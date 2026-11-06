use std::hash::{Hash, RandomState};

use super::{CollectionLog, Delta, Displaced, IncrCollection, Staged, StagedDelete, StagedInsert};
use crate::cells::Cells;
use crate::hashed_multiset::{AddAt, HashedMultiset};
use crate::runtime::Runtime;
use crate::value::Value;
use crate::value_code;

/// One change of a batch given to [`IncrCollection::apply`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Change<T> {
    Insert(T),
    Delete(T),
}

/// A batch prepared with no lock held: each change hashed (user `Hash`),
/// each insert's copies made (user `Clone`), and each change assigned the
/// group of the distinct value it names (user `Eq`).
struct StagedBatch<T> {
    changes: Vec<Staged<T>>,
    group_of: Vec<usize>,
    groups: Vec<Group>,
}

struct Group {
    hash: u64,
    /// The change whose value stands for the group in lookups.
    first: usize,
}

/// A staged change's value, equal to another by the value alone, keyed in
/// the batch's own multiset to find the group it belongs to.
struct Member<'a, T> {
    value: &'a T,
    group: usize,
}

impl<T: PartialEq> PartialEq for Member<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T> Staged<T> {
    fn key(&self) -> (u64, &T) {
        match self {
            Staged::Insert(s) => (s.hash, &s.stored),
            Staged::Delete(s) => (s.hash, &s.value),
        }
    }

    fn inserted_copy(&self) -> Option<T>
    where
        T: Clone,
    {
        match self {
            Staged::Insert(s) => Some(s.stored.clone()),
            Staged::Delete(_) => None,
        }
    }
}

impl<T: Hash + Eq + Clone> StagedBatch<T> {
    fn new(hasher: &RandomState, changes: impl IntoIterator<Item = Change<T>>) -> Self {
        let changes: Vec<Staged<T>> = changes
            .into_iter()
            .map(|change| match change {
                Change::Insert(value) => Staged::Insert(StagedInsert::new(hasher, value)),
                Change::Delete(value) => Staged::Delete(StagedDelete::owned(hasher, value)),
            })
            .collect();
        let mut group_of = Vec::with_capacity(changes.len());
        let mut groups = Vec::new();
        let mut seen = HashedMultiset::new();
        value_code::run(|| {
            for (i, change) in changes.iter().enumerate() {
                let (hash, value) = change.key();
                let member = Member {
                    value,
                    group: groups.len(),
                };
                match seen.locate_add(hash, &member) {
                    AddAt::Existing(at) => group_of.push(seen.stored(at).group),
                    at => {
                        group_of.push(member.group);
                        groups.push(Group { hash, first: i });
                        let _ = seen.add(at, member);
                    }
                }
            }
        });
        drop(seen);
        Self {
            changes,
            group_of,
            groups,
        }
    }
}

/// What a batch does to one distinct value: `inserts` or `deletes` net
/// occurrences (at most one of them non-zero), at its entry `at`.
struct Planned<T> {
    at: Option<usize>,
    inserts: usize,
    deletes: usize,
    /// Copies for the caller of the values this group's net changes add
    /// to the table or remove from it while the table keeps one.
    copies: Vec<T>,
}

/// A committed batch: the new version if anything changed, the changes
/// applied when asked for, and what the batch displaced, for the caller
/// to drop once the version is set.
struct CommittedBatch<T> {
    version: Option<u64>,
    applied: Vec<Change<T>>,
    unused: Vec<T>,
    displaced: Displaced<T>,
}

impl<T: Hash + Eq + Clone> CollectionLog<T> {
    /// Applies a staged batch under the log guard. Every lookup and user
    /// `Clone` runs before the first change, changing nothing but the
    /// table's capacity, so a panic leaves the log as it was; from the
    /// first change on no user code runs.
    fn apply_batch(&mut self, batch: StagedBatch<T>, report: bool) -> CommittedBatch<T> {
        let mut plan = value_code::run(|| self.plan_batch(&batch, report));
        let mut applied = Vec::new();
        let mut unused = Vec::new();
        let start = self.deltas.len();
        // The indices in `plan` stay valid to the end: the reserve in
        // `plan_batch` covers every absent value, so nothing below
        // rehashes, and filling a free slot or emptying a full one never
        // moves another entry.
        for (change, group) in batch.changes.into_iter().zip(batch.group_of) {
            let planned = &mut plan[group];
            match change {
                Staged::Insert(s) if planned.inserts > 0 => {
                    planned.inserts -= 1;
                    self.deltas.push(Delta::Insert(s.delta));
                    let kept = match planned.at {
                        Some(at) => self.elements.add(AddAt::Existing(at), s.stored),
                        None => {
                            planned.at = Some(self.elements.insert_absent_probe(s.hash, s.stored));
                            planned.copies.pop()
                        }
                    };
                    if let Some(value) = kept {
                        if report {
                            applied.push(Change::Insert(value));
                        } else {
                            unused.push(value);
                        }
                    }
                }
                Staged::Delete(s) if planned.deletes > 0 => {
                    planned.deletes -= 1;
                    self.deltas.push(Delta::Delete(s.value));
                    let at = planned.at.expect("a net delete names a present value");
                    let removed = self.elements.decrement_or_remove(at);
                    match (removed, report) {
                        (Some(value), true) => applied.push(Change::Delete(value)),
                        (Some(value), false) => unused.push(value),
                        (None, _) => applied.extend(planned.copies.pop().map(Change::Delete)),
                    }
                }
                Staged::Insert(s) => unused.extend([s.delta, s.stored]),
                Staged::Delete(s) => unused.push(s.value),
            }
        }
        let pushed = (self.deltas.len() - start) as u64;
        self.version += pushed;
        CommittedBatch {
            version: (pushed > 0).then_some(self.version),
            applied,
            unused,
            displaced: Displaced {
                values: [None],
                compacted: self.maybe_compact(),
            },
        }
    }

    /// Phase 1 of [`Self::apply_batch`]: finds each distinct value once,
    /// replays the batch's changes against its count to get the net
    /// change one-by-one changes would make, and reserves room for the
    /// values it adds.
    fn plan_batch(&mut self, batch: &StagedBatch<T>, report: bool) -> Vec<Planned<T>> {
        let find = |elements: &HashedMultiset<T>, group: &Group| {
            elements.find(group.hash, batch.changes[group.first].key().1)
        };
        let mut plan: Vec<Planned<T>> = batch
            .groups
            .iter()
            .map(|group| Planned {
                at: find(&self.elements, group),
                inserts: 0,
                deletes: 0,
                copies: Vec::new(),
            })
            .collect();
        let mut have: Vec<usize> = plan
            .iter()
            .map(|p| p.at.map_or(0, |at| self.elements.count(at)))
            .collect();
        let start = have.clone();
        let mut first_insert: Vec<Option<usize>> = vec![None; plan.len()];
        for (i, (change, &group)) in batch.changes.iter().zip(&batch.group_of).enumerate() {
            match change {
                Staged::Insert(_) => {
                    have[group] += 1;
                    first_insert[group].get_or_insert(i);
                }
                Staged::Delete(_) => have[group] = have[group].saturating_sub(1),
            }
        }
        for (p, (&before, &after)) in plan.iter_mut().zip(start.iter().zip(&have)) {
            p.inserts = after.saturating_sub(before);
            p.deletes = before.saturating_sub(after);
        }
        let absent = plan
            .iter()
            .filter(|p| p.at.is_none() && p.inserts > 0)
            .count();
        if self.elements.reserve(absent) {
            for (p, group) in plan.iter_mut().zip(&batch.groups) {
                if p.at.is_some() {
                    p.at = find(&self.elements, group);
                }
            }
        }
        if report {
            for (g, p) in plan.iter_mut().enumerate() {
                p.copies = match p.at {
                    None if p.inserts > 0 => first_insert[g]
                        .and_then(|i| batch.changes[i].inserted_copy())
                        .into_iter()
                        .collect(),
                    Some(at) if p.deletes > 0 => {
                        let removes_entry = have[g] == 0;
                        let stored = self.elements.stored(at);
                        std::iter::repeat_with(|| stored.clone())
                            .take(p.deletes - usize::from(removes_entry))
                            .collect()
                    }
                    _ => Vec::new(),
                };
            }
        }
        plan
    }
}

impl<T: Value + Hash + Eq, C: Cells> IncrCollection<T, C> {
    /// Applies `changes` in order as one atomic step, the same as calling
    /// `insert` and `delete` for each but with one version set: a reader
    /// sees the batch's net effect or none of it. A delete of a value not
    /// present at its point in the batch is skipped, as `delete` skips it.
    /// A panic in user `Hash`, `Eq` or `Clone` leaves the collection
    /// unchanged.
    ///
    /// Returns the net changes made, one per occurrence added or removed;
    /// a removal carries the stored value it removed, which can differ
    /// from the argument that named it if `Eq` equates distinct values.
    ///
    /// Panics if this collection is the output of an operator.
    pub fn apply(
        &self,
        rt: &Runtime<C>,
        changes: impl IntoIterator<Item = Change<T>>,
    ) -> Vec<Change<T>> {
        self.apply_reporting(rt, changes, true)
    }

    /// [`Self::apply`] with every change an insert.
    ///
    /// Panics if this collection is the output of an operator.
    pub fn extend(&self, rt: &Runtime<C>, values: impl IntoIterator<Item = T>) {
        let applied = self.apply_reporting(rt, values.into_iter().map(Change::Insert), false);
        debug_assert!(applied.is_empty());
    }

    fn apply_reporting(
        &self,
        rt: &Runtime<C>,
        changes: impl IntoIterator<Item = Change<T>>,
        report: bool,
    ) -> Vec<Change<T>> {
        self.check_mutable();
        value_code::check();
        let batch = StagedBatch::new(&self.hasher, changes);
        let committed = self.log.write().apply_batch(batch, report);
        let routers = committed
            .version
            .is_some_and(|version| rt.set_unrouted(self.version_node, version));
        drop(committed.unused);
        committed.displaced.release(&self.log);
        if routers {
            rt.route_queued();
        }
        committed.applied
    }
}
