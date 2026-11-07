//! The key index of a `KeyedCollection` and the two phases every keyed
//! operation runs in. Phase 1 ([`StagedKeys`]) runs all user key and row
//! code against the index and changes nothing but the index's capacity;
//! the pair collection then applies the staged changes atomically; phase 2
//! ([`KeyIndex::commit`]) updates the index from what the collection
//! reports it applied, running no user code, so the two never diverge.

use std::cell::{Cell, RefCell, RefMut};
use std::collections::HashMap;
use std::hash::{BuildHasher, RandomState};
use std::ops::{Deref, DerefMut};

use hashbrown::HashTable;
use incr_compute::{Change, Held, StateKind, StateMemory, StateStats};
use incr_core::lock_order::{self, LockKind};
use incr_core::{Local, RoutingHold};
use incr_python_protocol::PyValue;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// How many keys an operation inserted, gave a new row, and removed.
pub(crate) type KeyCounts = (usize, usize, usize);

/// What phase 2 treats as applied to the pair collection.
#[derive(Clone, Copy)]
pub(crate) enum Applied {
    /// The changes the collection reports, matched to staged pairs by
    /// identity: for a collection whose stored pairs are the index's own.
    Reported,
    /// Every staged change, for a persisted collection, whose stored pairs
    /// are decoded copies that are equal to the index's but never the same
    /// objects. The index mirrors the collection, so a staged delete of a
    /// held pair always finds its equal in the collection.
    AsStaged,
}

/// One index change an operation made, kept by an overlay to undo it:
/// the pair, its stored hash, and whether the operation inserted it (else
/// it removed it).
pub(crate) struct KeyStep {
    hash: u64,
    pair: Py<PyTuple>,
    inserted: bool,
}

/// Each held key's hash and the `(key, row)` tuple the pair collection
/// stores for it, the same object, so a pair is named to the collection
/// and found here by identity. Rebuildable from the pairs alone.
pub(crate) struct KeyIndex {
    table: HashTable<(u64, Py<PyTuple>)>,
    hasher: RandomState,
}

/// A `KeyedCollection`'s index, which the runtime's state registry also
/// reaches, through a `Weak`. `settled` holds the index's pair count and
/// allocation as of the last operation's end, for a stats walk run from a
/// key or row method while an operation holds the index.
pub(crate) struct SharedIndex {
    index: RefCell<KeyIndex>,
    settled: Cell<(usize, usize)>,
}

// SAFETY: the registry needs `Send + Sync`, but it lives in a
// `Runtime<Local>`, which is neither, and the index otherwise only in the
// `unsendable` `KeyedCollection`, so it is only ever reached from the one
// thread that runtime lives on.
unsafe impl Send for SharedIndex {}
// SAFETY: see `Send`.
unsafe impl Sync for SharedIndex {}

impl SharedIndex {
    pub(crate) fn new() -> Self {
        Self::holding(KeyIndex::new())
    }

    pub(crate) fn holding(index: KeyIndex) -> Self {
        let settled = (index.table.len(), index.table.allocation_size());
        Self {
            index: RefCell::new(index),
            settled: Cell::new(settled),
        }
    }

    pub(crate) fn try_borrow_mut(&self) -> Option<IndexGuard<'_>> {
        let index = self.index.try_borrow_mut().ok()?;
        Some(IndexGuard {
            index,
            settled: &self.settled,
            _binding: None,
            _routing: None,
        })
    }

    /// [`Self::try_borrow_mut`] for an operation that writes, which runs
    /// user key and row code with the index held: `routing` must outlive
    /// the borrow, since routing runs key functions that may use the index.
    pub(crate) fn try_borrow_for_write<'a>(
        &'a self,
        routing: RoutingHold<'a, Local>,
    ) -> Option<IndexGuard<'a>> {
        let binding = lock_order::acquire(LockKind::Binding);
        let index = self.index.try_borrow_mut().ok()?;
        Some(IndexGuard {
            index,
            settled: &self.settled,
            _binding: Some(binding),
            _routing: Some(routing),
        })
    }
}

/// Mutable access to a [`SharedIndex`] that records its size on release.
pub(crate) struct IndexGuard<'a> {
    index: RefMut<'a, KeyIndex>,
    settled: &'a Cell<(usize, usize)>,
    _binding: Option<lock_order::Held>,
    _routing: Option<RoutingHold<'a, Local>>,
}

impl Deref for IndexGuard<'_> {
    type Target = KeyIndex;

    fn deref(&self) -> &KeyIndex {
        &self.index
    }
}

impl DerefMut for IndexGuard<'_> {
    fn deref_mut(&mut self) -> &mut KeyIndex {
        &mut self.index
    }
}

impl Drop for IndexGuard<'_> {
    fn drop(&mut self) {
        self.settled
            .set((self.index.table.len(), self.index.table.allocation_size()));
    }
}

impl StateStats for SharedIndex {
    fn state_stats(&self) -> StateMemory {
        let (pairs, bytes) = match self.index.try_borrow() {
            Ok(index) => (index.table.len(), index.table.allocation_size()),
            Err(_) => self.settled.get(),
        };
        StateMemory::new(
            StateKind::Other("keyed_index"),
            pairs,
            vec![Held::of::<Py<PyTuple>>(pairs)],
            bytes,
        )
    }
}

fn item<'py>(pair: &Bound<'py, PyTuple>, i: usize) -> Bound<'py, PyAny> {
    pair.get_item(i).expect("a pair has two items")
}

/// Python `==` with the identity shortcut dicts and tuples use.
fn equal(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(a.is(b) || a.eq(b)?)
}

/// The entry of `table` with `hash` for which `matches` holds, stopping at
/// the first error `matches` raises.
fn find<T>(
    table: &HashTable<T>,
    hash: u64,
    hash_of: impl Fn(&T) -> u64,
    mut matches: impl FnMut(&T) -> PyResult<bool>,
) -> PyResult<Option<&T>> {
    let mut error = None;
    let found = table.find(hash, |entry| {
        error.is_none()
            && hash_of(entry) == hash
            && matches(entry).unwrap_or_else(|e| {
                error = Some(e);
                false
            })
    });
    error.map_or(Ok(found), Err)
}

impl KeyIndex {
    pub(crate) fn new() -> Self {
        Self {
            table: HashTable::new(),
            hasher: RandomState::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.table.len()
    }

    pub(crate) fn hash(&self, key: &Bound<'_, PyAny>) -> PyResult<u64> {
        Ok(self.hasher.hash_one(key.hash()?))
    }

    /// The row held for `key`.
    pub(crate) fn row<'py>(&self, key: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let hash = self.hash(key)?;
        let found = self.find(hash, key)?;
        Ok(found.map(|pair| item(pair.bind(key.py()), 1)))
    }

    fn find(&self, hash: u64, key: &Bound<'_, PyAny>) -> PyResult<Option<&Py<PyTuple>>> {
        let found = find(
            &self.table,
            hash,
            |e| e.0,
            |(_, pair)| equal(&item(pair.bind(key.py()), 0), key),
        )?;
        Ok(found.map(|(_, pair)| pair))
    }

    /// Ends phase 1 by reserving room for every key `staged` adds, passes
    /// its changes to `apply` (the pair collection's atomic `apply`), and
    /// runs phase 2 from what that returns. Nothing changes if `apply`
    /// fails; nothing is applied if the staged keys change nothing.
    pub(crate) fn apply_staged(
        &mut self,
        py: Python<'_>,
        staged: StagedKeys,
        apply: impl FnOnce(Vec<Change<PyValue>>) -> PyResult<Vec<Change<PyValue>>>,
        journal: Option<&mut Vec<KeyStep>>,
        applied_as: Applied,
    ) -> PyResult<(KeyCounts, Vec<Change<PyValue>>)> {
        let changes = staged.changes(py);
        if changes.is_empty() {
            return Ok(((0, 0, 0), Vec::new()));
        }
        self.reserve_for(&staged);
        let applied = apply(changes)?;
        let kept = journal.is_some().then(|| applied.clone());
        let committed = match applied_as {
            Applied::Reported => applied,
            Applied::AsStaged => staged.changes(py),
        };
        let counts = self.commit(py, &staged, committed, journal);
        Ok((counts, kept.unwrap_or_default()))
    }

    fn reserve_for(&mut self, staged: &StagedKeys) {
        let added = staged
            .keys
            .iter()
            .filter(|k| k.held.is_none() && k.next.is_some())
            .count();
        self.table.reserve(added, |e| e.0);
    }

    /// Phase 2 for a pair collection the store has applied `staged` to:
    /// the index takes the staged keys, through stored hashes and pair
    /// identity, with no user code.
    pub(crate) fn commit_staged(&mut self, py: Python<'_>, staged: &StagedKeys) {
        self.reserve_for(staged);
        self.commit(py, staged, staged.changes(py), None);
    }

    /// An index holding exactly `pairs`, which must be `(key, row)` tuples.
    /// Runs the keys' `__hash__`.
    pub(crate) fn of_pairs<'py>(
        pairs: impl IntoIterator<Item = Bound<'py, PyTuple>>,
    ) -> PyResult<KeyIndex> {
        let mut index = KeyIndex::new();
        for pair in pairs {
            index.insert_pair(pair)?;
        }
        Ok(index)
    }

    /// Holds `pair` alongside any pair held for an equal key, as the pair
    /// collection, a multiset, does.
    pub(crate) fn insert_pair(&mut self, pair: Bound<'_, PyTuple>) -> PyResult<()> {
        let hash = self.hash(&item(&pair, 0))?;
        self.table.reserve(1, |e| e.0);
        self.table
            .insert_unique(hash, (hash, pair.unbind()), |e| e.0);
        Ok(())
    }

    /// Drops one held pair `==` to `pair`, if any, as the pair collection
    /// drops one occurrence of an element.
    pub(crate) fn remove_pair(&mut self, pair: &Bound<'_, PyTuple>) -> PyResult<()> {
        let hash = self.hash(&item(pair, 0))?;
        let py = pair.py();
        let held = find(
            &self.table,
            hash,
            |e| e.0,
            |(_, held)| equal(held.bind(py).as_any(), pair.as_any()),
        )?
        .map(|(_, held)| held.clone_ref(py));
        if let Some(held) = held {
            if let Ok(entry) = self.table.find_entry(hash, |e| e.1.is(&held)) {
                entry.remove();
            }
        }
        Ok(())
    }

    /// Reverses `steps` newest first, through stored hashes and pair
    /// identity, with no user code.
    pub(crate) fn undo(&mut self, py: Python<'_>, steps: &[KeyStep]) {
        for step in steps.iter().rev() {
            if step.inserted {
                if let Ok(entry) = self.table.find_entry(step.hash, |e| e.1.is(&step.pair)) {
                    entry.remove();
                }
            } else {
                self.table
                    .insert_unique(step.hash, (step.hash, step.pair.clone_ref(py)), |e| e.0);
            }
        }
    }

    /// Phase 2: makes the index hold what `applied` reports the pair
    /// collection added and removed, through stored hashes and pair
    /// identity, with no user code.
    pub(crate) fn commit(
        &mut self,
        py: Python<'_>,
        staged: &StagedKeys,
        applied: Vec<Change<PyValue>>,
        mut journal: Option<&mut Vec<KeyStep>>,
    ) -> KeyCounts {
        let mut by_pair = HashMap::new();
        for (i, k) in staged.keys.iter().enumerate() {
            for pair in [&k.held, &k.next].into_iter().flatten() {
                by_pair.insert(pair.as_ptr(), i);
            }
        }
        let mut done = vec![(false, false); staged.keys.len()];
        let mut stray_removals = 0;
        for change in applied {
            match change {
                Change::Delete(PyValue(pair)) => match by_pair.get(&pair.as_ptr()) {
                    Some(&i) => {
                        done[i].0 = true;
                        if let Some(steps) = journal.as_deref_mut() {
                            let k = &staged.keys[i];
                            if let Some(tuple) = [&k.held, &k.next]
                                .into_iter()
                                .flatten()
                                .find(|t| t.as_ptr() == pair.as_ptr())
                            {
                                steps.push(KeyStep {
                                    hash: k.hash,
                                    pair: tuple.clone_ref(py),
                                    inserted: false,
                                });
                            }
                        }
                        if let Ok(entry) = self
                            .table
                            .find_entry(staged.keys[i].hash, |e| e.1.is(&pair))
                        {
                            entry.remove();
                        }
                    }
                    None => {
                        stray_removals += 1;
                        if let Some(steps) = journal.as_deref_mut() {
                            steps.extend(self.table.iter().filter(|e| e.1.is(&pair)).map(|e| {
                                KeyStep {
                                    hash: e.0,
                                    pair: e.1.clone_ref(py),
                                    inserted: false,
                                }
                            }));
                        }
                        self.table.retain(|e| !e.1.is(&pair));
                    }
                },
                Change::Insert(PyValue(pair)) => {
                    let i = *by_pair
                        .get(&pair.as_ptr())
                        .expect("apply reports a staged pair as inserted");
                    done[i].1 = true;
                    let k = &staged.keys[i];
                    let next = k
                        .next
                        .as_ref()
                        .expect("an inserted pair is a staged next pair");
                    if let Some(steps) = journal.as_deref_mut() {
                        steps.push(KeyStep {
                            hash: k.hash,
                            pair: next.clone_ref(py),
                            inserted: true,
                        });
                    }
                    self.table
                        .insert_unique(k.hash, (k.hash, next.clone_ref(py)), |e| e.0);
                }
            }
        }
        done.iter()
            .fold((0, 0, stray_removals), |(ins, rep, rem), d| match d {
                (false, true) => (ins + 1, rep, rem),
                (true, true) => (ins, rep + 1, rem),
                (true, false) => (ins, rep, rem + 1),
                (false, false) => (ins, rep, rem),
            })
    }
}

/// One key as an operation stages it: the pair the index holds for it and
/// the pair it is to hold, `None` for absent.
struct StagedKey {
    hash: u64,
    key: Py<PyAny>,
    held: Option<Py<PyTuple>>,
    next: Option<Py<PyTuple>>,
}

/// Phase 1 of a keyed operation: the per-key changes staged so far, keyed
/// like the index and consulted before it, so several changes to one key
/// compose. All user key and row code runs here.
#[derive(Default)]
pub(crate) struct StagedKeys {
    keys: Vec<StagedKey>,
    by_key: HashTable<usize>,
}

impl StagedKeys {
    /// Stages `key` to hold `row`, or to be absent for `None`, over what is
    /// staged or held for it. A row `==` the current one changes nothing,
    /// and a held key object is kept over an equal new one, as a dict does.
    /// Returns whether the key was present before this change.
    pub(crate) fn stage(
        &mut self,
        index: &KeyIndex,
        key: &Bound<'_, PyAny>,
        row: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let py = key.py();
        let hash = index.hash(key)?;
        let i = match self.find(py, hash, key)? {
            Some(i) => i,
            None => {
                let held = index.find(hash, key)?.map(|p| p.clone_ref(py));
                let key = match &held {
                    Some(pair) => item(pair.bind(py), 0).unbind(),
                    None => key.clone().unbind(),
                };
                self.push(StagedKey {
                    hash,
                    key,
                    next: held.as_ref().map(|p| p.clone_ref(py)),
                    held,
                })
            }
        };
        let staged = &mut self.keys[i];
        let present = staged.next.is_some();
        match row {
            Some(row) => {
                if let Some(next) = &staged.next {
                    if equal(&item(next.bind(py), 1), row)? {
                        return Ok(true);
                    }
                }
                staged.next = Some(PyTuple::new(py, [staged.key.bind(py), row])?.unbind());
            }
            None => staged.next = None,
        }
        Ok(present)
    }

    /// Stages the removal of every key the index holds that nothing
    /// staged so far names.
    pub(crate) fn stage_removal_of_the_rest(
        &mut self,
        py: Python<'_>,
        index: &KeyIndex,
    ) -> PyResult<()> {
        for (hash, pair) in index.table.iter() {
            let key = item(pair.bind(py), 0);
            if self.find(py, *hash, &key)?.is_none() {
                self.push(StagedKey {
                    hash: *hash,
                    key: key.unbind(),
                    held: Some(pair.clone_ref(py)),
                    next: None,
                });
            }
        }
        Ok(())
    }

    fn find(&self, py: Python<'_>, hash: u64, key: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        let keys = &self.keys;
        let found = find(
            &self.by_key,
            hash,
            |&i| keys[i].hash,
            |&i| equal(keys[i].key.bind(py), key),
        )?;
        Ok(found.copied())
    }

    fn push(&mut self, key: StagedKey) -> usize {
        let i = self.keys.len();
        let hash = key.hash;
        self.keys.push(key);
        let keys = &self.keys;
        self.by_key.insert_unique(hash, i, |&j| keys[j].hash);
        i
    }

    /// How many keys the staged changes insert, give a new row, and remove.
    pub(crate) fn counts(&self) -> KeyCounts {
        self.keys
            .iter()
            .fold((0, 0, 0), |(ins, rep, rem), k| match (&k.held, &k.next) {
                (None, Some(_)) => (ins + 1, rep, rem),
                (Some(held), Some(next)) if !held.is(next) => (ins, rep + 1, rem),
                (Some(_), None) => (ins, rep, rem + 1),
                _ => (ins, rep, rem),
            })
    }

    /// Stages `rows`, a mapping or an iterable of `(key, row)` pairs (a
    /// later pair for a key wins), as the complete set of keys, so every
    /// other held key is staged for removal.
    pub(crate) fn stage_reconcile(
        &mut self,
        index: &KeyIndex,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let pairs = if rows.hasattr("items")? {
            rows.call_method0("items")?
        } else {
            rows.clone()
        };
        for pair in pairs.try_iter()? {
            let (key, row): (Bound<'_, PyAny>, Bound<'_, PyAny>) = pair?.extract()?;
            self.stage(index, &key, Some(&row))?;
        }
        self.stage_removal_of_the_rest(rows.py(), index)
    }

    /// The changes to the pair collection: each staged key whose pair
    /// changes removes the held pair and inserts the next one.
    pub(crate) fn changes(&self, py: Python<'_>) -> Vec<Change<PyValue>> {
        let pair = |p: &Py<PyTuple>| PyValue(p.clone_ref(py).into_any());
        let mut changes = Vec::new();
        for k in &self.keys {
            match (&k.held, &k.next) {
                (Some(held), Some(next)) if held.is(next) => {}
                (held, next) => {
                    changes.extend(held.iter().map(|p| Change::Delete(pair(p))));
                    changes.extend(next.iter().map(|p| Change::Insert(pair(p))));
                }
            }
        }
        changes
    }
}
