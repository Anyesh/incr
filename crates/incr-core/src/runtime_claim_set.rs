//! Deletes of owned sets: `claim_set`, which claims a set of roots and
//! everything they own under one write hold or rolls back, the
//! non-blocking variant a creator's sweep and rollback use, and the
//! cascade over units.
//!
//! A member is claimed `Deleting`, never straight to `Dead`, so that a
//! busy member found later in the set lets every earlier claim roll back
//! to the state it was taken from. `Deleting` lives only inside the write
//! hold: a `set` claims under a read hold, so it never sees it, and a
//! lock-free path that does spins until the hold ends.

use std::collections::HashSet;

use super::*;

/// What a `claim_set` over a set of roots did.
pub(super) enum SetEnd<'rt, C: Cells> {
    /// Every member is unhooked and recycled; the bag holds what they
    /// drop, and the count is the members'.
    Released(DeleteBag<'rt, C>, usize),
    /// Every root was already deleted.
    Gone,
    /// The member at this slot has a dependent outside the set.
    HasDependents(u32),
    /// This member is busy (computing, held by a set, or claimed by
    /// another delete); every claim was rolled back.
    Busy(ErasedIncr),
}

/// The members of an owned set, owners before the nodes they own, with
/// a hash set beside the order once the set is large enough for a scan
/// to cost more than a hash.
#[derive(Default)]
struct Members {
    order: Vec<u32>,
    index: Option<HashSet<u32, FxBuildHasher>>,
}

const SCAN_LIMIT: usize = 16;

impl Members {
    fn insert(&mut self, slot: u32) -> bool {
        if self.contains(slot) {
            return false;
        }
        self.order.push(slot);
        if let Some(index) = self.index.as_mut() {
            index.insert(slot);
        } else if self.order.len() > SCAN_LIMIT {
            self.index = Some(self.order.iter().copied().collect());
        }
        true
    }

    fn contains(&self, slot: u32) -> bool {
        match &self.index {
            Some(index) => index.contains(&slot),
            None => self.order.contains(&slot),
        }
    }
}

/// One entry of the cascade's stack. Entries from `base` to the top form
/// one segment, deleted together; a segment of one is a unit of its own.
struct Pending {
    handle: ErasedIncr,
    base: usize,
}

/// One step of the cascade, decided under a read hold.
enum Step {
    /// Every root of the top segment is gone.
    Pop,
    /// The segment needs [`Runtime::set_step`].
    Set,
    /// A dependent outside the top segment's owned set.
    Next(ErasedIncr),
    /// The top segment is a unit with no owned nodes and no dependents.
    DeleteSingle,
    /// The top segment's roots and their owned sets have no dependent
    /// outside them.
    DeleteSet,
}

impl<C: Cells> Runtime<C> {
    /// Adds the live roots and everything they own to `members`. Only
    /// meaningful under an `inner` hold.
    fn gather(&self, inner: &Inner<C>, roots: &[ErasedIncr], members: &mut Members) {
        for r in roots {
            if self.nodes.get(r.slot()).generation() == r.generation() {
                members.insert(r.slot());
            }
        }
        let mut i = 0;
        while i < members.order.len() {
            if let Some(owned) = self.owned_by(inner, members.order[i]) {
                for c in owned {
                    members.insert(c.0);
                }
            }
            i += 1;
        }
    }

    /// A member of `members` with a dependent outside it.
    fn outside_dependent(&self, inner: &Inner<C>, members: &Members) -> Option<(u32, NodeId)> {
        members.order.iter().find_map(|&m| {
            inner.dependents[m as usize]
                .iter()
                .find(|d| !members.contains(d.0))
                .map(|d| (m, d))
        })
    }

    /// Claims each of `members`, in order, from its state to `Deleting`,
    /// recording the state it took in `taken`. A busy member stops the
    /// claim and is returned, with the claims taken left for the caller
    /// to roll back.
    fn take_members(&self, members: &[u32], taken: &mut Vec<(u32, NodeState)>) -> Result<(), u32> {
        for &m in members {
            let node = self.nodes.get(m);
            loop {
                // Under the write hold no set can take its claim, which
                // needs a read hold, so one taken before is the only kind.
                if node.set_claimed() {
                    return Err(m);
                }
                let cur = state::load::<C>(node.state_cell());
                match cur {
                    NodeState::Computing
                    | NodeState::ComputingDirty
                    | NodeState::Dead
                    | NodeState::Deleting => return Err(m),
                    _ => {
                        if state::try_transition::<C>(node.state_cell(), cur, NodeState::Deleting)
                            .is_ok()
                        {
                            taken.push((m, cur));
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Returns every claim in `taken` to the state it was taken from.
    /// Only this thread moves a `Deleting` node, so each CAS succeeds.
    fn roll_back(&self, taken: &mut Vec<(u32, NodeState)>) {
        for (m, prior) in taken.drain(..).rev() {
            let back = state::try_transition::<C>(
                self.nodes.get(m).state_cell(),
                NodeState::Deleting,
                prior,
            );
            debug_assert!(back.is_ok(), "only the claimant moves a Deleting node");
        }
    }

    /// Marks every claimed member deleted and releases them, the nodes
    /// owned before their owners, under the caller's write hold: every
    /// generation is bumped before any user `Drop`, which `sink` defers.
    fn release_set<'rt>(
        &'rt self,
        inner: &mut Inner<C>,
        members: &[u32],
        mut sink: impl FnMut(NodeDrops<'rt, C>, Option<RetiredDeps<C>>),
    ) {
        for &m in members {
            let held = self.nodes.get(m).mark_deleted();
            debug_assert!(!held, "a set claimed a Deleting member");
        }
        for &m in members.iter().rev() {
            let retired = self.retire_claimed(self.erased_at(m));
            let (drops, dep_list) = self.unhook_retired(inner, retired);
            sink(drops, dep_list);
        }
    }

    /// Deletes `roots` and everything they own, transitively, as one
    /// set or not at all, under one write hold: the set is gathered,
    /// refused if any member has a dependent outside it (edges inside it
    /// do not count), and every member is claimed `Deleting`, owners
    /// first. A busy member rolls every claim back and is returned for
    /// the caller to wait out with no guard held.
    pub(super) fn claim_set(&self, roots: &[ErasedIncr], root: NodeId) -> SetEnd<'_, C> {
        let _change = self.changes.change(replay::speculative());
        let mut inner = self.inner.write();
        let mut members = Members::default();
        self.gather(&inner, roots, &mut members);
        if members.order.is_empty() {
            return SetEnd::Gone;
        }
        for &m in &members.order {
            self.refuse_delete_under_own_set(NodeId(m));
        }
        if let Some((m, _)) = self.outside_dependent(&inner, &members) {
            return SetEnd::HasDependents(m);
        }
        let mut taken = Vec::with_capacity(members.order.len());
        if let Err(busy) = self.take_members(&members.order, &mut taken) {
            self.park(ParkPoint::ClaimSetBeforeRollback(root));
            self.roll_back(&mut taken);
            return SetEnd::Busy(self.erased_at(busy));
        }
        let mut bag: Option<DeleteBag<'_, C>> = None;
        let count = members.order.len();
        self.release_set(&mut inner, &members.order, |drops, dep_list| {
            match bag.as_mut() {
                None => {
                    let mut first = DeleteBag::new(drops, dep_list);
                    first.reserve_batch(count - 1);
                    bag = Some(first);
                }
                Some(bag) => bag.push_node(drops, dep_list),
            }
        });
        SetEnd::Released(bag.expect("incr-core: a released set has a member"), count)
    }

    /// The strict delete of `root`, which owns nodes: its whole owned set
    /// through [`Self::claim_set`], waiting out busy members.
    pub(super) fn delete_owned_set(
        &self,
        root: ErasedIncr,
        spins: &mut u32,
    ) -> Result<(), StrictRefusal> {
        let id = NodeId(root.slot());
        if self.dep_stack.current_frame_active(self.runtime_id) {
            panic!(
                "incr-core: delete of node {} inside a compute, but it owns nodes; a node \
                 that owns nodes is deleted outside every compute",
                id.0,
            );
        }
        loop {
            match self.claim_set(&[root], id) {
                SetEnd::Released(bag, _) => {
                    bag.drain(self);
                    return Ok(());
                }
                SetEnd::Gone => return Err(StrictRefusal::Gone),
                SetEnd::HasDependents(slot) => return Err(StrictRefusal::HasDependents(slot)),
                SetEnd::Busy(member) => {
                    self.park(ParkPoint::ClaimSetRolledBack(id));
                    self.wait_out_member(member, spins);
                }
            }
        }
    }

    /// Waits, with no guard held, until busy `member` is no longer busy
    /// in the way `claim_set` found it. Panics on a claim this thread
    /// holds, which no wait releases.
    fn wait_out_member(&self, member: ErasedIncr, spins: &mut u32) {
        let id = NodeId(member.slot());
        let node = self.nodes.get(member.slot());
        if node.generation() != member.generation() {
            return;
        }
        if self.holds_claim(id) {
            panic!(
                "incr-core: delete of an owned set whose member {} this thread is computing",
                id.0,
            );
        }
        if node.set_claimed() {
            self.refuse_wait_on_own_set(id);
            self.spin_on(id, spins, || !node.set_claimed());
            return;
        }
        match state::load::<C>(node.state_cell()) {
            NodeState::Dead => self.wait_out_claim(member, spins),
            NodeState::Deleting => {
                *spins += 1;
                self.wait_out_deleting(id, *spins);
            }
            s if is_computing(s) => {
                self.refuse_wait_on_computing(id);
                self.spin_on(id, spins, || {
                    !is_computing(state::load::<C>(node.state_cell()))
                });
            }
            _ => {}
        }
    }

    /// Deletes every candidate's owned set that can go without waiting,
    /// under one write hold, for creator `creator`'s sweep or rollback.
    /// A candidate whose set has a busy member, or a dependent outside
    /// the sets being deleted, stays. The released members' drops join
    /// `bag`. Returns the candidates deleted, and those found already
    /// gone.
    pub(super) fn sweep_units(
        &self,
        creator: NodeId,
        candidates: &[ErasedIncr],
        bag: &mut Deferred<C>,
    ) -> HashSet<ErasedIncr> {
        let _change = self.changes.change(replay::speculative());
        let mut inner = self.inner.write();
        let mut done: HashSet<ErasedIncr> = HashSet::new();
        let mut units: Vec<(ErasedIncr, Members)> = Vec::with_capacity(candidates.len());
        for &c in candidates {
            let mut members = Members::default();
            self.gather(&inner, &[c], &mut members);
            if members.order.is_empty() {
                done.insert(c);
            } else {
                units.push((c, members));
            }
        }
        let mut taken = Vec::new();
        'claim: loop {
            // Drop units with a dependent outside the union, until none
            // has one: dropping a unit can strand a dependent of another.
            loop {
                let mut union = Members::default();
                for (_, members) in &units {
                    for &m in &members.order {
                        union.insert(m);
                    }
                }
                let before = units.len();
                units.retain(|(_, members)| {
                    !members.order.iter().any(|&m| {
                        inner.dependents[m as usize]
                            .iter()
                            .any(|d| !union.contains(d.0))
                    })
                });
                if units.len() == before {
                    break;
                }
            }
            for at in 0..units.len() {
                if self.take_members(&units[at].1.order, &mut taken).is_err() {
                    self.park(ParkPoint::ClaimSetBeforeRollback(creator));
                    self.roll_back(&mut taken);
                    units.swap_remove(at);
                    continue 'claim;
                }
            }
            break;
        }
        for (root, members) in &units {
            self.release_set(&mut inner, &members.order, |drops, dep_list| {
                bag.push_node(drops, dep_list)
            });
            done.insert(*root);
        }
        self.defer_buried_tickets(bag);
        done
    }

    /// Delete `root` together with every node that depends on it,
    /// transitively, and every node any of them owns (such as the groups
    /// of a `group_by` router, or a compute's children). Returns how many
    /// nodes this call deleted; a root that is already gone counts as
    /// done and returns 0.
    ///
    /// The cascade works on units, a node and everything it owns: a unit
    /// goes once no node outside it depends on any of its members. Units
    /// whose dependents lead back to each other (a dependency cycle, or
    /// owners that read each other's children) are deleted together.
    ///
    /// Dependents are found through the reverse edges a compute records,
    /// so a query that was created but has never run is not reached: it
    /// survives and panics with a stale-handle message on its first
    /// compute. Concurrent cascades and plain deletes over the same nodes
    /// are safe, and a node deleted by one of them is not counted here.
    /// A cascade does not terminate while other threads keep attaching
    /// new dependents to its nodes faster than it removes them.
    ///
    /// Panics if called from inside a compute closure: the computing
    /// node may depend on `root`, and the cascade would wait on it
    /// forever.
    pub fn delete_with_dependents(&self, root: impl Into<ErasedIncr>) -> usize {
        value_code::check();
        let root = root.into();
        if root.runtime_id() == RuntimeId::PENDING {
            panic!(
                "incr-core: delete_with_dependents with a pending handle {root:?}, which \
                 names no node",
            );
        }
        assert!(
            !self.dep_stack.current_frame_active(self.runtime_id),
            "incr-core: delete_with_dependents called during compute; not permitted",
        );
        if root.runtime_id() != self.runtime_id {
            panic!(
                "incr-core: delete_with_dependents with a handle from a foreign runtime \
                 (handle was minted by runtime {}, used on runtime {})",
                root.runtime_id().get(),
                self.runtime_id.get(),
            );
        }

        let mut deleted = 0;
        let mut stack = Stack {
            entries: vec![Pending {
                handle: root,
                base: 0,
            }],
            buckets: None,
        };
        while let Some(&Pending { handle: top, base }) = stack.entries.last() {
            let mut step = if base + 1 == stack.entries.len() {
                self.unit_step(top)
            } else {
                Step::Set
            };
            if let Step::Set = step {
                step = self.set_step(&stack.entries[base..]);
            }
            match step {
                Step::Pop => stack.pop_segment(base),
                Step::Next(next) => match stack.position(next) {
                    // A unit lower on the stack depends on this segment
                    // and this segment on it: every segment from there
                    // up merges into one.
                    Some(at) => stack.merge_from(at),
                    None => stack.push(next),
                },
                // One delete per entry, so a wound retries only the
                // delete it interrupted and the count stays exact.
                Step::DeleteSingle => match self.enter(|| self.delete_erased(top)) {
                    DeleteOutcome::Deleted => {
                        deleted += 1;
                        stack.pop_segment(base);
                    }
                    DeleteOutcome::AlreadyGone => stack.pop_segment(base),
                    // A dependent attached after the step; walk to it next.
                    DeleteOutcome::HasDependents => {}
                },
                Step::DeleteSet => {
                    let seg = &stack.entries[base..];
                    if let Some(n) = self.enter(|| self.delete_segment(seg)) {
                        deleted += n;
                        stack.pop_segment(base);
                    }
                }
                Step::Set => unreachable!("set_step decides a segment"),
            }
        }
        deleted
    }

    /// The step for a segment that is one unit, under one read hold in
    /// which every generation compared is the linked node's.
    #[inline(always)]
    fn unit_step(&self, h: ErasedIncr) -> Step {
        let inner = self.inner.read();
        if self.nodes.get(h.slot()).generation() != h.generation() {
            return Step::Pop;
        }
        if self.owns_nodes(&inner, h.slot()) {
            return Step::Set;
        }
        // The last dependent, because a delete removes it from this list
        // by scanning from the back, so tearing down a fan-in of n stays
        // linear in n.
        match inner.dependents[h.slot() as usize].last() {
            Some(d) => Step::Next(self.erased_at(d.0)),
            None => Step::DeleteSingle,
        }
    }

    /// The step for a segment of several units, or of one that owns
    /// nodes: a dependent outside the segment's owned set, or the delete
    /// of the set.
    #[cold]
    #[inline(never)]
    fn set_step(&self, seg: &[Pending]) -> Step {
        let inner = self.inner.read();
        let roots: Vec<ErasedIncr> = seg.iter().map(|p| p.handle).collect();
        let mut members = Members::default();
        self.gather(&inner, &roots, &mut members);
        if members.order.is_empty() {
            return Step::Pop;
        }
        match self.outside_dependent(&inner, &members) {
            Some((_, d)) => Step::Next(self.erased_at(d.0)),
            None => Step::DeleteSet,
        }
    }

    /// One `claim_set` of a segment's roots: the count released, or
    /// `None` to decide the segment again after a dependent attached or a
    /// busy member was waited out.
    #[cold]
    #[inline(never)]
    fn delete_segment(&self, seg: &[Pending]) -> Option<usize> {
        let roots: Vec<ErasedIncr> = seg.iter().map(|p| p.handle).collect();
        let mut spins = 0;
        match self.claim_set(&roots, NodeId(roots[0].slot())) {
            SetEnd::Released(bag, n) => {
                bag.drain(self);
                Some(n)
            }
            SetEnd::Gone => Some(0),
            SetEnd::HasDependents(_) => None,
            SetEnd::Busy(member) => {
                self.wait_out_member(member, &mut spins);
                None
            }
        }
    }
}

/// The cascade's stack. A shallow stack is scanned to find an entry.
/// Past `SCAN_DEPTH` entries a table counts the entries per slot bucket
/// beside it, so a dependent whose bucket is empty, the common case, is
/// known not to be on the stack without a scan.
struct Stack {
    entries: Vec<Pending>,
    buckets: Option<Box<[u32; BUCKETS]>>,
}

const SCAN_DEPTH: usize = 16;
const BUCKETS: usize = 4096;

fn bucket(h: ErasedIncr) -> usize {
    h.slot() as usize % BUCKETS
}

impl Stack {
    #[inline(always)]
    fn position(&self, h: ErasedIncr) -> Option<usize> {
        if let Some(buckets) = &self.buckets {
            if buckets[bucket(h)] == 0 {
                return None;
            }
        }
        self.entries.iter().position(|p| p.handle == h)
    }

    #[inline(always)]
    fn push(&mut self, h: ErasedIncr) {
        let at = self.entries.len();
        self.entries.push(Pending {
            handle: h,
            base: at,
        });
        match &mut self.buckets {
            Some(buckets) => buckets[bucket(h)] += 1,
            None if at >= SCAN_DEPTH => self.count_buckets(),
            None => {}
        }
    }

    #[cold]
    #[inline(never)]
    fn count_buckets(&mut self) {
        let mut buckets = Box::new([0_u32; BUCKETS]);
        for p in &self.entries {
            buckets[bucket(p.handle)] += 1;
        }
        self.buckets = Some(buckets);
    }

    fn merge_from(&mut self, at: usize) {
        let merged = self.entries[at].base;
        for p in &mut self.entries[merged..] {
            p.base = merged;
        }
    }

    #[inline(always)]
    fn pop_segment(&mut self, base: usize) {
        match &mut self.buckets {
            Some(buckets) => {
                for p in self.entries.drain(base..) {
                    buckets[bucket(p.handle)] -= 1;
                }
            }
            None => self.entries.truncate(base),
        }
    }
}
