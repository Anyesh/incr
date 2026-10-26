//! The runtime's side of [`crate::hold`]: what this thread sees of its own
//! held landings, holding a landing, and the attempt's end, which
//! publishes them or, after a wound, retracts them.

use std::any::Any;
use std::collections::HashMap;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::Arc;

use crate::cells::Cells;
use crate::cycle::ComputeCtl;
use crate::cycle_state::{self, UnwindKind};
use crate::deferred::Deferred;
use crate::fx::FxBuildHasher;
use crate::hold::{self, Held};
use crate::node::{NodeData, NodeId};
use crate::segmented_nodes::IN_GROUP;
use crate::state::{self, NodeState};

use super::{is_computing, ChildLanding, GroupInfo, Inner, ParkPoint, Runtime, SingleClaim};
use crate::dep_stack::DepStack;
use crate::dependent_list::DependentList;
use crate::handle::ErasedIncr;

/// Each held slot's landing target, for publication's new-dep check.
type Targets = HashMap<u32, NodeState, FxBuildHasher>;

impl<C: Cells> Runtime<C> {
    /// `raw`, loaded from `cell`, as this thread sees it: a node its
    /// attempt holds reads as the landing's state. Only a `Computing`
    /// load pays for the check.
    #[inline(always)]
    pub(crate) fn seen(&self, cell: &C::State, raw: NodeState) -> NodeState {
        if !C::SINGLE_THREADED && is_computing(raw) {
            self.held_seen(cell, raw)
        } else {
            raw
        }
    }

    /// [`Self::seen`] for engine bookkeeping that is no read by the
    /// running frame, which a hit would hold for nothing.
    #[inline(always)]
    pub(crate) fn seen_quietly(&self, cell: &C::State, raw: NodeState) -> NodeState {
        if !C::SINGLE_THREADED && is_computing(raw) {
            hold::peek(state::cell_key::<C>(cell)).unwrap_or(raw)
        } else {
            raw
        }
    }

    /// A read of a held landing makes the reading frame's landing depend
    /// on it, so that landing is held too.
    #[cold]
    #[inline(never)]
    fn held_seen(&self, cell: &C::State, raw: NodeState) -> NodeState {
        match hold::view(state::cell_key::<C>(cell)) {
            Some(target) => {
                self.dep_stack.raise_hold();
                target
            }
            None => raw,
        }
    }

    /// What a landing's `write_erased` handed back: if it `wrote`, what it
    /// displaced, kept for the undo record of a node this thread holds;
    /// else the rejected equal value, which is not what the slot held.
    /// Either is dropped at the drain when not kept.
    pub(super) fn displaced(
        &self,
        node: &NodeData<C>,
        displaced: Option<Box<dyn Any>>,
        wrote: bool,
        bag: &mut Deferred<C>,
    ) {
        let back = if wrote && !C::SINGLE_THREADED && hold::active() {
            hold::keep_value(state::cell_key::<C>(node.state_cell()), displaced)
        } else {
            displaced
        };
        if let Some(d) = back {
            bag.push_erased(d);
        }
    }

    /// Raises `HOLD` on a run of a node this thread holds and has claimed
    /// again, or of a claim a cut kept and took out of its group, so its
    /// write keeps what it displaces for the undo record and its landing
    /// is held.
    #[inline(always)]
    pub(super) fn rerun_hold(&self, id: NodeId, node: &NodeData<C>, ctl: &ComputeCtl) {
        if !C::SINGLE_THREADED && hold::active() {
            self.raise_rerun_hold(id, node, ctl);
        }
    }

    #[cold]
    #[inline(never)]
    fn raise_rerun_hold(&self, id: NodeId, node: &NodeData<C>, ctl: &ComputeCtl) {
        if hold::holds(state::cell_key::<C>(node.state_cell()))
            || hold::cut_kept(self.runtime_id, id.0)
        {
            ctl.raise(crate::cycle::HOLD);
        }
    }

    /// `land_as` while this thread holds landings: the landing is held if
    /// the node already has an entry, if it was decided by reading a held
    /// landing since the claim, or if a cut kept the claim; `None` lets it
    /// land. A wound's unwind lands nothing before the records the attempt
    /// replaced are back ([`hold::take_recs`]), whichever runtime raised
    /// it, so a cut's claim then lands unrun under its old record.
    #[cold]
    #[inline(never)]
    pub(super) fn held_landing(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        target: NodeState,
        flagged: NodeState,
        unrun: bool,
    ) -> Option<NodeState> {
        if cycle_state::wounded() {
            self.retract_held_records();
        }
        let cell = state::cell_key::<C>(node.state_cell());
        if !hold::holds(cell) {
            if !hold::consulted() && !hold::cut_kept(self.runtime_id, id.0) {
                return None;
            }
            // Only a verify or an unrun release lands without opening an
            // entry first, and a release to New is of a claim from New; a
            // cut's claim released unrun takes its origin from the cut.
            let first = target == NodeState::New;
            let own = hold::open(
                cell,
                self.runtime_id,
                id.0,
                node.generation(),
                first,
                None,
                None,
            );
            debug_assert!(own.is_none(), "incr-core: a new entry hands back no value");
        }
        hold::land(cell, target, flagged, unrun);
        Some(target)
    }

    /// Opens the entry of a landing about to be held, before its writes,
    /// so the undo record sees what they replace.
    #[cold]
    #[inline(never)]
    pub(super) fn open_hold(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        first: bool,
        ctl: Option<&ComputeCtl>,
        bag: &mut Deferred<C>,
    ) {
        let kept = ctl.and_then(hold::take_kept);
        let cell = state::cell_key::<C>(node.state_cell());
        let verified_at = Some(node.verified_at());
        if let Some(own) = hold::open(
            cell,
            self.runtime_id,
            id.0,
            node.generation(),
            first,
            verified_at,
            kept,
        ) {
            bag.push_erased(own);
        }
    }

    /// Whether `c` has a dependent as `Local` would list it: a held
    /// reader's reverse edges change only at its publication, so its new
    /// deps count and the ones it dropped do not.
    #[inline(always)]
    pub(super) fn has_dependents_seen(&self, c: u32, listed: &DependentList) -> bool {
        if C::SINGLE_THREADED || !hold::active() {
            return !listed.is_empty();
        }
        self.has_dependents_held(c, listed)
    }

    /// Scans every held node's dep lists, O(held deps) per check; a
    /// delete inside a read that holds landings is rare.
    #[cold]
    #[inline(never)]
    fn has_dependents_held(&self, c: u32, listed: &DependentList) -> bool {
        let reads = |r: u32| {
            let mut found = false;
            self.nodes.get(r).for_each_dep(|d| found |= d.0 == c);
            found
        };
        let held = |r: u32| hold::holds(state::cell_key::<C>(self.nodes.get(r).state_cell()));
        listed.iter().any(|r| !held(r.0) || reads(r.0))
            || hold::keys(self.runtime_id)
                .into_iter()
                .any(|r| reads(r) && !listed.iter().any(|l| l.0 == r))
    }

    /// [`Self::claim_single`] of a node this thread holds: claimed `Dead`
    /// as `Local` claims its landing, and taken out of the hold.
    #[inline(always)]
    pub(super) fn claim_held_dead(&self, handle: ErasedIncr) -> Option<SingleClaim<'_>> {
        if C::SINGLE_THREADED || !hold::active() {
            return None;
        }
        self.claim_held_dead_cold(handle)
    }

    #[cold]
    #[inline(never)]
    fn claim_held_dead_cold(&self, handle: ErasedIncr) -> Option<SingleClaim<'_>> {
        let slot = handle.slot();
        let node = self.nodes.get(slot);
        let cell = state::cell_key::<C>(node.state_cell());
        hold::view(cell)?;
        let change = {
            let inner = self.inner.read();
            if node.generation() != handle.generation() {
                return Some(SingleClaim::Gone);
            }
            if self.owns_nodes(&inner, slot) {
                return Some(SingleClaim::Owner);
            }
            if self.has_dependents_seen(slot, &inner.dependents[slot as usize]) {
                return Some(SingleClaim::HasDependents);
            }
            // Only another thread's set walk moves the word meanwhile, and
            // only between Computing and ComputingDirty.
            while state::try_transition::<C>(
                node.state_cell(),
                state::load::<C>(node.state_cell()),
                NodeState::Dead,
            )
            .is_err()
            {}
            // Under the hold, as `claim_single` takes it.
            self.counted_change()
        };
        let held = hold::forget(cell).expect("incr-core: a viewed landing has its entry");
        if let Some(old) = &held.undo.deps {
            let mut now: Vec<NodeId> = Vec::new();
            node.for_each_dep(|d| now.push(d));
            let mut inner = self.inner.write();
            for (o, _) in old {
                if !now.contains(o) {
                    inner.dependents[o.0 as usize].remove(NodeId(slot));
                }
            }
        }
        // As a retraction ends an entry: children the held run opened are
        // discarded, and its values drop at the drain.
        let mut bag = Deferred::new();
        if let Some((_, opened)) = held.children {
            self.land_children_now(NodeId(slot), ChildLanding::Discard, opened, &mut bag);
        }
        if let Some(Some(old)) = held.undo.value {
            bag.push_erased(old);
        }
        bag.drain(self);
        Some(SingleClaim::Claimed(change))
    }

    /// The attempt ended without a wound: every held landing lands, in
    /// landing order, with what it deferred.
    #[cold]
    #[inline(never)]
    pub(super) fn publish_held(&self) {
        self.publish_own();
    }

    /// The attempt was wounded: every held landing goes back.
    #[cold]
    #[inline(never)]
    pub(super) fn retract_held(&self) {
        let hold::Taken {
            entries, cut_recs, ..
        } = hold::take(self.runtime_id);
        self.retract_entries(entries, cut_recs);
    }

    /// A wound's first step, before its unwind lands anything: the
    /// records the attempt replaced go back ([`hold::take_recs`]).
    #[cold]
    #[inline(never)]
    pub(super) fn retract_held_records(&self) {
        let recs = hold::take_recs(self.runtime_id);
        if recs.is_empty() {
            return;
        }
        let mut inner = self.inner.write();
        for (slot, rec, in_group) in recs {
            self.restore_rec(&mut inner, slot, rec, in_group);
        }
    }

    fn restore_rec(
        &self,
        inner: &mut Inner<C>,
        slot: u32,
        rec: Option<Arc<GroupInfo>>,
        in_group: bool,
    ) {
        match rec {
            Some(r) => {
                inner.member_recs.insert(slot, r);
            }
            None => {
                inner.member_recs.remove(&slot);
            }
        }
        let flags = self.nodes.get_with_flags(slot).1;
        let bits = C::u8_load_relaxed(flags);
        C::u8_store_relaxed(
            flags,
            if in_group {
                bits | IN_GROUP
            } else {
                bits & !IN_GROUP
            },
        );
    }

    /// A non-Clean read of this runtime from another runtime's frame. If
    /// one of this runtime's entries is open below and the read rested on
    /// a held landing, the reading run depends on state this attempt has
    /// not published, so its runtime is tainted and its landing held; a
    /// read that met only claims got what `Local` gets, a cycle. A first
    /// entry's holds end at its exit ([`Self::first_entry`]).
    #[cold]
    #[inline(never)]
    pub(super) fn foreign_entry<R>(&self, read: impl FnOnce() -> R) -> R {
        if !cycle_state::open_entry_of(self.runtime_id) {
            return read();
        }
        let before = hold::looks();
        let r = read();
        if hold::looks() != before {
            self.taint_reader();
        }
        r
    }

    /// Whether an entry of this runtime made now is its first inside
    /// another entry on the thread, whose exit ends its held landings
    /// ([`Self::first_entry`]).
    #[inline]
    pub(super) fn first_nested(&self) -> bool {
        !C::SINGLE_THREADED && cycle_state::first_nested_of(self.runtime_id)
    }

    /// An entry of this runtime other than a read (a stabilize, an
    /// observer's walk, a delete) made inside another entry on the thread.
    /// A read is wrapped in `read_slow` instead, around the walks it
    /// retries.
    #[inline]
    pub(super) fn nested_entry<R>(&self, entry: impl FnOnce() -> R) -> R {
        if self.first_nested() {
            return self.first_entry(entry);
        }
        entry()
    }

    /// Runs this runtime's first entry inside another runtime's entry,
    /// of any kind, and ends its held landings at the exit, since nothing
    /// may keep a pointer to the runtime past it: they publish if they
    /// read only published state, and retract (landing as not computed)
    /// if the runtime is tainted or the thread wounded, which taints the
    /// reader in turn.
    #[cold]
    #[inline(never)]
    pub(super) fn first_entry<R>(&self, entry: impl FnOnce() -> R) -> R {
        // No frame carries `HOLD` while no hold is active, so the reader is
        // looked up only when one is, or when this entry made one.
        let reader_held = hold::active() && self.hold_reader_has(crate::cycle::HOLD);
        debug_assert!(reader_held || !self.hold_reader_has(crate::cycle::HOLD));
        let outcome = catch_unwind(AssertUnwindSafe(entry));
        if hold::holds_runtime(self.runtime_id) {
            let unwound = outcome.is_err()
                && matches!(
                    cycle_state::unwind_kind(),
                    Some(UnwindKind::Wound | UnwindKind::Discard)
                );
            if unwound || cycle_state::wounded() || hold::tainted(self.runtime_id) {
                self.retract_held();
                self.taint_reader();
            } else {
                let closed = hold::closed(self.runtime_id);
                self.publish_own();
                // The frames' pops passed their HOLD to the reader, but
                // what it read is published now, unless a cycle closed in
                // the entry: the reader's value depends on where it entered.
                if closed {
                    self.hold_closed_cycle();
                } else if !reader_held {
                    if let Some(c) = self.dep_stack.hold_reader() {
                        // SAFETY: the reading frame is below this entry on
                        // the stack, so its control block outlives it.
                        unsafe { &*c }.clear(crate::cycle::HOLD);
                    }
                }
            }
        }
        match outcome {
            Ok(r) => r,
            Err(payload) => resume_unwind(payload),
        }
    }

    fn hold_reader_has(&self, flag: u16) -> bool {
        // SAFETY: the reading frame is on the stack, so its control block
        // is live (see `Frame::ctl`).
        self.dep_stack
            .hold_reader()
            .is_some_and(|c| unsafe { &*c }.has(flag))
    }

    /// A cycle closed on this thread's stack: what catches it depends on the
    /// member the attempt entered it at, so the nearest `Shared` frame is
    /// held, and its pops pass the hold down to the catcher and its readers.
    #[cold]
    #[inline(never)]
    pub(super) fn hold_closed_cycle(&self) {
        if let Some(rt) = self.dep_stack.raise_hold() {
            hold::note_closed(rt);
        }
    }

    /// The reading frame's landing depends on state this attempt has not
    /// published: it is held, and its runtime tainted.
    fn taint_reader(&self) {
        if let Some(reader) = self.dep_stack.raise_hold() {
            hold::taint(reader);
        }
    }

    fn publish_own(&self) {
        // A cut claim's record stays as the cut left it.
        let hold::Taken {
            entries, episodes, ..
        } = hold::take(self.runtime_id);
        // Before the members land, so a read of one of their landings also
        // reads the count.
        let publishing =
            (episodes != 0).then(|| self.count_landed(episodes, entries.iter().map(|e| e.slot)));
        // A release counted when its claims landed held: a walk planned
        // since then expanded through their held states, and must see the
        // count move when they publish.
        if entries.iter().any(|e| e.unrun) {
            self.count_release(entries.iter().filter(|e| e.unrun).map(|e| e.slot));
        }
        let mut bag = Deferred::new();
        let held: Targets = entries.iter().map(|e| (e.slot, e.target)).collect();
        let mut rest = entries.into_iter();
        let published = catch_unwind(AssertUnwindSafe(|| {
            while let Some(slot) = rest.as_slice().first().map(|e| e.slot) {
                self.park(ParkPoint::BeforePublish(NodeId(slot)));
                if let Some(e) = rest.next() {
                    self.publish_one(e, &held, &mut bag);
                }
            }
        }));
        // An engine panic partway leaves no node Computing without an owner:
        // what is left goes back as on a wound (G9).
        if let Err(payload) = published {
            self.retract_entries(rest.collect(), Vec::new());
            self.landed(publishing.flatten());
            bag.drain_carried(payload);
        }
        self.landed(publishing.flatten());
        bag.drain(self);
    }

    fn publish_one(&self, e: Held, held: &Targets, bag: &mut Deferred<C>) {
        let id = NodeId(e.slot);
        let node = self.nodes.get(e.slot);
        debug_assert_eq!(node.generation(), e.generation);
        let mut target = e.target;
        if let Some(old) = &e.undo.deps {
            let mut new: Vec<NodeId> = Vec::new();
            node.for_each_dep(|d| new.push(d));
            let was = |d: &NodeId| old.iter().any(|(o, _)| o == d);
            {
                let mut inner = self.inner.write();
                for (o, _) in old {
                    if !new.contains(o) {
                        inner.dependents[o.0 as usize].remove(id);
                    }
                }
                for n in &new {
                    if !was(n) {
                        inner.dependents[n.0 as usize].push_unique(id);
                    }
                }
            }
            // A set after the read could not reach the node through an edge
            // that waited until now. A Clean landing's verified_at is the
            // revision it read at.
            let rev = node.verified_at();
            if target == NodeState::Clean
                && new
                    .iter()
                    .any(|d| !was(d) && !self.settled_at_publication(*d, held, rev))
            {
                target = NodeState::Dirty;
            }
        }
        // The node lands whatever its children's landing does.
        let children = e.children.map(|(landing, opened)| {
            let landing = match landing {
                ChildLanding::Sweep if target != NodeState::Clean => ChildLanding::Keep,
                other => other,
            };
            catch_unwind(AssertUnwindSafe(|| {
                self.land_children_now(id, landing, opened, bag)
            }))
        });
        for action in e.trace {
            self.push_trace_now(id, action);
        }
        let landed = self.land_now(id, node, target, e.flagged, e.unrun);
        // A set flagged the held Failed landing, as `land_failed` handles at
        // an unheld one.
        if target == NodeState::Failed && landed != NodeState::Failed {
            let token = self.failed_run_token(e.slot, e.generation);
            self.forget_failed_run_if(e.slot, e.generation, token);
            self.left_failed_landing(id, u64::MAX, bag);
        }
        if let Some(Some(own)) = e.undo.value {
            bag.push_erased(own);
        }
        if let Some(Err(payload)) = children {
            resume_unwind(payload);
        }
    }

    /// A new dep of a published landing: one this publication has yet to
    /// land counts as its target, unless a set flagged it since.
    fn settled_at_publication(&self, d: NodeId, held: &Targets, rev: u64) -> bool {
        let dep = self.nodes.get(d.0);
        let raw = state::load::<C>(dep.state_cell());
        match held.get(&d.0) {
            Some(&t) if is_computing(raw) => {
                raw == NodeState::Computing
                    && matches!(t, NodeState::Clean | NodeState::Cyclic)
                    && dep.changed_at() <= rev
            }
            _ => self.observed_clean(d, rev),
        }
    }

    /// Every entry goes back to what it replaced, and its node lands as
    /// its claim found it, unrun. Records and dep lists first, then
    /// values, then the state words, so no node is visible before its
    /// fields are back. Runs no user code; the attempt's values drop at
    /// the drain.
    fn retract_entries(
        &self,
        mut entries: Vec<Held>,
        cut_recs: Vec<(u32, Option<Arc<GroupInfo>>)>,
    ) {
        entries.reverse();
        // The retracted claims land unrun, as a release's do.
        if !entries.is_empty() {
            self.count_release(entries.iter().map(|e| e.slot));
        }
        let mut bag = Deferred::new();
        let mut lost = vec![false; entries.len()];
        {
            let mut inner = self.inner.write();
            for (i, e) in entries.iter_mut().enumerate() {
                let node = self.nodes.get(e.slot);
                if let Some(old) = &e.undo.deps {
                    let live: Vec<NodeId> = old
                        .iter()
                        .filter(|&&(d, g)| {
                            let dep = self.nodes.get(d.0);
                            dep.generation() == g && dep.state() != NodeState::Dead
                        })
                        .map(|&(d, _)| d)
                        .collect();
                    lost[i] = live.len() != old.len();
                    if let Some(displaced) = node.install_deps(&live) {
                        bag.push_dep_list(displaced);
                    }
                }
                if let Some((rec, in_group)) = e.undo.rec.take() {
                    self.restore_rec(&mut inner, e.slot, rec, in_group);
                }
                // An entry over a list that lost a dep counted the whole
                // old list.
                if let Some(prior) = e.undo.released.take() {
                    self.set_released(&mut inner, e.slot, node, prior.filter(|_| !lost[i]));
                }
            }
            for (slot, rec) in cut_recs {
                self.restore_rec(&mut inner, slot, rec, true);
            }
        }
        for (i, e) in entries.iter_mut().enumerate() {
            let node = self.nodes.get(e.slot);
            if let Some(old) = e.undo.value.take() {
                let arena = self.arenas.erased(node.type_tag());
                if let Some(own) = arena.restore_erased(node.arena_slot(), old) {
                    bag.push_erased(own);
                }
            }
            // A dep list that lost an entry no longer names the input the
            // old value came from, so it must not verify.
            let verified = if lost[i] {
                0
            } else {
                e.undo.verified_at.unwrap_or(0)
            };
            node.set_verified_at(verified);
            if let Some((_, opened)) = e.children {
                self.land_children_now(NodeId(e.slot), ChildLanding::Discard, opened, &mut bag);
            }
        }
        let failed: Vec<u32> = entries
            .iter()
            .filter(|e| e.target == NodeState::Failed)
            .map(|e| e.slot)
            .collect();
        self.release_retracted(&failed, &mut bag);
        for e in entries {
            let back = if e.undo.first {
                NodeState::New
            } else {
                NodeState::Dirty
            };
            self.land_now(NodeId(e.slot), self.nodes.get(e.slot), back, back, true);
        }
        bag.drain(self);
    }
}
