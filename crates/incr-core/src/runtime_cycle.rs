//! Cycle resolution: the read rules, discovery's deferrals, episodes and
//! their landing, group verification, discard landings and the engine
//! entries that turn cycle unwinds into results.
//!
//! Discovery is Tarjan's algorithm over the conceptual stack, whose
//! positions (`cidx`) are frame depths. A read that reaches a node this
//! thread is computing, on its frames or in its held map, lowers the
//! reading frame's `lowlink` and marks it `LOW`; a `LOW` frame buffers its
//! value instead of writing it. At its pop a `LOW` frame whose `lowlink`
//! is below its own position is deferred: its claim moves to the held map
//! and its parent inherits the lowlink. A frame whose `lowlink` is its
//! own position is the root of the cycle, and runs an episode over itself
//! and every claim held at or above it.

use std::any::Any;
use std::cell::Cell;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use super::*;
use crate::cycle::{downcast, CycleUnwind, Erased, WoundUnwind};
use crate::cycle_state::{Episode, Held, Look, Round, Seen, Step};
use crate::deferred::Unbuilt;
use crate::{hold, reserve, set_stack, waits};

/// What a compute attempt carries into its landings.
#[derive(Copy, Clone)]
pub(super) struct Attempt {
    /// The claim was taken from New.
    pub(super) first: bool,
    pub(super) cidx: u32,
    pub(super) stamp_rev: u64,
    pub(super) verify_rev: u64,
}

/// An episode's members taken out of the held map, with what their
/// landing needs.
struct EpisodeLanding<'a> {
    members: &'a [u32],
    cu: &'a [u32],
    uncut: bool,
    cycle: Cycle,
    stamp_rev: u64,
    verify_rev: u64,
    entries: Vec<(u32, Seen<Held>)>,
}

/// What cutting an episode to its root's SCC did.
enum Shrink {
    Kept,
    Shrunk,
    Dissolved,
}

/// How a `LOW` pop, or an episode, left the root.
pub(super) enum PopEnd {
    /// The node is held, for a lower root to land.
    Deferred,
    /// The episode landed every member.
    Landed,
    /// Validation found the root alone: its caller ends the root's held
    /// entry, and it computes as an ordinary node, under the claim it holds.
    Dissolved,
    /// A recorded dep went stale: every member landed New, and the root
    /// claims again.
    Retry,
    /// A doom discarded a member run: the episode waits in a replay
    /// context, the root's claim with it, and the root's reader is
    /// answered as a discarded compute's.
    Suspended,
    /// A speculative run's pop would have run an episode and dropped its
    /// chain cooperatively instead, its claims discarded: the reader is
    /// answered as a discarded compute's.
    Dropped,
    /// The root's compute is a walk's entry, which contained the episode's
    /// abort: its claims have landed, and the walk goes on.
    Contained,
}

/// What an episode did, for tests.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EpisodeSeen {
    Start(u64),
    /// A doomed member run suspended the episode in iteration `k`, with
    /// its captured `(stamp_rev, verify_rev)`.
    Suspend {
        id: u64,
        k: u32,
        revs: (u64, u64),
    },
    /// A member run's virtual frame popped and resumed its episode, with
    /// `queued_above` entries still queued above that frame's level.
    Resume {
        id: u64,
        k: u32,
        revs: (u64, u64),
        queued_above: usize,
    },
    /// A run of member `slot` started under the episode's `MEMBER` mark.
    Member {
        id: Option<u64>,
        slot: u32,
    },
}

#[cfg(test)]
impl EpisodeSeen {
    pub(crate) fn of(ep: &Episode, queued_above: Option<usize>) -> Self {
        let r = ep
            .round
            .as_ref()
            .expect("incr-core: a suspended episode has a round");
        let revs = (r.stamp_rev, r.verify_rev);
        match queued_above {
            Some(queued_above) => Self::Resume {
                id: ep.id,
                k: r.k,
                revs,
                queued_above,
            },
            None => Self::Suspend {
                id: ep.id,
                k: r.k,
                revs,
            },
        }
    }
}

/// How group verification left the node `compute_one` claimed.
pub(super) enum GroupEnd {
    Done(Computed),
    /// A member could not be claimed: compute the node as an ordinary one.
    Ordinary,
    /// The episode dissolved: compute the node as an ordinary one.
    Dissolved,
    Retry,
}

type Payload = Box<dyn Any + Send>;

/// The strongly connected component of every node of `adj`, as indices
/// into one numbering of the components. Iterative, so a long ring does
/// not recurse.
fn components(adj: &[Vec<usize>]) -> Vec<usize> {
    const UNSEEN: usize = usize::MAX;
    let n = adj.len();
    let mut index = vec![UNSEEN; n];
    let mut low = vec![0; n];
    let mut on_stack = vec![false; n];
    let mut comp = vec![UNSEEN; n];
    let mut stack = Vec::new();
    let mut next_index = 0;
    let mut next_comp = 0;
    for start in 0..n {
        if index[start] != UNSEEN {
            continue;
        }
        let mut calls: Vec<(usize, usize)> = vec![(start, 0)];
        index[start] = next_index;
        low[start] = next_index;
        next_index += 1;
        stack.push(start);
        on_stack[start] = true;
        while let Some(&mut (v, ref mut edge)) = calls.last_mut() {
            if let Some(&w) = adj[v].get(*edge) {
                *edge += 1;
                if index[w] == UNSEEN {
                    index[w] = next_index;
                    low[w] = next_index;
                    next_index += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    calls.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            calls.pop();
            if let Some(&(parent, _)) = calls.last() {
                low[parent] = low[parent].min(low[v]);
            }
            if low[v] == index[v] {
                while let Some(w) = stack.pop() {
                    on_stack[w] = false;
                    comp[w] = next_comp;
                    if w == v {
                        break;
                    }
                }
                next_comp += 1;
            }
        }
    }
    comp
}

/// The members of `adj` that lie in a component with more than one
/// member or with a self-edge.
fn in_cycles(adj: &[Vec<usize>]) -> Vec<bool> {
    let comp = components(adj);
    let mut size = vec![0usize; adj.len()];
    for &c in &comp {
        size[c] += 1;
    }
    (0..adj.len())
        .map(|v| size[comp[v]] > 1 || adj[v].contains(&v))
        .collect()
}

/// Ends what an outermost engine entry leaves of its replay, whichever
/// way it exits: contexts of another runtime that its unwinds kept, or a
/// doom's PENDING reservations from a `mark` that unwound. A reservation
/// left behind would keep another thread waiting until this one enters
/// the engine again.
struct OutermostExit;

impl Drop for OutermostExit {
    fn drop(&mut self) {
        if !reserve::none_held() {
            replay::reset();
            reserve::end_contexts();
        }
    }
}

/// [`OutermostExit`] for the outermost `Shared` entry under a `Local`
/// runtime's outermost entry: it removes only the contexts at or above
/// `cidx`, as the `Local` entry's below stay. A wound unwinding through it
/// waits out its winner here, where the claim the winner waited on is
/// this runtime's to check.
struct NestedSharedExit<'a, C: Cells> {
    rt: &'a Runtime<C>,
    cidx: u32,
}

impl<C: Cells> Drop for NestedSharedExit<'_, C> {
    fn drop(&mut self) {
        self.rt.end_nested_shared(self.cidx);
    }
}

/// Removes this thread's contexts and ends its reservations, those a doom
/// noted before `mark` opened their context included: `reset` alone
/// misses them when no context is open.
fn end_replay() {
    replay::reset();
    reserve::end_all();
}

impl<C: Cells> Runtime<C> {
    fn active_ctl_has(&self, flag: u16) -> bool {
        self.dep_stack
            .active_frame(self.runtime_id)
            // SAFETY: the active frame's control block is live.
            .is_some_and(|(ctl, _)| unsafe { &*ctl }.has(flag))
    }

    /// Whether the compute running on this thread is a member run of a
    /// known episode member.
    pub(crate) fn running_as_member(&self) -> bool {
        self.active_ctl_has(cycle::BUFFERED)
    }

    /// Whether the compute running on this thread has read a provisional
    /// value (R2 to R5 set `LOW`).
    pub(crate) fn read_provisional(&self) -> bool {
        self.active_ctl_has(cycle::LOW)
    }

    /// Raises R3's outcome if the running compute has read a provisional
    /// value: the cycle from its lowlink up.
    pub(crate) fn refuse_provisional(&self) {
        let Some((ctl, _)) = self.dep_stack.active_frame(self.runtime_id) else {
            return;
        };
        // SAFETY: the active frame's control block is live.
        let ctl = unsafe { &*ctl };
        if ctl.has(cycle::LOW) {
            self.raise_cycle(self.cycle_at(ctl.lowlink.get()));
        }
    }

    /// Raises the cycle a `get` read found: an engine unwind inside an
    /// engine entry, which the outermost entry turns into a panic, or that
    /// panic itself at depth 0.
    #[cold]
    #[inline(never)]
    pub(super) fn raise_cycle(&self, cycle: Cycle) -> ! {
        if cycle_state::depth() == 0 {
            cycle_panic(cycle);
        }
        // A closure that swallowed its wound reads on; its cycle must not
        // take the wound's place in flight, or a catch returns it. A drop
        // the wound's own unwind runs gets the cycle: a raise there aborts,
        // and `caught` puts the wound back.
        if cycle_state::wounded() && !std::thread::panicking() {
            self.raise_wound();
        }
        cycle_state::set_unwind_kind(Some(UnwindKind::Cycle));
        resume_unwind(Box::new(CycleUnwind(cycle)))
    }

    /// Runs `f`, an engine entry, under the entry depth. The outermost
    /// entry resets the per-entry cycle state and turns a cycle unwind
    /// that reaches it into a panic; a `try_get` at any depth turns it
    /// into `Err`.
    #[inline(always)]
    pub(super) fn entered<R>(&self, kind: ReadKind, mut f: impl FnMut() -> R) -> Result<R, Cycle> {
        let entry = EntryDepth::enter(self.runtime_id, !C::SINGLE_THREADED);
        let _nested = (!C::SINGLE_THREADED && entry.shared_outermost() && !entry.is_outermost())
            .then(|| self.nested_shared_exit());
        if !entry.is_outermost() && kind == ReadKind::Get {
            return Ok(f());
        }
        self.entered_catching(entry.is_outermost(), kind, &mut f)
    }

    #[cold]
    #[inline(never)]
    fn nested_shared_exit(&self) -> NestedSharedExit<'_, C> {
        reserve::op_begin();
        NestedSharedExit {
            rt: self,
            cidx: self.dep_stack.next_cidx(),
        }
    }

    #[cold]
    #[inline(never)]
    fn end_nested_shared(&self, cidx: u32) {
        replay::drop_all_from(cidx);
        if cycle_state::wounded() {
            // The winner may wait on one of them. Only `Local` frames lie
            // below the thread's first `Shared` entry, and none takes `HOLD`,
            // so no reader is tainted.
            if hold::holds_runtime(self.runtime_id) {
                self.retract_held();
            }
            debug_assert!(self.dep_stack.hold_reader().is_none());
            reserve::end_all();
            debug_assert!(
                cycle_state::held_keys()
                    .iter()
                    .all(|&(rt, _)| rt != self.runtime_id),
                "incr-core: a wounded thread waits for its winner holding claims"
            );
            self.await_winner();
        } else if !reserve::none_held() {
            reserve::end_contexts();
        }
    }

    /// [`Self::entered`] for an entry that is not a `try_get`, where a
    /// cycle unwind is raised, never returned.
    #[inline(always)]
    pub(super) fn enter<R>(&self, f: impl FnMut() -> R) -> R {
        match self.entered(ReadKind::Get, f) {
            Ok(r) => r,
            Err(_) => unreachable!("incr-core: a get entry raises its cycles"),
        }
    }

    #[inline(always)]
    fn entered_catching<R>(
        &self,
        outermost: bool,
        kind: ReadKind,
        f: &mut impl FnMut() -> R,
    ) -> Result<R, Cycle> {
        let _exit = (outermost && !C::SINGLE_THREADED).then(|| {
            reserve::op_begin();
            OutermostExit
        });
        loop {
            if outermost {
                cycle_state::reset_outermost();
                replay::reset();
            }
            match catch_unwind(AssertUnwindSafe(&mut *f)) {
                Ok(r) => {
                    if outermost && !C::SINGLE_THREADED && hold::active() {
                        self.publish_held();
                    }
                    if outermost {
                        cycle_state::debug_assert_exit_clear();
                    }
                    return Ok(r);
                }
                Err(payload) => {
                    if let Some(cycle) = self.caught(outermost, kind, payload) {
                        return Err(cycle);
                    }
                }
            }
        }
    }

    /// An unwind out of an entry: `None` to retry after a wound, or the
    /// cycle a `try_get` returns; anything else unwinds on.
    #[cold]
    #[inline(never)]
    fn caught(
        &self,
        outermost: bool,
        kind: ReadKind,
        payload: Box<dyn std::any::Any + Send>,
    ) -> Option<Cycle> {
        replay::drop_from(self.runtime_id, self.dep_stack.next_cidx());
        if outermost {
            self.retire_notes();
        }
        if outermost {
            replay::set_carry(None);
        }
        // A wound wins over any kind: whatever unwound last, this
        // thread gave up its claims to an older one, and retries.
        if outermost && cycle_state::wounded() {
            cycle_state::take_unwind_kind();
            cycle_state::set_wounded(false);
            drop(payload);
            // The winner may wait on a held landing: nothing the attempt
            // landed stays, in any order.
            if !C::SINGLE_THREADED && hold::active() {
                self.retract_held();
            }
            debug_assert!(
                cycle_state::held_keys().is_empty(),
                "incr-core: a wounded thread waits for its winner holding claims"
            );
            // Another runtime's contexts, which `drop_from` keeps, would keep
            // their reservations while the winner waits on one of them.
            end_replay();
            debug_assert!(
                reserve::none_held(),
                "incr-core: a wounded thread waits for its winner holding reservations"
            );
            self.park(ParkPoint::WoundRetracted);
            self.await_winner();
            return None;
        }
        // What a nested `try_get` caught is no result for a wounded thread:
        // the wound unwinds on to the outermost entry, unless a drop of the
        // wound's own unwind made the read. A cycle `try_get` returns without
        // an unwind leaves the kind Wound, so the closure's pop raises it.
        let wounded = !outermost && cycle_state::wounded();
        if wounded && !std::thread::panicking() {
            drop(payload);
            self.raise_wound();
        }
        // Any other unwind out of the outermost entry ends the attempt as
        // its read's outcome, which every landing it made is part of.
        if outermost && !C::SINGLE_THREADED && hold::active() {
            self.publish_held();
        }
        if outermost {
            cycle_state::debug_assert_exit_clear();
        }
        match cycle_state::take_unwind_kind() {
            Some(UnwindKind::Cycle) => {
                let cycle = match payload.downcast::<CycleUnwind>() {
                    Ok(unwind) => unwind.0,
                    // A closure swallowed the cycle and then panicked
                    // on its own: that panic is the one to deliver.
                    Err(payload) => resume_unwind(payload),
                };
                if kind == ReadKind::TryGet {
                    if wounded {
                        cycle_state::set_unwind_kind(Some(UnwindKind::Wound));
                    }
                    return Some(cycle);
                }
                cycle_panic(cycle);
            }
            other => {
                cycle_state::set_unwind_kind(other);
                resume_unwind(payload);
            }
        }
    }

    /// The head start of a wounded thread, which holds nothing now: until
    /// its winner has taken the claim it waited on, if that is one of this
    /// runtime's.
    fn await_winner(&self) {
        waits::await_winner(C::backoff, |(rt, slot)| {
            #[cfg(all(test, not(loom)))]
            if rt == self.runtime_id {
                waits::OWN_CLAIM_CHECKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            rt == self.runtime_id
                && !matches!(
                    state::load::<C>(self.nodes.get(slot).state_cell()),
                    NodeState::New | NodeState::Dirty
                )
        });
    }

    /// An engine unwind that a closure caught and returned past: a cycle
    /// becomes an ordinary result, as under `try_get`. A wound or a
    /// speculative discard stays in flight and the pop is told to treat
    /// its closure as unwound, since the claims the wound must release are
    /// still held and the closure's value rests on the swallowed panic.
    #[cold]
    pub(super) fn swallowed(&self, kind: Option<UnwindKind>) -> bool {
        match kind {
            Some(UnwindKind::Cycle) => {
                cycle_state::set_unwind_kind(None);
                false
            }
            Some(UnwindKind::Wound) | Some(UnwindKind::Discard) => true,
            None => false,
        }
    }

    /// Raises a wound: this thread unwinds to its outermost engine entry,
    /// every claim it holds is released on the way, and the entry
    /// retries. Also raised again at a pop whose closure swallowed it.
    #[cold]
    #[inline(never)]
    pub(super) fn raise_wound(&self) -> ! {
        cycle_state::set_wounded(true);
        cycle_state::set_unwind_kind(Some(UnwindKind::Wound));
        resume_unwind(Box::new(WoundUnwind))
    }

    /// A claim wait on `on` that has spun `REGISTER_AFTER` times: it
    /// registers with the claims this thread holds, `also` included, and
    /// waits until `done`, unless a registration that closes a waits-for
    /// cycle wounds it.
    #[cold]
    #[inline(never)]
    pub(super) fn wait_registered(
        &self,
        on: NodeId,
        also: Option<NodeId>,
        mut done: impl FnMut() -> bool,
    ) {
        let mut holds: Vec<waits::Claim> = self.dep_stack.with(|s| {
            s.frames
                .iter()
                .filter(|f| f.rt() != RuntimeId::SENTINEL)
                .map(|f| (f.rt(), f.node.0))
                .collect()
        });
        holds.extend(also.map(|x| (self.runtime_id, x.0)));
        holds.extend(cycle_state::held_keys());
        holds.extend(hold::claims());
        holds.extend(set_stack::claims());
        holds.extend(
            reserve::held()
                .into_iter()
                .filter(|&c| c != (self.runtime_id, on.0)),
        );
        // A thread that holds nothing is no one's owner, so no waits-for
        // cycle can pass through it and it never needs a registration.
        if holds.is_empty() {
            let mut spins = waits::REGISTER_AFTER;
            while !done() {
                spins += 1;
                C::backoff(spins);
            }
            return;
        }
        let reg = waits::Registration::new(
            |next| cycle_state::ticket(next),
            (self.runtime_id, on.0),
            holds,
        );
        let mut spins = waits::REGISTER_AFTER;
        loop {
            // A wait that is satisfied proceeds even if it was wounded
            // meanwhile: the oldest thread never needed its claims.
            if done() {
                return;
            }
            if reg.wounded() {
                drop(reg);
                self.raise_wound();
            }
            spins += 1;
            C::backoff(spins);
        }
    }

    /// One spin of a claim wait on `on`, which registers once it has
    /// spun long enough and then waits until `done`.
    #[inline]
    pub(super) fn spin_on(&self, on: NodeId, spins: &mut u32, done: impl FnMut() -> bool) {
        self.spin_holding(on, None, spins, done);
    }

    /// [`Self::spin_on`] by a thread that also holds `also`'s claim
    /// outside its frames and held claims.
    #[inline]
    fn spin_holding(
        &self,
        on: NodeId,
        also: Option<NodeId>,
        spins: &mut u32,
        done: impl FnMut() -> bool,
    ) {
        if *spins == 0 {
            crate::lock_order::assert_may_wait_on_claim();
        }
        *spins += 1;
        if !C::SINGLE_THREADED && *spins >= waits::REGISTER_AFTER {
            self.wait_registered(on, also, done);
        } else {
            C::backoff(*spins);
        }
    }

    /// Waits, as a claim wait on `id`, until no other thread reserves it
    /// at `generation`, holding `also`'s claim outside its frames. Leaves
    /// `spins` non-zero, so the claim retried after it is a waited one.
    #[cold]
    #[inline(never)]
    pub(super) fn wait_out_reservation(
        &self,
        id: NodeId,
        generation: u32,
        also: Option<NodeId>,
        spins: &mut u32,
    ) {
        let claim = (self.runtime_id, id.0);
        // Release, so a test that sees the count also sees this thread's
        // unclaim of `id` before it.
        #[cfg(test)]
        if reserve::elsewhere(claim, generation) {
            self.reservation_waits
                .fetch_add(1, std::sync::atomic::Ordering::Release);
        }
        while reserve::elsewhere(claim, generation) {
            self.spin_holding(id, also, spins, || !reserve::elsewhere(claim, generation));
        }
        *spins = (*spins).max(1);
    }

    /// Runs a cycle callback with no lock held, as value code, under a
    /// barrier frame that belongs to no runtime: a read of a Clean node
    /// returns its value and records nothing, and any other call into the
    /// runtime panics.
    pub(super) fn callback<R>(&self, f: impl FnOnce() -> R) -> R {
        struct Barrier<'a, C: Cells>(&'a Runtime<C>);
        impl<C: Cells> Drop for Barrier<'_, C> {
            fn drop(&mut self) {
                self.0.dep_stack.pop_frame();
            }
        }
        if !C::SINGLE_THREADED {
            reserve::callback_code();
        }
        let ctl = ComputeCtl::new(0);
        self.dep_stack
            .push_frame_at(RuntimeId::SENTINEL, NodeId::SENTINEL, &ctl, u32::MAX);
        self.dep_stack.settle_foreign_keeping(false);
        let _barrier = Barrier(self);
        value_code::run(f)
    }

    fn erased_at_live(&self, slot: u32) -> ErasedIncr {
        ErasedIncr::new(slot, self.nodes.get(slot).generation(), self.runtime_id)
    }

    /// The cycle a read at conceptual-stack position `p` closes: this
    /// runtime's frames at or above `p` and its claims held at or above.
    fn cycle_at(&self, p: u32) -> Cycle {
        let rt = self.runtime_id;
        let mut slots: Vec<u32> = self.dep_stack.with(|s| {
            s.frames
                .iter()
                .filter(|f| f.rt() == rt && f.cidx >= p)
                .map(|f| f.node.0)
                .collect()
        });
        if value_code::thread_word() & value_code::REPLAY_BIT != 0 {
            slots.extend(self.virtual_slots_from(p));
        }
        cycle_state::with_held(|h| slots.extend(h.anchored_from(rt, p)));
        Cycle::new(slots.into_iter().map(|s| self.erased_at_live(s)).collect())
    }

    /// Whether a frame of another runtime sits at or above `p`: a cycle
    /// closed through another runtime's closure is never resolved as a
    /// group.
    fn foreign_frame_above(&self, p: u32) -> bool {
        let rt = self.runtime_id;
        self.dep_stack.with(|s| {
            s.frames
                .iter()
                .any(|f| f.cidx >= p && f.cidx != u32::MAX && f.rt() != rt)
        })
    }

    /// The read rules, for a read of `handle` that found it `Cyclic` or
    /// Computing. `None` when the node is not this thread's: the read
    /// waits like any other. `armed` is the reading frame when the read is
    /// past the frame budget and may doom it.
    #[cold]
    #[inline(never)]
    pub(super) fn cycle_read<T: Value>(
        &self,
        handle: Incr<T>,
        observed: NodeState,
        kind: ReadKind,
        armed: Option<ActiveRead>,
    ) -> Option<Result<T, Cycle>> {
        let rt = self.runtime_id;
        let slot = handle.slot();
        if observed == NodeState::Cyclic {
            return Some(Err(self
                .landed_cycle(handle)
                .unwrap_or_else(|| Cycle::new(vec![handle.erase()]))));
        }
        let on_stack = self.dep_stack.with(|s| {
            s.frames
                .iter()
                .rev()
                .find(|f| f.node.0 == slot && f.rt() == rt)
                .map(|f| f.cidx)
        });
        let held = self.look_held(slot, Look::here(), |e| {
            (
                e.anchor(),
                e.in_cu,
                e.buf().is_some(),
                e.episode.clone(),
                e.ep,
            )
        });
        // A held node with a virtual frame stands for a frame on the stack.
        let on_stack = match on_stack {
            None if held.is_some() && value_code::thread_word() & value_code::REPLAY_BIT != 0 => {
                self.virtual_position(slot, handle.generation())
            }
            on_stack => on_stack,
        };
        let p = match (on_stack, &held) {
            (Some(cidx), _) => cidx,
            (None, Some((anchor, ..))) => *anchor,
            (None, None) => return None,
        };
        let active = self
            .dep_stack
            .active_frame(rt)
            .filter(|_| !self.foreign_frame_above(p));
        let Some((ctl, at)) = active else {
            if !C::SINGLE_THREADED {
                self.hold_closed_cycle();
            }
            return Some(Err(self.cycle_at(p)));
        };
        // SAFETY: the active frame is on the stack, so its control block
        // is live.
        let ctl = unsafe { &*ctl };

        // A member of a suspended episode, read from below its member run's
        // virtual frame: only a doomed frame's speculative run reads there.
        if let Some((.., ep)) = held {
            if value_code::thread_word() & value_code::REPLAY_BIT != 0
                && replay::suspended_reachable(ep) == Some(false)
            {
                debug_assert!(
                    replay::speculative(),
                    "incr-core: only a speculative run reads a suspended member from below"
                );
                if let Some(v) = self
                    .placeholder::<T>()
                    .filter(|_| self.drop_cooperatively(Some(at)))
                {
                    ctl.raise(cycle::PH_SEEN);
                    return Some(Ok(v));
                }
                self.drop_unwind();
            }
        }

        // R0: an unseeded member that still reads another such member.
        if let Some((anchor, true, _, Some(cycle), _)) = &held {
            ctl.lower(*anchor);
            return Some(Err(cycle.clone()));
        }
        // R2: a strategy member reads as its current value.
        if C::u8_load_relaxed(self.nodes.get_with_flags(slot).1) & HAS_STRATEGY != 0 {
            let value = match &held {
                Some((_, _, _, Some(_), _)) => self.held_value::<T>(slot, true),
                _ => self.discovery_value::<T>(slot, p),
            };
            ctl.lower(p);
            return Some(Ok(value));
        }
        // R3: an unseeded node on the stack has no value to give.
        if on_stack.is_some() {
            ctl.lower(p);
            if !C::SINGLE_THREADED {
                self.hold_closed_cycle();
            }
            return Some(Err(self.cycle_at(p)));
        }
        let (anchor, _, buffered, ..) = held.expect("an unseeded node off the stack is held");
        // R5: run the member for its value, then R4: read its buffer. Past
        // the budget the reader is doomed instead, and the member becomes
        // a held entry the driver runs as a member run.
        if !buffered {
            if let Some(f) = armed {
                ctl.lower(anchor);
                return Some(Ok(self.doom_held(handle, f)));
            }
            // One position above its reader, as the driver runs a held
            // entry: nested member runs at one anchor would share it, and a
            // doom marks frames by position.
            let mut r5 = self.member_run(slot, at + 1, true, false, false);
            // A read past the budget in the member run doomed it. If the
            // member run was the doom's bottom frame, its context runs here
            // until the member is buffered, and the loop's member run passes
            // its lowlink down; otherwise the doom marked the reader, which
            // takes the placeholder.
            if let Ok(None) = r5 {
                match self.member_run_discarded(at + 1) {
                    Some(ran) => r5 = ran.map(|()| None),
                    None => {
                        ctl.lower(anchor);
                        ctl.raise(cycle::PH_SEEN);
                        return Some(Ok(self
                            .placeholder::<T>()
                            .expect("incr-core: a doomed read's type has a placeholder")));
                    }
                }
            }
            match r5 {
                Ok(Some(low)) => {
                    if low <= at {
                        self.pass_down(at + 1, low);
                    }
                    ctl.lower(low);
                }
                Ok(None) => {}
                Err(payload) => {
                    ctl.lower(anchor);
                    let kind_now = cycle_state::unwind_kind();
                    let mut bag = Deferred::new();
                    self.release_unwound_from(
                        at + 1,
                        slot,
                        kind_now,
                        kind_now.is_some(),
                        Some(&*payload),
                        &mut bag,
                    );
                    bag.drain(self);
                    return Some(Err(self.member_run_unwound(kind, payload)));
                }
            }
        }
        ctl.lower(anchor);
        Some(Ok(self.held_value::<T>(slot, false)))
    }

    /// An R5 member run that ended in an unwind: a cycle a `get` in it
    /// raised is the reading `try_get`'s `Err`, as if the member had been
    /// read through an engine entry; anything else unwinds on.
    #[cold]
    #[inline(never)]
    fn member_run_unwound(&self, kind: ReadKind, payload: Payload) -> Cycle {
        if kind == ReadKind::TryGet && cycle_state::unwind_kind() == Some(UnwindKind::Cycle) {
            if let Some(unwind) = payload.downcast_ref::<CycleUnwind>() {
                cycle_state::take_unwind_kind();
                return unwind.0.clone();
            }
        }
        self.note_unwound_read(&*payload, cycle_state::unwind_kind());
        resume_unwind(payload)
    }

    /// A clone of a held node's current value (`s`) or buffer.
    fn held_value<T: Value>(&self, slot: u32, current: bool) -> T {
        self.look_held(slot, Look::here(), |seen| {
            let v = if current { &seen.s } else { seen.buf() };
            let v = v
                .as_ref()
                .expect("incr-core: a held member has a value to read");
            value_code::run(|| downcast::<T>(v).clone())
        })
        .expect("incr-core: a read member is held")
    }

    /// A strategy member's value when no episode runs: `g` of the cycle
    /// the read closes, or the fixpoint's initial value.
    fn discovery_value<T: Value>(&self, slot: u32, p: u32) -> T {
        let strategy = self.strategy_of(slot);
        // A Fixpoint's initial value ignores the cycle, which costs a
        // scan of every member to build.
        let cycle = strategy
            .max_iterations()
            .is_none()
            .then(|| self.cycle_at(p));
        let v = self.callback(|| strategy.v0(cycle.as_ref()));
        *v.downcast::<T>()
            .expect("incr-core: a cycle value has its member's type")
    }

    fn strategy_of(&self, slot: u32) -> Arc<dyn ErasedStrategy> {
        self.inner
            .read()
            .cycle_strategies
            .get(&slot)
            .cloned()
            .expect("incr-core: HAS_STRATEGY names a node with a strategy")
    }

    /// Runs held node `slot`'s closure as a member run at position
    /// `cidx`: its value is buffered in the held map, with the deps it
    /// recorded. Returns the run's lowlink, or the panic it ended in, with
    /// the node marked failed unless the run was speculative. A `walked`
    /// run finishes a walk's compute, so the virtual frames a doom in it
    /// makes contain a panic as that walk does. A `stand_in` run of a node
    /// outside any episode finishes a frame the unbounded run computes
    /// natively, so its panic marks the node failed as that compute's
    /// landing does, never counted as a member's failure nested in its
    /// readers' runs. Otherwise a run that finishes a doomed one counts the
    /// failures marked since the earliest of those runs began as nested in
    /// it.
    pub(super) fn member_run(
        &self,
        slot: u32,
        cidx: u32,
        doomable: bool,
        walked: bool,
        stand_in: bool,
    ) -> Result<Option<u32>, Payload> {
        let id = NodeId(slot);
        let compute = {
            let inner = self.inner.read();
            inner
                .compute_fns
                .get(slot as usize)
                .and_then(|f| f.as_deref())
                .map(std::ptr::from_ref)
                .expect("incr-core: a held node is a query")
        };
        let ctl = ComputeCtl::new(
            if doomable {
                cycle::BUFFERED
            } else {
                cycle::BUFFERED | cycle::NO_DOOM
            } | if walked { cycle::WALKED } else { 0 },
        );
        ctl.lowlink.set(cidx);
        self.dep_stack
            .push_frame_at(self.runtime_id, id, &ctl, cidx);
        if doomable && self.budgeted() {
            self.dep_stack.settle_foreign_keeping(true);
        }
        let mut began = None;
        if value_code::thread_word() & value_code::REPLAY_BIT != 0 {
            self.start_run(id, cidx, &ctl);
            if let Some(low) = self.take_low(id, cidx) {
                ctl.lower(low);
            }
            began = self.take_run_failures(id, cidx);
            self.divert_for_deliveries(id, cidx);
        }
        // SAFETY: this thread holds the node's claim, so its closure is
        // not freed (see `compute_attempts`).
        let compute = unsafe { &*compute };
        let failures = began.unwrap_or_else(cycle_state::failures);
        let first = replay::peek_id();
        let mut result = catch_unwind(AssertUnwindSafe(|| compute(self, slot, &ctl)));
        let (deps, reading) = self.dep_stack.pop_frame_reading();
        let word = value_code::thread_word();
        // A member run starts after its episode root's pop, so nothing
        // below it drops what it kept.
        if word & value_code::KEPT_BIT != 0 {
            replay::drop_kept_from(self.runtime_id, cidx);
        }
        if let Err(payload) = &result {
            if word & value_code::UNWIND_MASK != 0 {
                clear_caught_cycle(&**payload, &deps, reading);
            }
        }
        if word & (value_code::MEMO_BIT | value_code::UNWIND_MASK) != 0
            && self.after_pop(word, result.is_ok())
        {
            result = Err(match UnwindKind::of_word(word) {
                Some(UnwindKind::Discard) => Box::new(cycle::DiscardUnwind),
                _ => Box::new(WoundUnwind),
            });
        }
        if ctl.has(cycle::DOOMED)
            && (result.is_ok() || !matches!(cycle_state::unwind_kind(), Some(UnwindKind::Wound)))
        {
            replay::note_run_failures(
                self.runtime_id,
                slot,
                self.nodes.get(slot).generation(),
                cidx,
                failures,
            );
            self.member_run_doomed(slot, cidx, first, &ctl, result.err());
            return Ok(None);
        }
        if result.is_err() {
            replay::unwind_past_since(self.runtime_id, cidx, first);
            // Callers land the run after its speculative guard has dropped.
            cycle_state::set_unwind_kind(replay::run_unwind_kind(
                cycle_state::unwind_kind(),
                replay::speculative(),
            ));
            // A discard unwinds a run the unbounded run never makes: the
            // member keeps what its last run recorded, or its episode's
            // landing installs deps that the discard leaves New.
            if cycle_state::unwind_kind() == Some(UnwindKind::Discard) {
                return result.map(|_| None);
            }
        }
        let look = Look::here();
        let (replaced, spec_buf) = cycle_state::with_held(|h| {
            let e = h
                .get_mut(&(self.runtime_id, slot))
                .expect("incr-core: a member run's node is held");
            let out = e.record(
                look,
                deps,
                ctl.buf.take(),
                !ctl.children.get().is_null(),
                ctl.poison(),
            );
            // An engine unwind is no fault of the member's.
            if result.is_err() && cycle_state::unwind_kind().is_none() {
                if stand_in && e.episode.is_none() {
                    e.failed = true;
                } else if cycle_state::failures() == failures {
                    cycle_state::mark_failed(e);
                }
            }
            out
        });
        drop(replaced);
        drop(spec_buf);
        if word & value_code::REPLAY_BIT != 0
            && (result.is_ok() || cycle_state::unwind_kind().is_none())
        {
            self.end_run(id, cidx);
        }
        result.map(|_| Some(ctl.lowlink.get()))
    }

    /// A member run a doom discarded: it stood on placeholders, so nothing
    /// it computed is kept, but the member keeps its claim, for the run's
    /// retry at its virtual frame. The claims anchored above it take the
    /// discard landing, but for those virtual frames above it own.
    #[cold]
    #[inline(never)]
    fn member_run_doomed(
        &self,
        slot: u32,
        cidx: u32,
        first: u64,
        ctl: &ComputeCtl,
        payload: Option<Payload>,
    ) {
        cycle_state::take_unwind_kind();
        let mut bag = Deferred::new();
        if let Some(payload) = payload.and_then(|p| self.restore_contained(p)) {
            bag.push_erased(Box::new(Unbuilt::new(payload)));
        }
        if let Some(value) = ctl.buf.take() {
            bag.push_erased(Box::new(Unbuilt::new(value)));
        }
        let mut keep = Vec::new();
        self.virtually_owned(cidx + 1, &mut keep);
        self.release_held_except(cidx + 1, &keep, &mut bag);
        let old = cycle_state::with_held(|h| {
            h.get_mut(&(self.runtime_id, slot))
                .and_then(Held::discard_buf)
        });
        if let Some(old) = old {
            bag.push_erased(Box::new(old));
        }
        replay::unwind_past_since(self.runtime_id, cidx + 1, first);
        bag.drain(self);
    }

    /// A pop that found its frame `LOW`: defer the node, or run the
    /// episode it roots.
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn cycle_pop(
        &self,
        id: NodeId,
        generation: u32,
        cidx: u32,
        first: bool,
        ctl: &ComputeCtl,
        deps: Vec<(NodeId, u32)>,
        bag: &mut Deferred<C>,
    ) -> PopEnd {
        let rt = self.runtime_id;
        let lowlink = ctl.lowlink.get().min(cidx);
        let seeded = C::u8_load_relaxed(self.nodes.get_with_flags(id.0).1) & HAS_STRATEGY != 0;
        let mut held = Held::new(generation, lowlink, seeded, first);
        held.wrote = ctl.wrote.get() != 0;
        let (replaced, spec_buf) = held.record(
            Look::here(),
            deps,
            ctl.buf.take(),
            !ctl.children.get().is_null(),
            ctl.poison(),
        );
        debug_assert!(
            replaced.is_none() && spec_buf.is_none(),
            "incr-core: a new claim replaces nothing"
        );
        held.own(cidx, id.0, None);
        let displaced = cycle_state::with_held(|h| h.insert((rt, id.0), held));
        debug_assert!(displaced.is_none(), "incr-core: node {} held twice", id.0);
        if lowlink < cidx {
            self.pass_down(cidx, lowlink);
            return PopEnd::Deferred;
        }
        if value_code::thread_word() & value_code::REPLAY_BIT != 0 && replay::speculative() {
            // A speculative entry never runs an episode: the unbounded run
            // may enter this cycle at another member, or not at all. The
            // claims take the discard landing and the chain is dropped.
            self.release_held_from(cidx, true, Some(UnwindKind::Discard), bag);
            if self.drop_cooperatively(None) {
                return PopEnd::Dropped;
            }
            self.set_drop_kind();
            bag.drain_carried(Box::new(cycle::DiscardUnwind));
        }
        self.episode(id, cidx, ctl.has(cycle::WALKED), bag)
    }

    /// Every claim of this runtime held at or above `cidx`, re-anchored at
    /// `cidx`, in slot order.
    fn claim_members(&self, cidx: u32) -> Vec<u32> {
        let rt = self.runtime_id;
        let mut members: Vec<u32> = cycle_state::with_held(|h| {
            let members = h.anchored_from(rt, cidx);
            h.root_at(rt, &members, cidx);
            members
        });
        members.sort_unstable();
        members
    }

    /// The member-to-member reads of each member's last run, as indices
    /// into `members`.
    fn member_graph(&self, members: &[u32]) -> Vec<Vec<usize>> {
        let rt = self.runtime_id;
        let look = Look::here();
        let mut spec_bufs = Vec::new();
        let graph = cycle_state::with_held(|h| {
            members
                .iter()
                .map(|m| {
                    let (seen, spec_buf) = h
                        .get_mut(&(rt, *m))
                        .expect("incr-core: an episode member is held")
                        .look(look);
                    debug_assert!(
                        spec_buf.is_none(),
                        "incr-core: an episode member is settled before its graph"
                    );
                    spec_bufs.extend(spec_buf);
                    let mut out: Vec<usize> = seen
                        .last()
                        .iter()
                        .filter_map(|(d, _)| members.binary_search(&d.0).ok())
                        .collect();
                    out.sort_unstable();
                    out.dedup();
                    out
                })
                .collect()
        });
        self.drop_spec_bufs(spec_bufs);
        graph
    }

    /// Runs `f`, user code attributed to member `slot`: a panic marks the
    /// member failed and ends the episode.
    fn attributed<R>(
        &self,
        slot: u32,
        cidx: u32,
        bag: &mut Deferred<C>,
        f: impl FnOnce() -> R,
    ) -> R {
        match catch_unwind(AssertUnwindSafe(f)) {
            Ok(r) => r,
            Err(payload) => {
                cycle_state::with_held(|h| {
                    if let Some(e) = h.get_mut(&(self.runtime_id, slot)) {
                        cycle_state::mark_failed(e);
                    }
                });
                self.abort_episode(cidx, payload, bag)
            }
        }
    }

    /// Ends an episode with an unwind: every member is released as
    /// [`Self::release_held_from`] lands it, and the unwind resumes, or
    /// stops at the episode's boundary if the walk's entry rooting it
    /// contains it.
    fn abort_episode(&self, cidx: u32, payload: Payload, bag: &mut Deferred<C>) -> ! {
        let kind = cycle_state::take_unwind_kind();
        let contain = self.episode_contains(cidx, kind, &*payload);
        self.release_unwound_from(cidx, u32::MAX, kind, kind.is_some(), Some(&*payload), bag);
        if !contain {
            self.note_unwound_read(&*payload, kind);
        }
        if contain {
            self.raise_contained(cidx, payload);
        }
        self.leave_frame(kind, payload, bag)
    }

    /// Whether the walk's entry whose compute roots the episode at `cidx`,
    /// if one does, contains its abort, asked before the releases as
    /// `compute_panicked` asks. A contained abort drops the replay contexts
    /// the root's frame opened, as that frame's unwind would; the caller
    /// tells the reading frame of any other that its read unwound, after
    /// the landings it keeps a copy of.
    fn episode_contains(
        &self,
        cidx: u32,
        kind: Option<UnwindKind>,
        payload: &(dyn Any + Send),
    ) -> bool {
        let contain =
            contain_impl::walked_episode(cidx).is_some() && self.containable(kind, payload);
        if contain {
            replay::unwind_past(self.runtime_id, cidx);
        }
        contain
    }

    /// Drops a user value an episode no longer needs, attributing a
    /// panicking `Drop` to the member it belonged to.
    fn drop_member_value<V>(&self, slot: u32, cidx: u32, value: Option<V>, bag: &mut Deferred<C>) {
        if let Some(value) = value {
            self.attributed(slot, cidx, bag, || drop(value));
        }
    }

    /// Whether episode member `slot` has no buffer, a speculative run's
    /// settled first.
    fn unbuffered(&self, slot: u32) -> bool {
        let (spec_buf, none) = self.take_held(slot, |e| {
            let (seen, spec_buf) = e.look(Look::here());
            (spec_buf, seen.buf().is_none())
        });
        drop(spec_buf);
        none
    }

    /// `f` on held node `slot`'s run as `look` sees it, if held.
    pub(super) fn look_held<R>(
        &self,
        slot: u32,
        look: Look,
        f: impl FnOnce(&Seen<&mut Held>) -> R,
    ) -> Option<R> {
        let mut f = Some(f);
        loop {
            // What a settle took is dropped before `f` runs, which may run
            // user code; the claim's next look settles nothing.
            let seen = cycle_state::with_held(|h| {
                let Some(e) = h.get_mut(&(self.runtime_id, slot)) else {
                    return Ok(None);
                };
                match e.look(look) {
                    (seen, None) => Ok(Some(f.take().expect("look_held calls f once")(&seen))),
                    (_, Some(spec_buf)) => Err(spec_buf),
                }
            });
            match seen {
                Ok(seen) => return seen,
                Err(spec_buf) => self.drop_spec_bufs(Some(spec_buf)),
            }
        }
    }

    /// Drops what settles took off claims, outside the held map.
    fn drop_spec_bufs(&self, spec_bufs: impl IntoIterator<Item = Unbuilt>) {
        spec_bufs.into_iter().for_each(drop);
    }

    fn take_held<R>(&self, slot: u32, f: impl FnOnce(&mut Held) -> R) -> R {
        cycle_state::with_held(|h| {
            f(h.get_mut(&(self.runtime_id, slot))
                .expect("incr-core: an episode member is held"))
        })
    }

    /// Runs one member of episode `ep`, as its id and root, and folds its
    /// lowlink into `low`; `false` if a doom discarded the run, which
    /// suspends the episode.
    fn run_member(
        &self,
        slot: u32,
        cidx: u32,
        ep: (u64, u32),
        low: &mut u32,
        bag: &mut Deferred<C>,
    ) -> bool {
        replay::enter_member_run(cidx, ep);
        #[cfg(test)]
        self.episodes_seen
            .lock()
            .unwrap()
            .push(EpisodeSeen::Member {
                id: replay::member_run_episode(cidx),
                slot,
            });
        let ran = self.member_run(slot, cidx, true, false, false);
        replay::leave_member_run();
        drop(replay::run_ended(
            self.runtime_id,
            cidx,
            matches!(ran, Ok(None)),
        ));
        match ran {
            Ok(Some(l)) => {
                *low = (*low).min(l);
                true
            }
            Ok(None) => false,
            Err(payload) => self.abort_episode(cidx, payload, bag),
        }
    }

    /// The episode rooted at `root`, at conceptual-stack position `cidx`:
    /// iterate the members to a landing, defer the group to a lower root,
    /// or dissolve it. `walked` if the root's compute is a walk's entry.
    pub(super) fn episode(
        &self,
        root: NodeId,
        cidx: u32,
        walked: bool,
        bag: &mut Deferred<C>,
    ) -> PopEnd {
        #[cfg(test)]
        self.episodes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let rt = self.runtime_id;
        let ep = Box::new(Episode {
            id: replay::next_id(),
            root,
            cidx,
            validated: cycle_state::has_dissolved(rt, root.0),
            pruned: cycle_state::has_dissolved(rt, root.0),
            members: self.claim_members(cidx),
            cu: Vec::new(),
            round: None,
            walked,
            uncut: cycle_state::has_dissolved(rt, root.0),
        });
        #[cfg(test)]
        self.episodes_seen
            .lock()
            .unwrap()
            .push(EpisodeSeen::Start(ep.id));
        self.run_episode(ep, bag)
    }

    /// Runs episode `ep` on from where it stands. A member run that a doom
    /// discards suspends it in the context of that run's virtual frame,
    /// whose pop resumes it with the same member in the same iteration.
    ///
    /// An episode rooted at a walk's entry is where its contained abort
    /// stops, so that the entry's compute ends `Contained`.
    pub(super) fn run_episode(&self, ep: Box<Episode>, bag: &mut Deferred<C>) -> PopEnd {
        if !ep.walked {
            return self.episode_rounds(ep, bag);
        }
        let _walked = contain_impl::WalkedEpisode::enter(ep.cidx, ep.root);
        match catch_unwind(AssertUnwindSafe(|| self.episode_rounds(ep, bag))) {
            Ok(end) => end,
            Err(payload) => {
                self.contain_unwind(payload, bag);
                PopEnd::Contained
            }
        }
    }

    fn episode_rounds(&self, mut ep: Box<Episode>, bag: &mut Deferred<C>) -> PopEnd {
        let rt = self.runtime_id;
        let cidx = ep.cidx;
        let id = ep.id;
        let root = ep.root.0;
        'restart: loop {
            if ep.round.is_none() {
                let cycle =
                    Cycle::new(ep.members.iter().map(|&m| self.erased_at_live(m)).collect());
                let strategies: Vec<(u32, Arc<dyn ErasedStrategy>)> = {
                    let inner = self.inner.read();
                    ep.members
                        .iter()
                        .filter_map(|&m| inner.cycle_strategies.get(&m).map(|s| (m, s.clone())))
                        .collect()
                };
                let limit = strategies
                    .iter()
                    .filter_map(|(_, s)| s.max_iterations())
                    .min()
                    .unwrap_or(1);
                let unseeded: Vec<u32> = ep
                    .members
                    .iter()
                    .copied()
                    .filter(|m| strategies.binary_search_by_key(m, |(s, _)| *s).is_err())
                    .collect();
                // A restart clears what the members installed, which a later
                // settle would put back.
                let look = Look::here();
                let settled: Vec<Unbuilt> = cycle_state::with_held(|h| {
                    h.root_at(rt, &ep.members, cidx);
                    ep.members
                        .iter()
                        .filter_map(|m| {
                            let (mut e, spec_buf) = h
                                .get_mut(&(rt, *m))
                                .expect("incr-core: an episode member is held")
                                .look(look);
                            e.episode = Some(cycle.clone());
                            e.ep = id;
                            e.in_cu = ep.cu.binary_search(m).is_ok();
                            e.clear_installed();
                            spec_buf
                        })
                        .collect()
                });
                drop(settled);
                // Every claim precedes this capture, because membership growth
                // restarts the episode.
                let stamp_rev = self.current_revision();
                let verify_rev = C::walked(&self.walk, &self.revision);
                for (m, s) in &strategies {
                    let v0 =
                        self.attributed(*m, cidx, bag, || self.callback(|| s.v0(Some(&cycle))));
                    let old = self.take_held(*m, |e| e.s.replace(v0));
                    self.drop_member_value(*m, cidx, old, bag);
                }
                // s_{k-1} of each Fixpoint member, for the convergence test and
                // the landing pass.
                let prev = (0..strategies.len()).map(|_| None).collect();
                ep.round = Some(Round {
                    cycle,
                    strategies,
                    limit,
                    unseeded,
                    stamp_rev,
                    verify_rev,
                    prev,
                    k: 1,
                    low: cidx,
                    step: Step::Clear,
                });
            }
            let r = ep
                .round
                .as_mut()
                .expect("incr-core: the episode has a round");
            loop {
                match &mut r.step {
                    // 1. Clear the unseeded buffers.
                    Step::Clear => {
                        for &m in &r.unseeded {
                            let old = self.take_held(m, Held::discard_buf);
                            self.drop_member_value(m, cidx, old, bag);
                        }
                        r.low = cidx;
                        r.step = Step::Seeded(0);
                    }
                    // 2. Run each seeded member once, in slot order.
                    Step::Seeded(i) => {
                        let i = *i;
                        let Some(&(m, _)) = r.strategies.get(i) else {
                            r.step = Step::Unseeded(0);
                            continue;
                        };
                        if !self.run_member(m, cidx, (id, root), &mut r.low, bag) {
                            return self.suspend_episode(ep);
                        }
                        r.step = Step::Seeded(i + 1);
                    }
                    // 3. Run the unseeded members no read has run.
                    Step::Unseeded(i) => {
                        let i = *i;
                        let Some(&m) = r.unseeded.get(i) else {
                            r.step = Step::Check;
                            continue;
                        };
                        if self.unbuffered(m)
                            && !self.run_member(m, cidx, (id, root), &mut r.low, bag)
                        {
                            return self.suspend_episode(ep);
                        }
                        r.step = Step::Unseeded(i + 1);
                    }
                    Step::Check => {
                        // 6. A run reached below the root: the group belongs to a
                        // lower root's cycle.
                        if r.low < cidx {
                            self.merge_down(cidx, r.low, bag);
                            return PopEnd::Deferred;
                        }
                        // 5. A non-member that read the group joined it.
                        if self.join_new(cidx, &mut ep.members) {
                            ep.round = None;
                            continue 'restart;
                        }
                        let graph = self.member_graph(&ep.members);
                        if !ep.validated {
                            ep.validated = true;
                            #[cfg(test)]
                            self.validations
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            match self.shrink_to_root_scc(
                                ep.root,
                                &mut ep.members,
                                &mut ep.cu,
                                &graph,
                                bag,
                            ) {
                                Shrink::Kept => {}
                                Shrink::Shrunk => {
                                    ep.round = None;
                                    continue 'restart;
                                }
                                Shrink::Dissolved => return PopEnd::Dissolved,
                            }
                        }
                        // CU: the unseeded members that still read each other.
                        let un_idx: Vec<usize> = r
                            .unseeded
                            .iter()
                            .map(|m| {
                                ep.members
                                    .binary_search(m)
                                    .expect("an unseeded member is a member")
                            })
                            .collect();
                        let sub: Vec<Vec<usize>> = un_idx
                            .iter()
                            .map(|&i| {
                                graph[i]
                                    .iter()
                                    .filter_map(|j| un_idx.binary_search(j).ok())
                                    .collect()
                            })
                            .collect();
                        let cyclic = in_cycles(&sub);
                        let mut grew = false;
                        for (i, &m) in r.unseeded.iter().enumerate() {
                            if cyclic[i] {
                                if let Err(at) = ep.cu.binary_search(&m) {
                                    ep.cu.insert(at, m);
                                    grew = true;
                                }
                            }
                        }
                        if grew {
                            ep.round = None;
                            continue 'restart;
                        }

                        // 4. The strategy members' next values.
                        let mut converged = vec![true; r.strategies.len()];
                        for (i, (m, s)) in r.strategies.iter().enumerate() {
                            let (out, spec_buf) = self.take_held(*m, |e| {
                                let (mut seen, spec_buf) = e.look(Look::here());
                                (seen.take_buf(), spec_buf)
                            });
                            debug_assert!(
                                spec_buf.is_none(),
                                "incr-core: a strategy member is settled before its step"
                            );
                            self.drop_member_value(*m, cidx, spec_buf, bag);
                            if s.max_iterations().is_none() {
                                self.drop_member_value(*m, cidx, out, bag);
                                continue;
                            }
                            let out = out.expect("incr-core: a member run buffers its value");
                            let s_prev = self
                                .take_held(*m, |e| e.s.take())
                                .expect("incr-core: a strategy member has a value");
                            let k = r.k;
                            let next = self.attributed(*m, cidx, bag, || {
                                self.callback(|| s.step(&s_prev, out, k))
                            });
                            converged[i] = self.attributed(*m, cidx, bag, || {
                                self.callback(|| s.converged(&s_prev, &next))
                            });
                            self.take_held(*m, |e| e.s = Some(next));
                            let older = r.prev[i].replace(s_prev);
                            self.drop_member_value(*m, cidx, older, bag);
                        }
                        if !(converged.iter().all(|&c| c) || r.k >= r.limit) {
                            r.k += 1;
                            r.step = Step::Clear;
                            continue;
                        }
                        for (i, (m, s)) in r.strategies.iter().enumerate() {
                            if !converged[i] {
                                let last = self
                                    .take_held(*m, |e| e.s.take())
                                    .expect("incr-core: a strategy member has a value");
                                let landed = self.attributed(*m, cidx, bag, || {
                                    self.callback(|| s.on_limit(&r.cycle, last))
                                });
                                self.take_held(*m, |e| e.s = Some(landed));
                            }
                        }

                        // The landing pass: unseeded members outside CU computed
                        // against s_{K-1} run once more against s*, unless no
                        // Fixpoint member moved.
                        let moved = r.strategies.iter().enumerate().any(|(i, (m, s))| {
                            r.prev[i].as_ref().is_some_and(|p| {
                                let cur = self.take_held(*m, |e| e.s.take());
                                let same = cur.as_ref().is_some_and(|c| {
                                    self.attributed(*m, cidx, bag, || {
                                        value_code::run(|| s.same(p, c))
                                    })
                                });
                                self.take_held(*m, |e| e.s = cur);
                                !same
                            })
                        });
                        let prev = std::mem::take(&mut r.prev);
                        for (i, p) in prev.into_iter().enumerate() {
                            self.drop_member_value(r.strategies[i].0, cidx, p, bag);
                        }
                        if !moved {
                            break;
                        }
                        let outside: Vec<u32> = r
                            .unseeded
                            .iter()
                            .copied()
                            .filter(|m| ep.cu.binary_search(m).is_err())
                            .collect();
                        for &m in &outside {
                            let old = self.take_held(m, Held::discard_buf);
                            self.drop_member_value(m, cidx, old, bag);
                        }
                        r.low = cidx;
                        r.step = Step::Landing { outside, next: 0 };
                    }
                    Step::Landing { outside, next } => {
                        let Some(&m) = outside.get(*next) else {
                            if r.low < cidx {
                                self.merge_down(cidx, r.low, bag);
                                return PopEnd::Deferred;
                            }
                            if self.join_new(cidx, &mut ep.members) {
                                ep.round = None;
                                continue 'restart;
                            }
                            break;
                        };
                        *next += 1;
                        if self.unbuffered(m)
                            && !self.run_member(m, cidx, (id, root), &mut r.low, bag)
                        {
                            *next -= 1;
                            return self.suspend_episode(ep);
                        }
                    }
                }
            }
            // Validation cuts to the SCC of iteration 1's reads only, so a
            // member that joined later, or whose reads moved, may be on no
            // cycle in the final reads. One more cut per episode keeps
            // growth the only change after it, so episodes still terminate.
            if !ep.pruned {
                ep.pruned = true;
                let graph = self.member_graph(&ep.members);
                match self.shrink_to_root_scc(ep.root, &mut ep.members, &mut ep.cu, &graph, bag) {
                    Shrink::Kept => {}
                    Shrink::Shrunk => {
                        ep.uncut = true;
                        ep.round = None;
                        continue 'restart;
                    }
                    Shrink::Dissolved => return PopEnd::Dissolved,
                }
            }
            let r = ep.round.take().expect("incr-core: the episode has a round");
            return self.land_episode(
                &ep.members,
                &ep.cu,
                ep.uncut,
                r.cycle,
                r.stamp_rev,
                r.verify_rev,
                cidx,
                bag,
            );
        }
    }

    /// Cuts `ep` to the root's SCC in `graph`, the member graph of the
    /// last runs, releasing the rest with the discard landing. A root
    /// alone with no self-edge dissolves. The members kept leave a group
    /// the cut released others from ([`Self::leave_cut_group`]).
    fn shrink_to_root_scc(
        &self,
        root: NodeId,
        members: &mut Vec<u32>,
        cu: &mut Vec<u32>,
        graph: &[Vec<usize>],
        bag: &mut Deferred<C>,
    ) -> Shrink {
        let rt = self.runtime_id;
        let comp = components(graph);
        let at = members
            .binary_search(&root.0)
            .expect("the root is a member");
        let scc: Vec<u32> = members
            .iter()
            .enumerate()
            .filter(|(i, _)| comp[*i] == comp[at])
            .map(|(_, &m)| m)
            .collect();
        if scc.len() == 1 && !graph[at].contains(&at) {
            cycle_state::dissolved_before(rt, root.0);
            self.release_members(root, members, Some(root.0), bag);
            self.leave_cut_group(&[root.0]);
            return Shrink::Dissolved;
        }
        if scc.len() == members.len() {
            return Shrink::Kept;
        }
        let gone: Vec<u32> = members
            .iter()
            .copied()
            .filter(|m| scc.binary_search(m).is_err())
            .collect();
        self.release_members(root, &gone, None, bag);
        self.leave_cut_group(&scc);
        cu.retain(|m| scc.binary_search(m).is_ok());
        *members = scc;
        Shrink::Shrunk
    }

    /// Ends the claim the root of a dissolved episode holds as a member,
    /// returning whether it was taken from New and whether it wrote. A root
    /// still in a landed group has that record noted, so this entry's retry
    /// of the root computes it alone.
    pub(super) fn end_dissolved(&self, root: NodeId, bag: &mut Deferred<C>) -> (bool, bool) {
        let e = cycle_state::with_held(|h| h.remove(&(self.runtime_id, root.0)))
            .expect("incr-core: a dissolved root is held");
        if C::u8_load_relaxed(self.nodes.get_with_flags(root.0).1) & IN_GROUP != 0 {
            let group = self.inner.read().member_recs.get(&root.0).cloned();
            if let Some(group) = group {
                cycle_state::note_dissolved_group(self.runtime_id, root.0, group);
            }
        }
        let ended = (e.first, e.wrote);
        bag.push_erased(Box::new(e.into_values()));
        ended
    }

    /// Stores episode `ep`, whose member run a doom discarded, in the
    /// context that holds that run's virtual frame.
    fn suspend_episode(&self, ep: Box<Episode>) -> PopEnd {
        #[cfg(test)]
        self.episodes_seen
            .lock()
            .unwrap()
            .push(EpisodeSeen::of(&ep, None));
        replay::suspend(self.runtime_id, ep);
        PopEnd::Suspended
    }

    /// Adds every claim held at or above `cidx` that is not yet a member.
    fn join_new(&self, cidx: u32, members: &mut Vec<u32>) -> bool {
        let rt = self.runtime_id;
        let joined: Vec<u32> = cycle_state::with_held(|h| {
            let mut joined = h.anchored_from(rt, cidx);
            joined.retain(|slot| members.binary_search(slot).is_err());
            h.root_at(rt, &joined, cidx);
            joined
        });
        if joined.is_empty() {
            return false;
        }
        members.extend(joined);
        members.sort_unstable();
        true
    }

    /// Defers the whole group to the root of the lower cycle a member
    /// run reached: every member is re-anchored at `low` and leaves the
    /// episode, and the root's reader inherits the lowlink.
    fn merge_down(&self, cidx: u32, low: u32, bag: &mut Deferred<C>) {
        let rt = self.runtime_id;
        let values: Vec<Erased> = cycle_state::with_held(|h| {
            h.anchored_from(rt, cidx)
                .into_iter()
                .filter_map(|m| {
                    h.set_anchor((rt, m), low);
                    let e = h.get_mut(&(rt, m)).expect("the slot was just listed");
                    e.episode = None;
                    e.ep = 0;
                    e.in_cu = false;
                    e.s.take()
                })
                .collect()
        });
        for v in values {
            bag.push_erased(v);
        }
        self.pass_down(cidx, low);
    }

    /// The frame at `cidx` popped deferred with lowlink `low`: the claims
    /// its subtree deferred re-anchor at `low` too, not at a later frame
    /// at their position, and pass to its parent, which inherits `low`.
    pub(super) fn pass_down(&self, cidx: u32, low: u32) {
        let rt = self.runtime_id;
        let (parent, node, virtual_frame) = self.lower_parent(cidx, low);
        let path = self.virtual_path_at(cidx);
        cycle_state::with_held(|h| {
            for m in h.anchored_from(rt, cidx) {
                h.set_anchor((rt, m), low);
            }
            let mut owned = h.owners(rt, cidx, u32::MAX);
            owned.retain(|&g| h.owner(g).owned_by(cidx, u32::MAX, path));
            h.hand_over(rt, &owned, &[], parent, node, virtual_frame);
        });
    }

    /// Takes each of `kept`, members a cut keeps claimed, out of a group
    /// whose record names a member that is no longer in it, so that the
    /// cut leaves no such record. They leave unverified, as a member
    /// landing Dirty out of its group does, since a wound lands them Dirty.
    /// Under `Shared` a cut is an exposure: the record is kept for a
    /// retraction, which puts the whole group back ([`hold::keep_cut_rec`]).
    fn leave_cut_group(&self, kept: &[u32]) {
        for &m in kept {
            let (node, flags) = self.nodes.get_with_flags(m);
            if C::u8_load_relaxed(flags) & IN_GROUP != 0 && self.names_departed(m) {
                if !C::SINGLE_THREADED && !hold::holds(state::cell_key::<C>(node.state_cell())) {
                    let rt = self.runtime_id;
                    let lands_new = cycle_state::with_held(|h| {
                        h.get(&(rt, m)).is_some_and(|e| e.first || e.wrote)
                    });
                    let rec = self.inner.read().member_recs.get(&m).cloned();
                    hold::keep_cut_rec(rt, m, rec, lands_new);
                }
                self.leave_group(NodeId(m), node, flags, NodeState::Dirty);
            }
        }
    }

    /// Whether `slot`'s group record names a member that is no longer in
    /// it.
    fn names_departed(&self, slot: u32) -> bool {
        let inner = self.inner.read();
        let Some(group) = inner.member_recs.get(&slot) else {
            return false;
        };
        group.members.iter().any(|&(m, _)| {
            !inner
                .member_recs
                .get(&m)
                .is_some_and(|g| Arc::ptr_eq(g, group))
        })
    }

    /// Counts a release of held claims, before they land, for the walks
    /// other threads planned while they were held: a claim that reads one
    /// of the landings also reads the count.
    pub(super) fn count_release(&self) {
        if !C::SINGLE_THREADED {
            cycle_state::own_episode(self.runtime_id, RELEASED);
            C::u64_fetch_add(&self.episodes_landed, RELEASED);
        }
    }

    /// Releases `slots` from the held map with the discard landing,
    /// except `keep`. They leave their group, as the unbounded run's
    /// episode at `root` cuts them at the same point.
    fn release_members(
        &self,
        root: NodeId,
        slots: &[u32],
        keep: Option<u32>,
        bag: &mut Deferred<C>,
    ) {
        let rt = self.runtime_id;
        let stamp_rev = self.current_revision();
        if slots.iter().any(|&m| Some(m) != keep) {
            self.count_release();
        }
        for &m in slots {
            if Some(m) == keep {
                continue;
            }
            let entry = cycle_state::with_held(|h| h.remove(&(rt, m)));
            if let Some(e) = entry {
                // A cut is an exposure: its released members stay Computing to
                // other threads until the attempt ends (`crate::hold`), and
                // leave their group at once, as under `Local`, with the record
                // kept for a retraction.
                if !C::SINGLE_THREADED {
                    self.open_hold(NodeId(m), self.nodes.get(m), e.first, None, bag);
                }
                self.discard_land(NodeId(m), e.first, e.wrote, stamp_rev, bag);
                bag.push_erased(Box::new(e.into_values()));
            }
        }
        self.park(ParkPoint::MembersReleased(root));
    }

    /// Lands every claim of this runtime held at or above `cidx`: with
    /// `discarding`, each takes the discard landing; otherwise, under a
    /// user panic, a claim whose own code panicked takes fix 5's panic
    /// path and every other claim [`Self::land_panic_released`].
    pub(super) fn release_held_from(
        &self,
        cidx: u32,
        discarding: bool,
        kind: Option<UnwindKind>,
        bag: &mut Deferred<C>,
    ) {
        self.release_held_where(cidx, discarding, kind, None, &[], None, bag);
    }

    /// [`Self::release_held_from`] for a frame at `cidx` that unwound with
    /// `kind`: the claims its subtree deferred to a root below it go too.
    /// A doom's discard keeps those, for the frame's retry to read, and so
    /// does a speculative run's unwind, which the unbounded run may never
    /// make: another frame at the position may own them. A discard also
    /// keeps what virtual frames at or above it own
    /// ([`Self::virtually_owned`]), as a doomed frame's does. `payload` is
    /// the unwind's, which a claim landing Failed lets out.
    pub(super) fn release_unwound_from(
        &self,
        cidx: u32,
        node: u32,
        kind: Option<UnwindKind>,
        discarding: bool,
        payload: Option<&(dyn std::any::Any + Send)>,
        bag: &mut Deferred<C>,
    ) -> Option<reserve::Peers> {
        let mut keep = Vec::new();
        if kind == Some(UnwindKind::Discard) {
            self.virtually_owned(cidx, &mut keep);
        }
        let subtree = (kind != Some(UnwindKind::Discard) && !replay::speculative()).then_some(node);
        self.release_held_where(cidx, discarding, kind, subtree, &keep, payload, bag)
    }

    /// [`Self::release_held_from`] for a doomed frame's discard, keeping
    /// the claims in `keep`.
    pub(super) fn release_held_except(&self, cidx: u32, keep: &[u32], bag: &mut Deferred<C>) {
        self.release_held_where(cidx, true, Some(UnwindKind::Discard), None, keep, None, bag);
    }

    #[allow(clippy::too_many_arguments)]
    fn release_held_where(
        &self,
        cidx: u32,
        discarding: bool,
        kind: Option<UnwindKind>,
        subtree: Option<u32>,
        keep: &[u32],
        payload: Option<&(dyn std::any::Any + Send)>,
        bag: &mut Deferred<C>,
    ) -> Option<reserve::Peers> {
        // The unwinding frame, of node `subtree`, owns the claims owned at
        // its position by it, and those owned above.
        let path = subtree.and_then(|_| self.virtual_path_at(cidx));
        let rt = self.runtime_id;
        // A suspended episode's members wait for its member run's virtual
        // frame, whatever frame runs at that position meanwhile: only the
        // episode's resumption, or its context abandoning it, releases them.
        // So do the claims its member runs deferred to its root, which the
        // resumption joins.
        let mut entries: Vec<(u32, Held)> = cycle_state::with_held(|h| {
            if h.is_empty() {
                return Vec::new();
            }
            let mut slots = h.anchored_from(rt, cidx);
            if let Some(node) = subtree {
                for g in h.owners(rt, cidx, u32::MAX) {
                    if h.owner(g).owned_by(cidx, node, path) {
                        slots.extend_from_slice(h.members(g));
                    }
                }
                slots.sort_unstable();
                slots.dedup();
            }
            slots.retain(|m| {
                !keep.contains(m) && h.get(&(rt, *m)).is_some_and(|e| released_at(rt, e, cidx))
            });
            slots
                .into_iter()
                .map(|m| (m, h.remove(&(rt, m)).expect("the slot was just listed")))
                .collect()
        });
        if entries.is_empty() {
            return None;
        }
        self.count_release();
        entries.sort_unstable_by_key(|(slot, _)| *slot);
        let released: Vec<u32> = entries.iter().map(|(slot, _)| *slot).collect();
        // One episode: the unwinding frame's node and what it released.
        let peers = (!discarding && kind.is_none() && self.lists_walks()).then(|| {
            reserve::Peers::new(
                released
                    .iter()
                    .copied()
                    .chain(subtree.filter(|&n| n != u32::MAX)),
            )
        });
        let stamp_rev = self.current_revision();
        let verify_rev = C::walked(&self.walk, &self.revision);
        let look = Look::here();
        for (slot, e) in entries {
            let (e, spec_buf) = e.into_look(look);
            if let Some(spec_buf) = spec_buf {
                bag.push_erased(Box::new(spec_buf));
            }
            let id = NodeId(slot);
            let attempt = Attempt {
                first: e.first,
                cidx,
                stamp_rev,
                verify_rev,
            };
            if discarding {
                self.unwound_land(id, e.first, e.wrote, stamp_rev, kind, bag);
            } else if e.failed {
                let (node, flags) = self.nodes.get_with_flags(slot);
                self.land_failed(
                    id,
                    node,
                    flags,
                    attempt,
                    e.wrote,
                    e.deps(),
                    &released,
                    payload,
                    false,
                    bag,
                );
            } else {
                self.land_panic_released(id, attempt, e.wrote, e.deps(), &released, bag);
            }
            // A cycle's retry may run a member again; a user panic ends its run.
            if !discarding && kind.is_none() {
                if let Some(peers) = &peers {
                    self.note_unwound_landing(id, self.nodes.get(slot), peers);
                }
                self.end_reservation(id, self.nodes.get(slot));
            }
            // A discarded claim's run is one the unbounded run never makes,
            // or makes again on the retry.
            if kind == Some(UnwindKind::Discard) {
                bag.push_erased(Box::new(Unbuilt::new(Box::new(e.into_values()))));
            } else {
                bag.push_erased(Box::new(e.into_values()));
            }
        }
        // A user panic's release is one the unbounded run makes too, and it
        // ends the runs of the released nodes' frames: a delivery, claim
        // watermark or failure count kept for a retry of one of them would
        // reach the node's next, fresh run at that position. A cycle unwind's release keeps them, since the
        // doomed run's retry reads the cycle from them again, as do a doom's
        // discard and a wound.
        if kind.is_none() {
            self.end_released_runs(&released);
        }
        peers
    }

    /// The discard landing of a claim this thread holds: the node lands as
    /// if the attempt never ran. A claim taken from New lands New, and
    /// stores `verified_at = 0` if it wrote nothing (rule (a)); one that
    /// wrote its slot is stamped and lands New, so that its retry writes
    /// and stamps unconditionally; any other lands Dirty. Recorded deps
    /// are dropped and installed ones kept. It never lands Failed. The node
    /// leaves its group, so callers use it only for an attempt the unbounded
    /// run releases the same way.
    pub(super) fn discard_land(
        &self,
        id: NodeId,
        first: bool,
        wrote: bool,
        stamp_rev: u64,
        bag: &mut Deferred<C>,
    ) {
        let target = self.discard_target(id, first, wrote, stamp_rev, ChildLanding::Discard, bag);
        let (node, flags) = self.nodes.get_with_flags(id.0);
        self.land_single_as(id, node, flags, target, target, true);
    }

    /// [`Self::discard_land`] of a claim released by an unwind of `kind`.
    /// A wound's and a T8 discard's keep the node in its group, as
    /// `unclaim_group` does: neither attempt is in the unbounded run or in
    /// any single-thread order, and a member it took out would make another
    /// thread compute alone what that order verifies as a group. A user
    /// panic or a cycle error outside a speculative run unwinds the
    /// unbounded run too, so its claims leave their group.
    pub(super) fn unwound_land(
        &self,
        id: NodeId,
        first: bool,
        wrote: bool,
        stamp_rev: u64,
        kind: Option<UnwindKind>,
        bag: &mut Deferred<C>,
    ) {
        if matches!(kind, Some(UnwindKind::Wound | UnwindKind::Discard)) {
            self.discard_land_in_group(id, first, wrote, stamp_rev, ChildLanding::Discard, bag);
            // After the landing, which puts a wound's replaced records back.
            #[cfg(test)]
            {
                let dissolved = cycle_state::has_dissolved(self.runtime_id, id.0);
                let stale = self.names_departed(id.0);
                let grouped = C::u8_load_relaxed(self.nodes.get_with_flags(id.0).1) & IN_GROUP != 0;
                let mut landings = self.unwound_landings.lock().unwrap();
                landings.push((id.0, dissolved, stale, grouped));
            }
        } else {
            self.discard_land(id, first, wrote, stamp_rev, bag);
        }
    }

    /// [`Self::discard_land`] landing the attempt's children as `children`
    /// and leaving the node's `MemberRec` and `IN_GROUP` as they were.
    pub(super) fn discard_land_in_group(
        &self,
        id: NodeId,
        first: bool,
        wrote: bool,
        stamp_rev: u64,
        children: ChildLanding,
        bag: &mut Deferred<C>,
    ) {
        let target = self.discard_target(id, first, wrote, stamp_rev, children, bag);
        self.land_unrun(id, self.nodes.get(id.0), target);
    }

    /// Lands the discarded attempt's children as `children` and stamps the
    /// node for its discard landing, which lands it in the state returned.
    pub(super) fn discard_target(
        &self,
        id: NodeId,
        first: bool,
        wrote: bool,
        stamp_rev: u64,
        children: ChildLanding,
        bag: &mut Deferred<C>,
    ) -> NodeState {
        let (node, flags) = self.nodes.get_with_flags(id.0);
        self.children_landing(id, flags, children, false, bag);
        if wrote && replay::speculative() {
            self.unwrite_speculative(node, bag);
            node.max_changed_at(stamp_rev);
            node.set_verified_at(0);
            NodeState::New
        } else if wrote {
            node.max_changed_at(stamp_rev);
            NodeState::New
        } else if first {
            node.set_verified_at(0);
            NodeState::New
        } else {
            NodeState::Dirty
        }
    }

    /// Takes back the value a discarded speculative run wrote into `node`'s
    /// slot, which the unbounded run may never build, so that no later
    /// write displaces it into a real run's bag. The node lands New with an
    /// empty slot, as a first compute that wrote nothing.
    #[cold]
    #[inline(never)]
    fn unwrite_speculative(&self, node: &NodeData<C>, bag: &mut Deferred<C>) {
        let arena = self.arenas.erased(node.type_tag());
        if let Some(value) = arena.clear_unbuilt(node.arena_slot()) {
            bag.push_erased(Box::new(value));
        }
    }

    /// Lands held node `slot` from its last run's buffer, as
    /// `compute_attempts` lands a run that popped neither LOW nor rooting a
    /// claim: a virtual hit's claim stood for its frame on the stack, and
    /// that frame turned out to be no cycle's member. `revs` are the stamp
    /// and verify revisions taken before the run. `Ok` with the state the
    /// node landed in, or `None` if a dep went stale, so that the node
    /// landed New for a retry; `Err` if writing the value panicked, with
    /// the claim still held.
    pub(super) fn land_alone(
        &self,
        slot: u32,
        (stamp_rev, verify_rev): (u64, u64),
        bag: &mut Deferred<C>,
    ) -> Result<Option<NodeState>, Payload> {
        let rt = self.runtime_id;
        let id = NodeId(slot);
        let (node, flags) = self.nodes.get_with_flags(slot);
        let e = cycle_state::with_held(|h| h.remove(&(rt, slot)))
            .expect("incr-core: a landing claim is held");
        debug_assert!(
            !e.spec_written() || replay::speculative(),
            "incr-core: a held node lands a speculative run's buffer"
        );
        let (mut e, spec_buf) = e.into_look(Look::here());
        if let Some(spec_buf) = spec_buf {
            bag.push_erased(Box::new(spec_buf));
        }
        let value = e.take_buf().expect("incr-core: a finished run has a value");
        let arena = self.arenas.erased(node.type_tag());
        let wrote = Cell::new(0_u8);
        let first = e.first;
        // A cut's claim lands held: its entry opens before the write, so
        // the undo keeps what the write displaces.
        if !C::SINGLE_THREADED && hold::cut_kept(rt, slot) {
            self.open_hold(id, node, first, None, bag);
        }
        let written = catch_unwind(AssertUnwindSafe(|| {
            arena.write_erased(node.arena_slot(), value, first, &wrote)
        }));
        e.wrote |= wrote.get() != 0;
        let changed = match written {
            Ok((changed, displaced)) => {
                self.displaced(node, displaced, changed || first, bag);
                changed
            }
            Err(payload) => {
                cycle_state::with_held(|h| h.insert((rt, slot), e.into_held()));
                return Err(payload);
            }
        };
        if changed || first {
            node.max_changed_at(stamp_rev);
        }
        let published = self.publish_deps(id, e.last(), None, verify_rev, bag);
        let landed = if published == Publish::Stale {
            if first && !e.wrote {
                node.set_verified_at(0);
            }
            self.children_landing(id, flags, ChildLanding::Discard, e.opened(), bag);
            // Leaving the group is safe: a New node's record is inert.
            self.land_single(id, node, flags, NodeState::New, NodeState::New);
            None
        } else {
            let poison = e.poison();
            if self.memo_landing(poison, verify_rev) {
                node.set_memo(verify_rev);
            } else if poison != 0 {
                node.set_verified_at(0);
            } else {
                node.max_verified_at(verify_rev);
            }
            let (target, children) = if published == Publish::Superseded || poison != 0 {
                (NodeState::Dirty, ChildLanding::Keep)
            } else {
                (NodeState::Clean, ChildLanding::Sweep)
            };
            self.children_landing(id, flags, children, e.opened(), bag);
            // The run computed the node alone, as the frame's run in the
            // unbounded run does, so the node leaves its group.
            let landed = self.land_single(id, node, flags, target, NodeState::Dirty);
            self.end_reservation_landed(id, node, landed);
            self.record_trace(
                id,
                crate::trace::TraceAction::Recomputed {
                    value_changed: changed,
                },
            );
            Some(landed)
        };
        bag.push_erased(Box::new(e.into_values()));
        Ok(landed)
    }

    /// Lands an episode whose claims are complete (the plan's Landing
    /// steps 2 to 5; step 1 is the iteration that produced the values).
    #[allow(clippy::too_many_arguments)]
    fn land_episode(
        &self,
        members: &[u32],
        cu: &[u32],
        uncut: bool,
        cycle: Cycle,
        stamp_rev: u64,
        verify_rev: u64,
        cidx: u32,
        bag: &mut Deferred<C>,
    ) -> PopEnd {
        let landing = EpisodeLanding {
            entries: self.take_episode(members, bag),
            members,
            cu,
            uncut,
            cycle,
            stamp_rev,
            verify_rev,
        };
        self.publish_episode(landing, cidx, bag)
    }

    /// Takes an episode's members out of the held map, as seen here.
    fn take_episode(&self, members: &[u32], bag: &mut Deferred<C>) -> Vec<(u32, Seen<Held>)> {
        let rt = self.runtime_id;
        let entries: Vec<(u32, Held)> = cycle_state::with_held(|h| {
            members
                .iter()
                .map(|&m| {
                    (
                        m,
                        h.remove(&(rt, m))
                            .expect("incr-core: an episode member is held"),
                    )
                })
                .collect()
        });
        debug_assert!(
            replay::speculative() || entries.iter().all(|(_, e)| !e.spec_written()),
            "incr-core: an episode lands a speculative run's record"
        );
        let look = Look::here();
        entries
            .into_iter()
            .map(|(m, e)| {
                let (e, spec_buf) = e.into_look(look);
                if let Some(spec_buf) = spec_buf {
                    bag.push_erased(Box::new(spec_buf));
                }
                (m, e)
            })
            .collect()
    }

    /// Steps 2 to 5 of `landing`: validates its members, installs their
    /// deps and group record, writes their values and lands them.
    fn publish_episode(
        &self,
        landing: EpisodeLanding<'_>,
        cidx: u32,
        bag: &mut Deferred<C>,
    ) -> PopEnd {
        let EpisodeLanding {
            members,
            cu,
            uncut,
            cycle,
            stamp_rev,
            verify_rev,
            mut entries,
        } = landing;

        // Step 2: validate, install and record the group under one hold.
        let mut inner = self.inner.write();
        let stale = entries.iter().any(|(_, e)| {
            e.deps().iter().any(|&(d, generation)| {
                let dep = self.nodes.get(d.0);
                dep.generation() != generation || dep.state() == NodeState::Dead
            })
        });
        if stale {
            drop(inner);
            self.count_release();
            for (m, e) in entries {
                let (node, flags) = self.nodes.get_with_flags(m);
                if e.first {
                    node.set_verified_at(0);
                }
                self.children_landing(NodeId(m), flags, ChildLanding::Discard, false, bag);
                // Leaving the group is safe: a New member's record is inert,
                // and the root's retry records the group again.
                self.land_single(NodeId(m), node, flags, NodeState::New, NodeState::New);
                bag.push_erased(Box::new(e.into_values()));
            }
            return PopEnd::Retry;
        }
        // Under Shared the episode is held until the attempt ends
        // (`crate::hold`), which counts it then.
        if !C::SINGLE_THREADED {
            hold::note_episode(self.runtime_id);
            for (m, e) in &entries {
                self.open_hold(NodeId(*m), self.nodes.get(*m), e.first, None, bag);
            }
        }
        let mut superseded = false;
        for (m, e) in &entries {
            let id = NodeId(*m);
            let node = self.nodes.get(*m);
            // Before the deps compare: the held branch below continues past the loop end.
            self.drop_landed_entry(&mut inner, *m, node);
            let mut old: Vec<NodeId> = Vec::new();
            node.for_each_dep(|d| old.push(d));
            let new: &[NodeId] = e.installed();
            superseded |= new.iter().any(|d| {
                members.binary_search(&d.0).is_err()
                    && !old.contains(d)
                    && !self.observed_clean(*d, verify_rev)
            });
            if old != new {
                if !C::SINGLE_THREADED {
                    hold::keep_deps(state::cell_key::<C>(node.state_cell()), || {
                        old.iter()
                            .map(|d| (*d, self.nodes.get(d.0).generation()))
                            .collect()
                    });
                    if let Some(displaced) = node.install_deps(new) {
                        bag.push_dep_list(displaced);
                    }
                    continue;
                }
                if let Some(displaced) = node.install_deps(new) {
                    bag.push_dep_list(displaced);
                }
                for o in &old {
                    if !new.contains(o) {
                        inner.dependents[o.0 as usize].remove(id);
                    }
                }
                for n in new {
                    if !old.contains(n) {
                        inner.dependents[n.0 as usize].push_unique(id);
                    }
                }
            }
        }
        let group = Arc::new(GroupInfo {
            members: entries.iter().map(|(m, e)| (*m, e.generation)).collect(),
            cycle,
        });
        for (m, _) in &entries {
            let (node, flags) = self.nodes.get_with_flags(*m);
            let old = inner.member_recs.insert(*m, group.clone());
            if !C::SINGLE_THREADED {
                let in_group = C::u8_load_relaxed(flags) & IN_GROUP != 0;
                hold::keep_rec(state::cell_key::<C>(node.state_cell()), old, in_group);
            }
            C::u8_store_relaxed(flags, C::u8_load_relaxed(flags) | IN_GROUP);
        }
        drop(inner);

        // A member's handled outcome holds for the revision, and every
        // member's value is built on it, so all of them keep a memo. Not
        // with CU left: a CU member run again alone may skip the reads
        // that put it in CU and read the memos as values. Nor if it skipped
        // the cut to the root's SCC (`uncut`): a panic in this entry may
        // have cut the cycle the strategy values were given for, and a read
        // later in the revision holds the nodes it failed failing.
        let poison = entries.iter().fold(0, |p, (_, e)| p | e.poison());
        let memo = cu.is_empty() && !uncut && self.memo_landing(poison, verify_rev);

        // Step 3: write the values and stamp.
        for i in 0..entries.len() {
            let m = entries[i].0;
            let node = self.nodes.get(m);
            let arena = self.arenas.erased(node.type_tag());
            if cu.binary_search(&m).is_ok() {
                let old = arena.clear_value(node.arena_slot());
                let wrote = old.is_some();
                let old = if C::SINGLE_THREADED {
                    old
                } else {
                    hold::keep_value(state::cell_key::<C>(node.state_cell()), old)
                };
                if let Some(old) = old {
                    bag.push_erased(old);
                }
                if wrote {
                    entries[i].1.wrote = true;
                }
                node.max_changed_at(stamp_rev);
                entries[i].1.changed = true;
            } else {
                let e = &mut entries[i].1;
                let value = if e.seeded { e.s.take() } else { e.take_buf() }
                    .expect("incr-core: a landing member has a value");
                let first = e.first;
                let wrote = Cell::new(0_u8);
                let written = catch_unwind(AssertUnwindSafe(|| {
                    arena.write_erased(node.arena_slot(), value, first, &wrote)
                }));
                e.wrote |= wrote.get() != 0;
                match written {
                    Ok((changed, displaced)) => {
                        if changed || first {
                            node.max_changed_at(stamp_rev);
                            e.changed = true;
                        }
                        self.displaced(node, displaced, changed || first, bag);
                    }
                    Err(payload) => {
                        e.failed = true;
                        self.abort_landing(&mut entries, stamp_rev, verify_rev, cidx, payload, bag);
                    }
                }
            }
            if entries[i].1.poison() != 0 {
                node.set_verified_at(0);
            } else {
                node.max_verified_at(verify_rev);
            }
        }
        // Only once every value is written: an abort in step 3 lands the
        // members as if the episode kept no memo.
        if memo {
            for (m, _) in &entries {
                self.nodes.get(*m).set_memo(verify_rev);
            }
        }

        // Steps 4 and 5: land, all Dirty if any member was flagged or a
        // new dep may hide a change.
        let dirty = superseded
            || poison != 0
            || entries
                .iter()
                .any(|(m, _)| self.nodes.get(*m).state() == NodeState::ComputingDirty);
        for (m, e) in entries {
            let node = self.nodes.get(m);
            self.record_trace(
                NodeId(m),
                crate::trace::TraceAction::Recomputed {
                    value_changed: e.changed,
                },
            );
            let (target, flagged) = match (cu.binary_search(&m).is_ok(), dirty) {
                (true, false) => (NodeState::Cyclic, NodeState::New),
                (true, true) => (NodeState::New, NodeState::New),
                (false, false) => (NodeState::Clean, NodeState::Dirty),
                (false, true) => (NodeState::Dirty, NodeState::Dirty),
            };
            // A member of CU lands with no value, so its runs count as
            // discarded; the others keep the children of their last run,
            // which is the final pass.
            let children = match target {
                NodeState::Clean => ChildLanding::Sweep,
                NodeState::Dirty => ChildLanding::Keep,
                _ => ChildLanding::Discard,
            };
            let flags = self.nodes.get_with_flags(m).1;
            self.children_landing(NodeId(m), flags, children, e.opened(), bag);
            let landed = self.land(NodeId(m), node, target, flagged);
            self.end_reservation_landed(NodeId(m), node, landed);
            bag.push_erased(Box::new(e.into_values()));
        }
        PopEnd::Landed
    }

    /// Step 3 ended in a panic (a `PartialEq` of a member's value): the
    /// member whose compare panicked lands through fix 5's panic path if
    /// it wrote nothing, and the others through
    /// [`Self::land_panic_released`] for a user panic,
    /// or the unwind's landing ([`Self::unwound_land`]) for an engine
    /// one, where a member whose slot was written is stamped and lands
    /// New.
    fn abort_landing(
        &self,
        entries: &mut [(u32, Seen<Held>)],
        stamp_rev: u64,
        verify_rev: u64,
        cidx: u32,
        payload: Payload,
        bag: &mut Deferred<C>,
    ) -> ! {
        // The landing counted itself, but its members now land unrun.
        self.count_release();
        let kind = cycle_state::take_unwind_kind();
        // Step 3 sets no memo before its last compare, and a claimed
        // member holds none from an earlier landing.
        debug_assert!(
            kind.is_none()
                || entries
                    .iter()
                    .all(|(m, _)| !self.nodes.get(*m).memo_at(|| verify_rev)),
            "incr-core: an engine unwind aborted an episode landing over a member's memo (T8-memo-on-aborted-episode)"
        );
        let contain = self.episode_contains(cidx, kind, &*payload);
        let members: Vec<u32> = entries.iter().map(|(m, _)| *m).collect();
        for (m, e) in entries.iter_mut() {
            let id = NodeId(*m);
            let attempt = Attempt {
                first: e.first,
                cidx,
                stamp_rev,
                verify_rev,
            };
            if e.failed && !e.wrote {
                let (node, flags) = self.nodes.get_with_flags(*m);
                let deps = e.take_deps();
                self.land_failed(
                    id,
                    node,
                    flags,
                    attempt,
                    false,
                    &deps,
                    &members,
                    Some(&*payload),
                    false,
                    bag,
                );
            } else if kind.is_none() {
                self.land_panic_released(id, attempt, e.wrote, e.deps(), &members, bag);
            } else {
                self.unwound_land(id, e.first, e.wrote, stamp_rev, kind, bag);
            }
            if kind.is_none() {
                self.end_reservation(id, self.nodes.get(*m));
            }
            bag.push_erased(Box::new((e.take_buf(), e.s.take())));
        }
        if !contain {
            self.note_unwound_read(&*payload, kind);
        }
        if contain {
            self.raise_contained(cidx, payload);
        }
        self.leave_frame(kind, payload, bag)
    }

    /// Claims group member `m`, Dirty to Computing, for the group verify
    /// of `x`, which holds `claimed` too. Another thread's claim is waited
    /// out, never given up: a verify that gave up would compute `x` alone,
    /// and a winner left holding an unseeded `x` under its member raises a
    /// cycle that no single-thread order raises. A wound in the wait lands
    /// `x` and `claimed` Dirty and in their group, as if never claimed.
    fn claim_member(&self, x: NodeId, m: u32, claimed: &[u32], bag: &mut Deferred<C>) -> bool {
        let node = self.nodes.get(m);
        let mut spins = 0;
        loop {
            match state::try_transition::<C>(
                node.state_cell(),
                NodeState::Dirty,
                NodeState::Computing,
            ) {
                Ok(()) if self.reserved_elsewhere(NodeId(m), node.generation()) => {
                    let generation = node.generation();
                    self.unclaim_unrun(node, true);
                    let waited = catch_unwind(AssertUnwindSafe(|| {
                        self.wait_out_reservation(NodeId(m), generation, Some(x), &mut spins)
                    }));
                    if let Err(payload) = waited {
                        self.unclaim_group(x, claimed, bag, payload);
                    }
                }
                Ok(()) => return true,
                Err(s)
                    if !C::SINGLE_THREADED && is_computing(s) && !self.holds_claim(NodeId(m)) =>
                {
                    let waited = catch_unwind(AssertUnwindSafe(|| {
                        self.spin_holding(NodeId(m), Some(x), &mut spins, || {
                            !is_computing(state::load::<C>(node.state_cell()))
                        })
                    }));
                    if let Err(payload) = waited {
                        self.unclaim_group(x, claimed, bag, payload);
                    }
                }
                Err(_) => return false,
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn unclaim_group(
        &self,
        x: NodeId,
        claimed: &[u32],
        bag: &mut Deferred<C>,
        payload: Payload,
    ) -> ! {
        let kind = cycle_state::take_unwind_kind();
        self.release_group_claims(claimed.iter().chain(std::iter::once(&x.0)));
        cycle_state::set_unwind_kind(kind);
        bag.drain_carried(payload)
    }

    /// Lands group verify's claims `claimed`, taken from Dirty, back to
    /// Dirty and in their group, as if never claimed: no member ran, so
    /// another thread's read of one must make the decision a read before
    /// the claims would have made.
    fn release_group_claims<'a>(&self, claimed: impl IntoIterator<Item = &'a u32>) {
        let rt = self.runtime_id;
        for &c in claimed {
            cycle_state::with_held(|h| h.remove(&(rt, c)));
            self.land_unrun(NodeId(c), self.nodes.get(c), NodeState::Dirty);
        }
    }

    /// Group verification of the Dirty group member `x`, whose claim
    /// `compute_one` holds at generation `g`, `walked` if as a walk's entry.
    pub(super) fn group_verify(
        &self,
        x: NodeId,
        g: u32,
        visit: Visit,
        walked: bool,
        bag: &mut Deferred<C>,
    ) -> GroupEnd {
        let rt = self.runtime_id;
        let group = self.inner.read().member_recs.get(&x.0).cloned();
        // The flags byte was loaded under the claim, and a record and its
        // bit change together under the claim, so a missing record is a
        // stale bit.
        #[cfg(test)]
        {
            assert!(
                group.is_some(),
                "incr-core: IN_GROUP set on node {} with no record",
                x.0
            );
            self.group_verifies.lock().unwrap().push(x.0);
        }
        let Some(group) = group else {
            return GroupEnd::Ordinary;
        };
        if !group.members.iter().any(|&(m, gen)| m == x.0 && gen == g) {
            return GroupEnd::Ordinary;
        }
        // A discard keeps the root in the group it dissolved, for other
        // threads, whose orders verify the group; this entry's retry of the
        // root computes it alone, as the unbounded run's frame does.
        if cycle_state::is_dissolved_group(rt, x.0, &group) {
            return GroupEnd::Ordinary;
        }
        // The members' runs sit where `x`'s frame would.
        let cidx = self.dep_stack.next_cidx();
        // 1. Claim the other members, Dirty to Computing, in slot order.
        let mut claimed: Vec<u32> = Vec::new();
        for &(m, generation) in &group.members {
            if m == x.0 {
                continue;
            }
            let node = self.nodes.get(m);
            // A member with a virtual frame stands for a frame computing on
            // this thread's stack, whose claim the unbounded run fails.
            let virtual_member = value_code::thread_word() & value_code::REPLAY_BIT != 0
                && self.virtual_position(m, generation).is_some();
            let mut ok = !virtual_member && self.claim_member(x, m, &claimed, bag);
            if ok && node.generation() != generation {
                self.land_unrun(NodeId(m), node, NodeState::Dirty);
                ok = false;
            }
            if !ok {
                self.release_group_claims(&claimed);
                return GroupEnd::Ordinary;
            }
            cycle_state::with_held(|h| {
                h.insert((rt, m), Held::new(generation, cidx, false, false))
            });
            claimed.push(m);
            self.park(ParkPoint::GroupMemberClaimed(NodeId(m)));
        }

        // 2. Did anything any member read change since it was verified?
        let verify_rev = C::walked(&self.walk, &self.revision);
        let mut changed = false;
        for &(m, _) in &group.members {
            let node = self.nodes.get(m);
            let verified = node.verified_at();
            if verified == 0 {
                #[cfg(test)]
                self.group_zero_fails
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                changed = true;
                break;
            }
            node.for_each_dep(|d| {
                if changed {
                    return;
                }
                let ok = if group
                    .members
                    .binary_search_by_key(&d.0, |&(s, _)| s)
                    .is_ok()
                {
                    self.nodes.get(d.0).changed_at() <= verified
                } else {
                    self.observed_clean(d, verified)
                };
                changed = !ok;
            });
            if changed {
                break;
            }
        }

        if !changed {
            // 3. Re-land the whole group Clean.
            for &(m, _) in &group.members {
                cycle_state::with_held(|h| h.remove(&(rt, m)));
                let node = self.nodes.get(m);
                debug_assert!(
                    self.arenas
                        .erased(node.type_tag())
                        .has_value(node.arena_slot()),
                    "incr-core: group verify re-lands member {m} over an empty slot"
                );
                node.max_verified_at(verify_rev);
                let landed = self.land(NodeId(m), node, NodeState::Clean, NodeState::Dirty);
                self.end_reservation_landed(NodeId(m), node, landed);
                if landed == NodeState::Clean {
                    self.record_trace(NodeId(m), crate::trace::TraceAction::VerifiedClean);
                }
            }
            return GroupEnd::Done(Computed::Done);
        }

        if let Visit::VerifyOnly(owner) = visit {
            // A tainted node never runs: release the group, and leave `x`
            // Dirty for the closure that still reads it, whose read the
            // taint memo makes compute `x` alone.
            self.release_group_claims(claimed.iter().chain(std::iter::once(&x.0)));
            self.memoize(x, owner);
            return GroupEnd::Done(Computed::Done);
        }

        // 4. Recompute the group through an episode rooted at `x`.
        let seeded = C::u8_load_relaxed(self.nodes.get_with_flags(x.0).1) & HAS_STRATEGY != 0;
        cycle_state::with_held(|h| {
            h.insert((rt, x.0), Held::new(g, cidx, seeded, false));
            for &c in &claimed {
                if let Some(e) = h.get_mut(&(rt, c)) {
                    e.seeded =
                        C::u8_load_relaxed(self.nodes.get_with_flags(c).1) & HAS_STRATEGY != 0;
                }
            }
        });
        match self.episode(x, cidx, walked, bag) {
            PopEnd::Landed => GroupEnd::Done(Computed::Done),
            PopEnd::Deferred => GroupEnd::Done(Computed::Deferred),
            PopEnd::Dissolved => {
                self.end_dissolved(x, bag);
                GroupEnd::Dissolved
            }
            PopEnd::Retry => GroupEnd::Retry,
            PopEnd::Suspended | PopEnd::Dropped => GroupEnd::Done(Computed::Discarded),
            PopEnd::Contained => GroupEnd::Done(Computed::Unsettled),
        }
    }
}

/// An uncaught cycle at depth 0: a panic naming the participants, with
/// the cycle kept for bindings that report it as a typed error.
#[cold]
#[inline(never)]
fn cycle_panic(cycle: Cycle) -> ! {
    let message = format!("incr-core: dependency cycle: {cycle}");
    cycle_state::set_last_cycle(cycle);
    panic!("{message}");
}

/// Whether an unwind from position `cidx` releases held claim `e`: an
/// episode suspended over it, or one it waits for, keeps it.
fn released_at(rt: RuntimeId, e: &Held, cidx: u32) -> bool {
    !replay::is_suspended(e.ep) && !replay::waits_for_suspended(rt, e.anchor(), cidx)
}

/// A run that unwinds with a cycle's kind and its own payload, having read
/// past the read the cycle came in through: its closure caught the cycle,
/// so the kind is stale and the panic is the closure's, as `read_cold`
/// makes it at a non-Clean read. A Clean read leaves the kind alone, so the
/// pop clears it here.
#[cold]
#[inline(never)]
pub(super) fn clear_caught_cycle(payload: &(dyn Any + Send), deps: &[(NodeId, u32)], reading: u64) {
    if cycle_state::unwind_kind() == Some(UnwindKind::Cycle)
        && !payload.is::<CycleUnwind>()
        && deps
            .last()
            .is_some_and(|&(d, g)| (u64::from(d.0) | (u64::from(g) << 32)) != reading)
    {
        cycle_state::set_unwind_kind(None);
    }
}
