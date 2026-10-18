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
use crate::cycle_state::Held;
use crate::{set_stack, waits};

/// What a compute attempt carries into its landings.
#[derive(Copy, Clone)]
pub(super) struct Attempt {
    /// The claim was taken from New.
    pub(super) first: bool,
    pub(super) cidx: u32,
    pub(super) stamp_rev: u64,
    pub(super) verify_rev: u64,
}

/// How a `LOW` pop, or an episode, left the root.
pub(super) enum PopEnd {
    /// The node is held, for a lower root to land.
    Deferred,
    /// The episode landed every member.
    Landed,
    /// Validation found the root alone: it computes as an ordinary node,
    /// under the claim it holds.
    Dissolved,
    /// A recorded dep went stale: every member landed New, and the root
    /// claims again.
    Retry,
}

/// How group verification left the node `compute_one` claimed.
pub(super) enum GroupEnd {
    Done(Computed),
    /// A member could not be claimed, or the episode dissolved: compute
    /// the node as an ordinary one.
    Ordinary,
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

impl<C: Cells> Runtime<C> {
    /// Raises the cycle a `get` read found: an engine unwind inside an
    /// engine entry, which the outermost entry turns into a panic, or that
    /// panic itself at depth 0.
    #[cold]
    #[inline(never)]
    pub(super) fn raise_cycle(&self, cycle: Cycle) -> ! {
        if cycle_state::depth() == 0 {
            cycle_panic(cycle);
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
        let entry = EntryDepth::enter();
        if !entry.is_outermost() && kind == ReadKind::Get {
            return Ok(f());
        }
        self.entered_catching(entry.is_outermost(), kind, &mut f)
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

    fn entered_catching<R>(
        &self,
        outermost: bool,
        kind: ReadKind,
        f: &mut impl FnMut() -> R,
    ) -> Result<R, Cycle> {
        loop {
            if outermost {
                cycle_state::reset_outermost();
            }
            let payload = match catch_unwind(AssertUnwindSafe(&mut *f)) {
                Ok(r) => return Ok(r),
                Err(payload) => payload,
            };
            // A wound wins over any kind: whatever unwound last, this
            // thread gave up its claims to an older one, and retries.
            if outermost && cycle_state::wounded() {
                cycle_state::take_unwind_kind();
                cycle_state::set_wounded(false);
                drop(payload);
                waits::await_winner(C::backoff);
                continue;
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
                        return Err(cycle);
                    }
                    cycle_panic(cycle);
                }
                other => {
                    cycle_state::set_unwind_kind(other);
                    resume_unwind(payload);
                }
            }
        }
    }

    /// An engine unwind that a closure caught and returned past: a cycle
    /// becomes an ordinary result, as under `try_get`. A wound stays in
    /// flight and the pop is told to treat its closure as unwound, since
    /// the claims the wound must release are still held.
    #[cold]
    pub(super) fn swallowed(&self, kind: Option<UnwindKind>) -> bool {
        match kind {
            Some(UnwindKind::Cycle) => {
                cycle_state::set_unwind_kind(None);
                false
            }
            Some(UnwindKind::Wound) => true,
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
    /// registers with the claims this thread holds and waits until `done`,
    /// unless a registration that closes a waits-for cycle wounds it.
    #[cold]
    #[inline(never)]
    pub(super) fn wait_registered(&self, on: NodeId, mut done: impl FnMut() -> bool) {
        let mut holds: Vec<waits::Claim> = self.dep_stack.with(|s| {
            s.frames
                .iter()
                .filter(|f| f.rt != RuntimeId::SENTINEL)
                .map(|f| (f.rt, f.node.0))
                .collect()
        });
        holds.extend(cycle_state::held_keys());
        holds.extend(set_stack::claims());
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
        *spins += 1;
        if !C::SINGLE_THREADED && *spins >= waits::REGISTER_AFTER {
            self.wait_registered(on, done);
        } else {
            C::backoff(*spins);
        }
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
        let ctl = ComputeCtl::new(0);
        self.dep_stack
            .push_frame_at(RuntimeId::SENTINEL, NodeId::SENTINEL, &ctl, u32::MAX);
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
                .filter(|f| f.rt == rt && f.cidx >= p)
                .map(|f| f.node.0)
                .collect()
        });
        cycle_state::with_held(|h| {
            slots.extend(
                h.iter()
                    .filter(|((r, _), e)| *r == rt && e.anchor >= p)
                    .map(|((_, slot), _)| *slot),
            )
        });
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
                .any(|f| f.cidx >= p && f.cidx != u32::MAX && f.rt != rt)
        })
    }

    /// The read rules, for a read of `handle` that found it `Cyclic` or
    /// Computing. `None` when the node is not this thread's: the read
    /// waits like any other.
    #[cold]
    #[inline(never)]
    pub(super) fn cycle_read<T: Value>(
        &self,
        handle: Incr<T>,
        observed: NodeState,
        kind: ReadKind,
    ) -> Option<Result<T, Cycle>> {
        let rt = self.runtime_id;
        let slot = handle.slot();
        if observed == NodeState::Cyclic {
            return Some(Err(self
                .cycle_of(handle)
                .unwrap_or_else(|| Cycle::new(vec![handle.erase()]))));
        }
        let on_stack = self.dep_stack.with(|s| {
            s.frames
                .iter()
                .rev()
                .find(|f| f.node.0 == slot && f.rt == rt)
                .map(|f| f.cidx)
        });
        let held = cycle_state::with_held(|h| {
            h.get(&(rt, slot))
                .map(|e| (e.anchor, e.in_cu, e.buf.is_some(), e.episode.clone()))
        });
        let p = match (on_stack, &held) {
            (Some(cidx), _) => cidx,
            (None, Some((anchor, ..))) => *anchor,
            (None, None) => return None,
        };
        let active = self
            .dep_stack
            .active_frame(rt)
            .filter(|_| !self.foreign_frame_above(p));
        let Some((ctl, _)) = active else {
            return Some(Err(self.cycle_at(p)));
        };
        // SAFETY: the active frame is on the stack, so its control block
        // is live.
        let ctl = unsafe { &*ctl };

        // R0: an unseeded member that still reads another such member.
        if let Some((anchor, true, _, Some(cycle))) = &held {
            ctl.lower(*anchor);
            return Some(Err(cycle.clone()));
        }
        // R2: a strategy member reads as its current value.
        if C::u8_load_relaxed(self.nodes.get_with_flags(slot).1) & HAS_STRATEGY != 0 {
            let value = match &held {
                Some((.., Some(_))) => self.held_value::<T>(slot, true),
                _ => self.discovery_value::<T>(slot, p),
            };
            ctl.lower(p);
            return Some(Ok(value));
        }
        // R3: an unseeded node on the stack has no value to give.
        if on_stack.is_some() {
            ctl.lower(p);
            return Some(Err(self.cycle_at(p)));
        }
        let (anchor, _, buffered, _) = held.expect("an unseeded node off the stack is held");
        // R5: run the member for its value, then R4: read its buffer.
        if !buffered {
            match self.member_run(slot, anchor) {
                Ok(low) => ctl.lower(low),
                Err(payload) => {
                    ctl.lower(anchor);
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
        resume_unwind(payload)
    }

    /// A clone of a held node's current value (`s`) or buffer.
    fn held_value<T: Value>(&self, slot: u32, current: bool) -> T {
        cycle_state::with_held(|h| {
            let e = &h[&(self.runtime_id, slot)];
            let v = if current { &e.s } else { &e.buf };
            let v = v
                .as_ref()
                .expect("incr-core: a held member has a value to read");
            value_code::run(|| downcast::<T>(v).clone())
        })
    }

    /// A strategy member's value when no episode runs: `g` of the cycle
    /// the read closes, or the fixpoint's initial value.
    fn discovery_value<T: Value>(&self, slot: u32, p: u32) -> T {
        let strategy = self.strategy_of(slot);
        let cycle = self.cycle_at(p);
        let v = self.callback(|| strategy.v0(&cycle));
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
    /// the node marked failed.
    pub(super) fn member_run(&self, slot: u32, cidx: u32) -> Result<u32, Payload> {
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
        let ctl = ComputeCtl::new(cycle::BUFFERED);
        ctl.lowlink.set(cidx);
        self.dep_stack
            .push_frame_at(self.runtime_id, id, &ctl, cidx);
        // SAFETY: this thread holds the node's claim, so its closure is
        // not freed (see `compute_attempts`).
        let compute = unsafe { &*compute };
        let failures = cycle_state::failures();
        let mut result = catch_unwind(AssertUnwindSafe(|| compute(self, slot, &ctl)));
        let deps = self.dep_stack.pop_frame();
        let word = value_code::thread_word();
        if word & (value_code::MEMO_BIT | value_code::UNWIND_MASK) != 0
            && self.after_pop(word, result.is_ok())
        {
            result = Err(Box::new(WoundUnwind));
        }
        let replaced = cycle_state::with_held(|h| {
            let e = h
                .get_mut(&(self.runtime_id, slot))
                .expect("incr-core: a member run's node is held");
            e.record_run(deps);
            e.poisoned |= ctl.has(cycle::POISONED);
            // An engine unwind is no fault of the member's.
            if result.is_err()
                && cycle_state::failures() == failures
                && cycle_state::unwind_kind().is_none()
            {
                cycle_state::mark_failed(e);
            }
            std::mem::replace(&mut e.buf, ctl.buf.take())
        });
        drop(replaced);
        result.map(|_| ctl.lowlink.get())
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
        held.buf = ctl.buf.take();
        held.poisoned = ctl.has(cycle::POISONED);
        held.wrote = ctl.wrote.get() != 0;
        held.record_run(deps);
        let displaced = cycle_state::with_held(|h| {
            if lowlink < cidx {
                // The claims this frame's subtree deferred belong to the
                // lower root too, not to a later frame at their position.
                for ((r, _), e) in h.iter_mut() {
                    if *r == rt && e.anchor >= cidx {
                        e.anchor = lowlink;
                    }
                }
            }
            h.insert((rt, id.0), held)
        });
        debug_assert!(displaced.is_none(), "incr-core: node {} held twice", id.0);
        if lowlink < cidx {
            if let Some((parent, _)) = self.dep_stack.active_frame(rt) {
                // SAFETY: the active frame's control block is live.
                unsafe { &*parent }.lower(lowlink);
            }
            return PopEnd::Deferred;
        }
        self.episode(id, cidx, bag)
    }

    /// Every claim of this runtime held at or above `cidx`, re-anchored at
    /// `cidx`, in slot order.
    fn claim_members(&self, cidx: u32) -> Vec<u32> {
        let rt = self.runtime_id;
        let mut members: Vec<u32> = cycle_state::with_held(|h| {
            h.iter_mut()
                .filter(|((r, _), e)| *r == rt && e.anchor >= cidx)
                .map(|((_, slot), e)| {
                    e.anchor = cidx;
                    *slot
                })
                .collect()
        });
        members.sort_unstable();
        members
    }

    /// The member-to-member reads of each member's last run, as indices
    /// into `members`.
    fn member_graph(&self, members: &[u32]) -> Vec<Vec<usize>> {
        let rt = self.runtime_id;
        cycle_state::with_held(|h| {
            members
                .iter()
                .map(|m| {
                    let mut out: Vec<usize> = h[&(rt, *m)]
                        .last
                        .iter()
                        .filter_map(|(d, _)| members.binary_search(&d.0).ok())
                        .collect();
                    out.sort_unstable();
                    out.dedup();
                    out
                })
                .collect()
        })
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

    /// Ends an episode with an unwind: every member is released, a member
    /// whose own code panicked through fix 5's panic path and the others
    /// with the discard landing, and the unwind resumes.
    fn abort_episode(&self, cidx: u32, payload: Payload, bag: &mut Deferred<C>) -> ! {
        let kind = cycle_state::take_unwind_kind();
        self.release_held_from(cidx, kind, bag);
        cycle_state::set_unwind_kind(kind);
        bag.drain_carried(payload)
    }

    /// Drops a user value an episode no longer needs, attributing a
    /// panicking `Drop` to the member it belonged to.
    fn drop_member_value(
        &self,
        slot: u32,
        cidx: u32,
        value: Option<Erased>,
        bag: &mut Deferred<C>,
    ) {
        if let Some(value) = value {
            self.attributed(slot, cidx, bag, || drop(value));
        }
    }

    fn take_held<R>(&self, slot: u32, f: impl FnOnce(&mut Held) -> R) -> R {
        cycle_state::with_held(|h| {
            f(h.get_mut(&(self.runtime_id, slot))
                .expect("incr-core: an episode member is held"))
        })
    }

    /// Runs one member and folds its lowlink into `low`.
    fn run_member(&self, slot: u32, cidx: u32, low: &mut u32, bag: &mut Deferred<C>) {
        match self.member_run(slot, cidx) {
            Ok(l) => *low = (*low).min(l),
            Err(payload) => self.abort_episode(cidx, payload, bag),
        }
    }

    /// The episode rooted at `root`, at conceptual-stack position `cidx`:
    /// iterate the members to a landing, defer the group to a lower root,
    /// or dissolve it.
    pub(super) fn episode(&self, root: NodeId, cidx: u32, bag: &mut Deferred<C>) -> PopEnd {
        let rt = self.runtime_id;
        let mut validated = cycle_state::has_dissolved(rt, root.0);
        let mut members = self.claim_members(cidx);
        let mut cu: Vec<u32> = Vec::new();

        'restart: loop {
            let cycle = Cycle::new(members.iter().map(|&m| self.erased_at_live(m)).collect());
            let strategies: Vec<(u32, Arc<dyn ErasedStrategy>)> = {
                let inner = self.inner.read();
                members
                    .iter()
                    .filter_map(|&m| inner.cycle_strategies.get(&m).map(|s| (m, s.clone())))
                    .collect()
            };
            let limit = strategies
                .iter()
                .filter_map(|(_, s)| s.max_iterations())
                .min()
                .unwrap_or(1);
            let unseeded: Vec<u32> = members
                .iter()
                .copied()
                .filter(|m| strategies.binary_search_by_key(m, |(s, _)| *s).is_err())
                .collect();
            cycle_state::with_held(|h| {
                for m in &members {
                    let e = h
                        .get_mut(&(rt, *m))
                        .expect("incr-core: an episode member is held");
                    e.anchor = cidx;
                    e.episode = Some(cycle.clone());
                    e.in_cu = cu.binary_search(m).is_ok();
                    e.installed.clear();
                }
            });
            // Every claim precedes this capture, because membership growth
            // restarts the episode.
            let stamp_rev = self.current_revision();
            let verify_rev = C::walked(&self.walk, &self.revision);
            for (m, s) in &strategies {
                let v0 = self.attributed(*m, cidx, bag, || self.callback(|| s.v0(&cycle)));
                let old = self.take_held(*m, |e| e.s.replace(v0));
                self.drop_member_value(*m, cidx, old, bag);
            }

            // s_{k-1} of each Fixpoint member, for the convergence test and
            // the landing pass.
            let mut prev: Vec<Option<Erased>> = (0..strategies.len()).map(|_| None).collect();
            let mut k: u32 = 1;
            loop {
                // 1. Clear the unseeded buffers.
                for &m in &unseeded {
                    let old = self.take_held(m, |e| e.buf.take());
                    self.drop_member_value(m, cidx, old, bag);
                }
                // 2. Run each seeded member once, in slot order.
                let mut low = cidx;
                for (m, _) in &strategies {
                    self.run_member(*m, cidx, &mut low, bag);
                }
                // 3. Run the unseeded members no read has run.
                for &m in &unseeded {
                    if self.take_held(m, |e| e.buf.is_none()) {
                        self.run_member(m, cidx, &mut low, bag);
                    }
                }
                // 6. A run reached below the root: the group belongs to a
                // lower root's cycle.
                if low < cidx {
                    self.merge_down(cidx, low, bag);
                    return PopEnd::Deferred;
                }
                // 5. A non-member that read the group joined it.
                if self.join_new(cidx, &mut members) {
                    continue 'restart;
                }
                let graph = self.member_graph(&members);
                if !validated {
                    validated = true;
                    let comp = components(&graph);
                    let r = members
                        .binary_search(&root.0)
                        .expect("the root is a member");
                    let scc: Vec<u32> = members
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| comp[*i] == comp[r])
                        .map(|(_, &m)| m)
                        .collect();
                    if scc.len() == 1 && !graph[r].contains(&r) {
                        cycle_state::dissolved_before(rt, root.0);
                        self.release_members(&members, Some(root.0), bag);
                        let entry = cycle_state::with_held(|h| h.remove(&(rt, root.0)));
                        if let Some(e) = entry {
                            bag.push_erased(Box::new((e.buf, e.s)));
                        }
                        return PopEnd::Dissolved;
                    }
                    if scc.len() < members.len() {
                        let gone: Vec<u32> = members
                            .iter()
                            .copied()
                            .filter(|m| scc.binary_search(m).is_err())
                            .collect();
                        self.release_members(&gone, None, bag);
                        members = scc;
                        continue 'restart;
                    }
                }
                // CU: the unseeded members that still read each other.
                let un_idx: Vec<usize> = unseeded
                    .iter()
                    .map(|m| {
                        members
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
                for (i, &m) in unseeded.iter().enumerate() {
                    if cyclic[i] {
                        if let Err(at) = cu.binary_search(&m) {
                            cu.insert(at, m);
                            grew = true;
                        }
                    }
                }
                if grew {
                    continue 'restart;
                }

                // 4. The strategy members' next values.
                let mut converged = vec![true; strategies.len()];
                for (i, (m, s)) in strategies.iter().enumerate() {
                    let out = self.take_held(*m, |e| e.buf.take());
                    if s.max_iterations().is_none() {
                        self.drop_member_value(*m, cidx, out, bag);
                        continue;
                    }
                    let out = out.expect("incr-core: a member run buffers its value");
                    let s_prev = self
                        .take_held(*m, |e| e.s.take())
                        .expect("incr-core: a strategy member has a value");
                    let next = self
                        .attributed(*m, cidx, bag, || self.callback(|| s.step(&s_prev, out, k)));
                    converged[i] = self.attributed(*m, cidx, bag, || {
                        self.callback(|| s.converged(&s_prev, &next))
                    });
                    self.take_held(*m, |e| e.s = Some(next));
                    let older = prev[i].replace(s_prev);
                    self.drop_member_value(*m, cidx, older, bag);
                }
                if converged.iter().all(|&c| c) || k >= limit {
                    for (i, (m, s)) in strategies.iter().enumerate() {
                        if !converged[i] {
                            let last = self
                                .take_held(*m, |e| e.s.take())
                                .expect("incr-core: a strategy member has a value");
                            let landed = self.attributed(*m, cidx, bag, || {
                                self.callback(|| s.on_limit(&cycle, last))
                            });
                            self.take_held(*m, |e| e.s = Some(landed));
                        }
                    }
                    break;
                }
                k += 1;
            }

            // The landing pass: unseeded members outside CU computed
            // against s_{K-1} run once more against s*, unless no
            // Fixpoint member moved.
            let moved = strategies.iter().enumerate().any(|(i, (m, s))| {
                prev[i].as_ref().is_some_and(|p| {
                    let cur = self.take_held(*m, |e| e.s.take());
                    let same = cur.as_ref().is_some_and(|c| {
                        self.attributed(*m, cidx, bag, || value_code::run(|| s.same(p, c)))
                    });
                    self.take_held(*m, |e| e.s = cur);
                    !same
                })
            });
            for (i, p) in prev.into_iter().enumerate() {
                self.drop_member_value(strategies[i].0, cidx, p, bag);
            }
            if moved {
                let outside: Vec<u32> = unseeded
                    .iter()
                    .copied()
                    .filter(|m| cu.binary_search(m).is_err())
                    .collect();
                for &m in &outside {
                    let old = self.take_held(m, |e| e.buf.take());
                    self.drop_member_value(m, cidx, old, bag);
                }
                let mut low = cidx;
                for &m in &outside {
                    if self.take_held(m, |e| e.buf.is_none()) {
                        self.run_member(m, cidx, &mut low, bag);
                    }
                }
                if low < cidx {
                    self.merge_down(cidx, low, bag);
                    return PopEnd::Deferred;
                }
                if self.join_new(cidx, &mut members) {
                    continue 'restart;
                }
            }
            return self.land_episode(&members, &cu, cycle, stamp_rev, verify_rev, cidx, bag);
        }
    }

    /// Adds every claim held at or above `cidx` that is not yet a member.
    fn join_new(&self, cidx: u32, members: &mut Vec<u32>) -> bool {
        let rt = self.runtime_id;
        let joined: Vec<u32> = cycle_state::with_held(|h| {
            h.iter_mut()
                .filter(|((r, slot), e)| {
                    *r == rt && e.anchor >= cidx && members.binary_search(slot).is_err()
                })
                .map(|((_, slot), e)| {
                    e.anchor = cidx;
                    *slot
                })
                .collect()
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
            h.iter_mut()
                .filter(|((r, _), e)| *r == rt && e.anchor >= cidx)
                .filter_map(|(_, e)| {
                    e.anchor = low;
                    e.episode = None;
                    e.in_cu = false;
                    e.s.take()
                })
                .collect()
        });
        for v in values {
            bag.push_erased(v);
        }
        if let Some((parent, _)) = self.dep_stack.active_frame(rt) {
            // SAFETY: the active frame's control block is live.
            unsafe { &*parent }.lower(low);
        }
    }

    /// Releases `slots` from the held map with the discard landing,
    /// except `keep`.
    fn release_members(&self, slots: &[u32], keep: Option<u32>, bag: &mut Deferred<C>) {
        let rt = self.runtime_id;
        let stamp_rev = self.current_revision();
        for &m in slots {
            if Some(m) == keep {
                continue;
            }
            let entry = cycle_state::with_held(|h| h.remove(&(rt, m)));
            if let Some(e) = entry {
                self.discard_land(NodeId(m), e.first, e.wrote, stamp_rev);
                bag.push_erased(Box::new((e.buf, e.s)));
            }
        }
    }

    /// Lands every claim of this runtime held at or above `cidx`: a claim
    /// whose own code panicked in user code takes fix 5's panic path, and
    /// every other claim, and every claim under an engine unwind, the
    /// discard landing.
    pub(super) fn release_held_from(
        &self,
        cidx: u32,
        kind: Option<UnwindKind>,
        bag: &mut Deferred<C>,
    ) {
        let rt = self.runtime_id;
        let mut entries: Vec<(u32, Held)> = cycle_state::with_held(|h| {
            if h.is_empty() {
                return Vec::new();
            }
            let keys: Vec<(RuntimeId, u32)> = h
                .iter()
                .filter(|((r, _), e)| *r == rt && e.anchor >= cidx)
                .map(|(k, _)| *k)
                .collect();
            keys.into_iter()
                .map(|k| (k.1, h.remove(&k).expect("the key was just listed")))
                .collect()
        });
        if entries.is_empty() {
            return;
        }
        entries.sort_unstable_by_key(|(slot, _)| *slot);
        let released: Vec<u32> = entries.iter().map(|(slot, _)| *slot).collect();
        let stamp_rev = self.current_revision();
        let verify_rev = C::walked(&self.walk, &self.revision);
        for (slot, e) in entries {
            let id = NodeId(slot);
            if e.failed && kind.is_none() {
                let (node, flags) = self.nodes.get_with_flags(slot);
                let attempt = Attempt {
                    first: e.first,
                    cidx,
                    stamp_rev,
                    verify_rev,
                };
                self.land_failed(id, node, flags, attempt, e.wrote, &e.deps, &released, bag);
            } else {
                self.discard_land(id, e.first, e.wrote, stamp_rev);
            }
            bag.push_erased(Box::new((e.buf, e.s)));
        }
    }

    /// The discard landing of a claim this thread holds: the node lands as
    /// if the attempt never ran. A claim taken from New lands New, and
    /// stores `verified_at = 0` if it wrote nothing (rule (a)); one that
    /// wrote its slot is stamped and lands New, so that its retry writes
    /// and stamps unconditionally; any other lands Dirty. Recorded deps
    /// are dropped and installed ones kept. It never lands Failed.
    pub(super) fn discard_land(&self, id: NodeId, first: bool, wrote: bool, stamp_rev: u64) {
        let (node, flags) = self.nodes.get_with_flags(id.0);
        let target = if wrote {
            node.max_changed_at(stamp_rev);
            NodeState::New
        } else if first {
            node.set_verified_at(0);
            NodeState::New
        } else {
            NodeState::Dirty
        };
        self.land_single(id, node, flags, target, target);
    }

    /// Lands an episode whose claims are complete (the plan's Landing
    /// steps 2 to 5; step 1 is the iteration that produced the values).
    #[allow(clippy::too_many_arguments)]
    fn land_episode(
        &self,
        members: &[u32],
        cu: &[u32],
        cycle: Cycle,
        stamp_rev: u64,
        verify_rev: u64,
        cidx: u32,
        bag: &mut Deferred<C>,
    ) -> PopEnd {
        let rt = self.runtime_id;
        let mut entries: Vec<(u32, Held)> = cycle_state::with_held(|h| {
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

        // Step 2: validate, install and record the group under one hold.
        let mut inner = self.inner.write();
        let stale = entries.iter().any(|(_, e)| {
            e.deps.iter().any(|&(d, generation)| {
                let dep = self.nodes.get(d.0);
                dep.generation() != generation || dep.state() == NodeState::Dead
            })
        });
        if stale {
            drop(inner);
            for (m, e) in entries {
                let (node, flags) = self.nodes.get_with_flags(m);
                if e.first {
                    node.set_verified_at(0);
                }
                self.land_single(NodeId(m), node, flags, NodeState::New, NodeState::New);
                bag.push_erased(Box::new((e.buf, e.s)));
            }
            return PopEnd::Retry;
        }
        let mut superseded = false;
        for (m, e) in &entries {
            let id = NodeId(*m);
            let node = self.nodes.get(*m);
            let mut old: Vec<NodeId> = Vec::new();
            node.for_each_dep(|d| old.push(d));
            let new: &[NodeId] = &e.installed;
            superseded |= new.iter().any(|d| {
                members.binary_search(&d.0).is_err()
                    && !old.contains(d)
                    && !self.observed_clean(*d, verify_rev)
            });
            if old != new {
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
            inner.member_recs.insert(*m, group.clone());
            let flags = self.nodes.get_with_flags(*m).1;
            C::u8_store_relaxed(flags, C::u8_load_relaxed(flags) | IN_GROUP);
        }
        drop(inner);

        // Step 3: write the values and stamp.
        for i in 0..entries.len() {
            let m = entries[i].0;
            let node = self.nodes.get(m);
            let arena = self.arenas.erased(node.type_tag());
            if cu.binary_search(&m).is_ok() {
                if let Some(old) = arena.clear_value(node.arena_slot()) {
                    bag.push_erased(old);
                    entries[i].1.wrote = true;
                }
                node.max_changed_at(stamp_rev);
                entries[i].1.changed = true;
            } else {
                let e = &mut entries[i].1;
                let value = if e.seeded { e.s.take() } else { e.buf.take() }
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
                        if let Some(d) = displaced {
                            bag.push_erased(d);
                        }
                    }
                    Err(payload) => {
                        e.failed = true;
                        self.abort_landing(&mut entries, stamp_rev, verify_rev, cidx, payload, bag);
                    }
                }
            }
            if entries[i].1.poisoned {
                node.set_verified_at(0);
            } else {
                node.max_verified_at(verify_rev);
            }
        }

        // Steps 4 and 5: land, all Dirty if any member was flagged or a
        // new dep may hide a change.
        let dirty = superseded
            || entries.iter().any(|(m, e)| {
                e.poisoned || self.nodes.get(*m).state() == NodeState::ComputingDirty
            });
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
            self.land(node, target, flagged);
            bag.push_erased(Box::new((e.buf, e.s)));
        }
        PopEnd::Landed
    }

    /// Step 3 ended in a panic (a `PartialEq` of a member's value): every
    /// member whose slot was written is stamped and lands New, the member
    /// whose compare panicked lands through fix 5's panic path, and the
    /// others take the discard landing. Each leaves its group.
    fn abort_landing(
        &self,
        entries: &mut [(u32, Held)],
        stamp_rev: u64,
        verify_rev: u64,
        cidx: u32,
        payload: Payload,
        bag: &mut Deferred<C>,
    ) -> ! {
        let kind = cycle_state::take_unwind_kind();
        let members: Vec<u32> = entries.iter().map(|(m, _)| *m).collect();
        for (m, e) in entries.iter_mut() {
            let id = NodeId(*m);
            if e.failed && !e.wrote {
                let (node, flags) = self.nodes.get_with_flags(*m);
                let attempt = Attempt {
                    first: e.first,
                    cidx,
                    stamp_rev,
                    verify_rev,
                };
                let deps = std::mem::take(&mut e.deps);
                self.land_failed(id, node, flags, attempt, false, &deps, &members, bag);
            } else {
                self.discard_land(id, e.first, e.wrote, stamp_rev);
            }
            bag.push_erased(Box::new((e.buf.take(), e.s.take())));
        }
        cycle_state::set_unwind_kind(kind);
        bag.drain_carried(payload)
    }

    /// Group verification of the Dirty group member `x`, whose claim
    /// `compute_one` holds at generation `g`.
    pub(super) fn group_verify(
        &self,
        x: NodeId,
        g: u32,
        visit: Visit,
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
        // The members' runs sit where `x`'s frame would.
        let cidx = self.dep_stack.depth() as u32;
        // 1. Claim the other members, Dirty to Computing, in slot order.
        let mut claimed: Vec<u32> = Vec::new();
        for &(m, generation) in &group.members {
            if m == x.0 {
                continue;
            }
            let node = self.nodes.get(m);
            let mut ok = state::try_transition::<C>(
                node.state_cell(),
                NodeState::Dirty,
                NodeState::Computing,
            )
            .is_ok();
            if ok && node.generation() != generation {
                self.land(node, NodeState::Dirty, NodeState::Dirty);
                ok = false;
            }
            if !ok {
                for c in claimed {
                    cycle_state::with_held(|h| h.remove(&(rt, c)));
                    let (node, flags) = self.nodes.get_with_flags(c);
                    self.land_single(NodeId(c), node, flags, NodeState::Dirty, NodeState::Dirty);
                }
                return GroupEnd::Ordinary;
            }
            cycle_state::with_held(|h| {
                h.insert((rt, m), Held::new(generation, cidx, false, false))
            });
            claimed.push(m);
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
                if self.land(node, NodeState::Clean, NodeState::Dirty) == NodeState::Clean {
                    self.record_trace(NodeId(m), crate::trace::TraceAction::VerifiedClean);
                }
            }
            return GroupEnd::Done(Computed::Done);
        }

        if let Visit::VerifyOnly(owner) = visit {
            // A tainted node never runs: release the group, and leave `x`
            // Dirty for the closure that still reads it.
            for c in claimed {
                cycle_state::with_held(|h| h.remove(&(rt, c)));
                let (node, flags) = self.nodes.get_with_flags(c);
                self.land_single(NodeId(c), node, flags, NodeState::Dirty, NodeState::Dirty);
            }
            let (node, flags) = self.nodes.get_with_flags(x.0);
            self.land_single(x, node, flags, NodeState::Dirty, NodeState::Dirty);
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
        match self.episode(x, cidx, bag) {
            PopEnd::Landed => GroupEnd::Done(Computed::Done),
            PopEnd::Deferred => GroupEnd::Done(Computed::Deferred),
            PopEnd::Dissolved => GroupEnd::Ordinary,
            PopEnd::Retry => GroupEnd::Retry,
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
