//! Deep chains by unwind and replay: the cold read paths a frame budget
//! sends reads down, the doom that releases the node a read wanted and
//! marks the frames below it, the discard of a doomed frame, and the
//! driver that runs a context's entries until its target lands.

use std::any::Any;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use super::*;
use crate::cycle_state::{Held, Look};
use crate::deferred::Unbuilt;
use crate::dep_stack::Placement;
use crate::locks::Lock;
use crate::replay::{self, Delivered, Delivery, Entry, ReadKey, Retry, VFrame};
use crate::reserve;
use crate::segmented_nodes::{NOT_REPLAYABLE, REPLAYABLE};

#[cfg(test)]
thread_local! {
    /// Diverts every frame `divert_for_deliveries` sees, delivery or not.
    pub(crate) static DIVERT_EVERY_RETRY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// What the doom point saw, for tests.
#[cfg(test)]
#[derive(Clone, Debug)]
// Read only by the replay tests, which loom does not build.
#[cfg_attr(loom, allow(dead_code))]
pub(crate) struct DoomSeen {
    pub(crate) node: u32,
    /// The node's state after its release.
    pub(crate) state: NodeState,
    pub(crate) verified_at: u64,
    /// Frames on the stack at the doom, read by a release-only test.
    #[cfg_attr(debug_assertions, allow(dead_code))]
    pub(crate) depth: usize,
    /// The id of the context the doom pushed its entry into.
    pub(crate) context: u64,
}

/// An unwind a reading frame took from a pending delivery, for tests.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeliverySeen {
    pub(crate) frame: u32,
    pub(crate) read: u32,
    pub(crate) cycle: bool,
    pub(crate) kind: ReadKind,
}

/// The context a driver found when it picked an entry, for tests.
#[cfg(test)]
#[cfg_attr(loom, allow(dead_code))]
#[derive(Clone, Debug)]
pub(crate) struct EntrySeen {
    pub(crate) context: u64,
    pub(crate) level: u32,
    /// The front entry of level 0, the target.
    pub(crate) target: Option<u32>,
    /// The number of levels; a copy of every level's front would make
    /// the log quadratic in a ring's length at budget 1.
    pub(crate) levels: usize,
}

/// How a virtual frame's pop left its context's driver.
enum VPop {
    Next,
    /// A doom (`true`) or a dropped chain pushed entries: the driver picks
    /// its next entry again.
    Restart(bool),
    /// The episode rooted at the node dissolved, which ended the run of the
    /// root's frame: the retry that computes the root alone is another run,
    /// as the unbounded run's frame runs the closure again. A resumed
    /// episode pops at a member run's frame, at the root's position, so the
    /// root is named.
    Dissolved(u32),
    /// The run of the frame at the popped position ended: no retry
    /// computes it again.
    Finished(u32, Ended),
}

/// How the run of a popped frame ended.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ended {
    Landed,
    /// Landed not Clean: the walk that computed the node goes on as after
    /// a compute that left it so.
    Unsettled,
    Unwound,
    /// The walk that computed the node contained the run's unwind, so the
    /// walk goes on as after a compute that did not settle.
    Contained,
}

/// Places the frames a virtual frame's pop pushes above its position, and
/// restores the stack's bias when the pop ends, an unwind included.
struct Placed<'a, D: DepStack>(&'a D, Placement);

impl<D: DepStack> Drop for Placed<'_, D> {
    fn drop(&mut self) {
        self.0.restore_placement(self.1);
    }
}

/// A frame below a doomed read, as marking sees it.
struct Below {
    node: u32,
    cidx: u32,
    /// The stack's depth with this frame on top.
    depth: u32,
    ctl: *const ComputeCtl,
    step: replay::Step,
    path: u64,
}

/// The ids of runtimes without a budget, which never doom a frame, so
/// another runtime's read from one of their frames keeps nothing for a
/// retry. `UNBUDGETED_LEN` mirrors the list's length, so a read finds it
/// empty without the lock.
static UNBUDGETED: std::sync::Mutex<Vec<RuntimeId>> = std::sync::Mutex::new(Vec::new());
static UNBUDGETED_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
thread_local! {
    /// Run once by this thread's next `BudgetEntry::note`, before it locks
    /// `UNBUDGETED`.
    pub(crate) static NOTE_DECIDED: std::cell::Cell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::Cell::new(None) };
    /// Landings this thread's `hold_landing` held as unsettled.
    pub(crate) static UNSETTLED_HOLDS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// A runtime's place in `UNBUDGETED`, which its drop gives up, and the
/// `BUDGET_CLOCK` count its budget last changed at.
pub(crate) struct BudgetEntry {
    id: RuntimeId,
    changed: std::sync::atomic::AtomicU64,
}

impl BudgetEntry {
    pub(crate) fn new(id: RuntimeId, max_frames: u32) -> Self {
        let entry = Self {
            id,
            changed: std::sync::atomic::AtomicU64::new(0),
        };
        if max_frames == u32::MAX {
            entry.note(|| true);
        }
        entry
    }

    /// Lists the runtime as unbudgeted, or takes it off the list, as
    /// `unbudgeted` says under the list's lock: setters store their budget
    /// inside `unbudgeted`, so the lock orders racing setters whole.
    fn note(&self, unbudgeted: impl FnOnce() -> bool) {
        #[cfg(test)]
        if let Some(hook) = NOTE_DECIDED.with(std::cell::Cell::take) {
            hook();
        }
        let mut list = UNBUDGETED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let unbudgeted = unbudgeted();
        match (list.iter().position(|&id| id == self.id), unbudgeted) {
            (None, true) => list.push(self.id),
            (Some(at), false) => {
                list.swap_remove(at);
            }
            _ => {}
        }
        UNBUDGETED_LEN.store(list.len(), std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for BudgetEntry {
    fn drop(&mut self) {
        if UNBUDGETED_LEN.load(std::sync::atomic::Ordering::Relaxed) != 0 {
            self.note(|| false);
        }
    }
}

/// Whether runtime `id` is listed in `UNBUDGETED`.
pub(super) fn unbudgeted(id: RuntimeId) -> bool {
    UNBUDGETED_LEN.load(std::sync::atomic::Ordering::Relaxed) != 0
        && UNBUDGETED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(&id)
}

impl<C: Cells> Runtime<C> {
    /// Sets the frame and byte budgets `budget` gives, and keeps
    /// `UNBUDGETED` in step with them, both under the list's lock. A change
    /// takes a `BUDGET_CLOCK` count, stored before the budgets so a read
    /// that sees them sees it (`hold_running_frames`).
    pub(super) fn set_budget(&self, budget: impl FnOnce() -> (u32, usize)) {
        use std::sync::atomic::Ordering::Relaxed;
        self.budget_entry.note(|| {
            let (frames, bytes) = budget();
            if (frames, bytes)
                != (
                    self.max_frames.load(Relaxed),
                    self.max_stack_bytes.load(Relaxed),
                )
            {
                self.budget_entry
                    .changed
                    .store(cycle_state::tick_budget_clock(), Relaxed);
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                self.max_frames.store(frames, Relaxed);
                self.max_stack_bytes.store(bytes, Relaxed);
            }
            !self.budgeted()
        });
    }

    /// Raises `NO_DOOM` on this runtime's frames on this thread when it
    /// first meets a budget change made since its outermost entry began,
    /// and returns whether it did: those frames may have read under the
    /// budget before, which kept nothing a retry under this one needs.
    #[cold]
    #[inline(never)]
    fn hold_running_frames(&self) -> bool {
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        let changed = self
            .budget_entry
            .changed
            .load(std::sync::atomic::Ordering::Relaxed);
        if !cycle_state::meet_budget_change(self.runtime_id, changed) {
            return false;
        }
        self.dep_stack
            .raise_on_undoomed(self.runtime_id, cycle::NO_DOOM);
        true
    }

    /// [`cycle::KeepResolver::keeps`] for this runtime: only a replayable
    /// compute of a budgeted runtime can be retried.
    ///
    /// # Safety
    ///
    /// `r` is a running compute's runtime's [`Self::keep_resolver_ptr`].
    pub(super) unsafe fn resolve_keep(r: *const cycle::KeepResolver, slot: u32) -> bool {
        // SAFETY: the running compute borrows the runtime, so it is in
        // place around its `keep_resolver`, and `r`'s provenance covers it.
        let rt = unsafe {
            &*r.byte_sub(std::mem::offset_of!(Self, keep_resolver))
                .cast::<Self>()
        };
        rt.budgeted() && rt.replayable(slot)
    }

    /// The `keep_resolver` pointer for [`Self::resolve_keep`], offset from
    /// the whole runtime: under Stacked Borrows one taken from the field,
    /// even by `&raw const`, may not reach the rest of the runtime.
    pub(super) fn keep_resolver_ptr(&self) -> *const cycle::KeepResolver {
        std::ptr::from_ref(self)
            .wrapping_byte_add(std::mem::offset_of!(Self, keep_resolver))
            .cast()
    }

    /// Whether node `slot`'s compute may be doomed.
    pub(super) fn replayable(&self, slot: u32) -> bool {
        let bits = C::u8_load_relaxed(self.nodes.get_with_flags(slot).1);
        bits & NOT_REPLAYABLE == 0 && (bits & REPLAYABLE != 0 || self.all_replayable)
    }

    /// Whether the node a frame is blocked reading (`slot | generation <<
    /// 32`) has a type with a placeholder, so the frame can take one.
    fn reading_has_placeholder(&self, reading: u64) -> bool {
        let node = self.nodes.get(reading as u32);
        let tag = node.type_tag();
        C::fence_acquire();
        node.generation() == (reading >> 32) as u32 && self.arenas.erased(tag).has_placeholder()
    }

    /// Whether the frame `depth` deep with control block `ctl` can't be
    /// doomed: it is `NO_DOOM`, or it took a foreign read's hold within this
    /// runtime's budget where it read, past which the stack bound wins, as
    /// for a ticket payload, and its retry reads the other runtime again.
    /// A frame running when this thread meets a budget change is
    /// `NO_DOOM` from then on.
    fn undoomable(&self, ctl: &ComputeCtl, depth: u32) -> bool {
        let flags = ctl.flags.get();
        flags & cycle::NO_DOOM != 0
            || (flags & cycle::FOREIGN_HOLD != 0
                && self
                    .dep_stack
                    .hold_position(ctl)
                    .is_some_and(|at| !self.over_budget_at(depth, at)))
            || (self.hold_running_frames() && ctl.has(cycle::NO_DOOM))
    }

    /// Whether a read from a frame `depth` frames deep is at or past a
    /// budget.
    fn over_budget(&self, depth: u32) -> bool {
        let here = 0_u8;
        self.over_budget_at(
            depth,
            std::hint::black_box(std::ptr::from_ref(&here)) as usize,
        )
    }

    /// [`Self::over_budget`] for a read made at stack address `here`.
    fn over_budget_at(&self, depth: u32, here: usize) -> bool {
        if depth >= self.max_frames.load(std::sync::atomic::Ordering::Relaxed) {
            return true;
        }
        let bytes = self
            .max_stack_bytes
            .load(std::sync::atomic::Ordering::Relaxed);
        bytes != 0 && here.abs_diff(cycle_state::stack_base()) > bytes
    }

    /// A speculative entry's run stashes only Clean values: a node it would
    /// land Dirty, poisoned, over a dep that is neither Clean nor stashed, or
    /// over a read of a value a later landing changed (`KEPT_READ`), has a
    /// history the unbounded run may never give it, and a later walk
    /// of its deps would differ from a first run of its closure. Nor does it
    /// stash a node whose marked frame still owes its retry an outcome: the
    /// unbounded run's frame lands the node first, and its readers read
    /// that landing. Such a pop drops the chain. A doomed frame's pop lands
    /// nothing anyway.
    #[cold]
    #[inline(never)]
    pub(super) fn speculative_lands_dirty(
        &self,
        id: NodeId,
        word: u64,
        ctl: &ComputeCtl,
        deps: &[(NodeId, u32)],
    ) -> bool {
        word & value_code::REPLAY_BIT != 0
            && replay::speculative()
            && !ctl.has(cycle::DOOMED)
            && (ctl.flags.get() & (cycle::POISONED | cycle::HANDLED | cycle::KEPT_READ) != 0
                || deps.iter().any(|&(d, generation)| {
                    !matches!(
                        state::load::<C>(self.nodes.get(d.0).state_cell()),
                        NodeState::Clean | NodeState::Cyclic
                    ) && !self.stashed(d.0, generation, false)
                })
                || self.frame_owes_outcome(id))
    }

    fn frame_owes_outcome(&self, id: NodeId) -> bool {
        let rt = self.runtime_id;
        let generation = self.nodes.get(id.0).generation();
        replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .any(|x| x.pending.owes_outcome(id.0, generation))
        })
    }

    /// A speculative run's pop that would land `id` Clean: the run's value
    /// and deps are stashed for later speculative reads, and the node lands
    /// back in the state it was claimed from, with no deps, stamp or value
    /// the unbounded run never gave it.
    #[cold]
    #[inline(never)]
    pub(super) fn stash_speculative(
        &self,
        id: NodeId,
        generation: u32,
        attempt: Attempt,
        ctl: &ComputeCtl,
        deps: Vec<(NodeId, u32)>,
        bag: &mut Deferred<C>,
    ) -> Computed {
        let value = ctl
            .buf
            .take()
            .expect("incr-core: a speculative run buffers its value");
        let node = self.nodes.get(id.0);
        let unchanged = !attempt.first
            && attempt.verify_rev == attempt.stamp_rev
            && match catch_unwind(AssertUnwindSafe(|| {
                self.arenas
                    .erased(node.type_tag())
                    .equals_erased(node.arena_slot(), &*value)
            })) {
                Ok(equal) => equal,
                Err(payload) => {
                    bag.push_erased(Box::new(Unbuilt::new(payload)));
                    false
                }
            };
        let index = self.stashed_index(id, ctl);
        let kept = self.stash(
            id.0,
            generation,
            Stash {
                outcome: Outcome::Value {
                    value: std::sync::Arc::new(StashedValue(Unbuilt::new(value))),
                    unchanged,
                },
                deps,
                verify_rev: attempt.verify_rev,
                index,
            },
            attempt.stamp_rev,
        );
        if let Err(refused) = kept {
            self.discard_stashed_children(id, refused.index, bag);
            bag.push_erased(Box::new(refused.outcome));
        }
        let target = if attempt.first {
            NodeState::New
        } else {
            NodeState::Dirty
        };
        self.land(id, node, target, target);
        Computed::Done
    }

    /// A speculative verify that would land `id` Clean, against the
    /// `since` it was verified at: the node lands back Dirty, and its stash
    /// stands for the verify, so a speculative read takes the node's value
    /// and a dependent's verify counts it Clean, and the real claim adopts
    /// it as the unbounded run's verify.
    #[cold]
    #[inline(never)]
    pub(super) fn stash_verify(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        generation: u32,
        since: u64,
        stamp_rev: u64,
        verify_rev: u64,
    ) -> Computed {
        // A set whose walk was pending at the claim may dirty a dep the
        // scan saw Clean.
        if verify_rev == stamp_rev {
            let _ = self.stash(
                id.0,
                generation,
                Stash {
                    outcome: Outcome::Verified { since },
                    deps: self.current_deps(node),
                    verify_rev,
                    index: None,
                },
                stamp_rev,
            );
        }
        self.land(id, node, NodeState::Dirty, NodeState::Dirty);
        Computed::Done
    }

    /// Whether, in a speculative run, node `slot` at `generation` has a
    /// stashed verify or a stashed recompute to the value it holds, which
    /// the unbounded run lands without a new `changed_at`.
    #[cold]
    #[inline(never)]
    pub(super) fn unchanged_stash(&self, slot: u32, generation: u32) -> bool {
        if !replay::speculative() || C::u32_load_relaxed(&self.stash_count) == 0 {
            return false;
        }
        let revision = self.current_revision();
        let mut s = self.stashes.write();
        s.revision == revision
            && (s.map.get_mut(&(slot, generation))).is_some_and(|e| match e.get_mut().outcome {
                Outcome::Verified { since } => self.nodes.get(slot).verified_at() == since,
                Outcome::Value { unchanged, .. } => unchanged,
                Outcome::Panic(_) | Outcome::Tried => false,
            })
    }

    /// Whether node `slot` at `generation` has a stashed verify this
    /// revision.
    #[cfg(all(test, not(loom)))]
    pub(super) fn stashed_verify(&self, slot: u32, generation: u32) -> bool {
        let revision = self.current_revision();
        let mut s = self.stashes.write();
        s.revision == revision
            && (s.map.get_mut(&(slot, generation)))
                .is_some_and(|e| matches!(e.get_mut().outcome, Outcome::Verified { .. }))
    }

    /// Node `node`'s deps, each at its current generation.
    fn current_deps(&self, node: &NodeData<C>) -> Vec<(NodeId, u32)> {
        let mut deps = Vec::new();
        node.for_each_dep(|d| deps.push((d, self.nodes.get(d.0).generation())));
        deps
    }

    /// A speculative run that unwound or was dropped after its closure
    /// returned: the value it buffered is one the unbounded run never built,
    /// so it drops under `isolate`.
    #[cold]
    #[inline(never)]
    pub(super) fn unbuild_speculative(&self, ctl: &ComputeCtl, bag: &mut Deferred<C>) {
        if let Some(value) = ctl.buf.take() {
            bag.push_erased(Box::new(Unbuilt::new(value)));
        }
    }

    /// A real claim of `id` at `generation`, from Dirty if `claimed_dirty`,
    /// finds the node stashed: the stash stands in for the run the unbounded
    /// run makes here, so the closure runs once. Its stashed deps are adopted
    /// first, depth first with an explicit stack, in the order that run
    /// lands them. Any dep that is neither Clean at or below the stash's
    /// verify revision nor adoptable fails the nodes above it, which keep
    /// their claims' source states and compute; the root keeps its claim.
    #[cold]
    #[inline(never)]
    pub(super) fn adopt_stash(
        &self,
        id: NodeId,
        generation: u32,
        claimed_dirty: bool,
        walked: bool,
        bag: &mut Deferred<C>,
    ) -> Adopted {
        let rt = self.runtime_id;
        if replay::speculative() || replay::frame_records(rt, id.0, generation) {
            return Adopted::No;
        }
        // A root's verify is its claim's own, after the walk settled its
        // deps; a taint memo's claim runs the closure, as unbounded.
        let Some(stash) = self.take_stash(id.0, generation, false) else {
            return Adopted::No;
        };
        let mut stack = vec![Adopting {
            id,
            claimed_dirty,
            stash,
            next: 0,
        }];
        let mut failed = false;
        loop {
            let top = stack.last_mut().expect("incr-core: adoption has a root");
            if !failed && top.next < top.stash.deps.len() {
                let (d, dg) = top.stash.deps[top.next];
                top.next += 1;
                let rev = match top.stash.outcome {
                    Outcome::Verified { since } => since,
                    _ => top.stash.verify_rev,
                };
                if self.nodes.get(d.0).generation() != dg {
                    failed = true;
                } else if !self.observed_clean(d, rev) {
                    match self.claim_stashed(d, dg) {
                        Some(dep) => stack.push(dep),
                        None => failed = true,
                    }
                }
                continue;
            }
            let a = stack.pop().expect("incr-core: adoption has a root");
            let root = stack.is_empty();
            if !failed && root {
                if let Outcome::Panic(payload) = a.stash.outcome {
                    return Adopted::Panicked(AdoptedPanic {
                        payload: payload.into_inner(),
                        deps: a.stash.deps,
                    });
                }
            }
            failed |= !matches!(
                a.stash.outcome,
                Outcome::Value { .. } | Outcome::Verified { .. }
            );
            if failed {
                bag.push_erased(Box::new(a.stash.outcome));
                if root {
                    return Adopted::No;
                }
                let target = if a.claimed_dirty {
                    NodeState::Dirty
                } else {
                    NodeState::New
                };
                self.discard_stashed_children(a.id, a.stash.index, bag);
                self.land(a.id, self.nodes.get(a.id.0), target, target);
                continue;
            }
            let adopted = a.id;
            match self.land_adopted(a, root, bag) {
                Landing::Landed(NodeState::Clean) if root => {
                    return Adopted::Landed(Computed::Done)
                }
                // A verify's dep must still be unchanged since the verify
                // once it lands from its own stash.
                Landing::Landed(NodeState::Clean) => {
                    if let Some(Outcome::Verified { since }) =
                        stack.last().map(|t| &t.stash.outcome)
                    {
                        failed = !self.observed_clean(adopted, *since);
                    }
                }
                Landing::Landed(_) if root => {
                    self.poison_reader(walked, cycle::POISONED);
                    return Adopted::Landed(Computed::Unsettled);
                }
                Landing::Stale if root => return Adopted::Stale,
                Landing::Unlanded if root => return Adopted::No,
                Landing::Panicked(raised) => return Adopted::Panicked(raised),
                Landing::Landed(_) | Landing::Stale | Landing::Unlanded => failed = true,
            }
        }
    }

    /// An adopted stash landed the claimed node `id` in place of the run the
    /// unbounded run makes here, at the next frame position: that run's pop
    /// retires the notes, and their copies, above it, and ends its run. No
    /// run started, so the run's path is the position's.
    pub(super) fn adopted_pop(&self, id: NodeId) {
        let cidx = self.dep_stack.next_cidx();
        self.retire_notes_above(cidx);
        self.retire_copies(&replay::Retire::Above(cidx));
        if value_code::thread_word() & value_code::REPLAY_BIT != 0 {
            self.end_run_along(id, cidx, |_| self.path_at(cidx));
        }
    }

    /// Drops the stash of node `id` at `generation`, which its claim will
    /// not adopt, with the children a creator's stashed run made.
    #[cold]
    #[inline(never)]
    pub(super) fn drop_stash(&self, id: NodeId, generation: u32, bag: &mut Deferred<C>) {
        if replay::speculative() {
            return;
        }
        if let Some(stash) = self.take_stash(id.0, generation, false) {
            self.discard_stashed_children(id, stash.index, bag);
            bag.push_erased(Box::new(stash.outcome));
        }
    }

    /// Claims stashed node `id` at `generation` for adoption, from New or
    /// Dirty, with its stash.
    fn claim_stashed(&self, id: NodeId, generation: u32) -> Option<Adopting> {
        let rt = self.runtime_id;
        if replay::frame_records(rt, id.0, generation) || !self.stashed(id.0, generation, false) {
            return None;
        }
        let node = self.nodes.get(id.0);
        let claimed_dirty =
            state::try_claim_compute::<C>(node.state_cell()).ok()? == NodeState::Dirty;
        let source = if claimed_dirty {
            NodeState::Dirty
        } else {
            NodeState::New
        };
        let stash = (node.generation() == generation)
            .then(|| self.take_stash(id.0, generation, true))
            .flatten();
        let Some(stash) = stash else {
            self.land(id, node, source, source);
            return None;
        };
        Some(Adopting {
            id,
            claimed_dirty,
            stash,
            next: 0,
        })
    }

    /// Lands claimed node `a.id` from its stash, as `compute_attempts`
    /// lands a run. The adoption's `root` keeps its claim when it does not
    /// land, for its claim's compute or for its write panic's raise.
    fn land_adopted(&self, a: Adopting, root: bool, bag: &mut Deferred<C>) -> Landing {
        let (node, flags) = self.nodes.get_with_flags(a.id.0);
        if let Outcome::Verified { since } = a.stash.outcome {
            // A real landing since the speculative verify supersedes it.
            if !a.claimed_dirty
                || node.verified_at() != since
                || self.current_deps(node) != a.stash.deps
            {
                let target = if a.claimed_dirty {
                    NodeState::Dirty
                } else {
                    NodeState::New
                };
                self.land(a.id, node, target, target);
                return Landing::Unlanded;
            }
            node.max_verified_at(
                a.stash
                    .verify_rev
                    .min(C::walked(&self.walk, &self.revision)),
            );
            let landed = self.land_single(a.id, node, flags, NodeState::Clean, NodeState::Dirty);
            self.end_reservation_landed(a.id, node, landed);
            if landed == NodeState::Clean {
                self.record_trace(a.id, crate::trace::TraceAction::VerifiedClean);
            }
            return Landing::Landed(landed);
        }
        let first = !a.claimed_dirty;
        let stamp_rev = self.current_revision();
        let verify_rev = a
            .stash
            .verify_rev
            .min(C::walked(&self.walk, &self.revision));
        let arena = self.arenas.erased(node.type_tag());
        let wrote = std::cell::Cell::new(0_u8);
        let Outcome::Value { value, .. } = a.stash.outcome else {
            unreachable!("incr-core: a panic's stash lands only through its raise")
        };
        let source = if first {
            NodeState::New
        } else {
            NodeState::Dirty
        };
        // A run that opened the index since took the run's children over.
        if a.stash
            .index
            .is_some_and(|i| self.index_at(a.id).with(|s| s.epoch) != i.epoch)
        {
            bag.push_erased(Box::new(value));
            if !root {
                self.land(a.id, node, source, source);
            }
            return Landing::Unlanded;
        }
        // A speculative read still cloning the value shares it, so the node
        // lands a clone. A panicking `Clone` is no part of the unbounded
        // run: the node computes.
        let value = match std::sync::Arc::try_unwrap(value) {
            Ok(value) => value.0.into_inner(),
            Err(shared) => {
                bag.push_erased(Box::new(std::sync::Arc::clone(&shared)));
                match catch_unwind(AssertUnwindSafe(|| arena.clone_erased(shared.0.value()))) {
                    Ok(value) => value,
                    Err(payload) => {
                        bag.push_erased(Box::new(Unbuilt::new(payload)));
                        if !root {
                            self.discard_stashed_children(a.id, a.stash.index, bag);
                            self.land(a.id, node, source, source);
                        }
                        return Landing::Unlanded;
                    }
                }
            }
        };
        let changed = match catch_unwind(AssertUnwindSafe(|| {
            arena.write_erased(node.arena_slot(), value, first, &wrote)
        })) {
            Ok((changed, displaced)) => {
                if let Some(d) = displaced {
                    bag.push_erased(d);
                }
                changed
            }
            // The unbounded run's write panics after its run too, and its
            // compute raises the payload.
            Err(payload) if root => {
                return Landing::Panicked(AdoptedPanic {
                    payload,
                    deps: a.stash.deps,
                });
            }
            Err(payload) => {
                bag.push_erased(Box::new(Unbuilt::new(payload)));
                self.discard_stashed_children(a.id, a.stash.index, bag);
                let target = self.discard_target(
                    a.id,
                    first,
                    wrote.get() != 0,
                    stamp_rev,
                    ChildLanding::Keep,
                    bag,
                );
                self.land(a.id, node, target, target);
                return Landing::Unlanded;
            }
        };
        if changed || first {
            node.max_changed_at(stamp_rev);
        }
        let published = self.publish_deps(a.id, &a.stash.deps, None, verify_rev, bag);
        if published == Publish::Stale {
            if first && wrote.get() == 0 {
                node.set_verified_at(0);
            }
            self.children_landing(a.id, flags, ChildLanding::Discard, true, bag);
            self.land_single(a.id, node, flags, NodeState::New, NodeState::New);
            return Landing::Stale;
        }
        node.max_verified_at(verify_rev);
        let target = if published == Publish::Superseded {
            NodeState::Dirty
        } else {
            NodeState::Clean
        };
        if let Some(index) = a.stash.index {
            let landing = if target == NodeState::Clean {
                ChildLanding::Sweep
            } else {
                ChildLanding::Keep
            };
            self.land_children(a.id, landing, index.opened, bag);
        }
        let landed = self.land_single(a.id, node, flags, target, NodeState::Dirty);
        self.end_reservation_landed(a.id, node, landed);
        self.record_trace(
            a.id,
            crate::trace::TraceAction::Recomputed {
                value_changed: changed,
            },
        );
        Landing::Landed(landed)
    }

    /// Discards the children that claimed creator `id`'s stashed run
    /// created, as its run's discard landing would: no landing adopts the
    /// run.
    fn discard_stashed_children(
        &self,
        id: NodeId,
        index: Option<StashedIndex>,
        bag: &mut Deferred<C>,
    ) {
        if index.is_some_and(|i| i.opened && self.index_at(id).with(|s| s.epoch) == i.epoch) {
            self.land_children(id, ChildLanding::Discard, true, bag);
        }
    }

    /// Whether a speculative run's compute of `id` at `generation` finds it
    /// stashed, so that it is not computed again: the stash is what the
    /// run's reads of it return.
    #[cold]
    #[inline(never)]
    pub(super) fn stashed_claim(&self, id: NodeId, generation: u32) -> bool {
        replay::speculative() && self.stashed(id.0, generation, true)
    }

    /// A speculative read's value of `handle` from its stash.
    #[cold]
    #[inline(never)]
    pub(super) fn stashed_read<T: Value>(&self, handle: Incr<T>) -> Option<T> {
        if !replay::speculative() {
            return None;
        }
        if C::u32_load_relaxed(&self.stash_count) == 0 {
            return None;
        }
        let revision = self.current_revision();
        let mut s = self.stashes.write();
        if s.revision != revision {
            return None;
        }
        let stash = s.map.get_mut(&(handle.slot(), handle.generation()))?;
        let read = match &stash.get_mut().outcome {
            Outcome::Value { value, .. } => Ok(Some(std::sync::Arc::clone(value))),
            Outcome::Verified { .. } => Ok(None),
            Outcome::Panic(_) | Outcome::Tried => Err(()),
        };
        drop(s);
        // A run that read a panic would unwind with it, which a speculative
        // run's discard stands in for. A node tried in this revision is not
        // run again, so its reader is discarded too.
        let value = read.unwrap_or_else(|()| self.drop_unwind())?;
        value_code::run(|| value.0.value().downcast_ref::<T>().cloned())
    }

    /// A speculative run's closure panicked, over a frame that read only
    /// settled values: the panic is stashed for the real claim to raise, and
    /// the run unwinds with a discard in its place. True with it if the
    /// stash holds children the run created, which its landing leaves.
    #[cold]
    #[inline(never)]
    pub(super) fn stash_panic(
        &self,
        id: NodeId,
        ctl: &ComputeCtl,
        deps: &[(NodeId, u32)],
        attempt: &Attempt,
        payload: Box<dyn Any + Send>,
    ) -> (Box<dyn Any + Send>, bool) {
        if ctl.flags.get() & (cycle::DOOMED | cycle::POISONED | cycle::KEPT_READ) != 0
            || !replay::speculative()
        {
            return (payload, false);
        }
        let index = self.stashed_index(id, ctl);
        let kept = self.stash(
            id.0,
            self.nodes.get(id.0).generation(),
            Stash {
                outcome: Outcome::Panic(Unbuilt::new(payload)),
                deps: deps.to_vec(),
                verify_rev: attempt.verify_rev,
                index,
            },
            attempt.stamp_rev,
        );
        let children = kept.is_ok() && index.is_some_and(|i| i.opened);
        (Box::new(cycle::DiscardUnwind), children)
    }

    /// Creator `id`'s child index as its claimed run leaves it, `None` for
    /// a node that has none.
    fn stashed_index(&self, id: NodeId, ctl: &ComputeCtl) -> Option<StashedIndex> {
        let flags = self.nodes.get_with_flags(id.0).1;
        (C::u8_load_relaxed(flags) & HAS_INDEX != 0).then(|| StashedIndex {
            epoch: self.index_at(id).with(|s| s.epoch),
            opened: !ctl.children.get().is_null(),
        })
    }

    /// The payload of an adopted panic, with its deps recorded in the frame
    /// it is raised in, as the closure's reads before the panic recorded
    /// them.
    #[cold]
    #[inline(never)]
    pub(super) fn raise_adopted(&self, raised: AdoptedPanic) -> Box<dyn Any + Send> {
        for &(d, generation) in &raised.deps {
            self.dep_stack
                .record_dep_or_foreign(self.runtime_id, d, generation);
        }
        raised.payload
    }

    /// Marks a speculative run of Native node `id` at `generation` as tried
    /// before its closure runs, so that no other speculative run computes it
    /// in this revision whether or not this one keeps a stash. A replayable
    /// node is not marked, since its closure may run again. False if another
    /// speculative run marked it after this claim's check.
    #[cold]
    #[inline(never)]
    pub(super) fn mark_tried(&self, id: NodeId, generation: u32) -> bool {
        let revision = self.current_revision();
        self.stash(
            id.0,
            generation,
            Stash {
                outcome: Outcome::Tried,
                deps: Vec::new(),
                verify_rev: 0,
                index: None,
            },
            revision,
        )
        .unwrap_or_else(|_| self.current_revision() != revision)
    }

    /// Keeps `stash` for node `slot` at `generation`, for `revision`, the
    /// one its run claimed in, and says whether the node had no entry. A
    /// node already stashed keeps its first stash, which speculative reads
    /// may have returned; a run's stash replaces its own tried mark. A
    /// stash not kept, because a set ended its revision or the node keeps
    /// its first, comes back.
    fn stash(
        &self,
        slot: u32,
        generation: u32,
        stash: Stash,
        revision: u64,
    ) -> Result<bool, Stash> {
        let (displaced, ended) = {
            let mut s = self.stashes.write();
            if self.current_revision() != revision {
                return Err(stash);
            }
            self.park(ParkPoint::StashAfterCheck(NodeId(slot)));
            let ended = (s.revision != revision).then(|| std::mem::take(&mut s.map));
            s.revision = revision;
            let displaced = match s.map.entry((slot, generation)) {
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    let kept = o.get_mut().get_mut();
                    if !matches!(kept.outcome, Outcome::Tried) {
                        return Err(stash);
                    }
                    Some(std::mem::replace(kept, stash))
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(Exclusive(stash));
                    None
                }
            };
            C::u32_store_relaxed(&self.stash_count, s.map.len() as u32);
            (displaced, ended)
        };
        let vacant = displaced.is_none();
        drop(displaced);
        if let Some(ended) = ended {
            self.drop_ended(ended);
        }
        // Pairs with the fence in a set between its bump and its load of
        // `stash_count`: a set that read the count before this store bumped
        // the revision before this load, and its stashes are this one's to
        // drop.
        C::fence_seqcst();
        if self.current_revision() == revision {
            return Ok(vacant);
        }
        let mut ended = {
            let mut s = self.stashes.write();
            if s.revision != revision {
                return Ok(vacant);
            }
            C::u32_store_relaxed(&self.stash_count, 0);
            std::mem::take(&mut s.map)
        };
        let mine = ended.remove(&(slot, generation));
        self.drop_ended(ended);
        mine.map_or(Ok(vacant), |mine| Err(mine.0))
    }

    /// Whether node `slot` at `generation` has a stash this revision, a
    /// tried mark included if `tried`.
    pub(super) fn stashed(&self, slot: u32, generation: u32, tried: bool) -> bool {
        if C::u32_load_relaxed(&self.stash_count) == 0 {
            return false;
        }
        let revision = self.current_revision();
        let mut s = self.stashes.write();
        s.revision == revision
            && (s.map.get_mut(&(slot, generation)))
                .is_some_and(|e| tried || !matches!(e.get_mut().outcome, Outcome::Tried))
    }

    /// Takes the stash of node `slot` at `generation` for a real run that
    /// adopts it, a verify's only if `verified`. A Native node's run keeps a
    /// tried mark in its place for the revision, whether or not the
    /// adoption lands it.
    fn take_stash(&self, slot: u32, generation: u32, verified: bool) -> Option<Stash> {
        if C::u32_load_relaxed(&self.stash_count) == 0 {
            return None;
        }
        let revision = self.current_revision();
        let native = !self.replayable(slot);
        let mut s = self.stashes.write();
        if s.revision != revision {
            return None;
        }
        let entry = s.map.get_mut(&(slot, generation))?.get_mut();
        let verify = matches!(entry.outcome, Outcome::Verified { .. });
        if matches!(entry.outcome, Outcome::Tried) || verify && !verified {
            return None;
        }
        if native && !verify {
            let tried = Stash {
                outcome: Outcome::Tried,
                deps: Vec::new(),
                verify_rev: 0,
                index: None,
            };
            return Some(std::mem::replace(entry, tried));
        }
        let stash = s.map.remove(&(slot, generation))?;
        C::u32_store_relaxed(&self.stash_count, s.map.len() as u32);
        Some(stash.0)
    }

    /// Drops every stash: a revision began, and a stash holds only for the
    /// revision its run read in.
    #[cold]
    #[inline(never)]
    pub(super) fn drop_stashes(&self) {
        let ended = {
            let mut s = self.stashes.write();
            C::u32_store_relaxed(&self.stash_count, 0);
            std::mem::take(&mut s.map)
        };
        self.drop_ended(ended);
    }

    /// Drops the stashes of an ended revision, with the children their
    /// creators' runs created.
    fn drop_ended(&self, mut ended: HashMap<(u32, u32), Exclusive<Stash>>) {
        let mut bag = Deferred::new();
        for (&(slot, generation), stash) in &mut ended {
            if let Some(index) = stash.get_mut().index.filter(|i| i.opened) {
                self.discard_dropped_children(NodeId(slot), generation, index, &mut bag);
            }
        }
        drop(ended);
        bag.drain_isolated();
    }

    /// Discards the children that creator `id`'s dropped stash at
    /// `generation` created, under a claim taken for that alone. A creator
    /// another claim holds leaves them to that claim's run.
    #[cold]
    #[inline(never)]
    fn discard_dropped_children(
        &self,
        id: NodeId,
        generation: u32,
        index: StashedIndex,
        bag: &mut Deferred<C>,
    ) {
        let node = self.nodes.get(id.0);
        let Ok(source) = state::try_claim_compute::<C>(node.state_cell()) else {
            return;
        };
        if node.generation() == generation {
            self.discard_stashed_children(id, Some(index), bag);
        }
        self.land_unrun(id, node, source);
    }

    /// Diverts the Clean reads of the frame just pushed for `node` at
    /// `cidx` while a replay context holds a delivery for one of its reads
    /// (see `Frame::read_rt`): the unbounded run's frame made that read
    /// before the node landed Clean, and its retry must take what the
    /// context kept for it, not the Clean value.
    #[cold]
    #[inline(never)]
    pub(super) fn divert_for_deliveries(&self, node: NodeId, cidx: u32) {
        if value_code::thread_word() & value_code::REPLAY_BIT == 0 || replay::speculative() {
            return;
        }
        #[cfg(test)]
        if DIVERT_EVERY_RETRY.with(std::cell::Cell::get) {
            self.dep_stack.divert_clean_reads();
            return;
        }
        if self.frame_awaits_deliveries(node.0, cidx) {
            self.dep_stack.divert_clean_reads();
        }
    }

    /// Whether a context holds a delivery that the frame of `node` at
    /// `cidx` can still take with its own runtime's reads.
    fn frame_awaits_deliveries(&self, node: u32, cidx: u32) -> bool {
        let rt = self.runtime_id;
        let frame = (node, self.nodes.get(node).generation(), cidx);
        replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .any(|x| x.pending.frame_awaits(frame))
        })
    }

    /// Whether a context holds a delivery for the frame `f`'s next read of
    /// `reading`, under its path, episode and run seq now. A frame's path
    /// can change while it runs, as the virtual frames below it do, so
    /// only a read's own lookup matches it. The read's ordinal and key
    /// parts are found only once a delivery for its node is listed, since
    /// they scan the frame's reads and the stack.
    fn read_awaits_delivery(&self, f: ActiveRead, reading: u64) -> bool {
        let rt = self.runtime_id;
        let frame = (f.node.0, self.nodes.get(f.node.0).generation(), f.cidx);
        let listed = |ordinal, parts| {
            replay::with(|c| {
                c.iter()
                    .filter(|x| x.rt == rt)
                    .any(|x| x.pending.read_awaits(frame, reading, ordinal, parts))
            })
        };
        if !listed(None, None) {
            return false;
        }
        let ordinal =
            1 + self
                .dep_stack
                .reads_of(rt, NodeId(reading as u32), (reading >> 32) as u32);
        // SAFETY: the frame's control block is live while it is on the stack.
        let member = unsafe { &*f.ctl }.has(cycle::BUFFERED);
        let (_, path, episode, seq) = self.key_parts(f, member);
        listed(Some(ordinal), Some((path, episode, seq)))
    }

    /// Whether the frame `f`'s read under the thread word `word` takes a
    /// delivery: a replayed read outside value code and the contexts'
    /// borrow, in a run that is not speculative, by a frame that is not
    /// doomed.
    fn takes_deliveries(word: u64, f: ActiveRead) -> bool {
        word & value_code::REPLAY_BIT != 0
            && word & value_code::COUNTER_MASK == 0
            && !replay::held()
            && !replay::speculative()
            // SAFETY: the active frame's control block is live.
            && !unsafe { &*f.ctl }.has(cycle::DOOMED)
    }

    /// Whether a replay context holds a delivery for this read of
    /// `handle` by the frame `f` under `word`, which the read takes (see
    /// [`Self::takes_deliveries`]).
    pub(super) fn delivery_waits<T: Value>(
        &self,
        f: ActiveRead,
        handle: Incr<T>,
        word: u64,
    ) -> bool {
        if !Self::takes_deliveries(word, f) {
            return false;
        }
        let reading = u64::from(handle.slot()) | (u64::from(handle.generation()) << 32);
        self.read_awaits_delivery(f, reading)
    }

    /// Ends the diversion of the frame `f`, `rt`'s top frame, once no
    /// context holds a delivery its reads can take, or once it is doomed,
    /// since a doomed run takes none.
    pub(super) fn undivert_when_taken(&self, f: ActiveRead) {
        #[cfg(test)]
        if DIVERT_EVERY_RETRY.with(std::cell::Cell::get) {
            return;
        }
        let rt = self.runtime_id;
        // SAFETY: the active frame's control block is live.
        let doomed = unsafe { &*f.ctl }.has(cycle::DOOMED);
        if self.dep_stack.diverts_clean_reads(rt)
            && (doomed || !self.frame_awaits_deliveries(f.node.0, f.cidx))
        {
            self.dep_stack.end_diverted_reads(rt);
        }
    }

    /// A slow read that value code, a doomed or budgeted frame, or an open
    /// replay context sends off the hot path.
    #[cold]
    #[inline(never)]
    pub(super) fn read_cold<T: Value>(
        &self,
        handle: Incr<T>,
        observed: NodeState,
        kind: ReadKind,
        frame: Option<ActiveRead>,
        mut word: u64,
    ) -> Result<T, Cycle> {
        let foreign = word & value_code::FOREIGN_BIT != 0;
        if foreign {
            value_code::clear_foreign_bit();
            word &= !value_code::FOREIGN_BIT;
        }
        if word & value_code::COUNTER_MASK != 0 {
            if foreign && self.dep_stack.foreign_keeper(self.runtime_id).is_some() {
                // As in `diverted_clean_read`: the reader's retry or rerun
                // would read this runtime again.
                self.dep_stack.hold_foreign_reader();
            }
            return Ok(self.value_code_read(handle, self.nodes.get(handle.slot())));
        }
        // A read while a cycle's unwind kind is set comes from a closure that
        // caught that unwind: the kind is stale, and a panic of the node read
        // next would land as a cycle's discard instead of Failed. A caught
        // discard is raised again here: left set, the next frame to pop
        // would take it as its own swallow, and a context can absorb that
        // frame's discard, so the catching closure's value would land.
        if word & value_code::UNWIND_MASK != 0 {
            match cycle_state::unwind_kind() {
                Some(UnwindKind::Cycle) => {
                    cycle_state::set_unwind_kind(None);
                    word = value_code::thread_word();
                }
                Some(UnwindKind::Discard) if !std::thread::panicking() => {
                    resume_unwind(Box::new(cycle::DiscardUnwind))
                }
                _ => {}
            }
        }
        if word & value_code::REGEN_BIT != 0 {
            self.regen_read(NodeId(handle.slot()), handle.generation(), observed);
        }
        if foreign {
            return self.foreign_read(handle, observed, kind, word);
        }
        // A delivery goes to its frame's retry in a non-speculative run: a
        // speculative run that computes the same node is on another path,
        // and a doomed frame is the run before the retry.
        let _scope = (word & value_code::REPLAY_BIT != 0 && !replay::speculative()).then(|| {
            replay::ReadScope {
                rt: self.runtime_id,
                reading: u64::from(handle.slot()) | (u64::from(handle.generation()) << 32),
            }
        });
        if let Some(read) = frame
            .filter(|&f| Self::takes_deliveries(word, f))
            .and_then(|f| self.take_delivery(f, handle, kind))
        {
            return read;
        }
        let mut mode = 0;
        let mut _speculating = None;
        if let Some(f) = frame {
            // SAFETY: the active frame's control block is live.
            let ctl = unsafe { &*f.ctl };
            if ctl.has(cycle::DOOMED) {
                self.park(ParkPoint::DoomedRead(NodeId(handle.slot())));
                if let Some(value) = self.read_doomed(handle, f) {
                    return Ok(value);
                }
                // A node with a virtual frame stands for a frame below this
                // one and can't be computed in it. With no placeholder to
                // read, the doomed frame's run ends here: it is discarded
                // whatever it would go on to read.
                if word & value_code::REPLAY_BIT != 0
                    && !cycle_state::holds(self.runtime_id, handle.slot())
                    && self
                        .virtual_position(handle.slot(), handle.generation())
                        .is_some()
                {
                    if self.placeholder::<T>().is_some() && self.drop_cooperatively(Some(f.cidx)) {
                        return Ok(self.placeholder_read::<T>(frame));
                    }
                    self.drop_unwind();
                }
                // A doomed frame has read a placeholder, so the unbounded
                // run may never make this read: what it computes natively
                // is speculative, as an entry pushed here would be.
                _speculating = Some(replay::Speculating::enter());
            } else {
                if word & value_code::REPLAY_BIT != 0 {
                    if let Some(read) = self.judge_virtual(handle, f, kind) {
                        return read;
                    }
                }
                if self.over_budget(f.depth)
                    && !self.undoomable(ctl, f.depth)
                    && self.replayable(f.node.0)
                    && self.placeholder::<T>().is_some()
                {
                    mode |= ARMED;
                }
            }
        } else if self.dep_stack.other_top_doomed(self.runtime_id) {
            // Another runtime's doomed frame reads this one: its
            // placeholder may have led it to a read the unbounded run never
            // makes, as a doomed frame's own native read is.
            _speculating = Some(replay::Speculating::enter());
        }
        // A speculative run of a runtime with no context, read from another
        // runtime's doomed frame, runs in VIRT mode too, so that its
        // landings are withheld.
        if word & value_code::REPLAY_BIT != 0
            && (replay::speculative()
                || replay::with(|c| replay::innermost(c, self.runtime_id).is_some()))
        {
            mode |= VIRT;
        }
        let node = self.nodes.get(handle.slot());
        match mode {
            0 => self.read_walk::<T, 0>(handle, node, observed, kind, word, frame),
            ARMED => self.read_walk::<T, ARMED>(handle, node, observed, kind, word, frame),
            VIRT => self.read_walk::<T, VIRT>(handle, node, observed, kind, word, frame),
            _ => self.read_walk::<T, { ARMED | VIRT }>(handle, node, observed, kind, word, frame),
        }
    }

    /// Passes `low` to the parent of the frame at position `cidx`: the
    /// virtual frame below it when there is one (the frame is a driver's
    /// entry, or an episode a virtual frame's pop runs), else the active
    /// frame. Returns the parent's position and node, and whether it is a
    /// virtual frame.
    pub(super) fn lower_parent(&self, cidx: u32, low: u32) -> (u32, u32, Option<u64>) {
        let rt = self.runtime_id;
        if value_code::thread_word() & value_code::REPLAY_BIT != 0 && cidx > 0 {
            let below = cidx - 1;
            let lowered = replay::with(|c| {
                c.iter_mut()
                    .rev()
                    .filter(|x| x.rt == rt && below >= x.boundary)
                    .find_map(|x| x.vstack.get_mut((below - x.boundary) as usize))
                    .map(|v| {
                        v.low = v.low.min(low);
                        (v.node, v.path)
                    })
            });
            if let Some((node, path)) = lowered {
                return (below, node, Some(path));
            }
        }
        match self.dep_stack.active_frame(rt) {
            Some((parent, at)) => {
                // SAFETY: the active frame's control block is live.
                unsafe { &*parent }.lower(low);
                let node = self
                    .dep_stack
                    .active_compute(rt)
                    .map_or(u32::MAX, |(n, _)| n.0);
                (at, node, None)
            }
            None => (cidx.saturating_sub(1), u32::MAX, None),
        }
    }

    /// The highest virtual frame of `slot` at `generation` in a context of
    /// this runtime, as a stack position.
    pub(super) fn virtual_position(&self, slot: u32, generation: u32) -> Option<u32> {
        let rt = self.runtime_id;
        replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .filter_map(|x| x.position(slot, generation).map(|pos| x.boundary + pos))
                .max()
        })
    }

    /// The nodes of this runtime's virtual frames at stack position `p`
    /// or above, and the claims their runs had deferred at or above `p`.
    pub(super) fn virtual_slots_from(&self, p: u32) -> Vec<u32> {
        let rt = self.runtime_id;
        replay::with(|c| {
            let vframes = c
                .iter()
                .filter(|x| x.rt == rt)
                .flat_map(|x| x.vstack.iter());
            vframes
                .clone()
                .filter(|v| v.cidx >= p)
                .map(|v| v.node)
                .chain(
                    vframes
                        .flat_map(|v| v.deferred.iter())
                        .filter(|&&(_, anchor)| anchor >= p)
                        .map(|&(slot, _)| slot),
                )
                .collect()
        })
    }

    /// The claims of this runtime the frame at `cidx`, of `node`, at
    /// `path`, owns, as `(slot, anchor)`, which the virtual frame it
    /// becomes owns from now. A suspended episode's members belong to its
    /// member run's virtual frame, not to a frame that runs at its position
    /// meanwhile.
    fn owned_claims(&self, cidx: u32, node: u32, path: u64) -> Vec<(u32, u32)> {
        let rt = self.runtime_id;
        cycle_state::with_held(|h| {
            // Groups whose every member is taken move whole; a group with a
            // suspended member moves the others one by one.
            let (mut whole, mut singles) = (Vec::new(), Vec::new());
            for g in h.owners(rt, cidx, cidx) {
                let owner = h.owner(g);
                if !(owner.node() == node || owner.node() == u32::MAX)
                    || owner.virtual_frame().is_some_and(|p| p != path)
                {
                    continue;
                }
                let members = h.members(g);
                let start = singles.len();
                singles.extend(
                    members
                        .iter()
                        .copied()
                        .filter(|&m| !replay::is_suspended(h[&(rt, m)].ep)),
                );
                if singles.len() - start == members.len() {
                    singles.truncate(start);
                    whole.push(g);
                }
            }
            let mut owned = singles.clone();
            for &g in &whole {
                owned.extend_from_slice(h.members(g));
            }
            h.hand_over(rt, &whole, &singles, cidx, node, Some(path));
            owned
                .into_iter()
                .map(|m| (m, h[&(rt, m)].anchor()))
                .collect()
        })
    }

    /// The judgement of a read by frame `f`, not doomed, of a node that is
    /// not Clean: `None` unless the node has a virtual frame this thread
    /// does not hold. Such a read never computes the node, which stands
    /// for a frame below `f` in the unbounded run. A suspect edge between
    /// that frame and `f` drops the chain above it; otherwise the read is a
    /// virtual hit, which claims the virtual frames' nodes from there up
    /// and is served by the read rules as a read of a frame on the stack.
    fn judge_virtual<T: Value>(
        &self,
        handle: Incr<T>,
        f: ActiveRead,
        kind: ReadKind,
    ) -> Option<Result<T, Cycle>> {
        let rt = self.runtime_id;
        let slot = handle.slot();
        // A member run sits at its episode's anchor but reads from the top
        // of the stack, above every frame between.
        let top = f.cidx.max(self.dep_stack.next_cidx().wrapping_sub(1));
        let mut spins = 0;
        loop {
            let node = self.nodes.get(slot);
            match state::load::<C>(node.state_cell()) {
                // Another thread's memo landing: the read takes the memo.
                NodeState::Dirty if node.memo_at(|| self.current_revision()) => return None,
                NodeState::New
                | NodeState::Dirty
                | NodeState::Computing
                | NodeState::ComputingDirty => {}
                _ => return None,
            }
            // The handle's frame would never land if the slot now holds
            // another node; generation loads after the state.
            if self.nodes.get(slot).generation() != handle.generation() {
                self.stale_handle(handle);
            }
            // A doomed frame of the node is still on the stack, under a
            // native compute its run makes: the read is one of that frame.
            if cycle_state::holds(rt, slot) || self.dep_stack.frame_cidx(rt, NodeId(slot)).is_some()
            {
                return None;
            }
            let p = self.virtual_position(slot, handle.generation())?;
            // Before its loop runs, a context's virtual frames stand for
            // doomed frames that have popped, so they are on no stack, as
            // are those a doom in the running entry's run made at or above
            // its level. Only speculative runs under the doomed frames read
            // them.
            if !replay::with(|c| {
                c.iter().any(|x| {
                    x.rt == rt
                        && x.position(slot, handle.generation()).is_some_and(|q| {
                            x.boundary + q == p && x.running.is_some_and(|l| q < l)
                        })
                })
            }) {
                debug_assert!(
                    replay::speculative(),
                    "incr-core: only a speculative run reads before a loop runs"
                );
                return Some(Ok(self.drop_at_read::<T>(f)));
            }
            if let Some(q) = self
                .landed_above(p, top)
                .filter(|&q| !self.suspect_above(p, q))
            {
                self.drop_at_landed(q);
            }
            if self.suspect_above(p, top) {
                debug_assert!(
                    replay::speculative(),
                    "incr-core: a suspect edge lies under a speculative entry"
                );
                return Some(Ok(self.drop_at_read::<T>(f)));
            }
            self.park(ParkPoint::BeforeVirtualClaim(NodeId(slot)));
            match self.claim_virtual(p, top) {
                Ok(()) => {
                    self.unregenerable(Some(f));
                    return Some(
                        self.cycle_read(handle, NodeState::Computing, kind, None)
                            .expect("incr-core: a virtual hit holds the node it read"),
                    );
                }
                Err(on) => {
                    let generation = self.nodes.get(on.0).generation();
                    self.wait_out_reservation(on, generation, None, &mut spins);
                    loop {
                        spins += 1;
                        self.wait_on_claim(on, spins);
                        if !is_computing(state::load::<C>(self.nodes.get(on.0).state_cell())) {
                            break;
                        }
                    }
                }
            }
        }
    }

    /// The lowest virtual frame above position `p`, up to `top`, whose
    /// node another thread has landed (a memo counts), deleted or recycled
    /// since the frame was made: a landed point.
    fn landed_above(&self, p: u32, top: u32) -> Option<u32> {
        let rt = self.runtime_id;
        replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .flat_map(|x| x.stacked())
                .filter(|v| v.cidx > p && v.cidx <= top)
                .filter(|v| {
                    let node = self.nodes.get(v.node);
                    node.generation() != v.generation
                        || match state::load::<C>(node.state_cell()) {
                            NodeState::Clean
                            | NodeState::Cyclic
                            | NodeState::Failed
                            | NodeState::Dead => true,
                            NodeState::Dirty => node.memo_at(|| self.current_revision()),
                            _ => false,
                        }
                })
                .map(|v| v.cidx)
                .min()
        })
    }

    /// Drops the chain above landed point `q`: the frames above it stand on
    /// a run the other thread's landing has made moot. The context that
    /// holds `q`'s virtual frame records it, and the drop unwind ends the
    /// running entry's run, after which that context's driver removes the
    /// entries above `q`'s level and keeps the one whose run made it.
    fn drop_at_landed(&self, q: u32) -> ! {
        let rt = self.runtime_id;
        replay::with(|c| {
            let x = c
                .iter_mut()
                .rev()
                .find(|x| {
                    x.rt == rt && x.boundary <= q && x.stacked().len() > (q - x.boundary) as usize
                })
                .expect("incr-core: a landed point is a virtual frame");
            x.drop_at = Some(q - x.boundary);
        });
        self.drop_unwind()
    }

    /// Whether an edge into a frame above position `p`, up to `top`, is
    /// suspect: a virtual frame's, or a running entry's.
    fn suspect_above(&self, p: u32, top: u32) -> bool {
        let rt = self.runtime_id;
        replay::with(|c| {
            c.iter().filter(|x| x.rt == rt).any(|x| {
                x.stacked()
                    .iter()
                    .any(|v| v.cidx > p && v.cidx <= top && v.suspect)
                    || x.running.is_some_and(|level| {
                        let at = x.boundary + level;
                        at > p && at <= top && x.running_entry.is_some_and(|e| e.suspect)
                    })
            })
        })
    }

    /// A speculative run reached a group verify, whose member runs and
    /// landings would outlive its discard and change what the native retry
    /// finds: the claim lands back as it was and the chain is dropped, so
    /// that the retry verifies the group.
    #[cold]
    #[inline(never)]
    pub(super) fn drop_speculative_verify(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        claimed_dirty: bool,
    ) -> Computed {
        let target = if claimed_dirty {
            NodeState::Dirty
        } else {
            node.set_verified_at(0);
            NodeState::New
        };
        self.land_unrun(id, node, target);
        if self.drop_cooperatively(None) {
            return Computed::Discarded;
        }
        self.drop_unwind()
    }

    /// Raises the discard unwind that drops a speculative chain: every frame
    /// it passes takes the discard landing, and the driver's catch of the
    /// speculative entry drops the chain.
    pub(super) fn drop_unwind(&self) -> ! {
        self.set_drop_kind();
        resume_unwind(Box::new(cycle::DiscardUnwind))
    }

    /// Frame `f`'s read drops the speculative chain it stands on: the
    /// placeholder, if the drop is cooperative, else a drop unwind.
    fn drop_at_read<T: Value>(&self, f: ActiveRead) -> T {
        if self.placeholder::<T>().is_some() && self.drop_cooperatively(Some(f.cidx)) {
            return self.placeholder_read::<T>(Some(f));
        }
        self.drop_unwind()
    }

    /// Drops the running speculative entry's chain without an unwind, when
    /// every frame of the entry's run can take a placeholder for the read
    /// it is blocked in, as **Marking** tests, and the drop point lies in
    /// the entry's context. Those frames are doomed, so the read in progress
    /// in the top one, at `reader` when given, returns a placeholder, each
    /// frame pops discarded and the run comes back `Discarded` to the
    /// driver, which drops the chain from the point recorded here. `false`
    /// leaves the drop to an unwind.
    pub(super) fn drop_cooperatively(&self, reader: Option<u32>) -> bool {
        let rt = self.runtime_id;
        // The chain starts at the lowest level whose virtual frames below
        // are all real ones', as for a drop unwind's catch.
        let found = replay::with(|c| {
            let x = &c[replay::innermost(c, rt)?];
            let level = x
                .running
                .filter(|_| x.running_entry.is_some_and(|e| e.spec))?;
            let mut q = level;
            while q > 0 && x.vstack.get(q as usize - 1).is_some_and(|v| v.spec) {
                q -= 1;
            }
            (q > 0).then_some((x.id, q))
        });
        let Some((id, q)) = found else {
            return false;
        };
        // The run's frames are those from where the driver placed the entry
        // up, not those at or above its position: a member run in the run
        // sits at its anchor, which can lie below that position.
        let frames: Vec<(bool, u32, u32, *const ComputeCtl, u64, u32)> = self.dep_stack.with(|s| {
            let placed = s.placed as usize;
            s.frames
                .get(placed..)
                .expect("incr-core: an entry above level 0 runs placed")
                .iter()
                .enumerate()
                .rev()
                .map(|(i, f)| {
                    let depth = (placed + i + 1) as u32;
                    (f.rt() == rt, f.node.0, f.cidx, f.ctl, f.reading, depth)
                })
                .collect()
        });
        if reader.is_some_and(|r| frames.first().is_none_or(|f| f.2 != r)) {
            return false;
        }
        let doomable = frames.iter().all(|&(ours, node, _, ctl, reading, depth)| {
            // SAFETY: a frame's control block is live while it is on the
            // stack.
            let ctl = unsafe { &*ctl };
            ours && !ctl.has(cycle::BUFFERED)
                && !self.undoomable(ctl, depth)
                && self.replayable(node)
                && self.reading_has_placeholder(reading)
        });
        if !doomable {
            return false;
        }
        for &(_, _, _, ctl, _, _) in &frames {
            // SAFETY: as above.
            unsafe { &*ctl }.raise(cycle::DOOMED);
        }
        replay::with(|c| {
            if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                x.drop_point = Some(q);
            }
        });
        true
    }

    /// A speculative run's frame of `id` at `attempt` popped where it would
    /// land other than Clean, or root an episode, and dropped its chain
    /// cooperatively: it lands as a drop unwind would land it, and its
    /// reader gets a placeholder.
    #[cold]
    #[inline(never)]
    pub(super) fn discard_dropped(
        &self,
        id: NodeId,
        attempt: Attempt,
        ctl: &ComputeCtl,
        bag: &mut Deferred<C>,
    ) -> Computed {
        replay::unwind_past(self.runtime_id, attempt.cidx);
        self.release_unwound_from(
            attempt.cidx,
            id.0,
            Some(UnwindKind::Discard),
            true,
            None,
            bag,
        );
        if ctl.has(cycle::SPECULATIVE) {
            self.unbuild_speculative(ctl, bag);
        } else if let Some(value) = ctl.buf.take() {
            bag.push_erased(value);
        }
        let wrote = ctl.wrote.get() != 0;
        let kind = Some(UnwindKind::Discard);
        self.unwound_land(id, attempt.first, wrote, attempt.stamp_rev, kind, bag);
        Computed::Discarded
    }

    /// Sets the unwind kind of a drop unwind about to be raised.
    pub(super) fn set_drop_kind(&self) {
        #[cfg(test)]
        self.drop_unwinds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        cycle_state::set_unwind_kind(Some(UnwindKind::Discard));
    }

    /// A virtual hit's claims: the node of every virtual frame from
    /// position `p` up to `top` that this thread does not hold, as held
    /// claims anchored at `p`. On a claim another thread holds, the claims
    /// taken are rolled back with the discard landing and that node is
    /// returned for the read to wait on.
    fn claim_virtual(&self, p: u32, top: u32) -> Result<(), NodeId> {
        let rt = self.runtime_id;
        let mut frames: Vec<(u32, u32, u32)> = replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .flat_map(|x| x.stacked())
                .filter(|v| v.cidx >= p && v.cidx <= top)
                .map(|v| (v.cidx, v.node, v.generation))
                .collect()
        });
        frames.sort_unstable();
        let mut taken: Vec<(u32, bool)> = Vec::new();
        for (_, slot, generation) in frames {
            // A doomed frame still running holds its node's claim itself.
            if cycle_state::holds(rt, slot) || self.dep_stack.frame_cidx(rt, NodeId(slot)).is_some()
            {
                continue;
            }
            let (node, flags) = self.nodes.get_with_flags(slot);
            let claimed = state::try_claim_compute::<C>(node.state_cell());
            let first = match claimed {
                // Another thread's memo landing since the scan, or a node
                // another context holds reserved.
                Ok(src)
                    if node.generation() == generation
                        && (src == NodeState::Dirty
                            && node.memo_at(|| self.current_revision())
                            || self.reserved_elsewhere(NodeId(slot), generation)) =>
                {
                    self.unclaim_unrun(node, src == NodeState::Dirty);
                    self.roll_back_hit(&taken);
                    return Err(NodeId(slot));
                }
                Ok(src) if node.generation() == generation => src == NodeState::New,
                Ok(src) => {
                    self.land_unrun(NodeId(slot), node, src);
                    self.roll_back_hit(&taken);
                    return Err(NodeId(slot));
                }
                Err(_) => {
                    self.roll_back_hit(&taken);
                    return Err(NodeId(slot));
                }
            };
            let seeded = C::u8_load_relaxed(flags) & HAS_STRATEGY != 0;
            cycle_state::with_held(|h| {
                h.insert((rt, slot), Held::new(generation, p, seeded, first));
            });
            replay::with(|c| {
                if let Some(x) = c
                    .iter_mut()
                    .rev()
                    .find(|x| x.rt == rt && x.running_entry.is_some())
                {
                    let level = x.running.expect("a context with an entry is running");
                    if p < x.boundary + level {
                        x.claims.push((level, slot));
                    }
                }
            });
            taken.push((slot, first));
        }
        Ok(())
    }

    fn roll_back_hit(&self, taken: &[(u32, bool)]) {
        let rt = self.runtime_id;
        let stamp_rev = self.current_revision();
        let mut bag = Deferred::new();
        if !taken.is_empty() {
            self.count_release();
        }
        for &(slot, first) in taken {
            cycle_state::with_held(|h| h.remove(&(rt, slot)));
            let kind = Some(UnwindKind::Discard);
            self.unwound_land(NodeId(slot), first, false, stamp_rev, kind, &mut bag);
        }
        bag.drain(self);
    }

    /// A read made in a doomed frame: the placeholder, after pushing the
    /// node as an entry above the frame unless this thread computes,
    /// holds or virtually computes it, or the frame was blocked reading
    /// it. `None` for a type with no placeholder, which the read computes
    /// natively.
    fn read_doomed<T: Value>(&self, handle: Incr<T>, f: ActiveRead) -> Option<T> {
        let placeholder = self.placeholder::<T>()?;
        let rt = self.runtime_id;
        let slot = handle.slot();
        if state::load::<C>(self.nodes.get(slot).state_cell()) == NodeState::Clean {
            return None;
        }
        // SAFETY: the active frame's control block is live.
        let ctl = unsafe { &*f.ctl };
        let reached =
            self.dep_stack.frame_cidx(rt, NodeId(slot)).is_some() || cycle_state::holds(rt, slot);
        let suspect = ctl.has(cycle::PH_SEEN);
        replay::with(|c| {
            if reached
                || c.iter()
                    .any(|x| x.rt == rt && x.position(slot, handle.generation()).is_some())
            {
                return;
            }
            let at = replay::covering(c, rt, f.cidx)
                .expect("incr-core: a doomed frame lies in a replay context");
            let x = &mut c[at];
            // A dropped chain's entries would be removed with it.
            if x.drop_point.is_some() {
                return;
            }
            let frame = &x.vstack[(f.cidx - x.boundary) as usize];
            // The retry makes the read the frame was blocked in before this
            // one, and that read settles the node as the unbounded run's
            // did: an entry landing it first would leave a stashed walk
            // unresumed.
            if frame.reading == u64::from(slot) | (u64::from(handle.generation()) << 32) {
                return;
            }
            let spec = frame.spec;
            x.push_entry(
                f.cidx + 1 - x.boundary,
                Entry {
                    node: slot,
                    generation: handle.generation(),
                    suspect,
                    spec: spec || suspect,
                    retry: Retry::Walk,
                    walked: false,
                    r5: false,
                },
            );
        });
        ctl.raise(cycle::PH_SEEN);
        Some(placeholder)
    }

    /// The answer to a read whose walk was doomed or discarded: the reading
    /// frame is doomed and gets its type's placeholder.
    #[cold]
    #[inline(never)]
    pub(super) fn placeholder_read<T: Value>(&self, frame: Option<ActiveRead>) -> T {
        let f = frame.expect("incr-core: only a frame's read is answered by a placeholder");
        // SAFETY: the active frame's control block is live.
        let ctl = unsafe { &*f.ctl };
        debug_assert!(
            ctl.has(cycle::DOOMED),
            "incr-core: a placeholder read dooms its frame"
        );
        ctl.raise(cycle::PH_SEEN);
        self.placeholder::<T>()
            .expect("incr-core: a doomed read's type has a placeholder")
    }

    /// The doom of the read that wanted `id`, which this thread claimed
    /// (from Dirty if `claimed_dirty`, else from New) and whose closure the
    /// read would run past the budget: the claim is released with the
    /// discard landing, keeping a group member's record, the frames below
    /// that can take a placeholder are marked, and `id` becomes an entry.
    #[cold]
    #[inline(never)]
    pub(super) fn doom(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        generation: u32,
        claimed_dirty: bool,
        direct: bool,
        walked: bool,
    ) -> Computed {
        let target = if claimed_dirty {
            NodeState::Dirty
        } else {
            node.set_verified_at(0);
            NodeState::New
        };
        // The context the doom marks computes `id` again; it may not be
        // open yet.
        let reserving = (!C::SINGLE_THREADED).then(|| reserve::Scope::new(Some(reserve::PENDING)));
        self.land_unrun(id, node, target);
        drop(reserving);
        #[cfg(test)]
        self.dooms.lock().unwrap().push(DoomSeen {
            node: id.0,
            state: node.state(),
            verified_at: node.verified_at(),
            depth: self.dep_stack.depth(),
            context: 0,
        });
        self.mark(
            id,
            generation,
            if direct {
                Retry::Direct
            } else {
                Retry::Compute
            },
            walked,
            false,
        );
        if let Some(context) = self.reserving_context() {
            reserve::adopt(context);
        }
        self.park(ParkPoint::AfterDoom(id));
        Computed::Doomed
    }

    /// The doom of an R5 read of `handle`, a member this thread holds
    /// without a buffer, from frame `f` past the budget: the member keeps
    /// its claim, the frames below that can take a placeholder are marked,
    /// and the member becomes an entry, which the driver runs as a member
    /// run (**Loop** step 4).
    #[cold]
    #[inline(never)]
    pub(super) fn doom_held<T: Value>(&self, handle: Incr<T>, f: ActiveRead) -> T {
        let id = NodeId(handle.slot());
        #[cfg(test)]
        self.dooms.lock().unwrap().push(DoomSeen {
            node: id.0,
            state: self.nodes.get(id.0).state(),
            verified_at: self.nodes.get(id.0).verified_at(),
            depth: self.dep_stack.depth(),
            context: 0,
        });
        self.mark(id, handle.generation(), Retry::Compute, false, true);
        self.park(ParkPoint::AfterDoom(id));
        self.placeholder_read::<T>(Some(f))
    }

    /// A clone of node `slot`'s value, if its slot holds one.
    fn slot_value(&self, slot: u32) -> Option<cycle::Erased> {
        let node = self.nodes.get(slot);
        self.arenas
            .erased(node.type_tag())
            .read_erased(node.arena_slot())
    }

    /// The earlier reads of the frames being marked that returned a value
    /// a landing had left Dirty, or took an unwind. A retry's read gets
    /// that outcome again, as the frame's first run did, where a fresh read
    /// would walk the node and run it again, or read what a later landing
    /// replaced it with.
    fn kept_reads(&self, marked: &[Below], vframes: &[VFrame]) -> Vec<(ReadKey, Delivered)> {
        let rt = self.runtime_id;
        let mut out = Vec::new();
        for f in marked {
            // SAFETY: a frame's control block is live while it is on the
            // stack, and these frames are being marked.
            if !unsafe { &*f.ctl }.has(cycle::KEPT_READ) {
                continue;
            }
            let Some(v) = vframes.iter().find(|v| v.cidx == f.cidx) else {
                continue;
            };
            for read in replay::take_kept_reads(rt, f.cidx, f.ctl) {
                let key = ReadKey {
                    reading: read.reading,
                    ordinal: read.ordinal,
                    foreign: read.foreign,
                    ..ReadKey::blocked(v)
                };
                out.push((key, read.what));
            }
        }
        out
    }

    /// A read by frame `f` returned `value` from a node a landing left
    /// Dirty. If a budget is set, the frame may be doomed and retried, and
    /// the retry's read must get this value again.
    #[cold]
    #[inline(never)]
    pub(super) fn note_dirty_read<T: Value>(&self, f: ActiveRead, handle: Incr<T>, value: &T) {
        if !self.budgeted() {
            return;
        }
        // SAFETY: the reading frame is on the stack, so its control block
        // is live.
        unsafe { &*f.ctl }.raise(cycle::UNREGENERABLE);
        let value: cycle::Erased = Box::new(value_code::run(|| value.clone()));
        let value = Unbuilt::new(value);
        let reading = u64::from(handle.slot()) | (u64::from(handle.generation()) << 32);
        let ordinal = self.dep_stack.reads_of_current(self.runtime_id);
        let poison = self.dirty_read_poison(self.nodes.get(handle.slot()));
        self.keep_read(
            f.ctl,
            f.cidx,
            reading,
            ordinal,
            Delivered::Landed(value, poison),
        );
    }

    /// The active frame's read took `payload`, a user panic or a cycle
    /// unwinding out of the compute it ran. If a budget is set, the frame
    /// may be doomed and retried, and the retry's read must take the
    /// unwind again: the unbounded run's frame read the node once, and a
    /// fresh read would run what the unwind left New, Dirty or Failed
    /// again, or panic as Failed with another payload. A payload no copy
    /// can stand for holds the frame instead (`hold_uncopied`).
    #[cold]
    #[inline(never)]
    pub(super) fn note_unwound_read(
        &self,
        payload: &(dyn std::any::Any + Send),
        kind: Option<UnwindKind>,
    ) {
        if !matches!(kind, None | Some(UnwindKind::Cycle))
            || replay::speculative()
            || !self.budgeted()
        {
            return;
        }
        let Some((f, reading, ordinal)) = self.dep_stack.unwound_read(self.runtime_id) else {
            return;
        };
        if kind == Some(UnwindKind::Cycle) {
            self.unregenerable(Some(f));
        }
        let (ctl, cidx) = (f.ctl, f.cidx);
        let regenerable = self.note_took(f, reading, payload);
        let copy = match replay::copy_payload(payload) {
            Some(copy) => copy,
            None if self.hold_uncopied(ctl, regenerable) => return,
            None => Box::new(replay::Regen(payload.type_id())),
        };
        let what = Delivered::Unwind {
            cycle: kind == Some(UnwindKind::Cycle),
            poison: self.unwound_poison(kind),
            attributed: false,
            payload: copy,
            nodes: self.copy_notes(payload, cidx + 1),
        };
        self.keep_read(ctl, cidx, reading, ordinal, what);
    }

    /// The active frame's read of `id` took `payload` from a walk's
    /// ticket. If a budget is set, the frame may be doomed after its closure
    /// has dropped the one-shot payload, and its retry's read must take it
    /// again, as the unbounded run's one read did: a copy is kept for the
    /// retry, or a payload no copy can stand for holds the frame
    /// (`hold_uncopied`).
    #[cold]
    #[inline(never)]
    pub(super) fn note_ticket_read(&self, id: NodeId, payload: &(dyn std::any::Any + Send)) {
        if !self.budgeted() {
            return;
        }
        let Some((f, reading, ordinal)) = self.dep_stack.unwound_read(self.runtime_id) else {
            return;
        };
        if reading != u64::from(id.0) | (u64::from(self.nodes.get(id.0).generation()) << 32) {
            return;
        }
        let (ctl, cidx) = (f.ctl, f.cidx);
        let regenerable = self.note_took(f, reading, payload);
        let copy = match replay::copy_payload(payload) {
            Some(copy) => copy,
            None if self.hold_uncopied(ctl, regenerable) => return,
            None => Box::new(replay::Regen(payload.type_id())),
        };
        let what = Delivered::Unwind {
            cycle: false,
            poison: cycle::HANDLED,
            attributed: false,
            payload: copy,
            nodes: self.copy_notes(payload, cidx + 1),
        };
        self.keep_read(ctl, cidx, reading, ordinal, what);
    }

    /// The reading frame with control block `ctl` took a payload no copy
    /// can stand for, which its closure may drop: it is not doomed while it
    /// is within the budget, so its one read is the unbounded run's. Past
    /// the budget a retry regenerates the payload (`regenerate`) if it is
    /// `regenerable`, so the stack bound holds; else the frame is held there
    /// too, since rerunning a closure not declared replayable is a run the
    /// unbounded run never makes. True if the frame is held.
    pub(super) fn hold_uncopied(&self, ctl: *const cycle::ComputeCtl, regenerable: bool) -> bool {
        if self.over_budget(self.dep_stack.depth() as u32) {
            if regenerable {
                return false;
            }
            #[cfg(test)]
            PAST_BUDGET_HOLDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            #[cfg(all(test, not(loom)))]
            THREAD_PAST_BUDGET_HOLDS.with(|n| n.set(n.get() + 1));
        }
        // SAFETY: the reading frame is on the stack, so its control block
        // is live.
        let flags = unsafe { &*ctl };
        if !flags.has(cycle::DOOMED) {
            flags.raise(cycle::NO_DOOM);
        }
        true
    }

    /// Marks frame `frame`'s run as one a rerun of its node does not
    /// repeat (`cycle::UNREGENERABLE`).
    pub(super) fn unregenerable(&self, frame: Option<ActiveRead>) {
        if let Some(f) = frame {
            // SAFETY: the reading frame is on the stack, so its control
            // block is live.
            unsafe { &*f.ctl }.raise(cycle::UNREGENERABLE);
        }
    }

    /// Notes for `regenerate` that frame `f`'s read of node `reading`
    /// (`slot | generation << 32`) took `payload` from a compute or a
    /// ticket, as the frame's run's read. Returns whether a retry can
    /// regenerate the payload (`regenerable`).
    pub(super) fn note_took(
        &self,
        f: ActiveRead,
        reading: u64,
        payload: &(dyn std::any::Any + Send),
    ) -> bool {
        // SAFETY: the reading frame is on the stack, so its control block
        // is live.
        let member = unsafe { &*f.ctl }.has(cycle::BUFFERED);
        let (generation, path, _, seq) = self.key_parts(f, member);
        let reader = (f.node.0, generation, path, seq);
        replay::note_took(self.runtime_id, reader, reading, payload);
        self.regenerable(reading)
    }

    /// Notes for `regenerate` that `id` at `generation`, its frame at `cidx`
    /// just popped, lands Failed with what its run's reads took, keyed as
    /// `note_took` keyed them. A run that is not `regenerable`, as a member
    /// landing after its run or a creator's run, which no rerun repeats,
    /// leaves no record, so its readers are held past the budget.
    pub(super) fn note_failed_run(
        &self,
        id: NodeId,
        generation: u32,
        cidx: u32,
        regenerable: bool,
    ) {
        if !self.budgeted() || replay::speculative() {
            return;
        }
        let reading = u64::from(id.0) | (u64::from(generation) << 32);
        let took = self.took_by_run(id, generation, cidx);
        if !regenerable || !self.replayable(id.0) {
            self.forget_failed_run(id.0, generation);
            return;
        }
        let ok = {
            let regens = self.regens.read();
            took.iter()
                .all(|(n, copy)| copy.is_some() || regens.by_node.get(n).is_some_and(|r| r.ok))
        };
        let took = took
            .into_iter()
            .map(|(n, copy)| (n, copy.map(std::sync::Mutex::new)))
            .collect();
        self.regens_held
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let mut regens = self.regens.write();
        regens.next_token += 1;
        let token = regens.next_token;
        regens
            .by_node
            .insert(reading, Regenerable { token, ok, took });
    }

    /// `id` at `generation`, its frame at `cidx` just popped, lands but not
    /// Failed: its run's takes and an earlier Failed landing's record go.
    pub(super) fn unfailed_landing(&self, id: NodeId, generation: u32, cidx: u32) {
        if !self.budgeted() || replay::speculative() {
            return;
        }
        drop(self.took_by_run(id, generation, cidx));
        self.forget_failed_run(id.0, generation);
    }

    /// `id` at `generation`, its frame at `cidx` just popped, lands a
    /// computed value: its run's takes go. Its earlier Failed record went
    /// when it left Failed (`flip_failed_from`), and one another thread's
    /// run lands once the claim is released must stay.
    pub(super) fn drop_takes(&self, id: NodeId, generation: u32, cidx: u32) {
        if !self.budgeted() || replay::speculative() {
            return;
        }
        drop(self.took_by_run(id, generation, cidx));
    }

    /// What the run of `id` at `generation`, whose frame at `cidx` just
    /// popped, noted its reads took (`replay::note_took`).
    fn took_by_run(&self, id: NodeId, generation: u32, cidx: u32) -> Vec<replay::Took> {
        let rt = self.runtime_id;
        if !replay::takes_noted(rt) {
            return Vec::new();
        }
        let reading = u64::from(id.0) | (u64::from(generation) << 32);
        let depth = self.dep_stack.depth() as u32;
        let (seq, path) =
            replay::ended_run(rt, depth, reading).unwrap_or_else(|| (0, self.path_at(cidx)));
        replay::took_by_run(rt, (id.0, generation, path, seq))
    }

    /// Whether a retry can regenerate the payload node `reading` landed
    /// Failed with: the node may run again, and so may every node its
    /// regeneration runs.
    pub(super) fn regenerable(&self, reading: u64) -> bool {
        self.regens
            .read()
            .by_node
            .get(&reading)
            .is_some_and(|r| r.ok)
    }

    /// What the reads of the run node `reading` landed Failed from took, a
    /// copy of each payload or `None` where its node runs again, if a
    /// budgeted run landed it.
    fn took_by(&self, reading: u64) -> Option<Vec<replay::Took>> {
        let regens = self.regens.read();
        let r = regens.by_node.get(&reading)?;
        Some(
            r.took
                .iter()
                .map(|(n, copy)| {
                    let copy = copy.as_ref().and_then(|c| {
                        replay::copy_payload(
                            &**c.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
                        )
                    });
                    (*n, copy)
                })
                .collect(),
        )
    }

    /// Drops the record of node `slot` at `generation`'s Failed landing,
    /// which it leaves.
    pub(super) fn forget_failed_run(&self, slot: u32, generation: u32) {
        if !self.regens_held.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let reading = u64::from(slot) | (u64::from(generation) << 32);
        self.regens.write().by_node.remove(&reading);
    }

    /// The token of node `slot` at `generation`'s record, to forget it by
    /// once the node's claim or Failed state no longer keeps another run
    /// from landing a new one (`forget_failed_run_if`).
    pub(super) fn failed_run_token(&self, slot: u32, generation: u32) -> Option<u64> {
        if !self.regens_held.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let reading = u64::from(slot) | (u64::from(generation) << 32);
        self.regens.read().by_node.get(&reading).map(|r| r.token)
    }

    /// Runs `flip` with the token of node `slot` at `generation`'s record,
    /// under the table's lock: a run writes or forgets its record before its
    /// landing publishes Failed, so the token is the newest Failed landing's
    /// record, if that landing wrote one.
    pub(super) fn with_failed_run_token<R>(
        &self,
        slot: u32,
        generation: u32,
        flip: impl FnOnce(Option<u64>) -> R,
    ) -> R {
        let reading = u64::from(slot) | (u64::from(generation) << 32);
        let regens = self.regens.read();
        flip(regens.by_node.get(&reading).map(|r| r.token))
    }

    /// Forgets node `slot` at `generation`'s record if it is still the one
    /// `token` names.
    pub(super) fn forget_failed_run_if(&self, slot: u32, generation: u32, token: Option<u64>) {
        let Some(token) = token else {
            return;
        };
        let reading = u64::from(slot) | (u64::from(generation) << 32);
        let mut regens = self.regens.write();
        if regens
            .by_node
            .get(&reading)
            .is_some_and(|r| r.token == token)
        {
            regens.by_node.remove(&reading);
        }
    }

    /// The payload a retry's read of Failed node `id` at `generation` raises
    /// where the unbounded run's read took a payload of type `ty` no copy
    /// can stand for: the node's closure runs again, as a replayed closure
    /// may, and a panic of that type is the payload. Each read its run took
    /// a payload from a compute or a ticket raises that payload again, a
    /// copy or one regenerated so before the run, so one run is on the
    /// stack at a time however deep the panics nest; any other read of a
    /// Failed node found it Failed already and panics with its message.
    /// `None` if a run aborts, returns, or panics with another type at the
    /// top: a set since may have changed what the runs read, so the read
    /// goes on as a read of the node as it is now, which a read racing the
    /// set may be.
    #[cold]
    #[inline(never)]
    pub(super) fn regenerate(
        &self,
        id: NodeId,
        generation: u32,
        ty: std::any::TypeId,
    ) -> Option<Box<dyn std::any::Any + Send>> {
        let top = u64::from(id.0) | (u64::from(generation) << 32);
        let mut payload = None;
        // Each level: its node, the reads its run took a payload from
        // still to make, and the payloads made for them.
        type Level = (
            u64,
            Vec<(u64, Option<Box<dyn std::any::Any + Send>>)>,
            Vec<(u64, Box<dyn std::any::Any + Send>)>,
        );
        let mut stack: Vec<Level> = Vec::new();
        // A node's last noted run can be newer than the run that read it,
        // so the reads need not nest.
        let mut on_stack: HashSet<u64> = HashSet::default();
        if let Some(took) = self.took_by(top) {
            stack.push((top, took, Vec::new()));
            on_stack.insert(top);
        }
        while let Some((at, took, made)) = stack.last_mut() {
            if let Some((n, copy)) = took.pop() {
                match copy {
                    Some(copy) => made.push((n, copy)),
                    None => match self.took_by(n) {
                        Some(took) if on_stack.insert(n) => stack.push((n, took, Vec::new())),
                        _ => break,
                    },
                }
                continue;
            }
            let at = *at;
            let made = std::mem::take(made);
            stack.pop();
            on_stack.remove(&at);
            let Some(p) = self.rerun(NodeId(at as u32), (at >> 32) as u32, made) else {
                break;
            };
            match stack.last_mut() {
                Some(level) => level.2.push((at, p)),
                None => payload = Some(p),
            }
        }
        for (_, took, made) in stack {
            drop_payloads(took.into_iter().filter_map(|t| t.1));
            drop_payloads(made.into_iter().map(|m| m.1));
        }
        match payload {
            Some(p) if (*p).type_id() == ty => {
                #[cfg(all(test, not(loom)))]
                THREAD_REGENERATIONS.with(|n| n.set(n.get() + 1));
                Some(p)
            }
            p => {
                drop_payloads(p);
                None
            }
        }
    }

    /// `payload`, taken by a read of `handle` under `key` from `rt`'s
    /// contexts, or for its Regen marker the payload regenerated, which the
    /// delivery then counts as raised: `None` if it can't be (`regenerate`).
    fn regenerated<T: Value>(
        &self,
        handle: Incr<T>,
        rt: RuntimeId,
        key: &ReadKey,
        payload: Box<dyn std::any::Any + Send>,
    ) -> Option<Box<dyn std::any::Any + Send>> {
        match payload.downcast::<replay::Regen>() {
            Ok(regen) => {
                let raised =
                    self.regenerate(NodeId(handle.slot()), handle.generation(), regen.0)?;
                replay::regenerated(rt, key, replay::payload_key(&*raised));
                Some(raised)
            }
            Err(payload) => Some(payload),
        }
    }

    /// One run of `regenerate`: node `id`'s closure, if the node is still
    /// Failed at `generation` and declared replayable, so its closure may
    /// run again, with its reads of Failed nodes taking from `made`. Returns
    /// the closure's panic. Nothing lands: the node is claimed from Failed
    /// for the run and goes back to it, the run's deps are dropped, and a
    /// read that would compute, wait or create aborts it. It runs outside
    /// `compute_attempts`, so no Native run limit counts it.
    ///
    /// No dep of a Failed node has changed since its run, so a closure that
    /// reads only the deps that run recorded reads what it read and panics
    /// as it did, unless it hides state the contract forbids. A run that
    /// reads any other node took another path, and makes no payload.
    fn rerun(
        &self,
        id: NodeId,
        generation: u32,
        made: Vec<(u64, Box<dyn std::any::Any + Send>)>,
    ) -> Option<Box<dyn std::any::Any + Send>> {
        let node = self.nodes.get(id.0);
        if !self.replayable(id.0) || node.generation() != generation {
            drop_payloads(made.into_iter().map(|m| m.1));
            return None;
        }
        // Another thread's rerun of the node holds it only for that run,
        // which never waits, so this waits it out. The claim is noted first,
        // so a rerun that finds the node Computing sees it.
        replay::regen_claim(self.runtime_id, id.0, true);
        let mut spins = 0;
        loop {
            // A ticket keeps the node while it reruns (`in_failed_landing`).
            let t = self.tickets.read();
            let moved = state::try_transition::<C>(
                node.state_cell(),
                NodeState::Failed,
                NodeState::Computing,
            );
            if moved.is_ok() {
                replay::regen_running(self.runtime_id, id.0, true);
            }
            drop(t);
            match moved {
                Ok(()) => break,
                Err(NodeState::Computing | NodeState::ComputingDirty)
                    if replay::regen_claims(self.runtime_id, id.0) > 1 =>
                {
                    spins += 1;
                    C::backoff(spins);
                }
                Err(_) => {
                    replay::regen_claim(self.runtime_id, id.0, false);
                    drop_payloads(made.into_iter().map(|m| m.1));
                    return None;
                }
            }
        }
        let unmark = Unmark(self.runtime_id, id.0);
        // SAFETY: as in `compute_attempts`: the closure is freed only under
        // a Dead claim, which can't succeed while this node is Computing.
        let compute = self
            .inner
            .read()
            .compute_fns
            .get(id.0 as usize)
            .and_then(|f| f.as_deref())
            .map(std::ptr::from_ref);
        let mut panicked = None;
        let mut left = made;
        let mut engine = None;
        if let Some(compute) = compute {
            let compute = unsafe { &*compute };
            let ctl = ComputeCtl::new(cycle::BUFFERED);
            self.dep_stack.push_frame(self.runtime_id, id, &ctl);
            replay::regen_open(self.runtime_id, left);
            value_code::set_bits(value_code::REGEN_BIT, true);
            let ran = catch_unwind(AssertUnwindSafe(|| compute(self, id.0, &ctl)));
            let aborted = value_code::thread_word() & value_code::REGEN_ABORTED != 0;
            value_code::set_bits(value_code::REGEN_BIT | value_code::REGEN_ABORTED, false);
            left = replay::regen_close();
            let read = self.dep_stack.pop_frame();
            let value = ctl.buf.take();
            let mut recorded = Vec::new();
            node.for_each_dep(|d| recorded.push(d.0));
            recorded.sort_unstable();
            let same_path = read
                .iter()
                .all(|(d, _)| recorded.binary_search(&d.0).is_ok());
            match ran {
                Err(p) if cycle_state::unwind_kind().is_some() => engine = Some(p),
                Err(p) if !aborted && same_path => panicked = Some(p),
                ran => drop_payloads(ran.err()),
            }
            drop(catch_unwind(AssertUnwindSafe(move || drop(value))));
        }
        let mut flipped = true;
        let t = self.tickets.read();
        let back =
            state::try_transition::<C>(node.state_cell(), NodeState::Computing, NodeState::Failed);
        if back.is_ok() {
            replay::regen_running(self.runtime_id, id.0, false);
        }
        drop(t);
        if back.is_ok() {
            self.park(ParkPoint::RerunBack(id));
        }
        if back.is_err() {
            // A set's walk found the node Computing and walked on through
            // it: the node leaves Failed as the walk would have flipped it,
            // in one step, so no read sees it Failed after the set.
            flipped = self.flip_failed_from(id, NodeState::ComputingDirty);
        }
        // The move back or the flip cleared the mark with the state: a
        // later clear could take the mark of the node's next rerun.
        std::mem::forget(unmark);
        replay::regen_claim(self.runtime_id, id.0, false);
        debug_assert!(flipped, "incr-core: a regenerating node left Computing");
        drop_payloads(left.into_iter().map(|m| m.1));
        // Raised by a wait, which a regenerating run never makes.
        if let Some(p) = engine {
            drop_payloads(panicked);
            resume_unwind(p);
        }
        panicked
    }

    /// A slow read made by a run of `regenerate`: the first read of a node
    /// the unbounded run's read took a payload from raises the payload made
    /// for it, whatever the node's state now; another read of a Failed node
    /// panics with its message; any other read would compute, wait or land,
    /// so it aborts the run.
    #[cold]
    #[inline(never)]
    pub(super) fn regen_read(&self, id: NodeId, generation: u32, observed: NodeState) -> ! {
        let key = u64::from(id.0) | (u64::from(generation) << 32);
        if let Some(payload) = replay::regen_take(self.runtime_id, key) {
            resume_unwind(payload);
        }
        if observed == NodeState::Failed {
            panic!("{}", super::provenance_impl::failed_read_message(id.0));
        }
        abort_regen()
    }

    /// The key parts a read by this runtime's frame `f` gets from this
    /// runtime: the node's generation, the frame's path, for a member run
    /// its episode, and its run's seq, looked up as [`Self::path_at`] and
    /// the virtual frames look it up, so a retry's key matches the
    /// deliveries made for it.
    fn key_parts(&self, f: ActiveRead, member: bool) -> replay::KeyParts {
        let rt = self.runtime_id;
        let node = f.node.0;
        let episode = member
            .then(|| cycle_state::held_episode(rt, node))
            .flatten();
        let (seq, path) = match replay::run_at(rt, f.ctl, f.depth) {
            Some((seq, path)) => {
                debug_assert_eq!(path, self.path_at(f.cidx), "the run's path at {}", f.cidx);
                (seq, path)
            }
            None => (0, self.path_at(f.cidx)),
        };
        (self.nodes.get(node).generation(), path, episode, seq)
    }

    fn frame_keys(&self) -> replay::FrameKeys {
        replay::FrameKeys {
            owner: std::ptr::from_ref(self).cast(),
            parts: Self::foreign_key_parts,
        }
    }

    /// [`Self::key_parts`] for a read another runtime makes from one of
    /// this runtime's frames.
    ///
    /// # Safety
    ///
    /// `owner` is a context's [`replay::FrameKeys::owner`], this runtime,
    /// and the context is open.
    unsafe fn foreign_key_parts(
        owner: *const (),
        node: u32,
        cidx: u32,
        member: bool,
    ) -> replay::KeyParts {
        // SAFETY: the caller's contract; the driver that opened the context
        // holds the runtime until it closes it.
        let rt = unsafe { &*owner.cast::<Self>() };
        let top = rt.dep_stack.top_frame();
        debug_assert!(top.is_some_and(|(r, n, _, c)| (r, n.0, c) == (rt.runtime_id, node, cidx)));
        let (_, node, ctl, cidx) = top.expect("incr-core: the reading frame is on the stack");
        let depth = rt.dep_stack.depth() as u32;
        rt.key_parts(
            ActiveRead {
                node,
                ctl,
                cidx,
                depth,
            },
            member,
        )
    }

    /// A read of `handle` by another runtime's frame, which may be doomed
    /// and retried. A second read of this runtime need not return what the
    /// first did (a cycle's outcome depends on where a walk enters it), so
    /// the retry takes the first read's outcome again, kept in the reader's
    /// runtime. A panic no copy can stand for keeps the reader from being
    /// doomed instead.
    #[cold]
    #[inline(never)]
    pub(super) fn foreign_read<T: Value>(
        &self,
        handle: Incr<T>,
        observed: NodeState,
        kind: ReadKind,
        word: u64,
    ) -> Result<T, Cycle> {
        let Some((reader, node, ctl, cidx)) = self.dep_stack.top_frame() else {
            return self.read_unframed(handle, observed, kind, word);
        };
        // SAFETY: the reading frame is on the stack, so its control block
        // is live.
        let flags = unsafe { &*ctl };
        // A doomed frame's reads are not the unbounded run's, and an
        // unbudgeted runtime's frame is never retried.
        if reader == self.runtime_id || flags.has(cycle::DOOMED) || unbudgeted(reader) {
            return self.read_unframed(handle, observed, kind, word);
        }
        // A rerun of the reader's node reads this runtime's node again, as
        // it is then.
        flags.raise(cycle::UNREGENERABLE);
        let rt = self.runtime_id;
        let reading = u64::from(handle.slot()) | (u64::from(handle.generation()) << 32);
        let kept = flags.has(cycle::KEPT_READ);
        let ordinal = replay::foreign_ordinal(reader, cidx, ctl, reading, rt, kept);
        let key = self.foreign_key(word, (reader, node, flags, cidx), reading, ordinal);
        let taken = key
            .as_ref()
            .and_then(|key| replay::take_pending(reader, key, clone_landed::<T>));
        let outcome = match taken {
            Some(Delivered::Landed(value, _)) => Ok(Ok(*value
                .into_inner()
                .downcast::<T>()
                .expect("incr-core: a landed value has its node's type"))),
            Some(Delivered::Unwind {
                cycle: true,
                payload,
                ..
            }) => Ok(Err(payload
                .downcast::<cycle::CycleUnwind>()
                .expect("incr-core: a kept cycle is a cycle unwind")
                .0)),
            Some(Delivered::Unwind { payload, .. }) => {
                let raised = match &key {
                    Some(key) => self.regenerated(handle, reader, key, payload),
                    None => Some(payload),
                };
                match raised {
                    Some(payload) => Err(payload),
                    None => catch_unwind(AssertUnwindSafe(|| {
                        self.read_unframed(handle, observed, kind, word)
                    })),
                }
            }
            Some(Delivered::Walk(..) | Delivered::Direct) => {
                unreachable!("incr-core: a walk's direction is not a value")
            }
            None => catch_unwind(AssertUnwindSafe(|| {
                self.read_unframed(handle, observed, kind, word)
            })),
        };
        let what = match &outcome {
            Ok(Ok(value)) => Some(Delivered::Landed(
                Unbuilt::new(clone_landed::<T>(value)),
                cycle::POISONED,
            )),
            Ok(Err(c)) => Some(Delivered::Unwind {
                cycle: true,
                poison: 0,
                attributed: false,
                payload: Box::new(cycle::CycleUnwind(c.clone())),
                nodes: replay::Landings::default(),
            }),
            // A cycle `get` raised, which the reader may catch.
            Err(payload) if cycle_state::unwind_kind() == Some(UnwindKind::Cycle) => payload
                .downcast_ref::<cycle::CycleUnwind>()
                .map(|c| Delivered::Unwind {
                    cycle: true,
                    poison: 0,
                    attributed: false,
                    payload: Box::new(cycle::CycleUnwind(c.0.clone())),
                    nodes: replay::Landings::default(),
                }),
            // A wound or a discard leaves the reader's run too.
            Err(_) if cycle_state::unwind_kind().is_some() => None,
            Err(payload) => {
                let copy = replay::copy_payload(&**payload);
                if copy.is_none() {
                    self.dep_stack.hold_foreign_reader();
                }
                copy.map(|payload| Delivered::Unwind {
                    cycle: false,
                    poison: 0,
                    attributed: false,
                    payload,
                    nodes: replay::Landings::default(),
                })
            }
        };
        if let Some(what) = what {
            flags.raise(cycle::KEPT_READ);
            replay::keep_read(
                replay::KeptRead {
                    rt: reader,
                    cidx,
                    ctl,
                    reading,
                    ordinal,
                    foreign: Some(rt),
                    what,
                },
                !kept,
            );
        }
        match outcome {
            Ok(read) => read,
            Err(payload) => resume_unwind(payload),
        }
    }

    /// A Clean read of `handle` by another runtime's frame (see
    /// [`Self::foreign_read`]) with no replay context open, so there is no
    /// kept outcome to take: the frame keeps a clone of the value. A slot
    /// read runs no compute, so there is no unwind to keep.
    #[cold]
    #[inline(never)]
    pub(super) fn foreign_clean_read<T: Value>(
        &self,
        handle: Incr<T>,
        node: &NodeData<C>,
        (reader, ctl, cidx): (RuntimeId, *const cycle::ComputeCtl, u32),
    ) -> Result<T, NodeState> {
        let (type_tag, arena_slot) = self.fields_under(handle, node);
        // Under `Shared` a landing can clear the slot after the Clean load;
        // the slow path reads such a node again.
        let value = self
            .try_read_slot(handle, node, type_tag, arena_slot)
            .ok_or(NodeState::Clean)?;
        // SAFETY: the reading frame is on the stack, so its control block
        // is live.
        let flags = unsafe { &*ctl };
        // A rerun of the reader's node reads this runtime's node again, as
        // it is then.
        flags.raise(cycle::UNREGENERABLE);
        // The frame's tag was set under a budget, so unlike `foreign_read`
        // this does not look the reader up: a frame that runs across a
        // budget change is `NO_DOOM`, and what it keeps goes at its pop.
        if flags.has(cycle::DOOMED) {
            return Ok(value);
        }
        let read = replay::KeptRead {
            rt: reader,
            cidx,
            ctl,
            reading: u64::from(handle.slot()) | (u64::from(handle.generation()) << 32),
            ordinal: 0,
            foreign: Some(self.runtime_id),
            what: Delivered::Landed(Unbuilt::new(clone_landed::<T>(&value)), cycle::POISONED),
        };
        let kept = flags.has(cycle::KEPT_READ);
        flags.raise(cycle::KEPT_READ);
        replay::keep_foreign_read(read, !kept);
        Ok(value)
    }

    /// The key of the read of this runtime's node `reading` that `frame`,
    /// another runtime's, makes for the `ordinal`th time, when the read is
    /// a retry's in an open context of that runtime.
    fn foreign_key(
        &self,
        word: u64,
        frame: (RuntimeId, NodeId, &cycle::ComputeCtl, u32),
        reading: u64,
        ordinal: u32,
    ) -> Option<ReadKey> {
        let (reader, node, flags, cidx) = frame;
        if word & value_code::REPLAY_BIT == 0 || replay::speculative() {
            return None;
        }
        let keys = replay::frame_keys(reader)?;
        let member = flags.has(cycle::BUFFERED);
        // SAFETY: the keys come from an open context of `reader`.
        let (generation, path, episode, seq) =
            unsafe { (keys.parts)(keys.owner, node.0, cidx, member) };
        Some(ReadKey {
            node: node.0,
            generation,
            cidx,
            reading,
            ordinal,
            path,
            seq,
            episode,
            foreign: Some(self.runtime_id),
        })
    }

    /// A read of `handle` that records nothing in the frame on top of the
    /// stack: `read_slow` with no frame of this runtime's.
    fn read_unframed<T: Value>(
        &self,
        handle: Incr<T>,
        mut observed: NodeState,
        kind: ReadKind,
        word: u64,
    ) -> Result<T, Cycle> {
        let node = self.nodes.get(handle.slot());
        if observed == NodeState::Clean {
            let (type_tag, arena_slot) = self.fields_under(handle, node);
            if let Some(value) = self.try_read_slot(handle, node, type_tag, arena_slot) {
                return Ok(value);
            }
            match self.reload_after_empty_slot(handle, node, type_tag, arena_slot) {
                Ok(value) => return Ok(value),
                Err(now) => observed = now,
            }
        }
        if word & (value_code::COUNTER_MASK | value_code::REPLAY_BIT | value_code::UNWIND_MASK) != 0
        {
            return self.read_cold(handle, observed, kind, None, word);
        }
        self.read_walk::<T, 0>(handle, node, observed, kind, word, None)
    }

    pub(super) fn budgeted(&self) -> bool {
        self.max_frames.load(std::sync::atomic::Ordering::Relaxed) != u32::MAX
            || self
                .max_stack_bytes
                .load(std::sync::atomic::Ordering::Relaxed)
                != 0
    }

    fn keep_read(
        &self,
        ctl: *const cycle::ComputeCtl,
        cidx: u32,
        reading: u64,
        ordinal: u32,
        what: Delivered,
    ) {
        // SAFETY: the frame is on the stack, so its control block is live.
        let flags = unsafe { &*ctl };
        let fresh = !flags.has(cycle::KEPT_READ);
        flags.raise(cycle::KEPT_READ);
        replay::keep_read(
            replay::KeptRead {
                rt: self.runtime_id,
                cidx,
                ctl,
                reading,
                ordinal,
                foreign: None,
                what,
            },
            fresh,
        );
    }

    /// The runtime's frames on top of the thread's stack, the reading
    /// frame first, each with its step and its path.
    fn frames_below(&self) -> Vec<Below> {
        let rt = self.runtime_id;
        let mut frames: Vec<Below> = self.dep_stack.with(|s| {
            let len = s.frames.len();
            s.frames
                .iter()
                .rev()
                .enumerate()
                .take_while(|(_, f)| f.rt() == rt)
                .map(|(i, f)| Below {
                    node: f.node.0,
                    cidx: f.cidx,
                    depth: (len - i) as u32,
                    ctl: f.ctl,
                    step: f.step(),
                    path: 0,
                })
                .collect()
        });
        // Bottom up, each frame's path extends the one below it.
        let mut below: Option<(u32, u64, replay::Step)> = None;
        for f in frames.iter_mut().rev() {
            f.path = match below {
                Some((cidx, path, step)) if cidx + 1 == f.cidx => replay::extend_path(path, step),
                _ => self.path_at(f.cidx),
            };
            f.step.seq = replay::seq_of(rt, f.ctl, f.depth);
            below = Some((f.cidx, f.path, f.step));
        }
        frames
    }

    /// Asserts that every frame [`Self::frames_below`] returns has the
    /// path [`Self::path_at`] gives it, and the frame above it the path
    /// its step extends that to, returning the steps, the reading frame
    /// first.
    #[cfg(test)]
    #[cfg_attr(loom, allow(dead_code))]
    pub(super) fn assert_frame_paths_agree(&self) -> Vec<replay::Step> {
        self.frames_below()
            .iter()
            .map(|f| {
                assert_eq!(f.path, self.path_at(f.cidx), "frame at {}", f.cidx);
                assert_eq!(
                    replay::extend_path(f.path, f.step),
                    self.path_at(f.cidx + 1),
                    "frame above {}",
                    f.cidx
                );
                f.step
            })
            .collect()
    }

    /// Marks the reading frame F and the frames below it that can be
    /// doomed, makes them virtual frames, and pushes `n` one level above F.
    fn mark(&self, n: NodeId, generation: u32, retry: Retry, walked: bool, r5: bool) {
        let rt = self.runtime_id;
        #[cfg(test)]
        assert!(
            self.virtual_position(n.0, generation).is_none(),
            "incr-core: a doom pushed node {} that has a virtual frame",
            n.0
        );
        let frames = self.frames_below();
        let top = frames
            .first()
            .expect("incr-core: a doom has a reading frame");
        let (running, floor, running_entry, base_spec) =
            replay::with(|c| match replay::covering(c, rt, top.cidx) {
                Some(at) => {
                    let x = &c[at];
                    match x.running {
                        Some(level) => {
                            (Some(x.id), x.boundary + level, x.running_entry, x.spec_base)
                        }
                        None => (None, x.boundary, None, x.spec_base),
                    }
                }
                None => (None, 0, None, false),
            });
        let mut count = 1;
        for f in &frames[1..] {
            // SAFETY: a frame's control block is live while it is on the
            // stack, and these frames are below the reading one.
            let ctl = unsafe { &*f.ctl };
            if f.cidx < floor
                || ctl.has(cycle::DOOMED)
                || self.undoomable(ctl, f.depth)
                || !self.replayable(f.node)
                || !self.reading_has_placeholder(f.step.reading)
            {
                break;
            }
            count += 1;
        }
        let marked = &frames[..count];
        let bottom = &marked[count - 1];
        let joined = running.filter(|_| bottom.cidx == floor);
        let spec = replay::speculative()
            || match (joined, running_entry) {
                (Some(_), Some(e)) => e.spec,
                _ => running_entry.map_or(base_spec, |e| e.spec),
            };
        // The compute whose discard runs the context is the bottom frame's,
        // whose retry goes straight to the closure as that frame had, or
        // for a member run, that of the episode's root: its pop or its
        // group verify ran the episode, and runs it again.
        // SAFETY: as above.
        let (target, target_retry, target_walked) = if unsafe { &*bottom.ctl }.has(cycle::BUFFERED)
        {
            let root = replay::member_run_root(bottom.cidx).unwrap_or(bottom.node);
            // The root's compute, run again, is a walk's entry if the
            // episode's was, and contains an unwind as that entry did.
            let walked = contain_impl::walked_episode(bottom.cidx) == Some(NodeId(root));
            (root, Retry::Compute, walked)
        } else {
            // SAFETY: as above.
            let walked = unsafe { &*bottom.ctl }.has(cycle::WALKED);
            (bottom.node, Retry::Direct, walked)
        };
        let id = match joined {
            Some(id) => id,
            None => replay::open(
                rt,
                bottom.cidx,
                Entry {
                    node: target,
                    generation: self.nodes.get(target).generation(),
                    suspect: false,
                    spec,
                    retry: target_retry,
                    walked: target_walked,
                    r5: false,
                },
                self.frame_keys(),
            ),
        };
        #[cfg(test)]
        if let Some(d) = self.dooms.lock().unwrap().last_mut() {
            d.context = id;
        }
        let vframes: Vec<VFrame> = (0..count)
            .rev()
            .map(|i| {
                let f = &marked[i];
                // SAFETY: as above.
                let ctl = unsafe { &*f.ctl };
                let suspect = if i + 1 == count {
                    joined.is_some() && running_entry.is_some_and(|e| e.suspect)
                } else {
                    // SAFETY: as above.
                    unsafe { &*marked[i + 1].ctl }.has(cycle::PH_SEEN)
                };
                VFrame {
                    node: f.node,
                    generation: self.nodes.get(f.node).generation(),
                    cidx: f.cidx,
                    low: ctl.lowlink.get().min(f.cidx),
                    suspect,
                    spec,
                    reading: f.step.reading,
                    ordinal: f.step.ordinal,
                    path: f.path,
                    seq: f.step.seq,
                    maker: 0,
                    ep: if ctl.has(cycle::BUFFERED) {
                        replay::member_run_episode(f.cidx)
                    } else {
                        None
                    },
                    member_of: f.step.episode,
                    computes: {
                        let n = if ctl.has(cycle::BUFFERED) {
                            replay::member_run_root(f.cidx).unwrap_or(f.node)
                        } else {
                            f.node
                        };
                        u64::from(n) | (u64::from(self.nodes.get(n).generation()) << 32)
                    },
                    deferred: if ctl.has(cycle::BUFFERED) {
                        Vec::new()
                    } else {
                        self.owned_claims(f.cidx, f.node, f.path)
                    },
                    walked: ctl.has(cycle::WALKED),
                }
            })
            .collect();
        let kept = self.kept_reads(marked, &vframes);
        // SAFETY: as above.
        let top_ctl = unsafe { &*top.ctl };
        let suspect = top_ctl.has(cycle::PH_SEEN);
        let unkept = replay::with(|c| {
            let x = c
                .iter_mut()
                .find(|x| x.id == id)
                .expect("incr-core: the doom's context is open");
            let maker = bottom.cidx - x.boundary;
            // The running entry's run got past its verify into the closure
            // that this doom discards: its next retry goes straight to the
            // closure, as the unbounded run verifies once, before deps that
            // the entries above land can change the verify's answer.
            // SAFETY: as above.
            if joined.is_some()
                && !unsafe { &*bottom.ctl }.has(cycle::BUFFERED)
                && running_entry.is_some_and(|e| e.node == bottom.node)
            {
                if let Some(e) = x
                    .levels
                    .get_mut(maker as usize)
                    .and_then(|q| q.front_mut())
                    .filter(|e| e.node == bottom.node)
                {
                    e.retry = Retry::Direct;
                }
            }
            for mut v in vframes {
                v.maker = maker;
                x.push_vframe(v);
            }
            let mut unkept = Vec::new();
            for (key, what) in kept {
                // A frame marked again keeps the delivery its first mark, or
                // the entry it was blocked on, made for the read.
                if x.pending
                    .find_keyed(&key, |d| {
                        matches!(d.what, Delivered::Landed(..) | Delivered::Unwind { .. })
                    })
                    .is_some()
                {
                    unkept.push(what);
                } else {
                    let mut d = Delivery::new(key, maker, what);
                    d.passed = true;
                    x.pending.push(d);
                }
            }
            #[cfg(test)]
            assert!(
                x.position(n.0, generation).is_none(),
                "incr-core: a doom pushed node {} which has a virtual frame",
                n.0
            );
            x.push_entry(
                top.cidx + 1 - x.boundary,
                Entry {
                    node: n.0,
                    generation,
                    suspect,
                    spec: spec || suspect,
                    retry,
                    walked,
                    r5,
                },
            );
            unkept
        });
        drop(unkept);
        for f in marked {
            // SAFETY: as above.
            unsafe { &*f.ctl }.raise(cycle::DOOMED);
        }
        top_ctl.raise(cycle::PH_SEEN);
        if value_code::thread_word() & value_code::MEMO_BIT != 0 {
            let parked = taint_memo::take_from(bottom.cidx + 1);
            replay::with(|c| {
                if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                    x.memo.extend(parked);
                }
            });
        }
    }

    /// The pop of a doomed frame: its run, its panic included, lands
    /// nothing, and every claim held at or above it takes the discard
    /// landing, except those virtual frames own, whose root stays held.
    #[cold]
    #[inline(never)]
    pub(super) fn discard_doomed(
        &self,
        id: NodeId,
        attempt: Attempt,
        ctl: &ComputeCtl,
        payload: Option<Box<dyn std::any::Any + Send>>,
        bag: &mut Deferred<C>,
    ) -> Computed {
        replay::note_claim_rev(
            self.runtime_id,
            id.0,
            self.nodes.get(id.0).generation(),
            attempt.cidx,
            attempt.verify_rev,
        );
        let mut keep = Vec::new();
        let rooted = self.keep_virtual(id, attempt.cidx, &mut keep);
        let dropping = self.dropping(attempt.cidx);
        // This thread's contexts compute again every claim this discard
        // releases, unless the covering one drops their chain.
        let reserving = (!C::SINGLE_THREADED)
            .then(|| reserve::Scope::new(self.reserving_context().filter(|_| !dropping)));
        self.release_held_except(attempt.cidx, &keep, bag);
        if let Some(value) = ctl.buf.take() {
            bag.push_erased(Box::new(Unbuilt::new(value)));
        }
        if let Some(payload) = payload.and_then(|p| self.restore_contained(p)) {
            bag.push_erased(Box::new(Unbuilt::new(payload)));
        }
        let wrote = ctl.wrote.get() != 0;
        match rooted {
            Some(path) => self.hold_doomed_root(id, attempt, wrote, path, bag),
            // A replayable closure is pure, so its retry asks for the same
            // children, which the entries above compute meanwhile; unbinding
            // them would make the retry create them New and doom again. A
            // frame of a dropped chain has no retry.
            None => self.discard_land_in_group(
                id,
                attempt.first,
                wrote,
                attempt.stamp_rev,
                if dropping {
                    ChildLanding::Discard
                } else {
                    ChildLanding::Keep
                },
                bag,
            ),
        }
        drop(reserving);
        replay::unwind_past(self.runtime_id, attempt.cidx + 1);
        Computed::Discarded
    }

    /// The context a doom's reservations belong to: this runtime's
    /// outermost on the thread, which outlives every context nested in it,
    /// so a reservation a nested context made for an outer one's entries
    /// lasts as long as they do. `None` under `Local`, which never reserves.
    fn reserving_context(&self) -> Option<u64> {
        if C::SINGLE_THREADED {
            return None;
        }
        replay::with(|c| c.iter().find(|x| x.rt == self.runtime_id).map(|x| x.id))
    }

    /// Whether the frame at `cidx` belongs to a chain its context drops.
    fn dropping(&self, cidx: u32) -> bool {
        replay::with(|c| {
            replay::covering(c, self.runtime_id, cidx).is_some_and(|at| c[at].drop_point.is_some())
        })
    }

    /// The claims that virtual frames at or above `cidx` own since the
    /// mark, added to `keep`: the unbounded run's frames are still on the
    /// stack with them held while the blocked read runs, so the entries
    /// above read them held, not New. When some are anchored at `cidx`, or
    /// a waiting episode's member run read the frame there, the doomed run
    /// of `id` there roots them, and the path of its virtual frame is
    /// returned.
    fn keep_virtual(&self, id: NodeId, cidx: u32, keep: &mut Vec<u32>) -> Option<u64> {
        let rt = self.runtime_id;
        let mut rooted = false;
        cycle_state::with_held(|h| {
            for g in h.virtual_owners(rt, cidx, u32::MAX) {
                for &slot in h.members(g) {
                    let anchor = h[&(rt, slot)].anchor();
                    if anchor >= cidx {
                        keep.push(slot);
                        rooted |= anchor == cidx;
                    }
                }
            }
        });
        rooted |= replay::suspended_low_at(rt, cidx);
        if !rooted || cycle_state::holds(rt, id.0) {
            return None;
        }
        replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .flat_map(|x| x.vstack.iter())
                .find(|v| v.cidx == cidx && v.node == id.0)
                .map(|v| v.path)
        })
    }

    /// The doomed root of kept claims stays claimed, as a virtual hit's
    /// claim of its frame, unbuffered: its virtual frame's pop finishes its
    /// run and roots the episode, as the unbounded run's frame does.
    fn hold_doomed_root(
        &self,
        id: NodeId,
        attempt: Attempt,
        wrote: bool,
        path: u64,
        bag: &mut Deferred<C>,
    ) {
        let rt = self.runtime_id;
        let (node, flags) = self.nodes.get_with_flags(id.0);
        self.children_landing(id, flags, ChildLanding::Keep, false, bag);
        let seeded = C::u8_load_relaxed(flags) & HAS_STRATEGY != 0;
        let mut held = Held::new(node.generation(), attempt.cidx, seeded, attempt.first);
        held.own(attempt.cidx, id.0, Some(path));
        held.wrote = wrote;
        cycle_state::with_held(|h| h.insert((rt, id.0), held));
    }

    /// A walk's compute came back discarded. If the doom opened a context
    /// at the position that compute ran at, the walk is that context's
    /// opener and runs its loop here, then goes on as if the compute had
    /// landed; otherwise (the reading frame is doomed, or the walk is a
    /// driver entry's) the walk stops.
    #[cold]
    #[inline(never)]
    pub(super) fn walk_discarded(&self, visit: Visit) -> Option<Computed> {
        let cidx = self.dep_stack.next_cidx();
        let rt = self.runtime_id;
        let id = replay::with(|c| {
            let at = replay::innermost(c, rt)?;
            let x = &c[at];
            (x.running.is_none() && x.boundary == cidx).then_some(x.id)
        })?;
        Some(self.run_loop(id, visit))
    }

    /// An R5 member run at `cidx` that a doom discarded: if the doom
    /// opened a context at its position, which it does when the member run
    /// is the doom's bottom frame, that context runs until the member it
    /// targets is buffered, or until its member run unwinds, which the
    /// result carries as the nested run's unwind. `None` if the doom marked
    /// the reader instead.
    pub(super) fn member_run_discarded(
        &self,
        cidx: u32,
    ) -> Option<Result<(), Box<dyn std::any::Any + Send>>> {
        let rt = self.runtime_id;
        let id = replay::with(|c| {
            let at = replay::innermost(c, rt)?;
            let x = &c[at];
            (x.running.is_none() && x.boundary == cidx).then_some(x.id)
        })?;
        Some(
            catch_unwind(AssertUnwindSafe(|| self.run_loop(id, Visit::Full))).map(|computed| {
                debug_assert_eq!(computed, Computed::Deferred);
            }),
        )
    }

    /// The driver of context `id`: runs its entries, highest level first
    /// and oldest first within a level, each at its level's position,
    /// until the level-0 target lands, and returns how the target's last
    /// run left it.
    fn run_loop(&self, id: u64, visit: Visit) -> Computed {
        'entries: loop {
            let (level, entry, parked, track) = replay::with(|c| {
                let x = c
                    .iter_mut()
                    .find(|x| x.id == id)
                    .expect("incr-core: a running context is open");
                let parked = std::mem::take(&mut x.memo);
                let (level, entry) = x
                    .next_entry()
                    .expect("incr-core: a context keeps its target until it lands");
                #[cfg(test)]
                self.entries.lock().unwrap().push(EntrySeen {
                    context: x.id,
                    level,
                    target: x.levels.first().and_then(|q| q.front()).map(|e| e.node),
                    levels: x.levels.len(),
                });
                // Running, with no entry yet while its virtual frames pop.
                x.running = Some(level);
                x.running_entry = None;
                x.drop_point = None;
                x.provenance = None;
                let track = x.pending.has_traced();
                (level, entry, parked, track)
            });
            self.park(ParkPoint::BeforeEntry(NodeId(entry.node)));
            let failures = cycle_state::failures();
            if !parked.is_empty() {
                taint_memo::restore(parked);
            }
            // The virtual frames at and above the entry's level pop, highest
            // first, each as the frame it stands for would.
            let mut popped_frames: Vec<(VFrame, bool)> = Vec::new();
            let mut dissolved: Vec<u32> = Vec::new();
            loop {
                let popped = replay::with(|c| {
                    let x = c
                        .iter_mut()
                        .find(|x| x.id == id)
                        .expect("incr-core: a running context is open");
                    if x.vstack.len() > level as usize {
                        let kept = match x.vstack.last() {
                            Some(v) if value_code::thread_word() & value_code::MEMO_BIT != 0 => {
                                taint_memo::take_from(v.cidx + 1)
                            }
                            _ => Vec::new(),
                        };
                        let v = x.pop_vframe()?;
                        if track {
                            x.pending.extend_below((v.cidx, v.node, v.generation));
                        }
                        Some((v, !x.vstack.is_empty(), kept))
                    } else {
                        None
                    }
                });
                let Some((v, virtual_below, mut kept)) = popped else {
                    // The frame left on top was blocked reading the compute
                    // popped last, which the same doomed run made, so no
                    // entry stands for it: the run of a later entry that
                    // retries the frame gets the direct compute below.
                    if let Some((u, false)) = popped_frames.last() {
                        let read = u.computes;
                        replay::with(|c| {
                            let Some(at) = c.iter().position(|x| x.id == id) else {
                                return;
                            };
                            if let Some(w) = c[at]
                                .vstack
                                .last()
                                .filter(|w| {
                                    w.cidx + 1 == u.cidx && w.reading == read && w.maker == u.maker
                                })
                                .cloned()
                            {
                                let runs = replay::run_snapshot(c, self.runtime_id, &w);
                                c[at].deliver_direct(&w, runs);
                            }
                        });
                    }
                    break;
                };
                // A frame still New or Dirty that its pop does not finish is
                // retried as the frame the unbounded run never popped, so it
                // keeps its memo entries: a read of the retry that takes a
                // delivery skips the walk that made them. Its pop runs nothing
                // that reads them, since only a held frame, never New or
                // Dirty, gets a stand-in run with its own entries.
                let node = self.nodes.get(v.node);
                let retried = v.ep.is_none()
                    && matches!(
                        state::load::<C>(node.state_cell()),
                        NodeState::New | NodeState::Dirty
                    )
                    && node.generation() == v.generation;
                self.note_ancestor(&v);
                // The frame that was reading the frame popped just above gets
                // the direct compute of that read, whether a stand-in run in
                // the pop finishes it or its maker's retry, in a later pass,
                // runs it again.
                if let Some((u, false)) = popped_frames.last() {
                    let read = u.computes;
                    if u.cidx == v.cidx + 1 && v.reading == read {
                        replay::with(|c| {
                            let runs = replay::run_snapshot(c, self.runtime_id, &v);
                            if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                                x.deliver_direct(&v, runs);
                            }
                        });
                    }
                }
                let popped = {
                    let _placed = Placed(
                        &self.dep_stack,
                        self.dep_stack.place_next_at(v.cidx, virtual_below),
                    );
                    self.tarjan_pop(id, &v, &mut kept)
                };
                match popped {
                    VPop::Next => {
                        if retried && !kept.is_empty() {
                            taint_memo::restore(kept);
                        }
                        popped_frames.push((v.clone(), false));
                    }
                    VPop::Dissolved(root) => {
                        let generation = self.nodes.get(root).generation();
                        if self.end_dissolved_reads(NodeId(root), generation, v.cidx) {
                            dissolved.push(v.cidx);
                        }
                        popped_frames.push((v.clone(), false));
                    }
                    VPop::Restart(doomed) => {
                        // A doom in the pop handed the context the entries
                        // above it, and the frame it left unfinished keeps its
                        // own with them, as the unbounded run's frame does.
                        if doomed
                            && v.ep.is_none()
                            && !kept.is_empty()
                            && self.vframe_unfinished(&v)
                        {
                            replay::with(|c| {
                                if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                                    x.memo.extend(kept);
                                }
                            });
                        }
                        continue 'entries;
                    }
                    VPop::Finished(node, ended) => {
                        popped_frames.push((v.clone(), ended != Ended::Unwound));
                        if let Some(computed) =
                            self.frame_finished(id, level, entry, &v, node, ended)
                        {
                            return computed;
                        }
                        continue 'entries;
                    }
                }
            }
            let boundary = replay::with(|c| {
                let x = c
                    .iter_mut()
                    .find(|x| x.id == id)
                    .expect("incr-core: a running context is open");
                x.running = Some(level);
                x.running_entry = Some(entry);
                x.boundary
            });
            // An entry whose node this thread holds is a cycle member that a
            // virtual hit claimed after the entry was pushed, or that an R5
            // read doomed. Without a buffer it runs as a buffered member run,
            // which finishes the run its frame would have; with one it is
            // done.
            let held = self
                .look_held(entry.node, Look::here_or(entry.spec), |e| {
                    (e.generation == entry.generation).then_some(e.buf().is_some())
                })
                .flatten();
            if let Some(buffered) = held {
                if !buffered {
                    let cidx = boundary + level;
                    self.retry_deliveries(id, entry.node, entry.generation, cidx);
                    let ran = {
                        // Its frames sit above `cidx`, as an entry's run's do:
                        // at a lower position a panic among them releases
                        // this member's own claim.
                        let _placed = Placed(
                            &self.dep_stack,
                            self.dep_stack.place_next_at(cidx, level > 0),
                        );
                        let _speculating = entry.spec.then(replay::Speculating::enter);
                        self.member_run(entry.node, cidx, true, entry.walked, !entry.r5)
                    };
                    drop(replay::run_ended(
                        self.runtime_id,
                        cidx,
                        matches!(ran, Ok(None)),
                    ));
                    if !matches!(ran, Ok(None)) {
                        self.end_deliveries(id, entry.node, entry.generation, cidx);
                    }
                    match ran {
                        Ok(Some(low)) if low < cidx => self.pass_down(cidx, low),
                        Ok(Some(_)) => {}
                        // A doom discarded the run: its entries run first,
                        // and this entry stays for the retry.
                        Ok(None) => continue,
                        Err(payload) => {
                            let kind = cycle_state::unwind_kind();
                            let contain = entry.walked
                                && !self.episode_member(entry.node)
                                && self.containable(NodeId(entry.node), cidx, kind, &*payload);
                            let mut bag = Deferred::new();
                            let payload =
                                self.continuation_unwound(entry.node, cidx, payload, &mut bag);
                            if !contain {
                                bag.drain(self);
                                match self
                                    .entry_unwound(id, level, entry, boundary, payload, failures)
                                {
                                    Some(computed) => return computed,
                                    None => continue,
                                }
                            }
                            // The run is the walk entry's compute, whose walk
                            // contains it as `compute_panicked` does.
                            cycle_state::take_unwind_kind();
                            self.contain(cidx, payload, &mut bag);
                            bag.drain(self);
                            if level > 0 {
                                self.hold_landing(id, level, entry.node, entry.generation, true);
                            }
                            replay::with(|c| {
                                if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                                    x.pop_entry(level);
                                    x.claims.retain(|&(l, _)| l < level);
                                }
                            });
                            if level == 0 {
                                self.close_context(id, boundary);
                                return Computed::Unsettled;
                            }
                            continue;
                        }
                    }
                }
                replay::with(|c| {
                    if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                        x.pop_entry(level);
                        x.claims.retain(|&(l, _)| l < level);
                    }
                });
                if level == 0 {
                    self.close_context(id, boundary);
                    return Computed::Deferred;
                }
                continue;
            }
            // The entry is the retry of the doomed run that its virtual frame
            // stood for, so it owns what that run deferred.
            let at = boundary + level;
            self.own_deferred(at, entry.node);
            for (v, _) in popped_frames
                .iter()
                .filter(|&(v, landed)| !landed && v.cidx > at)
            {
                let read = v.computes;
                if let Some((w, _)) = popped_frames.iter().find(|(w, _)| {
                    w.cidx + 1 == v.cidx && w.reading == read && !dissolved.contains(&w.cidx)
                }) {
                    replay::with(|c| {
                        let runs = replay::run_snapshot(c, self.runtime_id, w);
                        if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                            x.deliver_direct(w, runs);
                        }
                    });
                }
            }
            let placement = self.dep_stack.place_next_at(at, level > 0);
            let ran = {
                let _speculating = entry.spec.then(replay::Speculating::enter);
                catch_unwind(AssertUnwindSafe(|| {
                    if level == 0 {
                        // The target is the walk entry whose compute was
                        // discarded, and the walk has cleaned its deps, so
                        // it computes as that entry would, not as a new
                        // walk's root.
                        let visit = if entry.retry == Retry::Direct {
                            Visit::Direct
                        } else {
                            visit
                        };
                        self.compute_one::<VIRT>(
                            NodeId(entry.node),
                            visit,
                            Some(entry.generation),
                            entry.walked,
                            None,
                            Seen::Unwalked,
                        )
                    } else {
                        let (id, generation) = (NodeId(entry.node), Some(entry.generation));
                        match entry.retry {
                            Retry::Walk => self.ensure_clean_virt(id, entry.generation),
                            Retry::Compute => self.compute_one::<VIRT>(
                                id,
                                Visit::Full,
                                generation,
                                entry.walked,
                                None,
                                Seen::Unwalked,
                            ),
                            Retry::Direct => self.compute_one::<VIRT>(
                                id,
                                Visit::Direct,
                                generation,
                                entry.walked,
                                None,
                                Seen::Unwalked,
                            ),
                        }
                    }
                }))
            };
            self.dep_stack.restore_placement(placement);
            let dropped = replay::with(|c| {
                c.iter_mut()
                    .find(|x| x.id == id)
                    .and_then(|x| x.drop_point.take())
            });
            // A delivery its frame's retry never read is dropped once the
            // run ends, as the unbounded run drops it when a set changed
            // that frame's earlier reads. A run doomed again has not
            // ended: the frame comes back as an entry of its own, and its
            // next retry reads the delivery again.
            if matches!(ran, Ok(Computed::Discarded)) && dropped.is_none() {
                replay::with(|c| {
                    if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                        for i in x.pending.of_maker(level) {
                            x.pending.set_taken(i, false);
                        }
                    }
                });
            } else {
                let unread: Vec<Delivery> = replay::with(|c| {
                    c.iter_mut()
                        .find(|x| x.id == id)
                        .map(|x| x.end_maker(level))
                        .unwrap_or_default()
                });
                drop(unread);
            }
            let computed = match ran {
                Ok(computed) => computed,
                Err(payload) => {
                    match self.entry_unwound(id, level, entry, boundary, payload, failures) {
                        Some(computed) => computed,
                        None => continue,
                    }
                }
            };
            if computed == Computed::Discarded {
                if let Some(q) = dropped {
                    #[cfg(test)]
                    self.cooperative_drops
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let mut bag = Deferred::new();
                    self.drop_chain(id, q, false, &mut bag);
                    bag.drain(self);
                }
                continue;
            }
            debug_assert!(dropped.is_none(), "incr-core: a dropped run landed");
            debug_assert_ne!(computed, Computed::Doomed);
            if matches!(computed, Computed::Done | Computed::Unsettled) && level > 0 {
                self.hold_landing(
                    id,
                    level,
                    entry.node,
                    entry.generation,
                    computed == Computed::Unsettled,
                );
            }
            replay::with(|c| {
                if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                    x.pop_entry(level);
                    // The claims the run's hits took below its level stay
                    // held for the episode their position runs.
                    x.claims.retain(|&(l, _)| l < level);
                }
            });
            if level == 0 {
                self.close_context(id, boundary);
                return computed;
            }
        }
    }

    /// Whether the run `v` stands for has yet to end: its node is still
    /// New or Dirty, a claim this thread holds, or another thread's claim
    /// of a node this thread reserves, which comes back unrun.
    fn vframe_unfinished(&self, v: &VFrame) -> bool {
        let node = self.nodes.get(v.node);
        let state = state::load::<C>(node.state_cell());
        // After the load: a recycle bumps it before its node's state.
        if node.generation() != v.generation {
            return false;
        }
        match state {
            NodeState::New | NodeState::Dirty => true,
            NodeState::Computing | NodeState::ComputingDirty => {
                cycle_state::holds(self.runtime_id, v.node) || self.given_back(v)
            }
            _ => false,
        }
    }

    /// Whether another thread's claim of `v`'s node, which this thread
    /// reserves, came back unrun.
    #[cold]
    #[inline(never)]
    fn given_back(&self, v: &VFrame) -> bool {
        #[cfg(test)]
        self.give_back_waits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        matches!(
            self.given_back_state(v.node, v.generation),
            Some(NodeState::New | NodeState::Dirty)
        )
    }

    /// The state another thread's claim of `node`, which this thread
    /// reserves at `generation`, leaves once it ends, or `None` if this
    /// thread does not reserve it.
    #[cold]
    #[inline(never)]
    pub(crate) fn given_back_state(&self, node: u32, generation: u32) -> Option<NodeState> {
        if C::SINGLE_THREADED || !reserve::mine((self.runtime_id, node), generation) {
            return None;
        }
        let node_data = self.nodes.get(node);
        let mut spins = 0;
        loop {
            let observed = state::load::<C>(node_data.state_cell());
            // After the load: a recycled slot's occupant is no claim of
            // the reserved node.
            if node_data.generation() != generation {
                return None;
            }
            if !is_computing(observed) {
                return Some(observed);
            }
            // Another thread may end the reservation under the wait; its
            // claim is then no give-back and may wait on this thread.
            if !reserve::mine((self.runtime_id, node), generation) {
                return None;
            }
            spins += 1;
            C::backoff(spins);
        }
    }

    /// A virtual frame's pop, as its frame's pop would be: nothing if its
    /// node has landed or been replaced since, a lowlink below its position
    /// passed down, and at a lowlink equal to its position with claims
    /// anchored there, the episode it roots, run here as a task. A
    /// speculative frame never runs one: its chain is dropped instead, and
    /// the driver picks its next entry again (`true`).
    fn tarjan_pop(&self, id: u64, v: &VFrame, kept: &mut taint_memo::Parked) -> VPop {
        let rt = self.runtime_id;
        let failures = cycle_state::failures();
        if !self.vframe_unfinished(v) {
            return VPop::Next;
        }
        // What the pop runs sits at the frame's position: a doom in it joins
        // the context there.
        replay::with(|c| {
            if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                x.running = Some(v.cidx - x.boundary);
                x.running_entry = None;
            }
        });
        // The run that retries the frame takes its lowlink, a suspended
        // episode's member run as much as a stand-in.
        let mut low = v.low;
        let unbuffered = self.look_held(v.node, Look::here_or(v.spec), |e| e.buf().is_none());
        if low < v.cidx && unbuffered != Some(false) {
            replay::with(|c| {
                if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                    x.lows.push((v.node, v.generation, v.cidx, v.path, low));
                }
            });
        }
        if let Some(ep) = v.ep {
            let ep = replay::take_episode(id, ep)
                .expect("incr-core: a member run's virtual frame keeps its episode");
            #[cfg(test)]
            {
                let queued_above = replay::with(|c| {
                    c.iter().find(|x| x.id == id).map_or(0, |x| {
                        x.levels
                            .iter()
                            .skip((v.cidx - x.boundary) as usize + 1)
                            .map(|q| q.len())
                            .sum()
                    })
                });
                self.episodes_seen
                    .lock()
                    .unwrap()
                    .push(cycle_impl::EpisodeSeen::of(&ep, Some(queued_above)));
            }
            let (root, cidx) = (ep.root, ep.cidx);
            let mut bag = Deferred::new();
            let end = catch_unwind(AssertUnwindSafe(|| self.run_episode(ep, &mut bag)));
            return match end {
                Ok(end) => self.episode_task_end(root, cidx, end, bag),
                Err(payload) => self.vframe_unwound(id, v, root.0, false, payload, failures),
            };
        }
        // A virtual hit claimed the node, so no retry recomputes it: its
        // frame's run finishes here, as the unbounded run's frame does
        // after its blocked read returns, and buffers its value. The claim
        // stands for that frame on the stack, so the frame's pop decides
        // what becomes of it, as a compute's pop would.
        let mut stand_in = None;
        if unbuffered == Some(true) {
            stand_in = Some((
                self.current_revision(),
                C::walked(&self.walk, &self.revision),
            ));
            self.retry_deliveries(id, v.node, v.generation, v.cidx);
            // The run finishes the frame's doomed run, so it owns what that
            // run deferred, and its reads see the frame's own memo entries,
            // which the unbounded run's frame keeps until it pops.
            self.own_deferred(v.cidx, v.node);
            let (own, above) = std::mem::take(kept)
                .into_iter()
                .partition(|&(_, owner)| owner == v.cidx + 1);
            *kept = above;
            taint_memo::restore(own);
            let ran = {
                let _speculating = v.spec.then(replay::Speculating::enter);
                self.member_run(v.node, v.cidx, true, v.walked, true)
            };
            drop(replay::run_ended(rt, v.cidx, matches!(ran, Ok(None))));
            match ran {
                Ok(Some(l)) => {
                    self.end_deliveries(id, v.node, v.generation, v.cidx);
                    // The run started from the doomed runs' lowlink
                    // (`take_low`), so its own is the frame's.
                    low = l;
                }
                Ok(None) => return VPop::Restart(true),
                Err(payload) => {
                    self.end_deliveries(id, v.node, v.generation, v.cidx);
                    return self.vframe_unwound(id, v, v.node, true, payload, failures);
                }
            }
        }
        // An R5 member run's frame: the member stays held for its episode,
        // so the pop only finishes the run, as the run nested in its reader
        // does, and passes the run's lowlink down.
        if self.episode_member(v.node) {
            if low < v.cidx {
                if stand_in.is_some() {
                    self.pass_down(v.cidx, low);
                } else {
                    self.lower_parent(v.cidx, low);
                }
            }
            return VPop::Next;
        }
        if low < v.cidx {
            if stand_in.is_some() {
                // The run finished the frame, which defers as `cycle_pop`
                // defers a real one.
                cycle_state::with_held(|h| {
                    assert!(
                        h.contains_key(&(rt, v.node)),
                        "incr-core: a stand-in is held"
                    );
                    h.set_anchor((rt, v.node), v.cidx);
                    h.own((rt, v.node), v.cidx, v.node, None);
                });
                self.pass_down(v.cidx, low);
            } else if unbuffered.is_some() {
                self.lower_parent(v.cidx, low);
            }
            // Otherwise the frame's run is not over: its entry, or the run
            // that computes it natively, retries it at this position and
            // passes its own lowlink down, or unwinds and passes none.
            return VPop::Next;
        }
        let rooted = cycle_state::with_held(|h| {
            h.anchored(rt, v.cidx)
                .any(|slot| stand_in.is_none() || slot != v.node)
        }) || stand_in.is_some()
            && self
                .look_held(v.node, Look::here_or(v.spec), |e| {
                    e.last().iter().any(|&(d, _)| d.0 == v.node)
                })
                .expect("incr-core: a stand-in is held");
        if let (Some(revs), false) = (stand_in, rooted) {
            if !(v.spec && self.alone_lands_dirty(v.node, v.spec)) {
                if v.spec {
                    // `land_alone` takes a speculative run's buffer only when
                    // adopted: over Clean deps alone it is the value the
                    // unbounded run's own run of the node gives.
                    cycle_state::with_held(|h| {
                        h.get_mut(&(rt, v.node))
                            .expect("incr-core: a stand-in is held")
                            .adopt();
                    });
                }
                let mut bag = Deferred::new();
                return match self.land_alone(v.node, revs, &mut bag) {
                    Ok(landed) => {
                        bag.drain(self);
                        match landed {
                            Some(NodeState::Clean) => VPop::Finished(v.node, Ended::Landed),
                            Some(_) => VPop::Finished(v.node, Ended::Unsettled),
                            None => VPop::Next,
                        }
                    }
                    Err(payload) => {
                        // The landing finishes the stand-in's run, whose
                        // speculative guard has dropped.
                        cycle_state::set_unwind_kind(replay::run_unwind_kind(
                            cycle_state::unwind_kind(),
                            v.spec || replay::speculative(),
                        ));
                        bag.drain(self);
                        self.vframe_unwound(id, v, v.node, true, payload, failures)
                    }
                };
            }
        } else if !rooted {
            return VPop::Next;
        }
        let mut bag = Deferred::new();
        if v.spec {
            let q = replay::with(|c| {
                let x = c
                    .iter()
                    .find(|x| x.id == id)
                    .expect("incr-core: a running context is open");
                let mut q = v.cidx - x.boundary;
                while q > 0 && x.vstack[q as usize - 1].spec {
                    q -= 1;
                }
                q
            });
            if q == 0 {
                self.close_dropped(id);
                self.drop_unwind();
            }
            self.drop_chain(id, q, false, &mut bag);
            bag.drain(self);
            return VPop::Restart(false);
        }
        if !cycle_state::holds(rt, v.node) {
            debug_assert!(
                false,
                "incr-core: a virtual frame roots claims it does not hold"
            );
            return VPop::Next;
        }
        if stand_in.is_some() {
            cycle_state::with_held(|h| {
                if h.contains_key(&(rt, v.node)) {
                    h.set_anchor((rt, v.node), v.cidx);
                }
            });
        }
        let end = catch_unwind(AssertUnwindSafe(|| {
            self.episode(NodeId(v.node), v.cidx, v.walked, &mut bag)
        }));
        match end {
            Ok(end) => self.episode_task_end(NodeId(v.node), v.cidx, end, bag),
            Err(payload) => self.vframe_unwound(id, v, v.node, false, payload, failures),
        }
    }

    /// Whether held node `slot`, landing alone in a speculative run, would
    /// land other than Clean, which such a run never does. `spec` is the
    /// popping frame's own flag.
    fn alone_lands_dirty(&self, slot: u32, spec: bool) -> bool {
        self.look_held(slot, Look::here_or(spec), |e| {
            e.poison() != 0
                || e.last().iter().any(|&(d, _)| {
                    !matches!(
                        state::load::<C>(self.nodes.get(d.0).state_cell()),
                        NodeState::Clean | NodeState::Cyclic
                    )
                })
        })
        .expect("incr-core: a landing claim is held")
    }

    /// The end of an episode a virtual frame's pop ran. A dissolved root's
    /// claim lands as if its run never happened, since no frame computes it
    /// here: the entry below does. It keeps the root in its group, where the
    /// unbounded run's frame holds it until its alone landing, so another
    /// thread verifies the group as in the order where it reads first; this
    /// entry's retry computes the root alone (`end_dissolved`). A suspended
    /// episode restarts the driver, which runs the entries its doom pushed
    /// first.
    fn episode_task_end(&self, root: NodeId, cidx: u32, end: PopEnd, mut bag: Deferred<C>) -> VPop {
        if matches!(end, PopEnd::Dissolved | PopEnd::Retry) {
            self.rerun_root(root, cidx);
        }
        let pop = match end {
            PopEnd::Suspended => VPop::Restart(true),
            PopEnd::Landed => VPop::Finished(root.0, Ended::Landed),
            PopEnd::Contained => VPop::Finished(root.0, Ended::Contained),
            PopEnd::Dissolved => VPop::Dissolved(root.0),
            _ => VPop::Next,
        };
        if let PopEnd::Dissolved = end {
            let (first, wrote) = self.end_dissolved(root, &mut bag);
            let stamp_rev = self.current_revision();
            // The entry below computes the root again.
            let _reserving =
                (!C::SINGLE_THREADED).then(|| reserve::Scope::new(self.reserving_context()));
            self.discard_land_in_group(
                root,
                first,
                wrote,
                stamp_rev,
                ChildLanding::Discard,
                &mut bag,
            );
        }
        bag.drain(self);
        pop
    }

    /// The frame of `node` at virtual frame `v`'s position ended in an
    /// unwind, out of the run that finished it (`continuation`, after which
    /// the node lands as that frame's compute would) or out of the episode
    /// it rooted, which released its claims. The unwind goes on to the
    /// frame below, a virtual frame's retry or the read that ran the loop.
    /// A wound closes the context and unwinds on, as out of an entry: the
    /// virtual frames a doom in the ended run pushed may name an episode
    /// the wound aborted.
    fn vframe_unwound(
        &self,
        id: u64,
        v: &VFrame,
        node: u32,
        continuation: bool,
        payload: Box<dyn std::any::Any + Send>,
        failures: u64,
    ) -> VPop {
        let kind = cycle_state::unwind_kind();
        let mut bag = Deferred::new();
        let mut payload = payload;
        if continuation {
            let contain = v.walked
                && !self.episode_member(v.node)
                && self.containable(NodeId(v.node), v.cidx, kind, &*payload);
            payload = self.continuation_unwound(v.node, v.cidx, payload, &mut bag);
            if contain {
                cycle_state::take_unwind_kind();
                self.contain(v.cidx, payload, &mut bag);
                bag.drain(self);
                return VPop::Finished(node, Ended::Contained);
            }
        }
        let (boundary, pos) = replay::with(|c| {
            let x = c
                .iter()
                .find(|x| x.id == id)
                .expect("incr-core: a running context is open");
            (x.boundary, v.cidx - x.boundary)
        });
        if kind == Some(UnwindKind::Wound) {
            bag.drain(self);
            self.unwind_out_of(id, boundary, kind, payload);
        }
        if v.spec || kind == Some(UnwindKind::Discard) {
            let q = replay::with(|c| {
                let x = c
                    .iter()
                    .find(|x| x.id == id)
                    .expect("incr-core: a running context is open");
                let mut q = pos;
                while q > 0 && x.vstack[q as usize - 1].spec {
                    q -= 1;
                }
                q
            });
            if q == 0 {
                bag.drain(self);
                self.close_dropped(id);
                resume_unwind(payload);
            }
            cycle_state::take_unwind_kind();
            if let Some(payload) = self.restore_contained(payload) {
                bag.push_erased(payload);
            }
            self.drop_chain(id, q, false, &mut bag);
            bag.drain(self);
            return VPop::Restart(false);
        }
        if pos == 0 {
            // The read that ran the loop reads the target as a compute's
            // unwind leaves its reader.
            self.dep_stack
                .poison_current_frame_with(self.runtime_id, self.unwound_poison(kind));
            bag.drain(self);
            self.unwind_out_of(id, boundary, kind, payload);
        }
        cycle_state::take_unwind_kind();
        let poison = if continuation && self.episode_member(node) {
            0
        } else {
            self.failed_poison(kind, v.node)
        };
        let replaced = replay::with(|c| match c.iter_mut().find(|x| x.id == id) {
            Some(x) => x.deliver_unwind(
                v.cidx,
                Delivered::Unwind {
                    cycle: kind == Some(UnwindKind::Cycle),
                    poison,
                    attributed: cycle_state::failures() != failures,
                    nodes: self.take_notes(&*payload, v.cidx),
                    payload,
                },
            ),
            None => {
                bag.push_erased(payload);
                Vec::new()
            }
        });
        drop(replaced);
        bag.drain(self);
        VPop::Finished(node, Ended::Unwound)
    }

    /// Lands held node `slot`, whose run at `cidx` ended in an unwind, as
    /// a compute's frame would: the claims anchored at or above `cidx` are
    /// released, and the node lands Failed for a user panic, or takes the
    /// discard landing for an engine unwind, which a speculative run's
    /// panic is (`member_run`). Returns the payload.
    fn continuation_unwound(
        &self,
        slot: u32,
        cidx: u32,
        payload: Box<dyn std::any::Any + Send>,
        bag: &mut Deferred<C>,
    ) -> Box<dyn std::any::Any + Send> {
        let rt = self.runtime_id;
        let kind = cycle_state::unwind_kind();
        let discarding = kind.is_some();
        // An episode's member stays held for its episode, as when the run
        // nests in its reader: only what the run left above it goes.
        if self.episode_member(slot) {
            self.release_unwound_from(cidx, slot, kind, discarding, Some(&*payload), bag);
            return payload;
        }
        let Some(e) = cycle_state::with_held(|h| h.remove(&(rt, slot))) else {
            return payload;
        };
        self.count_release();
        let (e, spec_buf) = e.into_look(Look::here());
        self.release_unwound_from(cidx, slot, kind, discarding, Some(&*payload), bag);
        let id = NodeId(slot);
        let stamp_rev = self.current_revision();
        if discarding {
            self.unwound_land(id, e.first, e.wrote, stamp_rev, kind, bag);
        } else {
            let (node, flags) = self.nodes.get_with_flags(slot);
            let attempt = Attempt {
                first: e.first,
                cidx,
                stamp_rev,
                verify_rev: C::walked(&self.walk, &self.revision),
            };
            self.land_failed(
                id,
                node,
                flags,
                attempt,
                e.wrote,
                e.deps(),
                &[],
                Some(&*payload),
                false,
                bag,
            );
            self.end_reservation(id, node);
        }
        if let Some(spec_buf) = spec_buf {
            bag.push_erased(Box::new(spec_buf));
        }
        bag.push_erased(Box::new(e.into_values()));
        payload
    }

    /// Whether held node `slot` is a member of a running episode.
    fn episode_member(&self, slot: u32) -> bool {
        cycle_state::with_held(|h| {
            h.get(&(self.runtime_id, slot))
                .is_some_and(|e| e.episode.is_some())
        })
    }

    /// Virtual frame `v` pops: the claims frames above it own keep its
    /// path, so that only its retry takes them over.
    fn note_ancestor(&self, v: &VFrame) {
        cycle_state::with_held(|h| h.note_ancestor(self.runtime_id, v.cidx, v.path));
    }

    /// The claims a doomed run of `node` at `cidx` deferred become those of
    /// the run that retries or finishes it there, which releases them if
    /// it unwinds.
    fn own_deferred(&self, cidx: u32, node: u32) {
        let rt = self.runtime_id;
        cycle_state::with_held(|h| {
            for g in h.virtual_owners(rt, cidx, cidx) {
                if h.owner(g).node() == node {
                    h.own_retry(g);
                }
            }
        });
    }

    /// The run of `id` at `cidx`, whose control block is `ctl`, starts
    /// (see [`replay::start_run`]). Both a compute and a member run call it
    /// with the replay bit set; it gives a seq only under a context of this
    /// runtime.
    #[cold]
    #[inline(never)]
    pub(super) fn start_run(&self, id: NodeId, cidx: u32, ctl: &ComputeCtl) {
        let rt = self.runtime_id;
        if !replay::peek(|c| replay::innermost(c, rt).is_some()) {
            return;
        }
        let frame = (id.0, self.nodes.get(id.0).generation(), cidx);
        let depth = self.dep_stack.depth() as u32;
        let path = self
            .path_above_run(cidx, depth)
            .unwrap_or_else(|| self.path_at(cidx));
        debug_assert_eq!(path, self.path_at(cidx), "the path at {cidx}");
        replay::start_run(rt, ctl, depth, frame, path);
    }

    /// [`Self::path_at`] for a frame at `cidx`, `depth`, from the run of
    /// the frame below it, when that frame is this runtime's at `cidx - 1`
    /// and no virtual frame stands there: its path extended by its step.
    fn path_above_run(&self, cidx: u32, depth: u32) -> Option<u64> {
        let rt = self.runtime_id;
        let below = depth.checked_sub(1)?;
        let (ctl, mut step) = self.dep_stack.with(|s| {
            let f = s.frames.get(below.checked_sub(1)? as usize)?;
            (f.rt() == rt && f.cidx + 1 == cidx).then(|| (f.ctl, f.step()))
        })?;
        let (seq, path) = replay::run_at(rt, ctl, below)?;
        let virtual_below = replay::peek(|c| {
            (c.iter().filter(|x| x.rt == rt)).any(|x| x.vframe_at(cidx - 1).is_some())
        });
        if virtual_below {
            return None;
        }
        step.seq = seq;
        Some(replay::extend_path(path, step))
    }

    /// The root of an episode at `cidx` that dissolved or went stale
    /// computes again, as the unbounded run's does: another run, whose
    /// reads walk afresh instead of taking what this run's reads were
    /// given. Callers gate it on replay, so no other path pays for it.
    pub(super) fn rerun_root(&self, root: NodeId, cidx: u32) {
        self.end_run(root, cidx);
    }

    /// The run of `id` at `cidx` ended, not doomed: a later run of the
    /// frame there is another run (see [`replay::end_run`]).
    #[cold]
    #[inline(never)]
    pub(super) fn end_run(&self, id: NodeId, cidx: u32) {
        self.end_run_along(id, cidx, |generation| {
            self.run_path(id.0, generation, cidx).0
        });
    }

    /// [`Self::end_run`], with the run's path given its node's generation.
    fn end_run_along(&self, id: NodeId, cidx: u32, path: impl FnOnce(u32) -> u64) {
        let rt = self.runtime_id;
        let Some(below) = cidx.checked_sub(1) else {
            return;
        };
        if !replay::with(|c| replay::covering(c, rt, below).is_some()) {
            return;
        }
        let generation = self.nodes.get(id.0).generation();
        replay::end_run(rt, (id.0, generation, cidx), path(generation));
    }

    /// A run of `node` that finishes its virtual frame's frame starts: the
    /// deliveries held for that frame are read again, in order.
    fn retry_deliveries(&self, id: u64, node: u32, generation: u32, cidx: u32) {
        replay::with(|c| {
            if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                for i in x.pending.of_frame(node, generation, cidx) {
                    x.pending.set_taken(i, false);
                }
            }
        });
    }

    /// That run ended: the deliveries held for its frame are dropped.
    fn end_deliveries(&self, id: u64, node: u32, generation: u32, cidx: u32) {
        let ended: Vec<Delivery> = replay::with(|c| {
            c.iter_mut()
                .find(|x| x.id == id)
                .map(|x| x.pending.extract_frame(node, generation, cidx))
                .unwrap_or_default()
        });
        drop(ended);
    }

    /// Drops the deliveries, claim watermarks and failure counts held, in
    /// any of this runtime's contexts, for the frames of the nodes in
    /// `released`, whose claims a user panic's release just landed: that
    /// release ended the frames' runs. A
    /// speculative run's release is not the unbounded run's, so it keeps
    /// them.
    #[cold]
    #[inline(never)]
    pub(super) fn end_released_runs(&self, released: &[u32]) {
        if replay::speculative() {
            return;
        }
        let rt = self.runtime_id;
        let dropped: Vec<Delivery> = replay::with(|c| {
            let mut dropped = Vec::new();
            for x in c.iter_mut().filter(|x| x.rt == rt) {
                for &slot in released {
                    let generation = self.nodes.get(slot).generation();
                    dropped.extend(x.pending.extract_node(slot, generation));
                    x.end_node_runs(slot, generation);
                }
            }
            dropped
        });
        drop(dropped);
    }

    /// Drops the deliveries held, in any of this runtime's contexts, for
    /// reads by the frame of `root` at `cidx`, whose episode just
    /// dissolved: the unbounded run's frame runs the closure again, and the
    /// new run's reads are not the ended run's. A speculative run's dissolve
    /// is not the unbounded run's, so it keeps them; returns whether they
    /// were dropped.
    #[cold]
    #[inline(never)]
    pub(super) fn end_dissolved_reads(&self, root: NodeId, generation: u32, cidx: u32) -> bool {
        if replay::speculative() {
            return false;
        }
        let rt = self.runtime_id;
        let ended: Vec<Delivery> = replay::with(|c| {
            c.iter_mut()
                .filter(|x| x.rt == rt)
                .flat_map(|x| x.pending.extract_frame(root.0, generation, cidx))
                .collect()
        });
        // No retry reads the ended run's frame again, so its unread reads
        // are not marked as [`Self::drop_deliveries`] marks a retry's.
        drop(ended);
        true
    }

    /// Releases, with the discard landing, the claims virtual hits took
    /// for the entries of context `id` at or above `level` that are still
    /// held. Each is the node of a virtual frame below `level`, which that
    /// frame's retry or pop lands again, so a user panic needs no other
    /// landing; at level 0 every such frame has popped and none is held.
    fn release_claim_list(
        &self,
        id: u64,
        level: u32,
        kind: Option<UnwindKind>,
        bag: &mut Deferred<C>,
    ) {
        let rt = self.runtime_id;
        let slots: Vec<u32> = replay::with(|c| {
            c.iter_mut()
                .find(|x| x.id == id)
                .map(|x| {
                    x.claims
                        .extract_if(.., |&mut (l, _)| l >= level)
                        .map(|(_, slot)| slot)
                        .collect()
                })
                .unwrap_or_default()
        });
        let stamp_rev = self.current_revision();
        if !slots.is_empty() {
            self.count_release();
        }
        for slot in slots {
            if let Some(e) = cycle_state::with_held(|h| h.remove(&(rt, slot))) {
                self.unwound_land(NodeId(slot), e.first, e.wrote, stamp_rev, kind, bag);
                bag.push_erased(Box::new(e.into_values()));
            }
        }
    }

    /// An entry's run ended in an unwind. A wound unwinds on. A
    /// speculative entry's panic or cycle is swallowed, and the entry is
    /// dropped (`Some(Gone)`). Any other unwind of an entry above level 0
    /// is held for the virtual frame below it, whose entry's retry raises
    /// it at the read the frame was blocked in (`None`), and the level-0
    /// target's unwinds on to the read that ran the loop.
    fn entry_unwound(
        &self,
        id: u64,
        level: u32,
        entry: Entry,
        boundary: u32,
        payload: Box<dyn std::any::Any + Send>,
        failures: u64,
    ) -> Option<Computed> {
        let kind = cycle_state::unwind_kind();
        if kind == Some(UnwindKind::Discard) {
            let (own, enclosing) = replay::with(|c| {
                let at = c
                    .iter()
                    .position(|x| x.id == id)
                    .expect("incr-core: a running context is open");
                let enclosing = c[..at]
                    .iter()
                    .any(|x| x.rt == self.runtime_id && x.drop_at.is_some());
                (c[at].drop_at.take(), enclosing)
            });
            if let Some(q) = own {
                cycle_state::take_unwind_kind();
                let mut bag = Deferred::new();
                if let Some(payload) = self.restore_contained(payload) {
                    bag.push_erased(payload);
                }
                self.drop_chain(id, q, true, &mut bag);
                bag.drain(self);
                return None;
            }
            if enclosing {
                self.unwind_out_of(id, boundary, kind, payload);
            }
        }
        if kind == Some(UnwindKind::Wound) || (!entry.spec && level == 0) {
            self.unwind_out_of(id, boundary, kind, payload);
        }
        if entry.spec {
            // The speculative chain the entry belongs to starts at the
            // lowest level whose virtual frames below are all real ones'.
            // Dropping only the entry would livelock: the speculative
            // entry below re-reads the node on its retry and pushes it
            // again. The real path's retry reads whatever the unbounded
            // run reads, and raises the panic itself if it gets there.
            let q = replay::with(|c| {
                let x = c
                    .iter()
                    .find(|x| x.id == id)
                    .expect("incr-core: a running context is open");
                let mut q = level;
                while q > 0 && x.vstack[q as usize - 1].spec {
                    q -= 1;
                }
                q
            });
            if q == 0 {
                self.unwind_out_of(id, boundary, kind, payload);
            }
            cycle_state::take_unwind_kind();
            let mut bag = Deferred::new();
            if let Some(payload) = self.restore_contained(payload) {
                bag.push_erased(payload);
            }
            self.drop_chain(id, q, false, &mut bag);
            bag.drain(self);
            return None;
        }
        cycle_state::take_unwind_kind();
        // An R5 member run's unwind leaves its reader unpoisoned, as the
        // member stays held for its episode.
        let poison = if self.episode_member(entry.node) {
            0
        } else {
            self.failed_poison(kind, entry.node)
        };
        let (replaced, abandoned) = replay::with(|c| {
            let x = c
                .iter_mut()
                .find(|x| x.id == id)
                .expect("incr-core: a running context is open");
            let replaced = x.deliver_unwind(
                boundary + level,
                Delivered::Unwind {
                    cycle: kind == Some(UnwindKind::Cycle),
                    poison,
                    attributed: cycle_state::failures() != failures,
                    nodes: self.take_notes(&*payload, boundary + level),
                    payload,
                },
            );
            x.levels.truncate(level as usize);
            x.trim_levels();
            (replaced, x.truncate_vstack(level as usize))
        });
        drop(replaced);
        let mut bag = Deferred::new();
        for ep in abandoned {
            bag.push_erased(Box::new(ep));
        }
        self.release_held_from(boundary + level, true, kind, &mut bag);
        self.release_claim_list(id, level, kind, &mut bag);
        bag.drain(self);
        None
    }

    /// Drops the chain of context `id` from level `q` up: every entry
    /// above `q`, and the oldest entry at `q`, whose run made the chain,
    /// unless `keep` says it is kept for a retry. The virtual frames from
    /// `q` up are truncated with no Tarjan pops, and the claims held at or
    /// above `q` take the discard landing.
    fn drop_chain(&self, id: u64, q: u32, keep: bool, bag: &mut Deferred<C>) {
        debug_assert!(
            keep || q > 0,
            "incr-core: a drop removed the level-0 target"
        );
        let (boundary, abandoned) = replay::with(|c| {
            let x = c
                .iter_mut()
                .find(|x| x.id == id)
                .expect("incr-core: a running context is open");
            x.levels.truncate(q as usize + 1);
            if !keep {
                x.levels[q as usize].pop_front();
            }
            x.trim_levels();
            // Entries a doom handed over from the dropped frames go with
            // them, as a frame's pop clears what it owns (owner = position
            // + 1).
            let from = x.boundary + q + 1;
            x.memo.retain(|&(_, owner)| owner < from);
            (x.boundary, x.truncate_vstack(q as usize))
        });
        for ep in abandoned {
            bag.push_erased(Box::new(ep));
        }
        let kind = Some(UnwindKind::Discard);
        self.release_held_from(boundary + q, true, kind, bag);
        self.release_claim_list(id, q, kind, bag);
    }

    /// Closes context `id`, whose speculative chain reaches its level-0
    /// target, before the unwind that ends the speculative run which
    /// opened it: the claims held from its boundary up take the discard
    /// landing.
    fn close_dropped(&self, id: u64) {
        let mut bag = Deferred::new();
        self.drop_chain(id, 0, true, &mut bag);
        bag.drain(self);
        let boundary = replay::with(|c| {
            c.iter()
                .find(|x| x.id == id)
                .expect("incr-core: a running context is open")
                .boundary
        });
        self.close_context(id, boundary);
    }

    /// The unwind held for frame `f`'s read of `handle`, if this is the
    /// read it was held for: a cycle a `try_get` returns, or the unwind
    /// raised again.
    pub(super) fn take_delivery<T: Value>(
        &self,
        f: ActiveRead,
        handle: Incr<T>,
        kind: ReadKind,
    ) -> Option<Result<T, Cycle>> {
        let rt = self.runtime_id;
        let reading = u64::from(handle.slot()) | (u64::from(handle.generation()) << 32);
        // A delivery stays until its maker's run ends, because a retry
        // doomed again after the read comes back to it as an entry and must
        // read what the first one did. A payload that can't be copied is
        // handed over once, and a later take regenerates it (`regenerate`).
        // SAFETY: the active frame's control block is live.
        let member = unsafe { &*f.ctl }.has(cycle::BUFFERED);
        let (generation, path, episode, seq) = self.key_parts(f, member);
        let key = ReadKey {
            node: f.node.0,
            generation,
            cidx: f.cidx,
            reading,
            ordinal: self.dep_stack.reads_of_current(rt),
            path,
            seq,
            episode,
            foreign: None,
        };
        let above = replay::extend_path(
            path,
            replay::Step {
                node: f.node.0,
                episode,
                reading,
                ordinal: key.ordinal,
                seq,
            },
        );
        let what = replay::take_pending(rt, &key, clone_landed::<T>);
        let Some(what) = what else {
            let slot = handle.slot();
            let resumed = replay::with(|c| {
                let mut resumed = None;
                for x in c.iter_mut().filter(|x| x.rt == rt) {
                    x.direct = None;
                    if let Some(i) = x
                        .pending
                        .find_keyed(&key, |d| !d.taken() && matches!(d.what, Delivered::Direct))
                    {
                        x.pending.set_taken(i, true);
                        resumed = Some(x.pending.get(i).runs.clone());
                        x.direct = Some((slot, handle.generation(), f.cidx));
                    }
                    x.resume = None;
                    if let Some(i) = x.pending.find_keyed(&key, |d| {
                        !d.taken() && matches!(d.what, Delivered::Walk(..))
                    }) {
                        x.pending.set_taken(i, true);
                        let d = x.pending.get(i);
                        resumed = Some(d.runs.clone());
                        if let Delivered::Walk(walk, _) = &d.what {
                            let walk = (**walk).clone();
                            x.resume = Some((reading, f.cidx, walk));
                        }
                    }
                }
                resumed
            });
            replay::start_read(rt, f.cidx, f.depth, above, resumed);
            self.undivert_when_taken(f);
            return None;
        };
        replay::start_read(rt, f.cidx, f.depth, above, None);
        self.undivert_when_taken(f);
        // A read of a held node lowers its reader to the node's anchor,
        // whatever the read returns, as the read rules do.
        if let Some(anchor) = cycle_state::held_anchor(rt, handle.slot()) {
            // SAFETY: the active frame's control block is live.
            unsafe { &*f.ctl }.lower(anchor);
        }
        let (cycle, poison, attributed, payload, nodes) = match what {
            Delivered::Unwind {
                cycle,
                poison,
                attributed,
                payload,
                nodes,
            } => (cycle, poison, attributed, payload, nodes),
            Delivered::Walk(..) | Delivered::Direct => {
                unreachable!("incr-core: a walk's direction is not a value")
            }
            Delivered::Landed(value, poison) => {
                // SAFETY: the active frame's control block is live.
                unsafe { &*f.ctl }.raise(poison | cycle::UNREGENERABLE);
                return Some(Ok(*value
                    .into_inner()
                    .downcast::<T>()
                    .expect("incr-core: a landed value has its node's type")));
            }
        };
        #[cfg(test)]
        self.deliveries.lock().unwrap().push(DeliverySeen {
            frame: f.node.0,
            read: handle.slot(),
            cycle,
            kind,
        });
        // As at the panicking compute's landing: a closure that catches the
        // unwind, or a `try_get` that returns it, has still read a dep that
        // did not land. The native walk computes a Dirty reader's deps
        // before its closure, so that reader never lands Clean over them.
        if poison != 0 {
            // SAFETY: the active frame's control block is live.
            unsafe { &*f.ctl }.raise(poison);
        }
        // A payload no copy can stand for was delivered once, and a retry
        // after it regenerates the payload, or reads the node as it is.
        if cycle {
            // SAFETY: the active frame's control block is live.
            unsafe { &*f.ctl }.raise(cycle::UNREGENERABLE);
        }
        let payload = self.regenerated(handle, rt, &key, payload)?;
        if !replay::copyable(&*payload) {
            self.hold_uncopied(f.ctl, self.regenerable(reading));
        }
        if !cycle {
            // The member marked for the panic was marked in the run it came
            // out of, so the reader's member run is not, as when it nests.
            if attributed {
                cycle_state::count_failure();
            }
            self.note_delivered(nodes, &*payload);
            resume_unwind(payload);
        }
        // A closure that swallowed the cycle and then panicked on its own
        // sent that panic as the cycle unwind's payload, which a `try_get`
        // passes on as an ordinary panic, as `entered` does, unless the read
        // was of a member this thread holds: that read is R5's, from any
        // frame, whose unwind stays the cycle's (`member_run_unwound`). It
        // carries the landings noted for it before the catch, as the native
        // box does.
        let payload = match payload.downcast::<crate::cycle::CycleUnwind>() {
            Ok(unwind) if kind == ReadKind::TryGet => return Some(Err(unwind.0)),
            Ok(unwind) => unwind,
            Err(payload) => {
                self.note_delivered(nodes, &*payload);
                let r5 = cycle_state::held_episode(rt, handle.slot()).is_some();
                if kind == ReadKind::TryGet && !r5 {
                    resume_unwind(payload);
                }
                payload
            }
        };
        cycle_state::set_unwind_kind(Some(UnwindKind::Cycle));
        resume_unwind(payload)
    }

    /// A virtual frame's pop at position `v.cidx` finished the run of the
    /// frame there, which computed `node`. The frame below reads a landing
    /// as an entry's; if `node` is the entry at the driver's `level`, the
    /// pop was that entry's retry, and the entry is done. `Some` when that
    /// entry was the context's target.
    fn frame_finished(
        &self,
        id: u64,
        level: u32,
        entry: Entry,
        v: &VFrame,
        node: u32,
        ended: Ended,
    ) -> Option<Computed> {
        let landed = ended != Ended::Unwound;
        let generation = self.nodes.get(node).generation();
        let (pos, boundary) = replay::with(|c| {
            let x = c
                .iter()
                .find(|x| x.id == id)
                .expect("incr-core: a running context is open");
            (v.cidx - x.boundary, x.boundary)
        });
        if landed && pos > 0 {
            let unsettled = matches!(ended, Ended::Contained | Ended::Unsettled);
            self.hold_landing(id, pos, node, generation, unsettled);
        }
        if pos != level || entry.node != node || entry.generation != generation {
            return None;
        }
        replay::with(|c| {
            if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                x.pop_entry(level);
                x.claims.retain(|&(l, _)| l < level);
            }
        });
        if level == 0 {
            self.close_context(id, boundary);
            if ended == Ended::Contained {
                return Some(Computed::Unsettled);
            }
            // The read that ran the loop reads the target as the native
            // compute's reader does after its landing, which a walk entry's
            // landing leaves unpoisoned.
            if landed {
                self.episode_landed(self.nodes.get(node), entry.walked);
            }
            if ended == Ended::Unsettled {
                return Some(Computed::Unsettled);
            }
            return Some(Computed::Done);
        }
        None
    }

    /// Node `node` at `generation` landed from position `level`. If it
    /// landed Dirty and the frame below was blocked reading it, that
    /// frame's retry reads the slot as it is, as the unbounded run reads
    /// it right after the compute. If an episode landed it with no value,
    /// that read gets the episode's cycle, as the read that ran it would.
    /// A walk stopped short of its root drops the node's other entries if
    /// the compute left it `unsettled`, as `walk_stopped` does.
    fn hold_landing(&self, id: u64, level: u32, node: u32, generation: u32, unsettled: bool) {
        #[cfg(test)]
        if unsettled {
            UNSETTLED_HOLDS.with(|n| n.set(n.get() + 1));
        }
        self.park(ParkPoint::HoldLanding(NodeId(node)));
        let mut state = state::load::<C>(self.nodes.get(node).state_cell());
        if is_computing(state) {
            debug_assert!(
                !cycle_state::holds(self.runtime_id, node),
                "incr-core: a held landing's node is claimed by this thread"
            );
            // A claim made since the landing, of a node this thread
            // reserves, gives back the state the landing left.
            state = self.given_back_state(node, generation).unwrap_or(state);
        }
        let reading = u64::from(node) | (u64::from(generation) << 32);
        let what = match state {
            NodeState::Dirty => self.slot_value(node).map(|v| {
                Delivered::Landed(
                    Unbuilt::new(v),
                    self.dirty_read_poison(self.nodes.get(node)),
                )
            }),
            NodeState::New | NodeState::Cyclic => {
                self.inner
                    .read()
                    .member_recs
                    .get(&node)
                    .map(|g| Delivered::Unwind {
                        cycle: true,
                        attributed: false,
                        // As `episode_landed` poisons a reader of a node
                        // left New, and not of one left Cyclic.
                        poison: if state == NodeState::New {
                            cycle::POISONED
                        } else {
                            0
                        },
                        payload: Box::new(cycle::CycleUnwind(g.cycle.clone())),
                        nodes: replay::Landings::default(),
                    })
            }
            _ => None,
        };
        // Seqlock order, as `fields_under`: what was read is the entry's
        // node's only if a recycle has not bumped the generation since.
        C::fence_acquire();
        let what = what.filter(|_| self.nodes.get(node).generation() == generation);
        // What the contexts don't keep drops after their borrow: a landed
        // value is the user's.
        let unkept = replay::with(|c| {
            let Some(x) = c.iter_mut().find(|x| x.id == id) else {
                return (what, Vec::new());
            };
            let below = x.vstack[level as usize - 1].clone();
            // A walk the doom stopped short of its root goes on in the
            // retry after this compute, as the unbounded run's goes on.
            let key = ReadKey::blocked(&below);
            let walk = x
                .pending
                .find_keyed(&key, |d| matches!(d.what, Delivered::Walk(..)))
                .and_then(|i| match &mut x.pending.get_mut(i).what {
                    Delivered::Walk(walk, pushed) => Some((walk, pushed)),
                    _ => None,
                });
            match (walk, what) {
                (Some((walk, pushed)), what) => {
                    if pushed.take() == Some(node) {
                        walk.ran(node);
                    }
                    if unsettled {
                        walk.unsettled(NodeId(node));
                    }
                    (what, Vec::new())
                }
                (None, Some(what)) if below.reading == reading => {
                    (None, x.deliver_blocked(&below, what))
                }
                (None, what) => (what, Vec::new()),
            }
        });
        drop(unkept);
    }

    /// Closes context `id` and unwinds `payload` on to the read that ran
    /// its loop, which keeps it for its frame's retry, as a compute's
    /// unwind out to its reader is kept.
    fn unwind_out_of(
        &self,
        id: u64,
        boundary: u32,
        kind: Option<UnwindKind>,
        payload: Box<dyn std::any::Any + Send>,
    ) -> ! {
        self.close_unwound(id, boundary, kind);
        self.note_unwound_read(&*payload, kind);
        resume_unwind(payload)
    }

    /// Closes context `id` on an unwind out of its loop, which also ends
    /// the runs whose virtual hits took the claims it lists. A wound out of
    /// an entry above level 0 pops no frame at the positions below it, so
    /// the claims anchored at the context's virtual frames, and those its
    /// closed episodes waited for, are released here.
    fn close_unwound(&self, id: u64, boundary: u32, kind: Option<UnwindKind>) {
        let mut bag = Deferred::new();
        self.release_claim_list(id, 0, kind, &mut bag);
        bag.drain(self);
        self.close_context(id, boundary);
        let mut bag = Deferred::new();
        self.release_held_from(boundary, true, kind, &mut bag);
        bag.drain(self);
    }

    fn close_context(&self, id: u64, boundary: u32) {
        // The frame below the boundary outlives the context, and so do the
        // entries it owns.
        if value_code::thread_word() & value_code::MEMO_BIT != 0 {
            taint_memo::clear_from(boundary + 1);
        }
        drop(replay::close(id));
    }

    /// [`Self::ensure_clean`] for a driver entry, whose walk skips nodes
    /// with virtual frames.
    fn ensure_clean_virt(&self, id: NodeId, gen: u32) -> Computed {
        let word = value_code::checked_word();
        loop {
            let observed = state::load::<C>(self.nodes.get(id.0).state_cell());
            if observed == NodeState::Clean {
                return Computed::Done;
            }
            match self.clean_walk::<VIRT>(id, gen, word, observed) {
                Computed::Replan => {}
                computed => return computed,
            }
        }
    }

    /// Whether a `Direct` delivery sends this read's walk of `id` straight
    /// to its compute.
    #[cold]
    #[inline(never)]
    pub(super) fn take_direct(&self, id: NodeId) -> bool {
        let rt = self.runtime_id;
        let Some((_, at)) = self.dep_stack.active_frame(rt) else {
            return false;
        };
        let key = (id.0, self.nodes.get(id.0).generation(), at);
        replay::with(|c| {
            c.iter_mut()
                .filter(|x| x.rt == rt)
                .any(|x| x.direct.take_if(|d| *d == key).is_some())
        })
    }

    /// The path of the frame at `cidx` if a virtual frame at or above it
    /// owns claims, which only a run along the same path takes over.
    pub(super) fn virtual_path_at(&self, cidx: u32) -> Option<u64> {
        if value_code::thread_word() & value_code::REPLAY_BIT == 0 {
            return None;
        }
        let rt = self.runtime_id;
        cycle_state::with_held(|h| h.any_virtually_owned_from(rt, cidx)).then(|| self.path_at(cidx))
    }

    /// The path and seq of the frame of `node` at `cidx`: its run's, when
    /// one started, and [`Self::path_at`] otherwise.
    pub(super) fn run_path(&self, node: u32, generation: u32, cidx: u32) -> (u64, u32) {
        let (seq, path) = replay::run_seq(self.runtime_id, (node, generation, cidx));
        let path = path.unwrap_or_else(|| self.path_at(cidx));
        debug_assert_eq!(path, self.path_at(cidx), "the run's path at {cidx}");
        (path, seq)
    }

    /// The path of a frame at position `cidx`: the highest virtual frame
    /// below it extended by the frames on the stack between them.
    pub(super) fn path_at(&self, cidx: u32) -> u64 {
        let rt = self.runtime_id;
        let base = replay::peek(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .filter_map(|x| x.vframe_below(cidx))
                .max_by_key(|v| v.cidx)
                .map(|v| {
                    let step = replay::Step::virtual_frame(v);
                    (v.cidx, replay::extend_path(v.path, step))
                })
        });
        let (from, path) = base.map_or((0, 0), |(at, path)| (at + 1, path));
        replay::with_seqs(|seqs| {
            self.dep_stack.with(|s| {
                (s.frames.iter().zip(1..))
                    .filter(|(f, _)| f.rt() == rt && (from..cidx).contains(&f.cidx))
                    .fold(path, |path, (f, depth)| {
                        let mut step = f.step();
                        step.seq = seqs.of(rt, f.ctl, depth);
                        replay::extend_path(path, step)
                    })
            })
        })
    }

    /// The walked watermark the earliest doomed run of the frame of `node`
    /// at `cidx`, along this frame's path, claimed at, for the run that
    /// finishes it.
    #[cold]
    #[inline(never)]
    pub(super) fn take_claim_rev(&self, node: NodeId, cidx: u32) -> Option<u64> {
        self.take_earliest(node, cidx, |x| &mut x.claim_revs)
    }

    /// The thread's failure count when the earliest doomed run of the
    /// member run of `node` at `cidx`, along this frame's path, began, for
    /// the run that finishes it.
    #[cold]
    #[inline(never)]
    pub(super) fn take_run_failures(&self, node: NodeId, cidx: u32) -> Option<u64> {
        self.take_earliest(node, cidx, |x| &mut x.run_failures)
    }

    fn take_earliest(
        &self,
        node: NodeId,
        cidx: u32,
        records: fn(&mut replay::Context) -> &mut replay::EarliestByFrame,
    ) -> Option<u64> {
        let rt = self.runtime_id;
        let generation = self.nodes.get(node.0).generation();
        let frame = (node.0, generation, cidx);
        if !replay::with(|c| {
            c.iter_mut()
                .filter(|x| x.rt == rt)
                .any(|x| records(x).has(frame))
        }) {
            return None;
        }
        let (path, _) = self.run_path(node.0, generation, cidx);
        replay::with(|c| {
            c.iter_mut()
                .filter(|x| x.rt == rt)
                .find_map(|x| records(x).take(frame, path))
        })
    }

    /// The lowlink a popped virtual frame of `node` at `cidx`, along this
    /// frame's path, left for the run that finishes it.
    #[cold]
    #[inline(never)]
    pub(super) fn take_low(&self, node: NodeId, cidx: u32) -> Option<u32> {
        let rt = self.runtime_id;
        let generation = self.nodes.get(node.0).generation();
        let key = |l: &(u32, u32, u32, u64, u32)| l.0 == node.0 && l.1 == generation && l.2 == cidx;
        if !replay::peek(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .any(|x| x.lows.iter().any(key))
        }) {
            return None;
        }
        let (path, _) = self.run_path(node.0, generation, cidx);
        replay::with(|c| {
            c.iter_mut().filter(|x| x.rt == rt).find_map(|x| {
                let at = x.lows.iter().position(|l| key(l) && l.3 == path)?;
                Some(x.lows.remove(at).4)
            })
        })
    }

    /// Whether this read of `root` has a walk to resume (`take_resume`).
    #[cold]
    #[inline(never)]
    pub(super) fn resume_pending(&self, root: NodeId) -> bool {
        let rt = self.runtime_id;
        let Some((_, at)) = self.dep_stack.active_frame(rt) else {
            return false;
        };
        let reading = u64::from(root.0) | (u64::from(self.nodes.get(root.0).generation()) << 32);
        let pending = replay::with(|c| {
            c.iter().filter(|x| x.rt == rt).any(|x| {
                x.resume
                    .as_ref()
                    .is_some_and(|r| r.0 == reading && r.1 == at)
            })
        });
        #[cfg(test)]
        if pending {
            self.resumes_past_held[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        pending
    }

    /// The walk this read of `root` resumes: a frame's retry making the
    /// read whose walk a doom stopped goes on from where it stopped.
    #[cold]
    #[inline(never)]
    pub(super) fn take_resume(&self, root: NodeId) -> Option<Walk> {
        let rt = self.runtime_id;
        let (_, at) = self.dep_stack.active_frame(rt)?;
        let reading = u64::from(root.0) | (u64::from(self.nodes.get(root.0).generation()) << 32);
        let walk = replay::with(|c| {
            c.iter_mut()
                .filter(|x| x.rt == rt)
                .find_map(|x| x.resume.take_if(|r| r.0 == reading && r.1 == at))
                .map(|r| r.2)
        });
        #[cfg(test)]
        if walk.is_some() {
            let observed = state::load::<C>(self.nodes.get(root.0).state_cell());
            if observed == NodeState::Cyclic || super::is_computing(observed) {
                self.resumes_past_held[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        walk
    }

    /// A walk of `root` whose compute of `entry` came back other than
    /// `Done`: `Ok` with the result the walk goes on with, or `Err` with
    /// the one it stops on, its rest stashed for the reading frame's retry.
    /// Takes the walk by reference so the hot loop never moves it.
    #[cold]
    #[inline(never)]
    pub(super) fn walk_stopped(
        &self,
        root: NodeId,
        entry: super::Entry,
        visit: Visit,
        is_root: bool,
        walk: &mut Walk,
        computed: Computed,
    ) -> Result<Computed, Computed> {
        let computed = match computed {
            Computed::Doomed => {
                self.stash_walk(root, (!is_root).then_some(walk), None);
                return Err(computed);
            }
            Computed::Discarded => match self.walk_discarded(visit) {
                Some(after) => after,
                None => {
                    // The discarded compute is a frame the reading frame's
                    // retry runs again.
                    walk.repush(entry);
                    self.stash_walk(root, (!is_root).then_some(walk), Some(entry.id.0));
                    return Err(computed);
                }
            },
            Computed::Skipped(owner) => {
                walk.taint(owner);
                return Ok(Computed::Done);
            }
            _ => computed,
        };
        if computed == Computed::Unsettled {
            if !is_root {
                walk.unsettled(entry.id);
            }
            return Ok(Computed::Done);
        }
        Ok(computed)
    }

    /// A doom stopped the reading frame's walk of `root`: its retry
    /// resumes `walk`, the rest of it, or with none left, takes the root's
    /// landing as the read's outcome.
    fn stash_walk(&self, root: NodeId, walk: Option<&mut Walk>, pushed: Option<u32>) {
        // A doomed frame's walk reads past placeholders, and a speculative
        // run's retry takes no delivery.
        if replay::speculative() {
            return;
        }
        let rt = self.runtime_id;
        let Some((_, at)) = self.dep_stack.active_frame(rt) else {
            return;
        };
        let reading = u64::from(root.0) | (u64::from(self.nodes.get(root.0).generation()) << 32);
        replay::with(|c| {
            for i in (0..c.len()).rev().filter(|&i| c[i].rt == rt) {
                if let Some(v) = c[i]
                    .vstack
                    .iter()
                    .rev()
                    .find(|v| v.cidx == at && v.reading == reading)
                    .cloned()
                {
                    match walk {
                        Some(walk) => {
                            let runs = replay::run_snapshot(c, rt, &v);
                            c[i].deliver_walk(&v, std::mem::take(walk), pushed, runs);
                        }
                        None => {
                            let key = ReadKey::blocked(&v);
                            c[i].pending.remove_walks(&key);
                        }
                    }
                    return;
                }
            }
        });
    }

    /// The taint-memo owner for a walk's skip of `id`, if it has a virtual
    /// frame in a context of this runtime: one more than that frame's
    /// position, as for a skip of a frame on the stack, so that its pop
    /// clears what the skip memoized.
    #[cold]
    #[inline(never)]
    pub(super) fn virtual_owner(&self, id: NodeId) -> Option<u32> {
        self.virtual_position(id.0, self.nodes.get(id.0).generation())
            .map(|p| p + 1)
    }

    /// The taint-memo owner for a walk's compute of `id` whose claim wait
    /// ended with `id` New or Dirty, if `id` has a virtual frame. Never
    /// one: the visit that pushed the compute checked the same frames,
    /// only this thread makes them, and every one it makes while cleaning
    /// the node's deps sits above the walk's frame and pops before the
    /// walk goes on.
    #[cold]
    #[inline(never)]
    pub(super) fn post_wait_owner(&self, id: NodeId) -> Option<u32> {
        let owner = self.virtual_owner(id);
        #[cfg(test)]
        assert!(
            owner.is_none(),
            "incr-core: post-wait claim of a node with a virtual frame: node {}",
            id.0
        );
        owner
    }

    /// Whether position `cidx` holds one of this runtime's virtual frames.
    #[cfg(all(test, not(loom)))]
    pub(super) fn is_virtual_cidx(&self, cidx: u32) -> bool {
        let rt = self.runtime_id;
        replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .any(|x| x.vstack.iter().any(|v| v.cidx == cidx))
        })
    }
}

/// Readers held past the budget because a payload they took can't be
/// regenerated (`hold_uncopied`), for the differential scans to report.
#[cfg(test)]
pub(crate) static PAST_BUDGET_HOLDS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(test, not(loom)))]
thread_local! {
    /// [`PAST_BUDGET_HOLDS`] on this thread, which tests running beside
    /// one another leave alone.
    pub(crate) static THREAD_PAST_BUDGET_HOLDS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    /// Payloads `Runtime::regenerate` made on this thread.
    pub(crate) static THREAD_REGENERATIONS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// The Failed landings of a runtime's nodes from budgeted runs, by node
/// as a `ReadKey`'s `reading`, until each node leaves Failed.
#[derive(Default)]
pub(crate) struct Regens {
    by_node: HashMap<u64, Regenerable>,
    /// The token the next record gets (`Regenerable::token`).
    next_token: u64,
}

pub(crate) fn regens<C: Cells>() -> C::Lock<Regens> {
    <C::Lock<Regens> as crate::locks::Lock<Regens>>::new(Regens::default())
}

/// A node's Failed landing, as `Runtime::regenerate` runs it again.
struct Regenerable {
    /// Which landing wrote the record: a forget that runs after the node's
    /// claim is released removes the record only if it is still this one,
    /// since another thread's run may have landed the node Failed again.
    token: u64,
    /// Whether the node may run again, and so may every node a
    /// regeneration of its payload runs.
    ok: bool,
    /// What its run's reads took (`replay::Took`), the copies under a
    /// mutex only so that the table is `Sync`.
    took: Vec<(u64, Option<SyncPayload>)>,
}

type SyncPayload = std::sync::Mutex<Box<dyn std::any::Any + Send>>;

/// Drops payloads a regeneration does not raise; a user `Drop` that
/// panics there is not the read's panic.
fn drop_payloads(payloads: impl IntoIterator<Item = Box<dyn std::any::Any + Send>>) {
    for p in payloads {
        drop(catch_unwind(AssertUnwindSafe(move || drop(p))));
    }
}

/// Clears a rerun's `REGEN_RUNNING` mark if `rerun` unwinds before it moves
/// the node back.
struct Unmark(RuntimeId, u32);

impl Drop for Unmark {
    fn drop(&mut self) {
        replay::regen_running(self.0, self.1, false);
    }
}

/// The payload of a read that aborts the run `Runtime::regenerate` marks.
struct RegenAbort;

#[cold]
#[inline(never)]
pub(super) fn abort_regen() -> ! {
    value_code::set_bits(value_code::REGEN_ABORTED, true);
    resume_unwind(Box::new(RegenAbort))
}

/// A clone of a landed value of type `T`, as a delivery holds it.
fn clone_landed<T: Value>(value: &(dyn std::any::Any + Send)) -> cycle::Erased {
    Box::new(value_code::run(|| {
        value
            .downcast_ref::<T>()
            .expect("incr-core: a landed value has its node's type")
            .clone()
    }))
}

/// What [`Runtime::adopt_stash`] did with a claimed node.
pub(super) enum Adopted {
    /// No stash stands in for the run: the claim computes.
    No,
    /// The node landed from its stash.
    Landed(Computed),
    /// A dep went stale under the stash: the node landed New for a retry.
    Stale,
    /// The stash is a panic, which the claim's run raises in place of its
    /// closure.
    Panicked(AdoptedPanic),
}

/// A stashed panic a real claim adopts: its payload and the deps its run
/// read before it.
pub(super) struct AdoptedPanic {
    payload: Box<dyn Any + Send>,
    deps: Vec<(NodeId, u32)>,
}

/// How [`Runtime::land_adopted`] ended.
enum Landing {
    Landed(NodeState),
    /// A dep went stale after the run's end; the node landed back New.
    Stale,
    /// Nothing landed: a dep landed back as claimed, a root kept its claim.
    Unlanded,
    /// The root's write panicked, its claim kept for the raise.
    Panicked(AdoptedPanic),
}

/// A node claimed for adoption, with its stash and the next dep to check.
struct Adopting {
    id: NodeId,
    claimed_dirty: bool,
    stash: Stash,
    next: usize,
}

/// A speculative run's landing withheld: the node landed back in the state
/// it was claimed from, and its run's value and deps wait here for
/// speculative reads of the node and for the real claim that adopts them.
pub(super) struct Stash {
    outcome: Outcome,
    deps: Vec<(NodeId, u32)>,
    verify_rev: u64,
    /// A creator's child index as its run left it, unlanded.
    index: Option<StashedIndex>,
}

/// The child index of a creator whose run is stashed: the index's run
/// number, and whether the run reached it. Children the run created stay
/// the claim in progress's until the adoption sweeps or keeps them, or the
/// stash's drop discards them; a later run that opens the index takes them
/// over instead.
#[derive(Copy, Clone)]
struct StashedIndex {
    epoch: u64,
    opened: bool,
}

enum Outcome {
    /// A run's value, and whether it equals the value the node holds.
    Value {
        value: std::sync::Arc<StashedValue>,
        unchanged: bool,
    },
    /// The payload of the run's panic.
    Panic(Unbuilt<dyn Any + Send>),
    /// A verify that found every dep unchanged since `since`, the node's
    /// `verified_at`.
    Verified { since: u64 },
    /// A run still going, or one that kept nothing.
    Tried,
}

/// A stashed run's value, which speculative reads clone with the stash
/// lock released.
struct StashedValue(Unbuilt<dyn Any + Send>);

// SAFETY: a stash holds only a query's buffered value, whose type is a
// `Value`, which is `Sync`.
unsafe impl Sync for StashedValue {}

/// A runtime's stashes, all of one revision, by node and generation.
#[derive(Default)]
pub(crate) struct Stashes {
    revision: u64,
    map: HashMap<(u32, u32), Exclusive<Stash>>,
}

/// A value reached only through `&mut`, so sharing `&Exclusive` across
/// threads shares nothing: the stash's value and payload are `Send` only.
struct Exclusive<T>(T);

// SAFETY: no method takes `&self`, so a shared reference gives no access
// to the inner value; access needs `&mut`, which the stash lock's write
// guard alone provides.
unsafe impl<T: Send> Sync for Exclusive<T> {}

impl<T> Exclusive<T> {
    fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }
}
