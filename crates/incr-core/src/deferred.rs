//! `Deferred` and `DeleteBag`: the items an engine operation removed and
//! must not drop until it has landed.
//!
//! Dropping a displaced dep list, value, closure or observer can run user
//! code (a `Drop` impl, or any pending reclaim haphazard folds into a
//! retire), and haphazard may run any retired value's `Drop` inside any
//! retire. So no such drop runs under an incr lock or before the owning
//! operation's landing CAS: the operation puts the item in its bag and
//! drains the bag at its exit, after the landing and after any
//! `poison_current_frame()` that exit does. The owner is a `compute_one`
//! call (one `Deferred` across its attempts) or a delete (a `DeleteBag`).
//!
//! Each bag holds only what its owner can defer, in typed inline slots,
//! so the hot path boxes and allocates nothing. A compute's `Deferred` is
//! a dep-list slot and a spill pointer, and draining an empty one is two
//! null tests. A delete's bag is built around the first released node's
//! drops, so a single-node delete never tests for an empty node slot.
//! Batch releases push type-erased items into the spill `Vec` (`None`
//! until first use, which allocates nothing) and reserve it per batch, so
//! each batch costs at most one buffer growth.
//!
//! The drain has two modes, chosen by the exit and never by
//! `std::thread::panicking()`. On an exit that returns normally, items
//! drop one at a time under one unwind guard, each leaving the bag only
//! as it drops; the first panic poisons the reading frame (so an
//! enclosing closure that swallows it cannot land Clean over a node whose
//! cleanup failed) and then resumes as an ordinary panic, after which the
//! remaining items drop, in order, by ordinary drop glue as the bag
//! unwinds. On an exit that already carries a panic payload, every item
//! drops under [`isolate`], so a cleanup panic never replaces the carried
//! payload, and the payload then resumes. A user `Drop` that panics while
//! the thread is already unwinding aborts the process, as in std's own
//! containers; the bag's own drop glue runs only when an unwind passes
//! straight through the owner, since every normal and carried exit drains
//! explicitly.

use std::any::Any;
use std::mem;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::Arc;

use crate::arena_registry::ErasedArena;
use crate::cells::Cells;
use crate::dep_stack::DepStack;
use crate::node::RetiredDeps;
use crate::reclaim_probe::{self, RetireSite};
use crate::runtime::{ComputeFn, ObserverEntry, Runtime};

/// A type-erased spill item; dropping it is the deferred step.
pub(crate) type ErasedDrop = Box<dyn Any>;

/// What a delete removes from one node, kept in the arena slot and the
/// containers the delete already built (no new allocation) until the
/// drain. The arena release itself is deferred: the value stays in its
/// slot, and `arena.release(slot)` runs as a drain step, dropping or
/// retiring the value outside the arena lock.
pub(crate) struct NodeDrops<C: Cells> {
    pub(crate) compute: Option<ComputeFn<C>>,
    pub(crate) observers: Vec<ObserverEntry<C>>,
    pub(crate) arena: Arc<dyn ErasedArena<C>>,
    pub(crate) arena_slot: u32,
}

impl<C: Cells> NodeDrops<C> {
    /// The closure, then the observers, then the arena release. Each
    /// part leaves `self` only as it drops, so after a panic the parts
    /// not yet reached drop with `self`, and the release never runs
    /// twice because no drop glue runs it.
    #[inline(always)]
    fn drop_parts(&mut self) {
        drop(self.compute.take());
        drop(mem::take(&mut self.observers));
        self.arena.release(self.arena_slot);
        reclaim_probe::reached(RetireSite::DeleteRetire);
    }
}

/// What a compute defers: the displaced dep list's retire token, and
/// spilled items.
pub(crate) struct Deferred<C: Cells> {
    /// The retire token of a displaced overflow dep list. Lock-free
    /// walkers may still be reading the list under a hazard pointer, so
    /// draining retires it; a plain drop of the list would free memory a
    /// walker is iterating.
    dep_list: Option<RetiredDeps<C>>,
    spill: Option<Vec<ErasedDrop>>,
}

/// Spill buffer growths since the test started, so a test can show a
/// batch costs at most one.
#[cfg(test)]
pub(crate) static SPILL_ALLOCS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
fn count_spill_alloc() {
    SPILL_ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(not(test))]
#[inline(always)]
fn count_spill_alloc() {}

impl<C: Cells> Deferred<C> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            dep_list: None,
            spill: None,
        }
    }

    pub(crate) fn push_dep_list(&mut self, token: RetiredDeps<C>) {
        if self.dep_list.is_none() {
            self.dep_list = Some(token);
        } else {
            self.push_erased(Box::new(RetireOnDrop::<C>(Some(token))));
        }
    }

    /// Reserve room for a batch's `items` erased pushes, so the batch
    /// costs at most one buffer growth.
    fn reserve(&mut self, items: usize) {
        let spill = self.spill_mut();
        if spill.capacity() - spill.len() < items {
            spill.reserve(items);
            count_spill_alloc();
        }
    }

    pub(crate) fn push_erased(&mut self, item: ErasedDrop) {
        let spill = self.spill_mut();
        if spill.len() == spill.capacity() {
            count_spill_alloc();
        }
        spill.push(item);
    }

    fn spill_mut(&mut self) -> &mut Vec<ErasedDrop> {
        self.spill.get_or_insert_with(Vec::new)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.dep_list.is_none() && self.spill.is_none()
    }

    /// Normal mode: see the module docs.
    #[inline(always)]
    pub(crate) fn drain(self, rt: &Runtime<C>) {
        if self.is_empty() {
            mem::forget(self);
        } else {
            self.drain_items(rt);
        }
    }

    #[inline(never)]
    fn drain_items(mut self, rt: &Runtime<C>) {
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| self.drop_items())) {
            self.unwind(rt, payload);
        }
        self.forget_empty();
    }

    /// Drops every item, each leaving the bag only as it drops, and
    /// leaves the bag empty. The spill is reversed so items pop in push
    /// order; [`Self::unwind`] restores the order of what a panic left.
    fn drop_items(&mut self) {
        if let Some(token) = self.dep_list.take() {
            retire_dep_list::<C>(token);
        }
        if let Some(spill) = self.spill.as_mut() {
            spill.reverse();
            while let Some(item) = spill.pop() {
                drop(item);
            }
            self.spill = None;
        }
    }

    /// The normal-mode panic exit: poison the reading frame and resume,
    /// leaving the items not yet dropped to drop with the bag.
    #[cold]
    fn unwind(&mut self, rt: &Runtime<C>, payload: Box<dyn Any + Send>) -> ! {
        if let Some(spill) = self.spill.as_mut() {
            spill.reverse();
        }
        rt.dep_stack.poison_current_frame();
        resume_unwind(payload)
    }

    fn forget_empty(self) {
        debug_assert!(self.dep_list.is_none() && self.spill.is_none());
        mem::forget(self);
    }

    /// Carried mode: drops every item under [`isolate`], then resumes
    /// `payload` with its kind restored.
    #[cold]
    pub(crate) fn drain_carried(&mut self, payload: Box<dyn Any + Send>) -> ! {
        if let Some(token) = self.dep_list.take() {
            isolate(|| retire_dep_list::<C>(token));
        }
        if let Some(spill) = self.spill.take() {
            for item in spill {
                isolate(|| drop(item));
            }
        }
        resume_unwind(payload)
    }
}

/// A delete's bag: the first released node's drops inline, everything
/// after them in `rest`.
pub(crate) struct DeleteBag<C: Cells> {
    node: NodeDrops<C>,
    rest: Deferred<C>,
}

impl<C: Cells> DeleteBag<C> {
    #[inline(always)]
    pub(crate) fn new(node: NodeDrops<C>, dep_list: Option<RetiredDeps<C>>) -> Self {
        Self {
            node,
            rest: Deferred {
                dep_list,
                spill: None,
            },
        }
    }

    /// Reserve for a batch's further `nodes` node pushes and `dep_lists`
    /// token pushes, less the dep-list slot if it is still free.
    pub(crate) fn reserve_batch(&mut self, nodes: usize, dep_lists: usize) {
        let spilled_lists = dep_lists.saturating_sub(usize::from(self.rest.dep_list.is_none()));
        let items = nodes + spilled_lists;
        if items > 0 {
            self.rest.reserve(items);
        }
    }

    pub(crate) fn push_node(&mut self, drops: NodeDrops<C>, dep_list: Option<RetiredDeps<C>>) {
        self.rest.push_erased(Box::new(SpilledNode(drops)));
        if let Some(token) = dep_list {
            self.rest.push_dep_list(token);
        }
    }

    /// Normal mode: see the module docs.
    #[inline(always)]
    pub(crate) fn drain(mut self, rt: &Runtime<C>) {
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
            self.node.drop_parts();
            if !self.rest.is_empty() {
                self.rest.drop_items();
            }
        })) {
            self.rest.unwind(rt, payload);
        }
        let DeleteBag {
            node:
                NodeDrops {
                    compute,
                    observers,
                    arena,
                    ..
                },
            rest,
        } = self;
        // `drop_parts` left these empty, so forgetting them leaks nothing
        // and skips their drop glue.
        debug_assert!(compute.is_none() && observers.capacity() == 0);
        mem::forget(compute);
        mem::forget(observers);
        drop(arena);
        rest.forget_empty();
    }
}

/// A node's drops spilled behind the bag's inline node; its drop runs
/// the inline node's steps in the same order, the arena release included.
struct SpilledNode<C: Cells>(NodeDrops<C>);

impl<C: Cells> Drop for SpilledNode<C> {
    fn drop(&mut self) {
        self.0.drop_parts();
    }
}

/// A second displaced dep list in one bag, spilled as an erased item;
/// its drop is the retire, like the typed slot's.
struct RetireOnDrop<C: Cells>(Option<RetiredDeps<C>>);

impl<C: Cells> Drop for RetireOnDrop<C> {
    fn drop(&mut self) {
        if let Some(token) = self.0.take() {
            retire_dep_list::<C>(token);
        }
    }
}

fn retire_dep_list<C: Cells>(token: RetiredDeps<C>) {
    drop(token);
    reclaim_probe::reached(RetireSite::DepListRetire);
}

/// Runs one cleanup step and swallows its panic, so an exit that already
/// carries a payload delivers that payload unchanged. The secondary
/// payload is forgotten, not dropped, because its own `Drop` could
/// panic again.
///
/// T5's thread-local engine unwind kind is saved before the step and
/// restored after it here, discarding whatever the step set, so a wound
/// in flight still arrives as a wound.
#[inline(never)]
pub(crate) fn isolate(step: impl FnOnce()) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(step)) {
        std::mem::forget(payload);
    }
}
