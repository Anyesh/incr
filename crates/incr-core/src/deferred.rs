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
//! drops, so a single-node delete never tests for an empty node slot, and
//! an owned-tree delete's later members go into a typed buffer reserved
//! per batch, so each batch costs at most one growth and boxes nothing.
//! A compute's rare extra items go into a type-erased spill `Vec` (`None`
//! until first use, which allocates nothing).
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
use crate::children::ChildIndex;
use crate::cycle::ErasedStrategy;
use crate::cycle_state;
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
/// retiring the value outside the arena lock. The arena is borrowed from
/// the runtime, which frees arenas only when it drops.
pub(crate) struct NodeDrops<'rt, C: Cells> {
    /// A creator's child index, whose keys are user values.
    pub(crate) children: Option<Box<ChildIndex<C>>>,
    pub(crate) compute: Option<ComputeFn<C>>,
    pub(crate) strategy: Option<Arc<dyn ErasedStrategy>>,
    pub(crate) observers: Vec<ObserverEntry<C>>,
    pub(crate) arena: &'rt dyn ErasedArena<C>,
    pub(crate) arena_slot: u32,
}

impl<C: Cells> NodeDrops<'_, C> {
    /// The child index, the closure and cycle strategy, then the observers, then the arena release. Each
    /// part leaves `self` only as it drops, so after a panic the parts
    /// not yet reached drop with `self`, and the release never runs
    /// twice because no drop glue runs it.
    #[inline(always)]
    fn drop_parts(&mut self) {
        drop(self.children.take());
        drop(self.compute.take());
        drop(self.strategy.take());
        drop(mem::take(&mut self.observers));
        self.arena.release(self.arena_slot);
        reclaim_probe::reached(RetireSite::DeleteRetire);
    }
}

/// What a compute defers: the displaced dep list's retire token, and
/// spilled items. incr-persist's restore keeps one for the values its
/// failed seeds hand back.
#[doc(hidden)]
pub struct Deferred<C: Cells> {
    /// The retire token of a displaced overflow dep list. Lock-free
    /// walkers may still be reading the list under a hazard pointer, so
    /// draining retires it; a plain drop of the list would free memory a
    /// walker is iterating.
    dep_list: Option<RetiredDeps<C>>,
    spill: Option<Vec<ErasedDrop>>,
}

#[cfg(test)]
thread_local! {
    /// Spill buffer growths on this thread since the test reset it, so a
    /// test can show a batch costs at most one. Per thread because tests
    /// run in parallel and their deletes would add to a shared count.
    pub(crate) static SPILL_ALLOCS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn count_spill_alloc() {
    SPILL_ALLOCS.with(|c| c.set(c.get() + 1));
}

#[cfg(not(test))]
#[inline(always)]
fn count_spill_alloc() {}

impl<C: Cells> Deferred<C> {
    #[inline(always)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            dep_list: None,
            spill: None,
        }
    }

    /// Keeps `value` to drop at the drain.
    pub fn push<T: 'static>(&mut self, value: T) {
        self.push_erased(Box::new(value));
    }

    pub(crate) fn push_dep_list(&mut self, token: RetiredDeps<C>) {
        if self.dep_list.is_none() {
            self.dep_list = Some(token);
        } else {
            self.push_erased(Box::new(RetireOnDrop::<C>(Some(token))));
        }
    }

    /// A node a sweep or rollback released while its creator's claim is
    /// held, dropped at the drain like any spilled item.
    pub(crate) fn push_node(&mut self, node: NodeDrops<'_, C>, dep_list: Option<RetiredDeps<C>>) {
        // SAFETY: the only borrow in `NodeDrops` is the arena, which the
        // runtime frees only when it drops. Every `Deferred` is created,
        // drained and dropped inside a call that borrows that runtime, so
        // the arena outlives the item.
        let node: NodeDrops<'static, C> = unsafe { mem::transmute(node) };
        self.push_erased(Box::new(ReleasedNode {
            node,
            _dep_list: RetireOnDrop(dep_list),
        }));
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
    pub fn drain(self, rt: &Runtime<C>) {
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
        rt.dep_stack.poison_current_frame(rt.runtime_id);
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

/// A delete's bag: the first released node's drops inline, the first
/// displaced dep list in its slot, and every later member in `members`.
/// The drain runs them in that order, which is push order except that
/// the slot's list goes before every later member.
pub(crate) struct DeleteBag<'rt, C: Cells> {
    node: NodeDrops<'rt, C>,
    dep_list: Option<RetiredDeps<C>>,
    members: Option<Vec<SpilledMember<'rt, C>>>,
}

impl<'rt, C: Cells> DeleteBag<'rt, C> {
    #[inline(always)]
    pub(crate) fn new(node: NodeDrops<'rt, C>, dep_list: Option<RetiredDeps<C>>) -> Self {
        Self {
            node,
            dep_list,
            members: None,
        }
    }

    /// Reserve for a batch's further `nodes` node pushes, so the batch
    /// grows the member buffer at most once.
    pub(crate) fn reserve_batch(&mut self, nodes: usize) {
        let members = self.members.get_or_insert_with(Vec::new);
        if members.capacity() - members.len() < nodes {
            members.reserve(nodes);
            count_spill_alloc();
        }
    }

    pub(crate) fn push_node(&mut self, node: NodeDrops<'rt, C>, dep_list: Option<RetiredDeps<C>>) {
        let dep_list = match dep_list {
            Some(token) if self.dep_list.is_none() => {
                self.dep_list = Some(token);
                None
            }
            other => other,
        };
        let members = self.members.get_or_insert_with(Vec::new);
        if members.len() == members.capacity() {
            count_spill_alloc();
        }
        members.push(SpilledMember {
            node,
            _dep_list: RetireOnDrop(dep_list),
        });
    }

    /// Normal mode: see the module docs.
    #[inline(always)]
    pub(crate) fn drain(mut self, rt: &Runtime<C>) {
        if let Some(payload) = rt.drop_buried_tickets() {
            self.drain_after(rt, payload);
        }
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
            self.node.drop_parts();
            if self.dep_list.is_some() || self.members.is_some() {
                self.drop_rest();
            }
        })) {
            self.unwind(rt, payload);
        }
        let DeleteBag {
            node:
                NodeDrops {
                    compute,
                    strategy,
                    observers,
                    ..
                },
            dep_list,
            members,
        } = self;
        // `drop_parts` left these empty, so forgetting them leaks nothing
        // and skips their drop glue.
        debug_assert!(compute.is_none() && strategy.is_none() && observers.capacity() == 0);
        debug_assert!(dep_list.is_none() && members.is_none());
        mem::forget(compute);
        mem::forget(strategy);
        mem::forget(observers);
        mem::forget(dep_list);
        mem::forget(members);
    }

    /// Retires the slot's list, then drops every member, each leaving
    /// the bag only as it drops, and leaves the bag's rest empty. The
    /// members are reversed so they pop in push order; [`Self::unwind`]
    /// restores the order of what a panic left.
    fn drop_rest(&mut self) {
        if let Some(token) = self.dep_list.take() {
            retire_dep_list::<C>(token);
        }
        if let Some(members) = self.members.as_mut() {
            members.reverse();
            while let Some(member) = members.pop() {
                drop(member);
            }
            self.members = None;
        }
    }

    /// A buried ticket's payload panicked as it dropped: the inline node's
    /// steps still run, a second panic among them isolated, and the rest
    /// drops, in push order, with the bag as that panic resumes.
    #[cold]
    fn drain_after(mut self, rt: &Runtime<C>, payload: Box<dyn Any + Send>) -> ! {
        isolate(|| self.node.drop_parts());
        rt.dep_stack.poison_current_frame(rt.runtime_id);
        resume_unwind(payload)
    }

    /// The normal-mode panic exit: poison the reading frame and resume,
    /// leaving the items not yet dropped to drop, in order, with the bag.
    /// The tickets the delete buried drop first, a panic of theirs
    /// isolated, so no later delete raises it.
    #[cold]
    fn unwind(&mut self, rt: &Runtime<C>, payload: Box<dyn Any + Send>) -> ! {
        if let Some(secondary) = rt.drop_buried_tickets() {
            mem::forget(secondary);
        }
        if let Some(members) = self.members.as_mut() {
            members.reverse();
        }
        rt.dep_stack.poison_current_frame(rt.runtime_id);
        resume_unwind(payload)
    }
}

/// A member spilled behind the bag's inline node, with the dep list it
/// displaced unless that took the bag's slot. Its drop runs the inline
/// node's steps in the same order, the arena release included, and the
/// list retires after them as a field, even when a step panics.
struct SpilledMember<'rt, C: Cells> {
    node: NodeDrops<'rt, C>,
    _dep_list: RetireOnDrop<C>,
}

impl<C: Cells> Drop for SpilledMember<'_, C> {
    fn drop(&mut self) {
        self.node.drop_parts();
    }
}

/// A node released into a compute's bag: its drop runs the same steps
/// as a delete bag's member.
struct ReleasedNode<C: Cells> {
    node: NodeDrops<'static, C>,
    _dep_list: RetireOnDrop<C>,
}

impl<C: Cells> Drop for ReleasedNode<C> {
    fn drop(&mut self) {
        self.node.drop_parts();
    }
}

/// A displaced dep list behind a bag's typed slot; its drop is the
/// retire, like the slot's.
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

/// A value only a speculative or discarded run made, which the unbounded
/// run never builds, or builds again on the retry: its `Drop` is no event
/// of that run's, so it drops under [`isolate`] wherever it drops.
pub(crate) struct Unbuilt<V: ?Sized = dyn Any>(Option<Box<V>>);

impl<V: ?Sized> Unbuilt<V> {
    pub(crate) fn new(value: Box<V>) -> Self {
        Self(Some(value))
    }

    pub(crate) fn value(&self) -> &V {
        self.0
            .as_deref()
            .expect("an unbuilt value is held until it drops")
    }

    /// The value, for a read that delivers it: from then on it is built.
    pub(crate) fn into_inner(mut self) -> Box<V> {
        self.0
            .take()
            .expect("an unbuilt value is held until it drops")
    }
}

impl<V: ?Sized> Drop for Unbuilt<V> {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            #[cfg(all(test, not(loom)))]
            crate::replay::note_drop();
            isolate(|| drop(value));
        }
    }
}

/// Runs one cleanup step and swallows its panic, so an exit that already
/// carries a payload delivers that payload unchanged. The secondary
/// payload is dropped under its own catch. If that drop panics too, its
/// payload is dropped when it is a message and forgotten otherwise, since
/// its own `Drop` could panic again.
///
/// The thread's engine unwind kind is saved before the step and
/// restored after it here, discarding whatever the step set, so a wound
/// in flight still arrives as a wound.
#[inline(never)]
pub(crate) fn isolate(step: impl FnOnce()) {
    let kind = cycle_state::unwind_kind();
    if let Err(payload) = catch_unwind(AssertUnwindSafe(step)) {
        if let Err(again) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
            if again.is::<&'static str>() || again.is::<String>() {
                drop(again);
            } else {
                std::mem::forget(again);
            }
        }
    }
    cycle_state::set_unwind_kind(kind);
}
