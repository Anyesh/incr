//! `Deferred`: the items an engine operation removed and must not drop
//! until it has landed.
//!
//! Dropping a displaced dep list, value, closure or observer can run user
//! code (a `Drop` impl, or any pending reclaim haphazard folds into a
//! retire), and haphazard may run any retired value's `Drop` inside any
//! retire. So no such drop runs under an incr lock or before the owning
//! operation's landing CAS: the operation puts the item in its bag and
//! drains the bag at its exit, after the landing and after any
//! `poison_current_frame()` that exit does. The owner is a `compute_one`
//! call (one bag across its attempts) or a delete.
//!
//! The common items have typed inline slots, so the hot path boxes and
//! allocates nothing: constructing a bag is three null stores and a
//! drain of an empty bag is three null tests. Batch releases push
//! type-erased items into a spill `Vec` (`None` until first use, which
//! allocates nothing) and reserve it per batch, so each batch costs at
//! most one buffer growth.
//!
//! The drain has two modes, chosen by the exit and never by
//! `std::thread::panicking()`. On an exit that returns normally, items
//! drop one at a time; the first panic poisons the reading frame (so an
//! enclosing closure that swallows it cannot land Clean over a node whose
//! cleanup failed) and then resumes as an ordinary panic, after which the
//! remaining items drop by ordinary drop glue as the bag unwinds. On an
//! exit that already carries a panic payload, every item drops under
//! [`isolate`], so a cleanup panic never replaces the carried payload, and
//! the payload then resumes. A user `Drop` that panics while the thread is
//! already unwinding aborts the process, as in std's own containers; the
//! bag's own drop glue runs only when an unwind passes straight through
//! the owner, since every normal and carried exit drains explicitly.

use std::any::Any;
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

pub(crate) struct Deferred<C: Cells> {
    /// The retire token of a displaced overflow dep list. Lock-free
    /// walkers may still be reading the list under a hazard pointer, so
    /// draining retires it; a plain drop of the list would free memory a
    /// walker is iterating.
    dep_list: Option<RetiredDeps<C>>,
    node: Option<NodeDrops<C>>,
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
            node: None,
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

    pub(crate) fn push_node(&mut self, drops: NodeDrops<C>) {
        if self.node.is_none() {
            self.node = Some(drops);
        } else {
            self.push_erased(Box::new(drops));
        }
    }

    /// Reserve room for a batch's `items` erased pushes, so the batch
    /// costs at most one buffer growth.
    pub(crate) fn reserve(&mut self, items: usize) {
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

    /// Normal mode: see the module docs.
    #[inline(always)]
    pub(crate) fn drain(&mut self, rt: &Runtime<C>) {
        self.drain_in(rt, Mode::Normal);
    }

    /// Carried mode: drops every item under [`isolate`], then resumes
    /// `payload` with its kind restored.
    pub(crate) fn drain_carried(&mut self, rt: &Runtime<C>, payload: Box<dyn Any + Send>) -> ! {
        self.drain_in(rt, Mode::Carried);
        resume_unwind(payload)
    }

    #[inline(always)]
    fn drain_in(&mut self, rt: &Runtime<C>, mode: Mode) {
        if let Some(NodeDrops {
            compute,
            observers,
            arena,
            arena_slot,
        }) = self.node.take()
        {
            step(rt, mode, || drop(compute));
            step(rt, mode, || drop(observers));
            step(rt, mode, || {
                arena.release(arena_slot);
                reclaim_probe::reached(RetireSite::DeleteRetire);
            });
        }
        if let Some(token) = self.dep_list.take() {
            step(rt, mode, || retire_dep_list::<C>(token));
        }
        if let Some(spill) = self.spill.as_mut() {
            for item in spill.drain(..) {
                step(rt, mode, || drop(item));
            }
        }
    }
}

#[derive(Copy, Clone)]
enum Mode {
    Normal,
    Carried,
}

fn step<C: Cells>(rt: &Runtime<C>, mode: Mode, step: impl FnOnce()) {
    match mode {
        Mode::Normal => drain_step(rt, step),
        Mode::Carried => isolate(step),
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

/// One item of a normal-mode drain. A panic poisons the reading frame
/// before it resumes, so a parent that swallows it still lands Dirty.
#[inline(never)]
fn drain_step<C: Cells>(rt: &Runtime<C>, step: impl FnOnce()) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(step)) {
        rt.dep_stack.poison_current_frame();
        resume_unwind(payload);
    }
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
