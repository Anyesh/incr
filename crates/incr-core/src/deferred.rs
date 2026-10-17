//! `Deferred`: the items an engine operation removed and must not drop
//! until it has landed.
//!
//! Dropping a displaced dep list, value, closure or observer can run user
//! code (a `Drop` impl, or any pending reclaim haphazard folds into a
//! retire), and haphazard may run any retired value's `Drop` inside any
//! retire. So no such drop runs under an incr lock or before the owning
//! operation's landing CAS: the operation puts the item in its bag and
//! drains the bag at its exit, after the landing and after any
//! `poison_current_frame()` that exit does.
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
//! bag's own `Drop` runs only when an unwind passes straight through the
//! owner, since every normal and carried exit drains explicitly.

use std::any::Any;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use crate::cells::Cells;
use crate::dep_stack::DepStack;
use crate::node::RetiredDeps;
use crate::reclaim_probe::{self, RetireSite};
use crate::runtime::Runtime;

pub(crate) struct Deferred<C: Cells> {
    /// The retire token of a displaced overflow dep list. Lock-free
    /// walkers may still be reading the list under a hazard pointer, so
    /// draining retires it; a plain drop of the list would free memory a
    /// walker is iterating.
    dep_list: Option<RetiredDeps<C>>,
}

impl<C: Cells> Deferred<C> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self { dep_list: None }
    }

    pub(crate) fn push_dep_list(&mut self, token: RetiredDeps<C>) {
        debug_assert!(
            self.dep_list.is_none(),
            "incr-core: one attempt displaces at most one dep list"
        );
        self.dep_list = Some(token);
    }

    /// Normal mode: see the module docs.
    #[inline(always)]
    pub(crate) fn drain(&mut self, rt: &Runtime<C>) {
        if let Some(token) = self.dep_list.take() {
            drain_step(rt, || retire_dep_list::<C>(token));
        }
    }

    /// Carried mode: drops every item under [`isolate`], then resumes
    /// `payload` with its kind restored.
    pub(crate) fn drain_carried(&mut self, payload: Box<dyn Any + Send>) -> ! {
        if let Some(token) = self.dep_list.take() {
            isolate(|| retire_dep_list::<C>(token));
        }
        resume_unwind(payload)
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
