//! The binding's own state for one runtime, which every Python callable
//! handed to the engine carries, so code running inside a query reaches it.

use std::borrow::Borrow;
use std::cell::Cell;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;

use pyo3::prelude::*;

use crate::lookups::Lookups;

/// One runtime's binding state: how many Python callables its engine holds
/// (one per live [`PyCallable`] made from it), and its lookups.
#[derive(Clone, Default)]
pub(crate) struct RtState(Rc<Inner>);

#[derive(Default)]
struct Inner {
    callables: Cell<usize>,
    lookups: Arc<Lookups>,
}

impl RtState {
    pub(crate) fn hold(&self, f: Py<PyAny>) -> PyCallable {
        self.0.callables.set(self.0.callables.get() + 1);
        PyCallable {
            f,
            state: self.clone(),
        }
    }

    pub(crate) fn callables(&self) -> usize {
        self.0.callables.get()
    }

    pub(crate) fn lookups(&self) -> &Arc<Lookups> {
        &self.0.lookups
    }
}

/// A Python callable handed to the engine, counted in its runtime's
/// [`RtState`] from creation until drop.
pub(crate) struct PyCallable {
    f: Py<PyAny>,
    state: RtState,
}

impl PyCallable {
    pub(crate) fn state(&self) -> &RtState {
        &self.state
    }
}

// SAFETY: the engine's closures must be `Send + Sync`, but these live only
// in a `Runtime<Local>`, which is neither, and in the `unsendable` classes
// of this crate, so they are created, used and dropped on the one thread
// that runtime lives on and the state's `Rc` is never touched from another.
// `&PyCallable` reaches the `Py`, which is `Sync` itself, and through
// `state()` the `Rc`, which only that thread uses for the same reason.
unsafe impl Send for PyCallable {}
// SAFETY: see `Send`.
unsafe impl Sync for PyCallable {}

impl Drop for PyCallable {
    fn drop(&mut self) {
        let count = &self.state.0.callables;
        count.set(count.get() - 1);
    }
}

impl Deref for PyCallable {
    type Target = Py<PyAny>;

    fn deref(&self) -> &Py<PyAny> {
        &self.f
    }
}

impl Borrow<Py<PyAny>> for PyCallable {
    fn borrow(&self) -> &Py<PyAny> {
        &self.f
    }
}
