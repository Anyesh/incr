//! The Local binding's runtime state: incr-python-persist's [`RtState`]
//! carrying the runtime's [`Lookups`], and the Python callables handed to
//! a Local engine.

use std::borrow::Borrow;
use std::ops::Deref;
use std::sync::Arc;

use incr_python_persist::state::Held;
use pyo3::prelude::*;

use crate::lookups::Lookups;
use crate::persist::LocalBinding;

/// One runtime's binding state: its callables, overlay journal and lookups.
pub(crate) type RtState = incr_python_persist::state::RtState<LocalBinding, Arc<Lookups>>;

/// A Python callable handed to the engine, counted in its runtime's
/// [`RtState`] from creation until drop.
pub(crate) struct PyCallable(Held<LocalBinding, Arc<Lookups>>);

/// The Local parts of [`RtState`].
pub(crate) trait LocalState {
    /// `f` as a callable of this runtime.
    fn callable(&self, f: Py<PyAny>) -> PyCallable;
    fn lookups(&self) -> &Arc<Lookups>;
}

impl LocalState for RtState {
    fn callable(&self, f: Py<PyAny>) -> PyCallable {
        PyCallable(self.hold(f))
    }

    fn lookups(&self) -> &Arc<Lookups> {
        self.extra()
    }
}

impl PyCallable {
    pub(crate) fn state(&self) -> &RtState {
        self.0.state()
    }
}

// SAFETY: the engine's closures must be `Send + Sync`, but these live only
// in a `Runtime<Local>`, which is neither, and in the `unsendable` classes
// of this crate, so they are created, used and dropped on the one thread
// that runtime lives on, and the state's journal of `Local` handles is
// never touched from another. `&PyCallable` reaches the `Py`, which is
// `Sync` itself, and through `state()` the state, which only that thread
// uses for the same reason. The type is private to the Local binding.
unsafe impl Send for PyCallable {}
// SAFETY: see `Send`.
unsafe impl Sync for PyCallable {}

impl Deref for PyCallable {
    type Target = Py<PyAny>;

    fn deref(&self) -> &Py<PyAny> {
        &self.0
    }
}

impl Borrow<Py<PyAny>> for PyCallable {
    fn borrow(&self) -> &Py<PyAny> {
        &self.0
    }
}
