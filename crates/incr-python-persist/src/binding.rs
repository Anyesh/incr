//! How a package runs store calls: [`PyBinding`].

use incr_core::Cells;
use incr_persist::Store;
use pyo3::prelude::*;

use crate::backend::PyBackend;
use crate::PersistExceptions;

/// The incr-persist store a package's Python store wraps.
pub type CoreStore<B> = Store<<B as PyBinding>::C, PyBackend<<B as PyBinding>::Exceptions>>;

/// Marks code running where [`PyBinding::write`] put it. It carries no
/// `Python` token: engine-side code, which on Shared runs with the GIL
/// released, reaches Python only through `Python::attach`.
pub struct Detached(());

impl Detached {
    /// Only [`PyBinding::write`] implementations create one.
    #[doc(hidden)]
    pub fn assume() -> Self {
        Detached(())
    }
}

/// What a Python package supplies to the shared persistence layer: its
/// runtime's cells, its exception types, and how a store call runs.
pub trait PyBinding: Sized + 'static {
    type C: Cells;
    type Exceptions: PersistExceptions;

    /// Runs `f`, a store call that may write the runtime, on `store`, with
    /// a fresh [`Detached`]. The package decides whether the GIL stays
    /// held (Shared releases it, since the store mutex and the engine's
    /// locks run Python under them) and how writes route. An engine unwind
    /// or routing failure is raised as the package's exception.
    fn write<R: Send>(
        py: Python<'_>,
        store: &CoreStore<Self>,
        f: impl FnOnce(&CoreStore<Self>, Detached) -> R + Send,
    ) -> PyResult<R>;
}
