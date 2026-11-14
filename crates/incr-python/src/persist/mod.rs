//! `incr.persist`: durable inputs and collections for a Local runtime,
//! over any Python object that implements the backend protocol.
//!
//! A [`PyStore`] owns an `incr.Runtime` together with the incr-persist
//! store that keeps its persisted inputs and collections in a backend's
//! log. Values cross to bytes through [`codec`], one user codec per
//! persisted key.

mod batch;
mod memo;
mod memory;
mod store;

use std::sync::Arc;

use incr_core::Local;
use incr_python_persist::binding::{Carry, CoreStore, Detached, PyBinding};
use pyo3::prelude::*;
use pyo3::types::PyModule;

pub(crate) use memo::PyMemoTable;

pub(crate) type StoreCore = incr_python_persist::store::StoreCore<LocalBinding>;
pub(crate) type PyPersistedKeyed = incr_python_persist::store::Keyed<LocalBinding>;
pub(crate) type OverlayScope = incr_python_persist::scope::OverlayScope<LocalBinding>;

incr_python_persist::persist_exceptions!(Exceptions, incr.persist);

/// Local store calls keep the GIL, since the runtime is confined to the
/// opener's thread, and route as `engine_write` and `engine_read` do.
pub(crate) struct LocalBinding;

impl PyBinding for LocalBinding {
    type C = Local;
    type Exceptions = Exceptions;
    type Task<T> = LocalTask<T>;

    fn write<R: Send>(
        _py: Python<'_>,
        store: &Arc<CoreStore<Self>>,
        f: impl FnOnce(&Arc<CoreStore<Self>>, Detached) -> R + Send,
    ) -> PyResult<R> {
        crate::engine_write(store.runtime(), || f(store, Detached::assume()))
    }

    fn read<R: Send>(
        _py: Python<'_>,
        store: &Arc<CoreStore<Self>>,
        f: impl FnOnce(&Arc<CoreStore<Self>>, Detached) -> R + Send,
    ) -> PyResult<R> {
        crate::engine_read(store.runtime(), || f(store, Detached::assume()))
    }
}

/// A Local engine handle carried into or out of a store call.
pub(crate) struct LocalTask<T>(T);

impl<T> Carry<T> for LocalTask<T> {
    fn new(value: T) -> Self {
        LocalTask(value)
    }

    fn into_inner(self) -> T {
        self.0
    }
}

// SAFETY: a `LocalTask` holds `Runtime<Local>` handles, which must stay on
// the runtime's thread. Only `LocalBinding::write` and `LocalBinding::read`
// run the tasks that carry one, and both run them in place on the calling
// thread. The type is private to the Local binding, so no Shared path,
// which may run a task on another thread, can receive one.
unsafe impl<T> Send for LocalTask<T> {}

#[cfg(feature = "conformance")]
mod conformance {
    use super::Exceptions;

    incr_python_persist::conformance_functions!(
        Exceptions,
        incr.persist.conformance,
        "incr.persist.conformance"
    );
}

/// Creates `incr.persist`, adds it to `parent` and registers it in
/// `sys.modules`, so `import incr.persist` and `from incr.persist import X`
/// work although `incr` is a single extension module.
pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let module = PyModule::new(py, "persist")?;
    module.setattr("__name__", "incr.persist")?;
    module.add("PersistError", py.get_type::<PersistError>())?;
    module.add("Busy", py.get_type::<Busy>())?;
    module.add("ReadOnly", py.get_type::<ReadOnly>())?;
    module.add("InvalidCompaction", py.get_type::<InvalidCompaction>())?;
    module.add_class::<memory::PyMemoryBackend>()?;
    module.add_class::<memory::PyMemoryMemo>()?;
    module.add_class::<memory::PyMemoryMemoGuard>()?;
    module.add_class::<memo::PyMemo>()?;
    module.add_class::<memo::PyMemoTable>()?;
    module.add_class::<store::PyStore>()?;
    module.add_class::<batch::PyBatch>()?;
    #[cfg(feature = "conformance")]
    conformance::register_conformance(&module)?;
    incr_python_persist::codec::install_hook::<Exceptions>();
    parent.add("persist", &module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("incr.persist", &module)?;
    Ok(())
}
