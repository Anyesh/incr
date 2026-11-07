//! `incr.persist`: durable inputs and collections for a Local runtime,
//! over any Python object that implements the backend protocol.
//!
//! A [`PyStore`] owns an `incr.Runtime` together with the incr-persist
//! store that keeps its persisted inputs and collections in a backend's
//! log. Values cross to bytes through [`codec`], one user codec per
//! persisted key.

mod backend;
mod codec;
mod store;

use incr_persist::Error;
use pyo3::prelude::*;
use pyo3::types::PyModule;

pub(crate) use store::StoreCore;

pyo3::create_exception!(
    incr.persist,
    PersistError,
    pyo3::exceptions::PyRuntimeError,
    "A store or backend failure that no Python exception describes better."
);

/// The Python exception for `error`: the exception a Python backend
/// raised, as itself, or a `PersistError` carrying the whole chain.
pub(crate) fn persist_error(py: Python<'_>, error: Error) -> PyErr {
    if let Error::Backend(source) = &error {
        if let Some(raised) = source.downcast_ref::<backend::PyBackendError>() {
            return raised.err().clone_ref(py);
        }
    }
    let mut message = error.to_string();
    let mut source = std::error::Error::source(&error);
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    PersistError::new_err(message)
}

/// Creates `incr.persist`, adds it to `parent` and registers it in
/// `sys.modules`, so `import incr.persist` and `from incr.persist import X`
/// work although `incr` is a single extension module.
pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let module = PyModule::new(py, "persist")?;
    module.setattr("__name__", "incr.persist")?;
    module.add("PersistError", py.get_type::<PersistError>())?;
    module.add_class::<backend::PyMemoryBackend>()?;
    module.add_class::<store::PyStore>()?;
    codec::install_hook();
    parent.add("persist", &module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("incr.persist", &module)?;
    Ok(())
}
