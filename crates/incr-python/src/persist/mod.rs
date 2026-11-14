//! `incr.persist`: durable inputs and collections for a Local runtime,
//! over any Python object that implements the backend protocol.
//!
//! A [`PyStore`] owns an `incr.Runtime` together with the incr-persist
//! store that keeps its persisted inputs and collections in a backend's
//! log. Values cross to bytes through [`codec`], one user codec per
//! persisted key.

mod backend;
mod batch;
mod codec;
#[cfg(feature = "conformance")]
mod conformance;
mod keyed;
mod memo;
mod memo_backend;
mod scope;
mod store;

use incr_persist::Error;
use pyo3::prelude::*;
use pyo3::types::PyModule;

pub(crate) use keyed::PyPersistedKeyed;
pub(crate) use memo::PyMemoTable;
pub(crate) use scope::OverlayScope;
pub(crate) use store::StoreCore;

pyo3::create_exception!(
    incr.persist,
    PersistError,
    pyo3::exceptions::PyRuntimeError,
    "A store or backend failure that no Python exception describes better."
);

pyo3::create_exception!(
    incr.persist,
    Busy,
    PersistError,
    "Another handle holds the memo checkpoint lock or has written a newer generation: the checkpoint wrote nothing and can be retried."
);

pyo3::create_exception!(
    incr.persist,
    ReadOnly,
    PersistError,
    "A write reached a handle opened read-only."
);

pyo3::create_exception!(
    incr.persist,
    InvalidCompaction,
    PersistError,
    "`compact(upto, ..)` was given an `upto` below the snapshot position or beyond the tail; constructed as `InvalidCompaction(upto, snapshot_at, tail)`."
);

/// The Python exception for `error`: the exception a Python backend
/// raised, as itself, or a `PersistError` carrying the whole chain.
pub(crate) fn persist_error(py: Python<'_>, error: Error) -> PyErr {
    if let Error::Backend(source) = &error {
        if let Some(raised) = source.downcast_ref::<backend::PyBackendError>() {
            return raised.err().clone_ref(py);
        }
    }
    match error {
        Error::Busy => return Busy::new_err(error.to_string()),
        Error::ReadOnly => return ReadOnly::new_err(error.to_string()),
        Error::KeyedContention { .. } => {
            return pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
        }
        Error::InvalidCompaction {
            upto,
            snapshot_at,
            tail,
        } => {
            return InvalidCompaction::new_err((upto.0, snapshot_at.0, tail.0));
        }
        _ => {}
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
    module.add("Busy", py.get_type::<Busy>())?;
    module.add("ReadOnly", py.get_type::<ReadOnly>())?;
    module.add("InvalidCompaction", py.get_type::<InvalidCompaction>())?;
    module.add_class::<backend::PyMemoryBackend>()?;
    module.add_class::<memo_backend::PyMemoryMemo>()?;
    module.add_class::<memo_backend::PyMemoryMemoGuard>()?;
    module.add_class::<memo::PyMemo>()?;
    module.add_class::<memo::PyMemoTable>()?;
    module.add_class::<store::PyStore>()?;
    module.add_class::<batch::PyBatch>()?;
    #[cfg(feature = "conformance")]
    conformance::register(&module)?;
    codec::install_hook();
    parent.add("persist", &module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("incr.persist", &module)?;
    Ok(())
}
