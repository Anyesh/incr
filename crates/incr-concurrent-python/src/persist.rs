//! `incr_concurrent.persist`: durable inputs for a Shared runtime, over any
//! Python object that implements the backend protocol. Codecs run on
//! whichever thread encodes or decodes, so they must be thread-safe.

use std::sync::{Arc, Mutex, PoisonError};

use incr_python_persist::binding::{CoreStore, Detached, PyBinding};
use incr_python_persist::incr_persist::Shared;
use incr_python_persist::store::StoreCore;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::{engine_call, PyNodeId, PyRuntime};

incr_python_persist::persist_exceptions!(Exceptions, incr_concurrent.persist);

mod memory {
    use super::Exceptions;

    incr_python_persist::memory_classes!(Exceptions, "incr_concurrent.persist");
}

/// Store calls release the GIL like every engine call here: the store
/// mutex and the engine's locks run Python (backend methods, codecs,
/// value equality) under them, so a thread waiting on one with the GIL
/// held would deadlock against its holder.
pub(crate) struct SharedBinding;

impl PyBinding for SharedBinding {
    type C = Shared;
    type Exceptions = Exceptions;

    fn write<R: Send>(
        py: Python<'_>,
        store: &CoreStore<Self>,
        f: impl FnOnce(&CoreStore<Self>, Detached) -> R + Send,
    ) -> PyResult<R> {
        engine_call(py, store.runtime(), move || f(store, Detached::assume()))
    }
}

pub(crate) type Core = StoreCore<SharedBinding>;

/// Persisted inputs over a backend, and the runtime that reads them.
///
/// `store.input(key, default, codec)` declares a key and returns its node;
/// `codec` is any object with `encode(obj) -> bytes` and
/// `decode(bytes) -> obj`. Two contracts bind it: `decode(encode(x)) == x`,
/// and `encode` is deterministic (equal values give equal bytes), since
/// fingerprints hash the bytes. A codec runs on whichever thread writes or
/// replays, so it must be thread-safe. Write persisted inputs with
/// `store.set`; read them through `store.runtime`.
#[pyclass(name = "Store", module = "incr_concurrent.persist", frozen)]
pub(crate) struct PyStore {
    core: Mutex<Option<Arc<Core>>>,
    runtime: Py<PyRuntime>,
}

impl PyStore {
    fn core(&self) -> PyResult<Arc<Core>> {
        self.core
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("store is closed"))
    }
}

#[pymethods]
impl PyStore {
    /// Opens a store over `backend`, replaying its log. `build_id`, an int
    /// or a str, names the code that computes memoized values; incr hashes
    /// a str to an id. `durability` is `"sync"` (a write is durable when it
    /// returns) or `"buffered"`. With `auto_checkpoint_records=n`, a
    /// write first checkpoints once `n` records were applied since the last
    /// snapshot; a failed checkpoint fails that write and leaves the log
    /// unchanged.
    #[staticmethod]
    #[pyo3(signature = (backend, *, build_id, durability="sync", auto_checkpoint_records=None))]
    fn open(
        py: Python<'_>,
        backend: Py<PyAny>,
        build_id: &Bound<'_, PyAny>,
        durability: &str,
        auto_checkpoint_records: Option<u64>,
    ) -> PyResult<PyStore> {
        let core = Arc::new(Core::open(
            py,
            backend,
            build_id,
            durability,
            auto_checkpoint_records,
        )?);
        let runtime = Py::new(py, PyRuntime::of_store(Arc::clone(&core)))?;
        Ok(PyStore {
            core: Mutex::new(Some(core)),
            runtime,
        })
    }

    /// The runtime that reads the store's nodes. Close the store, not the
    /// runtime.
    #[getter]
    fn runtime(&self, py: Python<'_>) -> Py<PyRuntime> {
        self.runtime.clone_ref(py)
    }

    /// Declares the input `key`, which starts at its logged value or else at
    /// `default`, and returns its node. Write it with `store.set`.
    fn input(
        &self,
        py: Python<'_>,
        key: &str,
        default: Py<PyAny>,
        codec: Option<Py<PyAny>>,
    ) -> PyResult<PyNodeId> {
        let inner = self.core()?.input(py, key, default, codec)?;
        Ok(PyNodeId { inner })
    }

    /// Sets a persisted input: the value is logged, then applied.
    fn set(&self, py: Python<'_>, node: PyNodeId, value: Py<PyAny>) -> PyResult<()> {
        self.core()?.set(py, node.inner, value)
    }

    /// Closes the store: later store calls raise `RuntimeError`. The
    /// runtime keeps the values it holds.
    fn close(&self) {
        let closed = self
            .core
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        drop(closed);
    }
}

/// Creates `incr_concurrent.persist`, adds it to `parent` and registers it
/// in `sys.modules`.
pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let module = PyModule::new(py, "persist")?;
    module.setattr("__name__", "incr_concurrent.persist")?;
    module.add("PersistError", py.get_type::<PersistError>())?;
    module.add("Busy", py.get_type::<Busy>())?;
    module.add("ReadOnly", py.get_type::<ReadOnly>())?;
    module.add("InvalidCompaction", py.get_type::<InvalidCompaction>())?;
    module.add_class::<memory::PyMemoryBackend>()?;
    module.add_class::<PyStore>()?;
    incr_python_persist::codec::install_hook::<Exceptions>();
    parent.add("persist", &module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("incr_concurrent.persist", &module)?;
    Ok(())
}
