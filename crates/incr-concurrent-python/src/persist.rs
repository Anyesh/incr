//! `incr_concurrent.persist`: durable inputs for a Shared runtime, over any
//! Python object that implements the backend protocol. Codecs run on
//! whichever thread encodes or decodes, so they must be thread-safe.

use std::sync::Arc;

use ::incr_concurrent::Incr;
use incr_python_persist::binding::{CoreStore, Detached, Plain, PyBinding};
use incr_python_persist::classes::StoreRuntime;
use incr_python_persist::incr_persist::Shared;
use incr_python_persist::store::StoreCore;
use incr_python_protocol::PyValue;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::{engine_call, PyNodeId, PyRuntime};

incr_python_persist::persist_exceptions!(Exceptions, incr_concurrent.persist);
incr_python_persist::not_ready_signal!(Signal, incr_concurrent);

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
    type Signal = Signal;
    type Task<T> = Plain<T>;

    fn write<R: Send>(
        py: Python<'_>,
        store: &Arc<CoreStore<Self>>,
        f: impl FnOnce(&Arc<CoreStore<Self>>, Detached) -> R + Send,
    ) -> PyResult<R> {
        engine_call(py, store.runtime(), move || f(store, Detached::assume()))
    }

    fn read<R: Send>(
        py: Python<'_>,
        store: &Arc<CoreStore<Self>>,
        f: impl FnOnce(&Arc<CoreStore<Self>>, Detached) -> R + Send,
    ) -> PyResult<R> {
        engine_call(py, store.runtime(), move || f(store, Detached::assume()))
    }
}

pub(crate) type Core = StoreCore<SharedBinding>;

impl StoreRuntime<SharedBinding> for PyRuntime {
    type Node = PyNodeId;

    fn of_store(core: Arc<Core>) -> Self {
        PyRuntime::of_store(core)
    }

    fn check_store_call(&self) -> PyResult<()> {
        Ok(())
    }

    fn node(inner: Incr<PyValue>) -> PyNodeId {
        PyNodeId { inner }
    }

    fn node_inner(node: &PyNodeId) -> Incr<PyValue> {
        node.inner
    }
}

mod classes {
    use super::SharedBinding;
    use crate::PyRuntime;

    incr_python_persist::persist_classes!(
        minimal,
        binding = SharedBinding,
        runtime = PyRuntime,
        module = "incr_concurrent.persist",
        flags = [frozen]
    );
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
    module.add("TailMoved", py.get_type::<TailMoved>())?;
    module.add_class::<memory::PyMemoryBackend>()?;
    module.add_class::<classes::PyStore>()?;
    incr_python_persist::codec::install_hook::<Exceptions>();
    parent.add("persist", &module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("incr_concurrent.persist", &module)?;
    Ok(())
}
