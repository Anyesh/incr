//! `incr_concurrent.persist`: durable inputs for a Shared runtime, over any
//! Python object that implements the backend protocol. Codecs run on
//! whichever thread encodes or decodes, so they must be thread-safe.

use std::sync::Arc;

use ::incr_concurrent::{Incr, IncrCollection, Runtime};
use incr_python_persist::binding::{Carry, CoreStore, Detached, Plain, PyBinding};
use incr_python_persist::classes::{FullStoreRuntime, StoreRuntime};
use incr_python_persist::incr_persist::memo::MemoCtx;
use incr_python_persist::incr_persist::Shared;
use incr_python_persist::memo::MemoFn;
use incr_python_persist::ready;
use incr_python_persist::state::RtState;
use incr_python_persist::store::{Keyed, StoreCore};
use incr_python_protocol::{in_runtime, pyerr_boundary, PyValue};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::keyed::PyKeyedCollection;
use crate::{engine_call, with_ref, Exc, PyCollection, PyNodeId, PyRuntime, RtRef};

pub(crate) use classes::PyMemoTable;

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

    fn drop_detached<T>(py: Python<'_>, value: T)
    where
        Plain<T>: Send,
    {
        let value = Plain::new(value);
        py.detach(move || drop(value.into_inner()));
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

impl FullStoreRuntime<SharedBinding> for PyRuntime {
    type Extra = ();
    type Collection = PyCollection;
    type Keyed = PyKeyedCollection;

    fn state(&self) -> &RtState<SharedBinding> {
        &self.state
    }

    fn rt(&self) -> PyResult<&Runtime> {
        Ok(&self.inner)
    }

    fn store_core(&self) -> Option<&Arc<Core>> {
        match &self.inner {
            RtRef::Own(_) => None,
            RtRef::Store(core) => Some(core),
        }
    }

    /// Only a failed overlay restore poisons, and Shared registers no
    /// overlay.
    fn poison(_slf: &Bound<'_, Self>) {}

    /// A Shared store's runtime stays open after the store closes, since
    /// other threads may be reading it.
    fn close_store_runtime(_slf: &Bound<'_, Self>) -> PyResult<()> {
        Ok(())
    }

    fn write_on<R>(rt: &Runtime, f: impl FnOnce() -> R) -> PyResult<R> {
        pyerr_boundary::<Exc, _>(|| in_runtime(rt, f))
    }

    fn read_on<R>(rt: &Runtime, f: impl FnOnce() -> R) -> PyResult<R> {
        pyerr_boundary::<Exc, _>(|| in_runtime(rt, f))
    }

    fn memo_fn(_state: &RtState<SharedBinding>, func: Py<PyAny>) -> MemoFn<Shared> {
        Box::new(move |ctx: &MemoCtx<'_, Shared>, key: &PyValue| -> PyValue {
            let rt = ctx.rt();
            with_ref(rt, |py, rt_ref, missing| {
                let key = key.0.clone_ref(py);
                ready::call_memo::<Signal, _>(py, rt, &func, rt_ref, key, missing)
            })
        })
    }

    fn collection(slf: &Bound<'_, Self>, handle: IncrCollection<PyValue>) -> PyCollection {
        let runtime = slf.get();
        PyCollection {
            inner: handle,
            rt: runtime.inner.clone(),
            bridges: Arc::clone(&runtime.bridges),
        }
    }

    fn version_slot(collection: &PyCollection) -> u32 {
        collection.inner.version_node().slot()
    }

    fn keyed(
        slf: &Bound<'_, Self>,
        persisted: Keyed<SharedBinding>,
        label: Option<String>,
    ) -> PyResult<PyKeyedCollection> {
        PyKeyedCollection::persisted(slf.clone(), persisted, label)
    }

    fn keyed_handle(kc: &PyKeyedCollection) -> (Option<&Keyed<SharedBinding>>, &Py<Self>) {
        (kc.persisted_handle(), &kc.rt)
    }

    fn get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        PyRuntime::get(self, py, node)
    }

    fn get_with_status(&self, py: Python<'_>, node: PyNodeId) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        PyRuntime::get_with_status(self, py, node)
    }

    fn set(&self, node: PyNodeId, value: Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| PyRuntime::set(self, py, node, value))
    }
}

mod classes {
    use super::SharedBinding;
    use crate::PyRuntime;

    incr_python_persist::persist_classes!(
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
    module.add_class::<memory::PyMemoryMemo>()?;
    module.add_class::<memory::PyMemoryMemoGuard>()?;
    module.add_class::<classes::PyStore>()?;
    module.add_class::<classes::PyBatch>()?;
    module.add_class::<classes::PyMemo>()?;
    module.add_class::<PyMemoTable>()?;
    incr_python_persist::codec::install_hook::<Exceptions>();
    parent.add("persist", &module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("incr_concurrent.persist", &module)?;
    Ok(())
}
