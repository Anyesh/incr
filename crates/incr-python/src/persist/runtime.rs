//! How the persistence classes reach a Local runtime.

use std::sync::Arc;

use incr_compute::{Incr, IncrCollection, Runtime};
use incr_python_persist::classes::{FullStoreRuntime, StoreRuntime};
use incr_python_persist::memo::MemoFn;
use incr_python_protocol::PyValue;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use super::{memo, LocalBinding, PyPersistedKeyed, StoreCore};
use crate::lookups::Lookups;
use crate::overlay::Born;
use crate::state::{LocalState, RtState};
use crate::{PyCollection, PyKeyedCollection, PyNodeId, PyRuntime};

impl StoreRuntime<LocalBinding> for PyRuntime {
    type Node = PyNodeId;

    fn of_store(core: Arc<StoreCore>) -> Self {
        PyRuntime::persisted(core)
    }

    fn check_store_call(&self) -> PyResult<()> {
        if self.state.overlay_active() {
            return Err(PyRuntimeError::new_err(
                "store calls cannot run inside an overlay",
            ));
        }
        Ok(())
    }

    fn node(inner: Incr<PyValue>) -> PyNodeId {
        PyNodeId { inner }
    }

    fn node_inner(node: &PyNodeId) -> Incr<PyValue> {
        node.inner
    }
}

impl FullStoreRuntime<LocalBinding> for PyRuntime {
    type Extra = Arc<Lookups>;
    type Collection = PyCollection;
    type Keyed = PyKeyedCollection;

    fn state(&self) -> &RtState {
        &self.state
    }

    fn rt(&self) -> PyResult<&Runtime> {
        PyRuntime::rt(self)
    }

    fn store_core(&self) -> Option<&Arc<StoreCore>> {
        PyRuntime::store_core(self)
    }

    fn poison(slf: &Bound<'_, Self>) {
        slf.borrow_mut().poison();
    }

    fn close_store_runtime(slf: &Bound<'_, Self>) -> PyResult<()> {
        PyRuntime::close_persisted(slf)
    }

    fn write_on<R>(rt: &Runtime, f: impl FnOnce() -> R) -> PyResult<R> {
        crate::engine_write(rt, f)
    }

    fn read_on<R>(rt: &Runtime, f: impl FnOnce() -> R) -> PyResult<R> {
        crate::engine_read(rt, f)
    }

    fn memo_fn(state: &RtState, func: Py<PyAny>) -> MemoFn<incr_core::Local> {
        Box::new(memo::memo_fn(state.callable(func)))
    }

    fn collection(slf: &Bound<'_, Self>, handle: IncrCollection<PyValue>) -> PyCollection {
        PyCollection {
            inner: handle,
            rt: slf.clone().unbind(),
            born: Born::default(),
            persisted: true,
        }
    }

    fn version_slot(collection: &PyCollection) -> u32 {
        collection.inner.version_node().slot()
    }

    fn keyed(
        slf: &Bound<'_, Self>,
        persisted: PyPersistedKeyed,
        label: Option<String>,
    ) -> PyResult<PyKeyedCollection> {
        PyKeyedCollection::persisted(slf.clone(), persisted, label)
    }

    fn keyed_handle(kc: &PyKeyedCollection) -> (Option<&PyPersistedKeyed>, &Py<Self>) {
        (kc.persisted_handle(), &kc.rt)
    }

    fn get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        PyRuntime::get(self, py, node)
    }

    fn get_with_status(&self, py: Python<'_>, node: PyNodeId) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        PyRuntime::get_with_status(self, py, node)
    }

    fn set(&self, node: PyNodeId, value: Py<PyAny>) -> PyResult<()> {
        PyRuntime::set(self, node, value)
    }
}
