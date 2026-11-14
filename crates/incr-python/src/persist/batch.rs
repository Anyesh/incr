//! `store.batch()`: the writes of a `with` block, appended as one run when
//! the block ends without an exception.

use incr_python_persist::batch::Batch;
use pyo3::prelude::*;

use super::store::{keyed_of, version_slot, PyStore};
use super::LocalBinding;
use crate::{reconcile_pairs, PyCollection, PyKeyedCollection, PyNodeId};

/// The writes collected inside `with store.batch() as b:`.
///
/// Keyed writes compose per key against the rows each key held when first
/// staged, so `b.upsert` twice on one key, or `b.upsert` then `b.remove`,
/// leave exactly the final pair, and `b.remove` raises `KeyError` for a
/// key neither held nor staged. The collection takes the staged keys only
/// once the append has landed.
#[pyclass(name = "Batch", module = "incr.persist", unsendable)]
pub(crate) struct PyBatch {
    store: Py<PyStore>,
    batch: Batch<LocalBinding>,
}

impl PyBatch {
    pub(super) fn new(store: Py<PyStore>) -> Self {
        Self {
            store,
            batch: Batch::default(),
        }
    }
}

#[pymethods]
impl PyBatch {
    fn __enter__(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        slf.try_borrow_mut()?.batch.enter()?;
        Ok(slf)
    }

    #[pyo3(signature = (exc_type, _exc, _traceback))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let pending = self.batch.exit()?;
        if exc_type.is_none() {
            let core = self.store.bind(py).borrow().core_outside_overlay(py)?;
            core.write_pending(py, pending)?;
        }
        Ok(false)
    }

    fn set(&mut self, py: Python<'_>, node: PyNodeId, value: Py<PyAny>) -> PyResult<()> {
        let pending = self.batch.pending()?;
        let core = self.store.bind(py).borrow().core_outside_overlay(py)?;
        pending.set(&core, node.inner, value)
    }

    fn insert(
        &mut self,
        py: Python<'_>,
        collection: PyRef<'_, PyCollection>,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        let pending = self.batch.pending()?;
        let core = self.store.bind(py).borrow().core_outside_overlay(py)?;
        pending.insert(&core, version_slot(&collection), row)
    }

    fn delete(
        &mut self,
        py: Python<'_>,
        collection: PyRef<'_, PyCollection>,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        let pending = self.batch.pending()?;
        let core = self.store.bind(py).borrow().core_outside_overlay(py)?;
        pending.delete(&core, version_slot(&collection), row)
    }

    fn replace(
        &mut self,
        py: Python<'_>,
        collection: PyRef<'_, PyCollection>,
        old: Py<PyAny>,
        new: Py<PyAny>,
    ) -> PyResult<()> {
        let pending = self.batch.pending()?;
        let core = self.store.bind(py).borrow().core_outside_overlay(py)?;
        pending.replace(&core, version_slot(&collection), old, new)
    }

    fn upsert(
        &mut self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        key: Py<PyAny>,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        let pending = self.batch.pending()?;
        let core = self.store.bind(py).borrow().core_outside_overlay(py)?;
        pending.upsert(py, &core, &keyed_of(py, &core, &kc)?, key, row)
    }

    fn remove(
        &mut self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        key: Py<PyAny>,
    ) -> PyResult<()> {
        let pending = self.batch.pending()?;
        let core = self.store.bind(py).borrow().core_outside_overlay(py)?;
        pending.remove(py, &core, &keyed_of(py, &core, &kc)?, key)
    }

    fn reconcile(
        &mut self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let pending = self.batch.pending()?;
        let core = self.store.bind(py).borrow().core_outside_overlay(py)?;
        let keyed = keyed_of(py, &core, &kc)?;
        pending.reconcile(py, &core, &keyed, reconcile_pairs(rows)?)
    }
}
