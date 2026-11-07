//! `store.batch()`: the writes of a `with` block, appended as one run when
//! the block ends without an exception.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use super::keyed::Pending;
use super::store::PyStore;
use crate::{engine_write, PyCollection, PyKeyedCollection, PyNodeId};

#[derive(PartialEq)]
enum Phase {
    New,
    Open,
    Done,
}

/// The writes collected inside `with store.batch() as b:`.
///
/// Keyed writes read and write a map of staged keys before the committed
/// index, so `b.upsert` twice on one key, or `b.upsert` then `b.remove`,
/// leave exactly the final pair, and `b.remove` raises `KeyError` for a
/// key neither held nor staged. The index takes the staged keys only once
/// the append has landed.
#[pyclass(name = "Batch", module = "incr.persist", unsendable)]
pub(crate) struct PyBatch {
    store: Py<PyStore>,
    phase: Phase,
    pending: Pending,
}

impl PyBatch {
    pub(super) fn new(store: Py<PyStore>) -> Self {
        Self {
            store,
            phase: Phase::New,
            pending: Pending::default(),
        }
    }

    fn open(&self) -> PyResult<()> {
        if self.phase == Phase::Open {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(
                "the batch is not open: use it as `with store.batch() as b`",
            ))
        }
    }
}

#[pymethods]
impl PyBatch {
    fn __enter__(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        {
            let mut batch = slf.try_borrow_mut()?;
            if batch.phase != Phase::New {
                return Err(PyRuntimeError::new_err("a batch can be entered once"));
            }
            batch.phase = Phase::Open;
        }
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
        self.open()?;
        self.phase = Phase::Done;
        let pending = std::mem::take(&mut self.pending);
        if exc_type.is_none() {
            let core = self.store.bind(py).borrow().core_outside_overlay(py)?;
            engine_write(core.runtime(), || core.write(py, pending))??;
        }
        Ok(false)
    }

    fn set(&mut self, py: Python<'_>, node: PyNodeId, value: Py<PyAny>) -> PyResult<()> {
        self.open()?;
        let core = self.store.bind(py).borrow().core()?.clone();
        self.pending.set(&core, &node, value)
    }

    fn insert(
        &mut self,
        py: Python<'_>,
        collection: PyRef<'_, PyCollection>,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        self.open()?;
        let core = self.store.bind(py).borrow().core()?.clone();
        self.pending.insert(&core, &collection, row)
    }

    fn delete(
        &mut self,
        py: Python<'_>,
        collection: PyRef<'_, PyCollection>,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        self.open()?;
        let core = self.store.bind(py).borrow().core()?.clone();
        self.pending.delete(&core, &collection, row)
    }

    fn replace(
        &mut self,
        py: Python<'_>,
        collection: PyRef<'_, PyCollection>,
        old: Py<PyAny>,
        new: Py<PyAny>,
    ) -> PyResult<()> {
        self.open()?;
        let core = self.store.bind(py).borrow().core()?.clone();
        self.pending.replace(&core, &collection, old, new)
    }

    fn upsert(
        &mut self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        key: Bound<'_, PyAny>,
        row: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.open()?;
        let core = self.store.bind(py).borrow().core()?.clone();
        self.pending.upsert(&core, &kc, &key, &row)
    }

    fn remove(
        &mut self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        key: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.open()?;
        let core = self.store.bind(py).borrow().core()?.clone();
        self.pending.remove(&core, &kc, &key)
    }

    fn reconcile(
        &mut self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.open()?;
        let core = self.store.bind(py).borrow().core()?.clone();
        self.pending.reconcile(&core, &kc, rows)
    }
}
