//! Persisted keyed collections over incr-persist's `PersistedKeyed`, and
//! the writes of a store call or batch, collected in a `PendingWrite`.

use std::rc::Rc;

use incr_core::Local;
use incr_persist::{PendingWrite, PersistedKeyed};
use incr_python_protocol::{PyKey, PyValue};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

use super::store::StoreCore;
use crate::{
    engine_write, key_counts, reconcile_pairs, KeyCounts, PyCollection, PyKeyedCollection, PyNodeId,
};

pub(crate) type PyPersistedKeyed = PersistedKeyed<PyKey, PyValue, Local, PyValue>;

/// The writes of one store call or batch, appended as one run by
/// [`StoreCore::write`]. Keyed writes compose per key, against the rows
/// each key held when first staged.
#[derive(Default)]
pub(super) struct Pending(PendingWrite<Local>);

impl Pending {
    pub(super) fn set(
        &mut self,
        core: &StoreCore,
        node: &PyNodeId,
        value: Py<PyAny>,
    ) -> PyResult<()> {
        self.0.set(core.input_of(node)?, PyValue(value));
        Ok(())
    }

    pub(super) fn insert(
        &mut self,
        core: &StoreCore,
        collection: &PyCollection,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        self.0
            .insert(&core.collection_of(collection)?, PyValue(row));
        Ok(())
    }

    pub(super) fn delete(
        &mut self,
        core: &StoreCore,
        collection: &PyCollection,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        self.0
            .delete(&core.collection_of(collection)?, PyValue(row));
        Ok(())
    }

    pub(super) fn replace(
        &mut self,
        core: &StoreCore,
        collection: &PyCollection,
        old: Py<PyAny>,
        new: Py<PyAny>,
    ) -> PyResult<()> {
        self.0.replace(
            &core.collection_of(collection)?,
            Some(PyValue(old)),
            PyValue(new),
        );
        Ok(())
    }

    pub(super) fn upsert(
        &mut self,
        py: Python<'_>,
        core: &Rc<StoreCore>,
        kc: &PyKeyedCollection,
        key: Py<PyAny>,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        let keyed = core.keyed_of(py, kc)?;
        engine_write(core.runtime(), || {
            self.0.upsert(&keyed, PyKey(key), PyValue(row));
        })
    }

    /// Stages the removal of `key`, raising `KeyError` when it holds no
    /// row as staged so far.
    pub(super) fn remove(
        &mut self,
        py: Python<'_>,
        core: &Rc<StoreCore>,
        kc: &PyKeyedCollection,
        key: Py<PyAny>,
    ) -> PyResult<()> {
        let keyed = core.keyed_of(py, kc)?;
        let target = PyKey(key.clone_ref(py));
        if engine_write(core.runtime(), || self.0.remove(&keyed, &target))? {
            Ok(())
        } else {
            Err(PyKeyError::new_err(key))
        }
    }

    pub(super) fn reconcile(
        &mut self,
        py: Python<'_>,
        core: &Rc<StoreCore>,
        kc: &PyKeyedCollection,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let keyed = core.keyed_of(py, kc)?;
        let rows = reconcile_pairs(rows)?;
        engine_write(core.runtime(), || self.0.reconcile(&keyed, rows))
    }

    /// How many of `kc`'s keys the staged writes give a row they lacked,
    /// give another row, and leave without a row.
    pub(super) fn counts(
        &self,
        py: Python<'_>,
        core: &Rc<StoreCore>,
        kc: &PyKeyedCollection,
    ) -> PyResult<KeyCounts> {
        Ok(key_counts(self.0.counts(&core.keyed_of(py, kc)?)))
    }
}

impl StoreCore {
    /// The handle of `kc`, which must be a keyed collection this store
    /// persists.
    fn keyed_of(
        self: &Rc<Self>,
        py: Python<'_>,
        kc: &PyKeyedCollection,
    ) -> PyResult<PyPersistedKeyed> {
        let runtime = kc.rt.bind(py).borrow();
        kc.persisted_handle()
            .filter(|_| {
                runtime
                    .store_core()
                    .is_some_and(|core| Rc::ptr_eq(core, self))
            })
            .cloned()
            .ok_or_else(|| {
                PyValueError::new_err(
                    "collection is not a persisted keyed collection of this store",
                )
            })
    }

    /// Appends `pending` as one run and applies it.
    pub(super) fn write(&self, py: Python<'_>, pending: Pending) -> PyResult<()> {
        self.run(py, |store| store.write_pending(pending.0).map(drop))
    }
}
