//! The writes of one store call or batch, collected in a `PendingWrite`
//! and appended as one run by [`StoreCore::write_pending`].

use incr_core::{Incr, KeyCounts};
use incr_persist::PendingWrite;
use incr_python_protocol::{PyKey, PyValue};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;

use crate::binding::{carry, Carry, PyBinding};
use crate::store::{Keyed, StoreCore};

/// The staged writes of a store call or batch. Keyed writes compose per
/// key, against the rows each key held when first staged.
pub struct Pending<B: PyBinding>(PendingWrite<B::C>);

impl<B: PyBinding> Default for Pending<B> {
    fn default() -> Self {
        Pending(PendingWrite::default())
    }
}

/// What a keyed staging call reaches: the writes and the collection.
pub struct Staging<'a, B: PyBinding> {
    write: &'a mut PendingWrite<B::C>,
    keyed: &'a Keyed<B>,
}

impl<B: PyBinding> Pending<B> {
    pub(crate) fn into_write(self) -> PendingWrite<B::C> {
        self.0
    }

    /// Stages setting the persisted input `node` of `core`.
    pub fn set(
        &mut self,
        core: &StoreCore<B>,
        node: Incr<PyValue>,
        value: Py<PyAny>,
    ) -> PyResult<()> {
        self.0.set(core.input_of(node)?, PyValue(value));
        Ok(())
    }

    /// Stages appending `row` to the collection whose version node has
    /// slot `version_slot`.
    pub fn insert(
        &mut self,
        core: &StoreCore<B>,
        version_slot: u32,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        self.0
            .insert(&core.collection_of(version_slot)?, PyValue(row));
        Ok(())
    }

    /// Stages removing one occurrence of `row`.
    pub fn delete(
        &mut self,
        core: &StoreCore<B>,
        version_slot: u32,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        self.0
            .delete(&core.collection_of(version_slot)?, PyValue(row));
        Ok(())
    }

    /// Stages replacing one occurrence of `old`, if present, with `new`.
    pub fn replace(
        &mut self,
        core: &StoreCore<B>,
        version_slot: u32,
        old: Py<PyAny>,
        new: Py<PyAny>,
    ) -> PyResult<()> {
        self.0.replace(
            &core.collection_of(version_slot)?,
            Some(PyValue(old)),
            PyValue(new),
        );
        Ok(())
    }

    /// How many of `keyed`'s keys the staged writes give a row they lacked,
    /// give another row, and leave without a row.
    pub fn counts(&self, keyed: &Keyed<B>) -> KeyCounts {
        self.0.counts(keyed)
    }
}

impl<B: PyBinding> Pending<B>
where
    for<'a> B::Task<Staging<'a, B>>: Send,
{
    /// Runs `f` on the writes and `keyed` as an engine write of `core`,
    /// since staging reads `keyed`'s index.
    fn stage<R: Send>(
        &mut self,
        py: Python<'_>,
        core: &StoreCore<B>,
        keyed: &Keyed<B>,
        f: impl FnOnce(&mut PendingWrite<B::C>, &Keyed<B>) -> R + Send,
    ) -> PyResult<R> {
        let staging = carry::<B, _>(Staging {
            write: &mut self.0,
            keyed,
        });
        B::write(py, core.store(), move |_, _| {
            let Staging { write, keyed } = staging.into_inner();
            f(write, keyed)
        })
    }

    /// Stages holding `row` for `key` in `keyed`.
    pub fn upsert(
        &mut self,
        py: Python<'_>,
        core: &StoreCore<B>,
        keyed: &Keyed<B>,
        key: Py<PyAny>,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        self.stage(py, core, keyed, move |write, keyed| {
            write.upsert(keyed, PyKey(key), PyValue(row));
        })
    }

    /// Stages the removal of `key`, raising `KeyError` when it holds no
    /// row as staged so far.
    pub fn remove(
        &mut self,
        py: Python<'_>,
        core: &StoreCore<B>,
        keyed: &Keyed<B>,
        key: Py<PyAny>,
    ) -> PyResult<()> {
        let target = PyKey(key.clone_ref(py));
        if self.stage(py, core, keyed, move |write, keyed| {
            write.remove(keyed, &target)
        })? {
            Ok(())
        } else {
            Err(PyKeyError::new_err(key))
        }
    }

    /// Stages making `keyed` hold exactly `rows`.
    pub fn reconcile(
        &mut self,
        py: Python<'_>,
        core: &StoreCore<B>,
        keyed: &Keyed<B>,
        rows: Vec<(PyKey, PyValue)>,
    ) -> PyResult<()> {
        self.stage(py, core, keyed, move |write, keyed| {
            write.reconcile(keyed, rows)
        })
    }
}
