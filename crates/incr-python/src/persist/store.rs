//! `incr.persist.Store`: a runtime whose persisted inputs and collections
//! live in a backend's log. Writes to them go through the store; the
//! runtime and the graph API read them like any other node.

use std::sync::Arc;

use incr_python_persist::keyed::Pending;
use incr_python_persist::memo::MemoCore;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::batch::PyBatch;
use super::memo::{self, PyMemo, PyMemoTable};
use super::{LocalBinding, PyPersistedKeyed, StoreCore};
use crate::overlay::Born;
use crate::{
    key_counts, reconcile_pairs, KeyCounts, PyCollection, PyKeyedCollection, PyNodeId, PyRuntime,
};

/// The version slot of `collection`, which store calls name it by.
pub(super) fn version_slot(collection: &PyCollection) -> u32 {
    collection.inner.version_node().slot()
}

/// The handle of `kc`, which must be a keyed collection `core` persists.
pub(super) fn keyed_of(
    py: Python<'_>,
    core: &Arc<StoreCore>,
    kc: &PyKeyedCollection,
) -> PyResult<PyPersistedKeyed> {
    let runtime = kc.rt.bind(py).borrow();
    kc.persisted_handle()
        .filter(|_| {
            runtime
                .store_core()
                .is_some_and(|owner| Arc::ptr_eq(owner, core))
        })
        .cloned()
        .ok_or_else(|| {
            PyValueError::new_err("collection is not a persisted keyed collection of this store")
        })
}

/// Persisted inputs and collections over a backend, and the runtime that
/// reads them.
///
/// `store.input(key, default, codec)` and `store.collection(key, codec)`
/// declare a key and return its node or `Collection`; `codec` is any object
/// with `encode(obj) -> bytes` and `decode(bytes) -> obj`. Two contracts
/// bind it: `decode(encode(x)) == x`, and `encode` is deterministic (equal
/// values give equal bytes), since fingerprints hash the bytes. A codec
/// need not know `Pending`, `LookupFailed`, `Unavailable` or `Incomplete`:
/// those persist as themselves, and only their `row`, `value`, `key` and
/// `reason` go through the codec. A value written to a slot with no codec
/// raises, naming the slot.
///
/// Persisted nodes are written through the store (`store.set`,
/// `store.insert`, ...); `runtime.set` and a persisted collection's own
/// `insert`, `delete`, `extend` and `replace` raise `RuntimeError`.
#[pyclass(name = "Store", module = "incr.persist", unsendable)]
pub(crate) struct PyStore {
    core: Option<Arc<StoreCore>>,
    pub(super) memo: MemoCore<LocalBinding>,
    pub(super) runtime: Py<PyRuntime>,
}

const _: () = assert!(std::mem::align_of::<PyStore>() <= 16);

impl PyStore {
    pub(super) fn core(&self) -> PyResult<&Arc<StoreCore>> {
        self.core
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("store is closed"))
    }

    /// The core of a store whose runtime is usable: a runtime poisoned by
    /// a failed overlay restore refuses every store call, since a
    /// checkpoint would fold the half-restored collections.
    pub(super) fn core_live(&self, py: Python<'_>) -> PyResult<Arc<StoreCore>> {
        let core = Arc::clone(self.core()?);
        self.runtime.bind(py).borrow().rt()?;
        Ok(core)
    }

    pub(super) fn core_outside_overlay(&self, py: Python<'_>) -> PyResult<Arc<StoreCore>> {
        if self.runtime.bind(py).borrow().state.overlay_active() {
            return Err(PyRuntimeError::new_err(
                "store calls cannot run inside an overlay",
            ));
        }
        self.core_live(py)
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
        #[expect(
            clippy::arc_with_non_send_sync,
            reason = "the shared store core is reached through `Arc` in both packages; a Local one stays on its runtime's thread"
        )]
        let core = Arc::new(StoreCore::open(
            py,
            backend,
            build_id,
            durability,
            auto_checkpoint_records,
        )?);
        let runtime = Py::new(py, PyRuntime::persisted(Arc::clone(&core)))?;
        Ok(PyStore {
            core: Some(core),
            memo: MemoCore::default(),
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
        let core = self.core_outside_overlay(py)?;
        let inner = core.input(py, key, default, codec)?;
        Ok(PyNodeId { inner })
    }

    /// Declares the collection `key`, which starts with its logged elements,
    /// and returns it as a `Collection`. Write it with `store.insert`,
    /// `store.delete`, `store.replace` and `store.extend`.
    fn collection(
        &self,
        py: Python<'_>,
        key: &str,
        codec: Option<Py<PyAny>>,
    ) -> PyResult<PyCollection> {
        let core = self.core_outside_overlay(py)?;
        let persisted = core.collection(py, key, codec)?;
        Ok(PyCollection {
            inner: persisted.handle().clone(),
            rt: self.runtime.clone_ref(py),
            born: Born::default(),
            persisted: true,
        })
    }

    /// Sets a persisted input: the value is logged, then applied.
    fn set(&self, py: Python<'_>, node: PyNodeId, value: Py<PyAny>) -> PyResult<()> {
        let core = self.core_outside_overlay(py)?;
        core.set(py, node.inner, value)
    }

    /// Appends one occurrence of `row` to a persisted collection.
    fn insert(
        &self,
        py: Python<'_>,
        collection: PyRef<'_, PyCollection>,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        let core = self.core_outside_overlay(py)?;
        let persisted = core.collection_of(version_slot(&collection))?;
        core.insert(py, persisted, row)
    }

    /// Removes one occurrence of `row` from a persisted collection. Removing
    /// an absent row is logged and changes nothing.
    fn delete(
        &self,
        py: Python<'_>,
        collection: PyRef<'_, PyCollection>,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        let core = self.core_outside_overlay(py)?;
        let persisted = core.collection_of(version_slot(&collection))?;
        core.delete(py, persisted, row)
    }

    /// Removes one occurrence of `old`, if present, and inserts `new`, as
    /// one logged step.
    fn replace(
        &self,
        py: Python<'_>,
        collection: PyRef<'_, PyCollection>,
        old: Py<PyAny>,
        new: Py<PyAny>,
    ) -> PyResult<()> {
        let core = self.core_outside_overlay(py)?;
        let persisted = core.collection_of(version_slot(&collection))?;
        core.replace(py, persisted, old, new)
    }

    /// Appends every row of `rows` in one logged step.
    fn extend(
        &self,
        py: Python<'_>,
        collection: PyRef<'_, PyCollection>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let core = self.core_outside_overlay(py)?;
        let persisted = core.collection_of(version_slot(&collection))?;
        core.extend(py, persisted, rows)
    }

    /// Declares the keyed collection `key`: a `KeyedCollection` over a
    /// persisted collection of `(key, row)` pairs, holding the pairs the
    /// log restores. `key_codec` and `codec` encode the
    /// keys and the rows, under the contracts of `store.input`. Write it
    /// with `store.upsert`, `store.remove`, `store.reconcile` or a
    /// `store.batch`; its own writes raise. `label` names the collection's
    /// group in `graph_summary`, labelling its pair and row nodes.
    #[pyo3(signature = (key, key_codec, codec, label=None))]
    fn keyed_collection(
        &self,
        py: Python<'_>,
        key: &str,
        key_codec: Py<PyAny>,
        codec: Py<PyAny>,
        label: Option<String>,
    ) -> PyResult<PyKeyedCollection> {
        let core = self.core_outside_overlay(py)?;
        let persisted = core.keyed_collection(py, key, key_codec, codec)?;
        PyKeyedCollection::persisted(self.runtime.bind(py).clone(), persisted, label)
    }

    /// Holds `row` for `key` in a persisted keyed collection, as
    /// `KeyedCollection.upsert` does, logging the change.
    fn upsert(
        &self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        key: Py<PyAny>,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        let core = self.core_outside_overlay(py)?;
        let keyed = keyed_of(py, &core, &kc)?;
        let mut pending = Pending::default();
        pending.upsert(py, &core, &keyed, key, row)?;
        core.write_pending(py, pending)
    }

    /// Removes `key`'s pair from a persisted keyed collection. Raises
    /// `KeyError` if no row is held for it.
    fn remove(
        &self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        key: Py<PyAny>,
    ) -> PyResult<()> {
        let core = self.core_outside_overlay(py)?;
        let keyed = keyed_of(py, &core, &kc)?;
        let mut pending = Pending::default();
        pending.remove(py, &core, &keyed, key)?;
        core.write_pending(py, pending)
    }

    /// Makes a persisted keyed collection hold exactly `rows`, as
    /// `KeyedCollection.reconcile` does, in one logged step. Returns how
    /// many keys were `(inserted, replaced, removed)`.
    fn reconcile(
        &self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<KeyCounts> {
        let core = self.core_outside_overlay(py)?;
        let keyed = keyed_of(py, &core, &kc)?;
        let mut pending = Pending::default();
        pending.reconcile(py, &core, &keyed, reconcile_pairs(rows)?)?;
        let counts = key_counts(pending.counts(&keyed));
        core.write_pending(py, pending)?;
        Ok(counts)
    }

    /// Starts a batch: `with store.batch() as b:` collects `b.set`,
    /// `b.insert`, `b.delete`, `b.replace`, `b.upsert`, `b.remove` and
    /// `b.reconcile`, and on leaving the block appends them all as one
    /// run, so a restore sees all of them or none. A block that raises
    /// writes nothing.
    fn batch(slf: &Bound<'_, Self>) -> PyResult<PyBatch> {
        let py = slf.py();
        slf.try_borrow()?.core_outside_overlay(py)?;
        Ok(PyBatch::new(slf.clone().unbind()))
    }

    /// Declares the memo table `name`, whose entries `func(ctx, key)`
    /// computes, and returns it. `ctx` is the `RuntimeRef` queries receive,
    /// and `ctx.memo(table, key)` reads another entry as a dependency.
    /// `key_codec` and `codec` encode the keys and the values under the
    /// contracts of `store.input`, and `func` must be a pure function of
    /// its key and what it reads through `ctx`. Bump `version` when it
    /// changes meaning: entries are kept only while the name and version
    /// match. Declare every table before `store.open_memo`.
    fn memo_table(
        slf: &Bound<'_, Self>,
        name: &str,
        version: u32,
        key_codec: Py<PyAny>,
        codec: Py<PyAny>,
        func: Py<PyAny>,
    ) -> PyResult<PyMemoTable> {
        memo::declare_table(slf, name, version, key_codec, codec, func)
    }

    /// Opens the memo over `backend`, an object that implements the memo
    /// backend protocol (see `incr.persist.MemoryMemo`), and loads its
    /// current generation. Entries written under another `build_id`, or by
    /// a table of another name, version or type, are ignored.
    fn open_memo(slf: &Bound<'_, Self>, backend: Py<PyAny>) -> PyResult<PyMemo> {
        memo::open(slf, backend)
    }

    /// Applies the records other writers appended since this store last
    /// read the log, across a compaction too, and returns the position it
    /// has applied through.
    fn refresh(&self, py: Python<'_>) -> PyResult<u64> {
        self.core_outside_overlay(py)?.refresh(py)
    }

    /// `position` (the last log position applied), `snapshot_position`
    /// (the snapshot this store last wrote or crossed, 0 if none) and
    /// `records_since_checkpoint`, read without touching the backend.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.core()?.stats(py)?;
        let dict = PyDict::new(py);
        dict.set_item("position", stats.position.0)?;
        dict.set_item("snapshot_position", stats.snapshot_position.0)?;
        dict.set_item("records_since_checkpoint", stats.records_since_checkpoint)?;
        Ok(dict)
    }

    /// Folds the log into a snapshot, compacting it, and returns the
    /// position the snapshot covers.
    fn checkpoint(&self, py: Python<'_>) -> PyResult<u64> {
        self.core_outside_overlay(py)?.checkpoint(py)
    }

    /// Closes the store and its runtime: every later call on either raises.
    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let runtime = slf.try_borrow()?.runtime.clone_ref(py);
        PyRuntime::close_persisted(runtime.bind(py))?;
        let (core, memo) = {
            let mut store = slf.try_borrow_mut()?;
            (store.core.take(), std::mem::take(&mut store.memo))
        };
        drop(memo);
        drop(core);
        Ok(())
    }
}
