//! `incr.persist.Store`: a runtime whose persisted inputs and collections
//! live in a backend's log. Writes to them go through the store; the
//! runtime and the graph API read them like any other node.

use std::cell::{OnceCell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use incr_core::Local;
use incr_persist::memo::{fingerprint, Memo, MemoBuilder};
use incr_persist::{Durability, Input, Options, Store};
use incr_python_protocol::PyValue;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;

use super::backend::PyBackend;
use super::batch::PyBatch;
use super::codec::{PairCodec, Registry, Slot};
use super::keyed::{Following, KeyedBinding, Pending, PyPersistedCollection};
use super::memo::{self, PyMemo, PyMemoTable};
use crate::keyed::KeyCounts;
use crate::overlay::Born;
use crate::{engine_write, PyCollection, PyKeyedCollection, PyNodeId, PyRuntime};

/// What the store's Python objects share: the incr-persist store with the
/// runtime it owns, the codecs of its slots, and the persisted nodes the
/// binding must keep direct writes off.
pub(crate) struct StoreCore {
    pub(super) store: Arc<Store<Local, PyBackend>>,
    pub(super) registry: Rc<Registry>,
    pub(super) build_id: u128,
    pub(super) memo_builder: RefCell<Option<MemoBuilder<Local>>>,
    pub(super) memo: OnceCell<Memo<Local, PyBackend>>,
    inputs: RefCell<HashMap<u32, Input<PyValue>>>,
    collections: RefCell<HashMap<u32, PyPersistedCollection>>,
    pub(super) keyed: RefCell<Vec<Rc<KeyedBinding>>>,
    pub(super) following: RefCell<Option<Following>>,
    pub(super) foreign: RefCell<HashSet<String>>,
}

impl StoreCore {
    pub(crate) fn runtime(&self) -> &incr_compute::Runtime {
        self.store.runtime()
    }

    pub(crate) fn is_persisted_input(&self, slot: u32) -> bool {
        self.inputs.borrow().contains_key(&slot)
    }

    pub(super) fn input_of(&self, node: &PyNodeId) -> PyResult<Input<PyValue>> {
        self.inputs
            .borrow()
            .get(&node.inner.slot())
            .copied()
            .filter(|input| input.handle() == node.inner)
            .ok_or_else(|| PyValueError::new_err("node is not a persisted input of this store"))
    }

    pub(super) fn collection_of(
        &self,
        collection: &PyCollection,
    ) -> PyResult<PyPersistedCollection> {
        self.collections
            .borrow()
            .get(&collection.inner.version_node().slot())
            .cloned()
            .ok_or_else(|| {
                PyValueError::new_err("collection is not a persisted collection of this store")
            })
    }

    fn declare_codec(&self, slot: &Slot, codec: Option<Py<PyAny>>) -> PyResult<bool> {
        match codec {
            Some(codec) => self.registry.register(slot.clone(), codec),
            None if self.registry.is_registered(slot) => Err(PyValueError::new_err(
                "the key is already declared with a codec",
            )),
            None => Ok(false),
        }
    }

    /// Runs a store call, then applies the records it queued for the keyed
    /// collections. A failure of that is the call's own when the call
    /// succeeded, else its cause.
    pub(super) fn run<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&Store<Local, PyBackend>) -> incr_persist::Result<R>,
    ) -> PyResult<R> {
        self.foreign.borrow_mut().clear();
        let result = self.registry.run(py, || f(&self.store));
        let drained = self.registry.scoped(|| self.drain(py));
        match (result, drained) {
            (Ok(value), Ok(())) => Ok(value),
            (Ok(_), Err(drain)) => Err(drain),
            (Err(error), Ok(())) => Err(error),
            (Err(error), Err(drain)) => {
                error.set_cause(py, Some(drain));
                Err(error)
            }
        }
    }
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
    core: Option<Rc<StoreCore>>,
    pub(super) runtime: Py<PyRuntime>,
}

const _: () = assert!(std::mem::align_of::<PyStore>() <= 16);

impl PyStore {
    pub(super) fn core(&self) -> PyResult<&Rc<StoreCore>> {
        self.core
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("store is closed"))
    }

    /// The core of a store whose runtime is usable: a runtime poisoned by
    /// a failed overlay restore refuses every store call, since a
    /// checkpoint would fold the half-restored collections.
    pub(super) fn core_live(&self, py: Python<'_>) -> PyResult<Rc<StoreCore>> {
        let core = Rc::clone(self.core()?);
        self.runtime.bind(py).borrow().rt()?;
        Ok(core)
    }

    pub(super) fn core_outside_overlay(&self, py: Python<'_>) -> PyResult<Rc<StoreCore>> {
        if self.runtime.bind(py).borrow().state.overlay_active() {
            return Err(PyRuntimeError::new_err(
                "store calls cannot run inside an overlay",
            ));
        }
        self.core_live(py)
    }
}

fn build_id_of(build_id: &Bound<'_, PyAny>) -> PyResult<u128> {
    if let Ok(text) = build_id.downcast::<PyString>() {
        return Ok(fingerprint(text.to_str()?.as_bytes()));
    }
    build_id.extract::<u128>()
}

fn durability_of(durability: &str) -> PyResult<Durability> {
    match durability {
        "sync" => Ok(Durability::Sync),
        "buffered" => Ok(Durability::Buffered),
        other => Err(PyValueError::new_err(format!(
            "durability must be 'sync' or 'buffered', not {other:?}"
        ))),
    }
}

#[pymethods]
impl PyStore {
    /// Opens a store over `backend`, replaying its log. `build_id`, an int
    /// or a str, names the code that computes memoized values; incr hashes
    /// a str to an id. `durability` is `"sync"` (a write is durable when it
    /// returns) or `"buffered"`.
    #[staticmethod]
    #[pyo3(signature = (backend, *, build_id, durability="sync"))]
    fn open(
        py: Python<'_>,
        backend: Py<PyAny>,
        build_id: &Bound<'_, PyAny>,
        durability: &str,
    ) -> PyResult<PyStore> {
        let build_id = build_id_of(build_id)?;
        let options = Options {
            durability: durability_of(durability)?,
            ..Options::default()
        };
        let registry = Rc::new(Registry::default());
        let store = registry.run(py, || Store::open(PyBackend::new(backend), options))?;
        #[expect(
            clippy::arc_with_non_send_sync,
            reason = "MemoBuilder::open takes the store as an Arc"
        )]
        let core = Rc::new(StoreCore {
            store: Arc::new(store),
            registry,
            build_id,
            memo_builder: RefCell::default(),
            memo: OnceCell::new(),
            inputs: RefCell::default(),
            collections: RefCell::default(),
            keyed: RefCell::default(),
            following: RefCell::default(),
            foreign: RefCell::default(),
        });
        let runtime = Py::new(py, PyRuntime::persisted(Rc::clone(&core)))?;
        Ok(PyStore {
            core: Some(core),
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
        let slot = Slot::Key(key.to_owned());
        let added = core.declare_codec(&slot, codec)?;
        let declared = engine_write(core.runtime(), || {
            core.run(py, |store| store.input(key, PyValue(default)))
        })
        .and_then(|declared| declared);
        match declared {
            Ok(input) => {
                core.inputs
                    .borrow_mut()
                    .insert(input.handle().slot(), input);
                Ok(PyNodeId {
                    inner: input.handle(),
                })
            }
            Err(e) => {
                if added {
                    core.registry.unregister(&slot);
                }
                Err(e)
            }
        }
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
        let slot = Slot::Key(key.to_owned());
        let added = core.declare_codec(&slot, codec)?;
        let declared = engine_write(core.runtime(), || {
            core.run(py, |store| store.collection::<PyValue>(key))
        })
        .and_then(|declared| declared);
        match declared {
            Ok(persisted) => {
                let inner = persisted.handle().clone();
                core.collections
                    .borrow_mut()
                    .insert(inner.version_node().slot(), persisted);
                Ok(PyCollection {
                    inner,
                    rt: self.runtime.clone_ref(py),
                    born: Born::default(),
                    persisted: true,
                })
            }
            Err(e) => {
                if added {
                    core.registry.unregister(&slot);
                }
                Err(e)
            }
        }
    }

    /// Sets a persisted input: the value is logged, then applied.
    fn set(&self, py: Python<'_>, node: PyNodeId, value: Py<PyAny>) -> PyResult<()> {
        let core = self.core_outside_overlay(py)?;
        let input = core.input_of(&node)?;
        engine_write(core.runtime(), || {
            core.run(py, |store| store.set(input, PyValue(value)).map(drop))
        })?
    }

    /// Appends one occurrence of `row` to a persisted collection.
    fn insert(
        &self,
        py: Python<'_>,
        collection: PyRef<'_, PyCollection>,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        let core = self.core_outside_overlay(py)?;
        let persisted = core.collection_of(&collection)?;
        engine_write(core.runtime(), || {
            core.run(py, |store| persisted.insert(store, PyValue(row)).map(drop))
        })?
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
        let persisted = core.collection_of(&collection)?;
        engine_write(core.runtime(), || {
            core.run(py, |store| persisted.delete(store, &PyValue(row)).map(drop))
        })?
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
        let persisted = core.collection_of(&collection)?;
        engine_write(core.runtime(), || {
            core.run(py, |store| {
                persisted
                    .replace(store, Some(&PyValue(old)), PyValue(new))
                    .map(drop)
            })
        })?
    }

    /// Appends every row of `rows` in one logged step.
    fn extend(
        &self,
        py: Python<'_>,
        collection: PyRef<'_, PyCollection>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let core = self.core_outside_overlay(py)?;
        let persisted = core.collection_of(&collection)?;
        let rows = rows
            .try_iter()?
            .map(|row| row.map(|row| PyValue(row.unbind())))
            .collect::<PyResult<Vec<_>>>()?;
        engine_write(core.runtime(), || {
            core.run(py, |store| {
                store
                    .batch(|batch| {
                        for row in rows {
                            batch.insert(&persisted, row);
                        }
                    })
                    .map(drop)
            })
        })?
    }

    /// Declares the keyed collection `key`: a `KeyedCollection` over a
    /// persisted collection of `(key, row)` pairs, whose index is rebuilt
    /// from the pairs the log restores. `key_codec` and `codec` encode the
    /// keys and the rows, under the contracts of `store.input`. Write it
    /// with `store.upsert`, `store.remove`, `store.reconcile` or a
    /// `store.batch`; its own writes raise.
    fn keyed_collection(
        &self,
        py: Python<'_>,
        key: &str,
        key_codec: Py<PyAny>,
        codec: Py<PyAny>,
    ) -> PyResult<PyKeyedCollection> {
        let core = self.core_outside_overlay(py)?;
        let slot = Slot::Key(key.to_owned());
        if core.registry.is_registered(&slot) {
            return Err(PyValueError::new_err(format!(
                "the key {key:?} is already declared"
            )));
        }
        let pairs = Py::new(py, PairCodec::new(key_codec, codec))?;
        core.registry.register(slot.clone(), pairs.into_any())?;
        let declared = engine_write(core.runtime(), || core.declare_keyed(py, key))
            .and_then(|declared| declared);
        match declared {
            Ok(binding) => PyKeyedCollection::persisted(
                self.runtime.bind(py).clone().into_any().downcast_into()?,
                binding.pairs.handle().clone(),
                Arc::clone(&binding.index),
            ),
            Err(e) => {
                core.registry.unregister(&slot);
                Err(e)
            }
        }
    }

    /// Holds `row` for `key` in a persisted keyed collection, as
    /// `KeyedCollection.upsert` does, logging the change.
    fn upsert(
        &self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        key: Bound<'_, PyAny>,
        row: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let core = self.core_outside_overlay(py)?;
        let mut pending = Pending::default();
        pending.upsert(&core, &kc, &key, &row)?;
        engine_write(core.runtime(), || core.write(py, pending))?
    }

    /// Removes `key`'s pair from a persisted keyed collection. Raises
    /// `KeyError` if no row is held for it.
    fn remove(
        &self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        key: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let core = self.core_outside_overlay(py)?;
        let mut pending = Pending::default();
        pending.remove(&core, &kc, &key)?;
        engine_write(core.runtime(), || core.write(py, pending))?
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
        let mut pending = Pending::default();
        pending.reconcile(&core, &kc, rows)?;
        let counts = pending.counts();
        engine_write(core.runtime(), || core.write(py, pending))??;
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
        let core = self.core_outside_overlay(py)?;
        engine_write(core.runtime(), || {
            core.run(py, |store| store.refresh().map(|at| at.0))
        })?
    }

    /// Folds the log into a snapshot, compacting it, and returns the
    /// position the snapshot covers.
    fn checkpoint(&self, py: Python<'_>) -> PyResult<u64> {
        let core = self.core_outside_overlay(py)?;
        engine_write(core.runtime(), || {
            core.run(py, |store| store.checkpoint().map(|at| at.0))
        })?
    }

    /// Closes the store and its runtime: every later call on either raises.
    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let runtime = slf.try_borrow()?.runtime.clone_ref(py);
        PyRuntime::close_persisted(runtime.bind(py))?;
        let core = slf.try_borrow_mut()?.core.take();
        drop(core);
        Ok(())
    }
}
