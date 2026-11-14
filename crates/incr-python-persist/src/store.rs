//! The state a package's `persist.Store` class wraps: the incr-persist
//! store with the runtime it owns, the codecs of its slots, and its
//! persisted inputs.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;

use incr_core::Incr;
use incr_persist::memo::fingerprint;
use incr_persist::{
    Durability, Input, Options, PendingWrite, PersistedCollection, PersistedKeyed, Store,
    StoreStats,
};
use incr_python_protocol::{PyKey, PyValue};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::backend::PyBackend;
use crate::binding::{carry, Carry, CoreStore, Detached, PyBinding};
use crate::codec::{PairCodec, Registry, Slot};
use crate::keyed::Pending;

/// A persisted collection of a package's store.
pub type Collection<B> = PersistedCollection<PyValue, <B as PyBinding>::C>;

/// A persisted keyed collection of a package's store, whose pairs are
/// `(key, row)` tuples.
pub type Keyed<B> = PersistedKeyed<PyKey, PyValue, <B as PyBinding>::C, PyValue>;

/// The `build_id` a store is opened with: an int, or a str incr hashes.
pub fn build_id_of(build_id: &Bound<'_, PyAny>) -> PyResult<u128> {
    if let Ok(text) = build_id.downcast::<PyString>() {
        return Ok(fingerprint(text.to_str()?.as_bytes()));
    }
    build_id.extract::<u128>()
}

/// `"sync"` or `"buffered"` as a [`Durability`]; `ValueError` otherwise.
pub fn durability_of(durability: &str) -> PyResult<Durability> {
    match durability {
        "sync" => Ok(Durability::Sync),
        "buffered" => Ok(Durability::Buffered),
        other => Err(PyValueError::new_err(format!(
            "durability must be 'sync' or 'buffered', not {other:?}"
        ))),
    }
}

/// `auto_checkpoint_records` as the store option; `ValueError` for zero.
pub fn auto_checkpoint_of(records: Option<u64>) -> PyResult<Option<u64>> {
    if records == Some(0) {
        return Err(PyValueError::new_err(
            "auto_checkpoint_records must be a positive count or None",
        ));
    }
    Ok(records)
}

/// A store, its codecs and its persisted inputs and collections, shareable across threads when `B::C` is. The input and collection
/// tables are leaf locks under which no Python runs.
pub struct StoreCore<B: PyBinding> {
    store: Arc<CoreStore<B>>,
    registry: Arc<Registry<B::Exceptions>>,
    build_id: u128,
    inputs: Mutex<HashMap<u32, Input<PyValue>>>,
    collections: Mutex<HashMap<u32, Collection<B>>>,
    /// The labelled keyed collections by key, with their labels. A leaf
    /// lock.
    labelled: Mutex<HashMap<String, (Keyed<B>, String)>>,
    declaring: DeclareLock,
}

/// Runs one declaration at a time, so a failed one unregisters only a
/// codec no concurrent declaration of the same slot relies on. Waited on
/// with the GIL released, since a declaration runs Python under the store
/// lock.
#[derive(Default)]
struct DeclareLock {
    holder: Mutex<Option<ThreadId>>,
    freed: Condvar,
}

/// The declaration lock, held until dropped.
pub struct Declaring<'a>(&'a DeclareLock);

impl Drop for Declaring<'_> {
    fn drop(&mut self) {
        *self.0.holder.lock().unwrap_or_else(PoisonError::into_inner) = None;
        self.0.freed.notify_one();
    }
}

impl<B: PyBinding> StoreCore<B> {
    /// Opens a store over the Python `backend`, replaying its log.
    pub fn open(
        py: Python<'_>,
        backend: Py<PyAny>,
        build_id: &Bound<'_, PyAny>,
        durability: &str,
        auto_checkpoint_records: Option<u64>,
    ) -> PyResult<Self> {
        let build_id = build_id_of(build_id)?;
        let registry = Arc::new(Registry::default());
        let options = Options {
            durability: durability_of(durability)?,
            auto_checkpoint_records: auto_checkpoint_of(auto_checkpoint_records)?,
            value_scope: Some(registry.value_scope()),
        };
        let backend = PyBackend::new(py, backend)?;
        let store = registry.run(py, || Store::open(backend, options))?;
        Ok(StoreCore {
            store: Arc::new(store),
            registry,
            build_id,
            inputs: Mutex::default(),
            collections: Mutex::default(),
            labelled: Mutex::default(),
            declaring: DeclareLock::default(),
        })
    }

    pub fn runtime(&self) -> &incr_core::Runtime<B::C> {
        self.store.runtime()
    }

    pub fn store(&self) -> &Arc<CoreStore<B>> {
        &self.store
    }

    pub fn registry(&self) -> &Arc<Registry<B::Exceptions>> {
        &self.registry
    }

    pub fn is_persisted_input(&self, slot: u32) -> bool {
        self.inputs().contains_key(&slot)
    }

    /// The id naming the code that computes memoized values.
    pub fn build_id(&self) -> u128 {
        self.build_id
    }

    /// Takes the lock every declaration of this store holds from its
    /// first registry check until it has registered or unregistered its
    /// codecs. `RuntimeError` when this thread holds it already, as a
    /// codec that declares from inside a declaration would.
    pub fn declare_lock(&self, py: Python<'_>) -> PyResult<Declaring<'_>> {
        let lock = &self.declaring;
        let me = std::thread::current().id();
        let taken = py.detach(|| {
            let mut holder = lock.holder.lock().unwrap_or_else(PoisonError::into_inner);
            loop {
                match *holder {
                    None => {
                        *holder = Some(me);
                        return true;
                    }
                    Some(thread) if thread == me => return false,
                    Some(_) => {
                        holder = lock
                            .freed
                            .wait(holder)
                            .unwrap_or_else(PoisonError::into_inner);
                    }
                }
            }
        });
        if !taken {
            return Err(PyRuntimeError::new_err(
                "a declaration on this store is already running on this thread",
            ));
        }
        Ok(Declaring(lock))
    }

    /// Declares the input `key` with `codec` and returns its node, which
    /// starts at its logged value or else at `default`. Without a codec
    /// the key must already have one.
    pub fn input(
        &self,
        py: Python<'_>,
        key: &str,
        default: Py<PyAny>,
        codec: Option<Py<PyAny>>,
    ) -> PyResult<Incr<PyValue>> {
        let _declaring = self.declare_lock(py)?;
        let slot = Slot::Key(key.to_owned());
        let added = self.declare_codec(&slot, codec)?;
        let default = PyValue(default);
        match self.write(py, move |store, _| store.input(key, default)) {
            Ok(input) => {
                self.inputs().insert(input.handle().slot(), input);
                Ok(input.handle())
            }
            Err(error) => {
                if added {
                    self.registry.unregister(&slot);
                }
                Err(error)
            }
        }
    }

    /// Sets the persisted input `node`: the value is logged, then applied.
    pub fn set(&self, py: Python<'_>, node: Incr<PyValue>, value: Py<PyAny>) -> PyResult<()> {
        let input = self.input_of(node)?;
        let value = PyValue(value);
        self.write(py, move |store, _| store.set(input, value).map(drop))
    }

    fn inputs(&self) -> MutexGuard<'_, HashMap<u32, Input<PyValue>>> {
        self.inputs.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn collections(&self) -> MutexGuard<'_, HashMap<u32, Collection<B>>> {
        self.collections
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// The persisted collection whose version node has slot
    /// `version_slot`, if this store declared it.
    pub fn collection_of(&self, version_slot: u32) -> PyResult<Collection<B>> {
        self.collections()
            .get(&version_slot)
            .cloned()
            .ok_or_else(|| {
                PyValueError::new_err("collection is not a persisted collection of this store")
            })
    }

    /// Applies the records other writers appended since this store last
    /// read the log, and returns the position it has applied through.
    pub fn refresh(&self, py: Python<'_>) -> PyResult<u64> {
        self.write(py, |store, _| store.refresh().map(|at| at.0))
    }

    /// Folds the log into a snapshot and returns the position it covers.
    pub fn checkpoint(&self, py: Python<'_>) -> PyResult<u64> {
        self.write(py, |store, _| store.checkpoint().map(|at| at.0))
    }

    /// The store's positions, read without touching the backend. It goes
    /// through `B::write`, which routes nothing, so a stored routing
    /// failure is left for the next read.
    pub fn stats(&self, py: Python<'_>) -> PyResult<StoreStats> {
        self.write(py, |store, _| store.stats())
    }

    /// The input `node` is, if it is a persisted input of this store.
    pub fn input_of(&self, node: Incr<PyValue>) -> PyResult<Input<PyValue>> {
        self.inputs()
            .get(&node.slot())
            .copied()
            .filter(|input| input.handle() == node)
            .ok_or_else(|| PyValueError::new_err("node is not a persisted input of this store"))
    }

    /// Registers `codec` for `slot`; returns whether it is new. Without a
    /// codec, the slot must not have one yet.
    pub fn declare_codec(&self, slot: &Slot, codec: Option<Py<PyAny>>) -> PyResult<bool> {
        match codec {
            Some(codec) => self.registry.register(slot.clone(), codec),
            None if self.registry.is_registered(slot) => Err(PyValueError::new_err(
                "the key is already declared with a codec",
            )),
            None => Ok(false),
        }
    }

    /// Runs a store call through `B::write` with the store's codecs active,
    /// raising a codec's exception as itself.
    pub fn write<R: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&Arc<CoreStore<B>>, Detached) -> incr_persist::Result<R> + Send,
    ) -> PyResult<R> {
        drop(self.registry.take_failure());
        let registry = &self.registry;
        let result = B::write(py, &self.store, move |store, detached| {
            registry.scoped(|| f(store, detached))
        })?;
        self.registry.finish(py, result)
    }

    /// Runs `declare` with `slot`'s codec registered, unregistering it
    /// when `declare` fails and the registration was new.
    fn declaring<R>(
        &self,
        slot: &Slot,
        added: bool,
        declare: impl FnOnce() -> PyResult<R>,
    ) -> PyResult<R> {
        let declared = declare();
        if declared.is_err() && added {
            self.registry.unregister(slot);
        }
        declared
    }
}

impl<B: PyBinding> StoreCore<B>
where
    B::Task<Collection<B>>: Send,
{
    /// Declares the collection `key` with `codec` and returns it, holding
    /// its logged elements. Without a codec the key must already have one.
    pub fn collection(
        &self,
        py: Python<'_>,
        key: &str,
        codec: Option<Py<PyAny>>,
    ) -> PyResult<Collection<B>> {
        let _declaring = self.declare_lock(py)?;
        let slot = Slot::Key(key.to_owned());
        let added = self.declare_codec(&slot, codec)?;
        let persisted = self.declaring(&slot, added, || {
            self.write(py, move |store, _| {
                store.collection::<PyValue>(key).map(carry::<B, _>)
            })
        })?;
        let persisted = persisted.into_inner();
        self.collections()
            .insert(persisted.handle().version_node().slot(), persisted.clone());
        Ok(persisted)
    }

    /// Appends one occurrence of `row` to `persisted`.
    pub fn insert(&self, py: Python<'_>, persisted: Collection<B>, row: Py<PyAny>) -> PyResult<()> {
        let persisted = carry::<B, _>(persisted);
        self.write(py, move |store, _| {
            persisted.into_inner().insert(store, PyValue(row)).map(drop)
        })
    }

    /// Removes one occurrence of `row` from `persisted`; an absent row is
    /// logged and changes nothing.
    pub fn delete(&self, py: Python<'_>, persisted: Collection<B>, row: Py<PyAny>) -> PyResult<()> {
        let persisted = carry::<B, _>(persisted);
        self.write(py, move |store, _| {
            persisted
                .into_inner()
                .delete(store, &PyValue(row))
                .map(drop)
        })
    }

    /// Removes one occurrence of `old`, if present, and inserts `new`, as
    /// one logged step.
    pub fn replace(
        &self,
        py: Python<'_>,
        persisted: Collection<B>,
        old: Py<PyAny>,
        new: Py<PyAny>,
    ) -> PyResult<()> {
        let persisted = carry::<B, _>(persisted);
        self.write(py, move |store, _| {
            persisted
                .into_inner()
                .replace(store, Some(&PyValue(old)), PyValue(new))
                .map(drop)
        })
    }

    /// Appends every row of `rows` in one logged step.
    pub fn extend(
        &self,
        py: Python<'_>,
        persisted: Collection<B>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let rows = rows
            .try_iter()?
            .map(|row| row.map(|row| PyValue(row.unbind())))
            .collect::<PyResult<Vec<_>>>()?;
        let persisted = carry::<B, _>(persisted);
        self.write(py, move |store, _| {
            let persisted = persisted.into_inner();
            store
                .batch(|batch| {
                    for row in rows {
                        batch.insert(&persisted, row);
                    }
                })
                .map(drop)
        })
    }
}

impl<B: PyBinding> StoreCore<B>
where
    B::Task<Keyed<B>>: Send,
{
    /// Declares the keyed collection `key` over `(key, row)` pairs encoded
    /// by `key_codec` and `codec`, holding the pairs the log restores.
    /// Raises `ValueError` if the key is already declared.
    pub fn keyed_collection(
        &self,
        py: Python<'_>,
        key: &str,
        key_codec: Py<PyAny>,
        codec: Py<PyAny>,
        label: Option<&str>,
    ) -> PyResult<Keyed<B>> {
        let _declaring = self.declare_lock(py)?;
        let slot = Slot::Key(key.to_owned());
        if self.registry.is_registered(&slot) {
            return Err(PyValueError::new_err(format!(
                "the key {key:?} is already declared"
            )));
        }
        let pairs = Py::new(py, PairCodec::new(key_codec, codec))?;
        self.registry.register(slot.clone(), pairs.into_any())?;
        let persisted = self.declaring(&slot, true, || {
            self.write(py, move |store, _| {
                store
                    .keyed_collection::<PyKey, PyValue, PyValue>(key)
                    .map(carry::<B, _>)
            })
        })?;
        let persisted = persisted.into_inner();
        if let Some(label) = label {
            self.labelled
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(key.to_owned(), (persisted.clone(), label.to_owned()));
        }
        Ok(persisted)
    }

    /// Whether a keyed collection of this store has a label.
    pub fn has_labelled_keys(&self) -> bool {
        !self
            .labelled
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty()
    }

    /// Labels `<label>/<str(key)>` the version input of each of `keys`, a
    /// collection key and one of its keys as [`crate::codec::capture_keys`]
    /// gives them, that belongs to a labelled keyed collection and has one.
    /// A failure to label is reported as unraisable.
    pub fn label_keys(&self, py: Python<'_>, keys: Vec<(String, PyKey)>)
    where
        B::Task<Vec<(Keyed<B>, String, PyKey)>>: Send,
    {
        let mut unlabelled = Vec::new();
        let mut targets = Vec::new();
        {
            let labelled = self.labelled.lock().unwrap_or_else(PoisonError::into_inner);
            for (collection, key) in keys {
                match labelled.get(&collection) {
                    Some((keyed, label)) => targets.push((keyed.clone(), label.clone(), key)),
                    None => unlabelled.push(key),
                }
            }
        }
        drop(unlabelled);
        if targets.is_empty() {
            return;
        }
        let targets = carry::<B, _>(targets);
        let labelled = B::write(py, &self.store, move |store, _| {
            for (keyed, label, key) in targets.into_inner() {
                if let Some(handle) = keyed.version_input(&key) {
                    crate::classes::label_key(store.runtime(), handle.slot(), &label, &key);
                }
            }
        });
        if let Err(e) = labelled {
            e.write_unraisable(py, None);
        }
    }
}

impl<B: PyBinding> StoreCore<B>
where
    B::Task<PendingWrite<B::C>>: Send,
{
    /// Appends `pending` as one run and applies it.
    pub fn write_pending(&self, py: Python<'_>, pending: Pending<B>) -> PyResult<()> {
        let pending = carry::<B, _>(pending.into_write());
        self.write(py, move |store, _| {
            store.write_pending(pending.into_inner()).map(drop)
        })
    }
}
