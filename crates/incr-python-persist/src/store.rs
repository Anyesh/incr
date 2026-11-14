//! The state a package's `persist.Store` class wraps: the incr-persist
//! store with the runtime it owns, the codecs of its slots, and its
//! persisted inputs.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use incr_core::Incr;
use incr_persist::memo::fingerprint;
use incr_persist::{Durability, Input, Options, Store};
use incr_python_protocol::PyValue;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::backend::PyBackend;
use crate::binding::{CoreStore, Detached, PyBinding};
use crate::codec::{Registry, Slot};

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

/// A store, its codecs and its persisted inputs, shareable across threads
/// when `B::C` is. The input table is a leaf lock under which no Python
/// runs.
pub struct StoreCore<B: PyBinding> {
    store: Arc<CoreStore<B>>,
    registry: Arc<Registry<B::Exceptions>>,
    build_id: u128,
    inputs: Mutex<HashMap<u32, Input<PyValue>>>,
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
        let store = registry.run(py, || Store::open(PyBackend::new(backend), options))?;
        Ok(StoreCore {
            store: Arc::new(store),
            registry,
            build_id,
            inputs: Mutex::default(),
        })
    }

    pub fn runtime(&self) -> &incr_core::Runtime<B::C> {
        self.store.runtime()
    }

    /// The id naming the code that computes memoized values.
    pub fn build_id(&self) -> u128 {
        self.build_id
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

    fn input_of(&self, node: Incr<PyValue>) -> PyResult<Input<PyValue>> {
        self.inputs()
            .get(&node.slot())
            .copied()
            .filter(|input| input.handle() == node)
            .ok_or_else(|| PyValueError::new_err("node is not a persisted input of this store"))
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

    /// Runs a store call through `B::write` with the store's codecs active,
    /// raising a codec's exception as itself.
    fn write<R: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&CoreStore<B>, Detached) -> incr_persist::Result<R> + Send,
    ) -> PyResult<R> {
        drop(self.registry.take_failure());
        let registry = &self.registry;
        let result = B::write(py, &self.store, move |store, detached| {
            registry.scoped(|| f(store, detached))
        })?;
        self.registry.finish(py, result)
    }
}
