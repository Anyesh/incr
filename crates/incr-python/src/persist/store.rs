//! `incr.persist.Store`: a runtime whose persisted inputs and collections
//! live in a backend's log. Writes to them go through the store; the
//! runtime and the graph API read them like any other node.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use incr_core::Local;
use incr_persist::memo::fingerprint;
use incr_persist::{Durability, Input, Options, PersistedCollection, Store};
use incr_python_protocol::PyValue;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;

use super::backend::PyBackend;
use super::codec::{Registry, Slot};
use crate::overlay::Born;
use crate::{engine_write, PyCollection, PyNodeId, PyRuntime};

type PyPersistedCollection = PersistedCollection<PyValue, Local>;

/// What the store's Python objects share: the incr-persist store with the
/// runtime it owns, the codecs of its slots, and the persisted nodes the
/// binding must keep direct writes off.
pub(crate) struct StoreCore {
    store: Store<Local, PyBackend>,
    registry: Rc<Registry>,
    #[expect(dead_code, reason = "the memo layer opens with it")]
    build_id: u128,
    inputs: RefCell<HashMap<u32, Input<PyValue>>>,
    collections: RefCell<HashMap<u32, PyPersistedCollection>>,
}

impl StoreCore {
    pub(crate) fn runtime(&self) -> &incr_compute::Runtime {
        self.store.runtime()
    }

    pub(crate) fn is_persisted_input(&self, slot: u32) -> bool {
        self.inputs.borrow().contains_key(&slot)
    }

    fn input_of(&self, node: &PyNodeId) -> PyResult<Input<PyValue>> {
        self.inputs
            .borrow()
            .get(&node.inner.slot())
            .copied()
            .filter(|input| input.handle() == node.inner)
            .ok_or_else(|| PyValueError::new_err("node is not a persisted input of this store"))
    }

    fn collection_of(&self, collection: &PyCollection) -> PyResult<PyPersistedCollection> {
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

    fn run<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&Store<Local, PyBackend>) -> incr_persist::Result<R>,
    ) -> PyResult<R> {
        self.registry.run(py, || f(&self.store))
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
    runtime: Py<PyRuntime>,
}

const _: () = assert!(std::mem::align_of::<PyStore>() <= 16);

impl PyStore {
    fn core(&self) -> PyResult<&Rc<StoreCore>> {
        self.core
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("store is closed"))
    }

    fn core_outside_overlay(&self, py: Python<'_>) -> PyResult<Rc<StoreCore>> {
        let core = Rc::clone(self.core()?);
        if self.runtime.bind(py).borrow().state.overlay_active() {
            return Err(PyRuntimeError::new_err(
                "store calls cannot run inside an overlay",
            ));
        }
        Ok(core)
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
        let core = Rc::new(StoreCore {
            store,
            registry,
            build_id,
            inputs: RefCell::default(),
            collections: RefCell::default(),
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
