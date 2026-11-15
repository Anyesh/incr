//! Memo backends: [`PyMemoBackend`] adapts any Python object that
//! implements the memo backend protocol to incr-persist's `MemoBackend`,
//! and `MemoryMemo` is a reference implementation of that protocol over
//! incr-persist's own.
//!
//! The protocol, with a header as the `bytes` of a generation header and an
//! item as the tuple `(table, key_hash, loc, len)`:
//! `lock_checkpoint()` returns a guard whose `release()` ends the
//! checkpoint (incr calls it when the checkpoint finishes, and logs a
//! `release` that raises through `sys.unraisablehook`),
//! `load() -> (generation, header, items) | None`,
//! `read_entry(generation, item) -> bytes | None`,
//! `write_generation(base, generation, header, entries) -> items` for
//! `entries` of `(table, key_hash, frame)`, and
//! `remove_generation(generation)`. Raising `incr.persist.Busy` from
//! `lock_checkpoint`, `write_generation`, `write_generation_streamed` or
//! `write_delta` skips the checkpoint; any other exception fails it and
//! reaches the caller as itself.
//!
//! Four optional methods, looked up once when the store opens the backend,
//! add what a large store needs. A backend without them behaves
//! exactly as one written for the five methods above.
//!
//! - `read_entries(generation, items) -> [bytes | None]` reads the frames
//!   of `items` in order, none of them a tombstone, for a backend whose
//!   reads cost a round trip each.
//! - `write_generation_streamed(base, generation, header, frames) -> items`
//!   is `write_generation` with `frames` a lazy iterator of
//!   `(table, key_hash, frame)`, so a compaction never builds one list of
//!   every frame. Pull `frames` on the calling thread, during the call:
//!   it is invalid once the call returns, and a call that returns without
//!   exhausting it (reading until `StopIteration`) fails the checkpoint.
//!   Pulling may read this backend through `read_entry` and `read_entries`,
//!   so do not hold a lock of those across a pull.
//! - Delta chains, enabled when the backend has both `load_chain` and
//!   `write_delta`. A chain is a full generation, its base, followed by
//!   deltas, each written on the one before it; the newest layer is the
//!   tip, which `write_generation` and `write_delta` take as their `base`
//!   and `tip` and which they must find current, else raise `Busy`. A full
//!   write replaces the whole chain. `load()` keeps returning the chain's
//!   base, and `lock_checkpoint` must keep every layer of the current
//!   chain while removing other generations.
//!   `load_chain() -> [(generation, header, items)] | None` returns every
//!   layer, base first; a layer's header is the bytes its writer was
//!   given (the base's, or the delta header a `write_delta` received).
//!   `write_delta(tip, generation, header, entries, tombstones) -> items`
//!   stores the delta's `entries` as `write_generation` does, and hides
//!   each `(table, key_hash)` of `tombstones` from the layers below: a
//!   tombstone is an item `(table, key_hash, 0, 0)` with no frame, which
//!   the returned items and `load_chain` list like any other. An unreadable
//!   delta layer may be returned with a header incr cannot decode: it is
//!   then treated as holding nothing.
//!
//! incr decides which write a checkpoint makes: `write_delta` appends only
//! what changed since the tip, while `write_generation_streamed` (or
//! `write_generation` for a backend without it) rewrites everything, as a
//! compaction or a first checkpoint does. `load_chain` (or `load`) opens
//! the store, and `read_entries` (or `read_entry`) serves restores and the
//! frames a rewrite carries over from older layers.

use std::marker::PhantomData;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::thread::{self, ThreadId};

use incr_persist::memo::{
    decode_delta_header, decode_header, encode_delta_header, encode_header, Chain, DeltaHeader,
    EncodedEntry, GenId, Generation, IndexItem, MemoBackend, Tombstone,
};
use incr_persist::{Error, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString};

use crate::backend::{bad_return, raised};
use crate::PersistExceptions;

/// A memo index item as the protocol passes it: `(table, key_hash, loc, len)`.
pub type ItemTuple = (u32, u64, u64, u32);

pub fn item_tuple(item: &IndexItem) -> ItemTuple {
    (item.table, item.key_hash, item.loc, item.len)
}

fn item_of<E: PersistExceptions>(object: &Bound<'_, PyAny>, what: &str) -> Result<IndexItem> {
    let (table, key_hash, loc, len) = object
        .extract::<ItemTuple>()
        .map_err(|_| bad_return::<E>(what))?;
    if len == 0 {
        return Ok(IndexItem::tombstone(table, key_hash));
    }
    Ok(IndexItem {
        table,
        key_hash,
        loc,
        len,
    })
}

fn items_of<E: PersistExceptions>(object: &Bound<'_, PyAny>, what: &str) -> Result<Vec<IndexItem>> {
    object
        .try_iter()
        .map_err(|_| bad_return::<E>(what))?
        .map(|item| item_of::<E>(&item.map_err(raised::<E>)?, what))
        .collect()
}

/// A Python object implementing the memo backend protocol, as a
/// `MemoBackend` whose errors use the exceptions `E`.
pub struct PyMemoBackend<E> {
    backend: Py<PyAny>,
    chains: bool,
    streams: bool,
    batches: bool,
    exceptions: PhantomData<fn() -> E>,
}

impl<E: PersistExceptions> PyMemoBackend<E> {
    /// Wraps `backend`; raises what looking up its optional methods
    /// raises, other than `AttributeError`.
    pub fn new(backend: Py<PyAny>) -> PyResult<Self> {
        Python::attach(|py| {
            let object = backend.bind(py);
            let chains = object.hasattr(intern!(py, "load_chain"))?
                && object.hasattr(intern!(py, "write_delta"))?;
            let streams = object.hasattr(intern!(py, "write_generation_streamed"))?;
            let batches = object.hasattr(intern!(py, "read_entries"))?;
            Ok(PyMemoBackend {
                backend,
                chains,
                streams,
                batches,
                exceptions: PhantomData,
            })
        })
    }

    fn call<'py, A: pyo3::call::PyCallArgs<'py>>(
        &self,
        py: Python<'py>,
        method: &Bound<'py, PyString>,
        args: A,
    ) -> Result<Bound<'py, PyAny>> {
        self.backend
            .bind(py)
            .call_method1(method, args)
            .map_err(raised::<E>)
    }
}

/// The object `lock_checkpoint()` returned, released when the checkpoint
/// drops it.
pub struct PyMemoGuard {
    guard: Py<PyAny>,
}

impl Drop for PyMemoGuard {
    fn drop(&mut self) {
        Python::attach(|py| {
            let guard = self.guard.bind(py);
            if let Err(err) = guard.call_method0(intern!(py, "release")) {
                err.write_unraisable(py, Some(guard));
            }
        });
    }
}

impl<E: PersistExceptions> MemoBackend for PyMemoBackend<E> {
    type Guard = PyMemoGuard;

    fn lock_checkpoint(&self) -> Result<PyMemoGuard> {
        Python::attach(|py| {
            let guard = self.call(py, intern!(py, "lock_checkpoint"), ())?;
            Ok(PyMemoGuard {
                guard: guard.unbind(),
            })
        })
    }

    fn load(&self) -> Result<Option<Generation>> {
        Python::attach(|py| {
            let result = self.call(py, intern!(py, "load"), ())?;
            if result.is_none() {
                return Ok(None);
            }
            let (id, header, items) = result
                .extract::<(u64, Vec<u8>, Bound<'_, PyAny>)>()
                .map_err(|_| bad_return::<E>("load"))?;
            let id = GenId(id);
            let Ok(header) = decode_header(&header) else {
                return Ok(Some(Generation::empty(id)));
            };
            Ok(Some(Generation::new(
                id,
                header,
                items_of::<E>(&items, "load")?,
            )))
        })
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        Python::attach(|py| {
            let result = self.call(
                py,
                intern!(py, "read_entry"),
                (generation.0, item_tuple(item)),
            )?;
            if result.is_none() {
                return Ok(None);
            }
            let frame = result
                .downcast::<PyBytes>()
                .map_err(|_| bad_return::<E>("read_entry"))?;
            Ok(Some(frame.as_bytes().to_vec()))
        })
    }

    fn write_generation(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &incr_persist::memo::GenHeader,
        entries: &[EncodedEntry],
    ) -> Result<Generation> {
        Python::attach(|py| {
            let frames = entries
                .iter()
                .map(|entry| (entry.table, entry.key_hash, PyBytes::new(py, &entry.frame)))
                .collect::<Vec<_>>();
            let result = self.call(
                py,
                intern!(py, "write_generation"),
                (
                    base.map(|base| base.0),
                    id.0,
                    PyBytes::new(py, &encode_header(header)?),
                    PyList::new(py, frames).map_err(raised::<E>)?,
                ),
            )?;
            let items = items_of::<E>(&result, "write_generation")?;
            Ok(Generation::new(id, header.clone(), items))
        })
    }

    fn write_generation_streamed(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &incr_persist::memo::GenHeader,
        entries: &mut Pulls<'_>,
    ) -> Result<Generation> {
        if !self.streams {
            let entries = entries.collect::<Result<Vec<_>>>()?;
            return self.write_generation(base, id, header, &entries);
        }
        Python::attach(|py| {
            let lent = Lent::new(py, entries).map_err(raised::<E>)?;
            let result = self.call(
                py,
                intern!(py, "write_generation_streamed"),
                (
                    base.map(|base| base.0),
                    id.0,
                    PyBytes::new(py, &encode_header(header)?),
                    lent.frames.clone_ref(py),
                ),
            );
            let frames = lent.end(py);
            let frames = frames.get();
            match frames.failure() {
                Some(Failure::Error(error)) => return Err(error),
                Some(Failure::Panic(payload)) => panic::resume_unwind(payload),
                None => {}
            }
            let items = items_of::<E>(&result?, "write_generation_streamed")?;
            if !frames.exhausted() {
                return Err(raised::<E>(PyRuntimeError::new_err(
                    "write_generation_streamed returned before its frames were exhausted",
                )));
            }
            Ok(Generation::new(id, header.clone(), items))
        })
    }

    fn read_entries(&self, generation: GenId, items: &[IndexItem]) -> Result<Vec<Option<Vec<u8>>>> {
        if !self.batches {
            return items
                .iter()
                .map(|item| self.read_entry(generation, item))
                .collect();
        }
        Python::attach(|py| {
            let result = self.call(
                py,
                intern!(py, "read_entries"),
                (
                    generation.0,
                    items.iter().map(item_tuple).collect::<Vec<_>>(),
                ),
            )?;
            let frames = result
                .try_iter()
                .map_err(|_| bad_return::<E>("read_entries"))?
                .map(|frame| {
                    let frame = frame.map_err(raised::<E>)?;
                    if frame.is_none() {
                        return Ok(None);
                    }
                    let frame = frame
                        .downcast::<PyBytes>()
                        .map_err(|_| bad_return::<E>("read_entries"))?;
                    Ok(Some(frame.as_bytes().to_vec()))
                })
                .collect::<Result<Vec<_>>>()?;
            if frames.len() != items.len() {
                return Err(bad_return::<E>("read_entries"));
            }
            Ok(frames)
        })
    }

    fn supports_delta(&self) -> bool {
        self.chains
    }

    fn load_chain(&self, known: &[Arc<Generation>]) -> Result<Option<Chain>> {
        if !self.chains {
            let _ = known;
            return Ok(self.load()?.map(Chain::from));
        }
        Python::attach(|py| {
            let result = self.call(py, intern!(py, "load_chain"), ())?;
            if result.is_none() {
                return Ok(None);
            }
            let mut layers: Vec<Arc<Generation>> = Vec::new();
            for layer in result
                .try_iter()
                .map_err(|_| bad_return::<E>("load_chain"))?
            {
                let layer = layer.map_err(raised::<E>)?;
                let (id, header, items) = layer
                    .extract::<(u64, Vec<u8>, Bound<'_, PyAny>)>()
                    .map_err(|_| bad_return::<E>("load_chain"))?;
                let id = GenId(id);
                let items = items_of::<E>(&items, "load_chain")?;
                let generation = match (layers.first(), layers.last()) {
                    (Some(base), Some(parent)) => match decode_delta_header(&header) {
                        Ok(delta) => Generation::new_delta(id, delta, items),
                        Err(_) => Generation::empty_delta(id, base.id(), parent.id()),
                    },
                    _ => match decode_header(&header) {
                        Ok(header) => Generation::new(id, header, items),
                        Err(_) => Generation::empty(id),
                    },
                };
                layers.push(Arc::new(generation));
            }
            if layers.is_empty() {
                return Err(bad_return::<E>("load_chain"));
            }
            Chain::new(layers).map(Some)
        })
    }

    fn write_delta(
        &self,
        tip: GenId,
        id: GenId,
        header: &DeltaHeader,
        entries: &[EncodedEntry],
        tombstones: &[Tombstone],
    ) -> Result<Generation> {
        if !self.chains {
            return Err(Error::Unsupported("memo delta"));
        }
        Python::attach(|py| {
            let frames = entries
                .iter()
                .map(|entry| (entry.table, entry.key_hash, PyBytes::new(py, &entry.frame)))
                .collect::<Vec<_>>();
            let hidden = tombstones
                .iter()
                .map(|t| (t.table, t.key_hash))
                .collect::<Vec<_>>();
            let result = self.call(
                py,
                intern!(py, "write_delta"),
                (
                    tip.0,
                    id.0,
                    PyBytes::new(py, &encode_delta_header(header)?),
                    PyList::new(py, frames).map_err(raised::<E>)?,
                    hidden,
                ),
            )?;
            let items = items_of::<E>(&result, "write_delta")?;
            Ok(Generation::new_delta(id, header.clone(), items))
        })
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        Python::attach(|py| {
            self.call(py, intern!(py, "remove_generation"), (generation.0,))
                .map(drop)
        })
    }
}

type Pulls<'a> = dyn Iterator<Item = Result<EncodedEntry>> + 'a;

struct Source(*mut Pulls<'static>);

// The pointer is only dereferenced on the thread that lent it (`Frames`
// checks), while `Lent` keeps its referent alive.
unsafe impl Send for Source {}

/// The lazy `frames` iterator a streamed write hands its backend: it pulls
/// the engine's own iterator, and is invalid once the call returns.
#[pyclass(name = "MemoFrames", module = "incr_python_persist", frozen)]
pub struct Frames {
    source: Mutex<Option<Source>>,
    thread: ThreadId,
    exhausted: AtomicBool,
    failure: Mutex<Option<Failure>>,
}

/// What stopped a pull: an error the engine's iterator returned, or a
/// panic inside it, resumed once the backend's call has returned.
enum Failure {
    Error(Error),
    Panic(Box<dyn std::any::Any + Send>),
}

impl Frames {
    fn exhausted(&self) -> bool {
        self.exhausted.load(Ordering::Acquire)
    }

    fn failure(&self) -> Option<Failure> {
        self.failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }
}

#[pymethods]
impl Frames {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<(u32, u64, Py<PyBytes>)>> {
        if thread::current().id() != self.thread {
            return Err(PyRuntimeError::new_err(
                "memo frames must be pulled on the thread that called write_generation_streamed",
            ));
        }
        let mut source = match self.source.try_lock() {
            Ok(source) => source,
            Err(TryLockError::WouldBlock) => {
                return Err(PyRuntimeError::new_err("memo frames pulled re-entrantly"))
            }
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
        };
        let Some(Source(pulls)) = source.as_mut() else {
            return Err(PyRuntimeError::new_err(
                "memo frames are only valid during the write_generation_streamed call",
            ));
        };
        if self.exhausted() {
            return Ok(None);
        }
        // SAFETY: `Lent` keeps the pointee alive and clears this slot
        // before the call that lent it returns, and the lock excludes a
        // second pull.
        let next = panic::catch_unwind(AssertUnwindSafe(|| unsafe { (**pulls).next() }));
        let next = match next {
            Ok(next) => next,
            Err(payload) => {
                *source = None;
                *self.failure.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(Failure::Panic(payload));
                return Err(PyRuntimeError::new_err("memo frame source panicked"));
            }
        };
        match next {
            None => {
                self.exhausted.store(true, Ordering::Release);
                Ok(None)
            }
            Some(Ok(entry)) => Ok(Some((
                entry.table,
                entry.key_hash,
                PyBytes::new(py, &entry.frame).unbind(),
            ))),
            Some(Err(error)) => {
                let message = format!("memo frame source failed: {error}");
                *self.failure.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(Failure::Error(error));
                Err(PyRuntimeError::new_err(message))
            }
        }
    }
}

/// A [`Frames`] over an engine iterator, invalidated by [`Lent::end`] or
/// by dropping it.
struct Lent {
    frames: Py<Frames>,
}

impl Lent {
    fn new(py: Python<'_>, entries: &mut Pulls<'_>) -> PyResult<Self> {
        let ptr: *mut Pulls<'_> = entries;
        // SAFETY: erases the borrow's lifetime; `Frames` never uses the
        // pointer after `Lent` is gone, which its borrow outlives.
        let ptr = unsafe { std::mem::transmute::<*mut Pulls<'_>, *mut Pulls<'static>>(ptr) };
        let frames = Py::new(
            py,
            Frames {
                source: Mutex::new(Some(Source(ptr))),
                thread: thread::current().id(),
                exhausted: AtomicBool::new(false),
                failure: Mutex::new(None),
            },
        )?;
        Ok(Lent { frames })
    }

    fn end(self, py: Python<'_>) -> Bound<'_, Frames> {
        self.invalidate(py);
        self.frames.bind(py).clone()
    }

    fn invalidate(&self, py: Python<'_>) {
        let frames = self.frames.bind(py).get();
        *frames.source.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

impl Drop for Lent {
    fn drop(&mut self) {
        Python::attach(|py| self.invalidate(py));
    }
}
