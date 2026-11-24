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
//!   A pull that raises means the write must be abandoned with nothing
//!   stored; incr then reports the pull's own failure, whatever the call
//!   returns or raises.
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
//!   A `load_chain(known)` that takes one argument is passed a frozenset of
//!   the generation ids this handle already holds, and may send
//!   `(generation, None, None)` for each of them instead of loading the
//!   layer again; the list still names every layer of the chain, in order.
//!   A placeholder for an id outside `known` fails the checkpoint with a
//!   `TypeError`. Whether `load_chain` takes the argument is decided once,
//!   when the store opens the backend.
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
use std::panic;
use std::sync::Arc;

use incr_core::YieldHook;
use incr_persist::memo::{
    decode_delta_header, decode_header, encode_delta_header, encode_header, Chain, DeltaHeader,
    EncodedEntry, GenHeader, GenId, Generation, IndexItem, MemoBackend, Tombstone,
};
use incr_persist::{Error, Result};
use incr_python_protocol::gil::{self, Site};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyString};
use pyo3::IntoPyObjectExt;

use crate::backend::{bad_return, declares_concurrent, raised, sequence_error, Args};
use crate::lent::{self, Failure, Lending, Names};
use crate::PersistExceptions;

static OPEN: Site = Site::caller("memo.open");
static GUARD_RELEASE: Site = Site::one("memo.guard_release");
static LOCK_CHECKPOINT: Site = Site::one("memo.lock_checkpoint");
static LOAD: Site = Site::one("memo.load");
static LOAD_ITEMS: Site = Site::unbounded("memo.load.items");
static READ_ENTRY: Site = Site::one("memo.read_entry");
static WRITE_GENERATION: Site = Site::caller("memo.write_generation");
static WRITE_GENERATION_STREAMED: Site = Site::one("memo.write_generation_streamed");
static WRITTEN_ITEMS: Site = Site::unbounded("memo.write_generation_streamed.items");
static READ_ENTRIES: Site = Site::one("memo.read_entries");
static READ_ENTRIES_ITEMS: Site = Site::bounded("memo.read_entries.items", 256);
static READ_ENTRIES_FRAMES: Site = Site::bounded("memo.read_entries.frames", 256);
static LOAD_CHAIN: Site = Site::one("memo.load_chain");
static LOAD_CHAIN_KNOWN: Site = Site::one("memo.load_chain.known");
static LOAD_CHAIN_ITEMS: Site = Site::unbounded("memo.load_chain.items");
static WRITE_DELTA: Site = Site::one("memo.write_delta");
static WRITE_DELTA_FRAMES: Site = Site::unbounded("memo.write_delta.frames");
static WRITE_DELTA_TOMBSTONES: Site = Site::unbounded("memo.write_delta.tombstones");
static WRITE_DELTA_ITEMS: Site = Site::unbounded("memo.write_delta.items");
static REMOVE_GENERATION: Site = Site::one("memo.remove_generation");
static LENT_DROP: Site = Site::one("memo.lent_drop");

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

fn items_of<E: PersistExceptions>(
    site: &'static Site,
    object: Bound<'_, PyAny>,
    what: &'static str,
) -> Result<Vec<IndexItem>> {
    gil::drain(site, object, sequence_error::<E>(what), |item| {
        item_of::<E>(item, what)
    })
}

fn bytes_of<E: PersistExceptions>(object: &Bound<'_, PyAny>, what: &str) -> Result<Vec<u8>> {
    object
        .downcast::<PyBytes>()
        .map(|bytes| bytes.as_bytes().to_vec())
        .map_err(|_| bad_return::<E>(what))
}

/// A layer header, which a backend may return as any bytes-like object
/// (a driver's binary column type is often a `memoryview`), but not an
/// integer, which `bytes()` would read as a length.
fn header_of<E: PersistExceptions>(object: &Bound<'_, PyAny>, what: &str) -> Result<Vec<u8>> {
    if let Ok(bytes) = object.downcast::<PyBytes>() {
        return Ok(bytes.as_bytes().to_vec());
    }
    if object.hasattr("__index__").unwrap_or(true) {
        return Err(bad_return::<E>(what));
    }
    let bytes = object
        .py()
        .get_type::<PyBytes>()
        .call1((object,))
        .map_err(|_| bad_return::<E>(what))?;
    bytes_of::<E>(&bytes, what)
}

fn frames_of<'py, E: PersistExceptions>(
    py: Python<'py>,
    site: &'static Site,
    entries: &[EncodedEntry],
) -> Result<Bound<'py, PyAny>> {
    gil::build(py, site, entries, raised::<E>, |py, entry| {
        (entry.table, entry.key_hash, PyBytes::new(py, &entry.frame))
            .into_pyobject(py)
            .map(Bound::into_any)
            .map_err(raised::<E>)
    })
}

fn emptied<E: PersistExceptions>(list: Bound<'_, PyAny>) -> Result<()> {
    gil::empty(list).map_err(raised::<E>)
}

/// A Python object implementing the memo backend protocol, as a
/// `MemoBackend` whose errors use the exceptions `E`.
pub struct PyMemoBackend<E> {
    backend: Py<PyAny>,
    chains: bool,
    takes_known: bool,
    streams: bool,
    batches: bool,
    concurrent: bool,
    /// The store runtime's hook, for a native backend to tick on the
    /// engine's thread; never installed on a checkpoint worker.
    yield_hook: Option<YieldHook>,
    exceptions: PhantomData<fn() -> E>,
}

impl<E: PersistExceptions> PyMemoBackend<E> {
    /// Wraps `backend`; raises what looking up its optional methods
    /// raises, other than `AttributeError`.
    pub fn new(backend: Py<PyAny>) -> PyResult<Self> {
        gil::enter(&OPEN, |py| {
            let object = backend.bind(py);
            let chains = object.hasattr(intern!(py, "load_chain"))?
                && object.hasattr(intern!(py, "write_delta"))?;
            let takes_known = chains && takes_one_argument(py, object, "load_chain")?;
            let streams = object.hasattr(intern!(py, "write_generation_streamed"))?;
            let batches = object.hasattr(intern!(py, "read_entries"))?;
            let concurrent = streams && declares_concurrent(object)?;
            Ok(PyMemoBackend {
                backend,
                chains,
                takes_known,
                streams,
                batches,
                concurrent,
                yield_hook: None,
                exceptions: PhantomData,
            })
        })
    }

    pub fn with_yield_hook(self, yield_hook: Option<YieldHook>) -> Self {
        PyMemoBackend { yield_hook, ..self }
    }

    fn hook(&self) -> Option<&YieldHook> {
        self.yield_hook.as_ref().filter(|_| !gil::on_worker())
    }

    fn enter<R>(&self, site: &'static Site, f: impl for<'py> FnOnce(Python<'py>) -> R) -> R {
        incr_core::with_ambient_hook(self.hook(), || gil::enter(site, f))
    }

    fn call<'py>(
        &self,
        py: Python<'py>,
        method: &Bound<'py, PyString>,
        args: impl Args<'py>,
    ) -> Result<Bound<'py, PyAny>> {
        self.backend
            .bind(py)
            .call_method1(method, args)
            .map_err(raised::<E>)
    }
}

/// Whether `object.method` accepts one positional argument. A callable
/// with no readable signature is taken to accept none.
fn takes_one_argument(py: Python<'_>, object: &Bound<'_, PyAny>, method: &str) -> PyResult<bool> {
    let signature = match py
        .import(intern!(py, "inspect"))?
        .call_method1(intern!(py, "signature"), (object.getattr(method)?,))
    {
        Ok(signature) => signature,
        Err(error) if error.is_instance_of::<PyValueError>(py) => return Ok(false),
        Err(error) if error.is_instance_of::<PyTypeError>(py) => return Ok(false),
        Err(error) => return Err(error),
    };
    match signature.call_method1(intern!(py, "bind"), (py.None(),)) {
        Ok(_) => Ok(true),
        Err(error) if error.is_instance_of::<PyTypeError>(py) => Ok(false),
        Err(error) => Err(error),
    }
}

/// The object `lock_checkpoint()` returned, released when the checkpoint
/// drops it.
pub struct PyMemoGuard {
    guard: Py<PyAny>,
}

impl Drop for PyMemoGuard {
    fn drop(&mut self) {
        gil::enter(&GUARD_RELEASE, |py| {
            let guard = self.guard.bind(py);
            if let Err(err) = guard.call_method0(intern!(py, "release")) {
                err.write_unraisable(py, Some(guard));
            }
        });
    }
}

/// A layer as `load` and `load_chain` return it: `(generation, header,
/// items)`, read into Rust before any `Generation` is built.
struct Layer {
    id: GenId,
    header: Option<Vec<u8>>,
    items: Option<Vec<IndexItem>>,
}

impl<E: PersistExceptions> PyMemoBackend<E> {
    /// Reads the layer tuple `object`, which incr holds the only reference
    /// to, draining its items once the tuple is freed.
    fn layer(
        &self,
        object: Bound<'_, PyAny>,
        items_site: &'static Site,
        what: &'static str,
    ) -> Result<Layer> {
        let (id, header, items) = object
            .extract::<(u64, Bound<'_, PyAny>, Bound<'_, PyAny>)>()
            .map_err(|_| bad_return::<E>(what))?;
        drop(object);
        if header.is_none() && items.is_none() {
            return Ok(Layer {
                id: GenId(id),
                header: None,
                items: None,
            });
        }
        let header = header_of::<E>(&header, what)?;
        Ok(Layer {
            id: GenId(id),
            header: Some(header),
            items: Some(items_of::<E>(items_site, items, what)?),
        })
    }
}

impl<E: PersistExceptions> MemoBackend for PyMemoBackend<E> {
    type Guard = PyMemoGuard;

    fn lock_checkpoint(&self) -> Result<PyMemoGuard> {
        self.enter(&LOCK_CHECKPOINT, |py| {
            let guard = self.call(py, intern!(py, "lock_checkpoint"), ())?;
            Ok(PyMemoGuard {
                guard: guard.unbind(),
            })
        })
    }

    fn load(&self) -> Result<Option<Generation>> {
        let layer = self.enter(&LOAD, |py| {
            let result = self.call(py, intern!(py, "load"), ())?;
            if result.is_none() {
                return Ok(None);
            }
            self.layer(result, &LOAD_ITEMS, "load").map(Some)
        })?;
        let Some(Layer { id, header, items }) = layer else {
            return Ok(None);
        };
        let Ok(header) = decode_header(&header.ok_or_else(|| bad_return::<E>("load"))?) else {
            return Ok(Some(Generation::empty(id)));
        };
        Ok(Some(Generation::new(id, header, items.unwrap_or_default())))
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        self.enter(&READ_ENTRY, |py| {
            let result = self.call(
                py,
                intern!(py, "read_entry"),
                (
                    generation.0,
                    item_tuple(item)
                        .into_pyobject(py)
                        .map_err(raised::<E>)?
                        .into_any(),
                ),
            )?;
            if result.is_none() {
                return Ok(None);
            }
            bytes_of::<E>(&result, "read_entry").map(Some)
        })
    }

    fn write_generation(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &[EncodedEntry],
    ) -> Result<Generation> {
        let items = self.enter(&WRITE_GENERATION, |py| {
            let frames = frames_of::<E>(py, &WRITE_GENERATION, entries)?;
            let result = self.call(
                py,
                intern!(py, "write_generation"),
                (
                    base.map(|base| base.0),
                    id.0,
                    PyBytes::new(py, &encode_header(header)?),
                    frames.clone(),
                ),
            )?;
            emptied::<E>(frames)?;
            items_of::<E>(&WRITE_GENERATION, result, "write_generation")
        })?;
        Ok(Generation::new(id, header.clone(), items))
    }

    fn write_generation_streamed(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &mut Pulls<'_>,
    ) -> Result<Generation> {
        if !self.streams {
            let entries = entries.collect::<Result<Vec<_>>>()?;
            return self.write_generation(base, id, header, &entries);
        }
        let items = self.enter(&WRITE_GENERATION_STREAMED, |py| {
            let lent = Lent::new(py, entries).map_err(raised::<E>)?;
            let result = self.call(
                py,
                intern!(py, "write_generation_streamed"),
                (
                    base.map(|base| base.0),
                    id.0,
                    PyBytes::new(py, &encode_header(header)?),
                    lent.frames.clone_ref(py).into_any(),
                ),
            );
            let frames = lent.end(py);
            let frames = &frames.get().lending;
            match frames.failure() {
                Some(Failure::Error(error)) => return Err(error),
                Some(Failure::Panic(payload)) => panic::resume_unwind(payload),
                None => {}
            }
            let items = items_of::<E>(&WRITTEN_ITEMS, result?, "write_generation_streamed")?;
            if !frames.exhausted() {
                return Err(raised::<E>(PyRuntimeError::new_err(
                    "write_generation_streamed returned before its frames were exhausted",
                )));
            }
            Ok(items)
        })?;
        Ok(Generation::new(id, header.clone(), items))
    }

    fn read_entries(&self, generation: GenId, items: &[IndexItem]) -> Result<Vec<Option<Vec<u8>>>> {
        if !self.batches {
            return items
                .iter()
                .map(|item| self.read_entry(generation, item))
                .collect();
        }
        self.enter(&READ_ENTRIES, |py| {
            let asked = gil::build(py, &READ_ENTRIES_ITEMS, items, raised::<E>, |py, item| {
                item_tuple(item)
                    .into_pyobject(py)
                    .map(Bound::into_any)
                    .map_err(raised::<E>)
            })?;
            let result = self.call(
                py,
                intern!(py, "read_entries"),
                (generation.0, asked.clone()),
            )?;
            emptied::<E>(asked)?;
            let frames = gil::drain(
                &READ_ENTRIES_FRAMES,
                result,
                sequence_error::<E>("read_entries"),
                |frame| {
                    if frame.is_none() {
                        return Ok(None);
                    }
                    bytes_of::<E>(frame, "read_entries").map(Some)
                },
            )?;
            if frames.len() != items.len() {
                return Err(bad_return::<E>("read_entries"));
            }
            Ok(frames)
        })
    }

    fn supports_delta(&self) -> bool {
        self.chains
    }

    fn concurrent_compaction(&self) -> bool {
        self.concurrent
    }

    fn load_chain(&self, known: &[Arc<Generation>]) -> Result<Option<Chain>> {
        if !self.chains {
            let _ = known;
            return Ok(self.load()?.map(Chain::from));
        }
        let read = self.enter(&LOAD_CHAIN, |py| {
            let result = if self.takes_known {
                let ids = gil::build(py, &LOAD_CHAIN_KNOWN, known, raised::<E>, |py, layer| {
                    layer.id().0.into_bound_py_any(py).map_err(raised::<E>)
                })?;
                let ids = gil::frozen(ids).map_err(raised::<E>)?;
                self.call(py, intern!(py, "load_chain"), (ids,))?
            } else {
                self.call(py, intern!(py, "load_chain"), ())?
            };
            if result.is_none() {
                return Ok(None);
            }
            let layers = gil::drain(
                &LOAD_CHAIN,
                result,
                sequence_error::<E>("load_chain"),
                |layer| Ok(layer.clone()),
            )?;
            layers
                .into_iter()
                .map(|layer| self.layer(layer, &LOAD_CHAIN_ITEMS, "load_chain"))
                .collect::<Result<Vec<_>>>()
                .map(Some)
        })?;
        let Some(read) = read else {
            return Ok(None);
        };
        let mut layers: Vec<Arc<Generation>> = Vec::new();
        for Layer { id, header, items } in read {
            let (Some(header), Some(items)) = (header, items) else {
                if !self.takes_known {
                    return Err(bad_return::<E>("load_chain"));
                }
                let held = known
                    .iter()
                    .find(|held| held.id() == id)
                    .ok_or_else(|| bad_return::<E>("load_chain"))?;
                layers.push(Arc::clone(held));
                continue;
            };
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
        let items = self.enter(&WRITE_DELTA, |py| {
            let frames = frames_of::<E>(py, &WRITE_DELTA_FRAMES, entries)?;
            let hidden = gil::build(
                py,
                &WRITE_DELTA_TOMBSTONES,
                tombstones,
                raised::<E>,
                |py, t| {
                    (t.table, t.key_hash)
                        .into_pyobject(py)
                        .map(Bound::into_any)
                        .map_err(raised::<E>)
                },
            )?;
            let result = self.call(
                py,
                intern!(py, "write_delta"),
                (
                    tip.0,
                    id.0,
                    PyBytes::new(py, &encode_delta_header(header)?),
                    frames.clone(),
                    hidden.clone(),
                ),
            )?;
            emptied::<E>(frames)?;
            emptied::<E>(hidden)?;
            items_of::<E>(&WRITE_DELTA_ITEMS, result, "write_delta")
        })?;
        Ok(Generation::new_delta(id, header.clone(), items))
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        self.enter(&REMOVE_GENERATION, |py| {
            self.call(py, intern!(py, "remove_generation"), (generation.0,))
                .map(drop)
        })
    }
}

type Pulls<'a> = lent::Pulls<'a, EncodedEntry>;

const FRAMES: Names = Names {
    items: "memo frames",
    call: "write_generation_streamed",
    source: "memo frame source",
};

/// The lazy `frames` iterator a streamed write hands its backend: it pulls
/// the engine's own iterator, and is invalid once the call returns.
#[pyclass(name = "MemoFrames", module = "incr_python_persist", frozen)]
pub struct Frames {
    lending: Lending<EncodedEntry>,
}

#[pymethods]
impl Frames {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<(u32, u64, Py<PyBytes>)>> {
        Ok(self.lending.pull()?.map(|entry| {
            (
                entry.table,
                entry.key_hash,
                PyBytes::new(py, &entry.frame).unbind(),
            )
        }))
    }
}

/// A [`Frames`] over an engine iterator, invalidated by [`Lent::end`] or
/// by dropping it.
struct Lent {
    frames: Py<Frames>,
}

impl Lent {
    fn new(py: Python<'_>, entries: &mut Pulls<'_>) -> PyResult<Self> {
        let lending = Lending::new(entries, FRAMES);
        Ok(Lent {
            frames: Py::new(py, Frames { lending })?,
        })
    }

    fn end(self, py: Python<'_>) -> Bound<'_, Frames> {
        self.invalidate(py);
        self.frames.bind(py).clone()
    }

    fn invalidate(&self, py: Python<'_>) {
        self.frames.bind(py).get().lending.invalidate();
    }
}

impl Drop for Lent {
    fn drop(&mut self) {
        gil::enter(&LENT_DROP, |py| self.invalidate(py));
    }
}
