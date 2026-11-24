//! Backends: [`PyBackend`] adapts any Python object that implements the
//! backend protocol to incr-persist's `Backend`, and `MemoryBackend` is a
//! reference implementation of that protocol over incr-persist's own.
//!
//! The protocol, with records as `bytes` and positions as `int`:
//! `append(records) -> int`, `sync(upto)`, `read_from(after)` returning a
//! list of `(position, record)` or `("compacted", snapshot_position)`,
//! `load_snapshot() -> (position, records) | None`,
//! `snapshot_id() -> (position, generation) | None`,
//! `compact(upto, records)` and `writable() -> bool`, and optionally:
//!
//! - `append_at(after, records) -> int`, which appends only while the tail
//!   is `after` and otherwise raises `incr.persist.TailMoved`, writing
//!   nothing; `store.batch(apply="on_write")` needs it.
//! - `compact_streamed(upto, records)`, `compact` with `records` a lazy
//!   iterator of `bytes`, so a checkpoint never builds one list of every
//!   record. Pull `records` on the calling thread, during the call: it is
//!   invalid once the call returns, and a call that returns without
//!   exhausting it (reading until `StopIteration`) fails the checkpoint
//!   with `PersistError` and poisons the store. A pull that raises means
//!   the compaction must be abandoned with nothing stored: let the
//!   exception propagate and incr reports the pull's own failure. A call
//!   that returns normally after a pull raised is treated as having stored
//!   a truncated snapshot: the checkpoint raises (a codec's own exception
//!   when a codec raised, else `PersistError`) and the store is poisoned.
//!
//! A raised exception fails the store call it happened in and reaches the caller as itself,
//! except that `incr.persist.ReadOnly` (a write on a read-only handle),
//! `InvalidCompaction(upto, snapshot_at, tail)` (a `compact` outside the
//! snapshot and the tail) and `TailMoved` from `append_at` are the store's
//! own errors, which its logic branches on. `writable()` is the exception to raising: incr-persist's
//! trait gives it no error channel, so an exception from it reads as
//! `False`. The store itself never calls it (only the conformance cases
//! do), so a read-only handle is known by its `append` and `compact`
//! raising `ReadOnly`.

use std::fmt;
use std::marker::PhantomData;
use std::panic;

use incr_core::YieldHook;
use incr_persist::{
    Backend, ConditionalAppend, Error, Position, ReadOutcome, Record, Result, SnapshotId,
};
use incr_python_protocol::gil::{self, Site};
use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyString};

use crate::lent::{self, Failure, Lending, Names};
use crate::PersistExceptions;

static RAISED: Site = Site::one("log.raised");
static ERROR_TEXT: Site = Site::caller("log.error_text");
static APPEND: Site = Site::caller("log.append");
static APPEND_AT: Site = Site::caller("log.append_at");
static SYNC: Site = Site::caller("log.sync");
static READ_FROM: Site = Site::caller("log.read_from");
static LOAD_SNAPSHOT: Site = Site::caller("log.load_snapshot");
static SNAPSHOT_ID: Site = Site::caller("log.snapshot_id");
static COMPACT: Site = Site::caller("log.compact");
static COMPACT_STREAMED: Site = Site::one("log.compact_streamed");
static LENT_DROP: Site = Site::one("log.lent_drop");
static WRITABLE: Site = Site::caller("log.writable");

/// A Python exception a backend method raised, carried through
/// incr-persist's `Error::Backend` so the store call raises it unchanged.
#[derive(Debug)]
pub struct PyBackendError(PyErr);

impl PyBackendError {
    pub fn err(&self) -> &PyErr {
        &self.0
    }
}

impl fmt::Display for PyBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        gil::enter(&ERROR_TEXT, |py| {
            let value = self.0.value(py);
            let name = self.0.get_type(py).name();
            match (name, value.str()) {
                (Ok(name), Ok(message)) => write!(f, "{name}: {message}"),
                _ => f.write_str("a Python exception"),
            }
        })
    }
}

impl std::error::Error for PyBackendError {}

/// `err` as the store's error: `incr.persist.Busy`, `ReadOnly` and
/// `InvalidCompaction` are the store's own variants, so a skipped
/// checkpoint or a rejected write stays one, and any other exception is
/// carried so the call raises it unchanged.
pub fn raised<E: PersistExceptions>(err: PyErr) -> Error {
    if let Some(known) = gil::enter(&RAISED, |py| well_known::<E>(py, &err)) {
        return known;
    }
    Error::Backend(Box::new(PyBackendError(err)))
}

fn well_known<E: PersistExceptions>(py: Python<'_>, err: &PyErr) -> Option<Error> {
    if err.is_instance(py, &E::busy(py)) {
        return Some(Error::Busy);
    }
    if err.is_instance(py, &E::read_only(py)) {
        return Some(Error::ReadOnly);
    }
    if err.is_instance(py, &E::invalid_compaction(py)) {
        let (upto, snapshot_at, tail) = err
            .value(py)
            .getattr("args")
            .and_then(|args| args.extract::<(u64, u64, u64)>())
            .ok()?;
        return Some(Error::InvalidCompaction {
            upto: Position(upto),
            snapshot_at: Position(snapshot_at),
            tail: Position(tail),
        });
    }
    None
}

pub fn bad_return<E: PersistExceptions>(what: &str) -> Error {
    raised::<E>(PyTypeError::new_err(format!(
        "backend {what} returned a value of the wrong shape"
    )))
}

/// The error for a sequence a backend returned: a `TypeError` from walking
/// it means it had the wrong shape, anything else is the backend's own.
pub(crate) fn sequence_error<E: PersistExceptions>(what: &'static str) -> impl Fn(PyErr) -> Error {
    move |err| {
        if gil::enter(&RAISED, |py| err.is_instance_of::<PyTypeError>(py)) {
            bad_return::<E>(what)
        } else {
            raised::<E>(err)
        }
    }
}

/// An argument a backend call may pass: a scalar or a Python object, so a
/// Rust collection is never converted by PyO3 in one hold.
pub trait Arg<'py>: IntoPyObject<'py> {}

impl Arg<'_> for u32 {}
impl Arg<'_> for u64 {}
impl Arg<'_> for Option<u64> {}
impl<'py> Arg<'py> for Bound<'py, PyAny> {}
impl<'py> Arg<'py> for Bound<'py, PyBytes> {}
impl<'py> Arg<'py> for Py<PyAny> {}

/// The arguments of a backend call, each an [`Arg`].
pub trait Args<'py>: pyo3::call::PyCallArgs<'py> {}

impl Args<'_> for () {}
impl<'py, A: Arg<'py>> Args<'py> for (A,) where (A,): pyo3::call::PyCallArgs<'py> {}
impl<'py, A: Arg<'py>, B: Arg<'py>> Args<'py> for (A, B) where (A, B): pyo3::call::PyCallArgs<'py> {}
impl<'py, A: Arg<'py>, B: Arg<'py>, C: Arg<'py>, D: Arg<'py>> Args<'py> for (A, B, C, D) where
    (A, B, C, D): pyo3::call::PyCallArgs<'py>
{
}
impl<'py, A: Arg<'py>, B: Arg<'py>, C: Arg<'py>, D: Arg<'py>, F: Arg<'py>> Args<'py>
    for (A, B, C, D, F)
where
    (A, B, C, D, F): pyo3::call::PyCallArgs<'py>,
{
}

/// Whether `backend`'s class declares `concurrent_compaction` true. Read
/// from the class, so a wrapper that forwards attributes to another
/// backend does not inherit that backend's promise.
pub(crate) fn declares_concurrent(backend: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = backend.py();
    match backend
        .get_type()
        .getattr(intern!(py, "concurrent_compaction"))
    {
        Ok(flag) => flag.is_truthy(),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => Ok(false),
        Err(err) => Err(err),
    }
}

/// A Python object implementing the backend protocol, as a `Backend`
/// whose errors use the exceptions `E`. It offers a `ConditionalAppend`
/// exactly when the object had an `append_at` attribute when wrapped.
pub struct PyBackend<E> {
    backend: Py<PyAny>,
    has_append_at: bool,
    streams: bool,
    concurrent: bool,
    /// The store runtime's hook, ticked per record converted on the
    /// engine's thread and never on a checkpoint worker.
    yield_hook: Option<YieldHook>,
    exceptions: PhantomData<fn() -> E>,
}

impl<E: PersistExceptions> PyBackend<E> {
    /// Wraps `backend`; raises what looking up its `append_at`,
    /// `compact_streamed` or `concurrent_compaction` raises, other than
    /// `AttributeError`.
    pub fn new(py: Python<'_>, backend: Py<PyAny>) -> PyResult<Self> {
        let object = backend.bind(py);
        let has_append_at = object.hasattr(intern!(py, "append_at"))?;
        let streams = object.hasattr(intern!(py, "compact_streamed"))?;
        let concurrent = streams && declares_concurrent(object)?;
        Ok(PyBackend {
            backend,
            has_append_at,
            streams,
            concurrent,
            yield_hook: None,
            exceptions: PhantomData,
        })
    }

    pub fn with_yield_hook(self, yield_hook: Option<YieldHook>) -> Self {
        PyBackend { yield_hook, ..self }
    }

    fn hook(&self) -> Option<&YieldHook> {
        self.yield_hook.as_ref().filter(|_| !gil::on_worker())
    }

    /// Runs a backend method's whole body through `site` with the store
    /// runtime's hook ambient, so the gate's sequence functions tick it.
    fn enter<R>(&self, site: &'static Site, f: impl for<'py> FnOnce(Python<'py>) -> R) -> R {
        incr_core::with_ambient_hook(self.hook(), || gil::enter(site, f))
    }

    fn call_raw<'py>(
        &self,
        py: Python<'py>,
        method: &Bound<'py, PyString>,
        args: impl Args<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.backend.bind(py).call_method1(method, args)
    }

    fn call<'py>(
        &self,
        py: Python<'py>,
        method: &Bound<'py, PyString>,
        args: impl Args<'py>,
    ) -> Result<Bound<'py, PyAny>> {
        self.call_raw(py, method, args).map_err(raised::<E>)
    }
}

fn encoded<'py, E: PersistExceptions>(
    py: Python<'py>,
    site: &'static Site,
    records: &[Record],
) -> Result<Bound<'py, PyAny>> {
    gil::build(py, site, records, raised::<E>, |py, record| {
        Ok(PyBytes::new(py, &record.encode()?).into_any())
    })
}

fn decoded<E: PersistExceptions>(object: &Bound<'_, PyAny>) -> Result<Record> {
    let bytes = object
        .downcast::<PyBytes>()
        .map_err(|_| bad_return::<E>("record"))?;
    Record::decode(bytes.as_bytes())
}

fn position<E: PersistExceptions>(object: &Bound<'_, PyAny>, what: &str) -> Result<Position> {
    object
        .extract::<u64>()
        .map(Position)
        .map_err(|_| bad_return::<E>(what))
}

/// Frees an argument list once the call that took it has returned.
fn emptied<E: PersistExceptions>(list: Bound<'_, PyAny>) -> Result<()> {
    gil::empty(list).map_err(raised::<E>)
}

impl<E: PersistExceptions> Backend for PyBackend<E> {
    fn append(&self, records: &[Record]) -> Result<Position> {
        self.enter(&APPEND, |py| {
            let list = encoded::<E>(py, &APPEND, records)?;
            let returned = self.call(py, intern!(py, "append"), (list.clone(),));
            emptied::<E>(list)?;
            position::<E>(&returned?, "append")
        })
    }

    fn sync(&self, upto: Position) -> Result<()> {
        self.enter(&SYNC, |py| {
            self.call(py, intern!(py, "sync"), (upto.0,)).map(drop)
        })
    }

    fn read_from(&self, after: Position, out: &mut Vec<(Position, Record)>) -> Result<ReadOutcome> {
        self.enter(&READ_FROM, |py| {
            let result = self.call(py, intern!(py, "read_from"), (after.0,))?;
            if let Ok((tag, snapshot_at)) = result.extract::<(Bound<'_, PyString>, u64)>() {
                if tag.to_str().is_ok_and(|tag| tag == "compacted") {
                    return Ok(ReadOutcome::Compacted {
                        snapshot_at: Position(snapshot_at),
                    });
                }
            }
            let found = gil::drain(
                &READ_FROM,
                result,
                sequence_error::<E>("read_from"),
                |pair| {
                    let (at, record) = pair
                        .extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()
                        .map_err(|_| bad_return::<E>("read_from"))?;
                    Ok((position::<E>(&at, "read_from")?, decoded::<E>(&record)?))
                },
            )?;
            out.extend(found);
            Ok(ReadOutcome::Records)
        })
    }

    fn load_snapshot(&self) -> Result<Option<(Position, Vec<Record>)>> {
        self.enter(&LOAD_SNAPSHOT, |py| {
            let result = self.call(py, intern!(py, "load_snapshot"), ())?;
            if result.is_none() {
                return Ok(None);
            }
            let (at, records) = result
                .extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()
                .map_err(|_| bad_return::<E>("load_snapshot"))?;
            drop(result);
            let at = position::<E>(&at, "load_snapshot")?;
            let records = gil::drain(
                &LOAD_SNAPSHOT,
                records,
                sequence_error::<E>("load_snapshot"),
                decoded::<E>,
            )?;
            Ok(Some((at, records)))
        })
    }

    fn snapshot_id(&self) -> Result<Option<SnapshotId>> {
        self.enter(&SNAPSHOT_ID, |py| {
            let result = self.call(py, intern!(py, "snapshot_id"), ())?;
            if result.is_none() {
                return Ok(None);
            }
            let (at, generation) = result
                .extract::<(u64, u64)>()
                .map_err(|_| bad_return::<E>("snapshot_id"))?;
            Ok(Some(SnapshotId {
                position: Position(at),
                generation,
            }))
        })
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> Result<()> {
        self.enter(&COMPACT, |py| {
            let list = encoded::<E>(py, &COMPACT, &folded)?;
            let returned = self.call(py, intern!(py, "compact"), (upto.0, list.clone()));
            emptied::<E>(list)?;
            returned.map(drop)
        })
    }

    fn compact_streamed(
        &self,
        upto: Position,
        records: &mut dyn Iterator<Item = Result<Record>>,
    ) -> Result<()> {
        if !self.streams {
            let folded = records.collect::<Result<Vec<_>>>()?;
            return self.compact(upto, folded);
        }
        self.enter(&COMPACT_STREAMED, |py| {
            let lent = Lent::new(py, records).map_err(raised::<E>)?;
            let result = self.call(
                py,
                intern!(py, "compact_streamed"),
                (upto.0, lent.records.clone_ref(py).into_any()),
            );
            let records = lent.end(py);
            match (records.get().lending.failure(), result) {
                (Some(Failure::Panic(payload)), _) => panic::resume_unwind(payload),
                (Some(Failure::Error(error)), Err(_)) => Err(error),
                // A call that returns normally after a failed pull may have
                // committed what it pulled; the store poisons on this `Ok`
                // when its fold failed.
                (Some(Failure::Error(_)), Ok(_)) => Ok(()),
                (None, result) => result.map(drop),
            }
        })
    }

    fn writable(&self) -> bool {
        self.enter(&WRITABLE, |py| {
            self.call(py, intern!(py, "writable"), ())
                .ok()
                .and_then(|writable| writable.is_truthy().ok())
                .unwrap_or(false)
        })
    }

    fn concurrent_compaction(&self) -> bool {
        self.concurrent
    }

    fn conditional_append(&self) -> Option<&dyn ConditionalAppend> {
        self.has_append_at.then_some(self as &dyn ConditionalAppend)
    }
}

impl<E: PersistExceptions> ConditionalAppend for PyBackend<E> {
    fn append_at(&self, after: Position, records: &[Record]) -> Result<Position> {
        self.enter(&APPEND_AT, |py| {
            let list = encoded::<E>(py, &APPEND_AT, records)?;
            let landed = self.call_raw(py, intern!(py, "append_at"), (after.0, list.clone()));
            let landed = match landed {
                Ok(landed) => position::<E>(&landed, "append_at"),
                Err(err) if err.is_instance(py, &E::tail_moved(py)) => {
                    Err(Error::TailMoved { after })
                }
                Err(err) => Err(raised::<E>(err)),
            };
            emptied::<E>(list)?;
            landed
        })
    }
}

type Pulls<'a> = lent::Pulls<'a, Record>;

const RECORDS: Names = Names {
    items: "compaction records",
    call: "compact_streamed",
    source: "compaction record source",
};

/// The lazy `records` iterator a streamed compaction hands its backend:
/// it pulls the store's own iterator and encodes each record as `compact`'s
/// list does, and is invalid once the call returns.
#[pyclass(name = "CompactionRecords", module = "incr_python_persist", frozen)]
pub struct Records {
    lending: Lending<Record>,
}

#[pymethods]
impl Records {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Py<PyBytes>>> {
        match self.lending.pull()? {
            Some(record) => match record.encode() {
                Ok(bytes) => Ok(Some(PyBytes::new(py, &bytes).unbind())),
                Err(error) => Err(self.lending.fail(error)),
            },
            None => Ok(None),
        }
    }
}

struct Lent {
    records: Py<Records>,
}

impl Lent {
    fn new(py: Python<'_>, records: &mut Pulls<'_>) -> PyResult<Self> {
        let lending = Lending::new(records, RECORDS);
        Ok(Lent {
            records: Py::new(py, Records { lending })?,
        })
    }

    fn end(self, py: Python<'_>) -> Bound<'_, Records> {
        self.invalidate(py);
        self.records.bind(py).clone()
    }

    fn invalidate(&self, py: Python<'_>) {
        self.records.bind(py).get().lending.invalidate();
    }
}

impl Drop for Lent {
    fn drop(&mut self) {
        gil::enter(&LENT_DROP, |py| self.invalidate(py));
    }
}
