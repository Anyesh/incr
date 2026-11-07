//! Backends: [`PyBackend`] adapts any Python object that implements the
//! backend protocol to incr-persist's `Backend`, and `MemoryBackend` is a
//! reference implementation of that protocol over incr-persist's own.
//!
//! The protocol, with records as `bytes` and positions as `int`:
//! `append(records) -> int`, `sync(upto)`, `read_from(after)` returning a
//! list of `(position, record)` or `("compacted", snapshot_position)`,
//! `load_snapshot() -> (position, records) | None`,
//! `snapshot_id() -> (position, generation) | None`,
//! `compact(upto, records)` and `writable() -> bool`. A raised exception
//! fails the store call it happened in and reaches the caller as itself,
//! except that `incr.persist.ReadOnly` (a write on a read-only handle) and
//! `InvalidCompaction(upto, snapshot_at, tail)` (a `compact` outside the
//! snapshot and the tail) are the store's own errors, which its logic
//! branches on. `writable()` is the exception to raising: incr-persist's
//! trait gives it no error channel, so an exception from it reads as
//! `False`. The store itself never calls it (only the conformance cases
//! do), so a read-only handle is known by its `append` and `compact`
//! raising `ReadOnly`.

use std::fmt;

use incr_persist::{
    Backend, Error, MemoryBackend, Position, ReadOutcome, Record, Result, SnapshotId,
};
use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString, PyTuple};

use super::{persist_error, Busy, InvalidCompaction, ReadOnly};

/// A Python exception a backend method raised, carried through
/// incr-persist's `Error::Backend` so the store call raises it unchanged.
#[derive(Debug)]
pub(crate) struct PyBackendError(PyErr);

impl PyBackendError {
    pub(crate) fn err(&self) -> &PyErr {
        &self.0
    }
}

impl fmt::Display for PyBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Python::attach(|py| {
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
pub(super) fn raised(err: PyErr) -> Error {
    if let Some(known) = Python::attach(|py| well_known(py, &err)) {
        return known;
    }
    Error::Backend(Box::new(PyBackendError(err)))
}

fn well_known(py: Python<'_>, err: &PyErr) -> Option<Error> {
    if err.is_instance_of::<Busy>(py) {
        return Some(Error::Busy);
    }
    if err.is_instance_of::<ReadOnly>(py) {
        return Some(Error::ReadOnly);
    }
    if err.is_instance_of::<InvalidCompaction>(py) {
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

pub(super) fn bad_return(what: &str) -> Error {
    raised(PyTypeError::new_err(format!(
        "backend {what} returned a value of the wrong shape"
    )))
}

pub(crate) struct PyBackend {
    backend: Py<PyAny>,
}

impl PyBackend {
    pub(crate) fn new(backend: Py<PyAny>) -> Self {
        PyBackend { backend }
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
            .map_err(raised)
    }
}

fn encoded<'py>(py: Python<'py>, records: &[Record]) -> Result<Bound<'py, PyList>> {
    let items = records
        .iter()
        .map(|record| Ok(PyBytes::new(py, &record.encode()?)))
        .collect::<Result<Vec<_>>>()?;
    PyList::new(py, items).map_err(raised)
}

fn decoded(object: &Bound<'_, PyAny>) -> Result<Record> {
    let bytes = object
        .downcast::<PyBytes>()
        .map_err(|_| bad_return("record"))?;
    Record::decode(bytes.as_bytes())
}

fn decoded_all(object: &Bound<'_, PyAny>) -> Result<Vec<Record>> {
    object
        .try_iter()
        .map_err(raised)?
        .map(|item| decoded(&item.map_err(raised)?))
        .collect()
}

fn position(object: &Bound<'_, PyAny>, what: &str) -> Result<Position> {
    object
        .extract::<u64>()
        .map(Position)
        .map_err(|_| bad_return(what))
}

impl Backend for PyBackend {
    fn append(&self, records: &[Record]) -> Result<Position> {
        Python::attach(|py| {
            let list = encoded(py, records)?;
            position(&self.call(py, intern!(py, "append"), (list,))?, "append")
        })
    }

    fn sync(&self, upto: Position) -> Result<()> {
        Python::attach(|py| self.call(py, intern!(py, "sync"), (upto.0,)).map(drop))
    }

    fn read_from(&self, after: Position, out: &mut Vec<(Position, Record)>) -> Result<ReadOutcome> {
        Python::attach(|py| {
            let result = self.call(py, intern!(py, "read_from"), (after.0,))?;
            if let Ok(tuple) = result.downcast::<PyTuple>() {
                if let Some(snapshot_at) = compacted(tuple)? {
                    return Ok(ReadOutcome::Compacted { snapshot_at });
                }
            }
            let mut found = Vec::new();
            for item in result.try_iter().map_err(raised)? {
                let pair = item.map_err(raised)?;
                let pair = pair
                    .downcast::<PyTuple>()
                    .map_err(|_| bad_return("read_from"))?;
                if pair.len() != 2 {
                    return Err(bad_return("read_from"));
                }
                let at = position(&pair.get_item(0).map_err(raised)?, "read_from")?;
                found.push((at, decoded(&pair.get_item(1).map_err(raised)?)?));
            }
            out.extend(found);
            Ok(ReadOutcome::Records)
        })
    }

    fn load_snapshot(&self) -> Result<Option<(Position, Vec<Record>)>> {
        Python::attach(|py| {
            let result = self.call(py, intern!(py, "load_snapshot"), ())?;
            if result.is_none() {
                return Ok(None);
            }
            let pair = result
                .downcast::<PyTuple>()
                .map_err(|_| bad_return("load_snapshot"))?;
            if pair.len() != 2 {
                return Err(bad_return("load_snapshot"));
            }
            let at = position(&pair.get_item(0).map_err(raised)?, "load_snapshot")?;
            Ok(Some((at, decoded_all(&pair.get_item(1).map_err(raised)?)?)))
        })
    }

    fn snapshot_id(&self) -> Result<Option<SnapshotId>> {
        Python::attach(|py| {
            let result = self.call(py, intern!(py, "snapshot_id"), ())?;
            if result.is_none() {
                return Ok(None);
            }
            let (at, generation) = result
                .extract::<(u64, u64)>()
                .map_err(|_| bad_return("snapshot_id"))?;
            Ok(Some(SnapshotId {
                position: Position(at),
                generation,
            }))
        })
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> Result<()> {
        Python::attach(|py| {
            let list = encoded(py, &folded)?;
            self.call(py, intern!(py, "compact"), (upto.0, list))
                .map(drop)
        })
    }

    fn writable(&self) -> bool {
        Python::attach(|py| {
            self.call(py, intern!(py, "writable"), ())
                .ok()
                .and_then(|writable| writable.is_truthy().ok())
                .unwrap_or(false)
        })
    }
}

fn compacted(tuple: &Bound<'_, PyTuple>) -> Result<Option<Position>> {
    if tuple.len() != 2 {
        return Ok(None);
    }
    let first = tuple.get_item(0).map_err(raised)?;
    match first.downcast::<PyString>() {
        Ok(tag) if tag.to_str().is_ok_and(|tag| tag == "compacted") => {
            position(&tuple.get_item(1).map_err(raised)?, "read_from").map(Some)
        }
        _ => Ok(None),
    }
}

/// A log held in process memory: the reference implementation of the
/// backend protocol, for tests and development. Stores opened on the same
/// object share its log; `read_only()` gives a handle that cannot write.
#[pyclass(name = "MemoryBackend", module = "incr.persist")]
pub(crate) struct PyMemoryBackend {
    inner: MemoryBackend,
}

#[pymethods]
impl PyMemoryBackend {
    #[new]
    fn new() -> Self {
        PyMemoryBackend {
            inner: MemoryBackend::new(),
        }
    }

    fn read_only(&self) -> Self {
        PyMemoryBackend {
            inner: self.inner.read_only(),
        }
    }

    fn append(&self, py: Python<'_>, records: Vec<Vec<u8>>) -> PyResult<u64> {
        let records = records
            .iter()
            .map(|bytes| Record::decode(bytes))
            .collect::<Result<Vec<_>>>()
            .map_err(|e| persist_error(py, e))?;
        self.inner
            .append(&records)
            .map(|at| at.0)
            .map_err(|e| persist_error(py, e))
    }

    fn sync(&self, py: Python<'_>, upto: u64) -> PyResult<()> {
        self.inner
            .sync(Position(upto))
            .map_err(|e| persist_error(py, e))
    }

    fn read_from(&self, py: Python<'_>, after: u64) -> PyResult<Py<PyAny>> {
        let mut found = Vec::new();
        let outcome = self
            .inner
            .read_from(Position(after), &mut found)
            .map_err(|e| persist_error(py, e))?;
        if let ReadOutcome::Compacted { snapshot_at } = outcome {
            return Ok(("compacted", snapshot_at.0)
                .into_pyobject(py)?
                .into_any()
                .unbind());
        }
        let items = found
            .iter()
            .map(|(at, record)| {
                let bytes = record.encode().map_err(|e| persist_error(py, e))?;
                Ok((at.0, PyBytes::new(py, &bytes)))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, items)?.into_any().unbind())
    }

    fn load_snapshot(&self, py: Python<'_>) -> PyResult<Option<(u64, Vec<Py<PyBytes>>)>> {
        let Some((at, records)) = self
            .inner
            .load_snapshot()
            .map_err(|e| persist_error(py, e))?
        else {
            return Ok(None);
        };
        let records = records
            .iter()
            .map(|record| {
                let bytes = record.encode().map_err(|e| persist_error(py, e))?;
                Ok(PyBytes::new(py, &bytes).unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Some((at.0, records)))
    }

    fn snapshot_id(&self, py: Python<'_>) -> PyResult<Option<(u64, u64)>> {
        Ok(self
            .inner
            .snapshot_id()
            .map_err(|e| persist_error(py, e))?
            .map(|id| (id.position.0, id.generation)))
    }

    fn compact(&self, py: Python<'_>, upto: u64, folded: Vec<Vec<u8>>) -> PyResult<()> {
        let folded = folded
            .iter()
            .map(|bytes| Record::decode(bytes))
            .collect::<Result<Vec<_>>>()
            .map_err(|e| persist_error(py, e))?;
        self.inner
            .compact(Position(upto), folded)
            .map_err(|e| persist_error(py, e))
    }

    fn writable(&self) -> bool {
        self.inner.writable()
    }
}
