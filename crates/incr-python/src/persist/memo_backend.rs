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
//! `lock_checkpoint` or `write_generation` skips the checkpoint; any other
//! exception fails it and reaches the caller as itself.

use incr_persist::memo::{
    decode_header, encode_header, EncodedEntry, GenId, Generation, IndexItem, MemoBackend,
    MemoryMemo,
};
use incr_persist::Result;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString};

use super::backend::{bad_return, raised};
use super::persist_error;

type ItemTuple = (u32, u64, u64, u32);

fn item_tuple(item: &IndexItem) -> ItemTuple {
    (item.table, item.key_hash, item.loc, item.len)
}

fn item_of(object: &Bound<'_, PyAny>, what: &str) -> Result<IndexItem> {
    let (table, key_hash, loc, len) = object
        .extract::<ItemTuple>()
        .map_err(|_| bad_return(what))?;
    Ok(IndexItem {
        table,
        key_hash,
        loc,
        len,
    })
}

fn items_of(object: &Bound<'_, PyAny>, what: &str) -> Result<Vec<IndexItem>> {
    object
        .try_iter()
        .map_err(|_| bad_return(what))?
        .map(|item| item_of(&item.map_err(raised)?, what))
        .collect()
}

pub(crate) struct PyMemoBackend {
    backend: Py<PyAny>,
}

impl PyMemoBackend {
    pub(crate) fn new(backend: Py<PyAny>) -> Self {
        PyMemoBackend { backend }
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

/// The object `lock_checkpoint()` returned, released when the checkpoint
/// drops it.
pub(crate) struct PyMemoGuard {
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

impl MemoBackend for PyMemoBackend {
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
                .map_err(|_| bad_return("load"))?;
            let id = GenId(id);
            let Ok(header) = decode_header(&header) else {
                return Ok(Some(Generation::empty(id)));
            };
            Ok(Some(Generation::new(id, header, items_of(&items, "load")?)))
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
                .map_err(|_| bad_return("read_entry"))?;
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
                    PyList::new(py, frames).map_err(raised)?,
                ),
            )?;
            let items = items_of(&result, "write_generation")?;
            Ok(Generation::new(id, header.clone(), items))
        })
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        Python::attach(|py| {
            self.call(py, intern!(py, "remove_generation"), (generation.0,))
                .map(drop)
        })
    }
}

/// A memo store held in process memory: the reference implementation of
/// the memo backend protocol, for tests and development. Stores opened on
/// the same object share its generations.
#[pyclass(name = "MemoryMemo", module = "incr.persist")]
pub(crate) struct PyMemoryMemo {
    inner: MemoryMemo,
}

#[pyclass(name = "MemoryMemoGuard", module = "incr.persist")]
pub(crate) struct PyMemoryMemoGuard {
    guard: Option<<MemoryMemo as MemoBackend>::Guard>,
}

#[pymethods]
impl PyMemoryMemoGuard {
    fn release(&mut self) {
        self.guard.take();
    }
}

#[pymethods]
impl PyMemoryMemo {
    #[new]
    fn new() -> Self {
        PyMemoryMemo {
            inner: MemoryMemo::new(),
        }
    }

    fn lock_checkpoint(&self, py: Python<'_>) -> PyResult<PyMemoryMemoGuard> {
        let guard = self
            .inner
            .lock_checkpoint()
            .map_err(|e| persist_error(py, e))?;
        Ok(PyMemoryMemoGuard { guard: Some(guard) })
    }

    #[allow(clippy::type_complexity)]
    fn load(&self, py: Python<'_>) -> PyResult<Option<(u64, Py<PyBytes>, Vec<ItemTuple>)>> {
        let Some(generation) = self.inner.load().map_err(|e| persist_error(py, e))? else {
            return Ok(None);
        };
        let header = encode_header(generation.header()).map_err(|e| persist_error(py, e))?;
        Ok(Some((
            generation.id().0,
            PyBytes::new(py, &header).unbind(),
            generation.items().iter().map(item_tuple).collect(),
        )))
    }

    fn read_entry(
        &self,
        py: Python<'_>,
        generation: u64,
        item: ItemTuple,
    ) -> PyResult<Option<Py<PyBytes>>> {
        let item = IndexItem {
            table: item.0,
            key_hash: item.1,
            loc: item.2,
            len: item.3,
        };
        let frame = self
            .inner
            .read_entry(GenId(generation), &item)
            .map_err(|e| persist_error(py, e))?;
        Ok(frame.map(|frame| PyBytes::new(py, &frame).unbind()))
    }

    fn write_generation(
        &self,
        py: Python<'_>,
        base: Option<u64>,
        generation: u64,
        header: Vec<u8>,
        entries: Vec<(u32, u64, Vec<u8>)>,
    ) -> PyResult<Vec<ItemTuple>> {
        let header = decode_header(&header).map_err(|e| persist_error(py, e))?;
        let entries = entries
            .into_iter()
            .map(|(table, key_hash, frame)| EncodedEntry {
                table,
                key_hash,
                frame,
            })
            .collect::<Vec<_>>();
        let written = self
            .inner
            .write_generation(base.map(GenId), GenId(generation), &header, &entries)
            .map_err(|e| persist_error(py, e))?;
        Ok(written.items().iter().map(item_tuple).collect())
    }

    fn remove_generation(&self, py: Python<'_>, generation: u64) -> PyResult<()> {
        self.inner
            .remove_generation(GenId(generation))
            .map_err(|e| persist_error(py, e))
    }
}
