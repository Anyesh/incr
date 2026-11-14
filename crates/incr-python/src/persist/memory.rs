//! `MemoryBackend` and `MemoryMemo`: reference implementations of the
//! backend protocols over incr-persist's in-memory backends, for tests and
//! development.

use incr_persist::memo::{
    decode_header, encode_header, EncodedEntry, GenId, IndexItem, MemoBackend, MemoryMemo,
};
use incr_persist::{Backend, MemoryBackend, Position, ReadOutcome, Record, Result};
use incr_python_persist::memo_backend::{item_tuple, ItemTuple};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

use super::persist_error;

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
