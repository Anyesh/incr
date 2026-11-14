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

use std::marker::PhantomData;

use incr_persist::memo::{
    decode_header, encode_header, EncodedEntry, GenId, Generation, IndexItem, MemoBackend,
};
use incr_persist::Result;
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
    exceptions: PhantomData<fn() -> E>,
}

impl<E: PersistExceptions> PyMemoBackend<E> {
    pub fn new(backend: Py<PyAny>) -> Self {
        PyMemoBackend {
            backend,
            exceptions: PhantomData,
        }
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

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        Python::attach(|py| {
            self.call(py, intern!(py, "remove_generation"), (generation.0,))
                .map(drop)
        })
    }
}
