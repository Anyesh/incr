//! Persistent memoization for a store: memo tables whose entries survive a
//! restart, and the reads of them from Python.
//!
//! A table is declared with `store.memo_table`, which registers its key and
//! value codecs as the slots `MemoKey(name)` and `MemoValue(name)`. A table
//! function runs like a query: it receives the `RuntimeRef` and the entry's
//! key, and reads other entries with `ctx.memo(table, key)`.

use std::cell::Cell;
use std::collections::HashMap;

use incr_compute::Runtime;
use incr_core::Local;
use incr_persist::memo::MemoCtx;
use incr_python_persist::memo::{MemoFn, TableHandle};
use incr_python_protocol::{in_runtime, PyValue};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::store::PyStore;
use super::StoreCore;
use crate::state::{PyCallable, RtState};
use incr_python_persist::ready;

use crate::{InvalidateRef, PyNodeId, PyRuntimeRef};

/// The closure the engine runs for a memo table: `py_func` called with a
/// `RuntimeRef` that is invalidated when it returns, and the entry's key.
fn memo_fn(
    py_func: PyCallable,
) -> impl Fn(&MemoCtx<'_, Local>, &PyValue) -> PyValue + Send + Sync + 'static {
    move |ctx: &MemoCtx<'_, Local>, key: &PyValue| -> PyValue {
        let rt = ctx.rt();
        Python::attach(|py| {
            let missing = Cell::new(0);
            let rt_ref = Py::new(
                py,
                PyRuntimeRef {
                    ptr: Cell::new(rt as *const _),
                    state: py_func.state(),
                    missing: &missing,
                },
            )
            .unwrap();
            let _guard = InvalidateRef(rt_ref.bind(py).clone());
            in_runtime(rt, || {
                PyValue(ready::call_memo(
                    py,
                    rt,
                    &py_func,
                    rt_ref.clone_ref(py).into_any(),
                    key.0.clone_ref(py),
                    &missing,
                ))
            })
        })
    }
}

/// The memo closure maker of a store whose runtime state is `state`.
fn maker(state: RtState) -> impl Fn(Py<PyAny>) -> MemoFn<Local> {
    move |func| Box::new(memo_fn(state.hold(func)))
}

pub(super) fn declare_table(
    slf: &Bound<'_, PyStore>,
    name: &str,
    version: u32,
    key_codec: Py<PyAny>,
    codec: Py<PyAny>,
    func: Py<PyAny>,
) -> PyResult<PyMemoTable> {
    let py = slf.py();
    let store = slf.try_borrow()?;
    let core = store.core_outside_overlay(py)?;
    let state = store.runtime.bind(py).borrow().state.clone();
    let handle = store.memo.declare_table(
        &core,
        py,
        name,
        version,
        key_codec,
        codec,
        func,
        maker(state),
    )?;
    Ok(PyMemoTable {
        store: slf.clone().unbind(),
        name: name.to_owned(),
        handle,
    })
}

pub(super) fn open(slf: &Bound<'_, PyStore>, backend: Py<PyAny>) -> PyResult<PyMemo> {
    let py = slf.py();
    let store = slf.try_borrow()?;
    let core = store.core_outside_overlay(py)?;
    let state = store.runtime.bind(py).borrow().state.clone();
    store.memo.open(&core, py, backend, maker(state))?;
    Ok(PyMemo {
        store: slf.clone().unbind(),
    })
}

/// A memo table of a store: pass it to `memo.get_or_insert` and
/// `ctx.memo`.
#[pyclass(name = "MemoTable", module = "incr.persist", unsendable)]
pub(crate) struct PyMemoTable {
    store: Py<PyStore>,
    name: String,
    handle: TableHandle,
}

const _: () = assert!(std::mem::align_of::<PyMemoTable>() <= 16);

impl PyMemoTable {
    /// `key`'s value, restored from the memo or computed, read as a
    /// dependency of whatever query is running on `rt`.
    pub(crate) fn value_of(&self, py: Python<'_>, rt: &Runtime, key: PyValue) -> PyResult<PyValue> {
        let store = self.store.bind(py).try_borrow()?;
        let core = store.core()?;
        store.memo.value_of(core, rt, &self.name, &self.handle, key)
    }
}

#[pymethods]
impl PyMemoTable {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("MemoTable({:?})", self.name)
    }
}

/// An open memo: `memo.get_or_insert(table, key)` gives an entry's node, and
/// `memo.checkpoint()` writes the live entries to the backend.
#[pyclass(name = "Memo", module = "incr.persist", unsendable)]
pub(crate) struct PyMemo {
    store: Py<PyStore>,
}

const _: () = assert!(std::mem::align_of::<PyMemo>() <= 16);

impl PyMemo {
    fn core(&self, py: Python<'_>) -> PyResult<std::sync::Arc<StoreCore>> {
        self.store.bind(py).try_borrow()?.core_outside_overlay(py)
    }
}

#[pymethods]
impl PyMemo {
    /// The node of `table`'s entry for `key`: the live one, or one restored
    /// from the memo, or a new one. Read it with `runtime.get`. Inside an
    /// overlay it computes fresh and records nothing, so no candidate value
    /// is restored from the memo or reaches it.
    fn get_or_insert(
        &self,
        py: Python<'_>,
        table: PyRef<'_, PyMemoTable>,
        key: Py<PyAny>,
    ) -> PyResult<PyNodeId> {
        let core = self.store.bind(py).try_borrow()?.core_live(py)?;
        if !table.store.is(&self.store) {
            return Err(PyValueError::new_err(format!(
                "memo table '{}' belongs to another store",
                table.name
            )));
        }
        let store = self.store.bind(py).try_borrow()?;
        let inner = store.memo.get_or_insert(&core, py, &table.handle, key)?;
        Ok(PyNodeId { inner })
    }

    /// Writes every live entry whose dependencies are persisted to the
    /// backend as the next generation, and returns how many it `written`,
    /// `merged` from the previous generation, `evicted` and so on. Raises
    /// `Busy` when another handle is checkpointing. Raises `ValueError`
    /// naming the table when an entry's key or value did not round-trip
    /// through its codec: the generation was still written, without that
    /// entry, which is recomputed on the next run.
    fn checkpoint(&self, py: Python<'_>) -> PyResult<HashMap<&'static str, usize>> {
        let core = self.core(py)?;
        self.store.bind(py).try_borrow()?.memo.checkpoint(&core, py)
    }
}
