//! Persistent memoization for a store: memo tables whose entries survive a
//! restart, and the reads of them from Python.
//!
//! A table is declared with `store.memo_table`, which registers its key and
//! value codecs as the slots `MemoKey(name)` and `MemoValue(name)`. A table
//! function runs like a query: it receives the `RuntimeRef` and the entry's
//! key, and reads other entries with `ctx.memo(table, key)`.

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use incr_compute::Runtime;
use incr_core::Local;
use incr_persist::memo::{BuildId, Memo, MemoBuilder, MemoCtx, MemoOptions, MemoTable};
use incr_python_protocol::{in_runtime, PyValue};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use super::backend::PyBackend;
use super::codec::Slot;
use super::memo_backend::PyMemoBackend;
use super::persist_error;
use super::store::{PyStore, StoreCore};
use crate::state::{PyCallable, RtState};
use crate::{engine_read, engine_write, ready, InvalidateRef, PyNodeId, PyRuntimeRef};

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

type TableHandle = Rc<Cell<MemoTable<PyValue, PyValue>>>;

/// A declared memo table, kept so a builder consumed by a failed open can
/// be made again. Its handle is shared with the `MemoTable` object, whose
/// handle names the builder that declared it and so is replaced with it.
pub(super) struct MemoDecl {
    name: String,
    version: u32,
    func: Py<PyAny>,
    handle: TableHandle,
}

impl StoreCore {
    fn restore_memo_builder(&self, py: Python<'_>, state: &RtState) {
        let mut builder = self.new_memo_builder();
        for decl in self.memo_decls.borrow().iter() {
            let handle = builder.table::<PyValue, PyValue>(&decl.name, decl.version);
            builder.define(handle, memo_fn(state.hold(decl.func.clone_ref(py))));
            decl.handle.set(handle);
        }
        *self.memo_builder.borrow_mut() = Some(builder);
    }

    fn open_memo(&self) -> PyResult<&Memo<Local, PyBackend>> {
        self.memo
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("no memo is open: call store.open_memo first"))
    }

    fn new_memo_builder(&self) -> MemoBuilder<Local> {
        MemoBuilder::new(MemoOptions::new(BuildId::Fixed(self.build_id)))
    }
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
    if core.memo.get().is_some() {
        return Err(PyRuntimeError::new_err(
            "memo_table must be called before open_memo",
        ));
    }
    let key_slot = Slot::MemoKey(name.to_owned());
    let value_slot = Slot::MemoValue(name.to_owned());
    for (slot, given) in [(&key_slot, &key_codec), (&value_slot, &codec)] {
        if given.is_none(py) {
            return Err(PyValueError::new_err(format!(
                "no codec is registered for {}",
                slot.describe()
            )));
        }
    }
    if core.registry.is_registered(&key_slot) {
        return Err(PyValueError::new_err(format!(
            "memo table '{name}' is already declared"
        )));
    }
    if !func.bind(py).is_callable() {
        return Err(PyTypeError::new_err(
            "a memo table's function must be callable",
        ));
    }
    core.registry.register(key_slot, key_codec)?;
    core.registry.register(value_slot, codec)?;
    let callable = store
        .runtime
        .bind(py)
        .borrow()
        .state
        .hold(func.clone_ref(py));
    let mut builder = core.memo_builder.borrow_mut();
    let builder = builder.get_or_insert_with(|| core.new_memo_builder());
    let handle = builder.table::<PyValue, PyValue>(name, version);
    builder.define(handle, memo_fn(callable));
    let handle = Rc::new(Cell::new(handle));
    core.memo_decls.borrow_mut().push(MemoDecl {
        name: name.to_owned(),
        version,
        func,
        handle: Rc::clone(&handle),
    });
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
    if core.memo.get().is_some() {
        return Err(PyRuntimeError::new_err("the memo is already open"));
    }
    let builder = core
        .memo_builder
        .borrow_mut()
        .take()
        .unwrap_or_else(|| core.new_memo_builder());
    let opened = engine_write(core.runtime(), || {
        core.registry.run(py, || {
            builder.open(&core.store, PyMemoBackend::new(backend))
        })
    })
    .and_then(|opened| opened);
    let memo = match opened {
        Ok(memo) => memo,
        Err(e) => {
            let state = store.runtime.bind(py).borrow().state.clone();
            core.restore_memo_builder(py, &state);
            return Err(e);
        }
    };
    core.memo
        .set(memo)
        .map_err(|_| PyRuntimeError::new_err("the memo is already open"))?;
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
        let memo = core.open_memo()?;
        if !std::ptr::eq(core.runtime(), rt) {
            return Err(PyValueError::new_err(format!(
                "memo table '{}' belongs to the runtime of its store, not this one",
                self.name
            )));
        }
        let value = core.registry.scoped(|| {
            let node = memo.get_or_insert(self.handle.get(), key);
            rt.get(node)
        });
        drop(core.registry.take_failure());
        Ok(value)
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
    fn core(&self, py: Python<'_>) -> PyResult<std::rc::Rc<StoreCore>> {
        self.store.bind(py).try_borrow()?.core_outside_overlay(py)
    }
}

fn mismatch_message(tables: &[(String, &'static str)]) -> String {
    let names = tables
        .iter()
        .map(|(table, part)| format!("memo table '{table}' ({part})"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{names} did not round-trip through the codec (decode(encode(x)) != x): the generation \
         was written without those entries, they are recomputed on the next run, and nothing \
         wrong is stored"
    )
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
        let memo = core.open_memo()?;
        let node = engine_read(core.runtime(), || {
            core.registry
                .scoped(|| memo.get_or_insert(table.handle.get(), PyValue(key)))
        })?;
        drop(core.registry.take_failure());
        Ok(PyNodeId { inner: node })
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
        let memo = core.open_memo()?;
        drop(core.registry.take_failure());
        let done = engine_write(core.runtime(), || {
            core.registry.scoped(|| memo.checkpoint())
        })?;
        let cause = core.registry.take_failure();
        let stats = match done {
            Ok(stats) => stats,
            Err(error) => return Err(persist_error(py, error)),
        };
        let mismatches = memo.take_encode_mismatches();
        if !mismatches.is_empty() {
            let tables = mismatches
                .into_iter()
                .map(|m| {
                    let part = match m.part {
                        incr_persist::MemoPart::Key => "keys",
                        incr_persist::MemoPart::Value => "values",
                    };
                    (m.table, part)
                })
                .collect::<Vec<_>>();
            let err = PyValueError::new_err(mismatch_message(&tables));
            err.set_cause(py, cause);
            return Err(err);
        }
        Ok(HashMap::from([
            ("written", stats.written),
            ("unkept", stats.unkept),
            ("merged", stats.merged),
            ("corrupt", stats.corrupt),
            ("evicted", stats.evicted),
            ("oversized", stats.oversized),
        ]))
    }
}
