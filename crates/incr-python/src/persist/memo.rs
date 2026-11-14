//! The closure a Local memo table runs: the table function called like a
//! query, with the `RuntimeRef` and the entry's key.

use std::cell::Cell;

use incr_core::Local;
use incr_persist::memo::MemoCtx;
use incr_python_protocol::{in_runtime, PyValue};
use pyo3::prelude::*;

use crate::state::PyCallable;
use incr_python_persist::ready;

use crate::{InvalidateRef, PyRuntimeRef};

/// The closure the engine runs for a memo table: `py_func` called with a
/// `RuntimeRef` that is invalidated when it returns, and the entry's key.
pub(super) fn memo_fn(
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
                PyValue(ready::call_memo::<super::Signal, _>(
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
