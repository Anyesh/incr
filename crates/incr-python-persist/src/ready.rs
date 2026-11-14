//! The node rule for not-ready values. Inside a query, reading a
//! `NotReady` value raises `_NotReadySignal`, which the query's wrapper
//! turns into that value as the query's own; reading an `Incomplete` value
//! gives its `value` and adds its `missing` to the reading query's count,
//! which the wrapper reports as an `Incomplete` result.

use std::cell::Cell;

use incr_core::{Cells, Runtime};
use incr_python_protocol::notready::{incomplete, plainly_ready, status, with_missing, Status};
use incr_python_protocol::raise_into_engine;
use pyo3::intern;
use pyo3::prelude::*;

use pyo3::types::PyType;

/// A package's `_NotReadySignal` exception type. Implemented by
/// [`not_ready_signal!`](crate::not_ready_signal).
pub trait NotReadySignal: 'static {
    fn signal(py: Python<'_>) -> Bound<'_, PyType>;
}

/// Declares `_NotReadySignal` under the Python module path `$module` (the
/// package's top-level module, such as `incr`) and a unit type `$name`
/// implementing [`NotReadySignal`] for it.
#[macro_export]
macro_rules! not_ready_signal {
    ($name:ident, $module:expr) => {
        $crate::pyo3::create_exception!(
            $module,
            _NotReadySignal,
            $crate::pyo3::exceptions::PyBaseException,
            "Carries a not-ready value out of the query that read it; never caught by `except Exception`."
        );

        pub(crate) struct $name;

        impl $crate::ready::NotReadySignal for $name {
            fn signal(
                py: $crate::pyo3::Python<'_>,
            ) -> $crate::pyo3::Bound<'_, $crate::pyo3::types::PyType> {
                py.get_type::<_NotReadySignal>()
            }
        }
    };
}

/// `value` read inside a query whose missing contributions are `missing`.
#[inline(always)]
pub fn read<S: NotReadySignal>(
    py: Python<'_>,
    value: Py<PyAny>,
    missing: &Cell<u64>,
) -> PyResult<Py<PyAny>> {
    if plainly_ready(value.bind(py)) {
        return Ok(value);
    }
    read_any::<S>(py, value, missing)
}

#[inline(never)]
fn read_any<S: NotReadySignal>(
    py: Python<'_>,
    value: Py<PyAny>,
    missing: &Cell<u64>,
) -> PyResult<Py<PyAny>> {
    match status(value.bind(py)) {
        Status::Ready => Ok(value),
        Status::NotReady => Err(PyErr::from_type(S::signal(py), (value,))),
        Status::Incomplete(incomplete) => {
            let incomplete = incomplete.get();
            missing.set(missing.get() + incomplete.missing());
            Ok(incomplete.value().clone_ref(py))
        }
    }
}

/// The value a query of `rt` lands from calling `f(rt_ref)`, where
/// `missing` is the count `rt_ref`'s reads add to.
#[inline]
pub fn call_query<S: NotReadySignal, C: Cells>(
    py: Python<'_>,
    rt: &Runtime<C>,
    f: &Py<PyAny>,
    rt_ref: Py<PyAny>,
    missing: &Cell<u64>,
) -> Py<PyAny> {
    let value = f
        .call1(py, (rt_ref,))
        .unwrap_or_else(|e| not_ready_or_raise::<S, C>(py, rt, e));
    match missing.get() {
        0 => value,
        missing => short_by(py, value, missing),
    }
}

/// [`call_query`] for a memo table's function, which takes the entry's
/// `key` after the ref.
pub fn call_memo<S: NotReadySignal, C: Cells>(
    py: Python<'_>,
    rt: &Runtime<C>,
    f: &Py<PyAny>,
    rt_ref: Py<PyAny>,
    key: Py<PyAny>,
    missing: &Cell<u64>,
) -> Py<PyAny> {
    let value = f
        .call1(py, (rt_ref, key))
        .unwrap_or_else(|e| not_ready_or_raise::<S, C>(py, rt, e));
    match missing.get() {
        0 => value,
        missing => short_by(py, value, missing),
    }
}

/// The value a query of `rt` whose closure raised `e` lands: the
/// not-ready value `e` carries if it is the signal, which stopped the
/// closure before it asked for all its children, so they are kept. Any
/// other exception unwinds the engine, failing the node as before.
#[cold]
#[inline(never)]
pub fn not_ready_or_raise<S: NotReadySignal, C: Cells>(
    py: Python<'_>,
    rt: &Runtime<C>,
    e: PyErr,
) -> Py<PyAny> {
    if !e.is_instance(py, S::signal(py).as_any()) {
        raise_into_engine(e);
    }
    rt.keep_untouched_children();
    e.value(py)
        .getattr(intern!(py, "args"))
        .and_then(|args| args.get_item(0))
        .map(Bound::unbind)
        .unwrap_or_else(|e| raise_into_engine(e))
}

/// The value a query lands when it returned `value` having read `missing`
/// contributions short.
#[cold]
#[inline(never)]
pub fn short_by(py: Python<'_>, value: Py<PyAny>, missing: u64) -> Py<PyAny> {
    with_missing(py, value, missing).unwrap_or_else(|e| raise_into_engine(e))
}

/// `(value, status)`: the value a top-level read returns, and `None`, the
/// `NotReady` value or the `Incomplete` value.
pub fn with_status(py: Python<'_>, value: Py<PyAny>) -> (Py<PyAny>, Py<PyAny>) {
    match status(value.bind(py)) {
        Status::Ready => (value, py.None()),
        Status::NotReady => (value.clone_ref(py), value),
        Status::Incomplete(incomplete) => {
            let inner = incomplete.get().value().clone_ref(py);
            (inner, value)
        }
    }
}

/// The value a top-level read returns: an `Incomplete` value's `value`,
/// anything else as stored.
#[inline(always)]
pub fn unwrapped(py: Python<'_>, value: Py<PyAny>) -> Py<PyAny> {
    if plainly_ready(value.bind(py)) {
        return value;
    }
    unwrapped_any(py, value)
}

#[inline(never)]
fn unwrapped_any(py: Python<'_>, value: Py<PyAny>) -> Py<PyAny> {
    match incomplete(value.bind(py)) {
        Some(incomplete) => incomplete.get().value().clone_ref(py),
        None => value,
    }
}
