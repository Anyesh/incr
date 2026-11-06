//! The node rule for not-ready values. Inside a query, reading a
//! `NotReady` value raises `_NotReadySignal`, which the query's wrapper
//! turns into that value as the query's own; reading an `Incomplete` value
//! gives its `value` and adds its `missing` to the reading query's count,
//! which the wrapper reports as an `Incomplete` result.

use std::cell::Cell;

use incr_python_protocol::notready::{incomplete, plainly_ready, status, with_missing, Status};
use incr_python_protocol::raise_into_engine;
use pyo3::intern;
use pyo3::prelude::*;

pyo3::create_exception!(
    incr,
    _NotReadySignal,
    pyo3::exceptions::PyBaseException,
    "Carries a not-ready value out of the query that read it; never caught by `except Exception`."
);

/// `value` read inside a query whose missing contributions are `missing`.
#[inline(always)]
pub(crate) fn read(py: Python<'_>, value: Py<PyAny>, missing: &Cell<u64>) -> PyResult<Py<PyAny>> {
    if plainly_ready(value.bind(py)) {
        return Ok(value);
    }
    read_any(py, value, missing)
}

#[inline(never)]
fn read_any(py: Python<'_>, value: Py<PyAny>, missing: &Cell<u64>) -> PyResult<Py<PyAny>> {
    match status(value.bind(py)) {
        Status::Ready => Ok(value),
        Status::NotReady => Err(_NotReadySignal::new_err((value,))),
        Status::Incomplete(incomplete) => {
            let incomplete = incomplete.get();
            missing.set(missing.get() + incomplete.missing());
            Ok(incomplete.value().clone_ref(py))
        }
    }
}

/// The value a query lands from calling `f(rt_ref)`, where `missing` is
/// the count `rt_ref`'s reads add to.
#[inline]
pub(crate) fn call_query(
    py: Python<'_>,
    f: &Py<PyAny>,
    rt_ref: Py<PyAny>,
    missing: &Cell<u64>,
) -> Py<PyAny> {
    let value = f
        .call1(py, (rt_ref,))
        .unwrap_or_else(|e| not_ready_or_raise(py, e));
    match missing.get() {
        0 => value,
        missing => short_by(py, value, missing),
    }
}

/// The value a query whose closure raised `e` lands: the not-ready value
/// `e` carries if it is the signal. Any other exception unwinds the engine,
/// failing the node as before.
#[cold]
#[inline(never)]
pub(crate) fn not_ready_or_raise(py: Python<'_>, e: PyErr) -> Py<PyAny> {
    if !e.is_instance_of::<_NotReadySignal>(py) {
        raise_into_engine(e);
    }
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
pub(crate) fn short_by(py: Python<'_>, value: Py<PyAny>, missing: u64) -> Py<PyAny> {
    with_missing(py, value, missing).unwrap_or_else(|e| raise_into_engine(e))
}

/// `(value, status)`: the value a top-level read returns, and `None`, the
/// `NotReady` value or the `Incomplete` value.
pub(crate) fn with_status(py: Python<'_>, value: Py<PyAny>) -> (Py<PyAny>, Py<PyAny>) {
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
pub(crate) fn unwrapped(py: Python<'_>, value: Py<PyAny>) -> Py<PyAny> {
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
