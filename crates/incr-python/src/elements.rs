//! How collection operators treat not-ready elements. An element stands
//! for a row: a ready element is its own row, an `Unavailable` element with
//! a `row` and an `Incomplete` element (whose row is its `value`) stand for
//! theirs. Keyed operators key on the row, and `map` and `filter` apply to
//! it; a not-ready element with no row has no key and passes `map` and
//! `filter` unchanged. Totals leave not-ready elements out and count them.

use incr_python_protocol::notready::{
    incomplete, status, unavailable, with_missing, PyIncomplete, PyUnavailable, Status,
};
use incr_python_protocol::or_stash;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

/// What an element adds to the row it stands for.
pub(crate) enum Wrap {
    Ready,
    Unavailable(Py<PyAny>),
    Incomplete(u64),
}

/// The row `element` stands for and how it wraps it, or `None` for a
/// not-ready element with no row.
pub(crate) fn parts<'py>(element: &Bound<'py, PyAny>) -> Option<(Bound<'py, PyAny>, Wrap)> {
    match status(element) {
        Status::Ready => Some((element.clone(), Wrap::Ready)),
        Status::Incomplete(i) => {
            let i = i.get();
            Some((
                i.value().bind(element.py()).clone(),
                Wrap::Incomplete(i.missing()),
            ))
        }
        Status::NotReady => {
            let u = element.downcast_exact::<PyUnavailable>().ok()?.get();
            let row = u.row().bind(element.py());
            (!row.is_none()).then(|| {
                (
                    row.clone(),
                    Wrap::Unavailable(u.reason().clone_ref(element.py())),
                )
            })
        }
    }
}

/// `row` wrapped as the element it came from was.
pub(crate) fn rewrap(py: Python<'_>, row: Py<PyAny>, wrap: Wrap) -> Py<PyAny> {
    or_stash(match wrap {
        Wrap::Ready => Ok(row),
        Wrap::Unavailable(reason) => unavailable(py, reason, row),
        Wrap::Incomplete(missing) => with_missing(py, row, missing),
    })
}

/// The row a keyed operator keys `element` on. Raises `TypeError` into the
/// engine for a not-ready element with no row.
pub(crate) fn key_row<'py>(element: &Bound<'py, PyAny>) -> Bound<'py, PyAny> {
    match parts(element) {
        Some((row, _)) => row,
        None => or_stash(Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "a not-ready element with no row has no key: {}",
            element
                .repr()
                .map_or_else(|_| "<unprintable>".to_owned(), |r| r.to_string())
        )))),
    }
}

/// `map`'s image of `element`: `f` of its row, wrapped as the element was.
pub(crate) fn map(
    element: &Bound<'_, PyAny>,
    f: impl FnOnce(Bound<'_, PyAny>) -> Py<PyAny>,
) -> Py<PyAny> {
    match parts(element) {
        Some((row, wrap)) => rewrap(element.py(), f(row), wrap),
        None => element.clone().unbind(),
    }
}

/// Whether `filter` keeps `element`: `pred` of its row, or always for a
/// not-ready element with no row.
pub(crate) fn keep(
    element: &Bound<'_, PyAny>,
    pred: impl FnOnce(Bound<'_, PyAny>) -> bool,
) -> bool {
    parts(element).is_none_or(|(row, _)| pred(row))
}

/// Whether `element` counts toward a `count`'s value.
pub(crate) fn is_ready(element: &Bound<'_, PyAny>) -> bool {
    matches!(status(element), Status::Ready)
}

/// A total's value: `value` itself, or `Incomplete(value, missing)`.
pub(crate) fn total(py: Python<'_>, value: Py<PyAny>, missing: u64) -> Py<PyAny> {
    or_stash(with_missing(py, value, missing))
}

/// `aggregate`'s lift of `element`: `lift` of a ready element, the identity
/// one short for a not-ready element, and an `Incomplete` element's lifted
/// value carrying its `missing`.
pub(crate) fn lift(
    element: &Bound<'_, PyAny>,
    identity: &Py<PyAny>,
    lift: impl FnOnce(Bound<'_, PyAny>) -> Py<PyAny>,
) -> Py<PyAny> {
    let py = element.py();
    match status(element) {
        Status::Ready => lift(element.clone()),
        Status::Incomplete(i) => {
            let i = i.get();
            total(py, lift(i.value().bind(py).clone()), i.missing())
        }
        Status::NotReady => total(py, identity.clone_ref(py), 1),
    }
}

/// `aggregate`'s combine of two lifted values, with the missing counts the
/// lifts carry added and kept out of `combine`'s sight.
pub(crate) fn combine(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    combine: impl FnOnce(Bound<'_, PyAny>, Bound<'_, PyAny>) -> Py<PyAny>,
) -> Py<PyAny> {
    let (ia, ib) = (incomplete(a), incomplete(b));
    if ia.is_none() && ib.is_none() {
        return combine(a.clone(), b.clone());
    }
    let (va, ma) = split(a, ia);
    let (vb, mb) = split(b, ib);
    total(a.py(), combine(va, vb), ma + mb)
}

fn split<'py>(
    value: &Bound<'py, PyAny>,
    incomplete: Option<&Bound<'py, PyIncomplete>>,
) -> (Bound<'py, PyAny>, u64) {
    match incomplete {
        Some(i) => {
            let i = i.get();
            (i.value().bind(value.py()).clone(), i.missing())
        }
        None => (value.clone(), 0),
    }
}

/// `reduce`'s list of rows to fold, and how many contributions it leaves
/// out: one per not-ready element, and an `Incomplete` element's `missing`.
pub(crate) fn foldable<'py, 'a>(
    py: Python<'py>,
    elements: impl Iterator<Item = &'a Py<PyAny>>,
) -> (Bound<'py, PyList>, u64) {
    let list = PyList::empty(py);
    let mut missing = 0;
    for element in elements {
        let element = element.bind(py);
        let row = match status(element) {
            Status::Ready => element.clone(),
            Status::Incomplete(i) => {
                let i = i.get();
                missing += i.missing();
                i.value().bind(py).clone()
            }
            Status::NotReady => {
                missing += 1;
                continue;
            }
        };
        or_stash(list.append(row));
    }
    (list, missing)
}

/// A `join` output pair `(left, right)` of two keyed elements: their rows'
/// pair, `Unavailable` with the first unavailable side's reason if either
/// side is, else carrying both sides' missing counts.
pub(crate) fn pair(py: Python<'_>, left: &Py<PyAny>, right: &Py<PyAny>) -> Py<PyAny> {
    let (left, right) = (left.bind(py), right.bind(py));
    let (Some((lrow, lwrap)), Some((rrow, rwrap))) = (parts(left), parts(right)) else {
        unreachable!("join keys every element on its row")
    };
    let rows = or_stash(PyTuple::new(py, [lrow, rrow])).into_any().unbind();
    let wrap = match (lwrap, rwrap) {
        (Wrap::Unavailable(reason), _) | (_, Wrap::Unavailable(reason)) => {
            Wrap::Unavailable(reason)
        }
        (Wrap::Incomplete(a), Wrap::Incomplete(b)) => Wrap::Incomplete(a + b),
        (Wrap::Incomplete(m), Wrap::Ready) | (Wrap::Ready, Wrap::Incomplete(m)) => {
            Wrap::Incomplete(m)
        }
        (Wrap::Ready, Wrap::Ready) => Wrap::Ready,
    };
    rewrap(py, rows, wrap)
}
