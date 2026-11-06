//! Values that stand for a result not available yet (`NotReady` and its
//! subclasses) or available with parts missing (`Incomplete`). Equality is
//! by exact type and fields, with fields compared by Python `==`.

use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::PyTuple;
use pyo3::PyClassInitializer;

/// The base of every value standing for a result that is not ready.
#[pyclass(name = "NotReady", module = "incr", subclass, frozen)]
pub struct PyNotReady;

/// A result still being fetched, for `key` when it came from a lookup.
#[pyclass(name = "Pending", module = "incr", extends = PyNotReady, frozen)]
pub struct PyPending {
    #[pyo3(get)]
    key: Py<PyAny>,
}

/// A lookup that failed for `reason`.
#[pyclass(name = "LookupFailed", module = "incr", extends = PyNotReady, frozen)]
pub struct PyLookupFailed {
    #[pyo3(get)]
    reason: Py<PyAny>,
    #[pyo3(get)]
    key: Py<PyAny>,
}

/// A value withheld for `reason`. As a collection element, `row` is the
/// element it stands in for: keyed operators key on it, and `map` and
/// `filter` apply to it.
#[pyclass(name = "Unavailable", module = "incr", extends = PyNotReady, frozen)]
pub struct PyUnavailable {
    #[pyo3(get)]
    reason: Py<PyAny>,
    #[pyo3(get)]
    row: Py<PyAny>,
}

/// `value` computed with `missing` contributions left out because they
/// were not ready.
#[pyclass(name = "Incomplete", module = "incr", frozen)]
pub struct PyIncomplete {
    #[pyo3(get)]
    value: Py<PyAny>,
    #[pyo3(get)]
    missing: u64,
}

fn field_or_none(py: Python<'_>, field: Option<Py<PyAny>>) -> Py<PyAny> {
    field.unwrap_or_else(|| py.None())
}

/// The fields of `a` and `b` are pairwise `==`, for two values of one type.
fn fields_eq(py: Python<'_>, a: &[&Py<PyAny>], b: &[&Py<PyAny>]) -> PyResult<bool> {
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (x.bind(py), y.bind(py));
        if !(x.is(y) || x.eq(y)?) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn richcmp(py: Python<'_>, equal: bool, op: CompareOp) -> PyResult<Py<PyAny>> {
    match op {
        CompareOp::Eq => Ok(equal.into_pyobject(py)?.to_owned().into_any().unbind()),
        CompareOp::Ne => Ok((!equal).into_pyobject(py)?.to_owned().into_any().unbind()),
        _ => Ok(py.NotImplemented()),
    }
}

fn hash_of(py: Python<'_>, tag: &str, fields: &[&Py<PyAny>]) -> PyResult<isize> {
    let mut items = vec![tag.into_pyobject(py)?.into_any().unbind()];
    items.extend(fields.iter().map(|f| f.clone_ref(py)));
    PyTuple::new(py, items)?.hash()
}

fn repr_of(
    py: Python<'_>,
    name: &str,
    args: &[&Py<PyAny>],
    kwargs: &[(&str, &Py<PyAny>)],
) -> PyResult<String> {
    let mut parts = Vec::new();
    for arg in args {
        parts.push(arg.bind(py).repr()?.to_string());
    }
    for (name, value) in kwargs {
        parts.push(format!("{name}={}", value.bind(py).repr()?));
    }
    Ok(format!("{name}({})", parts.join(", ")))
}

#[pymethods]
impl PyPending {
    #[new]
    #[pyo3(signature = (key=None))]
    fn new(py: Python<'_>, key: Option<Py<PyAny>>) -> (Self, PyNotReady) {
        (
            Self {
                key: field_or_none(py, key),
            },
            PyNotReady,
        )
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let equal = match other.downcast_exact::<Self>() {
            Ok(other) => fields_eq(py, &[&self.key], &[&other.get().key])?,
            Err(_) => false,
        };
        richcmp(py, equal, op)
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        hash_of(py, "Pending", &[&self.key])
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        if self.key.is_none(py) {
            return Ok("Pending()".to_owned());
        }
        repr_of(py, "Pending", &[], &[("key", &self.key)])
    }
}

#[pymethods]
impl PyLookupFailed {
    #[new]
    #[pyo3(signature = (reason, key=None))]
    fn new(py: Python<'_>, reason: Py<PyAny>, key: Option<Py<PyAny>>) -> (Self, PyNotReady) {
        (
            Self {
                reason,
                key: field_or_none(py, key),
            },
            PyNotReady,
        )
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let equal = match other.downcast_exact::<Self>() {
            Ok(other) => fields_eq(
                py,
                &[&self.reason, &self.key],
                &[&other.get().reason, &other.get().key],
            )?,
            Err(_) => false,
        };
        richcmp(py, equal, op)
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        hash_of(py, "LookupFailed", &[&self.reason, &self.key])
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        if self.key.is_none(py) {
            return repr_of(py, "LookupFailed", &[&self.reason], &[]);
        }
        repr_of(py, "LookupFailed", &[&self.reason], &[("key", &self.key)])
    }
}

#[pymethods]
impl PyUnavailable {
    #[new]
    #[pyo3(signature = (reason, row=None))]
    fn new(py: Python<'_>, reason: Py<PyAny>, row: Option<Py<PyAny>>) -> (Self, PyNotReady) {
        (
            Self {
                reason,
                row: field_or_none(py, row),
            },
            PyNotReady,
        )
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let equal = match other.downcast_exact::<Self>() {
            Ok(other) => fields_eq(
                py,
                &[&self.reason, &self.row],
                &[&other.get().reason, &other.get().row],
            )?,
            Err(_) => false,
        };
        richcmp(py, equal, op)
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        hash_of(py, "Unavailable", &[&self.reason, &self.row])
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        if self.row.is_none(py) {
            return repr_of(py, "Unavailable", &[&self.reason], &[]);
        }
        repr_of(py, "Unavailable", &[&self.reason], &[("row", &self.row)])
    }
}

#[pymethods]
impl PyIncomplete {
    #[new]
    fn py_new(value: Py<PyAny>, missing: u64) -> PyResult<Self> {
        if missing == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "an Incomplete value has at least one missing contribution",
            ));
        }
        Ok(Self { value, missing })
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let equal = match other.downcast_exact::<Self>() {
            Ok(other) => {
                let other = other.get();
                self.missing == other.missing && fields_eq(py, &[&self.value], &[&other.value])?
            }
            Err(_) => false,
        };
        richcmp(py, equal, op)
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        let missing = self.missing.into_pyobject(py)?.into_any().unbind();
        hash_of(py, "Incomplete", &[&self.value, &missing])
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let missing = self.missing.into_pyobject(py)?.into_any().unbind();
        repr_of(py, "Incomplete", &[&self.value], &[("missing", &missing)])
    }
}

impl PyIncomplete {
    pub fn value(&self) -> &Py<PyAny> {
        &self.value
    }

    pub fn missing(&self) -> u64 {
        self.missing
    }
}

impl PyUnavailable {
    pub fn reason(&self) -> &Py<PyAny> {
        &self.reason
    }

    pub fn row(&self) -> &Py<PyAny> {
        &self.row
    }
}

/// `Pending(key)`.
pub fn pending(py: Python<'_>, key: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let init = PyClassInitializer::from(PyNotReady).add_subclass(PyPending { key });
    Ok(Py::new(py, init)?.into_any())
}

/// `LookupFailed(reason, key)`.
pub fn lookup_failed(py: Python<'_>, reason: Py<PyAny>, key: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let init = PyClassInitializer::from(PyNotReady).add_subclass(PyLookupFailed { reason, key });
    Ok(Py::new(py, init)?.into_any())
}

/// `Unavailable(reason, row)`.
pub fn unavailable(py: Python<'_>, reason: Py<PyAny>, row: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let init = PyClassInitializer::from(PyNotReady).add_subclass(PyUnavailable { reason, row });
    Ok(Py::new(py, init)?.into_any())
}

/// `value` as computed with `missing` contributions left out: `value`
/// itself when none is, an `Incomplete` with the two counts added when
/// `value` is one, and a not-ready `value` unchanged.
pub fn with_missing(py: Python<'_>, value: Py<PyAny>, missing: u64) -> PyResult<Py<PyAny>> {
    if missing == 0 {
        return Ok(value);
    }
    let (value, missing) = match status(value.bind(py)) {
        Status::NotReady => return Ok(value),
        Status::Incomplete(inner) => {
            let inner = inner.get();
            (inner.value.clone_ref(py), inner.missing + missing)
        }
        Status::Ready => (value, missing),
    };
    Ok(Py::new(py, PyIncomplete { value, missing })?.into_any())
}

/// What a value says about its own readiness.
pub enum Status<'a, 'py> {
    Ready,
    NotReady,
    Incomplete(&'a Bound<'py, PyIncomplete>),
}

/// Whether `value` is of a builtin type no not-ready value can have. One
/// load and compare per type, with no call, for the common case.
#[inline(always)]
fn plainly_ready(value: &Bound<'_, PyAny>) -> bool {
    use pyo3::ffi;
    // SAFETY: `value` is a live object; the type statics are only compared
    // by address.
    unsafe {
        let ty = ffi::Py_TYPE(value.as_ptr());
        ty == &raw mut ffi::PyLong_Type
            || ty == &raw mut ffi::PyFloat_Type
            || ty == &raw mut ffi::PyUnicode_Type
            || ty == &raw mut ffi::PyTuple_Type
    }
}

/// Whether `value` is an `Incomplete`.
#[inline]
pub fn incomplete<'a, 'py>(value: &'a Bound<'py, PyAny>) -> Option<&'a Bound<'py, PyIncomplete>> {
    if plainly_ready(value) {
        return None;
    }
    value.downcast_exact::<PyIncomplete>().ok()
}

/// The readiness of `value`. Every not-ready type is a heap type, so a
/// value of a static type is ready without a subclass check.
#[inline]
pub fn status<'a, 'py>(value: &'a Bound<'py, PyAny>) -> Status<'a, 'py> {
    if plainly_ready(value) {
        return Status::Ready;
    }
    if let Ok(incomplete) = value.downcast_exact::<PyIncomplete>() {
        return Status::Incomplete(incomplete);
    }
    // SAFETY: `value` is a live object, so its type is a live type object.
    let flags = unsafe { pyo3::ffi::PyType_GetFlags(pyo3::ffi::Py_TYPE(value.as_ptr())) };
    if flags & pyo3::ffi::Py_TPFLAGS_HEAPTYPE == 0 || !value.is_instance_of::<PyNotReady>() {
        Status::Ready
    } else {
        Status::NotReady
    }
}
