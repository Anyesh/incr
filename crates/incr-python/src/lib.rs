use pyo3::prelude::*;
use std::hash::{Hash, Hasher};

// ── PyValue: wraps a Python object for use as a value in the Rust engine ────

struct PyValue(PyObject);

impl Clone for PyValue {
    fn clone(&self) -> Self {
        Python::with_gil(|py| PyValue(self.0.clone_ref(py)))
    }
}

impl PartialEq for PyValue {
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| self.0.bind(py).eq(other.0.bind(py)).unwrap_or(false))
    }
}

impl Eq for PyValue {}

impl Hash for PyValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Python::with_gil(|py| {
            let h: isize = self.0.bind(py).hash().unwrap_or(0);
            state.write_isize(h);
        });
    }
}

// ── PyNodeId: typed handle exposed to Python ────────────────────────────────

#[pyclass(name = "NodeId")]
#[derive(Clone)]
struct PyNodeId {
    inner: incr_core::Incr<PyValue>,
}

#[pymethods]
impl PyNodeId {
    #[getter]
    fn id(&self) -> u32 {
        self.inner.node_id().raw()
    }
}

// ── PyRuntimeRef: temporary reference passed into query callbacks ────────────

#[pyclass(name = "RuntimeRef", unsendable)]
struct PyRuntimeRef {
    ptr: *const incr_core::Runtime,
}

#[pymethods]
impl PyRuntimeRef {
    fn get(&self, node: PyNodeId) -> PyResult<PyObject> {
        if self.ptr.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "RuntimeRef is no longer valid (used outside query callback)",
            ));
        }
        let rt = unsafe { &*self.ptr };
        let val: PyValue = rt.get(node.inner);
        Ok(val.0)
    }
}

// ── PyCollection: wraps IncrCollection<PyValue> ─────────────────────────────

#[pyclass(name = "Collection", unsendable)]
struct PyCollection {
    inner: incr_core::IncrCollection<PyValue>,
    rt_ptr: *const incr_core::Runtime,
}

#[pymethods]
impl PyCollection {
    fn insert(&self, value: PyObject) {
        let rt = unsafe { &*self.rt_ptr };
        self.inner.insert(rt, PyValue(value));
    }

    fn delete(&self, value: PyObject) {
        let rt = unsafe { &*self.rt_ptr };
        self.inner.delete(rt, &PyValue(value));
    }

    fn filter(&self, predicate: PyObject) -> PyResult<PyCollection> {
        let rt = unsafe { &*self.rt_ptr };
        let filtered = self.inner.filter(rt, move |val: &PyValue| -> bool {
            Python::with_gil(|py| {
                predicate
                    .call1(py, (val.0.clone_ref(py),))
                    .and_then(|r| r.is_truthy(py))
                    .unwrap_or(false)
            })
        });
        Ok(PyCollection {
            inner: filtered,
            rt_ptr: self.rt_ptr,
        })
    }

    fn map(&self, func: PyObject) -> PyResult<PyCollection> {
        let rt = unsafe { &*self.rt_ptr };
        let mapped = self.inner.map(rt, move |val: &PyValue| -> PyValue {
            Python::with_gil(|py| {
                let result = func
                    .call1(py, (val.0.clone_ref(py),))
                    .expect("map function raised an exception");
                PyValue(result)
            })
        });
        Ok(PyCollection {
            inner: mapped,
            rt_ptr: self.rt_ptr,
        })
    }

    fn count(&self) -> PyResult<PyNodeId> {
        let rt = unsafe { &*self.rt_ptr };
        let count_node: incr_core::Incr<usize> = self.inner.count(rt);
        // Bridge usize -> PyValue via a query
        let node = rt.create_query(move |rt| -> PyValue {
            let c: usize = rt.get(count_node);
            Python::with_gil(|py| PyValue(c.into_pyobject(py).unwrap().into_any().unbind()))
        });
        Ok(PyNodeId { inner: node })
    }

    #[getter]
    fn version_node_id(&self) -> u32 {
        self.inner.version_node_id().raw()
    }
}

// ── PyRuntime: the main runtime exposed to Python ───────────────────────────

#[pyclass(name = "Runtime", unsendable)]
struct PyRuntime {
    inner: incr_core::Runtime,
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> Self {
        PyRuntime {
            inner: incr_core::Runtime::new(),
        }
    }

    fn create_input(&self, value: PyObject) -> PyNodeId {
        let node = self.inner.create_input(PyValue(value));
        PyNodeId { inner: node }
    }

    fn get(&self, node: PyNodeId) -> PyObject {
        let val: PyValue = self.inner.get(node.inner);
        val.0
    }

    fn set(&self, node: PyNodeId, value: PyObject) {
        self.inner.set(node.inner, PyValue(value));
    }

    fn create_query(&self, py_func: PyObject) -> PyNodeId {
        let node = self
            .inner
            .create_query(move |rt: &incr_core::Runtime| -> PyValue {
                Python::with_gil(|py| {
                    let rt_ref = Py::new(
                        py,
                        PyRuntimeRef {
                            ptr: rt as *const _,
                        },
                    )
                    .unwrap();
                    let result = py_func
                        .call1(py, (rt_ref.clone_ref(py),))
                        .expect("query function raised an exception");
                    // Invalidate the ref so it can't be used after callback returns
                    rt_ref.bind(py).borrow_mut().ptr = std::ptr::null();
                    PyValue(result)
                })
            });
        PyNodeId { inner: node }
    }

    fn create_collection(&self) -> PyCollection {
        let col = self.inner.create_collection::<PyValue>();
        let rt_ptr: *const incr_core::Runtime = &self.inner;
        PyCollection { inner: col, rt_ptr }
    }

    // ── Introspection API ───────────────────────────────────────────────

    fn set_label(&self, node: PyNodeId, label: String) {
        self.inner.set_label(node.inner.node_id(), label);
    }

    fn set_label_by_id(&self, id: u32, label: String) {
        self.inner.set_label(incr_core::NodeId::from_raw(id), label);
    }

    fn set_tracing(&self, enabled: bool) {
        self.inner.set_tracing(enabled);
    }

    fn get_traced(&self, node: PyNodeId) -> PyResult<(PyObject, PyObject)> {
        let (val, trace): (PyValue, incr_core::PropagationTrace) =
            self.inner.get_traced(node.inner);
        Python::with_gil(|py| {
            let trace_dict = pyo3::types::PyDict::new(py);
            trace_dict.set_item("target", trace.target.raw())?;
            trace_dict.set_item("total_nodes", trace.total_nodes)?;
            trace_dict.set_item("nodes_recomputed", trace.nodes_recomputed)?;
            trace_dict.set_item("nodes_cutoff", trace.nodes_cutoff)?;
            trace_dict.set_item("elapsed_ns", trace.elapsed_ns)?;

            let node_traces = pyo3::types::PyList::empty(py);
            for nt in &trace.node_traces {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("id", nt.id.raw())?;
                d.set_item(
                    "action",
                    match &nt.action {
                        incr_core::TraceAction::VerifiedClean => "verified_clean",
                        incr_core::TraceAction::Recomputed {
                            value_changed: true,
                        } => "recomputed_changed",
                        incr_core::TraceAction::Recomputed {
                            value_changed: false,
                        } => "recomputed_cutoff",
                    },
                )?;
                node_traces.append(d)?;
            }
            trace_dict.set_item("node_traces", node_traces)?;

            Ok((val.0, trace_dict.into_any().unbind()))
        })
    }

    fn graph_snapshot(&self) -> PyResult<PyObject> {
        let infos = self.inner.graph_snapshot();
        Python::with_gil(|py| {
            let result = pyo3::types::PyList::empty(py);
            for info in &infos {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("id", info.id.raw())?;
                d.set_item(
                    "kind",
                    match info.kind {
                        incr_core::NodeKindInfo::Input => "input",
                        incr_core::NodeKindInfo::Compute => "compute",
                    },
                )?;
                d.set_item("label", &info.label)?;
                let deps: Vec<u32> = info.dependencies.iter().map(|n| n.raw()).collect();
                let depts: Vec<u32> = info.dependents.iter().map(|n| n.raw()).collect();
                d.set_item("dependencies", deps)?;
                d.set_item("dependents", depts)?;
                result.append(d)?;
            }
            Ok(result.into_any().unbind())
        })
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }
}

// ── Module definition ───────────────────────────────────────────────────────

#[pymodule]
fn incr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyNodeId>()?;
    m.add_class::<PyRuntimeRef>()?;
    m.add_class::<PyCollection>()?;
    Ok(())
}
