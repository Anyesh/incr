//! Python bindings for `incr-compute` (single-threaded).
//!
//! The Python module is named `incr`; `from incr import Runtime` opens
//! the door to creating inputs and queries against the v0.2 engine.
//! User values are wrapped in [`PyValue`] which provides `Clone`,
//! `PartialEq`, `Eq`, `Hash`, and `Ord` over arbitrary `PyObject`s via
//! the Python GIL. The runtime's `Value` bound (`Clone + PartialEq +
//! Send + Sync + 'static`) is satisfied because `Py<PyAny>` is `Send`
//! and `Sync` in PyO3 (you need the GIL to actually dereference).
//!
//! Runtime is `!Send + !Sync` under the Local strategy (the runtime's
//! dep_stack uses a `RefCell`), so `PyRuntime` is `unsendable` and the
//! GIL-bound nature of Python callbacks aligns nicely with that.

use pyo3::prelude::*;
use std::hash::{Hash, Hasher};

use incr_compute::{
    Incr, IncrCollection, NodeId, NodeKindInfo, PropagationTrace, Runtime, SortedCollection,
    TraceAction,
};

/// Newtype around `PyObject` that satisfies the `Value` bound. All
/// trait methods reacquire the GIL because Python objects are only
/// usable while holding it; this is the conventional PyO3 pattern for
/// embedding `PyObject` in trait-bounded Rust code.
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

impl PartialOrd for PyValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PyValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Python::with_gil(|py| {
            let self_ref = self.0.bind(py);
            let other_ref = other.0.bind(py);
            if self_ref.lt(other_ref).unwrap_or(false) {
                std::cmp::Ordering::Less
            } else if self_ref.eq(other_ref).unwrap_or(false) {
                std::cmp::Ordering::Equal
            } else {
                std::cmp::Ordering::Greater
            }
        })
    }
}

/// Typed node handle exposed to Python. Wraps `Incr<PyValue>`.
#[pyclass(name = "NodeId")]
#[derive(Clone)]
struct PyNodeId {
    inner: Incr<PyValue>,
}

#[pymethods]
impl PyNodeId {
    #[getter]
    fn id(&self) -> u32 {
        self.inner.slot()
    }

    fn __repr__(&self) -> String {
        format!("NodeId(slot={})", self.inner.slot())
    }
}

/// Read-only runtime handle passed to query closures. The pointer is
/// nulled out after the callback returns to make stale captures fail
/// loudly rather than silently corrupt memory.
#[pyclass(name = "RuntimeRef", unsendable)]
struct PyRuntimeRef {
    ptr: *const Runtime,
}

#[pymethods]
impl PyRuntimeRef {
    fn get(&self, node: PyNodeId) -> PyResult<PyObject> {
        if self.ptr.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "RuntimeRef is no longer valid (used outside query callback)",
            ));
        }
        // SAFETY: ptr is non-null only inside an active query callback;
        // the Runtime is borrowed by the runtime's own closure dispatch,
        // so the lifetime is guaranteed to outlive the callback.
        let rt = unsafe { &*self.ptr };
        let val: PyValue = rt.get(node.inner);
        Ok(val.0)
    }
}

#[pyclass(name = "Collection", unsendable)]
struct PyCollection {
    inner: IncrCollection<PyValue>,
    rt_ptr: *const Runtime,
}

#[pymethods]
impl PyCollection {
    fn insert(&self, value: PyObject) {
        let rt = unsafe { &*self.rt_ptr };
        self.inner.insert(rt, PyValue(value));
    }

    fn delete(&self, value: PyObject) -> bool {
        let rt = unsafe { &*self.rt_ptr };
        self.inner.delete(rt, &PyValue(value))
    }

    fn snapshot_len(&self) -> usize {
        self.inner.snapshot_len()
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
        let count_node: Incr<u64> = self.inner.count(rt);
        // Bridge u64 -> PyValue via a wrapper query so the Python side
        // receives a node returning an int (PyValue), matching the
        // single PyNodeId type the binding exposes.
        let node = rt.create_query(move |rt| -> PyValue {
            let c: u64 = rt.get(count_node);
            Python::with_gil(|py| PyValue(c.into_pyobject(py).unwrap().into_any().unbind()))
        });
        Ok(PyNodeId { inner: node })
    }

    fn reduce(&self, fold_fn: PyObject) -> PyResult<PyNodeId> {
        let rt = unsafe { &*self.rt_ptr };
        let reduce_node: Incr<PyValue> = self.inner.reduce(rt, move |elements| -> PyValue {
            Python::with_gil(|py| {
                let py_list = pyo3::types::PyList::empty(py);
                for elem in elements.iter() {
                    py_list.append(elem.0.clone_ref(py)).unwrap();
                }
                let result = fold_fn
                    .call1(py, (py_list,))
                    .expect("reduce function raised an exception");
                PyValue(result)
            })
        });
        Ok(PyNodeId { inner: reduce_node })
    }

    fn sort_by_key(&self, key_fn: PyObject) -> PyResult<PySortedCollection> {
        let rt = unsafe { &*self.rt_ptr };
        let sorted = self.inner.sort_by_key(rt, move |val: &PyValue| -> PyValue {
            Python::with_gil(|py| {
                let result = key_fn
                    .call1(py, (val.0.clone_ref(py),))
                    .expect("sort key function raised an exception");
                PyValue(result)
            })
        });
        Ok(PySortedCollection {
            inner: sorted,
            rt_ptr: self.rt_ptr,
        })
    }

    fn group_by(&self, key_fn: PyObject) -> PyResult<PyGroupedCollection> {
        let rt = unsafe { &*self.rt_ptr };
        let grouped = self.inner.group_by(rt, move |val: &PyValue| -> PyValue {
            Python::with_gil(|py| {
                let result = key_fn
                    .call1(py, (val.0.clone_ref(py),))
                    .expect("group_by key function raised an exception");
                PyValue(result)
            })
        });
        Ok(PyGroupedCollection {
            inner: grouped,
            rt_ptr: self.rt_ptr,
        })
    }

    fn join(
        &self,
        right: &PyCollection,
        left_key: PyObject,
        right_key: PyObject,
    ) -> PyResult<PyCollection> {
        let rt = unsafe { &*self.rt_ptr };
        let joined = self.inner.join(
            rt,
            &right.inner,
            move |val: &PyValue| -> PyValue {
                Python::with_gil(|py| {
                    let result = left_key
                        .call1(py, (val.0.clone_ref(py),))
                        .expect("left key function raised an exception");
                    PyValue(result)
                })
            },
            move |val: &PyValue| -> PyValue {
                Python::with_gil(|py| {
                    let result = right_key
                        .call1(py, (val.0.clone_ref(py),))
                        .expect("right key function raised an exception");
                    PyValue(result)
                })
            },
        );
        // join returns IncrCollection<(PyValue, PyValue)>; map pairs to
        // Python tuples wrapped in PyValue for the unified element type.
        let mapped = joined.map(rt, |pair: &(PyValue, PyValue)| -> PyValue {
            Python::with_gil(|py| {
                let tuple = pyo3::types::PyTuple::new(
                    py,
                    &[pair.0 .0.clone_ref(py), pair.1 .0.clone_ref(py)],
                )
                .unwrap();
                PyValue(tuple.into_any().unbind())
            })
        });
        Ok(PyCollection {
            inner: mapped,
            rt_ptr: self.rt_ptr,
        })
    }

    #[getter]
    fn version_node(&self) -> PyResult<PyNodeId> {
        let rt = unsafe { &*self.rt_ptr };
        let v: Incr<u64> = self.inner.version_node();
        // Wrap the u64 version node in a PyValue-returning bridge so
        // it can be passed to rt.get / set_label uniformly.
        let bridge = rt.create_query(move |rt| -> PyValue {
            let n: u64 = rt.get(v);
            Python::with_gil(|py| PyValue(n.into_pyobject(py).unwrap().into_any().unbind()))
        });
        Ok(PyNodeId { inner: bridge })
    }
}

#[pyclass(name = "SortedCollection", unsendable)]
struct PySortedCollection {
    inner: SortedCollection<PyValue, PyValue>,
    rt_ptr: *const Runtime,
}

#[pymethods]
impl PySortedCollection {
    fn pairwise(&self) -> PyResult<PyCollection> {
        let rt = unsafe { &*self.rt_ptr };
        let pair_collection = self.inner.pairwise(rt);
        let mapped = pair_collection.map(rt, |pair: &(PyValue, PyValue)| -> PyValue {
            Python::with_gil(|py| {
                let tuple = pyo3::types::PyTuple::new(
                    py,
                    &[pair.0 .0.clone_ref(py), pair.1 .0.clone_ref(py)],
                )
                .unwrap();
                PyValue(tuple.into_any().unbind())
            })
        });
        Ok(PyCollection {
            inner: mapped,
            rt_ptr: self.rt_ptr,
        })
    }

    fn window(&self, size: usize) -> PyResult<PyCollection> {
        let rt = unsafe { &*self.rt_ptr };
        let win_collection = self.inner.window(rt, size);
        let mapped = win_collection.map(rt, |window: &Vec<PyValue>| -> PyValue {
            Python::with_gil(|py| {
                let py_list = pyo3::types::PyList::empty(py);
                for elem in window.iter() {
                    py_list.append(elem.0.clone_ref(py)).unwrap();
                }
                PyValue(py_list.into_any().unbind())
            })
        });
        Ok(PyCollection {
            inner: mapped,
            rt_ptr: self.rt_ptr,
        })
    }

    fn snapshot(&self) -> PyResult<PyObject> {
        let entries = self.inner.snapshot();
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty(py);
            for entry in entries {
                list.append(entry.0.clone_ref(py))?;
            }
            Ok(list.into_any().unbind())
        })
    }

    fn snapshot_len(&self) -> usize {
        self.inner.snapshot_len()
    }

    #[getter]
    fn version_node(&self) -> PyResult<PyNodeId> {
        let rt = unsafe { &*self.rt_ptr };
        let ver_node: Incr<u64> = self.inner.version_node();
        let bridge = rt.create_query(move |rt| -> PyValue {
            let v: u64 = rt.get(ver_node);
            Python::with_gil(|py| PyValue(v.into_pyobject(py).unwrap().into_any().unbind()))
        });
        Ok(PyNodeId { inner: bridge })
    }
}

#[pyclass(name = "GroupedCollection", unsendable)]
struct PyGroupedCollection {
    inner: incr_compute::GroupedCollection<PyValue, PyValue>,
    rt_ptr: *const Runtime,
}

#[pymethods]
impl PyGroupedCollection {
    fn keys(&self) -> PyResult<PyObject> {
        let keys = self.inner.keys();
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty(py);
            for key in keys {
                list.append(key.0.clone_ref(py))?;
            }
            Ok(list.into_any().unbind())
        })
    }

    fn get_group(&self, key: PyObject) -> PyResult<Option<PyCollection>> {
        let py_key = PyValue(key);
        match self.inner.get_group(&py_key) {
            Some(collection) => Ok(Some(PyCollection {
                inner: collection,
                rt_ptr: self.rt_ptr,
            })),
            None => Ok(None),
        }
    }

    fn group_count(&self) -> usize {
        self.inner.group_count()
    }

    #[getter]
    fn version_node(&self) -> PyResult<PyNodeId> {
        let rt = unsafe { &*self.rt_ptr };
        let ver_node: Incr<u64> = self.inner.version_node();
        let bridge = rt.create_query(move |rt| -> PyValue {
            let v: u64 = rt.get(ver_node);
            Python::with_gil(|py| PyValue(v.into_pyobject(py).unwrap().into_any().unbind()))
        });
        Ok(PyNodeId { inner: bridge })
    }
}

#[pyclass(name = "Runtime", unsendable)]
struct PyRuntime {
    inner: Runtime,
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> Self {
        PyRuntime {
            inner: Runtime::new(),
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
        let node = self.inner.create_query(move |rt: &Runtime| -> PyValue {
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
                // Invalidate the ref so it can't be used after callback returns.
                rt_ref.bind(py).borrow_mut().ptr = std::ptr::null();
                PyValue(result)
            })
        });
        PyNodeId { inner: node }
    }

    fn create_collection(&self) -> PyCollection {
        let col = self.inner.create_collection::<PyValue>();
        let rt_ptr: *const Runtime = &self.inner;
        PyCollection { inner: col, rt_ptr }
    }

    fn set_label(&self, node: PyNodeId, label: String) {
        self.inner.set_label(node.inner.slot(), label);
    }

    fn set_label_by_id(&self, id: u32, label: String) {
        self.inner.set_label(id, label);
    }

    fn get_traced(&self, node: PyNodeId) -> PyResult<(PyObject, PyObject)> {
        let (val, trace): (PyValue, PropagationTrace) = self.inner.get_traced(node.inner);
        Python::with_gil(|py| {
            let trace_dict = pyo3::types::PyDict::new(py);
            trace_dict.set_item("target", trace.target.0)?;
            trace_dict.set_item("total_nodes", trace.total_nodes)?;
            trace_dict.set_item("nodes_recomputed", trace.nodes_recomputed)?;
            trace_dict.set_item("nodes_cutoff", trace.nodes_cutoff)?;
            trace_dict.set_item("elapsed_ns", trace.elapsed_ns)?;

            let node_traces = pyo3::types::PyList::empty(py);
            for nt in &trace.node_traces {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("id", nt.id.0)?;
                d.set_item(
                    "action",
                    match &nt.action {
                        TraceAction::VerifiedClean => "verified_clean",
                        TraceAction::Recomputed {
                            value_changed: true,
                        } => "recomputed_changed",
                        TraceAction::Recomputed {
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
                d.set_item("id", info.id.0)?;
                d.set_item(
                    "kind",
                    match info.kind {
                        NodeKindInfo::Input => "input",
                        NodeKindInfo::Compute => "compute",
                    },
                )?;
                d.set_item("label", &info.label)?;
                let deps: Vec<u32> = info.dependencies.iter().map(|n: &NodeId| n.0).collect();
                let depts: Vec<u32> = info.dependents.iter().map(|n: &NodeId| n.0).collect();
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

#[pymodule]
fn incr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyNodeId>()?;
    m.add_class::<PyRuntimeRef>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PySortedCollection>()?;
    m.add_class::<PyGroupedCollection>()?;
    Ok(())
}
