//! Python bindings for `incr-concurrent` (thread-safe).
//!
//! The Python module is named `incr_concurrent`; the API mirrors the
//! `incr` (single-threaded) binding exactly: same method names, same
//! callback shapes, same value bounds. Migration between the two is a
//! one-line import change.
//!
//! Unlike the single-threaded binding, the classes here are genuinely
//! shareable across Python threads: the runtime lives behind an
//! `Arc<Runtime<Shared>>` and every derived object holds a strong clone,
//! so nothing dangles and nothing needs `unsendable` (except
//! `RuntimeRef`, which is scoped to a single callback invocation by
//! design). Reader threads call `get` concurrently with a writer thread
//! calling `set`; that is the point of this crate.
//!
//! GIL discipline: every entry point that can compute or wait on
//! another thread's compute releases the GIL first (`allow_threads`).
//! Without that, thread A could hold the GIL while waiting on a node
//! thread B is computing, while B blocks acquiring the GIL inside its
//! own callback: a deadlock. Engine-internal locks never hold across
//! Python calls (operators stage user closures outside locks), so lock
//! ordering cannot deadlock against the GIL.
//!
//! Exception propagation matches the `incr` module: callbacks that
//! raise stash their `PyErr` thread-locally and panic; the API entry
//! point on the SAME thread re-raises the original exception. If a
//! different thread trips over the resulting Failed node it gets a
//! `RuntimeError` carrying the engine's message instead.

use pyo3::prelude::*;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ::incr_concurrent as engine;
use engine::{
    Incr, IncrCollection, NodeId, NodeKindInfo, PropagationTrace, Runtime, SortedCollection,
    TraceAction,
};

thread_local! {
    static PENDING_PYERR: RefCell<Option<PyErr>> = const { RefCell::new(None) };
}

/// Invoke a Python callable from inside an engine closure. A raised
/// exception is stashed for the API boundary to re-raise, then turned
/// into a Rust panic so the engine's compute boundary aborts this node
/// cleanly (Failed state, cursor unadvanced).
fn call_stashing<'py>(
    py: Python<'py>,
    f: &Py<PyAny>,
    args: impl pyo3::call::PyCallArgs<'py>,
) -> Py<PyAny> {
    match f.call1(py, args) {
        Ok(v) => v,
        Err(e) => {
            let msg = e.to_string();
            PENDING_PYERR.with(|c| *c.borrow_mut() = Some(e));
            panic!("incr: python callback raised: {}", msg);
        }
    }
}

/// Run an engine entry point that may execute Python callbacks. A
/// stashed Python exception is re-raised as itself; other engine panics
/// become RuntimeError with the engine's message.
fn pyerr_boundary<R>(f: impl FnOnce() -> R) -> PyResult<R> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(v) => Ok(v),
        Err(payload) => {
            if let Some(e) = PENDING_PYERR.with(|c| c.borrow_mut().take()) {
                return Err(e);
            }
            let msg = if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else {
                std::panic::resume_unwind(payload);
            };
            Err(pyo3::exceptions::PyRuntimeError::new_err(msg))
        }
    }
}

/// Newtype around `Py<PyAny>` that satisfies the `Value` bound
/// (`Py<PyAny>` is `Send + Sync`; dereferencing requires the GIL). All
/// trait methods reacquire the GIL, the conventional PyO3 pattern.
struct PyValue(Py<PyAny>);

impl Clone for PyValue {
    fn clone(&self) -> Self {
        Python::attach(|py| PyValue(self.0.clone_ref(py)))
    }
}

impl PartialEq for PyValue {
    fn eq(&self, other: &Self) -> bool {
        Python::attach(|py| self.0.bind(py).eq(other.0.bind(py)).unwrap_or(false))
    }
}

impl Eq for PyValue {}

impl Hash for PyValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Python::attach(|py| {
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
        Python::attach(|py| {
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

/// Token for an observer registration; pass to `Runtime.unobserve`.
#[pyclass(name = "ObserverId")]
#[derive(Clone)]
struct PyObserverId {
    inner: engine::ObserverId,
}

/// Read-only runtime handle passed to query closures. Scoped to one
/// callback invocation: the pointer is nulled out after the callback
/// returns so stale captures fail loudly. Unsendable because the borrow
/// it wraps belongs to the invoking thread's compute frame.
#[pyclass(name = "RuntimeRef", unsendable)]
struct PyRuntimeRef {
    ptr: *const Runtime,
}

#[pymethods]
impl PyRuntimeRef {
    fn get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        if self.ptr.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "RuntimeRef is no longer valid (used outside query callback)",
            ));
        }
        // SAFETY: ptr is non-null only inside an active query callback;
        // the Runtime is borrowed by the runtime's own closure dispatch,
        // so the lifetime is guaranteed to outlive the callback.
        let rt = unsafe { &*self.ptr };
        // Release the GIL: this nested get may wait on a node another
        // thread is computing, and that thread needs the GIL to finish.
        let val: PyValue = py.detach(|| rt.get(node.inner));
        Ok(val.0)
    }
}

#[pyclass(name = "Collection")]
struct PyCollection {
    inner: IncrCollection<PyValue>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl PyCollection {
    fn insert(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let v = PyValue(value);
        pyerr_boundary(|| py.detach(|| self.inner.insert(&self.rt, v)))
    }

    fn delete(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<bool> {
        let v = PyValue(value);
        pyerr_boundary(|| py.detach(|| self.inner.delete(&self.rt, &v)))
    }

    fn snapshot_len(&self) -> usize {
        self.inner.snapshot_len()
    }

    fn filter(&self, predicate: Py<PyAny>) -> PyResult<PyCollection> {
        let filtered = self.inner.filter(&self.rt, move |val: &PyValue| -> bool {
            Python::attach(|py| {
                call_stashing(py, &predicate, (val.0.clone_ref(py),))
                    .is_truthy(py)
                    .unwrap_or(false)
            })
        });
        Ok(PyCollection {
            inner: filtered,
            rt: Arc::clone(&self.rt),
        })
    }

    fn map(&self, func: Py<PyAny>) -> PyResult<PyCollection> {
        let mapped = self.inner.map(&self.rt, move |val: &PyValue| -> PyValue {
            Python::attach(|py| PyValue(call_stashing(py, &func, (val.0.clone_ref(py),))))
        });
        Ok(PyCollection {
            inner: mapped,
            rt: Arc::clone(&self.rt),
        })
    }

    fn count(&self) -> PyResult<PyNodeId> {
        let count_node: Incr<u64> = self.inner.count(&self.rt);
        let node = self.rt.create_query(move |rt| -> PyValue {
            let c: u64 = rt.get(count_node);
            Python::attach(|py| PyValue(c.into_pyobject(py).unwrap().into_any().unbind()))
        });
        Ok(PyNodeId { inner: node })
    }

    fn reduce(&self, fold_fn: Py<PyAny>) -> PyResult<PyNodeId> {
        let node = self.inner.reduce(&self.rt, move |elements| -> PyValue {
            Python::attach(|py| {
                let py_list = pyo3::types::PyList::empty(py);
                for elem in elements.iter() {
                    py_list.append(elem.0.clone_ref(py)).unwrap();
                }
                PyValue(call_stashing(py, &fold_fn, (py_list,)))
            })
        });
        Ok(PyNodeId { inner: node })
    }

    /// Incrementally maintained monoid fold; O(log n) per change. See
    /// the Rust docs for the associativity contract.
    fn aggregate(
        &self,
        identity: Py<PyAny>,
        lift: Py<PyAny>,
        combine: Py<PyAny>,
    ) -> PyResult<PyNodeId> {
        let node = self.inner.aggregate(
            &self.rt,
            PyValue(identity),
            move |val: &PyValue| -> PyValue {
                Python::attach(|py| PyValue(call_stashing(py, &lift, (val.0.clone_ref(py),))))
            },
            move |a: &PyValue, b: &PyValue| -> PyValue {
                Python::attach(|py| {
                    PyValue(call_stashing(
                        py,
                        &combine,
                        (a.0.clone_ref(py), b.0.clone_ref(py)),
                    ))
                })
            },
        );
        Ok(PyNodeId { inner: node })
    }

    fn sort_by_key(&self, key_fn: Py<PyAny>) -> PyResult<PySortedCollection> {
        let sorted = self
            .inner
            .sort_by_key(&self.rt, move |val: &PyValue| -> PyValue {
                Python::attach(|py| PyValue(call_stashing(py, &key_fn, (val.0.clone_ref(py),))))
            });
        Ok(PySortedCollection {
            inner: sorted,
            rt: Arc::clone(&self.rt),
        })
    }

    fn group_by(&self, key_fn: Py<PyAny>) -> PyResult<PyGroupedCollection> {
        let grouped = self
            .inner
            .group_by(&self.rt, move |val: &PyValue| -> PyValue {
                Python::attach(|py| PyValue(call_stashing(py, &key_fn, (val.0.clone_ref(py),))))
            });
        Ok(PyGroupedCollection {
            inner: grouped,
            rt: Arc::clone(&self.rt),
        })
    }

    fn join(
        &self,
        right: &PyCollection,
        left_key: Py<PyAny>,
        right_key: Py<PyAny>,
    ) -> PyResult<PyCollection> {
        let joined = self.inner.join(
            &self.rt,
            &right.inner,
            move |val: &PyValue| -> PyValue {
                Python::attach(|py| PyValue(call_stashing(py, &left_key, (val.0.clone_ref(py),))))
            },
            move |val: &PyValue| -> PyValue {
                Python::attach(|py| {
                    PyValue(call_stashing(py, &right_key, (val.0.clone_ref(py),)))
                })
            },
        );
        let mapped = joined.map(&self.rt, |pair: &(PyValue, PyValue)| -> PyValue {
            Python::attach(|py| {
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
            rt: Arc::clone(&self.rt),
        })
    }

    #[getter]
    fn version_node(&self) -> PyResult<PyNodeId> {
        let v: Incr<u64> = self.inner.version_node();
        let bridge = self.rt.create_query(move |rt| -> PyValue {
            let n: u64 = rt.get(v);
            Python::attach(|py| PyValue(n.into_pyobject(py).unwrap().into_any().unbind()))
        });
        Ok(PyNodeId { inner: bridge })
    }
}

#[pyclass(name = "SortedCollection")]
struct PySortedCollection {
    inner: SortedCollection<PyValue, PyValue>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl PySortedCollection {
    fn pairwise(&self) -> PyResult<PyCollection> {
        let pair_collection = self.inner.pairwise(&self.rt);
        let mapped = pair_collection.map(&self.rt, |pair: &(PyValue, PyValue)| -> PyValue {
            Python::attach(|py| {
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
            rt: Arc::clone(&self.rt),
        })
    }

    fn window(&self, size: usize) -> PyResult<PyCollection> {
        let win_collection = self.inner.window(&self.rt, size);
        let mapped = win_collection.map(&self.rt, |window: &Vec<PyValue>| -> PyValue {
            Python::attach(|py| {
                let py_list = pyo3::types::PyList::empty(py);
                for elem in window.iter() {
                    py_list.append(elem.0.clone_ref(py)).unwrap();
                }
                PyValue(py_list.into_any().unbind())
            })
        });
        Ok(PyCollection {
            inner: mapped,
            rt: Arc::clone(&self.rt),
        })
    }

    fn snapshot(&self) -> PyResult<Py<PyAny>> {
        let entries = self.inner.snapshot();
        Python::attach(|py| {
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
        let ver_node: Incr<u64> = self.inner.version_node();
        let bridge = self.rt.create_query(move |rt| -> PyValue {
            let v: u64 = rt.get(ver_node);
            Python::attach(|py| PyValue(v.into_pyobject(py).unwrap().into_any().unbind()))
        });
        Ok(PyNodeId { inner: bridge })
    }
}

#[pyclass(name = "GroupedCollection")]
struct PyGroupedCollection {
    inner: engine::GroupedCollection<PyValue, PyValue>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl PyGroupedCollection {
    fn keys(&self) -> PyResult<Py<PyAny>> {
        let keys = self.inner.keys();
        Python::attach(|py| {
            let list = pyo3::types::PyList::empty(py);
            for key in keys {
                list.append(key.0.clone_ref(py))?;
            }
            Ok(list.into_any().unbind())
        })
    }

    fn get_group(&self, key: Py<PyAny>) -> PyResult<Option<PyCollection>> {
        let py_key = PyValue(key);
        match self.inner.get_group(&py_key) {
            Some(collection) => Ok(Some(PyCollection {
                inner: collection,
                rt: Arc::clone(&self.rt),
            })),
            None => Ok(None),
        }
    }

    fn group_count(&self) -> usize {
        self.inner.group_count()
    }

    #[getter]
    fn version_node(&self) -> PyResult<PyNodeId> {
        let ver_node: Incr<u64> = self.inner.version_node();
        let bridge = self.rt.create_query(move |rt| -> PyValue {
            let v: u64 = rt.get(ver_node);
            Python::attach(|py| PyValue(v.into_pyobject(py).unwrap().into_any().unbind()))
        });
        Ok(PyNodeId { inner: bridge })
    }
}

#[pyclass(name = "Runtime")]
struct PyRuntime {
    inner: Arc<Runtime>,
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> Self {
        PyRuntime {
            inner: Arc::new(Runtime::new()),
        }
    }

    fn create_input(&self, value: Py<PyAny>) -> PyNodeId {
        let node = self.inner.create_input(PyValue(value));
        PyNodeId { inner: node }
    }

    fn get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        pyerr_boundary(|| {
            let val: PyValue = py.detach(|| self.inner.get(node.inner));
            val.0
        })
    }

    fn set(&self, py: Python<'_>, node: PyNodeId, value: Py<PyAny>) -> PyResult<()> {
        let v = PyValue(value);
        pyerr_boundary(|| py.detach(|| self.inner.set(node.inner, v)))
    }

    /// Delete a node and recycle its slot. Every surviving handle to it
    /// raises on use. The node must have no dependents.
    fn delete_node(&self, py: Python<'_>, node: PyNodeId) -> PyResult<()> {
        pyerr_boundary(|| py.detach(|| self.inner.delete_node(node.inner)))
    }

    /// Register `callback` to fire on `stabilize()` whenever the node's
    /// value changed since the last firing.
    fn observe(&self, node: PyNodeId, callback: Py<PyAny>) -> PyResult<PyObserverId> {
        let id = self.inner.observe(node.inner, move |v: &PyValue| {
            Python::attach(|py| {
                call_stashing(py, &callback, (v.0.clone_ref(py),));
            });
        });
        Ok(PyObserverId { inner: id })
    }

    fn unobserve(&self, id: PyObserverId) {
        self.inner.unobserve(id.inner);
    }

    /// Bring observed nodes up to date and fire callbacks for changed
    /// values. Releases the GIL; callbacks reacquire it individually.
    fn stabilize(&self, py: Python<'_>) -> PyResult<()> {
        pyerr_boundary(|| py.detach(|| self.inner.stabilize()))
    }

    fn create_query(&self, py_func: Py<PyAny>) -> PyNodeId {
        let node = self.inner.create_query(move |rt: &Runtime| -> PyValue {
            Python::attach(|py| {
                let rt_ref = Py::new(
                    py,
                    PyRuntimeRef {
                        ptr: rt as *const _,
                    },
                )
                .unwrap();
                let result = call_stashing(py, &py_func, (rt_ref.clone_ref(py),));
                // Invalidate the ref so it can't be used after callback returns.
                rt_ref.bind(py).borrow_mut().ptr = std::ptr::null();
                PyValue(result)
            })
        });
        PyNodeId { inner: node }
    }

    fn create_collection(&self) -> PyCollection {
        let col = self.inner.create_collection::<PyValue>();
        PyCollection {
            inner: col,
            rt: Arc::clone(&self.inner),
        }
    }

    fn set_label(&self, node: PyNodeId, label: String) {
        self.inner.set_label(node.inner.slot(), label);
    }

    fn set_label_by_id(&self, id: u32, label: String) {
        self.inner.set_label(id, label);
    }

    fn get_traced(&self, py: Python<'_>, node: PyNodeId) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (val, trace): (PyValue, PropagationTrace) =
            pyerr_boundary(|| py.detach(|| self.inner.get_traced(node.inner)))?;
        Python::attach(|py| {
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

    fn graph_snapshot(&self) -> PyResult<Py<PyAny>> {
        let infos = self.inner.graph_snapshot();
        Python::attach(|py| {
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
fn incr_concurrent(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyNodeId>()?;
    m.add_class::<PyObserverId>()?;
    m.add_class::<PyRuntimeRef>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PySortedCollection>()?;
    m.add_class::<PyGroupedCollection>()?;
    Ok(())
}
