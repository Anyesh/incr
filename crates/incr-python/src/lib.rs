//! Python bindings for `incr-compute` (single-threaded).
//!
//! The Python module is named `incr`; `from incr import Runtime` opens
//! the door to creating inputs and queries against the v0.2 engine.
//! User values are wrapped in [`PyValue`] which provides `Clone`,
//! `PartialEq`, `Eq`, `Hash`, and `Ord` over arbitrary `Py<PyAny>`s via
//! the Python GIL. The runtime's `Value` bound (`Clone + PartialEq +
//! Send + Sync + 'static`) is satisfied because `Py<PyAny>` is `Send`
//! and `Sync` in PyO3 (you need the GIL to actually dereference).
//!
//! Runtime is `!Send + !Sync` under the Local strategy, so every class
//! here is `unsendable`: Python may not move them across threads, which
//! matches the engine's confinement exactly. The GIL-deadlock concern
//! that applies to the concurrent binding (a compute waiting on another
//! thread's node while holding the GIL) cannot arise single-threaded.
//!
//! Exception propagation: Python callbacks that raise inside an engine
//! closure stash the `PyErr` in a thread-local and panic; the engine's
//! compute boundary marks the node Failed and unwinds back to the API
//! entry point, which re-raises the ORIGINAL Python exception. Engine
//! panics that are not Python exceptions (cycles, stale handles, failed
//! nodes) surface as `RuntimeError` with the engine's message.
//!
//! Lifetime safety: collections and views hold a strong `Py<PyRuntime>`
//! reference, so the runtime cannot be garbage-collected while any
//! derived object is alive. (The previous design held a raw pointer
//! into the PyRuntime allocation, which dangled if Python dropped the
//! runtime first.)

use pyo3::prelude::*;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};

use incr_compute::{
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

/// Newtype around `Py<PyAny>` that satisfies the `Value` bound. All
/// trait methods reacquire the GIL because Python objects are only
/// usable while holding it; this is the conventional PyO3 pattern for
/// embedding `Py<PyAny>` in trait-bounded Rust code.
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
#[pyclass(name = "NodeId", unsendable)]
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
#[pyclass(name = "ObserverId", unsendable)]
#[derive(Clone)]
struct PyObserverId {
    inner: incr_compute::ObserverId,
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
    fn get(&self, node: PyNodeId) -> PyResult<Py<PyAny>> {
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
    rt: Py<PyRuntime>,
}

impl PyCollection {
    fn with_rt<R>(&self, py: Python<'_>, f: impl FnOnce(&Runtime) -> R) -> R {
        let rt = self.rt.bind(py).borrow();
        f(&rt.inner)
    }
}

#[pymethods]
impl PyCollection {
    fn insert(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        pyerr_boundary(|| self.with_rt(py, |rt| self.inner.insert(rt, PyValue(value))))
    }

    fn delete(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<bool> {
        pyerr_boundary(|| self.with_rt(py, |rt| self.inner.delete(rt, &PyValue(value))))
    }

    fn snapshot_len(&self) -> usize {
        self.inner.snapshot_len()
    }

    fn filter(&self, py: Python<'_>, predicate: Py<PyAny>) -> PyResult<PyCollection> {
        let filtered = self.with_rt(py, |rt| {
            self.inner.filter(rt, move |val: &PyValue| -> bool {
                Python::attach(|py| {
                    call_stashing(py, &predicate, (val.0.clone_ref(py),))
                        .is_truthy(py)
                        .unwrap_or(false)
                })
            })
        });
        Ok(PyCollection {
            inner: filtered,
            rt: self.rt.clone_ref(py),
        })
    }

    fn map(&self, py: Python<'_>, func: Py<PyAny>) -> PyResult<PyCollection> {
        let mapped = self.with_rt(py, |rt| {
            self.inner.map(rt, move |val: &PyValue| -> PyValue {
                Python::attach(|py| PyValue(call_stashing(py, &func, (val.0.clone_ref(py),))))
            })
        });
        Ok(PyCollection {
            inner: mapped,
            rt: self.rt.clone_ref(py),
        })
    }

    fn count(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        let node = self.with_rt(py, |rt| {
            let count_node: Incr<u64> = self.inner.count(rt);
            // Bridge u64 -> PyValue so the Python side receives a node
            // returning an int, matching the single PyNodeId type.
            rt.create_query(move |rt| -> PyValue {
                let c: u64 = rt.get(count_node);
                Python::attach(|py| PyValue(c.into_pyobject(py).unwrap().into_any().unbind()))
            })
        });
        Ok(PyNodeId { inner: node })
    }

    fn reduce(&self, py: Python<'_>, fold_fn: Py<PyAny>) -> PyResult<PyNodeId> {
        let node = self.with_rt(py, |rt| {
            self.inner.reduce(rt, move |elements| -> PyValue {
                Python::attach(|py| {
                    let py_list = pyo3::types::PyList::empty(py);
                    for elem in elements.iter() {
                        py_list.append(elem.0.clone_ref(py)).unwrap();
                    }
                    PyValue(call_stashing(py, &fold_fn, (py_list,)))
                })
            })
        });
        Ok(PyNodeId { inner: node })
    }

    /// Incrementally maintained monoid fold: `lift` maps an element into
    /// the aggregate domain, `combine` merges two aggregates, `identity`
    /// is the unit. O(log n) per change; see the Rust docs for the
    /// associativity contract.
    fn aggregate(
        &self,
        py: Python<'_>,
        identity: Py<PyAny>,
        lift: Py<PyAny>,
        combine: Py<PyAny>,
    ) -> PyResult<PyNodeId> {
        let node = self.with_rt(py, |rt| {
            self.inner.aggregate(
                rt,
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
            )
        });
        Ok(PyNodeId { inner: node })
    }

    fn sort_by_key(&self, py: Python<'_>, key_fn: Py<PyAny>) -> PyResult<PySortedCollection> {
        let sorted = self.with_rt(py, |rt| {
            self.inner.sort_by_key(rt, move |val: &PyValue| -> PyValue {
                Python::attach(|py| PyValue(call_stashing(py, &key_fn, (val.0.clone_ref(py),))))
            })
        });
        Ok(PySortedCollection {
            inner: sorted,
            rt: self.rt.clone_ref(py),
        })
    }

    fn group_by(&self, py: Python<'_>, key_fn: Py<PyAny>) -> PyResult<PyGroupedCollection> {
        let grouped = self.with_rt(py, |rt| {
            self.inner.group_by(rt, move |val: &PyValue| -> PyValue {
                Python::attach(|py| PyValue(call_stashing(py, &key_fn, (val.0.clone_ref(py),))))
            })
        });
        Ok(PyGroupedCollection {
            inner: grouped,
            rt: self.rt.clone_ref(py),
        })
    }

    fn join(
        &self,
        py: Python<'_>,
        right: &PyCollection,
        left_key: Py<PyAny>,
        right_key: Py<PyAny>,
    ) -> PyResult<PyCollection> {
        let mapped = self.with_rt(py, |rt| {
            let joined = self.inner.join(
                rt,
                &right.inner,
                move |val: &PyValue| -> PyValue {
                    Python::attach(|py| {
                        PyValue(call_stashing(py, &left_key, (val.0.clone_ref(py),)))
                    })
                },
                move |val: &PyValue| -> PyValue {
                    Python::attach(|py| {
                        PyValue(call_stashing(py, &right_key, (val.0.clone_ref(py),)))
                    })
                },
            );
            // join yields (PyValue, PyValue) pairs; map to Python tuples
            // for the unified element type.
            joined.map(rt, |pair: &(PyValue, PyValue)| -> PyValue {
                Python::attach(|py| {
                    let tuple = pyo3::types::PyTuple::new(
                        py,
                        &[pair.0 .0.clone_ref(py), pair.1 .0.clone_ref(py)],
                    )
                    .unwrap();
                    PyValue(tuple.into_any().unbind())
                })
            })
        });
        Ok(PyCollection {
            inner: mapped,
            rt: self.rt.clone_ref(py),
        })
    }

    #[getter]
    fn version_node(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        let bridge = self.with_rt(py, |rt| {
            let v: Incr<u64> = self.inner.version_node();
            rt.create_query(move |rt| -> PyValue {
                let n: u64 = rt.get(v);
                Python::attach(|py| PyValue(n.into_pyobject(py).unwrap().into_any().unbind()))
            })
        });
        Ok(PyNodeId { inner: bridge })
    }
}

#[pyclass(name = "SortedCollection", unsendable)]
struct PySortedCollection {
    inner: SortedCollection<PyValue, PyValue>,
    rt: Py<PyRuntime>,
}

#[pymethods]
impl PySortedCollection {
    fn pairwise(&self, py: Python<'_>) -> PyResult<PyCollection> {
        let rt_ref = self.rt.bind(py).borrow();
        let pair_collection = self.inner.pairwise(&rt_ref.inner);
        let mapped = pair_collection.map(&rt_ref.inner, |pair: &(PyValue, PyValue)| -> PyValue {
            Python::attach(|py| {
                let tuple = pyo3::types::PyTuple::new(
                    py,
                    &[pair.0 .0.clone_ref(py), pair.1 .0.clone_ref(py)],
                )
                .unwrap();
                PyValue(tuple.into_any().unbind())
            })
        });
        drop(rt_ref);
        Ok(PyCollection {
            inner: mapped,
            rt: self.rt.clone_ref(py),
        })
    }

    fn window(&self, py: Python<'_>, size: usize) -> PyResult<PyCollection> {
        let rt_ref = self.rt.bind(py).borrow();
        let win_collection = self.inner.window(&rt_ref.inner, size);
        let mapped = win_collection.map(&rt_ref.inner, |window: &Vec<PyValue>| -> PyValue {
            Python::attach(|py| {
                let py_list = pyo3::types::PyList::empty(py);
                for elem in window.iter() {
                    py_list.append(elem.0.clone_ref(py)).unwrap();
                }
                PyValue(py_list.into_any().unbind())
            })
        });
        drop(rt_ref);
        Ok(PyCollection {
            inner: mapped,
            rt: self.rt.clone_ref(py),
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
    fn version_node(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        let rt_ref = self.rt.bind(py).borrow();
        let ver_node: Incr<u64> = self.inner.version_node();
        let bridge = rt_ref.inner.create_query(move |rt| -> PyValue {
            let v: u64 = rt.get(ver_node);
            Python::attach(|py| PyValue(v.into_pyobject(py).unwrap().into_any().unbind()))
        });
        Ok(PyNodeId { inner: bridge })
    }
}

#[pyclass(name = "GroupedCollection", unsendable)]
struct PyGroupedCollection {
    inner: incr_compute::GroupedCollection<PyValue, PyValue>,
    rt: Py<PyRuntime>,
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

    fn get_group(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Option<PyCollection>> {
        let py_key = PyValue(key);
        match self.inner.get_group(&py_key) {
            Some(collection) => Ok(Some(PyCollection {
                inner: collection,
                rt: self.rt.clone_ref(py),
            })),
            None => Ok(None),
        }
    }

    fn group_count(&self) -> usize {
        self.inner.group_count()
    }

    #[getter]
    fn version_node(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        let rt_ref = self.rt.bind(py).borrow();
        let ver_node: Incr<u64> = self.inner.version_node();
        let bridge = rt_ref.inner.create_query(move |rt| -> PyValue {
            let v: u64 = rt.get(ver_node);
            Python::attach(|py| PyValue(v.into_pyobject(py).unwrap().into_any().unbind()))
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

    fn create_input(&self, value: Py<PyAny>) -> PyNodeId {
        let node = self.inner.create_input(PyValue(value));
        PyNodeId { inner: node }
    }

    fn get(&self, node: PyNodeId) -> PyResult<Py<PyAny>> {
        pyerr_boundary(|| {
            let val: PyValue = self.inner.get(node.inner);
            val.0
        })
    }

    fn set(&self, node: PyNodeId, value: Py<PyAny>) -> PyResult<()> {
        pyerr_boundary(|| self.inner.set(node.inner, PyValue(value)))
    }

    /// Delete a node and recycle its slot. Every surviving handle to it
    /// raises on use. The node must have no dependents.
    fn delete_node(&self, node: PyNodeId) -> PyResult<()> {
        pyerr_boundary(|| self.inner.delete_node(node.inner))
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
    /// values.
    fn stabilize(&self) -> PyResult<()> {
        pyerr_boundary(|| self.inner.stabilize())
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

    fn create_collection(slf: PyRef<'_, Self>) -> PyCollection {
        let col = slf.inner.create_collection::<PyValue>();
        let rt: Py<PyRuntime> = slf.into();
        PyCollection { inner: col, rt }
    }

    fn set_label(&self, node: PyNodeId, label: String) {
        self.inner.set_label(node.inner.slot(), label);
    }

    fn set_label_by_id(&self, id: u32, label: String) {
        self.inner.set_label(id, label);
    }

    fn get_traced(&self, node: PyNodeId) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (val, trace): (PyValue, PropagationTrace) =
            pyerr_boundary(|| self.inner.get_traced(node.inner))?;
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
fn incr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyNodeId>()?;
    m.add_class::<PyObserverId>()?;
    m.add_class::<PyRuntimeRef>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PySortedCollection>()?;
    m.add_class::<PyGroupedCollection>()?;
    Ok(())
}
