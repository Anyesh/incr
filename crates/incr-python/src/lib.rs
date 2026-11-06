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

mod elements;
mod keyed;
mod lookups;
mod memory;
mod ready;
mod state;

use pyo3::prelude::*;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use incr_compute::{
    DeleteError, Incr, IncrCollection, NodeId, NodeKindInfo, NodeTable, PropagationTrace, Runtime,
    SortedCollection, TraceAction,
};
use keyed::{IndexGuard, KeyCounts, SharedIndex, StagedKeys};
use state::{PyCallable, RtState};

use incr_python_protocol::notready::{
    lookup_failed, PyIncomplete, PyLookupFailed, PyNotReady, PyPending, PyUnavailable,
};
use incr_python_protocol::{
    call_stashing, in_runtime, init_fn, or_stash, pyerr_boundary, query_opts, try_read, Exceptions,
    PyFixpoint, PyValue,
};
use ready::_NotReadySignal;

pyo3::create_exception!(
    incr,
    CycleError,
    pyo3::exceptions::PyRuntimeError,
    "A read closed a dependency cycle that has no value: `args` is the message and the participants' node ids."
);

pyo3::create_exception!(
    incr,
    HasDependentsError,
    pyo3::exceptions::PyRuntimeError,
    "The node, or a node it owns, still has dependents; nothing was deleted."
);

pyo3::create_exception!(
    incr,
    _IncrEngineUnwind,
    pyo3::exceptions::PyBaseException,
    "Carries an engine unwind through Python frames inside a query; never caught by `except Exception`."
);

/// This module's exception types, for the protocol shared with the other
/// binding.
struct Exc;

impl Exceptions for Exc {
    fn cycle_error(message: String, ids: Vec<u32>) -> PyErr {
        CycleError::new_err((message, ids))
    }

    fn engine_unwind() -> PyErr {
        _IncrEngineUnwind::new_err(())
    }
}

/// Runs a `#[pymethods]` engine call on `rt`. A Local runtime is touched
/// by one thread, so its waits never block on another thread and the GIL
/// stays held.
fn engine_call<R>(rt: &Runtime, f: impl FnOnce() -> R) -> PyResult<R> {
    pyerr_boundary::<Exc, _>(|| in_runtime(rt, f))
}

type TracedNode = (Option<String>, Option<Py<PyAny>>);

/// The memoized value of a node a trace just left Clean, so the read
/// recomputes nothing. `None` for a dead slot, a cycle, or a value type
/// other than `PyValue`, `u64` and `(u64, u64)`.
fn traced_value(rt: &Runtime, id: NodeId) -> Option<Py<PyAny>> {
    let handle = rt.erased_handle(id)?;
    if let Some(node) = rt.typed_handle::<PyValue>(handle) {
        return rt.try_get(node).ok().map(|v| v.0);
    }
    if let Some(node) = rt.typed_handle::<u64>(handle) {
        let n = rt.try_get(node).ok()?;
        return Some(Python::attach(|py| {
            n.into_pyobject(py).unwrap().into_any().unbind()
        }));
    }
    let pair = rt.try_get(rt.typed_handle::<(u64, u64)>(handle)?).ok()?;
    Some(Python::attach(|py| {
        pair.into_pyobject(py).unwrap().into_any().unbind()
    }))
}

fn delete_error(e: DeleteError, key: impl Into<Py<PyAny>>) -> PyErr {
    match e {
        DeleteError::HasDependents => HasDependentsError::new_err(e.to_string()),
        DeleteError::NotAlive => pyo3::exceptions::PyKeyError::new_err(key.into()),
    }
}

/// A binding object that reaches its runtime through the `Runtime` object
/// it holds.
trait OnRuntime {
    fn runtime(&self) -> &Py<PyRuntime>;

    /// [`engine_call`] on this object's runtime, which `f` receives.
    fn engine_call<R>(&self, py: Python<'_>, f: impl FnOnce(&Runtime) -> R) -> PyResult<R> {
        let rt = self.runtime().bind(py).borrow();
        let rt = rt.rt()?;
        engine_call(rt, || f(rt))
    }

    /// The runtime's binding state.
    fn state(&self, py: Python<'_>) -> RtState {
        self.runtime().bind(py).borrow().state.clone()
    }
}

/// The closure the engine runs for a Python query: `py_func` called with
/// a `RuntimeRef` that is invalidated when it returns.
fn query_fn(py_func: PyCallable) -> impl Fn(&Runtime) -> PyValue + Send + Sync + 'static {
    move |rt: &Runtime| -> PyValue {
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
            // Invalidates the ref when the callback returns or unwinds, so
            // one the callback kept never reaches a dropped runtime.
            let _guard = InvalidateRef(rt_ref.bind(py).clone());
            in_runtime(rt, || {
                PyValue(ready::call_query(
                    py,
                    rt,
                    &py_func,
                    rt_ref.clone_ref(py).into_any(),
                    &missing,
                ))
            })
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

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut h);
        h.finish()
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
    ptr: Cell<*const Runtime>,
    /// The runtime's binding state, valid while `ptr` is non-null: the
    /// running query's closure holds it.
    state: *const RtState,
    /// The missing contributions of the `Incomplete` values the running
    /// query read, valid while `ptr` is non-null.
    missing: *const Cell<u64>,
}

struct InvalidateRef<'py>(Bound<'py, PyRuntimeRef>);

impl Drop for InvalidateRef<'_> {
    fn drop(&mut self) {
        // No method borrows the ref mutably, so this borrow cannot fail.
        self.0.borrow().ptr.set(std::ptr::null());
    }
}

impl PyRuntimeRef {
    fn rt(&self) -> PyResult<&Runtime> {
        let ptr = self.ptr.get();
        if ptr.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "RuntimeRef is no longer valid (used outside query callback)",
            ));
        }
        // SAFETY: ptr is non-null only inside an active query callback;
        // the Runtime is borrowed by the runtime's own closure dispatch,
        // so the lifetime is guaranteed to outlive the callback.
        Ok(unsafe { &*ptr })
    }

    /// The runtime and the running query's missing count.
    fn rt_and_missing(&self) -> PyResult<(&Runtime, &Cell<u64>)> {
        let rt = self.rt()?;
        // SAFETY: `rt` succeeded, so the query whose frame owns the count
        // is still running.
        Ok((rt, unsafe { &*self.missing }))
    }

    /// The runtime and its binding state.
    fn rt_and_state(&self) -> PyResult<(&Runtime, &RtState)> {
        let rt = self.rt()?;
        // SAFETY: `rt` succeeded, so the query whose closure owns the
        // state is still running.
        Ok((rt, unsafe { &*self.state }))
    }
}

// Each method is a boundary, so an engine unwind crosses the calling
// Python frames as an exception instead of a foreign panic.
#[pymethods]
impl PyRuntimeRef {
    /// The node's value. A `NotReady` value makes this query's value that
    /// value; an `Incomplete` one gives its `value` and makes this query's
    /// value `Incomplete`, with every `missing` it read added up.
    fn get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        let (rt, missing) = self.rt_and_missing()?;
        let val: PyValue = engine_call(rt, || rt.get(node.inner))?;
        ready::read(py, val.0, missing)
    }

    /// Like `get`, but a cycle that leaves the node without a value
    /// raises `CycleError` as an ordinary exception the query may catch.
    fn try_get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        let (rt, missing) = self.rt_and_missing()?;
        let val = try_read::<Exc>(engine_call(rt, || rt.try_get(node.inner))?)?;
        ready::read(py, val, missing)
    }

    /// The value of `key`'s lookup (see `Runtime.create_lookup`), read as
    /// `get` reads a node: `Pending` makes this query's value `Pending`.
    fn lookup(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let (rt, state) = self.rt_and_state()?;
        let (_, missing) = self.rt_and_missing()?;
        let lookups = state.lookups();
        let val = engine_call(rt, || rt.get(lookups.node(rt, py, &key)))?;
        ready::read(py, val.0, missing)
    }

    /// `(value, status)` as `Runtime.get_with_status` gives them, with
    /// neither rule of `get` applied.
    fn get_status(&self, py: Python<'_>, node: PyNodeId) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let rt = self.rt()?;
        let val: PyValue = engine_call(rt, || rt.get(node.inner))?;
        Ok(ready::with_status(py, val.0))
    }

    /// The input child named `key` of this query, created with `init()`
    /// the first time this query asks for it. Children this query stops
    /// asking for are deleted when it next lands.
    fn child_input(&self, key: Py<PyAny>, init: Py<PyAny>) -> PyResult<PyNodeId> {
        let rt = self.rt()?;
        let (key, init) = (PyValue(key), init_fn(init));
        let node = engine_call(rt, || rt.child_input(key, init))?;
        Ok(PyNodeId { inner: node })
    }

    /// The query child named `key`, created with `f` the first time, like
    /// `child_input`; the options apply only when the call creates it.
    #[pyo3(signature = (key, f, *, fallback=None, fixpoint=None, replay=false))]
    fn child_query(
        &self,
        py: Python<'_>,
        key: Py<PyAny>,
        f: Py<PyAny>,
        fallback: Option<Py<PyAny>>,
        fixpoint: Option<PyRef<'_, PyFixpoint>>,
        replay: bool,
    ) -> PyResult<PyNodeId> {
        let (rt, callables) = self.rt_and_state()?;
        let opts = query_opts(py, fallback, fixpoint, replay, |f| callables.hold(f))?;
        let (key, f) = (PyValue(key), query_fn(callables.hold(f)));
        let node = engine_call(rt, || rt.child_query_with(key, f, opts))?;
        Ok(PyNodeId { inner: node })
    }
}

/// A map from keys to nodes of one runtime. Key `__eq__` and `__hash__`
/// run as value code: one that calls into incr raises the contract error.
#[pyclass(name = "NodeTable", unsendable)]
struct PyNodeTable {
    inner: NodeTable<PyValue, PyValue>,
    rt: Py<PyRuntime>,
}

impl OnRuntime for PyNodeTable {
    fn runtime(&self) -> &Py<PyRuntime> {
        &self.rt
    }
}

#[pymethods]
impl PyNodeTable {
    /// The input node for `key`, created with `init()`'s value on a miss.
    fn get_or_insert_input(
        &self,
        py: Python<'_>,
        key: Py<PyAny>,
        init: Py<PyAny>,
    ) -> PyResult<PyNodeId> {
        let (key, init) = (PyValue(key), init_fn(init));
        let node = self.engine_call(py, |rt| self.inner.get_or_insert_input(rt, key, init))?;
        Ok(PyNodeId { inner: node })
    }

    /// The query node for `key`, created with `f` and the options on a miss.
    #[pyo3(signature = (key, f, *, fallback=None, fixpoint=None, replay=false))]
    fn get_or_insert_query(
        &self,
        py: Python<'_>,
        key: Py<PyAny>,
        f: Py<PyAny>,
        fallback: Option<Py<PyAny>>,
        fixpoint: Option<PyRef<'_, PyFixpoint>>,
        replay: bool,
    ) -> PyResult<PyNodeId> {
        let callables = self.state(py);
        let opts = query_opts(py, fallback, fixpoint, replay, |f| callables.hold(f))?;
        let (key, f) = (PyValue(key), query_fn(callables.hold(f)));
        let node = self.engine_call(py, |rt| {
            self.inner.get_or_insert_query_with(rt, key, f, opts)
        })?;
        Ok(PyNodeId { inner: node })
    }

    /// The live node for `key`, or `None`. Not allowed inside a query.
    fn get(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Option<PyNodeId>> {
        let key = PyValue(key);
        let node = self.engine_call(py, |rt| self.inner.get(rt, &key))?;
        Ok(node.map(|inner| PyNodeId { inner }))
    }

    /// Deletes `key`'s node and removes the entry. Raises `KeyError` if
    /// the table holds no live node for it.
    fn remove(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<()> {
        let k = PyValue(key.clone_ref(py));
        match self.engine_call(py, |rt| self.inner.remove(rt, &k))? {
            Ok(()) => Ok(()),
            Err(e) => Err(delete_error(e, key)),
        }
    }

    /// The `(key, node)` pairs whose node is alive. Not allowed inside a
    /// query.
    fn items(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, PyNodeId)>> {
        let entries = self.engine_call(py, |rt| {
            let mut entries = Vec::new();
            self.inner
                .for_each(rt, |key, node| entries.push((key.clone(), node)));
            entries
        })?;
        Ok(entries
            .into_iter()
            .map(|(key, inner)| (key.0, PyNodeId { inner }))
            .collect())
    }
}

#[pyclass(name = "Collection", unsendable)]
struct PyCollection {
    inner: IncrCollection<PyValue>,
    rt: Py<PyRuntime>,
}

impl OnRuntime for PyCollection {
    fn runtime(&self) -> &Py<PyRuntime> {
        &self.rt
    }
}

#[pymethods]
impl PyCollection {
    fn insert(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        self.engine_call(py, |rt| self.inner.insert(rt, PyValue(value)))
    }

    fn delete(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<bool> {
        self.engine_call(py, |rt| self.inner.delete(rt, &PyValue(value)))
    }

    fn extend(&self, py: Python<'_>, rows: &Bound<'_, PyAny>) -> PyResult<()> {
        let values = rows
            .try_iter()?
            .map(|row| row.map(|row| PyValue(row.unbind())))
            .collect::<PyResult<Vec<_>>>()?;
        self.engine_call(py, |rt| self.inner.extend(rt, values))
    }

    fn replace(&self, py: Python<'_>, old: Py<PyAny>, new: Py<PyAny>) -> PyResult<bool> {
        self.engine_call(py, |rt| {
            self.inner.replace(rt, Some(&PyValue(old)), PyValue(new))
        })
    }

    fn snapshot_len(&self, py: Python<'_>) -> PyResult<usize> {
        self.engine_call(py, |_| self.inner.snapshot_len())
    }

    fn filter(&self, py: Python<'_>, predicate: Py<PyAny>) -> PyResult<PyCollection> {
        let predicate = self.state(py).hold(predicate);
        let filtered = self.engine_call(py, |rt| {
            self.inner.filter(rt, move |val: &PyValue| -> bool {
                Python::attach(|py| {
                    elements::keep(val.0.bind(py), |row| {
                        or_stash(call_stashing(py, &predicate, (row,)).is_truthy(py))
                    })
                })
            })
        })?;
        Ok(PyCollection {
            inner: filtered,
            rt: self.rt.clone_ref(py),
        })
    }

    fn map(&self, py: Python<'_>, func: Py<PyAny>) -> PyResult<PyCollection> {
        let func = self.state(py).hold(func);
        let mapped = self.engine_call(py, |rt| {
            self.inner.map(rt, move |val: &PyValue| -> PyValue {
                Python::attach(|py| {
                    PyValue(elements::map(val.0.bind(py), |row| {
                        call_stashing(py, &func, (row,))
                    }))
                })
            })
        })?;
        Ok(PyCollection {
            inner: mapped,
            rt: self.rt.clone_ref(py),
        })
    }

    /// The number of ready elements, `Incomplete` with the number of
    /// not-ready and `Incomplete` elements when there are any.
    fn count(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        let node = self.engine_call(py, |rt| {
            let parts = self.inner.count_partition(rt, |val: &PyValue| {
                Python::attach(|py| elements::is_ready(val.0.bind(py)))
            });
            rt.create_query(move |rt| -> PyValue {
                let (ready, missing) = rt.get(parts);
                Python::attach(|py| {
                    let ready = ready.into_pyobject(py).unwrap().into_any().unbind();
                    PyValue(elements::total(py, ready, missing))
                })
            })
        })?;
        Ok(PyNodeId { inner: node })
    }

    /// `fold_fn` of the list of rows, with not-ready elements left out
    /// and an `Incomplete` element's `value` in; `Incomplete` when any
    /// element is, counting each not-ready element once and each
    /// `Incomplete` element's `missing`.
    fn reduce(&self, py: Python<'_>, fold_fn: Py<PyAny>) -> PyResult<PyNodeId> {
        let fold_fn = self.state(py).hold(fold_fn);
        let node = self.engine_call(py, |rt| {
            self.inner.reduce(rt, move |values: &[PyValue]| -> PyValue {
                Python::attach(|py| {
                    let (rows, missing) = elements::foldable(py, values.iter().map(|v| &v.0));
                    PyValue(elements::total(
                        py,
                        call_stashing(py, &fold_fn, (rows,)),
                        missing,
                    ))
                })
            })
        })?;
        Ok(PyNodeId { inner: node })
    }

    /// Incrementally maintained monoid fold: `lift` maps an element into
    /// the aggregate domain, `combine` merges two aggregates, `identity`
    /// is the unit. O(log n) per change; see the Rust docs for the
    /// associativity contract.
    /// Not-ready elements are left out and counted as for `reduce`, and
    /// `lift` and `combine` never see a not-ready or `Incomplete` value.
    fn aggregate(
        &self,
        py: Python<'_>,
        identity: Py<PyAny>,
        lift: Py<PyAny>,
        combine: Py<PyAny>,
    ) -> PyResult<PyNodeId> {
        let callables = self.state(py);
        let (lift, combine) = (callables.hold(lift), callables.hold(combine));
        let lifted_identity = identity.clone_ref(py);
        let node = self.engine_call(py, |rt| {
            self.inner.aggregate(
                rt,
                PyValue(identity),
                move |val: &PyValue| -> PyValue {
                    Python::attach(|py| {
                        PyValue(elements::lift(val.0.bind(py), &lifted_identity, |row| {
                            call_stashing(py, &lift, (row,))
                        }))
                    })
                },
                move |a: &PyValue, b: &PyValue| -> PyValue {
                    Python::attach(|py| {
                        PyValue(elements::combine(a.0.bind(py), b.0.bind(py), |a, b| {
                            call_stashing(py, &combine, (a, b))
                        }))
                    })
                },
            )
        })?;
        Ok(PyNodeId { inner: node })
    }

    fn sort_by_key(&self, py: Python<'_>, key_fn: Py<PyAny>) -> PyResult<PySortedCollection> {
        let key_fn = self.state(py).hold(key_fn);
        let sorted = self.engine_call(py, |rt| {
            self.inner.sort_by_key(rt, move |val: &PyValue| -> PyValue {
                Python::attach(|py| {
                    PyValue(call_stashing(
                        py,
                        &key_fn,
                        (elements::key_row(val.0.bind(py)),),
                    ))
                })
            })
        })?;
        Ok(PySortedCollection {
            inner: sorted,
            rt: self.rt.clone_ref(py),
        })
    }

    fn group_by(&self, py: Python<'_>, key_fn: Py<PyAny>) -> PyResult<PyGroupedCollection> {
        let key_fn = self.state(py).hold(key_fn);
        let grouped = self.engine_call(py, |rt| {
            self.inner.group_by(rt, move |val: &PyValue| -> PyValue {
                Python::attach(|py| {
                    PyValue(call_stashing(
                        py,
                        &key_fn,
                        (elements::key_row(val.0.bind(py)),),
                    ))
                })
            })
        })?;
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
        let callables = self.state(py);
        let (left_key, right_key) = (callables.hold(left_key), callables.hold(right_key));
        let mapped = self.engine_call(py, |rt| {
            let joined = self.inner.join(
                rt,
                &right.inner,
                move |val: &PyValue| -> PyValue {
                    Python::attach(|py| {
                        PyValue(call_stashing(
                            py,
                            &left_key,
                            (elements::key_row(val.0.bind(py)),),
                        ))
                    })
                },
                move |val: &PyValue| -> PyValue {
                    Python::attach(|py| {
                        PyValue(call_stashing(
                            py,
                            &right_key,
                            (elements::key_row(val.0.bind(py)),),
                        ))
                    })
                },
            );
            joined.map(rt, |pair: &(PyValue, PyValue)| -> PyValue {
                Python::attach(|py| PyValue(elements::pair(py, &pair.0 .0, &pair.1 .0)))
            })
        })?;
        Ok(PyCollection {
            inner: mapped,
            rt: self.rt.clone_ref(py),
        })
    }

    #[getter]
    fn version_node(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        let this = self.rt.bind(py).borrow();
        let rt = this.rt()?;
        engine_call(rt, || this.version_bridge(rt, self.inner.version_node()))
    }
}

/// Rows by key: a collection of `(key, row)` pairs holding at most one row
/// per key, with keys matched as a dict matches them. `collection` is the
/// rows as a `Collection`, for the operators.
///
/// Rows must be hashable. A change hashes each added pair's row in the pair
/// collection, and each added or removed row again in `collection`'s output
/// staging, which also runs one row `__eq__` per removed row; before Python
/// 3.14, which caches a tuple's hash, a removed pair's row is hashed in the
/// pair collection too. So an upsert that inserts pays two row `__hash__`
/// calls, one that replaces pays three (four before 3.14) and one row
/// `__eq__` against the held row, and one with an `==` row changes nothing.
#[pyclass(name = "KeyedCollection", unsendable)]
struct PyKeyedCollection {
    pairs: IncrCollection<PyValue>,
    rows: IncrCollection<PyValue>,
    index: Arc<SharedIndex>,
    rt: Py<PyRuntime>,
}

impl OnRuntime for PyKeyedCollection {
    fn runtime(&self) -> &Py<PyRuntime> {
        &self.rt
    }
}

impl PyKeyedCollection {
    /// The index, refused to a key or row method that calls back into this
    /// collection while an operation on it runs.
    fn index(&self) -> PyResult<IndexGuard<'_>> {
        self.index.try_borrow_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "KeyedCollection used from a key or row method during an operation on it",
            )
        })
    }
}

#[pymethods]
impl PyKeyedCollection {
    #[new]
    fn new(runtime: Bound<'_, PyRuntime>) -> PyResult<Self> {
        let index = Arc::new(SharedIndex::new());
        let (pairs, rows) = runtime.borrow().engine_call(|rt| {
            let pairs = rt.create_collection::<PyValue>();
            let rows = pairs.map(rt, |pair: &PyValue| -> PyValue {
                Python::attach(|py| PyValue(or_stash(pair.0.bind(py).get_item(1)).unbind()))
            });
            let weak: std::sync::Weak<SharedIndex> = Arc::downgrade(&index);
            rt.register_state(Some(rows.version_node().erase()), weak);
            (pairs, rows)
        })?;
        Ok(PyKeyedCollection {
            pairs,
            rows,
            index,
            rt: runtime.unbind(),
        })
    }

    /// Hold `row` for `key`: nothing happens if the held row is `==` to
    /// it, otherwise the pair is replaced or inserted.
    fn upsert(&self, py: Python<'_>, key: Bound<'_, PyAny>, row: Bound<'_, PyAny>) -> PyResult<()> {
        let mut index = self.index()?;
        let mut staged = StagedKeys::default();
        staged.stage(&index, &key, Some(&row))?;
        index.apply_staged(py, staged, |changes| {
            self.engine_call(py, |rt| self.pairs.apply(rt, changes))
        })?;
        Ok(())
    }

    /// Remove `key`'s pair. Raises `KeyError` if no row is held for it.
    fn remove(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<()> {
        let mut index = self.index()?;
        let mut staged = StagedKeys::default();
        if !staged.stage(&index, &key, None)? {
            return Err(pyo3::exceptions::PyKeyError::new_err(key.unbind()));
        }
        index.apply_staged(py, staged, |changes| {
            self.engine_call(py, |rt| self.pairs.apply(rt, changes))
        })?;
        Ok(())
    }

    /// The row held for `key`, or `None`.
    fn get<'py>(&self, key: Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.index()?.row(&key)
    }

    fn __contains__(&self, key: Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.index()?.row(&key)?.is_some())
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.index()?.len())
    }

    /// Make the held rows equal `rows`, a mapping or an iterable of
    /// `(key, row)` pairs (a later pair for a key wins), in one atomic
    /// change. Returns how many keys were `(inserted, replaced, removed)`.
    fn reconcile(&self, py: Python<'_>, rows: &Bound<'_, PyAny>) -> PyResult<KeyCounts> {
        let mut index = self.index()?;
        let mut staged = StagedKeys::default();
        let pairs = if rows.hasattr("items")? {
            rows.call_method0("items")?
        } else {
            rows.clone()
        };
        for pair in pairs.try_iter()? {
            let (key, row): (Bound<'_, PyAny>, Bound<'_, PyAny>) = pair?.extract()?;
            staged.stage(&index, &key, Some(&row))?;
        }
        staged.stage_removal_of_the_rest(py, &index)?;
        index.apply_staged(py, staged, |changes| {
            self.engine_call(py, |rt| self.pairs.apply(rt, changes))
        })
    }

    #[getter]
    fn collection(&self, py: Python<'_>) -> PyCollection {
        PyCollection {
            inner: self.rows.clone(),
            rt: self.rt.clone_ref(py),
        }
    }
}

#[pyclass(name = "SortedCollection", unsendable)]
struct PySortedCollection {
    inner: SortedCollection<PyValue, PyValue>,
    rt: Py<PyRuntime>,
}

impl OnRuntime for PySortedCollection {
    fn runtime(&self) -> &Py<PyRuntime> {
        &self.rt
    }
}

#[pymethods]
impl PySortedCollection {
    fn pairwise(&self, py: Python<'_>) -> PyResult<PyCollection> {
        let mapped = self.engine_call(py, |rt| {
            let pair_collection = self.inner.pairwise(rt);
            pair_collection.map(rt, |pair: &(PyValue, PyValue)| -> PyValue {
                Python::attach(|py| {
                    let tuple = pyo3::types::PyTuple::new(
                        py,
                        &[pair.0 .0.clone_ref(py), pair.1 .0.clone_ref(py)],
                    )
                    .unwrap();
                    PyValue(tuple.into_any().unbind())
                })
            })
        })?;
        Ok(PyCollection {
            inner: mapped,
            rt: self.rt.clone_ref(py),
        })
    }

    fn window(&self, py: Python<'_>, size: usize) -> PyResult<PyCollection> {
        let mapped = self.engine_call(py, |rt| {
            let win_collection = self.inner.window(rt, size);
            win_collection.map(rt, |window: &Vec<PyValue>| -> PyValue {
                Python::attach(|py| {
                    let py_list = pyo3::types::PyList::empty(py);
                    for elem in window.iter() {
                        py_list.append(elem.0.clone_ref(py)).unwrap();
                    }
                    PyValue(py_list.into_any().unbind())
                })
            })
        })?;
        Ok(PyCollection {
            inner: mapped,
            rt: self.rt.clone_ref(py),
        })
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let entries = self.engine_call(py, |_| self.inner.snapshot())?;
        let list = pyo3::types::PyList::empty(py);
        for entry in entries {
            list.append(entry.0.clone_ref(py))?;
        }
        Ok(list.into_any().unbind())
    }

    fn snapshot_len(&self, py: Python<'_>) -> PyResult<usize> {
        self.engine_call(py, |_| self.inner.snapshot_len())
    }

    #[getter]
    fn version_node(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        let this = self.rt.bind(py).borrow();
        let rt = this.rt()?;
        engine_call(rt, || this.version_bridge(rt, self.inner.version_node()))
    }
}

#[pyclass(name = "GroupedCollection", unsendable)]
struct PyGroupedCollection {
    inner: incr_compute::GroupedCollection<PyValue, PyValue>,
    rt: Py<PyRuntime>,
}

impl OnRuntime for PyGroupedCollection {
    fn runtime(&self) -> &Py<PyRuntime> {
        &self.rt
    }
}

#[pymethods]
impl PyGroupedCollection {
    fn keys(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let keys = self.engine_call(py, |_| self.inner.keys())?;
        let list = pyo3::types::PyList::empty(py);
        for key in keys {
            list.append(key.0.clone_ref(py))?;
        }
        Ok(list.into_any().unbind())
    }

    fn get_group(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Option<PyCollection>> {
        let py_key = PyValue(key);
        let group = self.engine_call(py, |_| self.inner.get_group(&py_key))?;
        Ok(group.map(|collection| PyCollection {
            inner: collection,
            rt: self.rt.clone_ref(py),
        }))
    }

    fn group_count(&self, py: Python<'_>) -> PyResult<usize> {
        self.engine_call(py, |_| self.inner.group_count())
    }

    #[getter]
    fn version_node(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        let this = self.rt.bind(py).borrow();
        let rt = this.rt()?;
        engine_call(rt, || this.version_bridge(rt, self.inner.version_node()))
    }
}

#[pyclass(name = "Runtime", unsendable)]
struct PyRuntime {
    holder: Holder,
    /// The query each collection's `version_node` getter hands out,
    /// keyed by the collection's internal version node, so repeated
    /// access and every wrapper of one collection share a single node.
    version_bridges: RefCell<HashMap<Incr<u64>, Incr<PyValue>>>,
    state: RtState,
}

#[expect(
    clippy::large_enum_variant,
    reason = "one per runtime, and boxing would add a hop to every call"
)]
enum Holder {
    Owned(Runtime),
    Closed,
}

impl PyRuntime {
    fn rt(&self) -> PyResult<&Runtime> {
        match &self.holder {
            Holder::Owned(rt) => Ok(rt),
            Holder::Closed => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "runtime is closed",
            )),
        }
    }

    /// [`engine_call`] on this runtime, which `f` receives.
    fn engine_call<R>(&self, f: impl FnOnce(&Runtime) -> R) -> PyResult<R> {
        let rt = self.rt()?;
        engine_call(rt, || f(rt))
    }

    /// The Python-visible node mirroring `version`. Created on first use
    /// and again only if the user deleted the previous one, whose handle
    /// then stays stale like any deleted node's.
    fn version_bridge(&self, rt: &Runtime, version: Incr<u64>) -> PyNodeId {
        let mut bridges = self.version_bridges.borrow_mut();
        if let Some(&bridge) = bridges.get(&version) {
            if rt.is_alive(bridge) {
                return PyNodeId { inner: bridge };
            }
        }
        let bridge = rt.create_query(move |rt| -> PyValue {
            let n: u64 = rt.get(version);
            Python::attach(|py| PyValue(n.into_pyobject(py).unwrap().into_any().unbind()))
        });
        bridges.insert(version, bridge);
        PyNodeId { inner: bridge }
    }
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> Self {
        PyRuntime {
            holder: Holder::Owned(Runtime::new()),
            version_bridges: RefCell::new(HashMap::new()),
            state: RtState::default(),
        }
    }

    /// Drop the engine and everything it holds: values, closures and the
    /// Python objects they capture. Every later call on this runtime or an
    /// object derived from it raises. Refused while a call on it is running.
    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        let (holder, bridges) = {
            let mut this = slf.try_borrow_mut().map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "cannot close the runtime while a call on it is running",
                )
            })?;
            let holder = std::mem::replace(&mut this.holder, Holder::Closed);
            let bridges = std::mem::take(this.version_bridges.get_mut());
            let lookups = this.state.lookups().clear();
            (holder, (bridges, lookups))
        };
        // Dropped after the borrow ends, so a finalizer that calls back into
        // this runtime sees it closed instead of a held borrow.
        drop(bridges);
        drop(holder);
        Ok(())
    }

    fn create_input(&self, value: Py<PyAny>) -> PyResult<PyNodeId> {
        let node = self.engine_call(|rt| rt.create_input(PyValue(value)))?;
        Ok(PyNodeId { inner: node })
    }

    /// The node's value: a `NotReady` value as stored, and an `Incomplete`
    /// value's `value`.
    fn get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        let val: PyValue = self.engine_call(|rt| rt.get(node.inner))?;
        Ok(ready::unwrapped(py, val.0))
    }

    /// `(value, status)`: the value `get` returns, and `None`, the
    /// `NotReady` value, or the `Incomplete` value.
    fn get_with_status(&self, py: Python<'_>, node: PyNodeId) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let val: PyValue = self.engine_call(|rt| rt.get(node.inner))?;
        Ok(ready::with_status(py, val.0))
    }

    /// The status `get_with_status` gives.
    fn status(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        Ok(self.get_with_status(py, node)?.1)
    }

    /// Like `get`, but a cycle that leaves the node without a value raises
    /// `CycleError` as an ordinary exception.
    fn try_get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        let val = try_read::<Exc>(self.engine_call(|rt| rt.try_get(node.inner))?)?;
        Ok(ready::unwrapped(py, val))
    }

    fn set(&self, node: PyNodeId, value: Py<PyAny>) -> PyResult<()> {
        self.engine_call(|rt| rt.set(node.inner, PyValue(value)))
    }

    /// Delete a node and recycle its slot. Every surviving handle to it
    /// raises on use. The node must have no dependents.
    fn delete_node(&self, node: PyNodeId) -> PyResult<()> {
        self.engine_call(|rt| rt.delete_node(node.inner))
    }

    /// Delete the node and the nodes it owns only if none of them has a
    /// dependent. Raises `HasDependentsError` (deleting nothing) if one
    /// does, and `KeyError` if the node is not alive.
    fn try_delete_node(&self, py: Python<'_>, node: PyNodeId) -> PyResult<()> {
        let inner = node.inner;
        self.engine_call(|rt| rt.try_delete_node(inner))?
            .map_err(|e| {
                delete_error(
                    e,
                    node.into_pyobject(py)
                        .map(|b| b.into_any().unbind())
                        .unwrap_or_else(|_| py.None()),
                )
            })
    }

    /// Delete the node, every node that depends on it (transitively)
    /// and every node an operator created for them. Returns how many
    /// nodes were deleted; 0 if the node was already gone. A query that
    /// has never run is not reached, and raises as stale on first use.
    fn delete_with_dependents(&self, node: PyNodeId) -> PyResult<usize> {
        self.engine_call(|rt| rt.delete_with_dependents(node.inner))
    }

    /// True while the node exists in this runtime. False after it is
    /// deleted, and for a node from another runtime.
    fn is_alive(&self, node: PyNodeId) -> PyResult<bool> {
        self.engine_call(|rt| rt.is_alive(node.inner))
    }

    /// Report the node through `drain_dirtied_watched()` whenever a change
    /// moves it off Clean. A node that has no current value when watched
    /// is reported at once.
    fn watch(&self, node: PyNodeId) -> PyResult<()> {
        self.engine_call(|rt| rt.watch(node.inner))
    }

    /// Stop reporting the node, and drop its pending report.
    fn unwatch(&self, node: PyNodeId) -> PyResult<()> {
        self.engine_call(|rt| rt.unwatch(node.inner))
    }

    /// The watched nodes a change moved off Clean since the last drain,
    /// each once, in the order the changes reached them. A node deleted
    /// since is left out.
    fn drain_dirtied_watched(&self) -> PyResult<Vec<PyNodeId>> {
        self.engine_call(|rt| {
            rt.drain_dirtied_watched()
                .into_iter()
                .filter_map(|h| rt.typed_handle::<PyValue>(h))
                .map(|inner| PyNodeId { inner })
                .collect()
        })
    }

    /// Register `callback` to fire on `stabilize()` whenever the node's
    /// value changed since the last firing.
    fn observe(&self, node: PyNodeId, callback: Py<PyAny>) -> PyResult<PyObserverId> {
        let callback = self.state.hold(callback);
        let id = self.engine_call(|rt| {
            rt.observe(node.inner, move |v: &PyValue| {
                Python::attach(|py| {
                    call_stashing(py, &callback, (v.0.clone_ref(py),));
                });
            })
        })?;
        Ok(PyObserverId { inner: id })
    }

    fn unobserve(&self, id: PyObserverId) -> PyResult<()> {
        self.engine_call(|rt| rt.unobserve(id.inner))
    }

    /// Bring observed nodes up to date and fire callbacks for changed
    /// values.
    fn stabilize(&self) -> PyResult<()> {
        self.engine_call(|rt| rt.stabilize())
    }

    /// A query computing `py_func(runtime_ref)`. `fallback(ids)` or a
    /// `Fixpoint` resolves a cycle through it; `replay=True` lets a read
    /// past the frame budget replay instead of nesting.
    #[pyo3(signature = (py_func, *, fallback=None, fixpoint=None, replay=false))]
    fn create_query(
        &self,
        py: Python<'_>,
        py_func: Py<PyAny>,
        fallback: Option<Py<PyAny>>,
        fixpoint: Option<PyRef<'_, PyFixpoint>>,
        replay: bool,
    ) -> PyResult<PyNodeId> {
        let opts = query_opts(py, fallback, fixpoint, replay, |f| self.state.hold(f))?;
        let f = query_fn(self.state.hold(py_func));
        let node = self.engine_call(|rt| rt.create_query_with(f, opts))?;
        Ok(PyNodeId { inner: node })
    }

    fn create_table(slf: PyRef<'_, Self>) -> PyResult<PyNodeTable> {
        let table = slf.engine_call(|rt| rt.create_table::<PyValue, PyValue>())?;
        let rt: Py<PyRuntime> = slf.into();
        Ok(PyNodeTable { inner: table, rt })
    }

    /// The value a replayed read returns in place of the node it wanted.
    /// Set once per runtime.
    fn set_placeholder(&self, value: Py<PyAny>) -> PyResult<()> {
        self.engine_call(|rt| rt.set_placeholder(PyValue(value)))
    }

    /// The frame budget past which a replayable query's read replays.
    fn set_max_frames(&self, n: u32) -> PyResult<()> {
        self.engine_call(|rt| rt.set_max_frames(n))
    }

    /// A stack budget in bytes beside the frame budget; 0 turns it off.
    fn set_max_stack_bytes(&self, n: usize) -> PyResult<()> {
        self.engine_call(|rt| rt.set_max_stack_bytes(n))
    }

    fn create_collection(slf: PyRef<'_, Self>) -> PyResult<PyCollection> {
        let col = slf.engine_call(|rt| rt.create_collection::<PyValue>())?;
        let rt: Py<PyRuntime> = slf.into();
        Ok(PyCollection { inner: col, rt })
    }

    fn set_label(&self, node: PyNodeId, label: String) -> PyResult<()> {
        self.engine_call(|rt| rt.set_label(node.inner.slot(), label))
    }

    fn set_label_by_id(&self, id: u32, label: String) -> PyResult<()> {
        self.engine_call(|rt| rt.set_label(id, label))
    }

    fn get_traced(&self, node: PyNodeId) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (val, trace, nodes): (PyValue, PropagationTrace, Vec<TracedNode>) =
            self.engine_call(|rt| {
                let (val, trace) = rt.get_traced(node.inner);
                let nodes = trace
                    .node_traces
                    .iter()
                    .map(|nt| (rt.label(nt.id.0), traced_value(rt, nt.id)))
                    .collect();
                (val, trace, nodes)
            })?;
        Python::attach(|py| {
            let trace_dict = pyo3::types::PyDict::new(py);
            trace_dict.set_item("target", trace.target.0)?;
            trace_dict.set_item("total_nodes", trace.total_nodes)?;
            trace_dict.set_item("nodes_recomputed", trace.nodes_recomputed)?;
            trace_dict.set_item("nodes_cutoff", trace.nodes_cutoff)?;
            trace_dict.set_item("elapsed_ns", trace.elapsed_ns)?;

            let node_traces = pyo3::types::PyList::empty(py);
            for (nt, (label, value)) in trace.node_traces.iter().zip(nodes) {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("id", nt.id.0)?;
                d.set_item("label", label)?;
                d.set_item("value", value)?;
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

            Ok((ready::unwrapped(py, val.0), trace_dict.into_any().unbind()))
        })
    }

    fn graph_snapshot(&self) -> PyResult<Py<PyAny>> {
        let infos = self.engine_call(|rt| rt.graph_snapshot())?;
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

    /// Memory held by this runtime: `{"bytes", "nodes", "py_refs",
    /// "by_kind", "by_label"}`. `bytes` estimates what the engine holds from
    /// sizes and capacities (node slots, values' own slots, closures' captures,
    /// dependency lists, labels, collection logs and operator state); the
    /// Python objects themselves are not in it. `py_refs` counts the
    /// references the engine holds to Python objects: node values,
    /// collection elements and logged changes, operator state, keyed
    /// indexes, and every callable handed to it, but not the engine's own
    /// copies of a `window`'s windows. `by_kind` splits nodes into "input",
    /// "query" and "collection" (a collection's version node), and puts all
    /// collection and operator state under "collection". `by_label` totals
    /// labelled nodes by the label up to its first `/`, with a collection's
    /// state under the label of its `version_node`; callables are not
    /// attributed to labels. Walks the whole graph; nothing is counted
    /// between calls.
    fn memory_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let stats = self.engine_call(|rt| rt.memory_stats())?;
        let bridges = self
            .version_bridges
            .borrow()
            .iter()
            .map(|(version, bridge)| (version.slot(), bridge.slot()))
            .collect();
        memory::stats_dict(py, &stats, self.state.callables(), &bridges)
    }

    /// The lookup node for `key`: an input created as `Pending(key)`, and
    /// queued for `drain_pending_lookups`, the first time `key` is asked
    /// for here or through `RuntimeRef.lookup`. No query owns it, so it
    /// keeps its value when the queries reading it stop.
    fn create_lookup(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<PyNodeId> {
        let lookups = self.state.lookups();
        let node = self.engine_call(|rt| lookups.node(rt, py, &key))?;
        Ok(PyNodeId { inner: node })
    }

    /// The keys queued since the last drain whose lookup is still
    /// `Pending`, each once.
    fn drain_pending_lookups(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let lookups = self.state.lookups();
        let keys = self.engine_call(|rt| lookups.drain(rt, py))?;
        Ok(keys.into_iter().map(|key| key.0).collect())
    }

    /// Set each lookup to its value, from a mapping or an iterable of
    /// `(key, value)` pairs, creating the ones not asked for yet. Not
    /// atomic: if a key's `__hash__` or `__eq__` raises, the keys before it
    /// are set.
    fn resolve_lookups(&self, py: Python<'_>, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let pairs = if values.hasattr("items")? {
            values.call_method0("items")?
        } else {
            values.clone()
        };
        let pairs = pairs
            .try_iter()?
            .map(|pair| pair?.extract::<(Py<PyAny>, Py<PyAny>)>())
            .collect::<PyResult<Vec<_>>>()?;
        let lookups = self.state.lookups();
        self.engine_call(|rt| {
            for (key, value) in pairs {
                lookups.set(rt, py, &key, value);
            }
        })
    }

    /// Set `key`'s lookup to `LookupFailed(reason, key)`.
    fn fail_lookup(&self, py: Python<'_>, key: Py<PyAny>, reason: Py<PyAny>) -> PyResult<()> {
        let failed = lookup_failed(py, reason, key.clone_ref(py))?;
        let lookups = self.state.lookups();
        self.engine_call(|rt| lookups.set(rt, py, &key, failed))
    }

    /// Set each key's lookup back to `Pending(key)` and queue it again.
    fn retry_lookups(&self, py: Python<'_>, keys: &Bound<'_, PyAny>) -> PyResult<()> {
        let keys = keys
            .try_iter()?
            .map(|key| key.map(Bound::unbind))
            .collect::<PyResult<Vec<_>>>()?;
        let lookups = self.state.lookups();
        self.engine_call(|rt| {
            for key in &keys {
                lookups.retry(rt, py, key);
            }
        })
    }

    fn node_count(&self) -> PyResult<usize> {
        self.engine_call(|rt| rt.node_count())
    }
}

#[pymodule]
fn incr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CycleError", m.py().get_type::<CycleError>())?;
    m.add(
        "HasDependentsError",
        m.py().get_type::<HasDependentsError>(),
    )?;
    m.add("_NotReadySignal", m.py().get_type::<_NotReadySignal>())?;
    m.add_class::<PyNotReady>()?;
    m.add_class::<PyPending>()?;
    m.add_class::<PyLookupFailed>()?;
    m.add_class::<PyUnavailable>()?;
    m.add_class::<PyIncomplete>()?;
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyNodeId>()?;
    m.add_class::<PyObserverId>()?;
    m.add_class::<PyRuntimeRef>()?;
    m.add_class::<PyFixpoint>()?;
    m.add_class::<PyNodeTable>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PyKeyedCollection>()?;
    m.add_class::<PySortedCollection>()?;
    m.add_class::<PyGroupedCollection>()?;
    Ok(())
}
