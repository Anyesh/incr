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
mod lookups;
mod memory;
mod overlay;
mod persist;
mod state;

use pyo3::prelude::*;
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak};

use incr_compute::{
    Change, DeleteError, ErasedIncr, Incr, IncrCollection, NodeId, NodeTable, PropagationTrace,
    Runtime, SortedCollection,
};
use incr_core::{Local, RoutingFailed};
use incr_core::{PairOp, StateStats};
use incr_python_persist::classes::KeyedSource;
use incr_python_persist::classes::{key_counts, reconcile_pairs, KeyCounts};
use overlay::{Born, Undo};
use persist::PyPersistedKeyed;
use state::{LocalState, PyCallable, RtState};

use incr_python_persist::ready;
use incr_python_protocol::notready::{
    lookup_failed, PyIncomplete, PyLookupFailed, PyNotReady, PyPending, PyUnavailable,
};
use incr_python_protocol::snapshot::{
    bounded_snapshot_to_py, full_snapshot_to_py, SnapshotRequest,
};
use incr_python_protocol::{
    call_stashing, in_runtime, init_fn, or_stash, pyerr_boundary, query_opts, raise_into_engine,
    routing_error, try_read, Exceptions, PyFixpoint, PyKey, PyValue,
};
use persist::{_NotReadySignal, Signal};

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

/// Runs a `#[pymethods]` engine call that writes. Its writes only queue
/// `group_by` routers, so no key function runs inside a write; the next
/// read routes them. A Local runtime is touched by one thread, so its
/// waits never block on another thread and the GIL stays held.
fn engine_write<R>(rt: &Runtime, f: impl FnOnce() -> R) -> PyResult<R> {
    let _suppressed = rt.hold_routing_suppressed();
    pyerr_boundary::<Exc, _>(|| in_runtime(rt, f))
}

/// Runs a `#[pymethods]` engine call that reads. A routing failure an
/// earlier call stored, or one from routing the routers queued since, is
/// raised before `f` runs, so a consuming read never loses what it took.
/// A failure from routing inside `f` is stored for the next read.
fn engine_read<R>(rt: &Runtime, f: impl FnOnce() -> R) -> PyResult<R> {
    pyerr_boundary::<Exc, _>(|| {
        in_runtime(rt, || {
            if let Some(failure) = rt.take_routing_failure().or_else(|| rt.route_pending()) {
                raise_routing(failure);
            }
            f()
        })
    })
}

/// Runs a `RuntimeRef` call from inside a running query. An entry of the
/// runtime is open on this thread, so there is no stored routing failure
/// to raise and nothing to route: the call is the boundary and the
/// runtime marker only.
fn engine_in_query<R>(rt: &Runtime, f: impl FnOnce() -> R) -> PyResult<R> {
    pyerr_boundary::<Exc, _>(|| in_runtime(rt, f))
}

#[cold]
#[inline(never)]
fn raise_routing(failure: RoutingFailed) -> ! {
    raise_into_engine(Python::attach(|py| routing_error(py, failure)))
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

    /// Raises if this object was created inside an overlay that has ended.
    fn check_alive(&self) -> PyResult<()> {
        Ok(())
    }

    /// Lists a node this object's call created in the active overlay, so
    /// its exit deletes it.
    fn created(&self, py: Python<'_>, node: ErasedIncr) {
        self.state(py).journal(|overlay| overlay.created(node));
    }

    /// [`engine_write`] on this object's runtime, which `f` receives.
    fn engine_write<R>(&self, py: Python<'_>, f: impl FnOnce(&Runtime) -> R) -> PyResult<R> {
        self.check_alive()?;
        let rt = self.runtime().bind(py).borrow();
        let rt = rt.rt()?;
        engine_write(rt, || f(rt))
    }

    /// [`engine_read`] on this object's runtime, which `f` receives.
    fn engine_read<R>(&self, py: Python<'_>, f: impl FnOnce(&Runtime) -> R) -> PyResult<R> {
        self.check_alive()?;
        let rt = self.runtime().bind(py).borrow();
        let rt = rt.rt()?;
        engine_read(rt, || f(rt))
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
                PyValue(ready::call_query::<Signal, _>(
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
        let val: PyValue = engine_in_query(rt, || rt.get(node.inner))?;
        ready::read::<Signal>(py, val.0, missing)
    }

    /// Like `get`, but a cycle that leaves the node without a value
    /// raises `CycleError` as an ordinary exception the query may catch.
    fn try_get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        let (rt, missing) = self.rt_and_missing()?;
        let val = try_read::<Exc>(engine_in_query(rt, || rt.try_get(node.inner))?)?;
        ready::read::<Signal>(py, val, missing)
    }

    /// The value of `key`'s lookup (see `Runtime.create_lookup`), read as
    /// `get` reads a node: `Pending` makes this query's value `Pending`.
    fn lookup(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let (rt, state) = self.rt_and_state()?;
        let (_, missing) = self.rt_and_missing()?;
        let lookups = state.lookups();
        let val = engine_in_query(rt, || rt.get(lookups.node(rt, py, &key)))?;
        ready::read::<Signal>(py, val.0, missing)
    }

    /// `(value, status)` as `Runtime.get_with_status` gives them, with
    /// neither rule of `get` applied.
    fn get_status(&self, py: Python<'_>, node: PyNodeId) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let rt = self.rt()?;
        let val: PyValue = engine_in_query(rt, || rt.get(node.inner))?;
        Ok(ready::with_status(py, val.0))
    }

    /// The row `kc` holds for `key`, or `None`, read as `get` reads a node,
    /// with a dependency on that key alone: the query reruns when a write
    /// changes this key's row, inserts it or removes it, and not for other
    /// keys. Raises `ValueError` for a keyed collection of another runtime.
    fn get_key(
        &self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        key: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let (rt, missing) = self.rt_and_missing()?;
        let (source, label) = kc.key_source(py, rt)?;
        let key = PyKey(key);
        let row = engine_in_query(rt, || source.get_key(rt, &key, label))?;
        ready::read::<Signal>(py, row.map_or_else(|| py.None(), |row| row.0), missing)
    }

    /// `(row, status)` for `kc`'s `key` as `get_status` gives them, read
    /// with the dependency of `get_key`; `(None, None)` for an absent key.
    fn get_key_status(
        &self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        key: Py<PyAny>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let rt = self.rt()?;
        let (source, label) = kc.key_source(py, rt)?;
        let key = PyKey(key);
        let row = engine_in_query(rt, || source.get_key(rt, &key, label))?;
        Ok(ready::with_status(
            py,
            row.map_or_else(|| py.None(), |row| row.0),
        ))
    }

    /// The value of `table`'s entry for `key` (see `Store.memo_table`),
    /// restored from the memo or computed, read as `get` reads a node.
    fn memo(
        &self,
        py: Python<'_>,
        table: PyRef<'_, persist::PyMemoTable>,
        key: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let (rt, missing) = self.rt_and_missing()?;
        let val = table.value_of(py, rt, PyValue(key))?;
        ready::read::<Signal>(py, val.0, missing)
    }

    /// The input child named `key` of this query, created with `init()`
    /// the first time this query asks for it. Children this query stops
    /// asking for are deleted when it next lands.
    fn child_input(&self, key: Py<PyAny>, init: Py<PyAny>) -> PyResult<PyNodeId> {
        let rt = self.rt()?;
        let (key, init) = (PyValue(key), init_fn(init));
        let node = engine_in_query(rt, || rt.child_input(key, init))?;
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
        let opts = query_opts(py, fallback, fixpoint, replay, |f| callables.callable(f))?;
        let (key, f) = (PyValue(key), query_fn(callables.callable(f)));
        let node = engine_in_query(rt, || rt.child_query_with(key, f, opts))?;
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
        self.rt
            .bind(py)
            .borrow()
            .forbid_in_overlay("NodeTable.get_or_insert_input")?;
        let (key, init) = (PyValue(key), init_fn(init));
        let node = self.engine_read(py, |rt| self.inner.get_or_insert_input(rt, key, init))?;
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
        self.rt
            .bind(py)
            .borrow()
            .forbid_in_overlay("NodeTable.get_or_insert_query")?;
        let callables = self.state(py);
        let opts = query_opts(py, fallback, fixpoint, replay, |f| callables.callable(f))?;
        let (key, f) = (PyValue(key), query_fn(callables.callable(f)));
        let node = self.engine_read(py, |rt| {
            self.inner.get_or_insert_query_with(rt, key, f, opts)
        })?;
        Ok(PyNodeId { inner: node })
    }

    /// The live node for `key`, or `None`. Not allowed inside a query.
    fn get(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Option<PyNodeId>> {
        let key = PyValue(key);
        let node = self.engine_read(py, |rt| self.inner.get(rt, &key))?;
        Ok(node.map(|inner| PyNodeId { inner }))
    }

    /// Deletes `key`'s node and removes the entry. Raises `KeyError` if
    /// the table holds no live node for it.
    fn remove(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<()> {
        self.rt
            .bind(py)
            .borrow()
            .forbid_in_overlay("NodeTable.remove")?;
        let k = PyValue(key.clone_ref(py));
        match self.engine_write(py, |rt| self.inner.remove(rt, &k))? {
            Ok(()) => Ok(()),
            Err(e) => Err(delete_error(e, key)),
        }
    }

    /// The `(key, node)` pairs whose node is alive. Not allowed inside a
    /// query.
    fn items(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, PyNodeId)>> {
        let entries = self.engine_read(py, |rt| {
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
    born: Born,
    /// Set for a collection a store persists, which only the store writes.
    persisted: bool,
}

impl OnRuntime for PyCollection {
    fn runtime(&self) -> &Py<PyRuntime> {
        &self.rt
    }

    fn check_alive(&self) -> PyResult<()> {
        self.born.check()
    }
}

impl PyCollection {
    /// The collection a write goes to: this one, or for a persisted
    /// collection, which only the store writes, the unlogged handle an
    /// overlay's scope gives out.
    #[inline]
    fn target(&self, py: Python<'_>) -> PyResult<Cow<'_, IncrCollection<PyValue>>> {
        if !self.persisted {
            return Ok(Cow::Borrowed(&self.inner));
        }
        self.state(py)
            .journal_with(|overlay| overlay.writable_collection(py, self.inner.version_node().slot()))
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "collection is persisted: write it with store.insert, store.delete, store.replace or store.extend",
                )
            })?
            .map(Cow::Owned)
    }

    /// A collection a call on `parent` just created: dead when the active
    /// overlay ends, and its node deleted then.
    fn made(parent: &impl OnRuntime, py: Python<'_>, inner: IncrCollection<PyValue>) -> Self {
        parent.created(py, inner.version_node().erase());
        PyCollection {
            inner,
            rt: parent.runtime().clone_ref(py),
            born: parent.state(py).born(),
            persisted: false,
        }
    }

    /// Applies `changes` as one atomic step, and in an overlay journals
    /// what it reports applied.
    fn apply_journaled(
        &self,
        py: Python<'_>,
        target: &IncrCollection<PyValue>,
        changes: Vec<Change<PyValue>>,
    ) -> PyResult<Vec<Change<PyValue>>> {
        let applied = self.engine_write(py, |rt| target.apply(rt, changes))?;
        if !applied.is_empty() {
            self.state(py).journal(|overlay| {
                overlay.push(Undo::Collection(target.clone(), applied.clone()));
            });
        }
        Ok(applied)
    }
}

#[pymethods]
impl PyCollection {
    fn insert(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let target = self.target(py)?;
        if self.state(py).overlay_active() {
            self.apply_journaled(py, &target, vec![Change::Insert(PyValue(value))])?;
            return Ok(());
        }
        self.engine_write(py, |rt| target.insert(rt, PyValue(value)))
    }

    fn delete(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<bool> {
        let target = self.target(py)?;
        if self.state(py).overlay_active() {
            let applied =
                self.apply_journaled(py, &target, vec![Change::Delete(PyValue(value))])?;
            return Ok(!applied.is_empty());
        }
        self.engine_write(py, |rt| target.delete(rt, &PyValue(value)))
    }

    fn extend(&self, py: Python<'_>, rows: &Bound<'_, PyAny>) -> PyResult<()> {
        let target = self.target(py)?;
        let values = rows
            .try_iter()?
            .map(|row| row.map(|row| PyValue(row.unbind())))
            .collect::<PyResult<Vec<_>>>()?;
        if self.state(py).overlay_active() {
            self.apply_journaled(
                py,
                &target,
                values.into_iter().map(Change::Insert).collect(),
            )?;
            return Ok(());
        }
        self.engine_write(py, |rt| target.extend(rt, values))
    }

    fn replace(&self, py: Python<'_>, old: Py<PyAny>, new: Py<PyAny>) -> PyResult<bool> {
        let target = self.target(py)?;
        if self.state(py).overlay_active() {
            let applied = self.apply_journaled(
                py,
                &target,
                vec![Change::Delete(PyValue(old)), Change::Insert(PyValue(new))],
            )?;
            // An insert is never skipped, so an empty net change means the
            // delete happened and the equal insert cancelled it.
            return Ok(applied.is_empty() || applied.iter().any(|c| matches!(c, Change::Delete(_))));
        }
        self.engine_write(py, |rt| {
            target.replace(rt, Some(&PyValue(old)), PyValue(new))
        })
    }

    fn snapshot_len(&self, py: Python<'_>) -> PyResult<usize> {
        self.engine_read(py, |_| self.inner.snapshot_len())
    }

    fn filter(&self, py: Python<'_>, predicate: Py<PyAny>) -> PyResult<PyCollection> {
        let predicate = self.state(py).callable(predicate);
        let filtered = self.engine_read(py, |rt| {
            self.inner.filter(rt, move |val: &PyValue| -> bool {
                Python::attach(|py| {
                    elements::keep(val.0.bind(py), |row| {
                        or_stash(call_stashing(py, &predicate, (row,)).is_truthy(py))
                    })
                })
            })
        })?;
        Ok(PyCollection::made(self, py, filtered))
    }

    fn map(&self, py: Python<'_>, func: Py<PyAny>) -> PyResult<PyCollection> {
        let func = self.state(py).callable(func);
        let mapped = self.engine_read(py, |rt| {
            self.inner.map(rt, move |val: &PyValue| -> PyValue {
                Python::attach(|py| {
                    PyValue(elements::map(val.0.bind(py), |row| {
                        call_stashing(py, &func, (row,))
                    }))
                })
            })
        })?;
        Ok(PyCollection::made(self, py, mapped))
    }

    /// The number of ready elements, `Incomplete` with the number of
    /// not-ready and `Incomplete` elements when there are any.
    fn count(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        let (parts, node) = self.engine_read(py, |rt| {
            let parts = self.inner.count_partition(rt, |val: &PyValue| {
                Python::attach(|py| elements::is_ready(val.0.bind(py)))
            });
            let node = rt.create_query(move |rt| -> PyValue {
                let (ready, missing) = rt.get(parts);
                Python::attach(|py| {
                    let ready = ready.into_pyobject(py).unwrap().into_any().unbind();
                    PyValue(elements::total(py, ready, missing))
                })
            });
            (parts, node)
        })?;
        self.created(py, parts.erase());
        self.created(py, node.erase());
        Ok(PyNodeId { inner: node })
    }

    /// `fold_fn` of the list of rows, with not-ready elements left out
    /// and an `Incomplete` element's `value` in; `Incomplete` when any
    /// element is, counting each not-ready element once and each
    /// `Incomplete` element's `missing`.
    fn reduce(&self, py: Python<'_>, fold_fn: Py<PyAny>) -> PyResult<PyNodeId> {
        let fold_fn = self.state(py).callable(fold_fn);
        let node = self.engine_read(py, |rt| {
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
        self.created(py, node.erase());
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
        let (lift, combine) = (callables.callable(lift), callables.callable(combine));
        let lifted_identity = identity.clone_ref(py);
        let node = self.engine_read(py, |rt| {
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
        self.created(py, node.erase());
        Ok(PyNodeId { inner: node })
    }

    fn sort_by_key(&self, py: Python<'_>, key_fn: Py<PyAny>) -> PyResult<PySortedCollection> {
        let key_fn = self.state(py).callable(key_fn);
        let sorted = self.engine_read(py, |rt| {
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
        self.created(py, sorted.version_node().erase());
        Ok(PySortedCollection {
            inner: sorted,
            rt: self.rt.clone_ref(py),
            born: self.state(py).born(),
        })
    }

    fn group_by(&self, py: Python<'_>, key_fn: Py<PyAny>) -> PyResult<PyGroupedCollection> {
        let key_fn = self.state(py).callable(key_fn);
        let grouped = self.engine_read(py, |rt| {
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
        self.created(py, grouped.version_node().erase());
        Ok(PyGroupedCollection {
            inner: grouped,
            rt: self.rt.clone_ref(py),
            born: self.state(py).born(),
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
        let (left_key, right_key) = (callables.callable(left_key), callables.callable(right_key));
        let (joined_node, mapped) = self.engine_read(py, |rt| {
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
            let mapped = joined.map(rt, |pair: &(PyValue, PyValue)| -> PyValue {
                Python::attach(|py| PyValue(elements::pair(py, &pair.0 .0, &pair.1 .0)))
            });
            (joined.version_node(), mapped)
        })?;
        self.created(py, joined_node.erase());
        Ok(PyCollection::made(self, py, mapped))
    }

    #[getter]
    fn version_node(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        self.check_alive()?;
        let this = self.rt.bind(py).borrow();
        let rt = this.rt()?;
        engine_read(rt, || this.version_bridge(rt, self.inner.version_node()))
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
/// calls, one that replaces pays three (four before 3.14) and two row
/// `__eq__` against the held row, and one with an `==` row changes nothing.
///
/// Key and row methods run as value code: one that calls into incr during
/// an operation on the collection raises `RuntimeError`, which the
/// operation raises, changing nothing.
///
/// `label` names the collection's group in `Runtime.graph_summary`: it
/// labels the pair and row collections' version nodes, and the group
/// reports the key count as `"keys"`.
#[pyclass(name = "KeyedCollection", unsendable)]
struct PyKeyedCollection {
    kind: KeyedKind,
    rows: IncrCollection<PyValue>,
    rt: Py<PyRuntime>,
    born: Born,
    label: Option<String>,
}

/// A keyed collection whose pairs are `(key, row)` tuples, read by rank `R`.
type PyKeyedCore<R> = incr_core::KeyedCollection<PyKey, PyValue, Local, PyValue, R>;

/// A keyed collection whose pairs are `(key, row)` tuples.
pub(crate) type CoreKeyed = PyKeyedCore<()>;

/// The runtime collection behind a persisted keyed collection, whose pairs
/// are ranked by the fingerprint of their logged bytes.
pub(crate) type PersistedCore = PyKeyedCore<u128>;

/// A labelled keyed collection's index, label and pair version node.
type KeyedLabel = (Weak<dyn StateStats>, String, ErasedIncr);

enum KeyedKind {
    Plain(CoreKeyed),
    /// A keyed collection a store persists, which only the store writes,
    /// but for an overlay through its unlogged scope.
    Persisted(PyPersistedKeyed),
}

impl OnRuntime for PyKeyedCollection {
    fn runtime(&self) -> &Py<PyRuntime> {
        &self.rt
    }

    fn check_alive(&self) -> PyResult<()> {
        self.born.check()
    }
}

impl PyKeyedCollection {
    /// The handle the store persisting this collection knows it by.
    pub(crate) fn persisted_handle(&self) -> Option<&PyPersistedKeyed> {
        match &self.kind {
            KeyedKind::Plain(_) => None,
            KeyedKind::Persisted(persisted) => Some(persisted),
        }
    }

    /// A read that only consults state outside the engine must ask the
    /// runtime whether it is closed.
    fn check_open(&self, py: Python<'_>) -> PyResult<()> {
        self.check_alive()?;
        self.rt.bind(py).borrow().rt().map(drop)
    }

    /// Runs `f`, a read of the core index, at an engine boundary that
    /// routes nothing, as the index registers no dependency.
    fn read_core<R>(&self, py: Python<'_>, f: impl FnOnce() -> R) -> PyResult<R> {
        self.check_alive()?;
        let runtime = self.rt.bind(py).borrow();
        let rt = runtime.rt()?;
        pyerr_boundary::<Exc, _>(|| in_runtime(rt, f))
    }

    /// Runs `write` on `keyed`, given a journal in an overlay, and journals
    /// the pair ops it records as `undo` of them, so the exit can apply
    /// their inverses.
    fn write_core<R: incr_core::ElementRank + Send + Sync + 'static>(
        &self,
        py: Python<'_>,
        keyed: &PyKeyedCore<R>,
        undo: fn(PyKeyedCore<R>, Vec<PairOp<PyValue, R>>) -> Undo,
        write: impl FnOnce(
            &PyKeyedCore<R>,
            Option<&mut Vec<PairOp<PyValue, R>>>,
        ) -> PyResult<incr_core::KeyedChange<PyKey>>,
    ) -> PyResult<KeyCounts> {
        let state = self.state(py);
        let mut ops = Vec::new();
        let journal = state.overlay_active().then_some(&mut ops);
        let change = write(keyed, journal)?;
        if !ops.is_empty() {
            state.journal(|overlay| overlay.push(undo(keyed.clone(), ops)));
        }
        Ok(key_counts(change.counts))
    }

    /// Makes the wrapper over a new keyed collection's pair and row
    /// collections: journals them as created in an overlay, and labels
    /// them, reporting `stats` as the group's key count.
    fn wrap(
        runtime: Bound<'_, PyRuntime>,
        kind: KeyedKind,
        pairs: IncrCollection<PyValue>,
        rows: IncrCollection<PyValue>,
        stats: Weak<dyn StateStats>,
        label: Option<String>,
    ) -> PyResult<Self> {
        let state = runtime.borrow().state.clone();
        let persisted = matches!(kind, KeyedKind::Persisted(_));
        state.journal(|overlay| {
            if !persisted {
                overlay.created(pairs.version_node().erase());
            }
            overlay.created(rows.version_node().erase());
        });
        if let Some(label) = &label {
            let rt = runtime.borrow();
            rt.relabel(pairs.version_node().slot(), label.clone())?;
            rt.relabel(rows.version_node().slot(), label.clone())?;
            rt.keyed_labels
                .borrow_mut()
                .push((stats, label.clone(), pairs.version_node().erase()));
        }
        Ok(PyKeyedCollection {
            kind,
            rows,
            rt: runtime.unbind(),
            born: state.born(),
            label,
        })
    }

    /// A keyed collection over one a store persists.
    pub(crate) fn persisted(
        runtime: Bound<'_, PyRuntime>,
        persisted: PyPersistedKeyed,
        label: Option<String>,
    ) -> PyResult<Self> {
        let pairs = persisted.pairs();
        let rows = persisted.rows().clone();
        let stats: Weak<incr_core::KeyIndex<PyValue, u128, Local>> = persisted.index_weak();
        Self::wrap(
            runtime,
            KeyedKind::Persisted(persisted),
            pairs,
            rows,
            stats,
            label,
        )
    }

    /// What a per-key read on `rt` reaches, and the label of the version
    /// inputs it creates. Raises `ValueError` when `rt` is not this
    /// collection's runtime.
    fn key_source(
        &self,
        py: Python<'_>,
        rt: &Runtime,
    ) -> PyResult<(KeyedSource<'_, persist::LocalBinding>, Option<&str>)> {
        self.check_alive()?;
        if !std::ptr::eq(self.rt.bind(py).borrow().rt()?, rt) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "keyed collection belongs to another runtime",
            ));
        }
        let source = match &self.kind {
            KeyedKind::Plain(keyed) => KeyedSource::Plain(keyed),
            KeyedKind::Persisted(persisted) => KeyedSource::Persisted(persisted),
        };
        Ok((source, self.label.as_deref()))
    }

    /// The writable collection behind `persisted`, which only an overlay's
    /// unlogged scope gives out: outside one, the store writes it.
    fn unlogged(&self, py: Python<'_>, persisted: &PyPersistedKeyed) -> PyResult<PersistedCore> {
        self.check_alive()?;
        self.state(py)
            .journal_with(|overlay| overlay.writable_keyed(py, persisted))
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "keyed collection is persisted: write it with store.upsert, store.remove, store.reconcile or a store.batch",
                )
            })?
    }
}

#[pymethods]
impl PyKeyedCollection {
    #[new]
    #[pyo3(signature = (runtime, label=None))]
    fn new(runtime: Bound<'_, PyRuntime>, label: Option<String>) -> PyResult<Self> {
        let keyed = runtime.borrow().engine_read(CoreKeyed::new)?;
        let pairs = keyed.pairs();
        let rows = keyed.rows().clone();
        let stats: Weak<incr_core::KeyIndex<PyValue, (), Local>> = keyed.index_weak();
        Self::wrap(runtime, KeyedKind::Plain(keyed), pairs, rows, stats, label)
    }

    /// Hold `row` for `key`: nothing happens if the held row is `==` to
    /// it, otherwise the pair is replaced or inserted.
    fn upsert(&self, py: Python<'_>, key: Py<PyAny>, row: Py<PyAny>) -> PyResult<()> {
        match &self.kind {
            KeyedKind::Plain(keyed) => self.write_core(py, keyed, Undo::Keyed, |k, journal| {
                self.engine_write(py, |rt| {
                    k.upsert_journaled(rt, PyKey(key), PyValue(row), journal)
                })
            }),
            KeyedKind::Persisted(persisted) => self.write_core(
                py,
                &self.unlogged(py, persisted)?,
                Undo::PersistedKeyed,
                |k, journal| {
                    self.engine_write(py, |rt| {
                        k.upsert_journaled(rt, PyKey(key), PyValue(row), journal)
                    })
                },
            ),
        }
        .map(drop)
    }

    /// Remove `key`'s pair. Raises `KeyError` if no row is held for it.
    fn remove(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<()> {
        let target = PyKey(key.clone_ref(py));
        let (_, _, removed) = match &self.kind {
            KeyedKind::Plain(keyed) => self.write_core(py, keyed, Undo::Keyed, |k, journal| {
                self.engine_write(py, |rt| k.remove_journaled(rt, &target, journal))
            }),
            KeyedKind::Persisted(persisted) => self.write_core(
                py,
                &self.unlogged(py, persisted)?,
                Undo::PersistedKeyed,
                |k, journal| self.engine_write(py, |rt| k.remove_journaled(rt, &target, journal)),
            ),
        }?;
        if removed == 0 {
            return Err(pyo3::exceptions::PyKeyError::new_err(key));
        }
        Ok(())
    }

    /// The row held for `key`, or `None`. Registers no dependency: inside a
    /// query, read a key with `ctx.get_key(kc, key)` so the query reruns
    /// when that key's row changes.
    fn get(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Option<Py<PyAny>>> {
        let key = PyKey(key);
        let row = match &self.kind {
            KeyedKind::Plain(keyed) => self.read_core(py, || keyed.row(&key)),
            KeyedKind::Persisted(persisted) => self.read_core(py, || persisted.row(&key)),
        }?;
        Ok(row.map(|row| row.0))
    }

    fn __contains__(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<bool> {
        let key = PyKey(key);
        match &self.kind {
            KeyedKind::Plain(keyed) => self.read_core(py, || keyed.contains(&key)),
            KeyedKind::Persisted(persisted) => self.read_core(py, || persisted.contains(&key)),
        }
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        match &self.kind {
            KeyedKind::Plain(keyed) => self.read_core(py, || keyed.len()),
            KeyedKind::Persisted(persisted) => self.read_core(py, || persisted.len()),
        }
    }

    /// A new list of the held keys, in no particular order. Registers no
    /// dependency, like `get`.
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let keys = match &self.kind {
            KeyedKind::Plain(keyed) => self.read_core(py, || keyed.keys()),
            KeyedKind::Persisted(persisted) => self.read_core(py, || persisted.keys()),
        }?;
        pyo3::types::PyList::new(py, keys.into_iter().map(|key| key.0))
    }

    /// A new list of the held `(key, row)` tuples, in no particular order.
    /// Registers no dependency, like `get`.
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let items = match &self.kind {
            KeyedKind::Plain(keyed) => self.read_core(py, || keyed.items()),
            KeyedKind::Persisted(persisted) => self.read_core(py, || persisted.items()),
        }?;
        pyo3::types::PyList::new(py, items.into_iter().map(|pair| pair.0))
    }

    /// Make the held rows equal `rows`, a mapping or an iterable of
    /// `(key, row)` pairs (a later pair for a key wins), in one atomic
    /// change. Returns how many keys were `(inserted, replaced, removed)`.
    fn reconcile(&self, py: Python<'_>, rows: &Bound<'_, PyAny>) -> PyResult<KeyCounts> {
        self.check_open(py)?;
        match &self.kind {
            KeyedKind::Plain(keyed) => {
                let pairs = reconcile_pairs(rows)?;
                self.write_core(py, keyed, Undo::Keyed, |k, journal| {
                    self.engine_write(py, |rt| k.reconcile_journaled(rt, pairs, journal))
                })
            }
            KeyedKind::Persisted(persisted) => {
                let keyed = self.unlogged(py, persisted)?;
                let pairs = reconcile_pairs(rows)?;
                self.write_core(py, &keyed, Undo::PersistedKeyed, |k, journal| {
                    self.engine_write(py, |rt| k.reconcile_journaled(rt, pairs, journal))
                })
            }
        }
    }

    #[getter]
    fn collection(&self, py: Python<'_>) -> PyResult<PyCollection> {
        self.check_open(py)?;
        Ok(PyCollection {
            inner: self.rows.clone(),
            rt: self.rt.clone_ref(py),
            born: self.born.clone(),
            persisted: false,
        })
    }
}

#[pyclass(name = "SortedCollection", unsendable)]
struct PySortedCollection {
    inner: SortedCollection<PyValue, PyValue>,
    rt: Py<PyRuntime>,
    born: Born,
}

impl OnRuntime for PySortedCollection {
    fn runtime(&self) -> &Py<PyRuntime> {
        &self.rt
    }

    fn check_alive(&self) -> PyResult<()> {
        self.born.check()
    }
}

#[pymethods]
impl PySortedCollection {
    fn pairwise(&self, py: Python<'_>) -> PyResult<PyCollection> {
        let (pair_collection, mapped) = self.engine_read(py, |rt| {
            let pair_collection = self.inner.pairwise(rt);
            let mapped = pair_collection.map(rt, |pair: &(PyValue, PyValue)| -> PyValue {
                Python::attach(|py| {
                    let tuple = pyo3::types::PyTuple::new(
                        py,
                        &[pair.0 .0.clone_ref(py), pair.1 .0.clone_ref(py)],
                    )
                    .unwrap();
                    PyValue(elements::sequence(
                        py,
                        tuple.into_any().unbind(),
                        [&pair.0 .0, &pair.1 .0].into_iter(),
                    ))
                })
            });
            (pair_collection.version_node(), mapped)
        })?;
        self.created(py, pair_collection.erase());
        Ok(PyCollection::made(self, py, mapped))
    }

    fn window(&self, py: Python<'_>, size: usize) -> PyResult<PyCollection> {
        let (win_node, mapped) = self.engine_read(py, |rt| {
            let win_collection = self.inner.window(rt, size);
            let mapped = win_collection.map(rt, |window: &Vec<PyValue>| -> PyValue {
                Python::attach(|py| {
                    let tuple =
                        pyo3::types::PyTuple::new(py, window.iter().map(|e| e.0.clone_ref(py)))
                            .unwrap();
                    PyValue(elements::sequence(
                        py,
                        tuple.into_any().unbind(),
                        window.iter().map(|e| &e.0),
                    ))
                })
            });
            (win_collection.version_node(), mapped)
        })?;
        self.created(py, win_node.erase());
        Ok(PyCollection::made(self, py, mapped))
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let entries = self.engine_read(py, |_| self.inner.snapshot())?;
        let list = pyo3::types::PyList::empty(py);
        for entry in entries {
            list.append(entry.0.clone_ref(py))?;
        }
        Ok(list.into_any().unbind())
    }

    fn snapshot_len(&self, py: Python<'_>) -> PyResult<usize> {
        self.engine_read(py, |_| self.inner.snapshot_len())
    }

    #[getter]
    fn version_node(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        self.check_alive()?;
        let this = self.rt.bind(py).borrow();
        let rt = this.rt()?;
        engine_read(rt, || this.version_bridge(rt, self.inner.version_node()))
    }
}

#[pyclass(name = "GroupedCollection", unsendable)]
struct PyGroupedCollection {
    inner: incr_compute::GroupedCollection<PyValue, PyValue>,
    rt: Py<PyRuntime>,
    born: Born,
}

impl OnRuntime for PyGroupedCollection {
    fn runtime(&self) -> &Py<PyRuntime> {
        &self.rt
    }

    fn check_alive(&self) -> PyResult<()> {
        self.born.check()
    }
}

#[pymethods]
impl PyGroupedCollection {
    fn keys(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let keys = self.engine_read(py, |_| self.inner.keys())?;
        let list = pyo3::types::PyList::empty(py);
        for key in keys {
            list.append(key.0.clone_ref(py))?;
        }
        Ok(list.into_any().unbind())
    }

    fn get_group(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Option<PyCollection>> {
        let py_key = PyValue(key);
        let group = self.engine_read(py, |_| self.inner.get_group(&py_key))?;
        Ok(group.map(|collection| PyCollection {
            inner: collection,
            rt: self.rt.clone_ref(py),
            born: self.born.clone(),
            persisted: false,
        }))
    }

    fn group_count(&self, py: Python<'_>) -> PyResult<usize> {
        self.engine_read(py, |_| self.inner.group_count())
    }

    #[getter]
    fn version_node(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        self.check_alive()?;
        let this = self.rt.bind(py).borrow();
        let rt = this.rt()?;
        engine_read(rt, || this.version_bridge(rt, self.inner.version_node()))
    }
}

#[pyclass(name = "Runtime", unsendable)]
struct PyRuntime {
    holder: Holder,
    /// The query each collection's `version_node` getter hands out,
    /// keyed by the collection's internal version node, so repeated
    /// access and every wrapper of one collection share a single node.
    version_bridges: RefCell<HashMap<Incr<u64>, Incr<PyValue>>>,
    /// The labelled keyed collections made on this runtime, whose key
    /// counts `graph_summary` reports under their label's group while
    /// their pair collection's version node lives (one made in an overlay
    /// dies at its exit).
    keyed_labels: RefCell<Vec<KeyedLabel>>,
    state: RtState,
}

/// Boxed because `Runtime` is 64-byte aligned and Python aligns a
/// pyclass object to 16 bytes.
enum Holder {
    Owned(Box<Runtime>),
    Persisted(Arc<persist::StoreCore>),
    Poisoned(#[expect(dead_code, reason = "held so close drops it")] Box<Runtime>),
    PoisonedPersisted(
        #[expect(dead_code, reason = "held so close drops it")] Arc<persist::StoreCore>,
    ),
    Closed,
}

const _: () = {
    use std::mem::align_of;
    assert!(align_of::<PyNodeId>() <= 16);
    assert!(align_of::<PyObserverId>() <= 16);
    assert!(align_of::<PyRuntimeRef>() <= 16);
    assert!(align_of::<PyNodeTable>() <= 16);
    assert!(align_of::<PyCollection>() <= 16);
    assert!(align_of::<PyKeyedCollection>() <= 16);
    assert!(align_of::<PySortedCollection>() <= 16);
    assert!(align_of::<PyGroupedCollection>() <= 16);
    assert!(align_of::<PyRuntime>() <= 16);
};

impl PyRuntime {
    fn persisted(core: Arc<persist::StoreCore>) -> Self {
        PyRuntime {
            holder: Holder::Persisted(core),
            version_bridges: RefCell::new(HashMap::new()),
            keyed_labels: RefCell::new(Vec::new()),
            state: RtState::default(),
        }
    }

    /// Closes a store's runtime, for `store.close()`.
    fn close_persisted(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::close_any(slf)
    }

    fn close_any(slf: &Bound<'_, Self>) -> PyResult<()> {
        slf.try_borrow()
            .map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "cannot close the runtime while a call on it is running",
                )
            })?
            .forbid_in_overlay("close")?;
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

    fn rt(&self) -> PyResult<&Runtime> {
        match &self.holder {
            Holder::Owned(rt) => Ok(rt),
            Holder::Persisted(core) => Ok(core.runtime()),
            Holder::Poisoned(_) | Holder::PoisonedPersisted(_) => {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "runtime poisoned by a failed overlay restore",
                ))
            }
            Holder::Closed => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "runtime is closed",
            )),
        }
    }

    pub(crate) fn store_core(&self) -> Option<&Arc<persist::StoreCore>> {
        match &self.holder {
            Holder::Persisted(core) => Some(core),
            _ => None,
        }
    }

    fn relabel(&self, slot: u32, label: String) -> PyResult<()> {
        let old = self.engine_read(|rt| {
            let old = rt.label(slot);
            rt.set_label(slot, label);
            old
        })?;
        self.state
            .journal(|overlay| overlay.push(Undo::Label(slot, old)));
        Ok(())
    }

    /// Keeps the runtime for `close` to drop but refuses every other call:
    /// a half-restored base has no defined state.
    fn poison(&mut self) {
        match std::mem::replace(&mut self.holder, Holder::Closed) {
            Holder::Owned(rt) => self.holder = Holder::Poisoned(rt),
            Holder::Persisted(core) => self.holder = Holder::PoisonedPersisted(core),
            other => self.holder = other,
        }
    }

    /// Raises inside an overlay, for an operation it cannot undo.
    fn forbid_in_overlay(&self, what: &str) -> PyResult<()> {
        if self.state.overlay_active() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{what} cannot run inside an overlay"
            )));
        }
        Ok(())
    }

    /// [`engine_write`] on this runtime, which `f` receives.
    fn engine_write<R>(&self, f: impl FnOnce(&Runtime) -> R) -> PyResult<R> {
        let rt = self.rt()?;
        engine_write(rt, || f(rt))
    }

    /// [`engine_read`] on this runtime, which `f` receives.
    fn engine_read<R>(&self, f: impl FnOnce(&Runtime) -> R) -> PyResult<R> {
        let rt = self.rt()?;
        engine_read(rt, || f(rt))
    }

    fn set_journaled(
        &self,
        node: &PyNodeId,
        value: Py<PyAny>,
        first: bool,
        persisted: bool,
    ) -> PyResult<()> {
        if first && persisted {
            self.state
                .journal_with(|overlay| Python::attach(|py| overlay.touch(py, node.inner)))
                .transpose()?;
        }
        let old = if first {
            Some(self.engine_read(|rt| rt.get(node.inner))?)
        } else {
            None
        };
        self.engine_write(|rt| rt.set(node.inner, PyValue(value)))?;
        if let Some(old) = old {
            self.state
                .journal(|overlay| overlay.push(Undo::Set(node.inner, old)));
        }
        Ok(())
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
        self.state
            .journal(|overlay| overlay.created(bridge.erase()));
        PyNodeId { inner: bridge }
    }
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> Self {
        PyRuntime {
            holder: Holder::Owned(Box::new(Runtime::new())),
            version_bridges: RefCell::new(HashMap::new()),
            keyed_labels: RefCell::new(Vec::new()),
            state: RtState::default(),
        }
    }

    /// Drop the engine and everything it holds: values, closures and the
    /// Python objects they capture. Every later call on this runtime or an
    /// object derived from it raises. Refused while a call on it is running,
    /// and for the runtime of a store, which `store.close()` closes.
    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        if matches!(
            slf.try_borrow().map(|this| matches!(
                this.holder,
                Holder::Persisted(_) | Holder::PoisonedPersisted(_)
            )),
            Ok(true)
        ) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "this runtime belongs to a store: close it with store.close()",
            ));
        }
        Self::close_any(slf)
    }

    /// Run candidate changes against the live graph and put it back
    /// exactly on exit; see `Overlay`.
    fn overlay(slf: &Bound<'_, Self>) -> persist::PyOverlay {
        persist::PyOverlay::new(slf.clone().unbind())
    }

    fn create_input(&self, value: Py<PyAny>) -> PyResult<PyNodeId> {
        let node = self.engine_read(|rt| rt.create_input(PyValue(value)))?;
        self.state.journal(|overlay| overlay.created(node.erase()));
        Ok(PyNodeId { inner: node })
    }

    /// The node's value: a `NotReady` value as stored, and an `Incomplete`
    /// value's `value`.
    fn get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        let val: PyValue = self.engine_read(|rt| rt.get(node.inner))?;
        Ok(ready::unwrapped(py, val.0))
    }

    /// The row `kc` holds for `key`, or `None`, as stored. Outside a query
    /// it creates nothing; `ctx.get_key` is the read that depends on the
    /// key. Raises `ValueError` for a keyed collection of another runtime.
    fn get_key(
        &self,
        py: Python<'_>,
        kc: PyRef<'_, PyKeyedCollection>,
        key: Py<PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let rt = self.rt()?;
        let (source, label) = kc.key_source(py, rt)?;
        let key = PyKey(key);
        let row = engine_read(rt, || source.get_key(rt, &key, label))?;
        Ok(row.map(|row| row.0))
    }

    /// `(value, status)`: the value `get` returns, and `None`, the
    /// `NotReady` value, or the `Incomplete` value.
    fn get_with_status(&self, py: Python<'_>, node: PyNodeId) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let val: PyValue = self.engine_read(|rt| rt.get(node.inner))?;
        Ok(ready::with_status(py, val.0))
    }

    /// The status `get_with_status` gives.
    fn status(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        Ok(self.get_with_status(py, node)?.1)
    }

    /// Like `get`, but a cycle that leaves the node without a value raises
    /// `CycleError` as an ordinary exception.
    fn try_get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        let val = try_read::<Exc>(self.engine_read(|rt| rt.try_get(node.inner))?)?;
        Ok(ready::unwrapped(py, val))
    }

    fn set(&self, node: PyNodeId, value: Py<PyAny>) -> PyResult<()> {
        let persisted = self
            .store_core()
            .is_some_and(|core| core.is_persisted_input(node.inner.slot()));
        if persisted && !self.state.overlay_active() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "input is persisted: write it with store.set(node, value)",
            ));
        }
        let first = self
            .state
            .journal_with(|overlay| overlay.first_set(node.inner))
            .unwrap_or(false);
        let result = self.set_journaled(&node, value, first, persisted);
        if first && result.is_err() {
            self.state.journal(|overlay| overlay.forget_set(node.inner));
        }
        result
    }

    /// Delete a node and recycle its slot. Every surviving handle to it
    /// raises on use. The node must have no dependents.
    fn delete_node(&self, node: PyNodeId) -> PyResult<()> {
        self.forbid_in_overlay("delete_node")?;
        self.engine_write(|rt| rt.delete_node(node.inner))
    }

    /// Delete the node and the nodes it owns only if none of them has a
    /// dependent. Raises `HasDependentsError` (deleting nothing) if one
    /// does, and `KeyError` if the node is not alive.
    fn try_delete_node(&self, py: Python<'_>, node: PyNodeId) -> PyResult<()> {
        self.forbid_in_overlay("try_delete_node")?;
        let inner = node.inner;
        self.engine_write(|rt| rt.try_delete_node(inner))?
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
        self.forbid_in_overlay("delete_with_dependents")?;
        self.engine_write(|rt| rt.delete_with_dependents(node.inner))
    }

    /// True while the node exists in this runtime. False after it is
    /// deleted, and for a node from another runtime.
    fn is_alive(&self, node: PyNodeId) -> PyResult<bool> {
        self.engine_read(|rt| rt.is_alive(node.inner))
    }

    /// Report the node through `drain_dirtied_watched()` whenever a change
    /// moves it off Clean. A node that has no current value when watched
    /// is reported at once.
    fn watch(&self, node: PyNodeId) -> PyResult<()> {
        self.forbid_in_overlay("watch")?;
        self.engine_read(|rt| rt.watch(node.inner))
    }

    /// Stop reporting the node, and drop its pending report.
    fn unwatch(&self, node: PyNodeId) -> PyResult<()> {
        self.forbid_in_overlay("unwatch")?;
        self.engine_read(|rt| rt.unwatch(node.inner))
    }

    /// The watched nodes a change moved off Clean since the last drain,
    /// each once, in the order the changes reached them. A node deleted
    /// since is left out.
    fn drain_dirtied_watched(&self) -> PyResult<Vec<PyNodeId>> {
        self.forbid_in_overlay("drain_dirtied_watched")?;
        self.engine_read(|rt| {
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
        self.forbid_in_overlay("observe")?;
        let callback = self.state.callable(callback);
        let id = self.engine_read(|rt| {
            rt.observe(node.inner, move |v: &PyValue| {
                Python::attach(|py| {
                    call_stashing(py, &callback, (v.0.clone_ref(py),));
                });
            })
        })?;
        Ok(PyObserverId { inner: id })
    }

    fn unobserve(&self, id: PyObserverId) -> PyResult<()> {
        self.forbid_in_overlay("unobserve")?;
        self.engine_read(|rt| rt.unobserve(id.inner))
    }

    /// Bring observed nodes up to date and fire callbacks for changed
    /// values.
    fn stabilize(&self) -> PyResult<()> {
        self.forbid_in_overlay("stabilize")?;
        self.engine_read(|rt| rt.stabilize())
    }

    /// A query computing `py_func(runtime_ref)`. `fallback(ids)` or a
    /// `Fixpoint` resolves a cycle through it; `replay=True` lets a read
    /// past the frame budget replay instead of nesting.
    #[pyo3(signature = (py_func, *, fallback=None, fixpoint=None, replay=false, label=None))]
    fn create_query(
        &self,
        py: Python<'_>,
        py_func: Py<PyAny>,
        fallback: Option<Py<PyAny>>,
        fixpoint: Option<PyRef<'_, PyFixpoint>>,
        replay: bool,
        label: Option<String>,
    ) -> PyResult<PyNodeId> {
        let opts = query_opts(py, fallback, fixpoint, replay, |f| self.state.callable(f))?;
        let f = query_fn(self.state.callable(py_func));
        let node = self.engine_read(|rt| rt.create_query_with(f, opts))?;
        self.state.journal(|overlay| overlay.created(node.erase()));
        if let Some(label) = label {
            self.relabel(node.slot(), label)?;
        }
        Ok(PyNodeId { inner: node })
    }

    fn create_table(slf: PyRef<'_, Self>) -> PyResult<PyNodeTable> {
        slf.forbid_in_overlay("create_table")?;
        let table = slf.engine_read(|rt| rt.create_table::<PyValue, PyValue>())?;
        let rt: Py<PyRuntime> = slf.into();
        Ok(PyNodeTable { inner: table, rt })
    }

    /// The value a replayed read returns in place of the node it wanted.
    /// Set once per runtime.
    fn set_placeholder(&self, value: Py<PyAny>) -> PyResult<()> {
        self.engine_read(|rt| rt.set_placeholder(PyValue(value)))
    }

    /// The frame budget past which a replayable query's read replays.
    fn set_max_frames(&self, n: u32) -> PyResult<()> {
        self.engine_read(|rt| rt.set_max_frames(n))
    }

    /// A stack budget in bytes beside the frame budget; 0 turns it off.
    fn set_max_stack_bytes(&self, n: usize) -> PyResult<()> {
        self.engine_read(|rt| rt.set_max_stack_bytes(n))
    }

    fn create_collection(slf: PyRef<'_, Self>) -> PyResult<PyCollection> {
        let col = slf.engine_read(|rt| rt.create_collection::<PyValue>())?;
        let state = slf.state.clone();
        state.journal(|overlay| overlay.created(col.version_node().erase()));
        let born = state.born();
        let rt: Py<PyRuntime> = slf.into();
        Ok(PyCollection {
            inner: col,
            rt,
            born,
            persisted: false,
        })
    }

    fn set_label(&self, node: PyNodeId, label: String) -> PyResult<()> {
        self.relabel(node.inner.slot(), label)
    }

    fn set_label_by_id(&self, id: u32, label: String) -> PyResult<()> {
        self.relabel(id, label)
    }

    /// Starts or stops compute stats, which count per node the runs that
    /// landed a value, those cut off, the verifications that found it
    /// clean and its closure time. Off by default.
    fn enable_compute_stats(&self, on: bool) -> PyResult<()> {
        self.engine_read(|rt| rt.enable_compute_stats(on))
    }

    /// `{group: {"recomputed", "cutoff", "verified", "elapsed_ns"}}`
    /// counted since stats were enabled or last reset, grouped by the
    /// label up to its first `/` (`None` for unlabelled nodes).
    /// `recomputed` counts every run that landed, `cutoff` those whose
    /// value was equal, and `elapsed_ns` closure time net of nested
    /// computes. `reset=True` zeroes the counts after reading them.
    #[pyo3(signature = (reset=false))]
    fn compute_stats(&self, py: Python<'_>, reset: bool) -> PyResult<Py<PyAny>> {
        let by_label = self.engine_read(|rt| rt.compute_stats(reset))?;
        Ok(
            incr_python_protocol::inspect::compute_stats_dict(py, by_label)?
                .into_any()
                .unbind(),
        )
    }

    fn get_traced(&self, node: PyNodeId) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (val, trace, nodes): (PyValue, PropagationTrace, Vec<TracedNode>) =
            self.engine_read(|rt| {
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
                let d = incr_python_protocol::inspect::trace_entry(py, nt)?;
                d.set_item("label", label)?;
                d.set_item("value", value)?;
                node_traces.append(d)?;
            }
            trace_dict.set_item("node_traces", node_traces)?;

            Ok((ready::unwrapped(py, val.0), trace_dict.into_any().unbind()))
        })
    }

    /// The graph as a list of `{"id", "kind", "label", "dependencies",
    /// "dependents"}` dicts. With any argument given, a breadth-first walk
    /// from `roots` (node handles or collections) returns
    /// `{"nodes": [...], "truncated": bool}` instead: `direction` is
    /// `"dependencies"` (default), `"dependents"` or `"both"`; `depth`
    /// counts edges from a root; `label_prefix` filters the returned nodes
    /// while the walk still passes through the others; `limit` caps them.
    /// Each returned node keeps its full dependency and dependent lists.
    #[pyo3(signature = (roots=None, *, direction=None, depth=None, label_prefix=None, limit=None))]
    fn graph_snapshot(
        &self,
        py: Python<'_>,
        roots: Option<Vec<Bound<'_, PyAny>>>,
        direction: Option<&str>,
        depth: Option<u32>,
        label_prefix: Option<String>,
        limit: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let request = SnapshotRequest::parse(roots, direction, depth, label_prefix, limit, |o| {
            o.extract::<PyRef<'_, PyNodeId>>()
                .ok()
                .map(|n| n.inner.slot())
        })?;
        match request {
            None => {
                let infos = self.engine_read(|rt| rt.graph_snapshot())?;
                full_snapshot_to_py(py, &infos)
            }
            Some(request) => {
                let snap = self.engine_read(|rt| rt.graph_snapshot_bounded(&request.query()))?;
                bounded_snapshot_to_py(py, &snap)
            }
        }
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
        let stats = self.engine_read(|rt| rt.memory_stats())?;
        let bridges = self
            .version_bridges
            .borrow()
            .iter()
            .map(|(version, bridge)| (version.slot(), bridge.slot()))
            .collect();
        memory::stats_dict(py, &stats, self.state.callables(), &bridges)
    }

    /// `{"groups": [{"label", "kind", "nodes"}], "edges": [{"from", "to",
    /// "count"}]}` over the live graph, grouped by the label up to its
    /// first `/` (`None` for unlabelled nodes); `kind` is "input",
    /// "compute" or "mixed". An edge inside one group has `from == to`.
    /// A group a labelled `KeyedCollection` names also has `"keys"`, the
    /// collection's key count (summed over such collections). Walks every
    /// node under the engine's read lock, so the cost is linear in the
    /// node count.
    fn graph_summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let mut keyed = self.keyed_labels.borrow_mut();
        let (summary, alive) = self.engine_read(|rt| {
            let alive: Vec<bool> = keyed
                .iter()
                .map(|(_, _, node)| rt.is_alive(*node))
                .collect();
            (rt.graph_summary(), alive)
        })?;
        let mut keys: HashMap<String, usize> = HashMap::new();
        let mut alive = alive.into_iter();
        keyed.retain(|(index, label, _)| {
            let live = alive.next().unwrap_or(false);
            let Some(index) = index.upgrade() else {
                return false;
            };
            if live {
                *keys
                    .entry(incr_python_protocol::inspect::label_group(label).to_owned())
                    .or_default() += index.state_stats().elements;
            }
            live
        });
        incr_python_protocol::inspect::graph_summary_dict(py, summary, &keys)
    }

    /// The lookup node for `key`: an input created as `Pending(key)`, and
    /// queued for `drain_pending_lookups`, the first time `key` is asked
    /// for here or through `RuntimeRef.lookup`. No query owns it, so it
    /// keeps its value when the queries reading it stop.
    fn create_lookup(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<PyNodeId> {
        let lookups = self.state.lookups();
        let node = self.engine_read(|rt| lookups.node(rt, py, &key))?;
        Ok(PyNodeId { inner: node })
    }

    /// The keys queued since the last drain whose lookup is still
    /// `Pending`, each once.
    fn drain_pending_lookups(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let lookups = self.state.lookups();
        let keys = self.engine_read(|rt| lookups.drain(rt, py))?;
        Ok(keys.into_iter().map(|key| key.0).collect())
    }

    /// Set each lookup to its value, from a mapping or an iterable of
    /// `(key, value)` pairs, creating the ones not asked for yet. Not
    /// atomic: if a key's `__hash__` or `__eq__` raises, the keys before it
    /// are set.
    fn resolve_lookups(&self, py: Python<'_>, values: &Bound<'_, PyAny>) -> PyResult<()> {
        self.forbid_in_overlay("resolve_lookups")?;
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
        self.engine_write(|rt| {
            for (key, value) in pairs {
                lookups.set(rt, py, &key, value);
            }
        })
    }

    /// Set `key`'s lookup to `LookupFailed(reason, key)`.
    fn fail_lookup(&self, py: Python<'_>, key: Py<PyAny>, reason: Py<PyAny>) -> PyResult<()> {
        self.forbid_in_overlay("fail_lookup")?;
        let failed = lookup_failed(py, reason, key.clone_ref(py))?;
        let lookups = self.state.lookups();
        self.engine_write(|rt| lookups.set(rt, py, &key, failed))
    }

    /// Set each key's lookup back to `Pending(key)` and queue it again.
    fn retry_lookups(&self, py: Python<'_>, keys: &Bound<'_, PyAny>) -> PyResult<()> {
        self.forbid_in_overlay("retry_lookups")?;
        let keys = keys
            .try_iter()?
            .map(|key| key.map(Bound::unbind))
            .collect::<PyResult<Vec<_>>>()?;
        let lookups = self.state.lookups();
        self.engine_write(|rt| {
            for key in &keys {
                lookups.retry(rt, py, key);
            }
        })
    }

    fn node_count(&self) -> PyResult<usize> {
        self.engine_read(|rt| rt.node_count())
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
    m.add_class::<persist::PyOverlay>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PyKeyedCollection>()?;
    m.add_class::<PySortedCollection>()?;
    m.add_class::<PyGroupedCollection>()?;
    persist::register(m)?;
    Ok(())
}
