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

use pyo3::marker::Ungil;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use ::incr_concurrent as engine;
use engine::{
    Cycle, CycleStrategy, DeleteError, Fixpoint, Incr, IncrCollection, NodeId, NodeKindInfo,
    NodeTable, PropagationTrace, QueryOpts, Replay, Runtime, SortedCollection, TraceAction,
};

pyo3::create_exception!(
    incr_concurrent,
    CycleError,
    pyo3::exceptions::PyRuntimeError,
    "A read closed a dependency cycle that has no value: `args` is the message and the participants' node ids."
);

pyo3::create_exception!(
    incr_concurrent,
    _IncrEngineUnwind,
    pyo3::exceptions::PyBaseException,
    "Carries an engine unwind through Python frames inside a query; never caught by `except Exception`."
);

thread_local! {
    /// Each exception a Python callback raised in the current outermost
    /// boundary, under the token of the `PyStashPanic` that carries it. Kept
    /// until that boundary exits, because the replay driver can deliver a
    /// held panic to a read made after nested boundaries have returned.
    static PENDING_PYERR: RefCell<Vec<(u64, PyErr)>> = const { RefCell::new(Vec::new()) };
    static BOUNDARY_DEPTH: Cell<u32> = const { Cell::new(0) };
    static LAST_TOKEN: Cell<u64> = const { Cell::new(0) };
    /// The runtime of the innermost engine call or query callback on this
    /// thread, or null outside every one.
    static ACTIVE_RT: Cell<*const Runtime> = const { Cell::new(std::ptr::null()) };
    /// An engine unwind a nested boundary caught, carried through the
    /// Python frames between it and the engine as `_IncrEngineUnwind`
    /// and resumed unchanged where that exception reaches the engine again.
    static ENGINE_UNWIND: RefCell<Option<Box<dyn std::any::Any + Send>>> = const { RefCell::new(None) };
}

/// The panic payload for a stashed Python exception. A boundary re-raises
/// the stash only for the token it was stored under, so a stash whose
/// panic an engine catch swallowed never labels a later error.
struct PyStashPanic {
    token: u64,
    msg: String,
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
    or_stash(f.call1(py, args))
}

/// Unwraps the result of Python code the engine runs: a callback, or a
/// value's `__eq__`, `__hash__` or `__lt__`. An exception is stashed for
/// `pyerr_boundary` to re-raise as itself, and the panic unwinds the
/// engine as any user panic does, so it is never read as "not equal".
fn or_stash<T>(r: PyResult<T>) -> T {
    r.unwrap_or_else(|e| raise_into_engine(e))
}

fn raise_into_engine(e: PyErr) -> ! {
    // The run is discarded either way, so the exception is not kept.
    if engine::engine_unwind_in_progress() {
        drop(e);
        let payload = ENGINE_UNWIND.with(|u| u.borrow_mut().take());
        std::panic::resume_unwind(payload.unwrap_or_else(|| Box::new("incr: engine unwind")));
    }
    if active_frame_doomed() {
        drop(e);
        std::panic::resume_unwind(Box::new("incr: python code raised in a doomed frame"));
    }
    let token = LAST_TOKEN.with(|t| {
        t.set(t.get() + 1);
        t.get()
    });
    let msg = format!("incr: python code raised: {e}");
    PENDING_PYERR.with(|c| c.borrow_mut().push((token, e)));
    // Not `panic!`: the exception is re-raised as itself, so the panic
    // hook's message would only be noise on stderr.
    std::panic::resume_unwind(Box::new(PyStashPanic { token, msg }))
}

/// Whether Python code raising now runs in a doomed frame: the active
/// frame of the runtime whose engine call or query callback is innermost.
/// Another runtime's call can run Python code over a doomed frame without
/// pushing a frame of its own, and that code is not the doomed run's.
fn active_frame_doomed() -> bool {
    let rt = ACTIVE_RT.with(Cell::get);
    // SAFETY: non-null only while an `InRuntime` guard for it is alive,
    // inside a call that borrows that runtime throughout.
    !rt.is_null() && unsafe { &*rt }.active_frame_doomed()
}

/// Records `rt` as the active runtime, restoring the enclosing one when
/// the call returns or unwinds.
struct InRuntime(*const Runtime);

impl InRuntime {
    fn enter(rt: &Runtime) -> Self {
        InRuntime(ACTIVE_RT.with(|a| a.replace(rt as *const _)))
    }
}

impl Drop for InRuntime {
    fn drop(&mut self) {
        ACTIVE_RT.with(|a| a.set(self.0));
    }
}

fn take_stash(token: u64) -> Option<PyErr> {
    PENDING_PYERR.with(|c| {
        let mut stash = c.borrow_mut();
        let at = stash.iter().position(|(t, _)| *t == token)?;
        Some(stash.swap_remove(at).1)
    })
}

/// Marks a `pyerr_boundary` in progress; the outermost one drops every
/// exception still stashed when it exits, such as one an engine catch
/// swallowed, and the payload of an `_IncrEngineUnwind` Python swallowed.
struct Boundary;

impl Boundary {
    fn enter() -> Self {
        BOUNDARY_DEPTH.with(|d| d.set(d.get() + 1));
        Boundary
    }
}

impl Drop for Boundary {
    fn drop(&mut self) {
        let depth = BOUNDARY_DEPTH.with(|d| {
            d.set(d.get() - 1);
            d.get()
        });
        if depth == 0 {
            // Dropped after the borrow ends: a finalizer that runs may raise
            // into a callback that stashes again.
            let stale = PENDING_PYERR.with(|c| std::mem::take(&mut *c.borrow_mut()));
            drop(stale);
            let stale_unwind = ENGINE_UNWIND.with(|u| u.borrow_mut().take());
            drop(stale_unwind);
        }
    }
}

/// Run an engine entry point that may execute Python callbacks. A
/// stashed Python exception is re-raised as itself; other engine panics
/// become RuntimeError with the engine's message. Inside a query, an
/// engine unwind in flight on return becomes `_IncrEngineUnwind`.
fn pyerr_boundary<R>(f: impl FnOnce() -> R) -> PyResult<R> {
    // A call from Python code a carried cycle unwind runs, such as a
    // `finally`: it computes and raises as itself, and the carried unwind
    // goes on unless the call ended in an engine unwind of its own.
    let Some(aside) = engine::set_aside_cycle_unwind() else {
        return catch_into_python(f);
    };
    let carried = ENGINE_UNWIND.with(|u| u.borrow_mut().take());
    let result = catch_into_python(f);
    if aside.put_back() {
        ENGINE_UNWIND.with(|u| *u.borrow_mut() = carried);
    }
    result
}

fn catch_into_python<R>(f: impl FnOnce() -> R) -> PyResult<R> {
    engine::take_last_cycle();
    let _boundary = Boundary::enter();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    if engine::engine_unwind_in_progress() {
        // A call that returned, such as a Clean read in a `finally` the
        // unwind runs, keeps the payload of the unwind already carried.
        if let Err(payload) = result {
            ENGINE_UNWIND.with(|u| *u.borrow_mut() = Some(payload));
        }
        return Err(_IncrEngineUnwind::new_err(()));
    }
    let payload = match result {
        Ok(v) => return Ok(v),
        Err(payload) => payload,
    };
    // Taken on every error, so a cycle panic that Rust code caught
    // never labels a later error.
    let cycle = engine::take_last_cycle();
    if let Some(stashed) = payload.downcast_ref::<PyStashPanic>() {
        return Err(take_stash(stashed.token)
            .unwrap_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(stashed.msg.clone())));
    }
    if let Some(cycle) = cycle {
        return Err(cycle_error(&cycle));
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

fn cycle_ids(cycle: &Cycle) -> Vec<u32> {
    cycle.participants().iter().map(|p| p.slot()).collect()
}

fn cycle_error(cycle: &Cycle) -> PyErr {
    CycleError::new_err((format!("dependency cycle: {cycle}"), cycle_ids(cycle)))
}

/// A `try_get` result as Python sees it: the value, or `CycleError`
/// raised as an ordinary exception, never an engine unwind.
fn try_read(read: Result<PyValue, Cycle>) -> PyResult<Py<PyAny>> {
    read.map(|v| v.0).map_err(|cycle| cycle_error(&cycle))
}

/// Runs a `#[pymethods]` engine call on `rt` with the GIL released,
/// because the engine may wait on a claim whose holder needs the GIL to
/// finish: the GIL is lock level -1, never held while waiting on an incr
/// lock.
fn engine_call<R: Ungil>(
    py: Python<'_>,
    rt: &Runtime,
    f: impl FnOnce() -> R + Ungil,
) -> PyResult<R> {
    pyerr_boundary(|| {
        let _active = InRuntime::enter(rt);
        py.detach(f)
    })
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
        Python::attach(|py| {
            let result = or_stash(
                self.0
                    .bind(py)
                    .rich_compare(other.0.bind(py), CompareOp::Eq),
            );
            match result.is_truthy() {
                Ok(equal) => equal,
                // NumPy and pandas raise ValueError for an ambiguous truth value.
                Err(e) if e.is_instance_of::<pyo3::exceptions::PyValueError>(py) => false,
                Err(e) => or_stash(Err(e)),
            }
        })
    }
}

impl Eq for PyValue {}

impl Hash for PyValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Python::attach(|py| {
            let h: isize = or_stash(self.0.bind(py).hash());
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
            if or_stash(self_ref.lt(other_ref)) {
                std::cmp::Ordering::Less
            } else if or_stash(self_ref.eq(other_ref)) {
                std::cmp::Ordering::Equal
            } else {
                std::cmp::Ordering::Greater
            }
        })
    }
}

/// The closure the engine runs for a Python query: `py_func` called with
/// a `RuntimeRef` that is invalidated when it returns.
fn query_fn(py_func: Py<PyAny>) -> impl Fn(&Runtime) -> PyValue + Send + Sync + 'static {
    move |rt: &Runtime| -> PyValue {
        Python::attach(|py| {
            let rt_ref = Py::new(
                py,
                PyRuntimeRef {
                    ptr: Cell::new(rt as *const _),
                },
            )
            .unwrap();
            // Invalidates the ref when the callback returns or unwinds, so
            // one the callback kept never reaches a dropped runtime.
            let _guard = InvalidateRef(rt_ref.bind(py).clone());
            let _active = InRuntime::enter(rt);
            PyValue(call_stashing(py, &py_func, (rt_ref.clone_ref(py),)))
        })
    }
}

/// An input's initial value from a Python callable, called only on a miss.
fn init_fn(init: Py<PyAny>) -> impl FnOnce() -> PyValue {
    move || Python::attach(|py| PyValue(call_stashing(py, &init, ())))
}

/// The options of a query created with `fallback`, `fixpoint` and
/// `replay` keyword arguments.
fn query_opts(
    py: Python<'_>,
    fallback: Option<Py<PyAny>>,
    fixpoint: Option<PyRef<'_, PyFixpoint>>,
    replay: bool,
) -> PyResult<QueryOpts<PyValue>> {
    let cycle = match (fallback, fixpoint) {
        (Some(_), Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a query takes a fallback or a fixpoint, not both",
            ))
        }
        (Some(g), None) => CycleStrategy::fallback(move |cycle: &Cycle| {
            Python::attach(|py| PyValue(call_stashing(py, &g, (cycle_ids(cycle),))))
        }),
        (None, Some(fixpoint)) => CycleStrategy::Fixpoint(fixpoint.strategy(py)),
        (None, None) => CycleStrategy::None,
    };
    let replay = if replay {
        Replay::Allowed
    } else {
        Replay::Native
    };
    Ok(QueryOpts { cycle, replay })
}

/// A fixed-point cycle strategy: every member starts at `initial()`, and
/// iteration `k` gives it `step(previous, output, k)` (the output by
/// default) until every member has `converged(previous, current)` (equal
/// by default), or after `max_iterations`, where each member that has
/// not converged lands `on_limit(ids, current)` (the current value by
/// default).
#[pyclass(name = "Fixpoint", frozen)]
struct PyFixpoint {
    initial: Py<PyAny>,
    converged: Option<Py<PyAny>>,
    step: Option<Py<PyAny>>,
    max_iterations: u32,
    on_limit: Option<Py<PyAny>>,
}

#[pymethods]
impl PyFixpoint {
    #[new]
    #[pyo3(signature = (initial, *, converged=None, step=None, max_iterations=100, on_limit=None))]
    fn new(
        initial: Py<PyAny>,
        converged: Option<Py<PyAny>>,
        step: Option<Py<PyAny>>,
        max_iterations: u32,
        on_limit: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if max_iterations == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a fixpoint needs at least one iteration",
            ));
        }
        Ok(Self {
            initial,
            converged,
            step,
            max_iterations,
            on_limit,
        })
    }
}

impl PyFixpoint {
    fn strategy(&self, py: Python<'_>) -> Fixpoint<PyValue> {
        let initial = self.initial.clone_ref(py);
        let mut fixpoint =
            Fixpoint::new(move || Python::attach(|py| PyValue(call_stashing(py, &initial, ()))))
                .max_iterations(self.max_iterations);
        if let Some(f) = &self.converged {
            let f = f.clone_ref(py);
            fixpoint = fixpoint.converged(move |prev: &PyValue, cur: &PyValue| {
                Python::attach(|py| {
                    let args = (prev.0.clone_ref(py), cur.0.clone_ref(py));
                    or_stash(call_stashing(py, &f, args).is_truthy(py))
                })
            });
        }
        if let Some(f) = &self.step {
            let f = f.clone_ref(py);
            fixpoint = fixpoint.step(move |prev: &PyValue, out: PyValue, k: u32| {
                Python::attach(|py| {
                    PyValue(call_stashing(py, &f, (prev.0.clone_ref(py), out.0, k)))
                })
            });
        }
        if let Some(f) = &self.on_limit {
            let f = f.clone_ref(py);
            fixpoint = fixpoint.on_limit(move |cycle: &Cycle, cur: PyValue| {
                Python::attach(|py| PyValue(call_stashing(py, &f, (cycle_ids(cycle), cur.0))))
            });
        }
        fixpoint
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

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut h);
        h.finish()
    }
}

/// The query each collection's `version_node` getter hands out, keyed by
/// the collection's internal version node, so repeated access and every
/// wrapper of one collection share a single node. One per runtime,
/// shared by its collection wrappers.
#[derive(Default)]
struct VersionBridges(Mutex<HashMap<Incr<u64>, Incr<PyValue>>>);

impl VersionBridges {
    /// The Python-visible node mirroring `version`. Created on first use
    /// and again only if the user deleted the previous one, whose handle
    /// then stays stale like any deleted node's.
    fn get(&self, rt: &Runtime, version: Incr<u64>) -> PyNodeId {
        if let Some(bridge) = self.alive(rt, version) {
            return PyNodeId { inner: bridge };
        }
        // Created without the mutex, which is never held across an engine
        // call that may wait: a racing thread may create one too.
        let bridge = rt.create_query(move |rt| -> PyValue {
            let n: u64 = rt.get(version);
            Python::attach(|py| PyValue(n.into_pyobject(py).unwrap().into_any().unbind()))
        });
        let mut bridges = self.lock();
        if let Some(&existing) = bridges.get(&version) {
            if rt.is_alive(existing) {
                drop(bridges);
                rt.delete_node(bridge);
                return PyNodeId { inner: existing };
            }
        }
        bridges.insert(version, bridge);
        PyNodeId { inner: bridge }
    }

    fn alive(&self, rt: &Runtime, version: Incr<u64>) -> Option<Incr<PyValue>> {
        let bridge = self.lock().get(&version).copied()?;
        rt.is_alive(bridge).then_some(bridge)
    }

    /// Poison is ignored because the map is only ever inserted into whole.
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<Incr<u64>, Incr<PyValue>>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
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
    ptr: Cell<*const Runtime>,
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
}

// Every method releases the GIL, because a read may wait on a node
// another thread is computing, and that thread needs the GIL to finish.
// Each is a boundary too, so an engine unwind crosses the calling Python
// frames as an exception instead of a foreign panic.
#[pymethods]
impl PyRuntimeRef {
    fn get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        let rt = self.rt()?;
        let val: PyValue = engine_call(py, rt, || rt.get(node.inner))?;
        Ok(val.0)
    }

    /// Like `get`, but a cycle that leaves the node without a value
    /// raises `CycleError` as an ordinary exception the query may catch.
    fn try_get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        let rt = self.rt()?;
        try_read(engine_call(py, rt, || rt.try_get(node.inner))?)
    }

    /// The input child named `key` of this query, created with `init()`
    /// the first time this query asks for it. Children this query stops
    /// asking for are deleted when it next lands.
    fn child_input(&self, py: Python<'_>, key: Py<PyAny>, init: Py<PyAny>) -> PyResult<PyNodeId> {
        let rt = self.rt()?;
        let (key, init) = (PyValue(key), init_fn(init));
        let node = engine_call(py, rt, || rt.child_input(key, init))?;
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
        let rt = self.rt()?;
        let opts = query_opts(py, fallback, fixpoint, replay)?;
        let (key, f) = (PyValue(key), query_fn(f));
        let node = engine_call(py, rt, || rt.child_query_with(key, f, opts))?;
        Ok(PyNodeId { inner: node })
    }
}

/// A map from keys to nodes of one runtime. Key `__eq__` and `__hash__`
/// run as value code: one that calls into incr raises the contract error.
#[pyclass(name = "NodeTable")]
struct PyNodeTable {
    inner: NodeTable<PyValue, PyValue>,
    rt: Arc<Runtime>,
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
        let node = engine_call(py, &self.rt, || {
            self.inner.get_or_insert_input(&self.rt, key, init)
        })?;
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
        let opts = query_opts(py, fallback, fixpoint, replay)?;
        let (key, f) = (PyValue(key), query_fn(f));
        let node = engine_call(py, &self.rt, || {
            self.inner.get_or_insert_query_with(&self.rt, key, f, opts)
        })?;
        Ok(PyNodeId { inner: node })
    }

    /// The live node for `key`, or `None`. Not allowed inside a query.
    fn get(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Option<PyNodeId>> {
        let key = PyValue(key);
        let node = engine_call(py, &self.rt, || self.inner.get(&self.rt, &key))?;
        Ok(node.map(|inner| PyNodeId { inner }))
    }

    /// Deletes `key`'s node and removes the entry. Raises `KeyError` if
    /// the table holds no live node for it.
    fn remove(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<()> {
        let k = PyValue(key.clone_ref(py));
        match engine_call(py, &self.rt, || self.inner.remove(&self.rt, &k))? {
            Ok(()) => Ok(()),
            Err(DeleteError::NotAlive) => Err(pyo3::exceptions::PyKeyError::new_err(key)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// The `(key, node)` pairs whose node is alive. Not allowed inside a
    /// query.
    fn items(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, PyNodeId)>> {
        let entries = engine_call(py, &self.rt, || {
            let mut entries = Vec::new();
            self.inner
                .for_each(&self.rt, |key, node| entries.push((key.clone(), node)));
            entries
        })?;
        Ok(entries
            .into_iter()
            .map(|(key, inner)| (key.0, PyNodeId { inner }))
            .collect())
    }
}

#[pyclass(name = "Collection")]
struct PyCollection {
    inner: IncrCollection<PyValue>,
    rt: Arc<Runtime>,
    bridges: Arc<VersionBridges>,
}

#[pymethods]
impl PyCollection {
    fn insert(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let v = PyValue(value);
        engine_call(py, &self.rt, || self.inner.insert(&self.rt, v))
    }

    fn delete(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<bool> {
        let v = PyValue(value);
        engine_call(py, &self.rt, || self.inner.delete(&self.rt, &v))
    }

    fn snapshot_len(&self, py: Python<'_>) -> PyResult<usize> {
        engine_call(py, &self.rt, || self.inner.snapshot_len())
    }

    fn filter(&self, py: Python<'_>, predicate: Py<PyAny>) -> PyResult<PyCollection> {
        let filtered = engine_call(py, &self.rt, || {
            self.inner.filter(&self.rt, move |val: &PyValue| -> bool {
                Python::attach(|py| {
                    or_stash(call_stashing(py, &predicate, (val.0.clone_ref(py),)).is_truthy(py))
                })
            })
        })?;
        Ok(PyCollection {
            inner: filtered,
            rt: Arc::clone(&self.rt),
            bridges: Arc::clone(&self.bridges),
        })
    }

    fn map(&self, py: Python<'_>, func: Py<PyAny>) -> PyResult<PyCollection> {
        let mapped = engine_call(py, &self.rt, || {
            self.inner.map(&self.rt, move |val: &PyValue| -> PyValue {
                Python::attach(|py| PyValue(call_stashing(py, &func, (val.0.clone_ref(py),))))
            })
        })?;
        Ok(PyCollection {
            inner: mapped,
            rt: Arc::clone(&self.rt),
            bridges: Arc::clone(&self.bridges),
        })
    }

    fn count(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        let node = engine_call(py, &self.rt, || {
            let count_node: Incr<u64> = self.inner.count(&self.rt);
            self.rt.create_query(move |rt| -> PyValue {
                let c: u64 = rt.get(count_node);
                Python::attach(|py| PyValue(c.into_pyobject(py).unwrap().into_any().unbind()))
            })
        })?;
        Ok(PyNodeId { inner: node })
    }

    fn reduce(&self, py: Python<'_>, fold_fn: Py<PyAny>) -> PyResult<PyNodeId> {
        let node = engine_call(py, &self.rt, || {
            self.inner.reduce(&self.rt, move |elements| -> PyValue {
                Python::attach(|py| {
                    let py_list = pyo3::types::PyList::empty(py);
                    for elem in elements.iter() {
                        py_list.append(elem.0.clone_ref(py)).unwrap();
                    }
                    PyValue(call_stashing(py, &fold_fn, (py_list,)))
                })
            })
        })?;
        Ok(PyNodeId { inner: node })
    }

    /// Incrementally maintained monoid fold; O(log n) per change. See
    /// the Rust docs for the associativity contract.
    fn aggregate(
        &self,
        py: Python<'_>,
        identity: Py<PyAny>,
        lift: Py<PyAny>,
        combine: Py<PyAny>,
    ) -> PyResult<PyNodeId> {
        let node = engine_call(py, &self.rt, || {
            self.inner.aggregate(
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
            )
        })?;
        Ok(PyNodeId { inner: node })
    }

    fn sort_by_key(&self, py: Python<'_>, key_fn: Py<PyAny>) -> PyResult<PySortedCollection> {
        let sorted = engine_call(py, &self.rt, || {
            self.inner
                .sort_by_key(&self.rt, move |val: &PyValue| -> PyValue {
                    Python::attach(|py| PyValue(call_stashing(py, &key_fn, (val.0.clone_ref(py),))))
                })
        })?;
        Ok(PySortedCollection {
            inner: sorted,
            rt: Arc::clone(&self.rt),
            bridges: Arc::clone(&self.bridges),
        })
    }

    fn group_by(&self, py: Python<'_>, key_fn: Py<PyAny>) -> PyResult<PyGroupedCollection> {
        let grouped = engine_call(py, &self.rt, || {
            self.inner
                .group_by(&self.rt, move |val: &PyValue| -> PyValue {
                    Python::attach(|py| PyValue(call_stashing(py, &key_fn, (val.0.clone_ref(py),))))
                })
        })?;
        Ok(PyGroupedCollection {
            inner: grouped,
            rt: Arc::clone(&self.rt),
            bridges: Arc::clone(&self.bridges),
        })
    }

    fn join(
        &self,
        py: Python<'_>,
        right: &PyCollection,
        left_key: Py<PyAny>,
        right_key: Py<PyAny>,
    ) -> PyResult<PyCollection> {
        let mapped = engine_call(py, &self.rt, || {
            let joined = self.inner.join(
                &self.rt,
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
            joined.map(&self.rt, |pair: &(PyValue, PyValue)| -> PyValue {
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
            rt: Arc::clone(&self.rt),
            bridges: Arc::clone(&self.bridges),
        })
    }

    #[getter]
    fn version_node(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        engine_call(py, &self.rt, || {
            self.bridges.get(&self.rt, self.inner.version_node())
        })
    }
}

#[pyclass(name = "SortedCollection")]
struct PySortedCollection {
    inner: SortedCollection<PyValue, PyValue>,
    rt: Arc<Runtime>,
    bridges: Arc<VersionBridges>,
}

#[pymethods]
impl PySortedCollection {
    fn pairwise(&self, py: Python<'_>) -> PyResult<PyCollection> {
        let mapped = engine_call(py, &self.rt, || {
            let pair_collection = self.inner.pairwise(&self.rt);
            pair_collection.map(&self.rt, |pair: &(PyValue, PyValue)| -> PyValue {
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
            rt: Arc::clone(&self.rt),
            bridges: Arc::clone(&self.bridges),
        })
    }

    fn window(&self, py: Python<'_>, size: usize) -> PyResult<PyCollection> {
        let mapped = engine_call(py, &self.rt, || {
            let win_collection = self.inner.window(&self.rt, size);
            win_collection.map(&self.rt, |window: &Vec<PyValue>| -> PyValue {
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
            rt: Arc::clone(&self.rt),
            bridges: Arc::clone(&self.bridges),
        })
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let entries = engine_call(py, &self.rt, || self.inner.snapshot())?;
        let list = pyo3::types::PyList::empty(py);
        for entry in entries {
            list.append(entry.0.clone_ref(py))?;
        }
        Ok(list.into_any().unbind())
    }

    fn snapshot_len(&self, py: Python<'_>) -> PyResult<usize> {
        engine_call(py, &self.rt, || self.inner.snapshot_len())
    }

    #[getter]
    fn version_node(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        engine_call(py, &self.rt, || {
            self.bridges.get(&self.rt, self.inner.version_node())
        })
    }
}

#[pyclass(name = "GroupedCollection")]
struct PyGroupedCollection {
    inner: engine::GroupedCollection<PyValue, PyValue>,
    rt: Arc<Runtime>,
    bridges: Arc<VersionBridges>,
}

#[pymethods]
impl PyGroupedCollection {
    fn keys(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let keys = engine_call(py, &self.rt, || self.inner.keys())?;
        let list = pyo3::types::PyList::empty(py);
        for key in keys {
            list.append(key.0.clone_ref(py))?;
        }
        Ok(list.into_any().unbind())
    }

    fn get_group(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Option<PyCollection>> {
        let py_key = PyValue(key);
        let group = engine_call(py, &self.rt, || self.inner.get_group(&py_key))?;
        Ok(group.map(|collection| PyCollection {
            inner: collection,
            rt: Arc::clone(&self.rt),
            bridges: Arc::clone(&self.bridges),
        }))
    }

    fn group_count(&self, py: Python<'_>) -> PyResult<usize> {
        engine_call(py, &self.rt, || self.inner.group_count())
    }

    #[getter]
    fn version_node(&self, py: Python<'_>) -> PyResult<PyNodeId> {
        engine_call(py, &self.rt, || {
            self.bridges.get(&self.rt, self.inner.version_node())
        })
    }
}

#[pyclass(name = "Runtime")]
struct PyRuntime {
    inner: Arc<Runtime>,
    bridges: Arc<VersionBridges>,
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> Self {
        PyRuntime {
            inner: Arc::new(Runtime::new()),
            bridges: Arc::default(),
        }
    }

    fn create_input(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<PyNodeId> {
        let v = PyValue(value);
        let node = engine_call(py, &self.inner, || self.inner.create_input(v))?;
        Ok(PyNodeId { inner: node })
    }

    fn get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        let val: PyValue = engine_call(py, &self.inner, || self.inner.get(node.inner))?;
        Ok(val.0)
    }

    /// Like `get`, but a cycle that leaves the node without a value raises
    /// `CycleError` as an ordinary exception.
    fn try_get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        try_read(engine_call(py, &self.inner, || {
            self.inner.try_get(node.inner)
        })?)
    }

    fn set(&self, py: Python<'_>, node: PyNodeId, value: Py<PyAny>) -> PyResult<()> {
        let v = PyValue(value);
        engine_call(py, &self.inner, || self.inner.set(node.inner, v))
    }

    /// Delete a node and recycle its slot. Every surviving handle to it
    /// raises on use. The node must have no dependents.
    fn delete_node(&self, py: Python<'_>, node: PyNodeId) -> PyResult<()> {
        engine_call(py, &self.inner, || self.inner.delete_node(node.inner))
    }

    /// Delete the node, every node that depends on it (transitively)
    /// and every node an operator created for them. Returns how many
    /// nodes were deleted; 0 if the node was already gone. A query that
    /// has never run is not reached, and raises as stale on first use.
    fn delete_with_dependents(&self, py: Python<'_>, node: PyNodeId) -> PyResult<usize> {
        engine_call(py, &self.inner, || {
            self.inner.delete_with_dependents(node.inner)
        })
    }

    /// True while the node exists in this runtime. False after it is
    /// deleted, and for a node from another runtime.
    fn is_alive(&self, py: Python<'_>, node: PyNodeId) -> PyResult<bool> {
        engine_call(py, &self.inner, || self.inner.is_alive(node.inner))
    }

    /// Report the node through `drain_dirtied_watched()` whenever a change
    /// moves it off Clean. A node that has no current value when watched
    /// is reported at once.
    fn watch(&self, py: Python<'_>, node: PyNodeId) -> PyResult<()> {
        engine_call(py, &self.inner, || self.inner.watch(node.inner))
    }

    /// Stop reporting the node, and drop its pending report.
    fn unwatch(&self, py: Python<'_>, node: PyNodeId) -> PyResult<()> {
        engine_call(py, &self.inner, || self.inner.unwatch(node.inner))
    }

    /// The watched nodes a change moved off Clean since the last drain,
    /// each once, in the order the changes reached them. A node deleted
    /// since is left out.
    fn drain_dirtied_watched(&self, py: Python<'_>) -> PyResult<Vec<PyNodeId>> {
        engine_call(py, &self.inner, || {
            self.inner
                .drain_dirtied_watched()
                .into_iter()
                .filter_map(|h| self.inner.typed_handle::<PyValue>(h))
                .map(|inner| PyNodeId { inner })
                .collect()
        })
    }

    /// Register `callback` to fire on `stabilize()` whenever the node's
    /// value changed since the last firing.
    fn observe(
        &self,
        py: Python<'_>,
        node: PyNodeId,
        callback: Py<PyAny>,
    ) -> PyResult<PyObserverId> {
        let id = engine_call(py, &self.inner, || {
            self.inner.observe(node.inner, move |v: &PyValue| {
                Python::attach(|py| {
                    call_stashing(py, &callback, (v.0.clone_ref(py),));
                });
            })
        })?;
        Ok(PyObserverId { inner: id })
    }

    fn unobserve(&self, py: Python<'_>, id: PyObserverId) -> PyResult<()> {
        engine_call(py, &self.inner, || self.inner.unobserve(id.inner))
    }

    /// Bring observed nodes up to date and fire callbacks for changed
    /// values. Releases the GIL; callbacks reacquire it individually.
    fn stabilize(&self, py: Python<'_>) -> PyResult<()> {
        engine_call(py, &self.inner, || self.inner.stabilize())
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
        let opts = query_opts(py, fallback, fixpoint, replay)?;
        let f = query_fn(py_func);
        let node = engine_call(py, &self.inner, || self.inner.create_query_with(f, opts))?;
        Ok(PyNodeId { inner: node })
    }

    fn create_table(&self, py: Python<'_>) -> PyResult<PyNodeTable> {
        let table = engine_call(py, &self.inner, || {
            self.inner.create_table::<PyValue, PyValue>()
        })?;
        Ok(PyNodeTable {
            inner: table,
            rt: Arc::clone(&self.inner),
        })
    }

    /// The value a replayed read returns in place of the node it wanted.
    /// Set once per runtime.
    fn set_placeholder(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let v = PyValue(value);
        engine_call(py, &self.inner, || self.inner.set_placeholder(v))
    }

    /// The frame budget past which a replayable query's read replays.
    fn set_max_frames(&self, py: Python<'_>, n: u32) -> PyResult<()> {
        engine_call(py, &self.inner, || self.inner.set_max_frames(n))
    }

    /// A stack budget in bytes beside the frame budget; 0 turns it off.
    fn set_max_stack_bytes(&self, py: Python<'_>, n: usize) -> PyResult<()> {
        engine_call(py, &self.inner, || self.inner.set_max_stack_bytes(n))
    }

    fn create_collection(&self, py: Python<'_>) -> PyResult<PyCollection> {
        let col = engine_call(py, &self.inner, || {
            self.inner.create_collection::<PyValue>()
        })?;
        Ok(PyCollection {
            inner: col,
            rt: Arc::clone(&self.inner),
            bridges: Arc::clone(&self.bridges),
        })
    }

    fn set_label(&self, py: Python<'_>, node: PyNodeId, label: String) -> PyResult<()> {
        engine_call(py, &self.inner, || {
            self.inner.set_label(node.inner.slot(), label)
        })
    }

    fn set_label_by_id(&self, py: Python<'_>, id: u32, label: String) -> PyResult<()> {
        engine_call(py, &self.inner, || self.inner.set_label(id, label))
    }

    fn get_traced(&self, py: Python<'_>, node: PyNodeId) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (val, trace): (PyValue, PropagationTrace) =
            engine_call(py, &self.inner, || self.inner.get_traced(node.inner))?;
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

    fn graph_snapshot(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let infos = engine_call(py, &self.inner, || self.inner.graph_snapshot())?;
        {
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
        }
    }

    fn node_count(&self, py: Python<'_>) -> PyResult<usize> {
        engine_call(py, &self.inner, || self.inner.node_count())
    }
}

#[pymodule]
fn incr_concurrent(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CycleError", m.py().get_type::<CycleError>())?;
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyNodeId>()?;
    m.add_class::<PyObserverId>()?;
    m.add_class::<PyRuntimeRef>()?;
    m.add_class::<PyFixpoint>()?;
    m.add_class::<PyNodeTable>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PySortedCollection>()?;
    m.add_class::<PyGroupedCollection>()?;
    Ok(())
}
