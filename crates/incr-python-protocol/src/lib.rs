//! How incr's Python bindings carry Python exceptions and engine unwinds
//! across the engine, shared by `incr` (Local) and `incr_concurrent`
//! (Shared). Each binding builds its own copy, so the thread-local state
//! here is per extension module.
//!
//! A Python callback that raises inside an engine closure panics with the
//! `PyErr` in its payload; the engine's compute boundary marks the node
//! Failed and unwinds back to the API entry point, which re-raises the
//! original exception. Engine panics that are not Python
//! exceptions (cycles, stale handles, failed nodes) surface as
//! `RuntimeError` with the engine's message, or `CycleError` for a cycle.

use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, PoisonError};

use incr_core::{Cells, Cycle, CycleStrategy, Fixpoint, QueryOpts, Replay, Runtime};

/// The exception types of one binding's module.
pub trait Exceptions {
    /// `CycleError` with its `args`: the message and the participants'
    /// node ids.
    fn cycle_error(message: String, ids: Vec<u32>) -> PyErr;
    /// `_IncrEngineUnwind`, which carries an engine unwind through Python
    /// frames inside a query.
    fn engine_unwind() -> PyErr;
}

thread_local! {
    static BOUNDARY_DEPTH: Cell<u32> = const { Cell::new(0) };
    /// The runtime of the innermost engine call or query callback on this
    /// thread, with the check of its active frame, or `None` outside every
    /// one.
    static ACTIVE_RT: Cell<Option<ActiveRuntime>> = const { Cell::new(None) };
    /// An engine unwind a nested boundary caught, carried through the
    /// Python frames between it and the engine as `_IncrEngineUnwind`
    /// and resumed unchanged where that exception reaches the engine again.
    static ENGINE_UNWIND: RefCell<Option<Box<dyn std::any::Any + Send>>> = const { RefCell::new(None) };
}

#[derive(Clone, Copy)]
struct ActiveRuntime {
    rt: *const (),
    frame_doomed: unsafe fn(*const ()) -> bool,
}

/// Exceptions of `PyStashPanic`s the engine dropped unraised, such as one
/// an engine catch swallowed. Dropping a `PyErr` can run Python finalizers,
/// so they wait here for the next outermost boundary's exit. Global, as a
/// `Shared` runtime can drop a payload on a thread with no boundary.
static PARKED: Mutex<Vec<PyErr>> = Mutex::new(Vec::new());

/// The panic payload for a Python exception raised inside the engine. It
/// carries the exception itself, because the engine can hold the panic
/// past the call that raised it: a speculative run's stashed outcome is
/// raised by a later call's read.
struct PyStashPanic(Option<PyErr>);

impl Drop for PyStashPanic {
    fn drop(&mut self) {
        if let Some(err) = self.0.take() {
            PARKED
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(err);
        }
    }
}

/// Invoke a Python callable from inside an engine closure. A raised
/// exception is stashed for the API boundary to re-raise, then turned
/// into a Rust panic so the engine's compute boundary aborts this node
/// cleanly (Failed state, cursor unadvanced).
pub fn call_stashing<'py>(
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
pub fn or_stash<T>(r: PyResult<T>) -> T {
    r.unwrap_or_else(|e| raise_into_engine(e))
}

/// Unwinds the engine with `e`, a Python exception raised inside it.
pub fn raise_into_engine(e: PyErr) -> ! {
    // The run is discarded either way, so the exception is not kept.
    if incr_core::engine_unwind_in_progress() {
        drop(e);
        let payload = ENGINE_UNWIND.with(|u| u.borrow_mut().take());
        std::panic::resume_unwind(payload.unwrap_or_else(|| Box::new("incr: engine unwind")));
    }
    if active_frame_doomed() {
        drop(e);
        std::panic::resume_unwind(Box::new("incr: python code raised in a doomed frame"));
    }
    // Not `panic!`: the exception is re-raised as itself, so the panic
    // hook's message would only be noise on stderr.
    std::panic::resume_unwind(Box::new(PyStashPanic(Some(e))))
}

/// Whether Python code raising now runs in a doomed frame: the active
/// frame of the runtime whose engine call or query callback is innermost.
/// Another runtime's call can run Python code over a doomed frame without
/// pushing a frame of its own, and that code is not the doomed run's.
fn active_frame_doomed() -> bool {
    // SAFETY: set only inside `in_runtime`, which borrows `rt` throughout.
    ACTIVE_RT
        .with(Cell::get)
        .is_some_and(|active| unsafe { (active.frame_doomed)(active.rt) })
}

/// # Safety
/// `rt` points to a live `Runtime<C>`.
unsafe fn frame_doomed<C: Cells>(rt: *const ()) -> bool {
    unsafe { &*rt.cast::<Runtime<C>>() }.active_frame_doomed()
}

/// Runs `f` with `rt` as the active runtime, restoring the enclosing one
/// when `f` returns or unwinds.
pub fn in_runtime<C: Cells, R>(rt: &Runtime<C>, f: impl FnOnce() -> R) -> R {
    let active = ActiveRuntime {
        rt: (rt as *const Runtime<C>).cast(),
        frame_doomed: frame_doomed::<C>,
    };
    let _restore = RestoreActive(ACTIVE_RT.with(|a| a.replace(Some(active))));
    f()
}

struct RestoreActive(Option<ActiveRuntime>);

impl Drop for RestoreActive {
    fn drop(&mut self) {
        ACTIVE_RT.with(|a| a.set(self.0));
    }
}

/// Marks a `pyerr_boundary` in progress; the outermost one drops every
/// parked exception when it exits, and the payload of an
/// `_IncrEngineUnwind` Python swallowed.
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
            let parked =
                std::mem::take(&mut *PARKED.lock().unwrap_or_else(PoisonError::into_inner));
            drop(parked);
            let stale_unwind = ENGINE_UNWIND.with(|u| u.borrow_mut().take());
            drop(stale_unwind);
        }
    }
}

/// Run an engine entry point that may execute Python callbacks. A
/// stashed Python exception is re-raised as itself; other engine panics
/// become RuntimeError with the engine's message. Inside a query, an
/// engine unwind in flight on return becomes `_IncrEngineUnwind`.
pub fn pyerr_boundary<E: Exceptions, R>(f: impl FnOnce() -> R) -> PyResult<R> {
    // A call from Python code a carried cycle unwind runs, such as a
    // `finally`: it computes and raises as itself, and the carried unwind
    // goes on unless the call ended in an engine unwind of its own.
    let Some(aside) = incr_core::set_aside_cycle_unwind() else {
        return catch_into_python::<E, R>(f);
    };
    let carried = ENGINE_UNWIND.with(|u| u.borrow_mut().take());
    let result = catch_into_python::<E, R>(f);
    if aside.put_back() {
        ENGINE_UNWIND.with(|u| *u.borrow_mut() = carried);
    }
    result
}

fn catch_into_python<E: Exceptions, R>(f: impl FnOnce() -> R) -> PyResult<R> {
    incr_core::take_last_cycle();
    let _boundary = Boundary::enter();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    if incr_core::engine_unwind_in_progress() {
        // A call that returned, such as a Clean read in a `finally` the
        // unwind runs, keeps the payload of the unwind already carried.
        if let Err(payload) = result {
            ENGINE_UNWIND.with(|u| *u.borrow_mut() = Some(payload));
        }
        return Err(E::engine_unwind());
    }
    let payload = match result {
        Ok(v) => return Ok(v),
        Err(payload) => payload,
    };
    // Taken on every error, so a cycle panic that Rust code caught
    // never labels a later error.
    let cycle = incr_core::take_last_cycle();
    let payload = match payload.downcast::<PyStashPanic>() {
        Ok(mut stashed) => {
            return Err(stashed
                .0
                .take()
                .expect("incr: a stash payload is raised once"));
        }
        Err(payload) => payload,
    };
    if let Some(cycle) = cycle {
        return Err(cycle_error::<E>(&cycle));
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

pub fn cycle_ids(cycle: &Cycle) -> Vec<u32> {
    cycle.participants().iter().map(|p| p.slot()).collect()
}

fn cycle_error<E: Exceptions>(cycle: &Cycle) -> PyErr {
    E::cycle_error(format!("dependency cycle: {cycle}"), cycle_ids(cycle))
}

/// A `try_get` result as Python sees it: the value, or `CycleError`
/// raised as an ordinary exception, never an engine unwind.
pub fn try_read<E: Exceptions>(read: Result<PyValue, Cycle>) -> PyResult<Py<PyAny>> {
    read.map(|v| v.0).map_err(|cycle| cycle_error::<E>(&cycle))
}

/// Newtype around `Py<PyAny>` that satisfies the `Value` bound
/// (`Py<PyAny>` is `Send + Sync`; dereferencing requires the GIL). All
/// trait methods reacquire the GIL because Python objects are only
/// usable while holding it; this is the conventional PyO3 pattern for
/// embedding `Py<PyAny>` in trait-bounded Rust code.
pub struct PyValue(pub Py<PyAny>);

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

/// An input's initial value from a Python callable, called only on a miss.
pub fn init_fn(init: Py<PyAny>) -> impl FnOnce() -> PyValue {
    move || Python::attach(|py| PyValue(call_stashing(py, &init, ())))
}

/// The options of a query created with `fallback`, `fixpoint` and
/// `replay` keyword arguments.
pub fn query_opts(
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
pub struct PyFixpoint {
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

#[cfg(test)]
mod tests {
    use super::*;
    use incr_core::Local;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn active() -> Option<*const ()> {
        ACTIVE_RT.with(Cell::get).map(|a| a.rt)
    }

    fn addr(rt: &Runtime<Local>) -> Option<*const ()> {
        Some((rt as *const Runtime<Local>).cast())
    }

    #[test]
    fn nested_in_runtime_restores_the_outer_runtime() {
        let outer = Runtime::<Local>::new();
        let inner = Runtime::<Local>::new();
        in_runtime(&outer, || {
            assert_eq!(active(), addr(&outer));
            in_runtime(&inner, || assert_eq!(active(), addr(&inner)));
            assert_eq!(active(), addr(&outer));
        });
        assert_eq!(active(), None);
    }

    #[test]
    fn a_panicking_closure_restores_the_previous_runtime() {
        let outer = Runtime::<Local>::new();
        let inner = Runtime::<Local>::new();
        in_runtime(&outer, || {
            let unwound = catch_unwind(AssertUnwindSafe(|| {
                in_runtime(&inner, || std::panic::resume_unwind(Box::new("unwind")))
            }));
            assert!(unwound.is_err());
            assert_eq!(active(), addr(&outer));
        });
        assert_eq!(active(), None);
    }
}
