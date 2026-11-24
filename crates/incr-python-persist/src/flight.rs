//! `PendingCheckpoint`: a store or memo checkpoint begun with
//! `begin_checkpoint()` on a Local runtime, whose copying runs in `poll`
//! slices on the caller and whose encode and write run on a worker thread.
//!
//! The handle holds only the flight's token and a reference to its store
//! or memo, so it may be freed on any thread, the collector's included:
//! freeing it abandons the flight, which the store or memo settles itself.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::thread::{self, ThreadId};
use std::time::Duration;

use incr_core::Local;
use incr_persist::memo::{CheckpointStats, MemoCheckpoint};
use incr_persist::StoreCheckpoint;
use incr_python_protocol::gil;
use pyo3::exceptions::{PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use pyo3::PyClass;

use crate::binding::{CoreStore, PyBinding};
use crate::memo::{checkpoint_dict, MemoCore};
use crate::store::StoreCore;

/// A store class whose instances begin store flights.
pub trait StoreOwner<B: PyBinding>: PyClass {
    /// Runs `f` with the store's core: one usable for a new checkpoint
    /// when `begin`, else one that is merely open.
    fn with_store<R>(
        &self,
        py: Python<'_>,
        begin: bool,
        f: impl FnOnce(&Arc<StoreCore<B>>) -> PyResult<R>,
    ) -> PyResult<R>;
}

/// A memo class whose instances begin memo flights.
pub trait MemoOwner<B: PyBinding>: PyClass {
    /// [`StoreOwner::with_store`] with the memo's core too.
    fn with_memo<R>(
        &self,
        py: Python<'_>,
        begin: bool,
        f: impl FnOnce(&StoreCore<B>, &MemoCore<B>) -> PyResult<R>,
    ) -> PyResult<R>;
}

trait Pending: Send + Sync {
    fn poll(&self, py: Python<'_>, budget: Duration) -> PyResult<bool>;
    fn result(&self, py: Python<'_>, timeout: Option<Duration>) -> PyResult<Option<Py<PyAny>>>;
    fn cancel(&self);
}

/// A checkpoint in flight. `poll(budget_seconds=0.05)` copies for about
/// the budget on this thread and returns `True` once the checkpoint is
/// committed and finished, raising what failed it (`PersistError` once it
/// was cancelled); `result(timeout=None)` drives it to the end, waiting for
/// the worker with the GIL released, and returns what `checkpoint()`
/// returns, raising `TimeoutError` with the flight still open when the
/// timeout passes; `cancel()` stops a flight that has not committed.
/// `poll`, `result`, `cancel` and leaving a `with` block raise
/// `RuntimeError` on any thread but the one that began the flight. Leaving
/// a `with` block, or freeing the handle, abandons the flight: the store
/// or memo ends it on its own once the worker has stopped, and until then
/// `checkpoint()` raises `Busy`.
#[pyclass(name = "PendingCheckpoint", module = "incr.persist", frozen)]
pub struct PendingCheckpoint {
    opener: ThreadId,
    flight: Mutex<Option<Box<dyn Pending>>>,
}

impl PendingCheckpoint {
    fn new(flight: Box<dyn Pending>) -> Self {
        PendingCheckpoint {
            opener: thread::current().id(),
            flight: Mutex::new(Some(flight)),
        }
    }

    /// The flight's slot, for the opener thread only and never waiting:
    /// a call holding it may be waiting for the worker with the GIL
    /// released, and the worker needs the GIL to finish.
    fn slot(&self) -> PyResult<MutexGuard<'_, Option<Box<dyn Pending>>>> {
        if thread::current().id() != self.opener {
            return Err(PyRuntimeError::new_err(
                "a pending checkpoint is used only on the thread that began it",
            ));
        }
        match self.flight.try_lock() {
            Ok(flight) => Ok(flight),
            Err(TryLockError::Poisoned(poisoned)) => Ok(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => Err(PyRuntimeError::new_err(
                "the pending checkpoint is in use by another call",
            )),
        }
    }

    fn with_flight<R>(&self, f: impl FnOnce(&dyn Pending) -> PyResult<R>) -> PyResult<R> {
        let flight = self.slot()?;
        match flight.as_deref() {
            Some(flight) => f(flight),
            None => Err(PyRuntimeError::new_err(
                "the pending checkpoint was abandoned when its with block ended",
            )),
        }
    }
}

#[pymethods]
impl PendingCheckpoint {
    #[pyo3(signature = (budget_seconds=0.05))]
    fn poll(&self, py: Python<'_>, budget_seconds: f64) -> PyResult<bool> {
        let budget = seconds("budget_seconds", budget_seconds)?;
        self.with_flight(|flight| flight.poll(py, budget))
    }

    #[pyo3(signature = (timeout=None))]
    fn result(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<Py<PyAny>> {
        let timeout = timeout.map(|t| seconds("timeout", t)).transpose()?;
        self.with_flight(|flight| {
            flight.result(py, timeout)?.ok_or_else(|| {
                PyTimeoutError::new_err("the checkpoint did not finish within the timeout")
            })
        })
    }

    fn cancel(&self) -> PyResult<()> {
        self.with_flight(|flight| {
            flight.cancel();
            Ok(())
        })
    }

    fn __enter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_exc))]
    fn __exit__(&self, _exc: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<bool> {
        let flight = self.slot()?.take();
        drop(flight);
        Ok(false)
    }
}

fn seconds(name: &str, value: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(value).map_err(|_| {
        PyValueError::new_err(format!("{name} must be a non-negative number of seconds"))
    })
}

/// Begins a store flight on `owner`'s store.
pub fn begin_store<B, S>(owner: &Bound<'_, S>) -> PyResult<PendingCheckpoint>
where
    B: PyBinding<C = Local>,
    S: StoreOwner<B>,
{
    let py = owner.py();
    let handle = owner.borrow().with_store(py, true, |core| {
        store_call(core, py, None, |store| store.begin_checkpoint())
    })?;
    Ok(PendingCheckpoint::new(Box::new(StoreFlight::<B, S> {
        owner: owner.clone().unbind(),
        handle,
        binding: PhantomData,
    })))
}

/// Begins a memo flight on `owner`'s memo.
pub fn begin_memo<B, S>(owner: &Bound<'_, S>) -> PyResult<PendingCheckpoint>
where
    B: PyBinding<C = Local>,
    S: MemoOwner<B>,
    for<'a> B::Task<&'a crate::memo::OpenMemo<B>>: Send,
{
    let py = owner.py();
    let handle = owner.borrow().with_memo(py, true, |core, memo| {
        memo.run_memo_call(core, py, None, |memo| memo.begin_checkpoint(), |_| false)
    })?;
    Ok(PendingCheckpoint::new(Box::new(MemoFlight::<B, S> {
        owner: owner.clone().unbind(),
        handle,
        binding: PhantomData,
    })))
}

/// Runs a flight call on the store as a store call, raising its error, or
/// the exception a codec raised on the flight's `worker` when it failed
/// there.
fn store_call<B: PyBinding, R: Send>(
    core: &StoreCore<B>,
    py: Python<'_>,
    worker: Option<ThreadId>,
    f: impl FnOnce(&CoreStore<B>) -> incr_persist::Result<R> + Send,
) -> PyResult<R> {
    let registry = core.registry();
    drop(registry.take_failure());
    let result = B::write(py, core.store(), move |store, _| {
        registry.scoped(|| f(store))
    })?;
    if result.is_err() {
        if let Some(raised) = worker.and_then(|worker| registry.take_failure_of(worker)) {
            drop(registry.take_failure());
            return Err(raised);
        }
    }
    registry.finish(py, result)
}

struct StoreFlight<B, S> {
    owner: Py<S>,
    handle: StoreCheckpoint,
    binding: PhantomData<fn() -> B>,
}

impl<B, S> Pending for StoreFlight<B, S>
where
    B: PyBinding<C = Local>,
    S: StoreOwner<B>,
{
    fn poll(&self, py: Python<'_>, budget: Duration) -> PyResult<bool> {
        let handle = &self.handle;
        self.owner.bind(py).borrow().with_store(py, false, |core| {
            store_call(core, py, handle.worker(), |store| {
                handle.poll(store, budget)
            })
        })
    }

    fn result(&self, py: Python<'_>, timeout: Option<Duration>) -> PyResult<Option<Py<PyAny>>> {
        let handle = &self.handle;
        let upto = self.owner.bind(py).borrow().with_store(py, false, |core| {
            store_call(core, py, handle.worker(), |store| {
                handle.result(store, timeout)
            })
        })?;
        upto.map(|upto| upto.0.into_py_any(py)).transpose()
    }

    fn cancel(&self) {
        self.handle.cancel();
    }
}

struct MemoFlight<B, S> {
    owner: Py<S>,
    handle: MemoCheckpoint,
    binding: PhantomData<fn() -> B>,
}

impl<B, S> Pending for MemoFlight<B, S>
where
    B: PyBinding<C = Local>,
    S: MemoOwner<B>,
    for<'a> B::Task<&'a crate::memo::OpenMemo<B>>: Send,
{
    fn poll(&self, py: Python<'_>, budget: Duration) -> PyResult<bool> {
        let handle = &self.handle;
        self.owner
            .bind(py)
            .borrow()
            .with_memo(py, false, |core, memo| {
                memo.run_memo_call(
                    core,
                    py,
                    handle.worker(),
                    |memo| handle.poll(memo, budget),
                    |done| *done,
                )
            })
    }

    fn result(&self, py: Python<'_>, timeout: Option<Duration>) -> PyResult<Option<Py<PyAny>>> {
        let handle = &self.handle;
        let stats: Option<CheckpointStats> =
            self.owner
                .bind(py)
                .borrow()
                .with_memo(py, false, |core, memo| {
                    memo.run_memo_call(
                        core,
                        py,
                        handle.worker(),
                        |memo| handle.result(memo, timeout),
                        Option::is_some,
                    )
                })?;
        stats
            .map(|stats| checkpoint_dict(py, &stats).map(|dict| dict.into_any().unbind()))
            .transpose()
    }

    fn cancel(&self) {
        self.handle.cancel();
    }
}

/// Cancels every checkpoint flight still running and waits, with the GIL
/// released, for each worker thread to end. Registered with `atexit`.
#[pyfunction]
pub fn _join_checkpoint_workers(py: Python<'_>) {
    gil::release(py, incr_persist::join_checkpoint_workers);
}

/// Sets the longest memo frame a checkpoint writes, `None` for the format's
/// own limit, so a test can make an entry oversized. Process-wide.
#[cfg(feature = "gil-checks")]
#[pyfunction]
#[pyo3(signature = (limit=None))]
pub fn _set_memo_frame_limit(limit: Option<usize>) {
    incr_persist::memo::set_frame_limit(limit);
}
