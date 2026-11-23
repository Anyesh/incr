//! The yield hook of a runtime whose engine calls keep the GIL: once the
//! interval has passed since its last hand-over, it releases the GIL and
//! takes it back. CPython makes a thread that drops the GIL on another
//! thread's switch request wait until that thread has taken it, so a
//! waiting thread runs before the engine goes on.
//!
//! A waiting thread requests the GIL once a timed wait of one switch
//! interval runs out with no switch, and each release signals that wait,
//! starting it again. A release that comes before the request so hands
//! nothing over and pushes the request back by a switch interval, so the
//! hook releases a second time [`RETRY_AFTER`] later.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use incr_core::YieldHook;
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::PyResult;

/// The default for `gil_yield_ms`.
pub const DEFAULT_GIL_YIELD_MS: u64 = 100;

/// Twice CPython's default switch interval (5 ms) and 1 ms more.
const RETRY_AFTER: Duration = Duration::from_millis(11);

/// Raises `ValueError` for `0`: a release at every check would restart a
/// waiting thread's timed wait every few microseconds, so it would never
/// request the GIL.
pub fn check_gil_yield_ms(gil_yield_ms: Option<u64>) -> PyResult<()> {
    match gil_yield_ms {
        Some(0) => Err(PyValueError::new_err(
            "gil_yield_ms must be at least 1, or None to never hand the GIL over",
        )),
        _ => Ok(()),
    }
}

/// The hook for `gil_yield_ms`, `None` for none; see
/// [`check_gil_yield_ms`]. Only for a runtime whose engine runs on a
/// thread holding the GIL whenever it calls the hook, as a `Local` runtime
/// does: it is not `Send`, so no closure that releases the GIL can reach
/// it.
pub fn gil_yield_hook(gil_yield_ms: Option<u64>) -> PyResult<Option<YieldHook>> {
    check_gil_yield_ms(gil_yield_ms)?;
    let Some(gil_yield_ms) = gil_yield_ms else {
        return Ok(None);
    };
    let nanos = |d: Duration| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
    let interval = nanos(Duration::from_millis(gil_yield_ms));
    let retry = Some(nanos(RETRY_AFTER)).filter(|&retry| retry < interval);
    let start = Instant::now();
    let next = AtomicU64::new(interval);
    let retrying = AtomicBool::new(false);
    let since_start = move || nanos(start.elapsed());
    Ok(Some(YieldHook::new(move || {
        if since_start() < next.load(Ordering::Relaxed) {
            return;
        }
        // SAFETY: this thread holds the GIL (see above), and the pair
        // touches no Python object in between, so PyO3's count of the
        // GIL, which it leaves unchanged, is right again once it returns.
        // Unlike `Python::detach`, it runs none of PyO3's deferred
        // decrefs, so no finalizer runs at the yield.
        incr_core::with_value_code(|| unsafe {
            let state = ffi::PyEval_SaveThread();
            ffi::PyEval_RestoreThread(state);
        });
        let now = since_start();
        let wait = match retry {
            Some(retry) if !retrying.swap(true, Ordering::Relaxed) => retry,
            Some(retry) => {
                retrying.store(false, Ordering::Relaxed);
                interval - retry
            }
            None => interval,
        };
        next.store(now.saturating_add(wait), Ordering::Relaxed);
    })))
}
