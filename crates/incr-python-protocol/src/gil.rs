//! The one place that takes and gives up the GIL in the code a checkpoint
//! worker can run (`incr-python-persist`, the codec glue in `persist.rs`
//! and `incr-python`'s persist module); `test_gil_gate_scan.py` refuses any
//! other. A worker runs its whole body released and enters Python only
//! through [`enter`], which times every stretch it holds the GIL and hands
//! the GIL over once a stretch reaches [`STRETCH`]. On any other thread the
//! gate is `Python::attach` and `Python::detach`.
//!
//! Every entry names a static [`Site`] that says whether a worker may reach
//! it and how much one call carries. With the `gil-checks` feature a worker
//! that breaks a site's declaration panics, and so does a value type's own
//! attach on a worker that is released, and a worker that drops more than
//! [`MAX_RELEASED_DROPS`] values while released.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use pyo3::marker::Ungil;
use pyo3::prelude::*;
use pyo3::types::{PyFrozenSet, PyList, PyTuple};

/// The shortest stretch off the GIL that lets a waiting thread take it.
/// A bare release and reacquire hands nothing over.
pub const GAP: Duration = Duration::from_micros(50);
/// How long a worker holds the GIL, between ticks, before its next tick or
/// entry first spends a [`GAP`] off it.
pub const STRETCH: Duration = Duration::from_millis(2);
/// The values a worker may drop while released before the gate fails it:
/// each such drop queues a decref that the next attach of any thread
/// applies, so a container of them freed released lands on the caller.
pub const MAX_RELEASED_DROPS: u64 = 64;
/// Elements a sequence function converts between frees of a list slice.
const SLICE: usize = 64;
/// Stretches at least this long are kept for the probe's report.
const LOGGED_HOLD: Duration = Duration::from_millis(1);

/// What one entry through a site carries, declared where the site is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Not reached on a worker; reaching it there is a bug.
    Caller,
    /// One call or one value.
    One,
    /// A sequence of at most this many elements per call.
    Bounded(usize),
    /// A sequence of any length, which the probe's scale run must cover.
    Unbounded,
}

/// A named entry into Python. Declare each as a `static` with one of the
/// constructors and a name unique across the scanned code.
pub struct Site {
    name: &'static str,
    kind: Kind,
    entered: AtomicU64,
    most: AtomicU64,
    longest: AtomicU64,
    listed: AtomicBool,
}

impl Site {
    const fn new(name: &'static str, kind: Kind) -> Self {
        Site {
            name,
            kind,
            entered: AtomicU64::new(0),
            most: AtomicU64::new(0),
            longest: AtomicU64::new(0),
            listed: AtomicBool::new(false),
        }
    }

    pub const fn caller(name: &'static str) -> Self {
        Site::new(name, Kind::Caller)
    }

    pub const fn one(name: &'static str) -> Self {
        Site::new(name, Kind::One)
    }

    pub const fn bounded(name: &'static str, bound: usize) -> Self {
        Site::new(name, Kind::Bounded(bound))
    }

    pub const fn unbounded(name: &'static str) -> Self {
        Site::new(name, Kind::Unbounded)
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    fn list(&'static self) {
        if !self.listed.swap(true, Ordering::AcqRel) {
            SITES
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(self);
        }
    }

    fn note_entry(&'static self) {
        self.list();
        self.entered.fetch_add(1, Ordering::Relaxed);
        if cfg!(feature = "gil-checks") && self.kind == Kind::Caller {
            panic!(
                "gil-checks: a checkpoint worker entered Python through '{}', which is not \
                 declared for workers",
                self.name
            );
        }
    }

    fn note_elements(&'static self, count: usize) {
        if !on_worker() {
            return;
        }
        self.list();
        self.most.fetch_max(count as u64, Ordering::Relaxed);
        if cfg!(feature = "gil-checks") {
            if let Kind::Bounded(bound) = self.kind {
                assert!(
                    count <= bound,
                    "gil-checks: '{}' carried {count} elements on a worker, over its bound of \
                     {bound}",
                    self.name
                );
            }
        }
    }

    fn note_hold(&'static self, held: Duration, ended: Instant) {
        let nanos = nanos(held);
        self.longest.fetch_max(nanos, Ordering::Relaxed);
        if held >= LOGGED_HOLD {
            let end = since_base(ended);
            HOLDS.lock().unwrap_or_else(PoisonError::into_inner).push((
                end.saturating_sub(nanos),
                end,
                self.name,
            ));
        }
    }
}

static SITES: Mutex<Vec<&'static Site>> = Mutex::new(Vec::new());
static HOLDS: Mutex<Vec<(u64, u64, &'static str)>> = Mutex::new(Vec::new());
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// A worker's own gate state. `depth` counts the gate entries open on the
/// GIL this thread holds; zero means the worker is released.
struct Worker {
    on: Cell<bool>,
    depth: Cell<u32>,
    holder: Cell<Option<&'static Site>>,
    acquired: Cell<Option<Instant>>,
    released: Cell<Option<Instant>>,
    gap: Cell<Option<Instant>>,
    drops: Cell<u64>,
}

thread_local! {
    static WORKER: Worker = const {
        Worker {
            on: Cell::new(false),
            depth: Cell::new(0),
            holder: Cell::new(None),
            acquired: Cell::new(None),
            released: Cell::new(None),
            gap: Cell::new(None),
            drops: Cell::new(0),
        }
    };
}

/// Whether this thread is a checkpoint worker inside [`run_worker`].
pub fn on_worker() -> bool {
    WORKER.try_with(|w| w.on.get()).unwrap_or(false)
}

/// Runs a checkpoint worker's whole body on this thread: takes one Python
/// thread state for the worker's life, so a backend's `threading.local`
/// lasts as long, and runs `body` with the GIL released.
///
/// # Safety
///
/// As for [`idle`]: `body` reaches Python only through the gate.
pub unsafe fn run_worker(body: &mut dyn FnMut()) {
    Python::attach(|py| {
        let _worker = WorkerMark::start();
        detach(py, body);
    });
}

struct WorkerMark;

impl WorkerMark {
    fn start() -> Self {
        LIVE.fetch_add(1, Ordering::AcqRel);
        WORKER.with(|w| {
            w.on.set(true);
            w.depth.set(0);
            w.holder.set(None);
            w.acquired.set(None);
            w.released.set(Some(Instant::now()));
            w.gap.set(None);
            w.drops.set(0);
        });
        WorkerMark
    }
}

impl Drop for WorkerMark {
    fn drop(&mut self) {
        let drops = WORKER.with(|w| {
            w.on.set(false);
            w.drops.get()
        });
        LIVE.fetch_sub(1, Ordering::AcqRel);
        if !thread::panicking() {
            check_drops(drops);
        }
    }
}

fn check_drops(drops: u64) {
    if cfg!(feature = "gil-checks") && drops > MAX_RELEASED_DROPS {
        panic!(
            "gil-checks: a checkpoint worker dropped {drops} Python values while released, \
             over {MAX_RELEASED_DROPS}: values must leave a worker inside `hold`"
        );
    }
}

/// Runs `f` holding the GIL. On a worker, an outermost entry first spends
/// a [`GAP`] off the GIL when the last one was [`STRETCH`] ago, and times
/// its hold; a nested entry is a [`tick`].
pub fn enter<R>(site: &'static Site, f: impl for<'py> FnOnce(Python<'py>) -> R) -> R {
    if !on_worker() {
        return Python::attach(f);
    }
    site.note_entry();
    if WORKER.with(|w| w.depth.get()) > 0 {
        return Python::attach(|py| {
            tick(py);
            f(py)
        });
    }
    if !thread::panicking() {
        WORKER.with(|w| check_drops(w.drops.get()));
    }
    space_from_last_gap();
    Python::attach(|py| {
        let _held = Held::begin(site);
        f(py)
    })
}

/// The open stretch of an outermost entry on a worker.
struct Held;

impl Held {
    fn begin(site: &'static Site) -> Self {
        WORKER.with(|w| {
            w.depth.set(1);
            w.holder.set(Some(site));
            begin_stretch(w, Instant::now());
        });
        Held
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        WORKER.with(|w| {
            end_stretch(w, Instant::now());
            w.depth.set(0);
            w.holder.set(None);
        });
    }
}

fn end_stretch(w: &Worker, now: Instant) {
    if let (Some(site), Some(acquired)) = (w.holder.get(), w.acquired.take()) {
        site.note_hold(now.saturating_duration_since(acquired), now);
    }
    w.released.set(Some(now));
}

fn begin_stretch(w: &Worker, now: Instant) {
    if w.released
        .get()
        .is_some_and(|released| now.saturating_duration_since(released) >= GAP)
    {
        w.gap.set(Some(now));
    }
    w.acquired.set(Some(now));
}

/// Before an outermost acquire: a worker that has held the GIL for
/// [`STRETCH`] since its last gap and released it less than [`GAP`] ago
/// waits out the rest of a gap first.
fn space_from_last_gap() {
    WORKER.with(|w| {
        let now = Instant::now();
        let off = w
            .released
            .get()
            .map_or(GAP, |released| now.saturating_duration_since(released));
        if off >= GAP {
            return;
        }
        let since_gap = w
            .gap
            .get()
            .map_or(STRETCH, |gap| now.saturating_duration_since(gap));
        if since_gap >= STRETCH {
            thread::sleep(GAP - off);
        }
    });
}

/// A point inside an entry where a worker may hand the GIL over: once
/// [`STRETCH`] has passed since its last gap, it releases for a [`GAP`].
/// On any other thread it ticks the runtime's ambient yield hook.
pub fn tick(py: Python<'_>) {
    if !on_worker() {
        incr_core::Yielder::ambient().tick();
        return;
    }
    let due = WORKER.with(|w| {
        w.depth.get() > 0
            && w.gap.get().is_none_or(|gap| gap.elapsed() >= STRETCH)
            && w.acquired.get().is_some()
    });
    if due {
        released(py, || {
            let until = Instant::now() + GAP;
            while let Some(left) = until.checked_duration_since(Instant::now()) {
                thread::sleep(left);
            }
        });
    }
}

/// Runs `f`, incr's own Rust reached from a Python frame, released on a
/// worker and in place on any other thread. An [`enter`] inside it
/// acquires for real and releases on its return.
pub fn leave<R: Ungil>(py: Python<'_>, f: impl Ungil + FnOnce() -> R) -> R {
    if on_worker() {
        released(py, f)
    } else {
        f()
    }
}

/// [`leave`] for a caller with no `Python` token: on a worker holding the
/// GIL through the gate it releases around `f`, and otherwise runs `f` in
/// place.
///
/// # Safety
///
/// `f` must not use a Python object or token it captured: PyO3 cannot
/// check a `dyn` closure for that.
pub unsafe fn idle(f: &mut dyn FnMut()) {
    if on_worker() && WORKER.with(|w| w.depth.get()) > 0 {
        Python::attach(|py| released(py, f));
    } else {
        f();
    }
}

/// Runs `f`, a caller's wait for a worker, with the GIL released on any
/// thread. `f` must not touch Python.
pub fn release<R: Ungil>(py: Python<'_>, f: impl Ungil + FnOnce() -> R) -> R {
    released(py, f)
}

/// [`release`] for a caller that holds the GIL but has no token.
///
/// # Safety
///
/// As for [`idle`].
pub unsafe fn release_held(f: &mut dyn FnMut()) {
    Python::attach(|py| released(py, f));
}

fn released<R>(py: Python<'_>, f: impl FnOnce() -> R) -> R {
    if !on_worker() {
        return detach(py, f);
    }
    let saved = WORKER.with(|w| {
        let saved = (w.depth.get(), w.holder.get());
        end_stretch(w, Instant::now());
        w.depth.set(0);
        w.holder.set(None);
        saved
    });
    let _restore = Restore(saved);
    detach(py, f)
}

struct Restore((u32, Option<&'static Site>));

impl Drop for Restore {
    fn drop(&mut self) {
        let (depth, holder) = self.0;
        WORKER.with(|w| {
            w.depth.set(depth);
            w.holder.set(holder);
            if depth > 0 {
                begin_stretch(w, Instant::now());
            }
        });
    }
}

/// `py.detach(f)` for an `f` that PyO3 cannot prove free of Python
/// references: the gate's callers pass incr's own Rust and waits, which
/// touch no Python object until they enter again.
fn detach<R>(py: Python<'_>, f: impl FnOnce() -> R) -> R {
    struct Unchecked<T>(T);
    // SAFETY: see above; nothing in `f` uses a `Python` token or a borrowed
    // Python object while the GIL is released, and every attach inside it
    // goes through PyO3, which knows this thread is detached.
    unsafe impl<T> Send for Unchecked<T> {}
    impl<T> Unchecked<T> {
        fn into_inner(self) -> T {
            self.0
        }
    }
    let f = Unchecked(f);
    py.detach(move || Unchecked(f.into_inner()())).into_inner()
}

/// For a value type's own attach: with `gil-checks`, panics on a worker
/// that is released, where the attach would take the GIL past the gate.
pub fn check_value_attach() {
    if cfg!(feature = "gil-checks") && on_worker() && WORKER.with(|w| w.depth.get()) == 0 {
        panic!(
            "gil-checks: a Python value was cloned, compared or hashed on a checkpoint worker \
             outside the gate"
        );
    }
}

/// For a value type's `Drop`: with `gil-checks`, counts the drops a worker
/// makes while released.
pub fn note_value_drop() {
    if cfg!(feature = "gil-checks") {
        let _ = WORKER.try_with(|w| {
            if w.on.get() && w.depth.get() == 0 {
                w.drops.set(w.drops.get() + 1);
            }
        });
    }
}

/// Builds a new `list` from `items` through `convert`, ticking per element.
pub fn build<'py, T, X>(
    py: Python<'py>,
    site: &'static Site,
    items: impl IntoIterator<Item = T>,
    error: impl Fn(PyErr) -> X,
    mut convert: impl FnMut(Python<'py>, T) -> Result<Bound<'py, PyAny>, X>,
) -> Result<Bound<'py, PyAny>, X> {
    let list = PyList::empty(py);
    for item in items {
        tick(py);
        list.append(convert(py, item)?).map_err(&error)?;
    }
    site.note_elements(list.len());
    Ok(list.into_any())
}

/// Converts a sequence a backend returned, ticking per element. An exact
/// `list` incr holds the only reference to is converted from its tail and
/// freed a slice at a time; another `list` or a `tuple` is first copied
/// into a new list, so a tick never lets another thread change what is
/// being walked; any other iterable is walked through [`pull`].
pub fn drain<'py, T: Send, X>(
    site: &'static Site,
    sequence: Bound<'py, PyAny>,
    error: impl Fn(PyErr) -> X,
    mut convert: impl FnMut(&Bound<'py, PyAny>) -> Result<T, X>,
) -> Result<Vec<T>, X> {
    let list = if sequence.is_exact_instance_of::<PyList>() && sequence.get_refcnt() == 1 {
        sequence
            .downcast_into::<PyList>()
            .map_err(|e| error(e.into()))?
    } else if sequence.is_instance_of::<PyList>() || sequence.is_instance_of::<PyTuple>() {
        sequence
            .downcast::<pyo3::types::PySequence>()
            .map_err(|e| error(e.into()))?
            .to_list()
            .map_err(&error)?
    } else {
        return pull(site, &sequence, error, convert);
    };
    let py = list.py();
    let len = list.len();
    site.note_elements(len);
    let mut out = Vec::with_capacity(len);
    let mut walk = || {
        let mut end = len;
        while end > 0 {
            let start = end.saturating_sub(SLICE);
            for at in (start..end).rev() {
                tick(py);
                let item = list.get_item(at).map_err(&error)?;
                out.push(convert(&item)?);
            }
            list.del_slice(start, end).map_err(&error)?;
            end = start;
        }
        Ok(())
    };
    match walk() {
        Ok(()) => Ok(leave(py, move || {
            out.reverse();
            out
        })),
        Err(failed) => Err(discarded(py, out, failed)),
    }
}

/// Walks a Python iterable from Rust, ticking per element.
pub fn pull<'py, T: Send, X>(
    site: &'static Site,
    iterable: &Bound<'py, PyAny>,
    error: impl Fn(PyErr) -> X,
    mut convert: impl FnMut(&Bound<'py, PyAny>) -> Result<T, X>,
) -> Result<Vec<T>, X> {
    let py = iterable.py();
    let mut out = Vec::new();
    let mut walk = || {
        for item in iterable.try_iter().map_err(&error)? {
            tick(py);
            if out.len() == out.capacity() {
                out = grown(py, std::mem::take(&mut out));
            }
            out.push(convert(&item.map_err(&error)?)?);
        }
        Ok(())
    };
    match walk() {
        Ok(()) => {
            site.note_elements(out.len());
            Ok(out)
        }
        Err(failed) => Err(discarded(py, out, failed)),
    }
}

/// `failed`, after the elements converted before it are freed, off the GIL
/// on a worker: a walk that fails late holds most of the sequence.
fn discarded<T: Send, X>(py: Python<'_>, out: Vec<T>, failed: X) -> X {
    leave(py, move || drop(out));
    failed
}

/// `out` with room for as many elements again. A reallocation copies the
/// whole vector, so on a worker it runs off the GIL.
fn grown<T: Send>(py: Python<'_>, mut out: Vec<T>) -> Vec<T> {
    leave(py, move || {
        out.reserve(out.len().max(SLICE));
        out
    })
}

/// A `frozenset` of the elements of `list`, a list [`build`] made.
pub fn frozen(list: Bound<'_, PyAny>) -> PyResult<Bound<'_, PyAny>> {
    let py = list.py();
    Ok(PyFrozenSet::new(py, list.try_iter()?.collect::<PyResult<Vec<_>>>()?)?.into_any())
}

/// Frees a list incr built for an argument a slice at a time, ticking per
/// element, when the backend kept no reference to it.
pub fn empty(list: Bound<'_, PyAny>) -> PyResult<()> {
    if list.get_refcnt() != 1 {
        return Ok(());
    }
    let Ok(list) = list.downcast_into::<PyList>() else {
        return Ok(());
    };
    let py = list.py();
    let mut end = list.len();
    while end > 0 {
        let start = end.saturating_sub(SLICE);
        for _ in start..end {
            tick(py);
        }
        list.del_slice(start, end)?;
        end = start;
    }
    Ok(())
}

fn base() -> Instant {
    static BASE: OnceLock<Instant> = OnceLock::new();
    *BASE.get_or_init(Instant::now)
}

fn since_base(at: Instant) -> u64 {
    nanos(at.saturating_duration_since(base()))
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// What the probe and the coverage check read: the clock, the sites
/// workers reached and their figures, the long holds, and a probe thread
/// that times how long each request for the GIL waits.
#[cfg(feature = "gil-checks")]
pub mod checks {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, PoisonError};
    use std::thread::{self, JoinHandle};
    use std::time::{Duration, Instant};

    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;
    use pyo3::types::PyModule;

    use super::{base, since_base, Kind, HOLDS, LIVE, SITES};

    /// Nanoseconds on the clock every figure here uses.
    #[pyfunction]
    fn _gil_now_ns() -> u64 {
        since_base(Instant::now())
    }

    /// Checkpoint workers alive now.
    #[pyfunction]
    fn _gil_live_workers() -> usize {
        LIVE.load(Ordering::Acquire)
    }

    /// Per site a worker reached since the last reset: `(name, kind, bound,
    /// worker entries, most elements in one call, longest hold in ns)`.
    #[pyfunction]
    #[allow(clippy::type_complexity)]
    fn _gil_sites() -> Vec<(&'static str, &'static str, Option<usize>, u64, u64, u64)> {
        SITES
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|site| {
                let (kind, bound) = match site.kind {
                    Kind::Caller => ("caller", None),
                    Kind::One => ("one", None),
                    Kind::Bounded(bound) => ("bounded", Some(bound)),
                    Kind::Unbounded => ("unbounded", None),
                };
                (
                    site.name,
                    kind,
                    bound,
                    site.entered.load(Ordering::Relaxed),
                    site.most.load(Ordering::Relaxed),
                    site.longest.load(Ordering::Relaxed),
                )
            })
            .collect()
    }

    /// Worker holds of 1 ms or more since the last reset, as `(start_ns,
    /// end_ns, site)`.
    #[pyfunction]
    fn _gil_holds() -> Vec<(u64, u64, &'static str)> {
        HOLDS.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    #[pyfunction]
    fn _gil_reset() {
        for site in SITES.lock().unwrap_or_else(PoisonError::into_inner).iter() {
            site.entered.store(0, Ordering::Relaxed);
            site.most.store(0, Ordering::Relaxed);
            site.longest.store(0, Ordering::Relaxed);
        }
        HOLDS.lock().unwrap_or_else(PoisonError::into_inner).clear();
    }

    struct Probe {
        stop: Arc<AtomicBool>,
        waits: Arc<Mutex<Vec<(u64, u64, u64)>>>,
        thread: JoinHandle<()>,
    }

    static PROBE: Mutex<Option<Probe>> = Mutex::new(None);

    /// Starts a thread that asks for the GIL, lets go at once and sleeps
    /// `interval_us`, recording each request's start, its wait and the part
    /// of the wait the thread spent runnable but not scheduled.
    #[pyfunction]
    #[pyo3(signature = (interval_us=200))]
    fn _gil_probe_start(interval_us: u64) -> PyResult<()> {
        let mut probe = PROBE.lock().unwrap_or_else(PoisonError::into_inner);
        if probe.is_some() {
            return Err(PyRuntimeError::new_err("the GIL probe is already running"));
        }
        base();
        let stop = Arc::new(AtomicBool::new(false));
        let waits = Arc::new(Mutex::new(Vec::new()));
        let thread = {
            let (stop, waits) = (Arc::clone(&stop), Arc::clone(&waits));
            thread::Builder::new()
                .name("incr-gil-probe".to_owned())
                .spawn(move || {
                    Python::attach(|py| {
                        super::detach(py, || {
                            while !stop.load(Ordering::Acquire) {
                                let queued = run_delay();
                                let asked = Instant::now();
                                Python::attach(|_| {});
                                let waited = asked.elapsed();
                                let unscheduled = run_delay().saturating_sub(queued);
                                waits.lock().unwrap_or_else(PoisonError::into_inner).push((
                                    since_base(asked),
                                    super::nanos(waited),
                                    unscheduled,
                                ));
                                thread::sleep(Duration::from_micros(interval_us));
                            }
                        })
                    })
                })?
        };
        *probe = Some(Probe {
            stop,
            waits,
            thread,
        });
        Ok(())
    }

    /// Nanoseconds this thread has spent runnable but not scheduled, from
    /// Linux's schedstat; 0 where it cannot be read.
    fn run_delay() -> u64 {
        std::fs::read_to_string("/proc/thread-self/schedstat")
            .ok()
            .and_then(|stat| stat.split_whitespace().nth(1)?.parse().ok())
            .unwrap_or(0)
    }

    /// Stops the probe and returns its `(asked_ns, waited_ns,
    /// unscheduled_ns)` records.
    #[pyfunction]
    fn _gil_probe_stop(py: Python<'_>) -> PyResult<Vec<(u64, u64, u64)>> {
        let probe = PROBE.lock().unwrap_or_else(PoisonError::into_inner).take();
        let Some(probe) = probe else {
            return Err(PyRuntimeError::new_err("the GIL probe is not running"));
        };
        probe.stop.store(true, Ordering::Release);
        super::release(py, || probe.thread.join())
            .map_err(|_| PyRuntimeError::new_err("the GIL probe panicked"))?;
        let waits =
            std::mem::take(&mut *probe.waits.lock().unwrap_or_else(PoisonError::into_inner));
        Ok(waits)
    }

    /// Adds the probe's functions to `module`.
    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(_gil_now_ns, module)?)?;
        module.add_function(wrap_pyfunction!(_gil_live_workers, module)?)?;
        module.add_function(wrap_pyfunction!(_gil_sites, module)?)?;
        module.add_function(wrap_pyfunction!(_gil_holds, module)?)?;
        module.add_function(wrap_pyfunction!(_gil_reset, module)?)?;
        module.add_function(wrap_pyfunction!(_gil_probe_start, module)?)?;
        module.add_function(wrap_pyfunction!(_gil_probe_stop, module)?)?;
        Ok(())
    }
}

#[cfg(all(test, feature = "gil-checks"))]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    struct EntersOnDrop;

    impl Drop for EntersOnDrop {
        fn drop(&mut self) {
            static SITE: Site = Site::one("test.enters_on_drop");
            enter(&SITE, |_| ());
        }
    }

    fn message(payload: &(dyn std::any::Any + Send)) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_default()
    }

    #[test]
    fn an_entry_while_unwinding_over_the_drop_count_lets_the_first_panic_through() {
        Python::initialize();
        let caught = Arc::new(Mutex::new(None));
        let inside = Arc::clone(&caught);
        let worker = thread::spawn(move || {
            // SAFETY: the body touches Python only through `enter`.
            unsafe {
                run_worker(&mut || {
                    for _ in 0..=MAX_RELEASED_DROPS {
                        note_value_drop();
                    }
                    let unwound = catch_unwind(AssertUnwindSafe(|| {
                        let _entry = EntersOnDrop;
                        panic!("the first panic");
                    }));
                    *inside.lock().unwrap() = unwound.err().map(|payload| message(&*payload));
                })
            };
        });
        let exit = worker
            .join()
            .expect_err("the worker's exit reports the drops");
        assert_eq!(caught.lock().unwrap().as_deref(), Some("the first panic"));
        assert!(message(&*exit).contains("dropped"), "{}", message(&*exit));
    }
}
