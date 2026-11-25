//! A checkpoint in flight: the store or memo that begins it keeps a
//! [`Flight`] and settles it on its own thread, a worker thread encodes and
//! writes, and the caller's handle holds only a [`FlightToken`].

mod chunked;
mod parcel;

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

pub(crate) use chunked::Chunked;
pub(crate) use parcel::Parcel;

use crate::value_context::{drop_then_hold, enter_value_scope, start_worker};
use crate::{Error, Result, ValueScope};

/// What a flight's worker posts once it is done.
pub(crate) enum Verdict<T> {
    /// The backend committed and was handed every item.
    Committed(T),
    /// The backend returned `Ok` after it was handed an `Err`, or before it
    /// pulled every item, so what it committed may be truncated.
    Truncated,
    /// The backend or the worker's own work failed, and nothing was
    /// committed.
    Failed(Error),
    /// The flight was cancelled or abandoned before the backend committed.
    Cancelled,
    /// The worker panicked; the payload is the panic's.
    Panicked(Box<dyn Any + Send>),
}

/// The state a flight's owner, its worker and its handle share. Every
/// method is lock-free, so the handle's drop is safe on any thread, the
/// worker's included, and under any lock.
#[derive(Default)]
pub(crate) struct Control {
    cancelled: AtomicBool,
    abandoned: AtomicBool,
    worker: OnceLock<Thread>,
}

impl Control {
    /// Whether the worker should stop: the flight was cancelled or
    /// abandoned.
    pub(crate) fn stopped(&self) -> bool {
        self.cancelled.load(Ordering::Acquire) || self.abandoned.load(Ordering::Acquire)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn abandoned(&self) -> bool {
        self.abandoned.load(Ordering::Acquire)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.wake();
    }

    fn abandon(&self) {
        self.abandoned.store(true, Ordering::Release);
        self.wake();
    }

    fn wake(&self) {
        if let Some(worker) = self.worker.get() {
            worker.unpark();
        }
    }
}

/// The part of a flight its handle holds. Dropping it abandons the flight.
pub(crate) struct FlightToken {
    control: Arc<Control>,
}

impl FlightToken {
    /// Whether this is the token of the flight `control` steers.
    pub(crate) fn is_of(&self, control: &Control) -> bool {
        std::ptr::eq(&*self.control, control)
    }

    pub(crate) fn cancel(&self) {
        self.control.cancel();
    }

    pub(crate) fn worker(&self) -> Option<std::thread::ThreadId> {
        self.control.worker.get().map(Thread::id)
    }
}

impl Drop for FlightToken {
    fn drop(&mut self) {
        self.control.abandon();
    }
}

/// The owner's side of a flight: it sends the worker its messages and takes
/// the verdict. Dropping it abandons the flight without waiting for the
/// worker.
pub(crate) struct Flight<M, T> {
    control: Arc<Control>,
    outbox: Sender<M>,
    posted: Arc<Posted<T>>,
}

impl<M: Send + 'static, T: Send + 'static> Flight<M, T> {
    /// Spawns a worker thread that runs `body` with `scope` current, inside
    /// the scope's worker start. A panic in `body` or in the start is
    /// posted as [`Verdict::Panicked`].
    pub(crate) fn start(
        scope: Option<ValueScope>,
        body: impl FnOnce(Inbox<M>) -> Verdict<T> + Send + 'static,
    ) -> Result<(Self, FlightToken)> {
        let control = Arc::new(Control::default());
        let posted = Arc::new(Posted::default());
        let (outbox, rx) = mpsc::channel();
        let inbox = Inbox {
            rx,
            control: Arc::clone(&control),
        };
        let worker = {
            let posted = Arc::clone(&posted);
            thread::Builder::new()
                .name("incr-checkpoint".to_owned())
                .stack_size(WORKER_STACK)
                .spawn(move || {
                    posted.post(run_worker(scope.clone(), inbox, body));
                    drop_then_hold(scope, posted);
                })?
        };
        let _ = control.worker.set(worker.thread().clone());
        register(Arc::clone(&control), worker);
        let token = FlightToken {
            control: Arc::clone(&control),
        };
        Ok((
            Flight {
                control,
                outbox,
                posted,
            },
            token,
        ))
    }
}

impl<M, T> Flight<M, T> {
    /// Hands `message` to the worker. Once the worker has stopped, the
    /// message comes back instead.
    pub(crate) fn send(&self, message: M) -> Option<M> {
        match self.outbox.send(message) {
            Ok(()) => {
                self.control.wake();
                None
            }
            Err(unsent) => Some(unsent.0),
        }
    }

    pub(crate) fn cancel(&self) {
        self.control.cancel();
    }

    pub(crate) fn control(&self) -> &Arc<Control> {
        &self.control
    }

    pub(crate) fn has_verdict(&self) -> bool {
        self.posted.ready.load(Ordering::Acquire)
    }

    pub(crate) fn posted_committed(&self) -> bool {
        matches!(
            *self
                .posted
                .verdict
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
            Some(Verdict::Committed(_))
        )
    }

    /// The worker's verdict once posted, taken: a second call returns
    /// `None`.
    pub(crate) fn take_verdict(&self) -> Option<Verdict<T>> {
        self.posted
            .verdict
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// Blocks until the worker posts its verdict or `timeout` passes, and
    /// returns whether it was posted.
    pub(crate) fn wait(&self, timeout: Option<Duration>) -> bool {
        self.posted.wait(timeout)
    }

    /// A handle to wait for the verdict on without holding the flight's
    /// owner.
    pub(crate) fn waiter(&self) -> Waiter<T> {
        Waiter(Arc::clone(&self.posted))
    }
}

/// What [`Flight::waiter`] returns.
pub(crate) struct Waiter<T>(Arc<Posted<T>>);

impl<T> Waiter<T> {
    /// [`Flight::wait`].
    pub(crate) fn wait(&self, timeout: Option<Duration>) -> bool {
        self.0.wait(timeout)
    }
}

impl<M, T> Drop for Flight<M, T> {
    fn drop(&mut self) {
        self.control.abandon();
    }
}

/// The worker's side of a flight's messages.
pub(crate) struct Inbox<M> {
    rx: Receiver<M>,
    control: Arc<Control>,
}

impl<M> Inbox<M> {
    /// The next message, waiting for it, or `None` once the flight is
    /// cancelled or abandoned or its owner is gone. Messages still queued
    /// then are dropped with the inbox.
    pub(crate) fn recv(&self) -> Option<M> {
        loop {
            if self.control.stopped() {
                return None;
            }
            match self.rx.try_recv() {
                Ok(message) => return Some(message),
                Err(TryRecvError::Empty) => thread::park(),
                Err(TryRecvError::Disconnected) => return None,
            }
        }
    }

    pub(crate) fn control(&self) -> &Arc<Control> {
        &self.control
    }
}

struct Posted<T> {
    verdict: Mutex<Option<Verdict<T>>>,
    ready: AtomicBool,
    changed: Condvar,
}

impl<T> Default for Posted<T> {
    fn default() -> Self {
        Posted {
            verdict: Mutex::new(None),
            ready: AtomicBool::new(false),
            changed: Condvar::new(),
        }
    }
}

impl<T> Posted<T> {
    fn wait(&self, timeout: Option<Duration>) -> bool {
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        let mut verdict = self.verdict.lock().unwrap_or_else(PoisonError::into_inner);
        while !self.ready.load(Ordering::Acquire) {
            verdict = match deadline {
                None => self
                    .changed
                    .wait(verdict)
                    .unwrap_or_else(PoisonError::into_inner),
                Some(deadline) => {
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return false;
                    }
                    self.changed
                        .wait_timeout(verdict, left)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
            };
        }
        true
    }

    fn post(&self, verdict: Verdict<T>) {
        let mut slot = self.verdict.lock().unwrap_or_else(PoisonError::into_inner);
        *slot = Some(verdict);
        self.ready.store(true, Ordering::Release);
        drop(slot);
        self.changed.notify_all();
    }
}

fn run_worker<M, T>(
    scope: Option<ValueScope>,
    inbox: Inbox<M>,
    body: impl FnOnce(Inbox<M>) -> Verdict<T>,
) -> Verdict<T> {
    let _entered = scope.clone().map(enter_value_scope);
    let mut verdict = None;
    let started = panic::catch_unwind(AssertUnwindSafe(|| {
        start_worker(scope.as_ref(), || {
            verdict = Some(
                panic::catch_unwind(AssertUnwindSafe(|| body(inbox)))
                    .unwrap_or_else(Verdict::Panicked),
            );
        });
    }));
    match (verdict, started) {
        (_, Err(payload)) => Verdict::Panicked(payload),
        (Some(verdict), Ok(())) => verdict,
        (None, Ok(())) => unreachable!("start_worker returns only after its argument ran"),
    }
}

type Worker = (Arc<Control>, JoinHandle<()>);

/// The size of a main thread's stack on Linux, which the value code and
/// backends a worker runs were written against.
const WORKER_STACK: usize = 8 << 20;

static WORKERS: Mutex<Vec<Worker>> = Mutex::new(Vec::new());

fn register(control: Arc<Control>, worker: JoinHandle<()>) {
    register_in(&WORKERS, control, worker);
}

fn register_in(registry: &Mutex<Vec<Worker>>, control: Arc<Control>, worker: JoinHandle<()>) {
    let mut workers = registry.lock().unwrap_or_else(PoisonError::into_inner);
    workers.retain(|(_, worker)| !worker.is_finished());
    workers.push((control, worker));
}

/// Cancels every checkpoint flight whose worker thread is still running,
/// abandoned ones included, and waits for each of those threads to end. For
/// process or interpreter exit. A worker may be waiting for whatever a value
/// scope's `hold` takes, so call this having given that up.
pub fn join_checkpoint_workers() {
    join_registered(&WORKERS);
}

/// Checkpoint worker threads started and not yet ended, abandoned flights'
/// included. A thread counts once the call that spawned it has returned,
/// until its thread ends.
pub fn checkpoint_workers_running() -> usize {
    WORKERS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .filter(|(_, worker)| !worker.is_finished())
        .count()
}

fn join_registered(registry: &Mutex<Vec<Worker>>) {
    let workers = std::mem::take(&mut *registry.lock().unwrap_or_else(PoisonError::into_inner));
    for (control, _) in &workers {
        control.cancel();
    }
    for (_, worker) in workers {
        if let Err(payload) = worker.join() {
            panic::resume_unwind(payload);
        }
    }
}

#[cfg(test)]
mod chunked_tests;
#[cfg(test)]
mod parcel_tests;
#[cfg(test)]
mod tests;
