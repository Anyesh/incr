//! T8 replay under `Shared` with a second thread acting between replay
//! rounds: a thread parks its driver at a [`ParkPoint`] (or inside a
//! closure) while another lands, claims, deletes or sets the nodes its
//! virtual frames stand for. Each result is compared with the unbounded
//! run's in which the other thread's action came first.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering::SeqCst};
use std::sync::mpsc;
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::Duration;

use super::race_tests::{id, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::Shared;
use crate::cycle::{QueryOpts, Replay};
use crate::handle::Incr;
use crate::state::NodeState;

type Rt = Runtime<Shared>;

/// A replay that livelocks never returns, so a thread under test is
/// joined with a deadline.
const JOIN_TIMEOUT: Duration = Duration::from_secs(20);

fn replayable<T>() -> QueryOpts<T> {
    QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    }
}

/// Installs the runtime's park hook at `points` without arming the
/// gates, so a test arms each once the runs before it have passed.
fn hook(rt: &Rt, points: Vec<(ParkPoint, Gate)>) {
    let installed = rt.park_hook.set(Box::new(move |p| {
        for (point, gate) in &points {
            if *point == p {
                gate.park();
            }
        }
    }));
    assert!(installed.is_ok(), "park hook installed twice");
}

fn message(err: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = err.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        String::from("<non-string panic>")
    }
}

/// A read's outcome: its value, or its panic's message.
fn outcome<R>(f: impl FnOnce() -> R) -> Result<R, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|e| message(&*e))
}

/// Runs `f` on a new thread; [`Bounded::join`] fails the test if it has
/// not returned by [`JOIN_TIMEOUT`].
struct Bounded<R>(mpsc::Receiver<thread::Result<R>>);

fn spawn<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> Bounded<R> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(catch_unwind(AssertUnwindSafe(f)));
    });
    Bounded(rx)
}

impl<R> Bounded<R> {
    fn join(self) -> R {
        match self.0.recv_timeout(JOIN_TIMEOUT) {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => std::panic::resume_unwind(e),
            Err(_) => panic!("the thread under test never returned: the replay livelocked"),
        }
    }
}

/// How X's closure runs once the test switches it from the chain's own
/// run, which reads Y and so dooms at budget 1.
const CHAIN: u8 = 0;
const CLEAN: u8 = 1;
const CYCLIC: u8 = 2;
const FAILED: u8 = 3;
const PARKED: u8 = 4;

/// T reads X with `try_get`, X reads Y, and Y, when `y_reads_x`, reads X
/// with `try_get`; all replayable over a `u64` placeholder. A switched X
/// reads nothing (Clean 42), reads itself (Cyclic), panics, or parks on
/// `gate` and then lands 42.
struct Landed {
    rt: Arc<Rt>,
    t: Incr<u64>,
    x: Incr<u64>,
    y: Incr<u64>,
    mode: Arc<AtomicU8>,
    gate: Gate,
    x_runs: Arc<AtomicU32>,
}

fn landed_shape(rt: Rt, y_reads_x: bool) -> Landed {
    let rt = Arc::new(rt);
    rt.set_placeholder(0_u64);
    let mode = Arc::new(AtomicU8::new(CHAIN));
    let gate = Gate::new();
    let x_runs = Arc::new(AtomicU32::new(0));
    let xc: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let y = {
        let xc = xc.clone();
        rt.create_query_with(
            move |rt: &Rt| {
                if y_reads_x {
                    rt.try_get(*xc.get().unwrap()).map_or(5, |v| v + 100)
                } else {
                    7
                }
            },
            replayable(),
        )
    };
    let x = {
        let (xc, mode, gate, x_runs) = (xc.clone(), mode.clone(), gate.clone(), x_runs.clone());
        rt.create_query_with(
            move |rt: &Rt| {
                x_runs.fetch_add(1, SeqCst);
                match mode.load(SeqCst) {
                    CHAIN => rt.get(y) + 1,
                    CLEAN => 42,
                    CYCLIC => rt.try_get(*xc.get().unwrap()).map_or(9, |v| v + 1),
                    FAILED => panic!("x failed"),
                    _ => {
                        gate.park();
                        42
                    }
                }
            },
            replayable(),
        )
    };
    xc.set(x).unwrap();
    let t = rt.create_query_with(
        move |rt: &Rt| rt.try_get(x).map_or(1000, |v| v + 1),
        replayable(),
    );
    Landed {
        rt,
        t,
        x,
        y,
        mode,
        gate,
        x_runs,
    }
}

/// The unbounded run in which X's switched landing comes first:
/// `get(X)`'s outcome, then `get(T)`'s, and X's runs.
fn landed_first(mode: u8, y_reads_x: bool) -> (Result<u64, String>, Result<u64, String>, u32) {
    let s = landed_shape(Rt::new(), y_reads_x);
    s.mode.store(mode, SeqCst);
    let x = outcome(|| s.rt.get(s.x));
    let t = outcome(|| s.rt.get(s.t));
    (x, t, s.x_runs.load(SeqCst))
}

/// **A VFrame's node landed by another thread** (Virtual cycles step 1,
/// Loop step 2), at budget 1: T's run dooms on X and X's on Y, so T and
/// X are virtual frames when the driver takes Y. Parked there, another
/// thread lands X in `mode`; Y's read of X (when it makes one), the pop
/// of X's virtual frame and T's retry then behave as the unbounded run
/// in which that landing came first.
fn vframe_node_landed(mode: u8, y_reads_x: bool) {
    let s = landed_shape(Rt::new(), y_reads_x);
    s.rt.set_max_frames(1);
    let entry = Gate::new();
    hook(
        &s.rt,
        vec![(ParkPoint::BeforeEntry(id(s.y)), entry.clone())],
    );
    entry.arm();
    let reader = {
        let (rt, t) = (s.rt.clone(), s.t);
        spawn(move || outcome(|| rt.get(t)))
    };
    entry.wait_parked();
    assert_eq!(s.rt.state_of(s.t), NodeState::New);
    assert_eq!(s.rt.state_of(s.x), NodeState::New);
    s.mode.store(mode, SeqCst);
    let x = outcome(|| s.rt.get(s.x));
    entry.release();
    let t = reader.join();

    let (want_x, want_t, runs) = landed_first(mode, y_reads_x);
    assert_eq!(x, want_x);
    assert_eq!(t, want_t);
    assert_eq!(
        s.x_runs.load(SeqCst),
        runs + 1,
        "X ran once doomed and then as in the unbounded run, never again"
    );
    let landed = match mode {
        CLEAN => NodeState::Clean,
        CYCLIC => NodeState::Cyclic,
        _ => NodeState::Failed,
    };
    assert_eq!(s.rt.state_of(s.x), landed);
    assert_eq!(outcome(|| s.rt.get(s.t)), want_t, "a later read agrees");
    s.rt.assert_invariant_i();
}

#[test]
fn shared_vframe_node_landed_clean() {
    vframe_node_landed(CLEAN, false);
}

#[test]
fn shared_vframe_node_landed_clean_read_above() {
    vframe_node_landed(CLEAN, true);
}

#[test]
fn shared_vframe_node_landed_cyclic() {
    vframe_node_landed(CYCLIC, false);
}

#[test]
fn shared_vframe_node_landed_cyclic_read_above() {
    vframe_node_landed(CYCLIC, true);
}

#[test]
fn shared_vframe_node_landed_failed() {
    vframe_node_landed(FAILED, false);
}

#[test]
fn shared_vframe_node_landed_failed_read_above() {
    vframe_node_landed(FAILED, true);
}

/// The landed-VFrame shape with X Computing on the other thread at the
/// pop of its virtual frame: the pop does nothing, and X's entry below
/// waits on that claim, as the unbounded run's read of a node another
/// thread computes would.
#[test]
fn shared_vframe_node_computing_at_pop() {
    let s = landed_shape(Rt::new(), false);
    s.rt.set_max_frames(1);
    let (entry, wait) = (Gate::new(), Gate::new());
    hook(
        &s.rt,
        vec![
            (ParkPoint::BeforeEntry(id(s.y)), entry.clone()),
            (ParkPoint::BeforeComputeOne(id(s.x)), wait.clone()),
            (ParkPoint::ComputeWaits(id(s.x)), wait.clone()),
        ],
    );
    entry.arm();
    let reader = {
        let (rt, t) = (s.rt.clone(), s.t);
        spawn(move || rt.get(t))
    };
    entry.wait_parked();
    s.mode.store(PARKED, SeqCst);
    s.gate.arm();
    let lander = {
        let (rt, x) = (s.rt.clone(), s.x);
        spawn(move || rt.get(x))
    };
    s.gate.wait_parked();
    wait.arm();
    entry.release();
    wait.wait_parked();
    assert_eq!(s.rt.state_of(s.x), NodeState::Computing);
    s.gate.release();
    assert_eq!(lander.join(), 42);
    wait.release();
    assert_eq!(reader.join(), 43);
    assert_eq!(s.x_runs.load(SeqCst), 2);
    assert_eq!(s.rt.state_of(s.t), NodeState::Clean);
    s.rt.assert_invariant_i();
}
