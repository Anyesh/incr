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

/// T reads X1, X1 reads X2 (until switched to land 42 alone), X2 reads Y,
/// and Y reads T with `try_get`, all replayable; T and X1 count runs.
struct Below {
    rt: Arc<Rt>,
    t: Incr<u64>,
    x1: Incr<u64>,
    y: Incr<u64>,
    landed: Arc<AtomicU8>,
    t_runs: Arc<AtomicU32>,
}

fn below_shape(rt: Rt) -> Below {
    let rt = Arc::new(rt);
    rt.set_placeholder(0_u64);
    let landed = Arc::new(AtomicU8::new(CHAIN));
    let t_runs = Arc::new(AtomicU32::new(0));
    let tc: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let y = {
        let tc = tc.clone();
        rt.create_query_with(
            move |rt: &Rt| rt.try_get(*tc.get().unwrap()).map_or(5, |v| v + 100),
            replayable(),
        )
    };
    let x2 = rt.create_query_with(move |rt: &Rt| rt.get(y) + 1, replayable());
    let x1 = {
        let landed = landed.clone();
        rt.create_query_with(
            move |rt: &Rt| match landed.load(SeqCst) {
                CHAIN => rt.get(x2) + 1,
                _ => 42,
            },
            replayable(),
        )
    };
    let t = {
        let t_runs = t_runs.clone();
        rt.create_query_with(
            move |rt: &Rt| {
                t_runs.fetch_add(1, SeqCst);
                rt.get(x1) + 1
            },
            replayable(),
        )
    };
    tc.set(t).unwrap();
    Below {
        rt,
        t,
        x1,
        y,
        landed,
        t_runs,
    }
}

/// **A landed point below the running entry's maker** (Dropping a
/// speculative chain), at budget 3: T's run makes virtual frames of T, X1
/// and X2 and pushes Y, and another thread lands X1 before Y runs. Y's
/// read of T meets X1 landed above T's frame, a landed point, so the drop
/// removes only the entries above X1's level and keeps T, the level-0
/// target, whose retry reads X1 Clean.
#[test]
fn shared_landed_point_below_the_running_entrys_maker() {
    let s = below_shape(Rt::new());
    s.rt.set_max_frames(3);
    let entry = Gate::new();
    hook(
        &s.rt,
        vec![(ParkPoint::BeforeEntry(id(s.y)), entry.clone())],
    );
    entry.arm();
    let reader = {
        let (rt, t) = (s.rt.clone(), s.t);
        spawn(move || rt.get(t))
    };
    entry.wait_parked();
    assert_eq!(s.rt.state_of(s.x1), NodeState::New);
    s.landed.store(CLEAN, SeqCst);
    assert_eq!(s.rt.get(s.x1), 42);
    entry.release();

    let unbounded = below_shape(Rt::new());
    unbounded.landed.store(CLEAN, SeqCst);
    assert_eq!(unbounded.rt.get(unbounded.x1), 42);
    let want = unbounded.rt.get(unbounded.t);
    assert_eq!(reader.join(), want);
    assert_eq!(
        s.t_runs.load(SeqCst),
        2,
        "T ran doomed and then as its own retry"
    );
    assert_eq!(s.rt.state_of(s.t), NodeState::Clean);
    s.rt.assert_invariant_i();
}

/// The T1 taint shape at n = 2 over inputs s, t and j: P reads s and, when
/// s = 1, a New query A and then c0; c0 reads c1; c1 reads t and then P
/// when t = 0 or j when t = 1. Computed at s = t = 0, so c1 holds an
/// installed edge to P, then set to s = t = 1, j = 5, which leaves that
/// edge stale. At budget 1, P dooms on A and its doomed read of c0 pushes
/// c0 as a walk. P parks on `gate` once it is armed.
struct Taint {
    rt: Arc<Rt>,
    p: Incr<u64>,
    c0: Incr<u64>,
    gate: Gate,
    p_runs: Arc<AtomicU32>,
}

fn taint_shape() -> Taint {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    let (s, t, j) = (
        rt.create_input(0_u64),
        rt.create_input(0_u64),
        rt.create_input(0_u64),
    );
    let gate = Gate::new();
    let p_runs = Arc::new(AtomicU32::new(0));
    let pc: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let c1 = {
        let pc = pc.clone();
        rt.create_query_with(
            move |rt: &Rt| {
                if rt.get(t) == 0 {
                    rt.get(*pc.get().unwrap()) + 1
                } else {
                    rt.get(j) + 1
                }
            },
            replayable(),
        )
    };
    let c0 = rt.create_query_with(move |rt: &Rt| rt.get(c1) + 1, replayable());
    let a = rt.create_query_with(|_: &Rt| 1_u64, replayable());
    let p = {
        let (gate, p_runs) = (gate.clone(), p_runs.clone());
        rt.create_query_with(
            move |rt: &Rt| {
                p_runs.fetch_add(1, SeqCst);
                gate.park();
                if rt.get(s) == 1 {
                    rt.get(a) + 1000 + rt.get(c0)
                } else {
                    0
                }
            },
            replayable(),
        )
    };
    pc.set(p).unwrap();
    assert_eq!(rt.get(c0), 2);
    rt.set(s, 1);
    rt.set(t, 1);
    rt.set(j, 5);
    p_runs.store(0, SeqCst);
    Taint {
        rt,
        p,
        c0,
        gate,
        p_runs,
    }
}

/// **The walk's Computing arm at a VFrame** (T1 The Computing arm too),
/// at budget 1: thread 1's `get(P)` dooms P on its read of c0, so P is a
/// virtual frame when the driver takes c0. Thread 2 then claims P and
/// parks inside P's closure, and c0's entry walk reaches P Computing
/// through c1's stale edge: it skips P and taints the path instead of
/// waiting, so thread 1 reaches its level-0 target while thread 2 still
/// holds P. Both threads read the value of the unbounded run in which
/// thread 2's claim came first, and P runs only doomed and on thread 2.
#[test]
fn shared_walk_skips_a_vframe_computing_on_another_thread() {
    let s = taint_shape();
    s.rt.set_max_frames(1);
    let (entry, target) = (Gate::new(), Gate::new());
    hook(
        &s.rt,
        vec![
            (ParkPoint::BeforeEntry(id(s.c0)), entry.clone()),
            (ParkPoint::BeforeEntry(id(s.p)), target.clone()),
        ],
    );
    entry.arm();
    let reader = {
        let (rt, p) = (s.rt.clone(), s.p);
        spawn(move || rt.get(p))
    };
    entry.wait_parked();
    assert_eq!(s.p_runs.load(SeqCst), 1);
    s.gate.arm();
    let claimer = {
        let (rt, p) = (s.rt.clone(), s.p);
        spawn(move || rt.get(p))
    };
    s.gate.wait_parked();
    target.arm();
    entry.release();
    target.wait_parked();
    assert_eq!(s.rt.state_of(s.p), NodeState::Computing);
    assert_eq!(s.rt.state_of(s.c0), NodeState::Clean);
    s.gate.release();
    let unbounded = {
        let u = taint_shape();
        u.rt.get(u.p)
    };
    assert_eq!(claimer.join(), unbounded);
    target.release();
    assert_eq!(reader.join(), unbounded);
    assert_eq!(s.p_runs.load(SeqCst), 2, "thread 1's walk never ran P");
    s.rt.assert_invariant_i();
}

/// The landed-VFrame shape with X deleted by another thread between
/// rounds: X's entry is dropped as `Gone`, and T's retry reads X by
/// handle and panics as stale, as the unbounded run does when the delete
/// comes first.
#[test]
fn shared_delete_of_a_vframes_node_between_rounds() {
    let s = landed_shape(Rt::new(), false);
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
    s.rt.delete_node(s.x);
    entry.release();
    let got = reader.join();

    let unbounded = landed_shape(Rt::new(), false);
    unbounded.rt.delete_node(unbounded.x);
    let want = outcome(|| unbounded.rt.get(unbounded.t));
    assert!(
        want.as_ref().is_err_and(|m| m.contains("stale")),
        "{want:?}"
    );
    assert_eq!(got, want);
    assert_eq!(s.x_runs.load(SeqCst), 1, "only X's doomed run ran");
    s.rt.assert_invariant_i();
}

/// Where the recycling delete lands relative to Y's read of T.
#[derive(Clone, Copy)]
enum Recycle {
    BeforeScan,
    BeforeClaim,
}

/// **A delete during replay, with recycling** (judgement steps 2 and 4,
/// Loop step 2), at budget 1 on the shape where Y reads T: T, X1 and X2
/// are virtual frames when Y reads T, and another thread deletes X1 and
/// recycles its slot into a New query Z before the read's scan, or
/// between the scan and the claim. The read drops the chain at X1's
/// level, Z's closure never runs, and `get(T)` panics as stale, as the
/// unbounded run does with the delete first.
fn delete_and_recycle_during_replay(when: Recycle) {
    let s = below_shape(Rt::new());
    s.rt.set_max_frames(1);
    let gate = Gate::new();
    let point = match when {
        Recycle::BeforeScan => ParkPoint::BeforeEntry(id(s.y)),
        Recycle::BeforeClaim => ParkPoint::BeforeVirtualClaim(id(s.t)),
    };
    hook(&s.rt, vec![(point, gate.clone())]);
    gate.arm();
    let reader = {
        let (rt, t) = (s.rt.clone(), s.t);
        spawn(move || outcome(|| rt.get(t)))
    };
    gate.wait_parked();
    s.rt.delete_node(s.x1);
    let z_runs = Arc::new(AtomicU32::new(0));
    let z = {
        let z_runs = z_runs.clone();
        s.rt.create_query_with(
            move |_: &Rt| {
                z_runs.fetch_add(1, SeqCst);
                9_u64
            },
            replayable(),
        )
    };
    assert_eq!(z.slot(), s.x1.slot(), "Z reuses X1's slot");
    gate.release();
    let got = reader.join();

    let unbounded = below_shape(Rt::new());
    unbounded.rt.delete_node(unbounded.x1);
    unbounded
        .rt
        .create_query_with(|_: &Rt| 9_u64, replayable::<u64>());
    let want = outcome(|| unbounded.rt.get(unbounded.t));
    assert!(
        want.as_ref().is_err_and(|m| m.contains("stale")),
        "{want:?}"
    );
    assert_eq!(got, want);
    assert_eq!(z_runs.load(SeqCst), 0, "Z ran on the reader's behalf");
    assert_eq!(s.rt.state_of(z), NodeState::New);
    s.rt.assert_invariant_i();
}

#[test]
fn shared_delete_and_recycle_before_the_scan() {
    delete_and_recycle_during_replay(Recycle::BeforeScan);
}

#[test]
fn shared_delete_and_recycle_between_scan_and_claim() {
    delete_and_recycle_during_replay(Recycle::BeforeClaim);
}

/// The Local form of a delete during replay: Y's closure deletes X, whose
/// virtual frame stands for a frame on the stack below Y, and panics as
/// a delete from inside the node's own compute, as the unbounded run's
/// delete of a node it is computing does.
#[test]
fn local_delete_of_a_vframes_node_from_its_own_thread() {
    fn run(budget: Option<u32>) -> Result<u64, String> {
        let rt = Runtime::<crate::cells::Local>::new();
        rt.set_placeholder(0_u64);
        if let Some(n) = budget {
            rt.set_max_frames(n);
        }
        let xc: Arc<OnceLock<Incr<u64>>> = Arc::default();
        let y = {
            let xc = xc.clone();
            rt.create_query_with(
                move |rt: &Runtime<crate::cells::Local>| {
                    rt.delete_node(*xc.get().unwrap());
                    7_u64
                },
                replayable(),
            )
        };
        let x = rt.create_query_with(
            move |rt: &Runtime<crate::cells::Local>| rt.get(y) + 1,
            replayable(),
        );
        xc.set(x).unwrap();
        let t = rt.create_query_with(
            move |rt: &Runtime<crate::cells::Local>| rt.get(x) + 1,
            replayable(),
        );
        let got = outcome(|| rt.get(t));
        rt.assert_invariant_i();
        got
    }
    let want = run(None);
    assert!(
        want.as_ref().is_err_and(|m| m.contains("own compute")),
        "{want:?}"
    );
    assert_eq!(run(Some(1)), want);
}
