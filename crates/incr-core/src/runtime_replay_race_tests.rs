//! T8 replay under `Shared` with a second thread acting between replay
//! rounds: a thread parks its driver at a [`ParkPoint`] (or inside a
//! closure) while another lands, claims, deletes or sets the nodes its
//! virtual frames stand for. Each result is compared with the unbounded
//! run's in which the other thread's action came first.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering::SeqCst};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, ThreadId};
use std::time::Duration;

use super::race_tests::{id, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Shared};
use crate::cycle::{Cycle, CycleStrategy, QueryOpts, Replay};
use crate::handle::Incr;
use crate::segmented_nodes::IN_GROUP;
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

type Outcome = Result<u64, String>;

/// The unbounded run in which X's switched landing comes first:
/// `get(X)`'s outcome, then `get(T)`'s, X's runs, and a later `get(T)`'s
/// outcome, which for a Failed T names T rather than X.
fn landed_first(mode: u8, y_reads_x: bool) -> (Outcome, Outcome, u32, Outcome) {
    let s = landed_shape(Rt::new(), y_reads_x);
    s.mode.store(mode, SeqCst);
    let x = outcome(|| s.rt.get(s.x));
    let t = outcome(|| s.rt.get(s.t));
    let runs = s.x_runs.load(SeqCst);
    (x, t, runs, outcome(|| s.rt.get(s.t)))
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

    let (want_x, want_t, runs, want_later) = landed_first(mode, y_reads_x);
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
    assert_eq!(outcome(|| s.rt.get(s.t)), want_later, "a later read agrees");
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
    unbounded.rt.set_max_frames(u32::MAX);
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
/// c0 as a walk. P parks on `gate` once it is armed, and on `late` after
/// its read of A.
struct Taint {
    rt: Arc<Rt>,
    p: Incr<u64>,
    a: Incr<u64>,
    c0: Incr<u64>,
    gate: Gate,
    late: Gate,
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
    let (gate, late) = (Gate::new(), Gate::new());
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
        let (gate, late, p_runs) = (gate.clone(), late.clone(), p_runs.clone());
        rt.create_query_with(
            move |rt: &Rt| {
                p_runs.fetch_add(1, SeqCst);
                gate.park();
                if rt.get(s) == 1 {
                    let a = rt.get(a);
                    late.park();
                    a + 1000 + rt.get(c0)
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
        a,
        c0,
        gate,
        late,
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
    // c0's entry follows the placeholder P read for `a`, so it is
    // speculative: its walk ran, and its landing waits in a stash.
    assert_eq!(s.rt.state_of(s.c0), NodeState::Dirty);
    s.gate.release();
    let unbounded = {
        let u = taint_shape();
        u.rt.set_max_frames(u32::MAX);
        u.rt.get(u.p)
    };
    assert_eq!(claimer.join(), unbounded);
    target.release();
    assert_eq!(reader.join(), unbounded);
    assert_eq!(s.p_runs.load(SeqCst), 2, "thread 1's walk never ran P");
    s.rt.assert_invariant_i();
}

/// **The walk's Computing arm at a VFrame, the discard-landing form** (T1
/// The Computing arm too, row 22), at budget 1. The plan's form parks
/// thread 1's walk inside its wait on P and has thread 2 discard-land P,
/// so that the wait ends with P Dirty and a virtual frame of it on the
/// first thread. That precondition is unreachable: the walk looks P's
/// virtual frame up before its Computing arm waits, and only thread 1
/// makes its own virtual frames, which it cannot do while it waits, so a
/// wait on P begins only when P has none and ends the same way.
/// `post_wait_owner` asserts it on every post-wait claim. This test runs
/// the reachable order instead: thread 1's `get(P)` dooms P, then thread 2
/// claims P and dooms on A, which thread 1's driver has not run yet, and
/// parks inside P's doomed run. c0's entry walk reaches P Computing and
/// skips it without waiting; thread 2 then discard-lands P and parks
/// before its retry, and thread 1's target entry claims P from that
/// landing.
#[test]
fn shared_walk_skips_a_vframe_that_another_thread_discard_lands() {
    let s = taint_shape();
    s.rt.set_max_frames(1);
    let (first, target, held) = (Gate::new(), Gate::new(), Gate::new());
    let p = id(s.p);
    let seen: Arc<Mutex<Vec<(ThreadId, ParkPoint)>>> = Arc::default();
    let installed = s.rt.park_hook.set(Box::new({
        let gates = vec![
            (ParkPoint::BeforeEntry(id(s.a)), first.clone()),
            (ParkPoint::BeforeEntry(p), target.clone()),
            (ParkPoint::BeforeEntry(p), held.clone()),
        ];
        let seen = seen.clone();
        move |point| {
            if matches!(point, ParkPoint::BeforeComputeOne(n) | ParkPoint::ComputeWaits(n) if n == p)
            {
                seen.lock().unwrap().push((thread::current().id(), point));
            }
            for (at, gate) in &gates {
                if *at == point {
                    gate.park();
                }
            }
        }
    }));
    assert!(installed.is_ok(), "park hook installed twice");
    first.arm();
    let reader_id: Arc<OnceLock<ThreadId>> = Arc::default();
    let reader = {
        let (rt, p, reader_id) = (s.rt.clone(), s.p, reader_id.clone());
        spawn(move || {
            reader_id.set(thread::current().id()).unwrap();
            rt.get(p)
        })
    };
    first.wait_parked();
    assert_eq!(s.rt.state_of(s.a), NodeState::New);
    s.late.arm();
    let claimer = {
        let (rt, p) = (s.rt.clone(), s.p);
        spawn(move || rt.get(p))
    };
    s.late.wait_parked();
    assert_eq!(s.p_runs.load(SeqCst), 2);
    target.arm();
    first.release();
    target.wait_parked();
    assert_eq!(s.rt.state_of(s.p), NodeState::Computing);
    // c0's entry follows the placeholder P read for `a`, so it is
    // speculative: its walk ran, and its landing waits in a stash.
    assert_eq!(s.rt.state_of(s.c0), NodeState::Dirty);
    held.arm();
    s.late.release();
    held.wait_parked();
    assert_eq!(
        s.rt.state_of(s.p),
        NodeState::Dirty,
        "thread 2 discard-landed P"
    );
    assert_eq!(s.p_runs.load(SeqCst), 2);
    let unbounded = {
        let u = taint_shape();
        u.rt.set_max_frames(u32::MAX);
        u.rt.get(u.p)
    };
    target.release();
    assert_eq!(reader.join(), unbounded);
    held.release();
    assert_eq!(claimer.join(), unbounded);
    assert_eq!(
        s.p_runs.load(SeqCst),
        3,
        "P ran doomed on each thread and once more in thread 1's target entry"
    );
    let reader_id = *reader_id.get().unwrap();
    let waits: Vec<ParkPoint> = seen
        .lock()
        .unwrap()
        .iter()
        .filter(|(t, _)| *t == reader_id)
        .map(|&(_, point)| point)
        .collect();
    // A forced revisit of P computes it from the walk a second time.
    if !super::forced_revisit() {
        assert_eq!(
            waits,
            vec![ParkPoint::BeforeComputeOne(p)],
            "thread 1's only walk compute of P is its root's, and it never waited on P"
        );
    }
    s.rt.assert_invariant_i();
}

/// Thread 1's replayable chain: H, New, reads X; X reads input i while
/// t = 0, and otherwise parks on `x_gate` and reads Z. Thread 2's side is
/// not replayable: Z, New, parks on `z_gate` and reads W + 1, and W reads
/// X while s = 0 and k otherwise. W is computed at s = t = 0, so it holds
/// an edge to X, and then s = t = 1 leaves that edge stale. Z is New so
/// that its closure runs before any walk reaches X.
struct WoundEntry {
    rt: Arc<Rt>,
    h: Incr<u64>,
    x: Incr<u64>,
    z: Incr<u64>,
    x_gate: Gate,
    z_gate: Gate,
}

fn wound_entry_shape() -> WoundEntry {
    // The entry counts below assume only the queries marked replayable are.
    let mut rt = Rt::new();
    rt.all_replayable = false;
    let rt = Arc::new(rt);
    rt.set_placeholder(0_u64);
    let (s, t, i, k) = (
        rt.create_input(0_u64),
        rt.create_input(0_u64),
        rt.create_input(1_u64),
        rt.create_input(7_u64),
    );
    let (x_gate, z_gate) = (Gate::new(), Gate::new());
    let zc: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let x = {
        let (zc, x_gate) = (zc.clone(), x_gate.clone());
        rt.create_query_with(
            move |rt: &Rt| {
                if rt.get(t) == 0 {
                    rt.get(i)
                } else {
                    x_gate.park();
                    rt.get(*zc.get().unwrap())
                }
            },
            replayable(),
        )
    };
    let w = rt.create_query(move |rt: &Rt| if rt.get(s) == 0 { rt.get(x) } else { rt.get(k) });
    let z = {
        let z_gate = z_gate.clone();
        rt.create_query(move |rt: &Rt| {
            z_gate.park();
            rt.get(w) + 1
        })
    };
    zc.set(z).unwrap();
    assert_eq!(rt.get(w), 1);
    rt.set(s, 1);
    rt.set(t, 1);
    let h = rt.create_query_with(move |rt: &Rt| rt.get(x) + 1, replayable());
    WoundEntry {
        rt,
        h,
        x,
        z,
        x_gate,
        z_gate,
    }
}

/// **The entry guard spans the loop, the wound form** (T5 Entry depth,
/// Contexts), at `set_max_frames(1)`: thread 1's `get(H)` dooms H on its
/// read of X and parks at X's entry. Thread 2 claims Z, X's entry claims
/// X and parks in its closure, and Z's read of W waits on X through W's
/// stale edge, registered first; X's read of Z then closes the waits-for
/// cycle and the registry wounds thread 1, the younger, inside the entry.
/// Thread 1's retry runs `get(H)`'s `ensure_clean` and its loop again in
/// a new context and returns the value of the unbounded run in which
/// thread 2's `get(Z)` came first, and thread 2's result is that run's.
fn wound_inside_a_driver_entry() {
    let s = wound_entry_shape();
    s.rt.set_max_frames(1);
    let entry = Gate::new();
    hook(
        &s.rt,
        vec![(ParkPoint::BeforeEntry(id(s.x)), entry.clone())],
    );
    entry.arm();
    let reader = {
        let (rt, h) = (s.rt.clone(), s.h);
        spawn(move || {
            let got = rt.get(h);
            (got, crate::replay::with(|c| c.is_empty()))
        })
    };
    entry.wait_parked();
    let before: Vec<u64> =
        s.rt.entries
            .lock()
            .unwrap()
            .iter()
            .map(|e| e.context)
            .collect();
    assert_eq!(before.len(), 1, "X's entry is the loop's first");
    s.z_gate.arm();
    let claimer = {
        let (rt, z) = (s.rt.clone(), s.z);
        spawn(move || outcome(|| rt.get(z)))
    };
    s.z_gate.wait_parked();
    s.x_gate.arm();
    entry.release();
    s.x_gate.wait_parked();
    assert_eq!(s.rt.state_of(s.x), NodeState::Computing);
    s.z_gate.release();
    await_registered_wait(&s.rt, s.x);
    s.x_gate.release();
    let (got, empty) = reader.join();
    let claimed = claimer.join();

    let u = wound_entry_shape();
    let want_z = outcome(|| u.rt.get(u.z));
    let want_h = u.rt.get(u.h);
    assert_eq!(claimed, want_z);
    assert_eq!(got, want_h);
    assert!(empty, "the read returns with no context open");
    let entries = s.rt.entries.lock().unwrap();
    let after = &entries[before.len()..];
    assert!(
        after
            .iter()
            .any(|e| e.target == Some(id(s.h).0) && !before.contains(&e.context)),
        "the retry ran H's loop in a new context"
    );
    assert!(
        after.iter().all(|e| !before.contains(&e.context)),
        "no entry ran in the wounded context after the wound"
    );
    drop(entries);
    super::wound_tests::assert_wounded();
    assert_eq!(crate::waits::registered(), 0);
    s.rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_wound_inside_a_driver_entry_retries_from_an_empty_context() {
    crate::child_test::run_in_child(
        "runtime::replay_race_tests::shared_wound_inside_a_driver_entry_retries_from_an_empty_context",
        wound_inside_a_driver_entry,
    )
    .assert_clean();
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
    unbounded.rt.set_max_frames(u32::MAX);
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
    unbounded.rt.set_max_frames(u32::MAX);
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

/// Blocks until a thread has registered a wait on `h`'s claim.
fn await_registered_wait(rt: &Rt, h: Incr<u64>) {
    let start = std::time::Instant::now();
    while !crate::waits::waited_on((rt.runtime_id, h.slot())) {
        assert!(
            start.elapsed() < JOIN_TIMEOUT,
            "no thread waited on node {}",
            h.slot()
        );
        thread::yield_now();
    }
}

/// **Readers of released claims**, at budget 1 on the landed-VFrame
/// shape in its chain form: a doom releases the claim of the node its
/// read wanted before the reading frame gets its placeholder, and
/// another thread parked there computes that node; another thread also
/// computes a discarded frame's node between rounds. Each is computed
/// once, by the other thread, as a node no one claims, and T's value is
/// the unbounded run's.
fn released_claim_read(point: fn(&Landed) -> ParkPoint) {
    let s = landed_shape(Rt::new(), false);
    s.rt.set_max_frames(1);
    let gate = Gate::new();
    hook(&s.rt, vec![(point(&s), gate.clone())]);
    gate.arm();
    let reader = {
        let (rt, t) = (s.rt.clone(), s.t);
        spawn(move || rt.get(t))
    };
    gate.wait_parked();
    assert_eq!(s.rt.state_of(s.x), NodeState::New, "X's claim is released");
    let alone = landed_shape(Rt::new(), false);
    alone.rt.set_max_frames(1);
    assert_eq!(s.rt.get(s.x), alone.rt.get(alone.x));
    let runs_before = s.x_runs.load(SeqCst);
    gate.release();
    let unbounded = landed_shape(Rt::new(), false);
    unbounded.rt.set_max_frames(u32::MAX);
    assert_eq!(reader.join(), unbounded.rt.get(unbounded.t));
    assert_eq!(
        s.x_runs.load(SeqCst),
        runs_before,
        "the reader never ran X once it had landed"
    );
    assert_eq!(s.rt.state_of(s.t), NodeState::Clean);
    s.rt.assert_invariant_i();
}

#[test]
fn shared_reader_of_a_doomed_reads_released_claim() {
    released_claim_read(|s| ParkPoint::AfterDoom(id(s.x)));
}

#[test]
fn shared_reader_of_a_discarded_frames_released_claim() {
    released_claim_read(|s| ParkPoint::BeforeEntry(id(s.y)));
}

/// **A waited-out discard is not a landing**, at budget 1: F, Clean at
/// s = 0, is Dirty after `set(s, 1)`. Reader B plans its walk of F and
/// stops before its claim; reader A claims F and dooms on its read of the
/// New query N, and B waits on A's claim. A's doom discards F as if its
/// run never happened and A's read goes on, so B claims F as planned
/// instead of revisiting it, which only a landing of A's claim warrants.
#[test]
fn shared_reader_that_waited_out_a_discarded_claim_keeps_its_plan() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    let s = rt.create_input(0_u64);
    let n = rt.create_query_with(|_: &Rt| 1_u64, replayable());
    let f = rt.create_query_with(
        move |rt: &Rt| match rt.get(s) {
            0 => 0,
            v => v * 10 + rt.get(n),
        },
        replayable(),
    );
    assert_eq!(rt.get(f), 0);
    rt.set(s, 1);
    rt.set_max_frames(1);
    let gates = Arc::new([
        ("b", ParkPoint::BeforeClaim(id(f)), Gate::new()),
        ("a", ParkPoint::AfterDoom(id(n)), Gate::new()),
        ("b", ParkPoint::ComputeWaits(id(f)), Gate::new()),
        ("a", ParkPoint::BeforeEntry(id(n)), Gate::new()),
    ]);
    for (_, _, gate) in gates.iter() {
        gate.arm();
    }
    let installed = rt.park_hook.set(Box::new({
        let gates = gates.clone();
        move |p| {
            for (name, point, gate) in gates.iter() {
                if *point == p && thread::current().name() == Some(name) {
                    gate.park();
                }
            }
        }
    }));
    assert!(installed.is_ok(), "park hook installed twice");
    let read = |name: &str| {
        let rt = rt.clone();
        thread::Builder::new()
            .name(name.into())
            .spawn(move || rt.get(f))
            .unwrap()
    };
    let b = read("b");
    gates[0].2.wait_parked();
    let a = read("a");
    gates[1].2.wait_parked();
    gates[0].2.release();
    gates[2].2.wait_parked();
    gates[2].2.release();
    gates[1].2.release();
    gates[3].2.wait_parked();
    let b = b.join().unwrap();
    gates[3].2.release();
    assert_eq!((a.join().unwrap(), b), (11, 11));
    assert_eq!(rt.wait_revisits.load(SeqCst), 0, "B revisited F");
    rt.assert_invariant_i();
}

/// **A `set` during a discard**, at budget 1: F, Clean at s = 0, is
/// claimed from Dirty after `set(s, 1)` and dooms on its read of a New
/// query N; parked at the doom, another thread sets s = 2 under F's
/// claim. F's discard landing keeps the set's dirtying, so the value the
/// read returns is one of the two revisions', and the next read, after
/// both sets, is the unbounded run's at s = 2.
#[test]
fn shared_set_during_a_discard() {
    fn shape(rt: &Arc<Rt>) -> (Incr<u64>, Incr<u64>, Incr<u64>) {
        rt.set_placeholder(0_u64);
        let s = rt.create_input(0_u64);
        let n = rt.create_query_with(|_: &Rt| 1_u64, replayable());
        let f = rt.create_query_with(
            move |rt: &Rt| match rt.get(s) {
                0 => 0,
                v => v * 10 + rt.get(n),
            },
            replayable(),
        );
        assert_eq!(rt.get(f), 0);
        (s, n, f)
    }
    let rt = Arc::new(Rt::new());
    let (s, n, f) = shape(&rt);
    rt.set(s, 1);
    rt.set_max_frames(1);
    let gate = Gate::new();
    hook(&rt, vec![(ParkPoint::AfterDoom(id(n)), gate.clone())]);
    gate.arm();
    let reader = {
        let rt = rt.clone();
        spawn(move || rt.get(f))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(f), NodeState::Computing);
    rt.set(s, 2);
    gate.release();
    let got = reader.join();
    assert!(got == 11 || got == 21, "{got}");
    rt.assert_invariant_i();

    let unbounded = Arc::new(Rt::new());
    unbounded.set_max_frames(u32::MAX);
    let (us, _, uf) = shape(&unbounded);
    unbounded.set(us, 2);
    assert_eq!(rt.get(f), unbounded.get(uf));
    assert_eq!(rt.state_of(f), NodeState::Clean);
    rt.assert_invariant_i();
}

/// A Fallback cycle F -> X -> Y -> F, all replayable: F's fallback is
/// 100, X reads Y while input b is 0 and is 50 otherwise, and Y,
/// unseeded, reads F. X parks on `gate` once it is armed.
struct Hit {
    rt: Arc<Rt>,
    f: Incr<u64>,
    x: Incr<u64>,
    y: Incr<u64>,
    b: Incr<u64>,
    gate: Gate,
}

fn hit_shape() -> Hit {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    let b = rt.create_input(0_u64);
    let gate = Gate::new();
    let fc: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let y = {
        let fc = fc.clone();
        rt.create_query_with(move |rt: &Rt| rt.get(*fc.get().unwrap()) + 1, replayable())
    };
    let x = {
        let gate = gate.clone();
        rt.create_query_with(
            move |rt: &Rt| {
                gate.park();
                if rt.get(b) == 0 {
                    rt.get(y) + 1
                } else {
                    50
                }
            },
            replayable(),
        )
    };
    let f = rt.create_query_with(
        move |rt: &Rt| rt.get(x) + 1,
        QueryOpts {
            cycle: crate::cycle::CycleStrategy::Fallback(Box::new(|_| 100)),
            replay: Replay::Allowed,
        },
    );
    fc.set(f).unwrap();
    Hit {
        rt,
        f,
        x,
        y,
        b,
        gate,
    }
}

/// The values of F, X and Y in a single-threaded unbounded run entered
/// at F, with b set first.
fn hit_unbounded(b: u64) -> (u64, u64, Option<u64>) {
    let u = hit_shape();
    u.rt.set_max_frames(u32::MAX);
    u.rt.set(u.b, b);
    let f = u.rt.get(u.f);
    let x = u.rt.get(u.x);
    let y = (b == 0).then(|| u.rt.get(u.y));
    (f, x, y)
}

/// **A virtual-hit claim racing another thread** (Virtual cycles step 4),
/// at budget 1: F and X are virtual frames when the driver takes Y, and
/// another thread claims X and parks inside X's closure. Y's read of F is
/// a virtual hit whose claim of X fails, so the claims taken roll back
/// and the read waits on X, registered. With `b` still 0, X's closure on
/// the other thread reads Y, which the first thread computes: the
/// registry closes the waits-for cycle, and the results are the single
/// run's. With `b` set to 1 first, X lands Clean without reading Y, the
/// second judgement finds a landed frame above F's and drops the chain,
/// and F's retry reads X Clean.
fn virtual_hit_claim_race(b: u64) {
    let s = hit_shape();
    s.rt.set_max_frames(1);
    let entry = Gate::new();
    hook(
        &s.rt,
        vec![(ParkPoint::BeforeEntry(id(s.y)), entry.clone())],
    );
    entry.arm();
    let first = {
        let (rt, f) = (s.rt.clone(), s.f);
        spawn(move || rt.get(f))
    };
    entry.wait_parked();
    s.rt.set(s.b, b);
    s.gate.arm();
    let second = {
        let (rt, x) = (s.rt.clone(), s.x);
        spawn(move || rt.get(x))
    };
    s.gate.wait_parked();
    entry.release();
    await_registered_wait(&s.rt, s.x);
    s.gate.release();
    let (f, x) = (first.join(), second.join());

    let (want_f, want_x, want_y) = hit_unbounded(b);
    assert_eq!((f, x), (want_f, want_x));
    if let Some(y) = want_y {
        assert_eq!(s.rt.get(s.y), y);
    }
    s.rt.assert_invariant_i();
    for h in [s.f, s.x, s.y] {
        assert!(!crate::waits::waited_on((s.rt.runtime_id, h.slot())));
    }
}

#[test]
fn shared_virtual_hit_claim_race_closes_a_waits_for_cycle() {
    virtual_hit_claim_race(0);
}

#[test]
fn shared_virtual_hit_claim_race_then_a_landed_frame() {
    virtual_hit_claim_race(1);
}

/// X (no strategy) keeps a record of the group {X, M} and M (Fallback
/// 100) has left it, as a single-node landing of M leaves it. Once k is
/// 1, X reads D and then M, M reads X and then W (Fallback 500), and W
/// reads X. T reads M. All are replayable over a `u64` placeholder.
struct Member<C: Cells> {
    rt: Arc<Runtime<C>>,
    t: Incr<u64>,
    x: Incr<u64>,
    m: Incr<u64>,
    w: Incr<u64>,
    d: Incr<u64>,
}

fn member_shape<C: Cells>(frames: u32) -> Member<C> {
    let rt = Arc::new(Runtime::<C>::new());
    rt.set_max_frames(frames);
    rt.set_placeholder(0_u64);
    let k = rt.create_input(0_u64);
    let mc: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let wc: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let d = rt.create_query_with(move |rt: &Runtime<C>| rt.get(k) * 3, replayable());
    let x = {
        let mc = mc.clone();
        rt.create_query_with(
            move |rt: &Runtime<C>| match rt.get(k) {
                0 => 1,
                _ => rt.get(d) + rt.get(*mc.get().unwrap()) + 1,
            },
            replayable(),
        )
    };
    let fallback = |v: u64| QueryOpts {
        cycle: CycleStrategy::Fallback(Box::new(move |_: &Cycle| v)),
        replay: Replay::Allowed,
    };
    let w = rt.create_query_with(
        move |rt: &Runtime<C>| match rt.get(k) {
            0 => 0,
            _ => rt.get(x) + 1000,
        },
        fallback(500),
    );
    let m = {
        let wc = wc.clone();
        rt.create_query_with(
            move |rt: &Runtime<C>| match rt.get(k) {
                0 => 0,
                _ => rt.get(x) + rt.get(*wc.get().unwrap()),
            },
            fallback(100),
        )
    };
    mc.set(m).unwrap();
    wc.set(w).unwrap();
    let t = rt.create_query_with(move |rt: &Runtime<C>| rt.get(m) + 7, replayable());
    assert_eq!((rt.get(t), rt.get(x), rt.get(w)), (7, 1, 0));
    rt.force_group_for_test(&[x.erase(), m.erase()]);
    let (node, flags) = rt.nodes.get_with_flags(m.slot());
    rt.leave_group(id(m), node, flags, NodeState::Clean);
    rt.set(k, 1);
    Member { rt, t, x, m, w, d }
}

impl<C: Cells> Member<C> {
    fn in_group(&self, h: Incr<u64>) -> bool {
        C::u8_load_relaxed(self.rt.nodes.get_with_flags(h.slot()).1) & IN_GROUP != 0
    }

    fn reads(&self, order: &[Incr<u64>]) -> Vec<Result<u64, String>> {
        order.iter().map(|&h| outcome(|| self.rt.get(h))).collect()
    }
}

/// The nodes a run reads first, in order.
type Order<C> = fn(&Member<C>) -> Vec<Incr<u64>>;

/// The outcomes of `get` of each node in `order` and then of every node,
/// at budget `frames`, with whether some run doomed on D.
fn member_reads<C: Cells>(frames: u32, order: Order<C>) -> (Vec<Result<u64, String>>, bool) {
    let s = member_shape::<C>(frames);
    let got = [s.reads(&order(&s)), s.reads(&[s.t, s.w, s.m, s.x, s.d])].concat();
    s.rt.assert_invariant_i();
    let doomed_d =
        s.rt.dooms
            .lock()
            .unwrap()
            .iter()
            .any(|d| d.node == s.d.slot());
    (got, doomed_d)
}

/// At budget 2, T's run dooms on X's read of D, and X, which computes
/// alone after its group verify failed on M, is discarded with M and T.
/// Every order of reads gives the unbounded run's outcomes at every
/// budget.
fn member_discard_matches_unbounded<C: Cells>() {
    let orders: [Order<C>; 3] = [|s| vec![s.t, s.w], |s| vec![s.w, s.t], |s| vec![s.x, s.t]];
    for order in orders {
        let (want, _) = member_reads::<C>(u32::MAX, order);
        for frames in 1..=3 {
            let (got, _) = member_reads::<C>(frames, order);
            assert_eq!(got, want, "budget {frames}");
        }
    }
    assert!(
        member_reads::<C>(2, orders[0]).1,
        "T's run never doomed on D"
    );
}

#[test]
fn local_discarded_member_matches_unbounded() {
    member_discard_matches_unbounded::<crate::cells::Local>();
}

#[test]
fn shared_discarded_member_matches_unbounded() {
    member_discard_matches_unbounded::<Shared>();
}

/// The member shape at budget 2: T's reader parks before D's entry, with
/// X discarded. Another thread then reads W, which reads X, and must
/// verify X's group, as a run that reads W first does: computed alone, X
/// reads M, which reads X on the stack, a cycle error that neither order
/// of the two reads raises.
#[test]
fn shared_reader_of_a_discarded_member_verifies_its_group() {
    let s = member_shape::<Shared>(2);
    let entry = Gate::new();
    hook(
        &s.rt,
        vec![(ParkPoint::BeforeEntry(id(s.d)), entry.clone())],
    );
    entry.arm();
    let reader = {
        let (rt, t) = (s.rt.clone(), s.t);
        spawn(move || outcome(|| rt.get(t)))
    };
    entry.wait_parked();
    assert_eq!(s.rt.state_of(s.x), NodeState::Dirty, "X is discarded");
    let kept = s.in_group(s.x);
    let w = {
        let (rt, w) = (s.rt.clone(), s.w);
        spawn(move || outcome(|| rt.get(w)))
    }
    .join();
    entry.release();
    let t = reader.join();
    let got = [vec![t, w], s.reads(&[s.t, s.w, s.m, s.x, s.d])].concat();
    s.rt.assert_invariant_i();

    let (t_first, _) = member_reads::<Shared>(u32::MAX, |s| vec![s.t, s.w]);
    let (w_first, _) = member_reads::<Shared>(u32::MAX, |s| vec![s.w, s.t]);
    let w_first = [
        vec![w_first[1].clone(), w_first[0].clone()],
        w_first[2..].to_vec(),
    ]
    .concat();
    assert!(
        got == t_first || got == w_first,
        "X in its group at the discard: {kept}\n got {got:?}\n T first {t_first:?}\n W first {w_first:?}"
    );
    assert!(kept, "the discard took X out of its group");
}

/// F (Fallback 100) reads X, X reads Y and then Z, Y reads F while input
/// s is 0, and Z sets s to 1 on its first run. F is New with a record of
/// the group {F, Q}, as a walk leaves a Cyclic member. At budget 1 Z's
/// set comes before F's pop, whose episode dissolves, and the driver
/// parks before the level-0 entry claims F again. F's claim, taken from
/// New, lands New as if never taken, and another thread's read then gets
/// the unbounded run's value.
#[test]
fn shared_dissolved_root_lands_as_before_its_claim() {
    let run = |frames: u32, park: bool| {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(frames);
        let s = rt.create_input(0_u64);
        let gate = Gate::new();
        let z = {
            let gate = gate.clone();
            let ran = Arc::new(AtomicU8::new(0));
            rt.create_query_with(
                move |rt: &Rt| {
                    if ran.swap(1, SeqCst) == 0 {
                        if park {
                            gate.arm();
                        }
                        rt.set(s, 1);
                    }
                    10
                },
                replayable(),
            )
        };
        let fc: Arc<OnceLock<Incr<u64>>> = Arc::default();
        let y = {
            let fc = fc.clone();
            rt.create_query_with(
                move |rt: &Rt| match rt.get(s) {
                    0 => rt.get(*fc.get().unwrap()) + 1,
                    _ => 5,
                },
                replayable(),
            )
        };
        let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + rt.get(z), replayable());
        let f = rt.create_query_with(
            move |rt: &Rt| rt.get(x),
            QueryOpts {
                cycle: CycleStrategy::Fallback(Box::new(|_: &Cycle| 100)),
                replay: Replay::Allowed,
            },
        );
        fc.set(f).unwrap();
        let q = rt.create_query(|_: &Rt| 7_u64);
        assert_eq!(rt.get(q), 7);
        rt.force_group_for_test(&[f.erase(), q.erase()]);
        hook(&rt, vec![(ParkPoint::BeforeClaim(id(f)), gate.clone())]);
        let reader = {
            let rt = rt.clone();
            spawn(move || outcome(|| rt.get(f)))
        };
        let mut gap = None;
        if park {
            gate.wait_parked();
            let landed = rt.state_of(f);
            let other = {
                let rt = rt.clone();
                spawn(move || outcome(|| rt.get(f)))
            };
            gap = Some((landed, other.join()));
            gate.release();
        }
        let got = reader.join();
        rt.assert_invariant_i();
        (got, gap)
    };
    let (want, _) = run(u32::MAX, false);
    assert_eq!(want, Ok(15));
    let (got, gap) = run(1, true);
    let (state, other) = gap.expect("the driver parked");
    assert_eq!((got, other), (want.clone(), want));
    assert_eq!(
        state,
        NodeState::New,
        "F's claim taken from New landed Dirty"
    );
}

/// What a closure does with the panic of a read below it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Panics {
    /// Lets it out.
    Out,
    /// Catches it and panics with its own payload.
    Own,
    /// Catches it, reads the node again and lets that read's panic out.
    Twice,
}

fn panicking_reader(rt: &Rt, read: Incr<u64>, panics: Panics, own: &'static str) -> Incr<u64> {
    rt.create_query_with(
        move |rt: &Rt| match panics {
            Panics::Out => rt.get(read) + 1,
            Panics::Own => {
                let _ = catch_unwind(AssertUnwindSafe(|| rt.get(read)));
                panic!("{own}")
            }
            Panics::Twice => {
                let _ = catch_unwind(AssertUnwindSafe(|| rt.get(read)));
                rt.get(read) + 1
            }
        },
        replayable(),
    )
}

/// X reads Y, which panics "y boom", and X does `x` with that panic;
/// with `w`, W reads X and does `w` with its panic. One thread's
/// `get(top)`, W or else X, dooms the first frame whose read is at the
/// budget, one frame per node, so Y's entry lands Failed and its payload
/// waits for X's virtual frame. The driver parks before the next entry
/// while another thread reads `other_reads` (X or W), which reads Y
/// Failed and lands what it computed Failed. Each read's outcome must be
/// one of a serial order's, and no provenance record outlives the reads.
fn landed_failed_over_a_delivery(
    x: Panics,
    w: Option<Panics>,
    other_reads_w: bool,
) -> (Result<u64, String>, Result<u64, String>) {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1 + u32::from(w.is_some()));
    let gate = Gate::new();
    let y = {
        let gate = gate.clone();
        rt.create_query_with(
            move |_: &Rt| -> u64 {
                gate.arm();
                panic!("y boom")
            },
            replayable(),
        )
    };
    let xn = panicking_reader(&rt, y, x, "x own");
    let wn = w.map(|w| panicking_reader(&rt, xn, w, "w own"));
    let top = wn.unwrap_or(xn);
    let other_reads = if other_reads_w { top } else { xn };
    hook(&rt, vec![(ParkPoint::BeforeEntry(id(top)), gate.clone())]);
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(top)))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(y), NodeState::Failed);
    let other = outcome(|| rt.get(other_reads));
    assert_eq!(rt.state_of(other_reads), NodeState::Failed);
    gate.release();
    let first = first.join();
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
    (first, other)
}

fn failed(r: &Result<u64, String>) -> bool {
    r.as_ref()
        .is_err_and(|m| m.contains("last compute panicked"))
}

/// Class A of the two-thread all-seeds scan (seed 0x172), at budget 1:
/// the other thread's X let Y's failed read out, so only the order with
/// this thread first is serial, and its retry of X takes the payload.
#[test]
fn shared_vframe_landed_failed_keeps_the_payload() {
    let (first, other) = landed_failed_over_a_delivery(Panics::Out, None, false);
    assert!(failed(&other), "{other:?}");
    assert_eq!(first, Err("y boom".to_string()), "the payload was dropped");
}

/// X's own panic makes only the other thread's order serial.
#[test]
fn shared_vframe_landed_failed_by_its_own_panic() {
    let (first, other) = landed_failed_over_a_delivery(Panics::Own, None, false);
    assert_eq!(other, Err("x own".to_string()));
    assert!(failed(&first), "{first:?}");
}

/// X lets out its second read of Y, not the one the payload is for: in
/// either order both reads panic "last compute panicked".
#[test]
fn shared_vframe_landed_failed_by_a_later_read() {
    let (first, other) = landed_failed_over_a_delivery(Panics::Twice, None, false);
    assert!(failed(&other), "{other:?}");
    assert!(failed(&first), "{first:?}");
}

/// At budget 2 X's virtual frame sits above W's, and W's retry reads X
/// natively.
#[test]
fn shared_vframe_landed_failed_under_a_native_read() {
    let (first, other) = landed_failed_over_a_delivery(Panics::Out, Some(Panics::Out), false);
    assert!(failed(&other), "{other:?}");
    assert_eq!(first, Err("y boom".to_string()));
    let (first, other) = landed_failed_over_a_delivery(Panics::Own, Some(Panics::Out), false);
    assert_eq!(other, Err("x own".to_string()));
    assert!(failed(&first), "{first:?}");
}

/// The other thread lands both W and X, the frames of a chain, and only
/// a landing of W that let the failure out allows this thread first.
#[test]
fn shared_vframe_chain_landed_failed() {
    let (first, other) = landed_failed_over_a_delivery(Panics::Out, Some(Panics::Out), true);
    assert!(failed(&other), "{other:?}");
    assert_eq!(first, Err("y boom".to_string()));
    let (first, other) = landed_failed_over_a_delivery(Panics::Out, Some(Panics::Own), true);
    assert_eq!(other, Err("w own".to_string()));
    assert!(failed(&first), "{first:?}");
}

/// R reads F, then N; F and N both read V, which reads Y. At budget 3
/// V's read of Y dooms R, F and V, and R's read of N after F's
/// placeholder pushes N as a speculative entry. The driver parks before
/// it while the other thread lands V Failed over Y. N's run reads V at
/// V's frame position: the payload waiting there is owed to the real
/// retry's read of V through F, not to N.
#[test]
fn speculative_entry_leaves_a_landed_payload_for_the_real_retry() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(3);
    let gate = Gate::new();
    let y = {
        let gate = gate.clone();
        rt.create_query_with(
            move |_: &Rt| -> u64 {
                gate.arm();
                panic!("y boom")
            },
            replayable(),
        )
    };
    let v = panicking_reader(&rt, y, Panics::Out, "");
    let f = panicking_reader(&rt, v, Panics::Out, "");
    let n = panicking_reader(&rt, v, Panics::Out, "");
    let r = rt.create_query_with(move |rt: &Rt| rt.get(f) + rt.get(n), replayable());
    hook(&rt, vec![(ParkPoint::BeforeEntry(id(n)), gate.clone())]);
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(r)))
    };
    gate.wait_parked();
    let other = outcome(|| rt.get(v));
    assert_eq!(rt.state_of(v), NodeState::Failed);
    gate.release();
    let first = first.join();
    rt.assert_invariant_i();
    assert!(failed(&other), "{other:?}");
    assert_eq!(first, Err("y boom".to_string()), "the payload was taken");
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// X is still computing on the other thread when the retry of X claims
/// it, so the retry waits and then reads X Failed.
#[test]
fn shared_vframe_landed_failed_while_the_retry_waits() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let (gate, inside, waits) = (Gate::new(), Gate::new(), Gate::new());
    let y = {
        let gate = gate.clone();
        rt.create_query_with(
            move |_: &Rt| -> u64 {
                gate.arm();
                panic!("y boom")
            },
            replayable(),
        )
    };
    let x = {
        let inside = inside.clone();
        rt.create_query_with(
            move |rt: &Rt| {
                inside.park();
                rt.get(y) + 1
            },
            replayable(),
        )
    };
    hook(
        &rt,
        vec![
            (ParkPoint::BeforeEntry(id(x)), gate.clone()),
            (ParkPoint::ComputeWaits(id(x)), waits.clone()),
        ],
    );
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(x)))
    };
    gate.wait_parked();
    inside.arm();
    let other = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(x)))
    };
    inside.wait_parked();
    waits.arm();
    gate.release();
    waits.wait_parked();
    inside.release();
    let other = other.join();
    waits.release();
    let first = first.join();
    assert!(failed(&other), "{other:?}");
    assert_eq!(first, Err("y boom".to_string()));
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// The other thread's read of X dooms too, before Y lands, so its Failed
/// read of Y is its Y entry's compute, delivered to its own retry of X,
/// which lets it out.
#[test]
fn shared_vframe_landed_failed_by_a_replayed_read() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let (gate, other_entry) = (Gate::new(), Gate::new());
    let y = {
        let gate = gate.clone();
        rt.create_query_with(
            move |_: &Rt| -> u64 {
                gate.arm();
                panic!("y boom")
            },
            replayable(),
        )
    };
    let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 1, replayable());
    hook(
        &rt,
        vec![
            (ParkPoint::BeforeEntry(id(x)), gate.clone()),
            (ParkPoint::BeforeEntry(id(y)), other_entry.clone()),
        ],
    );
    other_entry.arm();
    let other = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(x)))
    };
    other_entry.wait_parked();
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(x)))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(y), NodeState::Failed);
    other_entry.release();
    let other = other.join();
    assert_eq!(rt.state_of(x), NodeState::Failed);
    gate.release();
    let first = first.join();
    assert!(failed(&other), "{other:?}");
    assert_eq!(first, Err("y boom".to_string()));
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// Y reads Z, which panics "z boom", so the other thread's read of Y
/// Failed walks to Z Failed first: that panic is still the failure of
/// X's read of Y, which X lets out.
#[test]
fn shared_vframe_landed_failed_by_a_walk_into_its_dep() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let gate = Gate::new();
    let z = rt.create_query_with(|_: &Rt| -> u64 { panic!("z boom") }, replayable());
    let y = {
        let gate = gate.clone();
        rt.create_query_with(
            move |rt: &Rt| {
                gate.arm();
                rt.get(z) + 1
            },
            replayable(),
        )
    };
    let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 1, replayable());
    hook(&rt, vec![(ParkPoint::BeforeEntry(id(x)), gate.clone())]);
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(x)))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(y), NodeState::Failed);
    let other = outcome(|| rt.get(x));
    assert_eq!(rt.state_of(x), NodeState::Failed);
    gate.release();
    let first = first.join();
    assert!(failed(&other), "{other:?}");
    assert_eq!(first, Err("z boom".to_string()));
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// The other thread's X lets Y's failed read out, a set flips X and Y
/// Dirty, and the next run of X lands it Failed by its own panic. The other
/// thread observed Y's failure, so only the order with this thread first
/// is serial, and the retry takes the payload over the later landing.
#[test]
fn shared_vframe_relanded_failed_by_its_own_panic() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let gate = Gate::new();
    let i = rt.create_input(0_u64);
    let y = {
        let gate = gate.clone();
        rt.create_query_with(
            move |rt: &Rt| -> u64 {
                if rt.get(i) == 1 {
                    return 7;
                }
                gate.arm();
                panic!("y boom")
            },
            replayable(),
        )
    };
    let x = rt.create_query_with(
        move |rt: &Rt| {
            if rt.get(i) == 1 {
                panic!("x own");
            }
            rt.get(y) + 1
        },
        replayable(),
    );
    hook(&rt, vec![(ParkPoint::BeforeEntry(id(x)), gate.clone())]);
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(x)))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(y), NodeState::Failed);
    let other = outcome(|| rt.get(x));
    assert!(failed(&other), "{other:?}");
    rt.set(i, 1);
    let again = outcome(|| rt.get(x));
    assert_eq!(again, Err("x own".to_string()));
    assert_eq!(rt.state_of(x), NodeState::Failed);
    gate.release();
    let first = first.join();
    assert_eq!(first, Err("y boom".to_string()));
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// The other thread's X lets Y's failed read out and a set flips X Dirty
/// before the retry reads X. The other thread observed Y's failure, so
/// only the order with this thread first is serial: the retry takes the
/// payload instead of computing X, which the next read computes.
#[test]
fn shared_vframe_dirtied_after_it_let_the_failure_out() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let gate = Gate::new();
    let i = rt.create_input(0_u64);
    let y = {
        let gate = gate.clone();
        rt.create_query_with(
            move |rt: &Rt| -> u64 {
                if rt.get(i) == 1 {
                    return 7;
                }
                gate.arm();
                panic!("y boom")
            },
            replayable(),
        )
    };
    let x = rt.create_query_with(
        move |rt: &Rt| {
            if rt.get(i) == 1 {
                panic!("x own");
            }
            rt.get(y) + 1
        },
        replayable(),
    );
    hook(&rt, vec![(ParkPoint::BeforeEntry(id(x)), gate.clone())]);
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(x)))
    };
    gate.wait_parked();
    let other = outcome(|| rt.get(x));
    assert!(failed(&other), "{other:?}");
    rt.set(i, 1);
    gate.release();
    let first = first.join();
    assert_eq!(first, Err("y boom".to_string()));
    assert_eq!(outcome(|| rt.get(x)), Err("x own".to_string()));
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// As `dirtied_after_it_let_the_failure_out`, but the retry loads X New
/// and parks before its claim while the other thread computes X, letting
/// Y's failure out, and a set flips X Dirty, so the claim succeeds on that
/// Dirty. The retry gives the claim back and takes the payload.
#[test]
fn shared_vframe_dirtied_while_the_retry_claims() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let (gate, claim) = (Gate::new(), Gate::new());
    let i = rt.create_input(0_u64);
    let y = {
        let gate = gate.clone();
        rt.create_query_with(
            move |rt: &Rt| -> u64 {
                if rt.get(i) == 1 {
                    return 7;
                }
                gate.arm();
                panic!("y boom")
            },
            replayable(),
        )
    };
    let x = rt.create_query_with(
        move |rt: &Rt| {
            if rt.get(i) == 1 {
                panic!("x own");
            }
            rt.get(y) + 1
        },
        replayable(),
    );
    hook(
        &rt,
        vec![
            (ParkPoint::BeforeEntry(id(x)), gate.clone()),
            (ParkPoint::BeforeClaim(id(x)), claim.clone()),
        ],
    );
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(x)))
    };
    gate.wait_parked();
    claim.arm();
    gate.release();
    claim.wait_parked();
    let other = outcome(|| rt.get(x));
    assert!(failed(&other), "{other:?}");
    rt.set(i, 1);
    claim.release();
    let first = first.join();
    assert_eq!(first, Err("y boom".to_string()));
    assert_eq!(rt.state_of(x), NodeState::Dirty);
    assert_eq!(outcome(|| rt.get(x)), Err("x own".to_string()));
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// W reads X, which reads Y. The other thread's X lets Y's failed read
/// out, and a set and a read reland X Clean before the retry of X's frame.
/// Only the order with this thread first is serial, so the retry takes the
/// payload over the Clean X, and W lets it out.
#[test]
#[ignore = "W's retry reads X Clean on the fast path and never takes the delivery"]
fn shared_vframe_relanded_clean_below_its_reader() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let gate = Gate::new();
    let i = rt.create_input(0_u64);
    let y = {
        let gate = gate.clone();
        rt.create_query_with(
            move |rt: &Rt| -> u64 {
                if rt.get(i) == 1 {
                    return 7;
                }
                gate.arm();
                panic!("y boom")
            },
            replayable(),
        )
    };
    let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 1, replayable());
    let w = rt.create_query_with(move |rt: &Rt| rt.get(x) + 1, replayable());
    hook(&rt, vec![(ParkPoint::BeforeEntry(id(x)), gate.clone())]);
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(w)))
    };
    gate.wait_parked();
    let other = outcome(|| rt.get(x));
    assert!(failed(&other), "{other:?}");
    rt.set(i, 1);
    assert_eq!(outcome(|| rt.get(x)), Ok(8));
    gate.release();
    let first = first.join();
    assert_eq!(first, Err("y boom".to_string()));
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// As `keeps_the_payload`, but after the retry's read of X found the
/// payload and loaded X's generation, another thread deletes X and
/// recycles its slot into Z, landed Failed by its own panic, before the
/// read takes the record lock. The other thread's read let the failure
/// out, so only the order with this thread first is serial, where the
/// retry takes the payload.
#[test]
fn shared_vframe_failed_read_of_a_recycled_slot() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let (gate, recheck) = (Gate::new(), Gate::new());
    let y = {
        let gate = gate.clone();
        rt.create_query_with(
            move |_: &Rt| -> u64 {
                gate.arm();
                panic!("y boom")
            },
            replayable(),
        )
    };
    let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 1, replayable());
    hook(
        &rt,
        vec![
            (ParkPoint::BeforeEntry(id(x)), gate.clone()),
            (ParkPoint::FailedReadBeforeRecheck(id(x)), recheck.clone()),
        ],
    );
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(x)))
    };
    gate.wait_parked();
    let other = outcome(|| rt.get(x));
    assert!(failed(&other), "{other:?}");
    recheck.arm();
    gate.release();
    recheck.wait_parked();
    rt.delete_node(x);
    let z = rt.create_query_with(|_: &Rt| -> u64 { panic!("z boom") }, replayable());
    assert_eq!(z.slot(), x.slot(), "Z reuses X's slot");
    assert_eq!(outcome(|| rt.get(z)), Err("z boom".to_string()));
    recheck.release();
    let first = first.join();
    assert_eq!(first, Err("y boom".to_string()));
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// Records whether it was dropped while its thread was unwinding, where a
/// panicking user `Drop` aborts.
struct DropWitness(Arc<AtomicU8>);

impl Drop for DropWitness {
    fn drop(&mut self) {
        let at = if thread::panicking() { 2 } else { 1 };
        self.0.store(at, SeqCst);
    }
}

/// The retry's first attempt at X reads Z, which is deleted under it, so
/// its publish is Stale and the attempt's bag carries an item into the
/// next. Meanwhile the other thread's X lets Y's failure out, so the next
/// attempt takes the payload, and the bag drains before it is raised.
#[test]
fn shared_vframe_owed_payload_drains_a_stale_retrys_bag_first() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let (inside, stale) = (Gate::new(), Gate::new());
    let z = rt.create_input(1_u64);
    let y = rt.create_query_with(|_: &Rt| -> u64 { panic!("y boom") }, replayable());
    let runs = Arc::new(AtomicU32::new(0));
    let x = {
        let (inside, runs) = (inside.clone(), Arc::clone(&runs));
        rt.create_query_with(
            move |rt: &Rt| {
                if runs.fetch_add(1, SeqCst) == 1 {
                    let v = rt.get(z);
                    inside.park();
                    return v;
                }
                rt.get(y) + 1
            },
            replayable(),
        )
    };
    hook(&rt, vec![(ParkPoint::StaleRetry(id(x)), stale.clone())]);
    let dropped = Arc::new(AtomicU8::new(0));
    let planted = Arc::clone(&dropped);
    let installed = rt.bag_hook.set(Box::new(move |p| {
        (p == ParkPoint::StaleRetry(id(x)))
            .then(|| Box::new(DropWitness(Arc::clone(&planted))) as crate::deferred::ErasedDrop)
    }));
    assert!(installed.is_ok(), "bag hook installed twice");
    inside.arm();
    stale.arm();
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(x)))
    };
    inside.wait_parked();
    rt.delete_node(z);
    inside.release();
    stale.wait_parked();
    let other = outcome(|| rt.get(x));
    assert!(failed(&other), "{other:?}");
    stale.release();
    let first = first.join();
    assert_eq!(first, Err("y boom".to_string()));
    assert_eq!(
        dropped.load(SeqCst),
        1,
        "the bag dropped its item while unwinding"
    );
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// R reads X, which reads Y only once `sel` is 1, so a set of `sel` makes
/// R's walk compute X as a walk entry, whose read of Y dooms it at budget
/// 1. Y's entry panics "y boom", parked before X's retry on `gate`.
struct WalkedFrame {
    rt: Arc<Rt>,
    i: Incr<u64>,
    x: Incr<u64>,
    r: Incr<u64>,
    gate: Gate,
}

fn walked_frame() -> WalkedFrame {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let gate = Gate::new();
    let (sel, i) = (rt.create_input(0_u64), rt.create_input(0_u64));
    let y = {
        let gate = gate.clone();
        rt.create_query_with(
            move |rt: &Rt| -> u64 {
                if rt.get(i) == 1 {
                    return 7;
                }
                gate.arm();
                panic!("y boom")
            },
            replayable(),
        )
    };
    let x = rt.create_query_with(
        move |rt: &Rt| {
            if rt.get(sel) == 0 {
                return 1;
            }
            rt.get(y) + 1
        },
        replayable(),
    );
    let r = rt.create_query_with(move |rt: &Rt| rt.get(x) + 1, replayable());
    assert_eq!(rt.get(r), 2);
    rt.set(sel, 1);
    hook(&rt, vec![(ParkPoint::BeforeEntry(id(x)), gate.clone())]);
    WalkedFrame { rt, i, x, r, gate }
}

/// The other thread's X lets Y's failed read out before the walk's retry
/// of X, which finds X Failed. Only the order with this thread's whole
/// read first is serial: its walk contains "y boom" on X, R's rerun takes
/// it, and the other thread reads X Failed.
#[test]
#[ignore = "a walked retry skips the owed check; T8-vframe-owed-walked-retry"]
fn shared_walked_vframe_failed_by_a_landing_that_let_the_failure_out() {
    let WalkedFrame { rt, x, r, gate, .. } = walked_frame();
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(r)))
    };
    gate.wait_parked();
    let other = outcome(|| rt.get(x));
    assert!(failed(&other), "{other:?}");
    gate.release();
    let first = first.join();
    assert_eq!(first, Err("y boom".to_string()));
    assert!(failed(&outcome(|| rt.get(x))));
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
    assert_eq!(rt.contained_payloads(), 0);
}

/// As `failed_by_a_landing_that_let_the_failure_out`, but a set that
/// mends Y flips X Dirty before the retry, which claims X. The serial
/// outcome is the same, and the set leaves X Dirty for the next read.
#[test]
#[ignore = "a walked retry skips the owed check; T8-vframe-owed-walked-retry"]
fn shared_walked_vframe_dirtied_after_it_let_the_failure_out() {
    let WalkedFrame { rt, i, x, r, gate } = walked_frame();
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(r)))
    };
    gate.wait_parked();
    let other = outcome(|| rt.get(x));
    assert!(failed(&other), "{other:?}");
    rt.set(i, 1);
    gate.release();
    let first = first.join();
    assert_eq!(first, Err("y boom".to_string()));
    assert_eq!(outcome(|| rt.get(r)), Ok(9));
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// T reads F, and F (Fallback 100), X and Y read each other in a
/// cycle. At budget 1 Y's read of F is a virtual hit, and F's virtual
/// frame pops as the root of an episode in which F panics "f boom", so
/// the episode lands F Failed and delivers the payload to T's virtual
/// frame, which holds no claim: the landing publishes a record first.
/// The other thread's T then reads F Failed through a walk that computes
/// X and Y, which the record's match does not cover, so only the record
/// and the reads' end are checked here.
#[test]
fn shared_vframe_landed_failed_over_an_episodes_delivery() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let gate = Gate::new();
    let fc: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let y = {
        let fc = fc.clone();
        rt.create_query_with(move |rt: &Rt| rt.get(*fc.get().unwrap()) + 1, replayable())
    };
    let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 1, replayable());
    let f_runs = Arc::new(AtomicU32::new(0));
    let f = {
        let (gate, f_runs) = (gate.clone(), f_runs.clone());
        rt.create_query_with(
            move |rt: &Rt| {
                let v = rt.get(x) + 1;
                // The doomed run and the stand-in come first; the episode's
                // run panics.
                if f_runs.fetch_add(1, SeqCst) >= 2 {
                    gate.arm();
                    panic!("f boom")
                }
                v
            },
            QueryOpts {
                cycle: CycleStrategy::Fallback(Box::new(|_| 100)),
                replay: Replay::Allowed,
            },
        )
    };
    fc.set(f).unwrap();
    let t = rt.create_query_with(move |rt: &Rt| rt.get(f) + 1, replayable());
    hook(&rt, vec![(ParkPoint::BeforeEntry(id(t)), gate.clone())]);
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(t)))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(f), NodeState::Failed);
    assert_eq!(
        rt.provenance_records(),
        1,
        "the episode published no record"
    );
    let other = outcome(|| rt.get(t));
    assert!(failed(&other), "{other:?}");
    gate.release();
    let first = first.join();
    assert!(first.is_err(), "{first:?}");
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// As `keeps_the_payload`, but the other thread reads R, which reads X,
/// and X reads R back before it reads Y, so X is a member of R's
/// episode. X's member run reads Y Failed and lets that panic out, which
/// aborts the episode, and X lands Failed at the release of its claim.
/// That landing let the failure out as X's own run's does, so the retry
/// of X takes the payload.
#[test]
fn shared_vframe_landed_failed_by_an_episode_member() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let gate = Gate::new();
    let in_cycle = Arc::new(AtomicU8::new(0));
    let y = {
        let gate = gate.clone();
        rt.create_query_with(
            move |_: &Rt| -> u64 {
                gate.arm();
                panic!("y boom")
            },
            replayable(),
        )
    };
    let rc: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let x = {
        let (rc, in_cycle) = (rc.clone(), in_cycle.clone());
        rt.create_query_with(
            move |rt: &Rt| {
                // The discovery run of R's cycle, which reads R as the
                // cycle closes, returns without reading Y; the episode's
                // runs read it.
                if in_cycle.load(SeqCst) > 0
                    && rt.get(*rc.get().unwrap()) > 0
                    && in_cycle.fetch_add(1, SeqCst) == 1
                {
                    return 0;
                }
                rt.get(y) + 1
            },
            replayable(),
        )
    };
    let r = rt.create_query_with(
        move |rt: &Rt| rt.get(x) + 1,
        QueryOpts {
            cycle: CycleStrategy::Fallback(Box::new(|_| 100)),
            replay: Replay::Allowed,
        },
    );
    rc.set(r).unwrap();
    hook(&rt, vec![(ParkPoint::BeforeEntry(id(x)), gate.clone())]);
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(x)))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(y), NodeState::Failed);
    in_cycle.store(1, SeqCst);
    let other = outcome(|| rt.get(r));
    assert!(failed(&other), "{other:?}");
    assert_eq!(rt.state_of(x), NodeState::Failed);
    gate.release();
    let first = first.join();
    assert_eq!(first, Err("y boom".to_string()), "the payload was dropped");
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// The position and path of an untaken delivery for a frame of X.
type Held = Option<(u32, u64)>;

/// W reads X, which reads Y, which panics "y boom"; once `switch` is set,
/// W first reads Z, which reads Z2. At budget 2 this thread's `get(W)`
/// dooms W and X, Y's entry lands Failed, and the driver parks before
/// W's retry with Y's payload held for X's frame at position 1. The other
/// thread lands X Failed, then sets the switch and reads W, whose read of
/// Z dooms W, so W's retry reads X Failed in a replay of its own at the
/// same position and path as the held payload's frame. A context is one
/// thread's, so that read raises "last compute panicked" and the payload
/// is still held when this thread goes on.
#[test]
fn shared_failed_read_at_an_equal_path_leaves_another_threads_delivery() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(2);
    let gate = Gate::new();
    let y_ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let y = {
        let (gate, y_ran) = (gate.clone(), y_ran.clone());
        rt.create_query_with(
            move |_: &Rt| -> u64 {
                y_ran.store(true, SeqCst);
                gate.arm();
                panic!("y boom")
            },
            replayable(),
        )
    };
    let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 1, replayable());
    let z2 = rt.create_query_with(|_: &Rt| 1_u64, replayable());
    let z = rt.create_query_with(move |rt: &Rt| rt.get(z2), replayable());
    let switch = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let w = {
        let switch = switch.clone();
        rt.create_query_with(
            move |rt: &Rt| {
                if switch.load(SeqCst) {
                    rt.get(z);
                }
                rt.get(x) + 1
            },
            replayable(),
        )
    };
    let held = |x: Incr<u64>| {
        crate::replay::with(|c| {
            c.iter().flat_map(|ctx| ctx.pending.iter()).find_map(|d| {
                (!d.taken && d.key.node == x.slot()).then_some((d.key.cidx, d.key.path))
            })
        })
    };
    let seen: Arc<Mutex<Vec<Held>>> = Arc::default();
    {
        let (gate, seen) = (gate.clone(), seen.clone());
        let installed = rt.park_hook.set(Box::new(move |p| {
            if p == ParkPoint::BeforeEntry(id(w)) && y_ran.swap(false, SeqCst) {
                seen.lock().unwrap().push(held(x));
                gate.park();
                seen.lock().unwrap().push(held(x));
            }
        }));
        assert!(installed.is_ok(), "park hook installed twice");
    }
    let first = {
        let rt = rt.clone();
        spawn(move || outcome(|| rt.get(w)))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(y), NodeState::Failed);
    let landed = outcome(|| rt.get(x));
    assert!(failed(&landed), "{landed:?}");
    switch.store(true, SeqCst);
    let other = outcome(|| rt.get(w));
    assert!(
        failed(&other),
        "the other thread took the held payload: {other:?}"
    );
    gate.release();
    let first = first.join();
    assert!(
        first == Err("y boom".to_string()) || failed(&first),
        "{first:?}"
    );
    let path = crate::replay::extend_path(
        0,
        crate::replay::Step::of(
            rt.runtime_id,
            w.slot(),
            false,
            u64::from(x.slot()) | (u64::from(x.generation()) << 32),
            1,
        ),
    );
    let seen = seen.lock().unwrap();
    assert_eq!(*seen, vec![Some((1, path)); 2], "the held payload moved");
    rt.assert_invariant_i();
    assert_eq!(rt.provenance_records(), 0, "a record outlived its delivery");
}

/// The two-deliveries shape of `runtime_replay_unwind_tests`, with X
/// switchable as in [`landed_shape`]: A (Fallback 4) catches its read of
/// X and reads X again; X catches its read of C and reads B; B and C read
/// A and panic on its fallback. Switched, X reads nothing and lands Clean
/// 42 or panics, and B and C read nothing.
struct Retaken {
    rt: Arc<Rt>,
    a: Incr<u64>,
    x: Incr<u64>,
    c: Incr<u64>,
    mode: Arc<AtomicU8>,
}

fn retaken_shape() -> Retaken {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    let mode = Arc::new(AtomicU8::new(CHAIN));
    let ac: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let member = |text: &'static str| {
        let (ac, mode) = (ac.clone(), mode.clone());
        rt.create_query_with(
            move |rt: &Rt| {
                if mode.load(SeqCst) != CHAIN {
                    return 0;
                }
                let a = rt.get(*ac.get().unwrap());
                if a == 4 {
                    std::panic::panic_any(String::from(text));
                }
                a
            },
            replayable(),
        )
    };
    let c = member("c boom");
    let b = member("b boom");
    let x = {
        let mode = mode.clone();
        rt.create_query_with(
            move |rt: &Rt| match mode.load(SeqCst) {
                CHAIN => {
                    let _ = catch_unwind(AssertUnwindSafe(|| rt.get(c)));
                    rt.get(b) + 4
                }
                CLEAN => 42,
                _ => std::panic::panic_any(String::from("x failed")),
            },
            replayable(),
        )
    };
    let a = rt.create_query_with(
        move |rt: &Rt| {
            let caught = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_or(9999);
            if caught % 2 == 0 {
                3
            } else {
                rt.get(x)
            }
        },
        QueryOpts {
            replay: Replay::Allowed,
            cycle: CycleStrategy::fallback(|_| 4),
        },
    );
    ac.set(a).unwrap();
    Retaken { rt, a, x, c, mode }
}

/// The unbounded run in which the other thread's switch of X to `mode`
/// and its `get(X)` come first: `get(A)`'s outcome and the states of A
/// and X.
fn retaken_first(mode: u8) -> (Outcome, [NodeState; 2]) {
    let s = retaken_shape();
    s.mode.store(mode, SeqCst);
    let _ = outcome(|| s.rt.get(s.x));
    let a = outcome(|| s.rt.get(s.a));
    (a, [s.a, s.x].map(|h| s.rt.state_of(h)))
}

/// **A frame's node landed by another thread between two takes of its
/// delivery**, at budget 2: the retry of A's caught read takes the
/// delivery for X's read of B, which lands X Dirty, and a retry doomed
/// again takes it a second time. Parked between the two takes, at the
/// driver's second entry for C, another thread switches X and lands it,
/// which it can since virtual frames hold no claims. The later take must
/// not land X over that state, and A ends as in the unbounded run in which
/// the other thread came first. The other thread's own read walks X's
/// deps as the first take left them, so for a panicking X it can raise
/// a Failed read of X rather than X's panic, as a serial walk would.
fn retaken_delivery_after_another_landing(mode: u8) {
    let s = retaken_shape();
    s.rt.set_max_frames(2);
    let gate = Gate::new();
    {
        let (gate, c, hits) = (gate.clone(), id(s.c), AtomicU32::new(0));
        let installed = s.rt.park_hook.set(Box::new(move |p| {
            if p == ParkPoint::BeforeEntry(c) && hits.fetch_add(1, SeqCst) == 1 {
                gate.park();
            }
        }));
        assert!(installed.is_ok(), "park hook installed twice");
    }
    gate.arm();
    let reader = {
        let (rt, a) = (s.rt.clone(), s.a);
        spawn(move || outcome(|| rt.get(a)))
    };
    gate.wait_parked();
    assert_eq!(
        s.rt.state_of(s.x),
        NodeState::Dirty,
        "the first take landed X Dirty"
    );
    s.mode.store(mode, SeqCst);
    let x = outcome(|| s.rt.get(s.x));
    let other = if mode == CLEAN {
        assert_eq!(x, Ok(42));
        NodeState::Clean
    } else {
        assert!(x.is_err(), "{x:?}");
        NodeState::Failed
    };
    assert_eq!(s.rt.state_of(s.x), other);
    gate.release();
    let a = reader.join();

    assert_eq!(
        s.rt.state_of(s.x),
        other,
        "the later take landed X over the other thread's landing"
    );
    assert_eq!(
        (a, [s.a, s.x].map(|h| s.rt.state_of(h))),
        retaken_first(mode)
    );
    s.rt.assert_invariant_i();
}

#[test]
fn shared_retaken_delivery_after_another_thread_lands_clean() {
    retaken_delivery_after_another_landing(CLEAN);
}

#[test]
fn shared_retaken_delivery_after_another_thread_lands_failed() {
    retaken_delivery_after_another_landing(FAILED);
}

/// Two replayable readers each doom at budget 1 and go on to the Native
/// `n`. Thread 1's speculative read of `n` passes the tried check and
/// parks before its claim; the main thread's speculative run then computes
/// `n`, stashes it and lands it back New. Thread 1's claim then succeeds
/// on New, and must not run `n` a second time in the revision.
#[test]
fn a_claim_after_another_threads_speculative_run_does_not_run_it_again() {
    let mut rt = Rt::new();
    rt.all_replayable = false;
    let rt = Arc::new(rt);
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let input = rt.create_input(1_u64);
    let runs = Arc::new(AtomicU32::new(0));
    let counted = runs.clone();
    let n = rt.create_query(move |rt: &Rt| {
        counted.fetch_add(1, SeqCst);
        rt.get(input) + 10
    });
    let reader = || {
        let b = rt.create_query(|_: &Rt| 5_u64);
        rt.create_query_with(
            move |rt: &Rt| {
                let x = rt.get(b);
                if x == 0 {
                    rt.get(n)
                } else {
                    x
                }
            },
            replayable(),
        )
    };
    let (r1, r2) = (reader(), reader());
    let gate = Gate::new();
    gate.arm();
    hook(&rt, vec![(ParkPoint::BeforeClaim(id(n)), gate.clone())]);
    let parked = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(r1))
    };
    gate.wait_parked();
    assert_eq!(rt.get(r2), 5);
    assert_eq!(runs.load(SeqCst), 1);
    gate.release();
    assert_eq!(parked.join().unwrap(), 5);
    assert_eq!(runs.load(SeqCst), 1, "n ran speculatively twice");
    assert_eq!(rt.get(n), 11);
    assert_eq!(runs.load(SeqCst), 1);
}

/// A speculative verify of `mid` scans its deps, then a set ends the
/// revision before it stashes: the stash must not hold for the set's
/// revision, where a speculative read would take `mid` as unchanged.
#[test]
fn a_speculative_verify_across_a_set_keeps_no_stash() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let input = rt.create_input(1_u64);
    let leaf = rt.create_query_with(move |rt: &Rt| rt.get(input) / 10, replayable());
    let mid = rt.create_query_with(move |rt: &Rt| rt.get(leaf), replayable());
    let top = rt.create_query_with(move |rt: &Rt| rt.get(mid) + 1, replayable());
    assert_eq!(rt.get(top), 1);
    rt.set(input, 2);
    let b = rt.create_query(|_: &Rt| 5_u64);
    let reader = rt.create_query_with(
        move |rt: &Rt| {
            let x = rt.get(b);
            if x == 0 {
                rt.get(top)
            } else {
                x
            }
        },
        replayable(),
    );
    let gate = Gate::new();
    gate.arm();
    hook(
        &rt,
        vec![(ParkPoint::VerifyAfterScan(id(mid)), gate.clone())],
    );
    let parked = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(reader))
    };
    gate.wait_parked();
    rt.set(input, 13);
    gate.release();
    assert_eq!(parked.join().unwrap(), 5);
    assert!(
        !rt.stashed_verify(id(mid).0, mid.generation()),
        "a verify from before the set is stashed for its revision"
    );
    assert_eq!(rt.get(top), 2);
}

/// A set bumps the revision and parks before its walk; a speculative
/// verify of `mid` then claims in the new revision but scans `leaf` as the
/// old one left it. It must not stash a verify the walk is about to undo.
#[test]
fn a_speculative_verify_before_a_sets_walk_keeps_no_stash() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let input = rt.create_input(1_u64);
    let leaf = rt.create_query_with(move |rt: &Rt| rt.get(input) / 10, replayable());
    let mid = rt.create_query_with(move |rt: &Rt| rt.get(leaf), replayable());
    let top = rt.create_query_with(move |rt: &Rt| rt.get(mid) + 1, replayable());
    assert_eq!(rt.get(top), 1);
    rt.set(input, 2);
    assert_eq!(rt.get(leaf), 0);
    let b = rt.create_query(|_: &Rt| 5_u64);
    let reader = rt.create_query_with(
        move |rt: &Rt| {
            let x = rt.get(b);
            if x == 0 {
                rt.get(top)
            } else {
                x
            }
        },
        replayable(),
    );
    let gate = Gate::new();
    gate.arm();
    hook(&rt, vec![(ParkPoint::SetAfterBump, gate.clone())]);
    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(input, 13))
    };
    gate.wait_parked();
    assert_eq!(rt.get(reader), 5);
    let stashed = rt.stashed_verify(id(mid).0, mid.generation());
    gate.release();
    setter.join().unwrap();
    assert!(
        !stashed,
        "a verify over a dep the pending walk dirties is stashed"
    );
    assert_eq!(rt.get(top), 2);
}

static SLOW_ARMED: AtomicBool = AtomicBool::new(false);
static SLOW_IN_CLONE: AtomicBool = AtomicBool::new(false);
static SLOW_RELEASED: AtomicBool = AtomicBool::new(false);

/// A value whose next `Clone` after [`SLOW_ARMED`] is set waits for
/// [`SLOW_RELEASED`].
#[derive(Debug, PartialEq)]
struct Slow(u64);

impl Clone for Slow {
    fn clone(&self) -> Self {
        if SLOW_ARMED.swap(false, SeqCst) {
            SLOW_IN_CLONE.store(true, SeqCst);
            while !SLOW_RELEASED.load(SeqCst) {
                thread::sleep(Duration::from_millis(1));
            }
        }
        Slow(self.0)
    }
}

/// A speculative read clones a stashed value with the stash lock released,
/// so a user `Clone` that waits on another thread cannot hold up that
/// thread's stash bookkeeping.
#[test]
fn a_stashed_value_is_cloned_outside_the_stash_lock() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let input = rt.create_input(7_u64);
    let x = rt.create_query_with(move |rt: &Rt| Slow(rt.get(input)), replayable());
    let reader = || {
        let b = rt.create_query(|_: &Rt| 5_u64);
        rt.create_query_with(
            move |rt: &Rt| {
                let v = rt.get(b);
                if v == 0 {
                    rt.get(x).0
                } else {
                    v
                }
            },
            replayable(),
        )
    };
    let (r1, r2) = (reader(), reader());
    assert_eq!(rt.get(r1), 5);
    assert!(rt.stashed(id(x).0, x.generation(), false), "x is stashed");
    SLOW_ARMED.store(true, SeqCst);
    let cloning = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(r2))
    };
    while !SLOW_IN_CLONE.load(SeqCst) {
        thread::sleep(Duration::from_millis(1));
    }
    let (tx, rx) = mpsc::channel();
    let looking = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || tx.send(rt.stashed(id(x).0, x.generation(), false)).unwrap())
    };
    let looked = rx.recv_timeout(Duration::from_secs(2));
    SLOW_RELEASED.store(true, SeqCst);
    assert_eq!(cloning.join().unwrap(), 5);
    looking.join().unwrap();
    assert_eq!(
        looked,
        Ok(true),
        "the stash lock is held across a user Clone"
    );
}
