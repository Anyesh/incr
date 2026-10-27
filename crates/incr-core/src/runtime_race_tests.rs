//! Deterministic race tests for the shared-protocol plan (B1-B6). Each
//! `Shared` test parks one thread at a named [`ParkPoint`] (or inside a
//! compute closure) and acts from another while it is parked. Every test
//! ends by checking invariant (I) and the value a fresh read returns.

use std::cell::RefCell;
use std::collections::HashSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, Weak};
use std::thread;
use std::time::Duration;

use super::{
    DeleteError, DeleteOutcome, Entry, EpisodeLog, Expanded, GroupInfo, OnUnwind, ParkPoint,
    Runtime, Walk, EPISODE_LOG_LEN, NO_TAINT, VISIT,
};
use crate::cells::{Cells, Local, Shared};
use crate::collection::{GroupedCollection, IncrCollection};
use crate::cycle_state::thread_key;
use crate::deferred::Deferred;
use crate::handle::Incr;
use crate::node::NodeId;
use crate::state::NodeState;

type Rt = Runtime<Shared>;

/// A park point that never gets reached would otherwise hang the test.
/// Under Miri the clock advances with every interpreted step of every
/// thread, so a timeout sized for native runs ends far sooner.
pub(super) const PARK_TIMEOUT: Duration = if cfg!(miri) {
    Duration::from_secs(120)
} else {
    Duration::from_secs(10)
};

thread_local! {
    /// The gates a test's thread made, let go when it ends, failed or not,
    /// since a test drives its gates from that thread: a failed test then
    /// fails alone instead of aborting the binary at the park timeout. A
    /// scope joins its threads before the test thread ends, so a scope that
    /// drives gates holds [`release_parked_on_unwind`], and a driver on
    /// another thread holds [`Gate::released_on_unwind`].
    static MADE_HERE: MadeHere = const { MadeHere(RefCell::new(Vec::new())) };
}

struct MadeHere(RefCell<Vec<Weak<GateInner>>>);

impl MadeHere {
    fn let_go(&self) {
        for gate in self.0.borrow_mut().drain(..) {
            if let Some(gate) = gate.upgrade() {
                Gate(gate).let_go();
            }
        }
    }
}

impl Drop for MadeHere {
    fn drop(&mut self) {
        self.let_go();
    }
}

/// Lets go of every gate this thread made if it is dropped by a panic, so
/// a scope that joins a parked thread does not wait out the park timeout.
pub(crate) fn release_parked_on_unwind() -> OnUnwind<impl FnMut()> {
    OnUnwind(|| MADE_HERE.with(MadeHere::let_go))
}

/// One-shot rendezvous. The thread under test calls [`Gate::park`]; the
/// driver calls [`Gate::wait_parked`], acts, then [`Gate::release`]. A
/// gate only parks once it is armed, so a closure can carry one through
/// setup computes without stopping.
#[derive(Clone)]
pub(crate) struct Gate(Arc<GateInner>);

pub(crate) struct GateInner {
    armed: AtomicBool,
    reached_tx: Mutex<Sender<()>>,
    reached_rx: Mutex<Receiver<()>>,
    release_tx: Mutex<Sender<()>>,
    release_rx: Mutex<Receiver<()>>,
    test: Option<String>,
}

impl Gate {
    pub(crate) fn new() -> Self {
        let (reached_tx, reached_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let gate = Self(Arc::new(GateInner {
            armed: AtomicBool::new(false),
            reached_tx: Mutex::new(reached_tx),
            reached_rx: Mutex::new(reached_rx),
            release_tx: Mutex::new(release_tx),
            release_rx: Mutex::new(release_rx),
            test: thread::current().name().map(str::to_owned),
        }));
        if gate.0.test.is_some() {
            MADE_HERE.with(|m| m.0.borrow_mut().push(Arc::downgrade(&gate.0)));
        }
        gate
    }

    pub(crate) fn arm(&self) {
        self.0.armed.store(true, Ordering::SeqCst);
    }

    pub(crate) fn park(&self) {
        // A gate stands for a step the test holds, not a stuck holder.
        self.park_as(true);
    }

    /// [`Self::park`] as a wait in the closure's own code, such as a join:
    /// a waiter on a landing the attempt holds may take it over.
    pub(crate) fn park_in_user_code(&self) {
        self.park_as(false);
    }

    fn park_as(&self, step: bool) {
        if !self.0.armed.swap(false, Ordering::SeqCst) {
            return;
        }
        let _parked = step.then(crate::attempt::Parked::new);
        self.0.reached_tx.lock().unwrap().send(()).unwrap();
        if self
            .0
            .release_rx
            .lock()
            .unwrap()
            .recv_timeout(PARK_TIMEOUT)
            .is_err()
        {
            super::park_timed_out(self.0.test.as_deref());
        }
    }

    pub(crate) fn wait_parked(&self) {
        self.0
            .reached_rx
            .lock()
            .unwrap()
            .recv_timeout(PARK_TIMEOUT)
            .expect("thread never reached its park point");
    }

    pub(crate) fn release(&self) {
        self.0.release_tx.lock().unwrap().send(()).unwrap();
    }

    /// Lets the gate go if this thread panics: for a driver that is not
    /// the thread that made the gate.
    pub(crate) fn released_on_unwind(&self) -> OnUnwind<impl FnMut() + '_> {
        OnUnwind(|| self.let_go())
    }

    fn let_go(&self) {
        self.0.armed.store(false, Ordering::SeqCst);
        self.release();
    }

    /// Whether an armed gate has been reached.
    pub(crate) fn reached(&self) -> bool {
        !self.0.armed.load(Ordering::SeqCst)
    }
}

/// Releases its gates if the test unwinds, so a failed assert fails the
/// test instead of leaving a thread parked until its timeout aborts the
/// binary. A release sent before a thread parks lets that park through.
pub(crate) struct ReleaseOnUnwind(pub(crate) Vec<Gate>);

impl Drop for ReleaseOnUnwind {
    fn drop(&mut self) {
        if thread::panicking() {
            for gate in &self.0 {
                gate.release();
            }
        }
    }
}

/// Install the runtime's park hook with armed gates at `points`.
pub(super) fn park_at<C: Cells>(rt: &Runtime<C>, points: Vec<(ParkPoint, Gate)>) {
    for (_, gate) in &points {
        gate.arm();
    }
    let installed = rt.park_hook.set(Box::new(move |p| {
        for (point, gate) in &points {
            if *point == p {
                gate.park();
            }
        }
    }));
    assert!(installed.is_ok(), "park hook installed twice");
}

/// Waits until `rt` counts a wait on a doom reservation past `before`:
/// another thread's read met the reservation and now waits it out.
pub(super) fn await_reservation_wait<C: Cells>(rt: &Runtime<C>, before: u32) {
    // Before the parked thread's own timeout, whose abort would hide this assert.
    let deadline = std::time::Instant::now() + PARK_TIMEOUT / 2;
    while rt.reservation_waits.load(Ordering::SeqCst) <= before {
        assert!(
            std::time::Instant::now() < deadline,
            "the other read never met the reservation"
        );
        std::thread::yield_now();
    }
}

pub(super) fn id<T>(h: Incr<T>) -> NodeId {
    NodeId(h.slot())
}

/// x -> Q = x / 2 (early cutoff) -> D = Q + 100, all Clean at x = 0.
fn chain(rt: &Rt) -> (Incr<u64>, Incr<u64>, Incr<u64>) {
    let x = rt.create_input(0_u64);
    let q = rt.create_query(move |rt: &Rt| rt.get(x) / 2);
    let d = rt.create_query(move |rt: &Rt| rt.get(q) + 100);
    assert_eq!(rt.get(d), 100);
    (x, q, d)
}

#[test]
fn b5_set_lands_between_verify_scan_and_cas() {
    let rt = Arc::new(Rt::new());
    let (x, _q, d) = chain(&rt);
    rt.set(x, 1);
    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::VerifyAfterScan(id(d)), gate.clone())]);

    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(d))
    };
    gate.wait_parked();
    rt.set(x, 2);
    gate.release();
    reader.join().unwrap();

    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 101);
}

#[test]
fn b5_set_lands_before_dependent_claim() {
    let rt = Arc::new(Rt::new());
    let (x, _q, d) = chain(&rt);
    rt.set(x, 1);
    let gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::BeforeComputeOne(id(d)), gate.clone())],
    );

    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(d))
    };
    gate.wait_parked();
    rt.set(x, 2);
    gate.release();
    reader.join().unwrap();

    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 101);
}

#[test]
fn b4_computing_dirty_cutoff_strands_dependent() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let q_gate = Gate::new();
    let q = {
        let q_gate = q_gate.clone();
        rt.create_query(move |rt: &Rt| {
            let v = rt.get(x);
            q_gate.park();
            v / 2
        })
    };
    let d = rt.create_query(move |rt: &Rt| rt.get(q) + 100);
    assert_eq!(rt.get(d), 100);
    // D Dirty over a Clean Q: Q recomputes to the same value.
    rt.set(x, 0);
    assert_eq!(rt.get(q), 0);

    let d_gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::BeforeComputeOne(id(d)), d_gate.clone())],
    );
    let reader_d = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(d))
    };
    d_gate.wait_parked();

    rt.set(x, 1);
    q_gate.arm();
    let reader_q = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(q))
    };
    q_gate.wait_parked();
    rt.set(x, 3);
    q_gate.release();
    reader_q.join().unwrap();
    d_gate.release();
    reader_d.join().unwrap();

    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 101);
}

/// x -> Q = x + 1 -> D = Q * 10 + y, with D Dirty through y while Q is
/// Clean. Returns (x, y, q, d).
fn b6_graph(rt: &Rt, d_gate: &Gate) -> (Incr<u64>, Incr<u64>, Incr<u64>, Incr<u64>) {
    let x = rt.create_input(0_u64);
    let y = rt.create_input(0_u64);
    let q = rt.create_query(move |rt: &Rt| rt.get(x) + 1);
    let d = {
        let d_gate = d_gate.clone();
        rt.create_query(move |rt: &Rt| {
            let v = rt.get(q) * 10 + rt.get(y);
            d_gate.park();
            v
        })
    };
    assert_eq!(rt.get(d), 10);
    rt.set(y, 1);
    (x, y, q, d)
}

#[test]
fn b6_verified_at_names_unwalked_revision_flagged_mid_compute() {
    let rt = Arc::new(Rt::new());
    let d_gate = Gate::new();
    let (x, _y, q, d) = b6_graph(&rt, &d_gate);

    let set_gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterBump, set_gate.clone())]);
    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(x, 5))
    };
    set_gate.wait_parked();

    d_gate.arm();
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(d))
    };
    d_gate.wait_parked();
    set_gate.release();
    setter.join().unwrap();
    d_gate.release();
    reader.join().unwrap();

    assert_eq!(rt.get(q), 6);
    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 61);
}

#[test]
fn b6_verified_at_names_unwalked_revision_landed_clean() {
    let rt = Arc::new(Rt::new());
    let d_gate = Gate::new();
    let (x, _y, q, d) = b6_graph(&rt, &d_gate);

    let set_gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterBump, set_gate.clone())]);
    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(x, 5))
    };
    set_gate.wait_parked();
    assert_eq!(rt.get(d), 11);
    set_gate.release();
    setter.join().unwrap();

    assert_eq!(rt.get(q), 6);
    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 61);
}

#[test]
fn b3_input_dep_set_after_first_read() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let gate = Gate::new();
    let d = {
        let gate = gate.clone();
        rt.create_query(move |rt: &Rt| {
            let v = rt.get(x) * 10;
            gate.park();
            v
        })
    };

    gate.arm();
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(d))
    };
    gate.wait_parked();
    rt.set(x, 5);
    gate.release();
    reader.join().unwrap();

    assert_eq!(rt.state_of(d), NodeState::Dirty);
    assert!(rt.has_reverse_edge(x, d));
    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 50);
}

#[test]
fn b3_query_dep_dirtied_after_first_read() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(0_u64);
    let s = rt.create_query(move |rt: &Rt| rt.get(x) + 1);
    assert_eq!(rt.get(s), 1);
    let gate = Gate::new();
    let d = {
        let gate = gate.clone();
        rt.create_query(move |rt: &Rt| {
            let v = rt.get(s) * 10;
            gate.park();
            v
        })
    };

    gate.arm();
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(d))
    };
    gate.wait_parked();
    rt.set(x, 5);
    gate.release();
    reader.join().unwrap();

    assert_eq!(rt.state_of(d), NodeState::Dirty);
    assert!(rt.has_reverse_edge(s, d));
    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 60);
}

/// D's first compute claims after a set's bump and before its walk, then
/// reads P Clean. The walk can't reach D, whose edge isn't published, and
/// P recomputed on another thread lands Clean with a `changed_at` equal to
/// D's stamp revision. Settled against the stamp revision, D would land
/// Clean over the old P for good (bug l).
#[test]
fn b3_dep_recomputed_for_an_unwalked_set_supersedes_new_dependent() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let p = rt.create_query(move |rt: &Rt| rt.get(x) + 1);
    assert_eq!(rt.get(p), 2);
    let d_gate = Gate::new();
    let d = {
        let d_gate = d_gate.clone();
        rt.create_query(move |rt: &Rt| {
            let v = rt.get(p) * 10;
            d_gate.park();
            v
        })
    };

    let set_gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterBump, set_gate.clone())]);
    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(x, 5))
    };
    set_gate.wait_parked();

    d_gate.arm();
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(d))
    };
    d_gate.wait_parked();
    set_gate.release();
    setter.join().unwrap();
    assert_eq!(rt.state_of(p), NodeState::Dirty);
    let recompute = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(p))
    };
    assert_eq!(recompute.join().unwrap(), 6);
    d_gate.release();
    assert_eq!(reader.join().unwrap(), 20);

    assert_eq!(rt.state_of(d), NodeState::Dirty);
    assert!(rt.has_reverse_edge(p, d));
    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 60);
}

/// `f` panics while `x` is even; `d` catches that panic as 7 and reads
/// `y`, parking on `gate` in its own code once its reads are done, while
/// its attempt holds `f`'s Failed landing.
fn caught_failed_dep(rt: &Rt, gate: &Gate) -> (Incr<u64>, Incr<u64>, Incr<u64>, Incr<u64>) {
    caught_failed_dep_counted(rt, gate, &Arc::new(AtomicUsize::new(0)))
}

/// [`caught_failed_dep`], counting `d`'s runs in `d_runs`.
fn caught_failed_dep_counted(
    rt: &Rt,
    gate: &Gate,
    d_runs: &Arc<AtomicUsize>,
) -> (Incr<u64>, Incr<u64>, Incr<u64>, Incr<u64>) {
    let x = rt.create_input(0_u64);
    let y = rt.create_input(0_u64);
    let f = rt.create_query(move |rt: &Rt| {
        let v = rt.get(x);
        assert!(v % 2 == 1, "even");
        v
    });
    let d = {
        let gate = gate.clone();
        let d_runs = Arc::clone(d_runs);
        rt.create_query(move |rt: &Rt| {
            d_runs.fetch_add(1, Ordering::SeqCst);
            let v = catch_unwind(AssertUnwindSafe(|| rt.get(f))).map_or(7, |v| v * 10) + rt.get(y);
            gate.park_in_user_code();
            v
        })
    };
    (x, y, f, d)
}

/// A set moves `d`'s Failed dep `f` between `d`'s first read and its
/// publish, before `d`'s edge exists, so the walk can't reach `d`: it
/// lands Dirty with no memo. `d`'s attempt holds `f`'s Failed landing
/// until it ends, so the set meets it held.
#[test]
fn caught_failed_dep_flipped_before_the_first_publish_keeps_no_memo() {
    let rt = Arc::new(Rt::new());
    let gate = Gate::new();
    let (x, _, f, d) = caught_failed_dep(&rt, &gate);
    gate.arm();
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(d))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(f), NodeState::Computing);
    rt.set(x, 1);
    gate.release();
    assert_eq!(reader.join().unwrap(), 7);

    assert_eq!(rt.state_of(d), NodeState::Dirty);
    assert!(!rt.has_memo_of(d));
    assert!(rt.has_reverse_edge(f, d));
    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 10);
}

/// As above, and `f` lands Failed again for the set before `d` publishes:
/// the reader that relands it takes `f`'s held landing over from `d`'s
/// parked attempt, wounding it, so `d` never publishes over a Failed state
/// it did not read: its retry reads the relanded one and keeps that memo.
#[test]
fn caught_failed_dep_relanded_failed_before_the_first_publish_retries_over_it() {
    let rt = Arc::new(Rt::new());
    let gate = Gate::new();
    let d_runs = Arc::new(AtomicUsize::new(0));
    let (x, _, f, d) = caught_failed_dep_counted(&rt, &gate, &d_runs);
    gate.arm();
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(d))
    };
    gate.wait_parked();
    rt.set(x, 2);
    let refail = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.get(f))).is_err())
    };
    assert!(refail.join().unwrap());
    assert_eq!(rt.state_of(f), NodeState::Failed);
    gate.release();
    assert_eq!(reader.join().unwrap(), 7);
    assert_eq!(d_runs.load(Ordering::SeqCst), 2);

    assert_eq!(rt.state_of(d), NodeState::Dirty);
    assert!(rt.has_memo_of(d));
    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 7);
    assert_eq!(d_runs.load(Ordering::SeqCst), 2);
}

/// `d` lands Dirty with a memo over Failed `f`; a recompute for `y` reads
/// `f` Failed again, and a set that moves `f` mid-compute leaves it no
/// memo.
#[test]
fn caught_failed_dep_flipped_during_a_recompute_keeps_no_memo() {
    let rt = Arc::new(Rt::new());
    let gate = Gate::new();
    let (x, y, f, d) = caught_failed_dep(&rt, &gate);
    assert_eq!(rt.get(d), 7);
    assert_eq!(rt.state_of(d), NodeState::Dirty);
    assert!(rt.has_memo_of(d));
    assert_eq!(rt.state_of(f), NodeState::Failed);
    rt.assert_invariant_i();

    rt.set(y, 1);
    gate.arm();
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(d))
    };
    gate.wait_parked();
    rt.set(x, 1);
    gate.release();
    assert_eq!(reader.join().unwrap(), 8);

    assert_eq!(rt.state_of(d), NodeState::Dirty);
    assert!(!rt.has_memo_of(d));
    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 11);
}

/// A read loads `d` Dirty with no memo for the revision, another thread
/// lands `d` with one, and the read claims `d` after: it takes the memo
/// instead of running `d` again in the revision.
#[test]
fn claim_after_another_threads_memo_takes_the_memo() {
    let rt = Arc::new(Rt::new());
    let gate = Gate::new();
    let (_, y, _, d) = caught_failed_dep(&rt, &gate);
    assert_eq!(rt.get(d), 7);
    rt.set(y, 1);
    let unclaims = rt.unrun_unclaims.load(Ordering::Relaxed);
    let before_claim = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::BeforeClaim(id(d)), before_claim.clone())],
    );
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(d))
    };
    before_claim.wait_parked();
    assert_eq!(rt.get(d), 8);
    assert!(rt.has_memo_of(d));
    before_claim.release();
    assert_eq!(reader.join().unwrap(), 8);
    assert!(rt.has_memo_of(d));
    assert_eq!(rt.unrun_unclaims.load(Ordering::Relaxed), unclaims + 1);
}

#[test]
fn b1_publish_lands_between_delete_check_and_claim() {
    let rt = Arc::new(Rt::new());
    let v = rt.create_input(7_u64);
    let q = rt.create_query(move |rt: &Rt| rt.get(v) + 1);
    let gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::DeleteAfterCheck(id(v)), gate.clone())],
    );

    let deleter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.delete_node(v))
    };
    gate.wait_parked();
    let read = catch_unwind(AssertUnwindSafe(|| rt.get(q)));
    gate.release();
    deleter.join().unwrap();

    if let Ok(value) = read {
        assert_eq!(value, 8);
    }
    assert_ne!(
        rt.state_of(q),
        NodeState::Clean,
        "Q is Clean over a deleted dep"
    );
    rt.assert_invariant_i();
}

#[test]
fn b2_delete_and_recycle_while_publisher_is_parked() {
    let rt = Arc::new(Rt::new());
    let v = rt.create_input(7_u64);
    let gate = Gate::new();
    let q = {
        let gate = gate.clone();
        rt.create_query(move |rt: &Rt| {
            let a = rt.get(v);
            gate.park();
            a + 1
        })
    };

    gate.arm();
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.get(q))).is_ok())
    };
    gate.wait_parked();
    rt.delete_node(v);
    let recycled = rt.create_input(99_u64);
    assert_eq!(
        recycled.slot(),
        v.slot(),
        "the delete should free v's slot for reuse"
    );
    gate.release();
    let returned = reader.join().unwrap();

    assert!(!returned, "the stale compute's value was returned");
    assert_eq!(rt.state_of(q), NodeState::Failed);
    assert!(
        !rt.has_reverse_edge(recycled, q),
        "phantom edge on the recycled slot"
    );
    rt.assert_invariant_i();
}

/// Parks a `delete_node(v)` at `DeleteAfterCheck`, so `v` is claimed
/// `Dead` but its generation is not bumped and its slot not released.
/// The returned waiter gate parks a second delete of `v` once it reaches
/// the wait on that claim.
fn park_delete_after_claim(rt: &Arc<Rt>, v: Incr<u64>) -> (Gate, Gate, thread::JoinHandle<()>) {
    let claimed = Gate::new();
    let waiter = Gate::new();
    park_at(
        rt,
        vec![
            (ParkPoint::DeleteAfterCheck(id(v)), claimed.clone()),
            (ParkPoint::DeleteWaitsOnClaim(id(v)), waiter.clone()),
        ],
    );
    let deleter = {
        let rt = Arc::clone(rt);
        thread::spawn(move || rt.delete_node(v))
    };
    claimed.wait_parked();
    (claimed, waiter, deleter)
}

#[test]
fn is_alive_is_false_while_dead_but_not_yet_released() {
    let rt = Arc::new(Rt::new());
    let v = rt.create_input(7_u64);
    let (claimed, _, deleter) = park_delete_after_claim(&rt, v);

    assert_eq!(rt.state_of(v), NodeState::Dead);
    assert_eq!(
        rt.nodes.get(v.slot()).generation(),
        v.erase().generation(),
        "the generation must not move before the write lock"
    );
    assert!(!rt.is_alive(v));

    claimed.release();
    deleter.join().unwrap();
    assert!(!rt.is_alive(v));
}

#[test]
fn delete_erased_waits_out_a_claimed_delete_then_reports_gone() {
    let rt = Arc::new(Rt::new());
    let v = rt.create_input(7_u64);
    let (claimed, waiter, deleter) = park_delete_after_claim(&rt, v);

    let second = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.delete_erased(v.erase()))
    };
    waiter.wait_parked();
    waiter.release();
    thread::sleep(Duration::from_millis(50));
    assert!(
        !second.is_finished(),
        "a second delete returned while the first still held its claim"
    );

    claimed.release();
    deleter.join().unwrap();
    assert_eq!(second.join().unwrap(), DeleteOutcome::AlreadyGone);
    assert_eq!(rt.node_count(), 0);
    rt.assert_invariant_i();
}

/// A strict delete passes its handle check, then loses the node to
/// another delete and its slot to a recycled node that has a dependent.
/// It must report the lost race, not the new occupant's dependents.
#[test]
fn strict_delete_that_lost_its_slot_to_a_recycle_reports_gone() {
    let rt = Arc::new(Rt::new());
    let v = rt.create_input(7_u64);
    let gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::DeleteAfterHandleCheck(id(v)), gate.clone())],
    );
    let late = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.delete_node(v))))
    };
    gate.wait_parked();
    assert_eq!(rt.delete_erased(v.erase()), DeleteOutcome::Deleted);
    let recycled = rt.create_input(1_u64);
    assert_eq!(recycled.slot(), v.slot());
    let reader = rt.create_query(move |rt: &Rt| rt.get(recycled));
    assert_eq!(rt.get(reader), 1);
    gate.release();

    let err = late.join().unwrap().unwrap_err();
    let msg = err.downcast_ref::<String>().unwrap();
    assert!(msg.contains("deleted twice concurrently"), "got: {msg}");
    assert!(rt.is_alive(recycled) && rt.is_alive(reader));
}

#[test]
fn concurrent_typed_delete_panics_deleted_twice() {
    let rt = Arc::new(Rt::new());
    let v = rt.create_input(7_u64);
    let (claimed, waiter, deleter) = park_delete_after_claim(&rt, v);

    let second = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.delete_node(v))))
    };
    waiter.wait_parked();
    claimed.release();
    deleter.join().unwrap();
    waiter.release();
    let msg = *second
        .join()
        .unwrap()
        .unwrap_err()
        .downcast::<String>()
        .unwrap();
    assert!(msg.contains("deleted twice concurrently"), "got: {msg}");
}

/// Q attaches while the delete is parked at its entry. A dependents check
/// made there, before a strict delete, would pass and the delete would
/// then panic; `try_delete_node` checks as part of its claim, so it sees Q.
#[test]
fn try_delete_reports_a_dependent_attached_after_its_entry() {
    let rt = Arc::new(Rt::new());
    let v = rt.create_input(7_u64);
    let q = rt.create_query(move |rt: &Rt| rt.get(v) + 1);
    let gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::DeleteAfterHandleCheck(id(v)), gate.clone())],
    );

    let deleter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.try_delete_node(v))
    };
    gate.wait_parked();
    assert_eq!(rt.get(q), 8);
    gate.release();

    assert_eq!(deleter.join().unwrap(), Err(DeleteError::HasDependents));
    assert!(rt.is_alive(v));
    rt.set(v, 9);
    assert_eq!(rt.get(q), 10);
    rt.assert_invariant_i();
}

#[test]
fn try_delete_claim_leaves_no_edge_for_a_compute_that_follows() {
    let rt = Arc::new(Rt::new());
    let v = rt.create_input(7_u64);
    let q = rt.create_query(move |rt: &Rt| rt.get(v) + 1);
    let gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::DeleteAfterCheck(id(v)), gate.clone())],
    );

    let deleter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.try_delete_node(v))
    };
    gate.wait_parked();
    let read = catch_unwind(AssertUnwindSafe(|| rt.get(q)));
    gate.release();

    assert_eq!(deleter.join().unwrap(), Ok(()));
    if let Ok(value) = read {
        assert_eq!(value, 8);
    }
    assert_ne!(rt.state_of(q), NodeState::Clean);
    rt.assert_invariant_i();
}

/// A router whose groups were created before it ran, all Clean.
pub(super) fn grouped(rt: &Rt) -> (Incr<u64>, Vec<Incr<u64>>) {
    let c: IncrCollection<i64, Shared> = rt.create_collection();
    c.insert(rt, 3);
    c.insert(rt, 4);
    let groups = c.group_by(rt, |x| x % 3);
    let _ = rt.get(groups.version_node());
    let versions = groups
        .keys()
        .iter()
        .map(|k| groups.get_group(k).unwrap().version_node())
        .collect();
    (groups.version_node(), versions)
}

/// A strict delete of a router finds one of its groups claimed by a
/// parked delete of that group. It waits that delete out instead of
/// claiming around it, then deletes the rest.
#[test]
fn strict_delete_of_an_owner_waits_out_a_claimed_owned_node() {
    let rt = Arc::new(Rt::new());
    let (router, versions) = grouped(&rt);
    let group = versions[0];
    let claimed = Gate::new();
    let waiter = Gate::new();
    park_at(
        &rt,
        vec![
            (ParkPoint::DeleteAfterCheck(id(group)), claimed.clone()),
            (ParkPoint::DeleteWaitsOnClaim(id(group)), waiter.clone()),
        ],
    );
    let group_delete = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.delete_node(group))
    };
    claimed.wait_parked();

    let router_delete = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.try_delete_node(router))
    };
    waiter.wait_parked();
    waiter.release();
    thread::sleep(Duration::from_millis(50));
    assert!(!router_delete.is_finished());
    assert!(rt.is_alive(router), "the router was claimed around a group");

    claimed.release();
    group_delete.join().unwrap();
    assert_eq!(router_delete.join().unwrap(), Ok(()));
    assert!(!rt.is_alive(router));
    assert!(versions.iter().all(|&v| !rt.is_alive(v)));
    assert_eq!(rt.node_count(), 1, "only the source is left");
    rt.assert_invariant_i();
}

/// The router's strict delete finds a group claimed by a parked delete
/// of that group: `claim_set` rolls every member it took back to its
/// state, waits the group's delete out, and then deletes the rest.
#[test]
fn strict_delete_of_an_owner_rolls_back_around_a_claimed_owned_node() {
    let rt = Arc::new(Rt::new());
    let (router, versions) = grouped(&rt);
    let group = versions[0];
    let claimed = Gate::new();
    let rolled_back = Gate::new();
    park_at(
        &rt,
        vec![
            (ParkPoint::DeleteAfterCheck(id(group)), claimed.clone()),
            (
                ParkPoint::ClaimSetRolledBack(id(router)),
                rolled_back.clone(),
            ),
        ],
    );
    let group_delete = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.delete_node(group))
    };
    claimed.wait_parked();
    let router_delete = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.try_delete_node(router))
    };
    rolled_back.wait_parked();
    assert_eq!(rt.state_of(router), NodeState::Clean);
    assert_eq!(rt.state_of(versions[1]), NodeState::Clean);
    assert!(rt.is_alive(router) && rt.is_alive(versions[1]));
    rolled_back.release();
    thread::sleep(Duration::from_millis(50));
    assert!(
        !router_delete.is_finished(),
        "the router delete waits for the group's"
    );

    claimed.release();
    group_delete.join().unwrap();
    assert_eq!(router_delete.join().unwrap(), Ok(()));
    assert!(versions.iter().all(|&v| !rt.is_alive(v)));
    assert_eq!(rt.node_count(), 1, "only the source is left");
    rt.assert_invariant_i();
}

type LocalRt = Runtime<Local>;
type RecycledSlot = Arc<Mutex<Option<Incr<u64>>>>;

/// Q reads v, deletes it, and recycles its slot inside its own compute.
/// Returns (q, the recycled input once Q has run).
fn self_deleting_query(rt: &LocalRt, stop_reading_on_retry: bool) -> (Incr<u64>, RecycledSlot) {
    let v = rt.create_input(7_u64);
    let recycled = Arc::new(Mutex::new(None));
    let first = AtomicBool::new(true);
    let q = {
        let recycled = Arc::clone(&recycled);
        rt.create_query(move |rt: &LocalRt| {
            if !first.swap(false, Ordering::SeqCst) && stop_reading_on_retry {
                return 42;
            }
            let a = rt.get(v);
            rt.delete_node(v);
            *recycled.lock().unwrap() = Some(rt.create_input_unchecked(99_u64));
            a + 1
        })
    };
    (q, recycled)
}

#[test]
fn b2_local_self_delete_retries_and_panics_stale() {
    let rt = LocalRt::new();
    let (q, recycled) = self_deleting_query(&rt, false);

    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(q))).is_err());
    let new = recycled.lock().unwrap().expect("Q's first compute ran");
    assert_eq!(rt.state_of(q), NodeState::Failed);
    assert!(
        !rt.has_reverse_edge(new, q),
        "phantom edge on the recycled slot"
    );
    rt.assert_invariant_i();
}

#[test]
fn b2_local_retry_that_stops_reading_publishes_cleanly() {
    let rt = LocalRt::new();
    let (q, recycled) = self_deleting_query(&rt, true);

    assert_eq!(rt.get(q), 42);
    let new = recycled.lock().unwrap().expect("Q's first compute ran");
    assert_eq!(rt.state_of(q), NodeState::Clean);
    assert!(
        !rt.has_reverse_edge(new, q),
        "phantom edge on the recycled slot"
    );
    assert!(rt.nodes.get(q.slot()).changed_at() > 0);
    rt.assert_invariant_i();
}

#[test]
fn b2_local_nested_self_delete_leaves_both_non_clean() {
    let rt = LocalRt::new();
    let (q, _recycled) = self_deleting_query(&rt, false);
    let d = rt.create_query(move |rt: &LocalRt| rt.get(q) + 100);

    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(d))).is_err());
    assert_ne!(rt.state_of(q), NodeState::Clean);
    assert_ne!(rt.state_of(d), NodeState::Clean);
    rt.assert_invariant_i();
}

#[test]
fn b2_local_panic_after_self_delete_lands_new() {
    let rt = LocalRt::new();
    let x = rt.create_input(0_u64);
    let v = rt.create_input(7_u64);
    let q = rt.create_query(move |rt: &LocalRt| {
        let xv = rt.get(x);
        if xv == 1 {
            rt.get(v);
            rt.delete_node(v);
            panic!("test closure panics after deleting its dep");
        }
        xv
    });
    assert_eq!(rt.get(q), 0);
    rt.set(x, 1);
    let d = rt.create_query(move |rt: &LocalRt| rt.get(q) + 100);

    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(d))).is_err());
    assert_eq!(rt.state_of(q), NodeState::New);
    assert!(
        !rt.has_reverse_edge(v, q),
        "edge left on the deleted dep's slot"
    );
    // D's new dep Q is a pending recompute, which a walk may already have
    // passed before D's edge existed, so D lands Dirty rather than Failed.
    assert_eq!(rt.state_of(d), NodeState::Dirty);
    rt.assert_invariant_i();
    rt.assert_edges_complete();
}

#[test]
fn walk_passes_through_new_to_failed_dependent() {
    let rt = LocalRt::new();
    let x = rt.create_input(0_u64);
    let q = rt.create_query(move |rt: &LocalRt| rt.get(x) + 1);
    let d = rt.create_query(move |rt: &LocalRt| rt.get(q) + 100);
    assert_eq!(rt.get(d), 101);
    // The shape a stale recovery leaves behind: Q New with its old edges,
    // and a Failed dependent reachable only through Q.
    rt.nodes.get(q.slot()).set_state(NodeState::New.as_u8());
    rt.nodes.get(d.slot()).set_state(NodeState::Failed.as_u8());

    rt.set(x, 1);
    assert_eq!(rt.state_of(d), NodeState::Dirty);
    assert_eq!(rt.get(d), 102);
}

#[test]
fn nested_superseded_dep_poisons_fast_path_publisher() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(0_u64);
    let s = rt.create_input(0_u64);
    let y = rt.create_input(0_u64);
    let q_gate = Gate::new();
    let q = {
        let q_gate = q_gate.clone();
        rt.create_query(move |rt: &Rt| {
            let xv = rt.get(x);
            if xv == 0 {
                return 0;
            }
            let sv = rt.get(s);
            q_gate.park();
            xv + sv
        })
    };
    let d = rt.create_query(move |rt: &Rt| rt.get(q) + rt.get(y) + 100);
    assert_eq!(rt.get(d), 100);
    // D Dirty through y while Q stays Clean, so ensure_clean leaves Q to
    // D's closure.
    rt.set(y, 1);

    let d_gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::BeforeComputeOne(id(d)), d_gate.clone())],
    );
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(d))
    };
    d_gate.wait_parked();
    rt.set(x, 1);
    q_gate.arm();
    d_gate.release();
    q_gate.wait_parked();
    // s is a new dep of Q with no reverse edge yet, so this walk reaches
    // neither Q nor D. Only Q's publish validation sees it.
    rt.set(s, 5);
    q_gate.release();
    reader.join().unwrap();

    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 107);
}

/// A reader loads Q `Dirty` and stops before its claim. Meanwhile another
/// reader cleans Q by cutoff and verifies D at revision 4, and a second
/// set dirties both again. The stopped reader's claim then succeeds on
/// the newer `Dirty`, so revisions captured before the claim predate the
/// second set: Q's change would be stamped 4 and D, verified at 4, would
/// stay Clean over it.
#[test]
fn claim_on_a_later_dirty_stamps_that_dirty_revision() {
    let rt = Arc::new(Rt::new());
    let (x, q, d) = chain(&rt);
    rt.set(x, 1);

    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::BeforeClaim(id(q)), gate.clone())]);
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(q))
    };
    gate.wait_parked();
    assert_eq!(rt.get(d), 100);
    rt.set(x, 2);
    gate.release();
    assert_eq!(reader.join().unwrap(), 1);

    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 101);
}

#[test]
fn lagging_watermark_costs_a_recompute_not_a_stale_value() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(0_u64);
    let q = rt.create_query(move |rt: &Rt| rt.get(x) + 1);
    let d = rt.create_query(move |rt: &Rt| rt.get(q) * 10);
    assert_eq!(rt.get(d), 10);

    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterWalk, gate.clone())]);
    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(x, 5))
    };
    gate.wait_parked();
    // The walk is done but not recorded, so these computes stamp
    // verified_at below Q's new changed_at.
    assert_eq!(rt.get(d), 60);
    gate.release();
    setter.join().unwrap();

    rt.assert_walks_complete();
    rt.assert_invariant_i();
    assert_eq!(rt.get(d), 60);
    rt.set(x, 6);
    assert_eq!(rt.get(d), 70);
}

fn iters(full: usize) -> usize {
    if cfg!(miri) {
        full / 100
    } else {
        full
    }
}

/// xorshift64, so every sweep thread draws a reproducible sequence.
struct Rng(u64);

impl Rng {
    fn next(&mut self, bound: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % bound
    }
}

/// The sweep's workload: an early-cutoff chain, a diamond, and a
/// collection feeding a `group_by`, over four inputs. Every query is a
/// known pure function of the inputs, so a final read checks (V).
struct Sweep<C: Cells> {
    x: [Incr<u64>; 4],
    chain: Incr<u64>,
    diamond: Incr<u64>,
    col: IncrCollection<u64, C>,
    groups: GroupedCollection<u64, u64, C>,
    total: Incr<u64>,
}

impl<C: Cells> Sweep<C> {
    fn build(rt: &Runtime<C>) -> Self {
        let x = [0, 1, 2, 3].map(|i| rt.create_input(i as u64));
        let [x0, x1, x2, _] = x;
        let q1 = rt.create_query(move |rt: &Runtime<C>| rt.get(x0) / 2);
        let q2 = rt.create_query(move |rt: &Runtime<C>| rt.get(q1) + rt.get(x1));
        let chain = rt.create_query(move |rt: &Runtime<C>| rt.get(q2) / 3);
        let a = rt.create_query(move |rt: &Runtime<C>| rt.get(x2) + 1);
        let b = rt.create_query(move |rt: &Runtime<C>| rt.get(x2) * 2);
        let diamond = rt.create_query(move |rt: &Runtime<C>| rt.get(a) + rt.get(b));
        let col = rt.create_collection::<u64>();
        let groups = col.group_by(rt, |v| v % 3);
        let total = col.count(rt);
        Self {
            x,
            chain,
            diamond,
            col,
            groups,
            total,
        }
    }

    /// One delete-and-recycle cycle: an input and a query over it and
    /// x[3], read, then deleted downstream first.
    fn churn(&self, rt: &Runtime<C>, value: u64) {
        let x3 = self.x[3];
        let v = rt.create_input(value);
        let w = rt.create_query(move |rt: &Runtime<C>| rt.get(v) + rt.get(x3));
        rt.get(w);
        rt.delete_node(w);
        rt.delete_node(v);
    }

    fn mutate(&self, rt: &Runtime<C>, rng: &mut Rng, present: &mut HashSet<u64>, next: &mut u64) {
        match rng.next(4) {
            0 => rt.set(self.x[rng.next(4) as usize], rng.next(50)),
            1 => self.churn(rt, rng.next(50)),
            2 => {
                *next += 1;
                self.col.insert(rt, *next);
                present.insert(*next);
            }
            _ => {
                if let Some(&v) = present.iter().next() {
                    assert!(self.col.delete(rt, &v));
                    present.remove(&v);
                }
            }
        }
    }

    /// (I), edge completeness and the watermark, then (V) for the
    /// scalar queries and the collection count.
    fn check(&self, rt: &Runtime<C>, present: &HashSet<u64>) {
        rt.assert_invariant_i();
        rt.assert_edges_complete();
        rt.assert_walks_complete();
        let [x0, x1, x2, _] = self.x.map(|h| rt.get(h));
        assert_eq!(rt.get(self.chain), (x0 / 2 + x1) / 3, "(V): chain");
        assert_eq!(rt.get(self.diamond), (x2 + 1) + x2 * 2, "(V): diamond");
        assert_eq!(rt.get(self.total), present.len() as u64, "(V): count");
    }

    /// Group counts are queries created on demand, so the sweep checks
    /// them once, at the end.
    fn check_groups(&self, rt: &Runtime<C>, present: &HashSet<u64>) {
        // The group_by operator applies upstream deltas when its version
        // node is read.
        rt.get(self.groups.version_node());
        for key in 0..3 {
            let expected = present.iter().filter(|v| *v % 3 == key).count() as u64;
            let got = self
                .groups
                .get_group(&key)
                .map_or(0, |g| rt.get(g.count(rt)));
            assert_eq!(got, expected, "(V): group {key}");
        }
    }
}

#[test]
fn sweep_local() {
    let rt = LocalRt::new();
    let sweep = Sweep::build(&rt);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut present = HashSet::new();
    let mut next = 0;
    for i in 0..iters(4000) {
        sweep.mutate(&rt, &mut rng, &mut present, &mut next);
        if rng.next(3) == 0 {
            rt.get(sweep.chain);
            rt.get(sweep.diamond);
            rt.get(sweep.groups.version_node());
        }
        if i % 97 == 0 {
            sweep.check(&rt, &present);
        }
    }
    sweep.check(&rt, &present);
    sweep.check_groups(&rt, &present);
}

#[test]
fn sweep_shared() {
    let rt = Arc::new(Rt::new());
    let sweep = Arc::new(Sweep::build(&rt));
    let n = iters(30000);
    let mut handles = Vec::new();

    for seed in 1..=2_u64 {
        let (rt, sweep) = (Arc::clone(&rt), Arc::clone(&sweep));
        handles.push(thread::spawn(move || {
            let mut rng = Rng(seed * 0x2545_f491_4f6c_dd1d);
            for _ in 0..n {
                rt.set(sweep.x[rng.next(4) as usize], rng.next(50));
            }
        }));
    }
    for _ in 0..2 {
        let (rt, sweep) = (Arc::clone(&rt), Arc::clone(&sweep));
        handles.push(thread::spawn(move || {
            for _ in 0..n {
                rt.get(sweep.chain);
                rt.get(sweep.diamond);
                rt.get(sweep.total);
                rt.get(sweep.groups.version_node());
            }
        }));
    }
    {
        let (rt, sweep) = (Arc::clone(&rt), Arc::clone(&sweep));
        handles.push(thread::spawn(move || {
            for i in 0..n as u64 {
                sweep.churn(&rt, i);
            }
        }));
    }
    let writer = {
        let (rt, sweep) = (Arc::clone(&rt), Arc::clone(&sweep));
        thread::spawn(move || {
            let mut rng = Rng(0xdead_beef_cafe_f00d);
            let mut present = HashSet::new();
            for i in 0..n as u64 {
                if rng.next(3) == 0 {
                    if let Some(&v) = present.iter().next() {
                        assert!(sweep.col.delete(&rt, &v));
                        present.remove(&v);
                    }
                } else {
                    sweep.col.insert(&rt, i);
                    present.insert(i);
                }
            }
            present
        })
    };
    for h in handles {
        h.join().unwrap();
    }
    let present = writer.join().unwrap();

    sweep.check(&rt, &present);
    sweep.check_groups(&rt, &present);
}

pub(super) fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(s) => *s,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(s) => (*s).to_string(),
            Err(_) => "<non-string panic payload>".to_string(),
        },
    }
}

/// A query over `x` whose closure counts its runs.
fn counted_query(rt: &Rt, x: Incr<u64>, add: u64) -> (Incr<u64>, Arc<AtomicUsize>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let q = {
        let runs = Arc::clone(&runs);
        rt.create_query(move |rt: &Rt| {
            runs.fetch_add(1, Ordering::SeqCst);
            rt.get(x) + add
        })
    };
    (q, runs)
}

/// Bug (d), the root: Q is deleted and its slot recycled while its first
/// compute is parked between the closure clone and the claim. The old
/// closure must not run against the new occupant, and the read of the
/// stale handle panics as stale.
#[test]
fn compute_vs_delete_recycle() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(7_u64);
    let (q, old_runs) = counted_query(&rt, x, 1);
    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::BeforeClaim(id(q)), gate.clone())]);

    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.get(q))))
    };
    gate.wait_parked();
    rt.delete_node(q);
    let (new, new_runs) = counted_query(&rt, x, 1000);
    assert_eq!(
        new.slot(),
        q.slot(),
        "the delete should free Q's slot for reuse"
    );
    gate.release();

    let outcome = reader.join().unwrap();
    let msg = panic_message(outcome.expect_err("the read of a deleted node returned a value"));
    assert!(msg.contains("stale"), "got: {msg}");
    assert_eq!(
        old_runs.load(Ordering::SeqCst),
        0,
        "the deleted query's closure ran against the recycled slot"
    );
    assert_eq!(new_runs.load(Ordering::SeqCst), 0);
    assert_eq!(rt.state_of(new), NodeState::New);
    assert_eq!(rt.get(new), 1007);
    rt.assert_invariant_i();
}

/// Bug (d), a walk entry: the Dirty transitive dep D of the root R is
/// deleted (with P and R) and its slot recycled while the walk's compute
/// of D is parked between the hold read and the claim. The old closure
/// must not run against the new occupant.
#[test]
fn walk_entry_vs_delete_recycle() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(0_u64);
    let (d, d_runs) = counted_query(&rt, x, 1);
    let p = rt.create_query(move |rt: &Rt| rt.get(d) + 10);
    let r = rt.create_query(move |rt: &Rt| rt.get(p) + 100);
    assert_eq!(rt.get(r), 111);
    rt.set(x, 1);
    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::BeforeClaim(id(d)), gate.clone())]);

    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.get(r))))
    };
    gate.wait_parked();
    assert_eq!(rt.delete_with_dependents(d), 3);
    let (new, new_runs) = counted_query(&rt, x, 1000);
    assert_eq!(
        new.slot(),
        d.slot(),
        "the cascade frees D's slot last, so it is reused first"
    );
    gate.release();

    let outcome = reader.join().unwrap();
    assert!(
        outcome.is_err(),
        "the read of a deleted root returned a value"
    );
    assert_eq!(
        d_runs.load(Ordering::SeqCst),
        1,
        "the deleted dep's closure ran against the recycled slot"
    );
    assert_eq!(new_runs.load(Ordering::SeqCst), 0);
    assert_eq!(rt.state_of(new), NodeState::New);
    assert_eq!(rt.get(new), 1001);
    rt.assert_invariant_i();
}

/// Bug (d) under `Local`: D's recompute deletes the walk root Q and
/// recycles its slot as an input of the same type, so the walk's root
/// entry and the read that follows it both meet a different node.
#[test]
fn local_dep_that_recycles_the_walk_root_panics_stale() {
    let rt = LocalRt::new();
    let x = rt.create_input(0_u64);
    let target: Arc<Mutex<Option<Incr<u64>>>> = Arc::new(Mutex::new(None));
    let recycled: RecycledSlot = Arc::new(Mutex::new(None));
    let d = {
        let target = Arc::clone(&target);
        let recycled = Arc::clone(&recycled);
        rt.create_query(move |rt: &LocalRt| {
            let xv = rt.get(x);
            if xv == 1 {
                let q = target
                    .lock()
                    .unwrap()
                    .expect("the target is set before the recompute");
                rt.delete_node(q);
                *recycled.lock().unwrap() = Some(rt.create_input_unchecked(5_u64));
            }
            xv + 1
        })
    };
    let q_runs = Arc::new(AtomicUsize::new(0));
    let q = {
        let q_runs = Arc::clone(&q_runs);
        rt.create_query(move |rt: &LocalRt| {
            q_runs.fetch_add(1, Ordering::SeqCst);
            rt.get(d) + 10
        })
    };
    *target.lock().unwrap() = Some(q);
    assert_eq!(rt.get(q), 11);
    rt.set(x, 1);

    let outcome = catch_unwind(AssertUnwindSafe(|| rt.get(q)));
    let msg = panic_message(outcome.expect_err("the read returned a value from the recycled slot"));
    assert!(msg.contains("stale"), "got: {msg}");
    let new = recycled.lock().unwrap().expect("D's recompute ran");
    assert_eq!(new.slot(), q.slot());
    assert_eq!(
        q_runs.load(Ordering::SeqCst),
        1,
        "the deleted query's closure ran against the recycled slot"
    );
    assert_eq!(rt.get(new), 5);
    assert_eq!(rt.get(d), 2);
    rt.assert_invariant_i();
}

/// Parks a `get(v)` on another thread at `point`; the join handle yields
/// the value or the panic message.
fn park_get(
    rt: &Arc<Rt>,
    v: Incr<u64>,
    point: ParkPoint,
) -> (Gate, thread::JoinHandle<Result<u64, String>>) {
    let gate = Gate::new();
    park_at(rt, vec![(point, gate.clone())]);
    let reader = {
        let rt = Arc::clone(rt);
        thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.get(v))).map_err(panic_message))
    };
    gate.wait_parked();
    (gate, reader)
}

#[test]
fn get_after_handle_check_vs_delete_recycle() {
    let rt = Arc::new(Rt::new());
    let v = rt.create_input(7_u64);
    let (gate, reader) = park_get(&rt, v, ParkPoint::GetAfterHandleCheck(id(v)));
    rt.delete_node(v);
    let new = rt.create_input(99_u64);
    assert_eq!(new.slot(), v.slot());
    gate.release();

    let msg = reader
        .join()
        .unwrap()
        .expect_err("the read returned the recycled slot's value");
    assert!(msg.contains("stale"), "got: {msg}");
    assert_eq!(rt.get(new), 99);
}

#[test]
fn get_before_read_vs_delete() {
    let rt = Arc::new(Rt::new());
    let v = rt.create_input(7_u64);
    let (gate, reader) = park_get(&rt, v, ParkPoint::GetBeforeRead(id(v)));
    rt.delete_node(v);
    gate.release();

    let msg = reader
        .join()
        .unwrap()
        .expect_err("the read of a deleted node returned a value");
    assert!(msg.contains("stale"), "got: {msg}");
    assert!(!msg.contains("never written"), "got: {msg}");
}

#[test]
fn get_before_read_vs_delete_recycle() {
    let rt = Arc::new(Rt::new());
    let v = rt.create_input(7_u64);
    let (gate, reader) = park_get(&rt, v, ParkPoint::GetBeforeRead(id(v)));
    rt.delete_node(v);
    let new = rt.create_input(99_u64);
    assert_eq!(new.slot(), v.slot());
    gate.release();

    let msg = reader
        .join()
        .unwrap()
        .expect_err("the read returned the recycled slot's value");
    assert!(msg.contains("stale"), "got: {msg}");
    assert_eq!(rt.get(new), 99);
}

/// A read whose walk lands its node New as a member of a cycle holds the
/// landing until the read ends (`crate::hold`), so another thread's claim
/// of the node waits for the read instead of claiming it between the
/// landing and the read's load of the state and group, where the read
/// once panicked that a Clean node had no value (replay seeds 0x33e9,
/// 0xa55 and 0x19f0 on two threads).
#[test]
fn read_of_cycle_member_vs_claim_after_landing() {
    let rt = Arc::new(Rt::new());
    let inp = rt.create_input(0_u64);
    // While inp is 0, `failed` reads itself, and a closure that catches
    // its cycle lands poisoned.
    let self_cell: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::default();
    let failed = {
        let self_cell = Arc::clone(&self_cell);
        rt.create_query(move |rt: &Rt| match rt.get(inp) {
            0 => rt.get(*self_cell.get().unwrap()),
            v => v,
        })
    };
    self_cell.set(failed).unwrap();
    let in_a = Gate::new();
    let b_cell: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::default();
    let a = {
        let (in_a, b_cell) = (in_a.clone(), Arc::clone(&b_cell));
        rt.create_query(move |rt: &Rt| {
            in_a.park();
            let b = *b_cell.get().unwrap();
            catch_unwind(AssertUnwindSafe(|| rt.try_get(b).unwrap_or(7777))).unwrap_or(9999)
        })
    };
    // b catches the cycle of a read, so the episode lands Dirty and its
    // members New, with their group.
    let b = rt.create_query(move |rt: &Rt| {
        let v = catch_unwind(AssertUnwindSafe(|| rt.get(failed))).unwrap_or(1);
        catch_unwind(AssertUnwindSafe(|| rt.get(a) + v)).unwrap_or(9999)
    });
    b_cell.set(b).unwrap();
    let x = rt.create_query(move |rt: &Rt| rt.try_get(a).map_or(1000, |v| v + 1));

    let before_read = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::GetBeforeRead(id(a)), before_read.clone())],
    );
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.get(x))).map_err(panic_message))
    };
    before_read.wait_parked();
    assert_eq!(rt.state_of(a), NodeState::Computing);
    assert!(rt.cycle_of(a).is_some());
    // So that the claimer's walk recomputes the failed dep instead of
    // panicking on it before it claims a.
    rt.set(inp, 1);

    in_a.arm();
    let claimer = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || {
            catch_unwind(AssertUnwindSafe(|| rt.try_get(a))).map_err(panic_message)
        })
    };
    thread::sleep(Duration::from_millis(50));
    assert!(!in_a.reached(), "the claimer ran a while the read held it");
    before_read.release();
    assert_eq!(reader.join().unwrap(), Ok(1000));
    in_a.wait_parked();
    in_a.release();

    let cycle = claimer
        .join()
        .unwrap()
        .unwrap()
        .expect_err("a is in a cycle");
    assert_eq!(cycle.participants().len(), 2);
    rt.assert_invariant_i();
}

/// a <-> b, a cycle `a`'s fallback resolves, with `b` swallowing the panic
/// of a failing node, so that every episode lands its members Dirty, and
/// `x` reading an input `t` and `a`. Setting `t` leaves the revision whose
/// memo `b`'s caught panic keeps, and makes the next read rerun `x`.
/// Returns `x`, `a`, `t` and the gate in `a`'s closure.
fn poisoned_episode(rt: &Rt) -> (Incr<u64>, Incr<u64>, Incr<u64>, Gate) {
    let t = rt.create_input(0_u64);
    let failed = rt.create_query(|_: &Rt| -> u64 { panic!("failed") });
    let in_a = Gate::new();
    let b_cell: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::default();
    let a = {
        let (in_a, b_cell) = (in_a.clone(), Arc::clone(&b_cell));
        rt.create_query_with_fallback(
            move |rt: &Rt| {
                in_a.park();
                rt.get(*b_cell.get().unwrap()) + 1
            },
            |_: &crate::Cycle| 0,
        )
    };
    let b = rt.create_query(move |rt: &Rt| {
        let v = catch_unwind(AssertUnwindSafe(|| rt.get(failed))).unwrap_or(1);
        rt.get(a) + v
    });
    b_cell.set(b).unwrap();
    let x = rt.create_query(move |rt: &Rt| rt.get(t) + rt.get(a) + 100);
    (x, a, t, in_a)
}

/// The episode that `x`'s read of `a` runs, as that read's walk root,
/// lands `a` Dirty with the memo `b`'s caught panic keeps for the
/// revision, and holds the landing until the read ends, so another
/// thread's read of `a` waits instead of claiming it under the read (seed
/// 0x74f's race). Once the read has published, that read takes the memo
/// and runs nothing.
#[test]
fn reader_of_group_episode_root_holds_it_then_shares_its_memo() {
    let rt = Arc::new(Rt::new());
    let (x, a, t, in_a) = poisoned_episode(&rt);
    rt.get(x);
    rt.set(t, 1);
    assert_eq!(rt.state_of(a), NodeState::Dirty);
    let walked = Gate::new();
    park_at(&rt, vec![(ParkPoint::AfterWalk(id(a)), walked.clone())]);
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(x))
    };
    walked.wait_parked();
    assert_eq!(rt.state_of(a), NodeState::Computing);
    in_a.arm();
    let claimer = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(a))
    };
    thread::sleep(Duration::from_millis(50));
    assert!(
        !claimer.is_finished(),
        "the claimer read `a` while the read held it"
    );
    walked.release();
    let x_value = reader.join().unwrap();
    let deadline = std::time::Instant::now() + PARK_TIMEOUT;
    while !claimer.is_finished() && std::time::Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    let ran = in_a.0.reached_rx.lock().unwrap().try_recv().is_ok();
    // A release sent ahead lets a later armed park of `a` pass at once.
    in_a.release();
    assert!(!ran, "the claimer ran `a` after the read");
    assert_eq!(x_value, claimer.join().unwrap() + 101);
    rt.assert_invariant_i();
}

/// `a` and `b` read each other while `g` is 0, and `a` reads only `g`
/// otherwise. Returns them with `a` Clean at 7.
pub(super) fn clean_node_that_can_turn_cyclic(rt: &Rt) -> (Incr<u64>, Incr<i64>, Incr<i64>) {
    let g = rt.create_input(1_u64);
    let b_cell: Arc<std::sync::OnceLock<Incr<i64>>> = Arc::default();
    let a = {
        let b_cell = Arc::clone(&b_cell);
        rt.create_query(move |rt: &Rt| {
            if rt.get(g) == 0 {
                rt.try_get(*b_cell.get().unwrap()).map_or(-1, |v| v + 1)
            } else {
                7
            }
        })
    };
    let b = rt.create_query(move |rt: &Rt| rt.try_get(a).map_or(-1, |v| v * 2));
    b_cell.set(b).unwrap();
    assert_eq!(rt.get(a), 7);
    assert_eq!(rt.state_of(a), NodeState::Clean);
    (g, a, b)
}

/// A read that loaded `a` Clean parks before its slot read; another
/// thread's set and episode land `a` Cyclic and clear its slot. The read
/// goes on from the state it reloads and returns the cycle, where it
/// used to panic that a Clean node had no value.
fn clean_read_vs_cyclic_landing<R: Send + 'static>(
    read: fn(&Rt, Incr<i64>) -> R,
) -> Result<R, String> {
    let rt = Arc::new(Rt::new());
    let (g, a, b) = clean_node_that_can_turn_cyclic(&rt);
    let before_read = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::GetBeforeRead(id(a)), before_read.clone())],
    );
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || {
            catch_unwind(AssertUnwindSafe(|| read(&rt, a))).map_err(panic_message)
        })
    };
    before_read.wait_parked();

    rt.set(g, 0);
    let cycle = rt.try_get(b).expect_err("b is in a cycle");
    assert_eq!(cycle.participants().len(), 2);
    assert_eq!(rt.state_of(a), NodeState::Cyclic);
    before_read.release();

    let read = reader.join().unwrap();
    rt.assert_invariant_i();
    read
}

#[test]
fn clean_try_get_vs_cyclic_landing() {
    let read =
        clean_read_vs_cyclic_landing(|rt, a| rt.try_get(a).map_err(|c| c.participants().len()));
    assert_eq!(read, Ok(Err(2)));
}

#[test]
fn clean_get_vs_cyclic_landing() {
    let msg = clean_read_vs_cyclic_landing(|rt, a| rt.get(a)).expect_err("a has no value");
    assert!(msg.contains("dependency cycle"), "got: {msg}");
}

#[cfg(debug_assertions)]
#[test]
fn a_get_waiting_on_a_compute_checks_the_locks_it_holds() {
    let rt = Arc::new(Rt::new());
    let gate = Gate::new();
    let q = {
        let gate = gate.clone();
        rt.create_query(move |_: &Rt| {
            gate.park();
            1_u64
        })
    };
    gate.arm();
    let computer = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(q))
    };
    gate.wait_parked();
    let read = catch_unwind(AssertUnwindSafe(|| {
        let _log = crate::lock_order::acquire(crate::lock_order::LockKind::CollectionLog);
        rt.get(q)
    }));
    gate.release();
    assert_eq!(computer.join().unwrap(), 1);
    let msg = panic_message(read.expect_err("a claim wait under a collection log"));
    assert!(
        msg.contains("waiting on a claim while holding [CollectionLog]"),
        "got: {msg}"
    );
}

/// A walk entry's claim released unrun after the episode check: a reader
/// waiting on that claim goes on, and reads the node once it lands.
#[test]
fn a_claim_released_unrun_ends_its_waiters_wait() {
    let rt = Arc::new(Rt::new());
    let (x, q, d) = chain(&rt);
    rt.set(x, 2);
    let (claimed, waiting) = (Gate::new(), Gate::new());
    // The walker passes the reader's park point before its claim, so the
    // reader's gate is armed only once the walker holds the claim.
    let points = [
        (ParkPoint::AfterWalkClaim(id(q)), claimed.clone()),
        (ParkPoint::BeforeComputeOne(id(q)), waiting.clone()),
    ];
    claimed.arm();
    let installed = rt.park_hook.set(Box::new(move |p| {
        for (point, gate) in &points {
            if *point == p {
                gate.park();
            }
        }
    }));
    assert!(installed.is_ok());
    let walker = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || {
            super::FORCE_REPLAN_HERE.with(|f| f.set(true));
            rt.get(d)
        })
    };
    claimed.wait_parked();
    waiting.arm();
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(q))
    };
    waiting.wait_parked();
    waiting.release();
    thread::sleep(Duration::from_millis(20));
    assert_eq!(rt.nodes.get(q.slot()).state(), NodeState::Computing);
    claimed.release();
    assert_eq!(reader.join().unwrap(), 1);
    assert_eq!(walker.join().unwrap(), 101);
    // A forced revisit releases claims unrun too.
    if !super::forced_revisit() {
        assert_eq!(rt.unrun_unclaims.load(Ordering::Relaxed), 1);
    }
    rt.assert_invariant_i();
}

/// A read of X finds it Computing on another thread and parks before its
/// walk waits on the root. X lands, is deleted, and its slot is recycled
/// into Z, which reads D and is left Dirty by a set under D. The read
/// panics as stale, as in the serial order with the delete first, and D's
/// closure does not run on its behalf.
#[test]
fn shared_root_wait_finds_its_slot_recycled_into_a_dirty_node() {
    let rt = Arc::new(Rt::new());
    let i = rt.create_input(1_u64);
    let d_runs = Arc::new(AtomicUsize::new(0));
    let d = {
        let d_runs = Arc::clone(&d_runs);
        rt.create_query(move |rt: &Rt| {
            d_runs.fetch_add(1, Ordering::SeqCst);
            rt.get(i) * 10
        })
    };
    let computing = Gate::new();
    let x = {
        let computing = computing.clone();
        rt.create_query(move |_: &Rt| {
            computing.park();
            5_u64
        })
    };
    computing.arm();
    let computer = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(x))
    };
    computing.wait_parked();
    // The computer passed its own `BeforeComputeOne(x)` before parking.
    let waiting = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::BeforeComputeOne(id(x)), waiting.clone())],
    );
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.get(x))).map_err(panic_message))
    };
    waiting.wait_parked();
    computing.release();
    assert_eq!(computer.join().unwrap(), 5);
    rt.delete_node(x);
    let z = rt.create_query(move |rt: &Rt| rt.get(d) + 1);
    assert_eq!(z.slot(), x.slot(), "Z reuses X's slot");
    assert_eq!(rt.get(z), 11);
    rt.set(i, 2);
    assert_eq!(rt.state_of(z), NodeState::Dirty);
    assert_eq!(d_runs.load(Ordering::SeqCst), 1);
    waiting.release();
    let got = reader.join().unwrap();
    assert!(got.as_ref().is_err_and(|m| m.contains("stale")), "{got:?}");
    assert_eq!(
        d_runs.load(Ordering::SeqCst),
        1,
        "D ran on the reader's behalf"
    );
    assert_eq!(rt.get(z), 21, "Z kept its own landing");
    rt.assert_invariant_i();
}

#[derive(Clone, Copy)]
enum DepEntry {
    Visit,
    Pending,
}

/// A read of D walks D's deps and parks at E's visit or pending entry.
/// Another thread recomputes D, which no longer reads E, and deletes E:
/// parked after its Dead claim, before the generation bump, or through
/// to a recycle of E's slot into Z. The read gets D's new value, as in
/// the serial order with the recompute and delete first, and Z's closure
/// does not run on its behalf.
fn walk_entry_finds_its_dep_deleted(at: DepEntry, recycled: bool) {
    let rt = Arc::new(Rt::new());
    let i = rt.create_input(0_u64);
    let j = rt.create_input(0_u64);
    let e = rt.create_query(move |rt: &Rt| rt.get(j) + 1);
    let d = rt.create_query(move |rt: &Rt| if rt.get(i) == 0 { rt.get(e) } else { 5 });
    assert_eq!(rt.get(d), 1);
    rt.set(j, 1);
    let (entry, delete) = (Gate::new(), Gate::new());
    let point = match at {
        DepEntry::Visit => ParkPoint::BeforeVisit(id(e)),
        DepEntry::Pending => ParkPoint::BeforeComputeOne(id(e)),
    };
    park_at(
        &rt,
        vec![
            (point, entry.clone()),
            (ParkPoint::DeleteAfterCheck(id(e)), delete.clone()),
        ],
    );
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.get(d))).map_err(panic_message))
    };
    entry.wait_parked();
    rt.set(i, 1);
    // The recompute's own walk passes E's points with their gates spent.
    assert_eq!(rt.get(d), 5);
    let z_runs = Arc::new(AtomicUsize::new(0));
    let new_z = || {
        let z_runs = Arc::clone(&z_runs);
        let z = rt.create_query(move |_: &Rt| {
            z_runs.fetch_add(1, Ordering::SeqCst);
            7_u64
        });
        assert_eq!(z.slot(), e.slot(), "Z reuses E's slot");
        z
    };
    let (deleter, z) = if recycled {
        delete.release();
        rt.delete_node(e);
        let z = new_z();
        assert_eq!(rt.state_of(z), NodeState::New);
        (None, Some(z))
    } else {
        let deleter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.delete_node(e))
        };
        delete.wait_parked();
        assert_eq!(rt.state_of(e), NodeState::Dead);
        (Some(deleter), None)
    };
    entry.release();
    let got = reader.join().unwrap();
    assert_eq!(got, Ok(5));
    if let Some(deleter) = deleter {
        delete.release();
        deleter.join().unwrap();
    }
    let z = z.unwrap_or_else(new_z);
    assert_eq!(
        z_runs.load(Ordering::SeqCst),
        0,
        "Z ran on the reader's behalf"
    );
    assert_eq!(rt.get(z), 7);
    rt.assert_invariant_i();
}

#[test]
fn shared_walk_visit_finds_its_dep_recycled() {
    walk_entry_finds_its_dep_deleted(DepEntry::Visit, true);
}

#[test]
fn shared_walk_visit_finds_its_dep_dead() {
    walk_entry_finds_its_dep_deleted(DepEntry::Visit, false);
}

#[test]
fn shared_walk_entry_finds_its_slot_recycled_into_a_new_node() {
    walk_entry_finds_its_dep_deleted(DepEntry::Pending, true);
}

#[test]
fn shared_walk_entry_finds_its_dep_dead() {
    walk_entry_finds_its_dep_deleted(DepEntry::Pending, false);
}

#[test]
fn a_claim_from_new_released_unrun_after_a_set_stays_new() {
    let rt = Arc::new(Rt::new());
    let (x, q, d) = chain(&rt);
    // Q New with its edges, as a stale recovery or an episode leaves it,
    // so the set's dirty walk reaches Q's claim.
    rt.nodes.get(q.slot()).set_state(NodeState::New.as_u8());
    rt.set(x, 2);
    let (claimed, rerun) = (Gate::new(), Gate::new());
    let points = [
        (ParkPoint::AfterWalkClaim(id(q)), claimed.clone()),
        (ParkPoint::BeforeComputeOne(id(q)), rerun.clone()),
    ];
    claimed.arm();
    let installed = rt.park_hook.set(Box::new(move |p| {
        for (point, gate) in &points {
            if *point == p {
                gate.park();
            }
        }
    }));
    assert!(installed.is_ok());
    let walker = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || {
            super::FORCE_REPLAN_HERE.with(|f| f.set(true));
            rt.get(d)
        })
    };
    claimed.wait_parked();
    rt.set(x, 4);
    assert_eq!(rt.nodes.get(q.slot()).state(), NodeState::ComputingDirty);
    rerun.arm();
    claimed.release();
    rerun.wait_parked();
    assert_eq!(rt.nodes.get(q.slot()).state(), NodeState::New);
    rerun.release();
    assert_eq!(walker.join().unwrap(), 102);
    // A forced revisit releases claims unrun too.
    if !super::forced_revisit() {
        assert_eq!(rt.unrun_unclaims.load(Ordering::Relaxed), 1);
    }
    rt.assert_invariant_i();
}

/// a = get(b); b = try_get(c), or 7777 on a cycle; c = get(a) with a
/// Fallback of 3. A read of `a` from New meets the cycle at `a` and gets
/// 7777, leaving `a` Dirty; a read after that walks to `c` first and gets 3.
fn caught_cycle_under_a_fallback(rt: &Rt) -> (Incr<u64>, Incr<u64>) {
    let cell: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
    let b_reads = cell.clone();
    let b = rt.create_query(move |rt| rt.try_get(*b_reads.get().unwrap()).unwrap_or(7777));
    let a = rt.create_query(move |rt| rt.get(b));
    let c = rt.create_query_with_fallback(move |rt| rt.get(a), |_| 3_u64);
    cell.set(c).unwrap();
    (a, b)
}

/// Reads of `a` one after the other give 7777 and then 3.
#[test]
fn a_caught_cycle_under_a_fallback_answers_a_second_read_from_its_walk() {
    let rt = Rt::new();
    let (a, _) = caught_cycle_under_a_fallback(&rt);
    assert_eq!(rt.get(a), 7777);
    assert_eq!(rt.get(a), 3);
}

/// Thread A loads `a` New, then B claims and lands it Dirty while A waits
/// on the claim. A's read comes after B's, so it gets 3, as a serial
/// second read does: its walk plans `a` again and expands its deps first,
/// where verifying `a` under its own claim met the cycle at `a`.
#[test]
fn a_walk_that_waited_out_a_dirty_landing_plans_the_node_again() {
    let rt = Arc::new(Rt::new());
    let (a, b) = caught_cycle_under_a_fallback(&rt);
    let (loaded, holding) = (Gate::new(), Gate::new());
    let points = [
        (ParkPoint::BeforeClaim(id(a)), "A", loaded.clone()),
        (ParkPoint::BeforeComputeOne(id(b)), "B", holding.clone()),
    ];
    loaded.arm();
    holding.arm();
    let installed = rt.park_hook.set(Box::new(move |p| {
        let name = thread::current().name().map(str::to_owned);
        for (point, who, gate) in &points {
            if *point == p && name.as_deref() == Some(*who) {
                gate.park();
            }
        }
    }));
    assert!(installed.is_ok());
    let first = {
        let rt = Arc::clone(&rt);
        thread::Builder::new()
            .name("A".into())
            .spawn(move || catch_unwind(AssertUnwindSafe(|| rt.get(a))))
            .unwrap()
    };
    loaded.wait_parked();
    let second = {
        let rt = Arc::clone(&rt);
        thread::Builder::new()
            .name("B".into())
            .spawn(move || rt.try_get(a))
            .unwrap()
    };
    holding.wait_parked();
    loaded.release();
    thread::sleep(Duration::from_millis(20));
    holding.release();
    assert_eq!(second.join().unwrap(), Ok(7777));
    assert_eq!(first.join().unwrap().ok(), Some(3));
}

/// Group verify releases the members it claimed unrun when a later member
/// fails its claim: A's verify of `p` in the group {p, x, q} claims `x`,
/// and `q`, which left the group Clean, fails. B planned its read of `x`
/// before A's claim and waited it out, so it finds `x` Dirty as it planned
/// it and claims it instead of revisiting it.
#[test]
fn a_walk_that_waited_out_a_released_group_claim_keeps_its_plan() {
    let rt = Arc::new(Rt::new());
    let k = rt.create_input(0_i64);
    let p = rt.create_query(move |rt| rt.get(k) + 1);
    let x = rt.create_query(move |rt| rt.get(k) * 10);
    let q = rt.create_query(|_| 5_i64);
    assert_eq!((rt.get(p), rt.get(x), rt.get(q)), (1, 0, 5));
    assert!(p.slot() < x.slot() && x.slot() < q.slot());
    rt.force_group_for_test(&[p.erase(), x.erase(), q.erase()]);
    let (node, flags) = rt.nodes.get_with_flags(q.slot());
    rt.leave_group(id(q), node, flags, NodeState::Clean);
    rt.set(k, 1);
    let (planned, claimed, waits) = (Gate::new(), Gate::new(), Gate::new());
    let points = [
        (ParkPoint::BeforeClaim(id(x)), "B", planned.clone()),
        (ParkPoint::GroupMemberClaimed(id(x)), "A", claimed.clone()),
        (ParkPoint::ComputeWaits(id(x)), "B", waits.clone()),
    ];
    for (_, _, gate) in &points {
        gate.arm();
    }
    let installed = rt.park_hook.set(Box::new(move |point| {
        let name = thread::current().name().map(str::to_owned);
        for (at, who, gate) in &points {
            if *at == point && name.as_deref() == Some(*who) {
                gate.park();
            }
        }
    }));
    assert!(installed.is_ok());
    let read = |name: &str, h: Incr<i64>| {
        let rt = Arc::clone(&rt);
        thread::Builder::new()
            .name(name.into())
            .spawn(move || rt.get(h))
            .unwrap()
    };
    let b = read("B", x);
    planned.wait_parked();
    let a = read("A", p);
    claimed.wait_parked();
    planned.release();
    waits.wait_parked();
    waits.release();
    claimed.release();
    assert_eq!((a.join().unwrap(), b.join().unwrap()), (2, 10));
    assert_eq!(rt.wait_revisits.load(Ordering::SeqCst), 0, "B revisited x");
    rt.assert_invariant_i();
}

/// Thread A's walk expands `p` as the group {p, x} and parks before its
/// compute; another thread lands both members out of the group, Dirty, and
/// with `regroup` lands {p, x, z} as a new group. A's claim of `p` then
/// plans it again as a read after those landings would: the node the new
/// plan expands (`x`, or `z`'s dep `w` outside the new group) runs as a
/// walk entry before `p` is claimed, not nested in `p`'s compute, which is
/// where a panicking member's failure is contained.
fn expanded_group_replaced_before_the_claim(regroup: bool) {
    let rt = Arc::new(Rt::new());
    let k = rt.create_input(0_i64);
    let w = rt.create_query(move |rt| rt.get(k) + 1);
    let x = rt.create_query(move |rt| rt.get(k) * 10);
    let z = rt.create_query(move |rt| rt.get(w) * 2);
    let p = rt.create_query(move |rt| rt.get(x) + rt.get(k));
    assert_eq!((rt.get(p), rt.get(z)), (0, 2));
    rt.force_group_for_test(&[p.erase(), x.erase()]);
    rt.set(k, 1);
    let probe = if regroup { id(w) } else { id(x) };
    let planned = Gate::new();
    planned.arm();
    let p_at_probe: Arc<Mutex<Vec<NodeState>>> = Arc::default();
    let installed = rt.park_hook.set(Box::new({
        let (rt, planned, p_at_probe) = (Arc::downgrade(&rt), planned.clone(), p_at_probe.clone());
        move |point| {
            if thread::current().name() != Some("A") {
                return;
            }
            if point == ParkPoint::BeforeComputeOne(id(p)) {
                planned.park();
            } else if point == ParkPoint::BeforeComputeOne(probe) {
                if let Some(rt) = rt.upgrade() {
                    p_at_probe.lock().unwrap().push(rt.state_of(p));
                }
            }
        }
    }));
    assert!(installed.is_ok());
    let a = {
        let rt = Arc::clone(&rt);
        thread::Builder::new()
            .name("A".into())
            .spawn(move || rt.get(p))
            .unwrap()
    };
    planned.wait_parked();
    for m in [p.erase(), x.erase()] {
        let (node, flags) = rt.nodes.get_with_flags(m.slot());
        rt.leave_group(NodeId(m.slot()), node, flags, NodeState::Dirty);
    }
    if regroup {
        rt.force_group_for_test(&[p.erase(), x.erase(), z.erase()]);
    }
    planned.release();
    assert_eq!(a.join().unwrap(), 11);
    assert_eq!(*p_at_probe.lock().unwrap(), [NodeState::Dirty]);
    assert_eq!((rt.get(p), rt.get(z)), (11, 4));
    rt.assert_invariant_i();
}

#[test]
fn a_walk_whose_expanded_group_dissolved_plans_its_entry_again() {
    expanded_group_replaced_before_the_claim(false);
}

#[test]
fn a_walk_whose_expanded_group_was_replaced_plans_its_entry_again() {
    expanded_group_replaced_before_the_claim(true);
}

/// As the regrouping case of `expanded_group_replaced_before_the_claim`,
/// but thread T then lands `p`, held, out of the new group and waits in its
/// closure. A's claim wait takes the landing over, which puts the new
/// group's record back as T's change, not A's, so A still plans `p` again.
#[test]
fn a_walk_that_took_over_a_held_landing_out_of_its_replaced_group_plans_its_entry_again() {
    let rt = Arc::new(Rt::new());
    let k = rt.create_input(0_i64);
    let w_runs = Arc::new(AtomicUsize::new(0));
    let w = {
        let w_runs = w_runs.clone();
        rt.create_query(move |rt| {
            w_runs.fetch_add(1, Ordering::SeqCst);
            rt.get(k) + 1
        })
    };
    let x = rt.create_query(move |rt| rt.get(k) * 10);
    let z = rt.create_query(move |rt| rt.get(w) * 2);
    let p = rt.create_query(move |rt| rt.get(x) + rt.get(k));
    let (planned, holding) = (Gate::new(), Gate::new());
    let left = Arc::new(AtomicBool::new(false));
    let t = {
        let (holding, left) = (holding.clone(), left.clone());
        rt.create_query(move |rt: &Rt| {
            if !left.swap(true, Ordering::SeqCst) {
                let (node, flags) = rt.nodes.get_with_flags(p.slot());
                rt.force_state_for_test(p, NodeState::Computing);
                let mut bag = Deferred::new();
                rt.open_hold(id(p), node, false, None, &mut bag);
                rt.land_single_as(id(p), node, flags, NodeState::Dirty, NodeState::Dirty, true);
                bag.drain(rt);
                holding.park_in_user_code();
            }
            0_i64
        })
    };
    assert_eq!((rt.get(p), rt.get(z)), (0, 2));
    w_runs.store(0, Ordering::SeqCst);
    rt.force_group_for_test(&[p.erase(), x.erase()]);
    rt.set(k, 1);
    planned.arm();
    holding.arm();
    let p_at_w: Arc<Mutex<Vec<NodeState>>> = Arc::default();
    let installed = rt.park_hook.set(Box::new({
        let (rt, planned, p_at_w) = (Arc::downgrade(&rt), planned.clone(), p_at_w.clone());
        move |point| {
            if thread::current().name() != Some("A") {
                return;
            }
            if point == ParkPoint::BeforeComputeOne(id(p)) {
                planned.park();
            } else if point == ParkPoint::BeforeComputeOne(id(w)) {
                if let Some(rt) = rt.upgrade() {
                    p_at_w.lock().unwrap().push(rt.state_of(p));
                }
            }
        }
    }));
    assert!(installed.is_ok());
    let read = |name: &str, h: Incr<i64>| {
        let rt = Arc::clone(&rt);
        thread::Builder::new()
            .name(name.into())
            .spawn(move || rt.get(h))
            .unwrap()
    };
    let a = read("A", p);
    planned.wait_parked();
    for m in [p.erase(), x.erase()] {
        let (node, flags) = rt.nodes.get_with_flags(m.slot());
        rt.leave_group(NodeId(m.slot()), node, flags, NodeState::Dirty);
    }
    rt.force_group_for_test(&[p.erase(), x.erase(), z.erase()]);
    let held = read("T", t);
    holding.wait_parked();
    planned.release();
    assert_eq!(a.join().unwrap(), 11);
    holding.release();
    assert_eq!(held.join().unwrap(), 0);
    let p_at_w = p_at_w.lock().unwrap().clone();
    assert!(
        !p_at_w.is_empty() && p_at_w.iter().all(|&s| s == NodeState::Dirty),
        "w ran under p's claim: {p_at_w:?}"
    );
    assert_eq!((rt.get(p), rt.get(z)), (11, 4));
    assert_eq!(w_runs.load(Ordering::SeqCst), 1);
    rt.assert_invariant_i();
}

/// Thread A's walk plans `a` New and parks at `at`; another read lands
/// `a` Dirty meanwhile. A's claim from Dirty plans `a` again, so it gets
/// 3, as a serial second read does.
fn planned_new_then_landed_dirty(at: fn(NodeId) -> ParkPoint) {
    let rt = Arc::new(Rt::new());
    let (a, _) = caught_cycle_under_a_fallback(&rt);
    let planned = Gate::new();
    planned.arm();
    let parks = planned.clone();
    let installed = rt.park_hook.set(Box::new(move |p| {
        if p == at(id(a)) && thread::current().name() == Some("A") {
            parks.park();
        }
    }));
    assert!(installed.is_ok());
    let first = {
        let rt = Arc::clone(&rt);
        thread::Builder::new()
            .name("A".into())
            .spawn(move || catch_unwind(AssertUnwindSafe(|| rt.get(a))))
            .unwrap()
    };
    planned.wait_parked();
    assert_eq!(rt.try_get(a), Ok(7777));
    assert_eq!(rt.state_of(a), NodeState::Dirty);
    planned.release();
    assert_eq!(first.join().unwrap().ok(), Some(3));
}

#[test]
fn a_claim_from_dirty_after_a_load_of_new_plans_the_node_again() {
    planned_new_then_landed_dirty(ParkPoint::BeforeClaim);
}

#[test]
fn a_dirty_landing_before_the_walks_first_load_plans_the_node_again() {
    planned_new_then_landed_dirty(ParkPoint::BeforeComputeOne);
}

/// A read whose walk lands its root Dirty returns that landing's value:
/// R, over a dependency cycle it catches, lands Dirty with its run count.
/// A second read's walk runs C as R's dep, before R's closure, so C's
/// cycle reaches that read without rerunning R, as on main
/// (B7-cycle-escapes-dirty-catcher). Parked between its walk and its slot
/// read, the first reader still returns its own landing.
#[test]
fn shared_read_returns_the_value_its_walk_landed_dirty() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let c_cell: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
    let cc = c_cell.clone();
    let c = rt.create_query(move |rt: &Runtime<Shared>| rt.get(*cc.get().unwrap()));
    c_cell.set(c).unwrap();
    let runs = Arc::new(AtomicUsize::new(0));
    let rr = runs.clone();
    let r = rt.create_query(move |rt: &Runtime<Shared>| {
        let n = rr.fetch_add(1, Ordering::SeqCst) as u64 + 1;
        rt.try_get(c).map_or(n * 100, |v| v + n)
    });
    let gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::AfterWalk(NodeId(r.slot())), gate.clone())],
    );
    let first = {
        let rt = rt.clone();
        thread::spawn(move || rt.get(r))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(r), NodeState::Dirty);
    let waits = rt.reservation_waits.load(Ordering::SeqCst);
    let second = {
        let rt = rt.clone();
        thread::spawn(move || rt.get(r))
    };
    while !second.is_finished() && rt.reservation_waits.load(Ordering::SeqCst) == waits {
        thread::sleep(Duration::from_millis(1));
    }
    gate.release();
    assert_eq!(first.join().unwrap(), 100);
    let cycle = panic_message(
        second
            .join()
            .expect_err("C's cycle reaches the second read"),
    );
    assert!(cycle.ends_with("dependency cycle: 0"), "{cycle}");
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

/// p stabilizes an observer of r, then reads x. r reads y and v = y + 1,
/// and once f is set only y; y then reads r and p. The observer's walk of
/// r from Dirty computes y before v: y's nested read of r leaves v Dirty
/// (tainted by y), r's read of y defers r into this thread's held map, and
/// y's read of p (on the stack below the walk) defers y too, so r stays
/// held by this thread. Another thread then releases claims before v,
/// whose forced replan check then finds a root this thread holds. Returns
/// p's read and whether the release ran.
fn observer_walk_with_its_root_held_here(release: bool) -> (Option<u64>, bool) {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let f = rt.create_input(0_u64);
    let cell: Arc<std::sync::OnceLock<(Incr<u64>, Incr<u64>)>> =
        Arc::new(std::sync::OnceLock::new());
    let y_reads = cell.clone();
    let y = rt.create_query_with_fallback(
        move |rt| {
            if rt.get(f) == 1 {
                let (r, p) = *y_reads.get().unwrap();
                rt.get(r) + rt.get(p)
            } else {
                rt.get(x)
            }
        },
        |_| 7_u64,
    );
    let v = rt.create_query(move |rt| rt.get(y) + 1);
    let r = rt.create_query_with_fallback(
        move |rt| {
            if rt.get(f) == 1 {
                rt.get(y)
            } else {
                rt.get(y) + rt.get(v)
            }
        },
        |_| 9_u64,
    );
    let p = rt.create_query_with_fallback(
        move |rt| {
            rt.stabilize();
            rt.get(x)
        },
        |_| 0_u64,
    );
    cell.set((r, p)).unwrap();
    assert_eq!(rt.get(r), 3);
    rt.observe(r, |_| {});
    assert_eq!(rt.get(p), 1);
    rt.set(f, 1);
    rt.set(x, 2);
    let released = Arc::new(AtomicBool::new(false));
    let hook_released = released.clone();
    let hook_rt = Arc::downgrade(&rt);
    let installed = rt.park_hook.set(Box::new(move |point| {
        let rt = hook_rt.upgrade().unwrap();
        if point == ParkPoint::BeforeComputeOne(id(v))
            && rt.own_claim_owner(id(r)).is_some()
            && release
            && !hook_released.swap(true, Ordering::Relaxed)
        {
            thread::spawn(move || rt.count_release([id(v).0]))
                .join()
                .unwrap();
        }
    }));
    assert!(installed.is_ok());
    super::FORCE_REPLAN_HERE.with(|f| f.set(true));
    let read = catch_unwind(AssertUnwindSafe(|| rt.get(p)));
    super::FORCE_REPLAN_HERE.with(|f| f.set(false));
    assert_eq!(rt.root_replans.load(Ordering::Relaxed), 0);
    (read.ok(), released.load(Ordering::Relaxed))
}

#[test]
fn an_observer_walk_replanned_with_its_root_held_here_reads_as_unreplanned() {
    let (unreplanned, _) = observer_walk_with_its_root_held_here(false);
    let (read, released) = observer_walk_with_its_root_held_here(true);
    assert!(released, "r was never held here");
    assert!(unreplanned.is_some());
    assert_eq!(read, unreplanned);
}

/// r reads a, which reads x. Another thread releases held claims before
/// each compute of a in the walk of r, as a budgeted read on another thread
/// does when it discards a doomed cycle's round each time it retries
/// (0x2997), so every plan of r is stale by the time it computes. The read
/// still finishes while the releases go on: they stop only after
/// `RELEASES`, far more than a bounded read waits out, and restarts from
/// its root at most `ROOT_RESTARTS` times.
#[test]
fn a_walk_another_thread_releases_under_on_every_plan_finishes() {
    const RELEASES: usize = 1000;
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let a = rt.create_query(move |rt| rt.get(x) + 1);
    let r = rt.create_query(move |rt| rt.get(a) + 1);
    assert_eq!(rt.get(r), 3);
    rt.set(x, 2);
    let releases = Arc::new(AtomicUsize::new(0));
    let hook_releases = releases.clone();
    let hook_rt = Arc::downgrade(&rt);
    let installed = rt.park_hook.set(Box::new(move |point| {
        if point == ParkPoint::BeforeComputeOne(id(a))
            && hook_releases.fetch_add(1, Ordering::Relaxed) < RELEASES
        {
            let rt = hook_rt.upgrade().unwrap();
            thread::spawn(move || rt.count_release([id(a).0]))
                .join()
                .unwrap();
        }
    }));
    assert!(installed.is_ok());
    assert_eq!(rt.get(r), 4);
    let released = releases.load(Ordering::Relaxed);
    assert!(
        released < RELEASES / 10,
        "the read restarted until the releases stopped: {released} releases, {} root replans",
        rt.root_replans.load(Ordering::Relaxed)
    );
    let replans = rt.root_replans.load(Ordering::Relaxed);
    assert!(
        replans <= super::ROOT_RESTARTS,
        "{replans} root replans past the bound"
    );
}

/// A released replan restarts only for a release after its plan that
/// landed one of the plan's nodes unrun, or whose slots the log has
/// evicted; release numbers compare across their wrap.
/// Pushes to `log` from another thread.
fn foreign(mut log: EpisodeLog, released: bool, number: u32, slots: Vec<u32>) -> EpisodeLog {
    thread::spawn(move || {
        log.push(released, number, thread_key(), slots);
        log
    })
    .join()
    .unwrap()
}

#[test]
fn an_episode_reaches_a_plan_only_through_another_threads_slots() {
    let mut log = foreign(EpisodeLog::default(), true, 7, vec![3, 4]);
    log = foreign(log, true, 8, vec![9]);
    log.push(true, 8, thread_key(), [1]);
    assert!(log.reached(7, [1, 9].into_iter()));
    assert!(!log.reached(7, [1].into_iter()), "an own release reached");
    assert!(!log.reached(7, [3, 4].into_iter()));
    assert!(!log.reached(8, [9].into_iter()));
    assert!(log.reached(6, [4].into_iter()));
    log.push(true, 9, 0, [12]);
    assert!(
        log.reached(8, [12].into_iter()),
        "this thread's release of another thread's claims missed"
    );

    log = foreign(log, false, 5, vec![2]);
    log.push(false, 6, thread_key(), [11]);
    assert!(!log.reached(0, [2].into_iter()), "a landed episode reached");
    let landed = |log: &EpisodeLog, since| {
        let mut slots: Vec<u32> = log
            .others_since(false, since)
            .unwrap()
            .into_iter()
            .collect();
        slots.sort_unstable();
        slots
    };
    assert_eq!(landed(&log, 4), [2]);
    assert_eq!(landed(&log, 5), [0_u32; 0]);

    let mut log = foreign(EpisodeLog::default(), true, u32::MAX, vec![5]);
    log = foreign(log, true, 0, vec![6]);
    assert!(log.reached(u32::MAX - 1, [5].into_iter()));
    assert!(log.reached(u32::MAX, [6].into_iter()));
    assert!(!log.reached(u32::MAX, [5].into_iter()));

    let all = || (0..EPISODE_LOG_LEN as u32).collect::<Vec<_>>();
    let mut log = foreign(EpisodeLog::default(), true, 1, all());
    log = foreign(log, true, 2, vec![u32::MAX]);
    assert!(log.reached(0, [u32::MAX - 1].into_iter()));
    assert!(!log.reached(1, [0].into_iter()));
    assert!(log.others_since(false, 0).is_some());
    log = foreign(log, false, 3, all());
    log = foreign(log, false, 4, vec![u32::MAX]);
    assert!(log.others_since(false, 2).is_none());
    assert_eq!(landed(&log, 3), [u32::MAX]);
}

/// Landings write nothing to the episode log, which only landed episodes
/// and releases of held claims fill, so a Shared recompute pays for no log.
#[test]
fn a_recompute_leaves_the_episode_log_empty() {
    let rt = Rt::new();
    let x = rt.create_input(0_u64);
    let fan: Vec<Incr<u64>> = (0..1000)
        .map(|_| rt.create_query(move |rt: &Rt| rt.get(x) + 1))
        .collect();
    let sum = rt.create_query(move |rt: &Rt| fan.iter().map(|&q| rt.get(q)).sum::<u64>());
    for v in 1..4 {
        rt.set(x, v);
        assert_eq!(rt.get(sum), (v + 1) * 1000);
    }
    assert_eq!(rt.episode_log.read().len(), 0);
}

#[test]
fn a_plan_is_reached_through_its_entries_and_expanded_groups() {
    let entry = |id, taint| Entry {
        id: NodeId(id),
        taint,
        generation: 0,
    };
    let rt = Rt::new();
    let member = rt.create_input(0_u64).erase();
    let mut walk = Walk::new(NodeId(0), 0, NodeState::Dirty);
    walk.push(entry(1, NO_TAINT));
    walk.push(entry(2, VISIT));
    let mut slots: Vec<u32> = walk.plan_slots(NodeId(3)).collect();
    slots.sort_unstable();
    assert_eq!(slots, [0, 1, 2, 3]);
    walk.expanded = Some(Box::new(Expanded {
        new: Default::default(),
        groups: vec![(
            NodeId(5),
            Arc::new(GroupInfo {
                members: vec![(5, 0), (6, 0)],
                cycle: crate::cycle::Cycle::new(vec![member]),
            }),
            0,
        )],
    }));
    let mut slots: Vec<u32> = walk.plan_slots(NodeId(3)).collect();
    slots.sort_unstable();
    assert_eq!(slots, [0, 1, 2, 3, 5, 6]);
}

/// r reads m, which reads a0..a3, which read x. In each plan of r another
/// thread holds the first a not yet landed, so the walk waits on its claim,
/// and the holder counts a release before it lands, as a thread whose
/// caught cycle or doomed round releases claims does (0x2997, 0x19a69):
/// every plan meets a claim a release may have ended. The walk restarts
/// from its root at least once and at most `ROOT_RESTARTS` times and then
/// plans on, where without the bound each plan would restart it again.
#[test]
fn a_walk_that_waits_on_a_released_claim_every_plan_restarts_a_bounded_number_of_times() {
    const MEMBERS: usize = 4;
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let holding = Arc::new(AtomicBool::new(false));
    let waiting = Arc::new(AtomicBool::new(false));
    let holder = Arc::new(Mutex::new(None));
    let member_slots = Arc::new(std::sync::OnceLock::<Vec<u32>>::new());
    let members: Vec<Incr<u64>> = (0..MEMBERS)
        .map(|_| {
            let (held, waited, holder, wrt) = (
                holding.clone(),
                waiting.clone(),
                holder.clone(),
                Arc::downgrade(&rt),
            );
            let slots = member_slots.clone();
            rt.create_query(move |rt| {
                if *holder.lock().unwrap() == Some(thread::current().id()) {
                    held.store(true, Ordering::SeqCst);
                    let start = std::time::Instant::now();
                    while !waited.load(Ordering::SeqCst) && start.elapsed() < Duration::from_secs(2)
                    {
                        thread::sleep(Duration::from_millis(1));
                    }
                    wrt.upgrade()
                        .unwrap()
                        .count_release(slots.get().unwrap().iter().copied());
                }
                rt.get(x) + 1
            })
        })
        .collect();
    member_slots
        .set(members.iter().map(|&a| id(a).0).collect())
        .unwrap();
    let read_members = members.clone();
    let m = rt.create_query(move |rt| read_members.iter().map(|&a| rt.get(a)).sum::<u64>());
    let r = rt.create_query(move |rt| rt.get(m) + 1);
    assert_eq!(rt.get(r), 9);
    rt.set(x, 2);
    let main = thread::current().id();
    let held_this_plan = Arc::new(AtomicBool::new(false));
    let holds = Arc::new(AtomicUsize::new(0));
    let holders = Arc::new(Mutex::new(Vec::new()));
    let (hook_holds, hook_holders, hook_plan) =
        (holds.clone(), holders.clone(), held_this_plan.clone());
    let hook_rt = Arc::downgrade(&rt);
    let hook_members = members.clone();
    let installed = rt.park_hook.set(Box::new(move |point| {
        if thread::current().id() != main {
            return;
        }
        if point == ParkPoint::BeforeVisit(id(m)) {
            hook_plan.store(false, Ordering::SeqCst);
            return;
        }
        if let ParkPoint::BeforeComputeOne(n) = point {
            if hook_members.iter().any(|&a| id(a) == n) {
                waiting.store(true, Ordering::SeqCst);
            }
            return;
        }
        let ParkPoint::BeforeVisit(n) = point else {
            return;
        };
        let Some(&a) = hook_members.iter().find(|&&a| id(a) == n) else {
            return;
        };
        let rt = hook_rt.upgrade().unwrap();
        if hook_plan.swap(true, Ordering::SeqCst) || rt.nodes.get(n.0).state() == NodeState::Clean {
            return;
        }
        hook_holds.fetch_add(1, Ordering::SeqCst);
        holding.store(false, Ordering::SeqCst);
        waiting.store(false, Ordering::SeqCst);
        let slot = holder.clone();
        let (tx, rx) = mpsc::channel();
        hook_holders.lock().unwrap().push(thread::spawn(move || {
            *slot.lock().unwrap() = Some(thread::current().id());
            tx.send(()).unwrap();
            rt.get(a)
        }));
        rx.recv().unwrap();
        let start = std::time::Instant::now();
        while !holding.load(Ordering::SeqCst) && start.elapsed() < Duration::from_secs(2) {
            thread::sleep(Duration::from_millis(1));
        }
    }));
    assert!(installed.is_ok());
    assert_eq!(rt.get(r), 13);
    for h in std::mem::take(&mut *holders.lock().unwrap()) {
        assert_eq!(h.join().unwrap(), 3);
    }
    let holds = holds.load(Ordering::SeqCst);
    assert!(
        holds > super::ROOT_RESTARTS as usize,
        "only {holds} plans waited on a held member"
    );
    let replans = rt.root_replans.load(Ordering::Relaxed);
    assert!(
        (1..=super::ROOT_RESTARTS).contains(&replans),
        "{replans} root replans over {holds} waited plans"
    );
}

/// Each of `walkers` threads reads the top of its own 200-query chain over
/// its own input after setting it, `ROUNDS` times, while `releasers` threads
/// call `release` on the same runtime without pause until every walk is done.
fn walks_under_releases(walkers: usize, releasers: usize, release: fn(&Rt)) -> Arc<Rt> {
    const DEPTH: usize = 200;
    const ROUNDS: u64 = 200;
    let rt = Arc::new(Rt::new());
    let done = Arc::new(AtomicBool::new(false));
    let releasing: Vec<_> = (0..releasers)
        .map(|_| {
            let (rt, done) = (rt.clone(), done.clone());
            thread::spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    release(&rt);
                }
            })
        })
        .collect();
    let walking: Vec<_> = (0..walkers)
        .map(|_| {
            let rt = rt.clone();
            thread::spawn(move || {
                let input = rt.create_input(0_u64);
                let mut top = rt.create_query(move |rt| rt.get(input));
                for _ in 1..DEPTH {
                    let below = top;
                    top = rt.create_query(move |rt| rt.get(below) + 1);
                }
                for round in 1..=ROUNDS {
                    rt.set(input, round);
                    assert_eq!(rt.get(top), round + DEPTH as u64 - 1);
                }
            })
        })
        .collect();
    for w in walking {
        w.join().unwrap();
    }
    done.store(true, Ordering::Relaxed);
    for r in releasing {
        r.join().unwrap();
    }
    rt
}

/// Times walks under releases that never pause, by the test's own
/// "finished in"; runs on trees without the walk's release check too.
#[test]
#[ignore = "a timing probe"]
fn walks_under_constant_releases_timing() {
    walks_under_releases(4, 4, |rt| rt.count_release(std::iter::empty()));
}

/// Releases do not make a walk reread; retractions do, once each per walk
/// that checks after them, so a stream of retractions that never pauses
/// costs rereads but still lets the walks finish. Prints the most rereads
/// one walk did under each.
#[test]
#[ignore = "a probe that prints counts"]
fn walk_rereads_under_constant_releases() {
    let retract = |rt: &Rt| {
        Shared::u64_fetch_add(&rt.retractions.0, 1);
    };
    for (kind, release) in [
        (
            "releases",
            (|rt: &Rt| rt.count_release(std::iter::empty())) as fn(&Rt),
        ),
        ("retractions", retract),
    ] {
        for releasers in [1, 4, 8] {
            let rt = walks_under_releases(4, releasers, release);
            println!(
                "{kind} from {releasers} threads: rereads {}, most in one walk {}, walks of 200 queries {}",
                rt.rereads.load(Ordering::Relaxed),
                rt.max_walk_rereads.load(Ordering::Relaxed),
                4 * 200
            );
        }
    }
}

/// Runs `drive` on a named thread like a test's, which parks a thread on
/// a gate it made and then panics; returns whether the parked thread was
/// released well inside the park timeout instead of aborting the binary.
fn parked_thread_released_after_a_failed_driver(drive: fn(Gate, mpsc::Sender<()>)) -> bool {
    let (resumed_tx, resumed_rx) = mpsc::channel();
    let driver = thread::Builder::new()
        .name("failing driver".into())
        .spawn(move || {
            let gate = Gate::new();
            gate.arm();
            drive(gate, resumed_tx);
        })
        .unwrap();
    assert!(driver.join().is_err(), "the driver panics");
    resumed_rx.recv_timeout(PARK_TIMEOUT / 2).is_ok()
}

#[test]
fn a_failed_tests_spawned_parked_thread_is_released() {
    let released = parked_thread_released_after_a_failed_driver(|gate, resumed| {
        let parked = gate.clone();
        thread::spawn(move || {
            parked.park();
            resumed.send(()).unwrap();
        });
        gate.wait_parked();
        panic!("the test fails while a thread is parked");
    });
    assert!(released);
}

#[test]
fn a_failed_tests_scoped_parked_thread_is_released() {
    let released = parked_thread_released_after_a_failed_driver(|gate, resumed| {
        thread::scope(|s| {
            let _released = release_parked_on_unwind();
            s.spawn(|| {
                gate.park();
                resumed.send(()).unwrap();
            });
            gate.wait_parked();
            panic!("the test fails while a scoped thread is parked");
        });
    });
    assert!(released);
}
