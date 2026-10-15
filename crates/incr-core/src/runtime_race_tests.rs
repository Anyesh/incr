//! Deterministic race tests for the shared-protocol plan (B1-B6). Each
//! `Shared` test parks one thread at a named [`ParkPoint`] (or inside a
//! compute closure) and acts from another while it is parked. Every test
//! ends by checking invariant (I) and the value a fresh read returns.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::handle::Incr;
use crate::node::NodeId;
use crate::state::NodeState;

type Rt = Runtime<Shared>;

/// A park point that never gets reached would otherwise hang the test.
const PARK_TIMEOUT: Duration = Duration::from_secs(10);

/// One-shot rendezvous. The thread under test calls [`Gate::park`]; the
/// driver calls [`Gate::wait_parked`], acts, then [`Gate::release`]. A
/// gate only parks once it is armed, so a closure can carry one through
/// setup computes without stopping.
#[derive(Clone)]
struct Gate(Arc<GateInner>);

struct GateInner {
    armed: AtomicBool,
    reached_tx: Mutex<Sender<()>>,
    reached_rx: Mutex<Receiver<()>>,
    release_tx: Mutex<Sender<()>>,
    release_rx: Mutex<Receiver<()>>,
}

impl Gate {
    fn new() -> Self {
        let (reached_tx, reached_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        Self(Arc::new(GateInner {
            armed: AtomicBool::new(false),
            reached_tx: Mutex::new(reached_tx),
            reached_rx: Mutex::new(reached_rx),
            release_tx: Mutex::new(release_tx),
            release_rx: Mutex::new(release_rx),
        }))
    }

    fn arm(&self) {
        self.0.armed.store(true, Ordering::SeqCst);
    }

    fn park(&self) {
        if !self.0.armed.swap(false, Ordering::SeqCst) {
            return;
        }
        self.0.reached_tx.lock().unwrap().send(()).unwrap();
        self.0
            .release_rx
            .lock()
            .unwrap()
            .recv_timeout(PARK_TIMEOUT)
            .expect("parked thread was never released");
    }

    fn wait_parked(&self) {
        self.0
            .reached_rx
            .lock()
            .unwrap()
            .recv_timeout(PARK_TIMEOUT)
            .expect("thread never reached its park point");
    }

    fn release(&self) {
        self.0.release_tx.lock().unwrap().send(()).unwrap();
    }
}

/// Install the runtime's park hook with armed gates at `points`.
fn park_at<C: Cells>(rt: &Runtime<C>, points: Vec<(ParkPoint, Gate)>) {
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

fn id<T>(h: Incr<T>) -> NodeId {
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
#[ignore = "fails until fix 1 (verify claims and checks dep state)"]
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
#[ignore = "fails until fix 1 (verify claims and checks dep state)"]
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
#[ignore = "fails until fix 1 (verify claims and checks dep state)"]
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
#[ignore = "fails until fix 7 (walked-revision watermark)"]
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
#[ignore = "fails until fix 7 (walked-revision watermark)"]
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

type LocalRt = Runtime<Local>;

/// Q reads v, deletes it, and recycles its slot inside its own compute.
/// Returns (q, the recycled input once Q has run).
fn self_deleting_query(
    rt: &LocalRt,
    stop_reading_on_retry: bool,
) -> (Incr<u64>, Arc<Mutex<Option<Incr<u64>>>>) {
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
    assert_eq!(rt.state_of(d), NodeState::Failed);
    rt.assert_invariant_i();

    rt.set(x, 2);
    assert_eq!(
        rt.state_of(d),
        NodeState::Dirty,
        "the walk must pass through New Q"
    );
}
