//! Deterministic race tests for the shared-protocol plan (B1-B6). Each
//! `Shared` test parks one thread at a named [`ParkPoint`] (or inside a
//! compute closure) and acts from another while it is parked. Every test
//! ends by checking invariant (I) and the value a fresh read returns.

use std::collections::HashSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use super::{DeleteError, DeleteOutcome, ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::collection::{GroupedCollection, IncrCollection};
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
pub(crate) struct Gate(Arc<GateInner>);

pub(crate) struct GateInner {
    armed: AtomicBool,
    reached_tx: Mutex<Sender<()>>,
    reached_rx: Mutex<Receiver<()>>,
    release_tx: Mutex<Sender<()>>,
    release_rx: Mutex<Receiver<()>>,
}

impl Gate {
    pub(crate) fn new() -> Self {
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

    pub(crate) fn arm(&self) {
        self.0.armed.store(true, Ordering::SeqCst);
    }

    pub(crate) fn park(&self) {
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

/// A read whose walk lands its node New as a member of a cycle finds the
/// slot empty, and another thread claims the node before the read loads
/// its state and group: the read waits for that claim and returns the
/// cycle, where it used to panic that a Clean node had no value (replay
/// seeds 0x33e9, 0xa55 and 0x19f0 on two threads).
#[test]
fn read_of_cycle_member_vs_claim_after_landing() {
    let rt = Arc::new(Rt::new());
    let inp = rt.create_input(0_u64);
    let failed = rt.create_query(move |rt: &Rt| match rt.get(inp) {
        0 => panic!("boom"),
        v => v,
    });
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
    // b catches the panic of a read, so the episode lands Dirty and its
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
    assert_eq!(rt.state_of(a), NodeState::New);
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
    in_a.wait_parked();
    assert_eq!(rt.state_of(a), NodeState::Computing);
    before_read.release();
    // Time for the reader to load a's state while the claim is held.
    thread::sleep(Duration::from_millis(50));
    in_a.release();

    assert_eq!(reader.join().unwrap(), Ok(1000));
    let cycle = claimer
        .join()
        .unwrap()
        .unwrap()
        .expect_err("a is in a cycle");
    assert_eq!(cycle.participants().len(), 2);
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
