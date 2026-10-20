//! `observer_read`: an observer's read of a node deleted or recycled after
//! `stabilize` took it, or of a node without a value, never panics.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use super::observe_impl::observed_is_clear;
use super::race_tests::{park_at, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{CycleStrategy, QueryOpts};
use crate::handle::Incr;
use crate::node::NodeId;

type Rt = Runtime<Shared>;

fn id<T>(h: Incr<T>) -> NodeId {
    NodeId(h.slot())
}

/// x -> q = x + 1, read Clean, observed, then x set so q is Dirty and the
/// observer's read has to compute it.
fn observed_leaf(rt: &Rt) -> (Incr<u64>, Incr<u64>, Arc<AtomicBool>) {
    let x = rt.create_input(1_u64);
    let q = rt.create_query(move |rt: &Rt| rt.get(x) + 1);
    assert_eq!(rt.get(q), 2);
    let fired = Arc::new(AtomicBool::new(false));
    let seen = Arc::clone(&fired);
    rt.observe(q, move |_| seen.store(true, Ordering::SeqCst));
    rt.set(x, 2);
    (x, q, fired)
}

fn stabilize_on_a_thread(rt: &Arc<Rt>) -> thread::JoinHandle<bool> {
    let rt = Arc::clone(rt);
    thread::spawn(move || {
        let ok = catch_unwind(AssertUnwindSafe(|| rt.stabilize())).is_ok();
        ok && observed_is_clear()
    })
}

/// (1) The delete is parked between its Dead claim and its generation
/// bump, so the observer meets Dead under a matching generation.
#[test]
fn observer_of_a_node_claimed_dead_returns_gone() {
    let rt = Arc::new(Rt::new());
    let (_, q, fired) = observed_leaf(&rt);
    let gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::DeleteAfterCheck(id(q)), gate.clone())],
    );
    let deleter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.delete_node(q))
    };
    gate.wait_parked();
    assert!(stabilize_on_a_thread(&rt).join().unwrap());
    gate.release();
    deleter.join().unwrap();
    assert!(!fired.load(Ordering::SeqCst));
}

/// (2) The slot is recycled into a Clean node before the observer runs,
/// so its entry compare fails.
#[test]
fn observer_of_a_recycled_node_returns_gone_at_entry() {
    let rt = Arc::new(Rt::new());
    let (_, q, fired) = observed_leaf(&rt);
    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::StabilizeAfterDrain, gate.clone())]);
    let stabilizer = stabilize_on_a_thread(&rt);
    gate.wait_parked();
    rt.delete_node(q);
    let recycled = rt.create_input(7_u64);
    assert_eq!(recycled.slot(), q.slot());
    gate.release();
    assert!(stabilizer.join().unwrap());
    assert!(!fired.load(Ordering::SeqCst));
}

/// (3) The node is deleted after `ensure_clean` returned, so the compare
/// after it fails.
#[test]
fn observer_of_a_node_deleted_after_ensure_clean_returns_gone() {
    let rt = Arc::new(Rt::new());
    let (_, q, fired) = observed_leaf(&rt);
    let gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::ObserverAfterEnsureClean(id(q)), gate.clone())],
    );
    let stabilizer = stabilize_on_a_thread(&rt);
    gate.wait_parked();
    rt.delete_node(q);
    gate.release();
    assert!(stabilizer.join().unwrap());
    assert!(!fired.load(Ordering::SeqCst));
}

/// (4) The slot is recycled into a New query after the entry compare, so
/// the walk's root compute finds the generation moved before it claims,
/// and the new query's closure never runs for the observer.
#[test]
fn observer_of_a_node_recycled_into_a_new_query_never_runs_it() {
    let rt = Arc::new(Rt::new());
    let (_, q, fired) = observed_leaf(&rt);
    let gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::ObserverAfterEntryCheck(id(q)), gate.clone())],
    );
    let stabilizer = stabilize_on_a_thread(&rt);
    gate.wait_parked();
    rt.delete_node(q);
    let runs = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&runs);
    let recycled = rt.create_query(move |_: &Rt| {
        counted.fetch_add(1, Ordering::SeqCst);
        0_u64
    });
    assert_eq!(recycled.slot(), q.slot());
    gate.release();
    assert!(stabilizer.join().unwrap());
    assert!(!fired.load(Ordering::SeqCst));
    assert_eq!(runs.load(Ordering::SeqCst), 0);
}

/// (5) The slot is recycled into a Clean node after the compare that
/// follows `ensure_clean`, so the read's pre-check fails.
#[test]
fn observer_of_a_node_recycled_before_the_pre_check_returns_gone() {
    let rt = Arc::new(Rt::new());
    let (_, q, fired) = observed_leaf(&rt);
    let gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::ObserverBeforePreCheck(id(q)), gate.clone())],
    );
    let stabilizer = stabilize_on_a_thread(&rt);
    gate.wait_parked();
    rt.delete_node(q);
    let recycled = rt.create_input(7_u64);
    assert_eq!(recycled.slot(), q.slot());
    gate.release();
    assert!(stabilizer.join().unwrap());
    assert!(!fired.load(Ordering::SeqCst));
}

macro_rules! both {
    ($name:ident, $local:ident, $shared:ident) => {
        #[test]
        fn $local() {
            $name::<Local>();
        }

        #[test]
        fn $shared() {
            $name::<Shared>();
        }
    };
}

fn a_panicking_observed_compute_leaves_the_observed_slot_clear<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(0_u64);
    let q = rt.create_query(move |rt: &Runtime<C>| {
        let v = rt.get(x);
        assert!(v == 0, "boom");
        v
    });
    rt.get(q);
    rt.observe(q, |_| {});
    rt.set(x, 1);
    assert!(catch_unwind(AssertUnwindSafe(|| rt.stabilize())).is_err());
    assert!(observed_is_clear());
}
both!(
    a_panicking_observed_compute_leaves_the_observed_slot_clear,
    local_a_panicking_observed_compute_leaves_the_observed_slot_clear,
    shared_a_panicking_observed_compute_leaves_the_observed_slot_clear
);

/// x -> a -> b -> a while x is odd: a and b land Cyclic. b reads x / 2,
/// which cuts off, once the cycle is broken.
fn an_observer_of_a_cyclic_node_fires_once_the_cycle_breaks<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    let b_slot: Arc<Mutex<Option<Incr<u64>>>> = Arc::new(Mutex::new(None));
    let to_b = Arc::clone(&b_slot);
    let a = rt.create_query_with(
        move |rt: &Runtime<C>| {
            let v = rt.get(x);
            if v % 2 == 1 {
                let b = to_b.lock().unwrap().expect("b is created");
                rt.get(b) + v
            } else {
                v
            }
        },
        QueryOpts {
            cycle: CycleStrategy::None,
            ..Default::default()
        },
    );
    let b = rt.create_query(move |rt: &Runtime<C>| rt.get(a) / 2);
    *b_slot.lock().unwrap() = Some(b);
    assert!(rt.try_get(a).is_err());
    let fired: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&fired);
    rt.observe(a, move |v| seen.lock().unwrap().push(*v));
    rt.stabilize();
    assert!(fired.lock().unwrap().is_empty(), "a Cyclic node fired");
    rt.set(x, 4);
    rt.stabilize();
    assert_eq!(*fired.lock().unwrap(), vec![4]);
    rt.stabilize();
    assert_eq!(*fired.lock().unwrap(), vec![4]);
}
both!(
    an_observer_of_a_cyclic_node_fires_once_the_cycle_breaks,
    local_an_observer_of_a_cyclic_node_fires_once_the_cycle_breaks,
    shared_an_observer_of_a_cyclic_node_fires_once_the_cycle_breaks
);
