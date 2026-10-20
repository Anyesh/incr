//! `stabilize` on the watch queue: it runs the observers of the nodes it
//! drains, and an unwind out of it puts the unprocessed ones back.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use super::observe_impl::observed_is_clear;
use super::race_tests::{park_at, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::handle::Incr;
use crate::state::NodeState;

type Rt = Runtime<Shared>;
type Fired = Arc<Mutex<Vec<(usize, u64)>>>;

macro_rules! both {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Local>();
                }
            )*
        }
        mod shared {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Shared>();
                }
            )*
        }
    };
}

both!(
    only_drained_nodes_run_their_observers,
    a_panicking_observed_compute_stays_in_stabilize,
    a_failed_observed_node_does_not_block_later_observers,
    observing_a_drained_watched_node_queues_it,
);

fn observe_into<C: Cells>(rt: &Runtime<C>, h: Incr<u64>, tag: usize, fired: &Fired) {
    let fired = Arc::clone(fired);
    rt.observe(h, move |&v| fired.lock().unwrap().push((tag, v)));
}

fn take(fired: &Fired) -> Vec<(usize, u64)> {
    std::mem::take(&mut *fired.lock().unwrap())
}

/// Observing a node watches it, so it fires when a change moved it off
/// Clean; after an `unwatch` its observer no longer runs.
fn only_drained_nodes_run_their_observers<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    let y = rt.create_input(10_u64);
    let p = rt.create_query(move |rt: &Runtime<C>| rt.get(x) + 1);
    let q = rt.create_query(move |rt: &Runtime<C>| rt.get(y) + 1);
    let fired: Fired = Arc::default();
    observe_into(&rt, p, 0, &fired);
    observe_into(&rt, q, 1, &fired);
    rt.stabilize();
    assert_eq!(take(&fired), vec![(0, 2), (1, 11)]);
    assert!(rt.drain_dirtied_watched().is_empty());

    rt.set(y, 20);
    rt.stabilize();
    assert_eq!(take(&fired), vec![(1, 21)]);
    rt.stabilize();
    assert!(take(&fired).is_empty());

    rt.unwatch(p);
    rt.set(x, 2);
    rt.stabilize();
    assert!(take(&fired).is_empty(), "an unwatched node's observer ran");
}

/// Three queued nodes; the second's callback panics once, after the
/// first's callback queued a fourth. The panic leaves `stabilize`, the
/// second and third go back in front of the fourth, in order, and the
/// next `stabilize` fires each of them once and the first not again.
#[test]
fn an_unwind_puts_the_rest_back_at_the_front() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let w = rt.create_input(0_u64);
    let nodes: Vec<Incr<u64>> = (0..3_u64)
        .map(|i| rt.create_query(move |rt: &Rt| rt.get(x) + i))
        .collect();
    let fourth = rt.create_query(move |rt: &Rt| rt.get(w));
    assert_eq!(rt.get(fourth), 0);
    rt.watch(fourth);

    let fired: Fired = Arc::default();
    {
        let fired = Arc::clone(&fired);
        let weak = Arc::downgrade(&rt);
        rt.observe(nodes[0], move |&v| {
            fired.lock().unwrap().push((0, v));
            weak.upgrade().unwrap().set(w, 1);
        });
    }
    let armed = Arc::new(AtomicBool::new(true));
    {
        let fired = Arc::clone(&fired);
        let armed = Arc::clone(&armed);
        rt.observe(nodes[1], move |&v| {
            assert!(!armed.swap(false, Ordering::SeqCst), "boom");
            fired.lock().unwrap().push((1, v));
        });
    }
    observe_into(&rt, nodes[2], 2, &fired);

    assert!(catch_unwind(AssertUnwindSafe(|| rt.stabilize())).is_err());
    assert_eq!(take(&fired), vec![(0, 1)]);
    assert_eq!(
        rt.queued_slots(),
        vec![nodes[1].slot(), nodes[2].slot(), fourth.slot()]
    );
    rt.stabilize();
    assert_eq!(take(&fired), vec![(1, 2), (2, 3)]);
    assert!(rt.queued_slots().is_empty());
    assert!(observed_is_clear());
}

/// A watched node whose observed compute panics lands Failed: its
/// observer does not fire and `stabilize` returns. The Failed landing
/// reports the node once more, the next `stabilize` fires nothing for it,
/// and once the input is fixed the observer fires.
fn a_panicking_observed_compute_stays_in_stabilize<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(0_u64);
    let q = rt.create_query(move |rt: &Runtime<C>| {
        let v = rt.get(x);
        assert!(v != 1, "boom");
        v
    });
    let fired: Fired = Arc::default();
    observe_into(&rt, q, 0, &fired);
    rt.stabilize();
    assert_eq!(take(&fired), vec![(0, 0)]);
    rt.set(x, 1);
    rt.stabilize();
    assert_eq!(rt.state_of(q), NodeState::Failed);
    assert!(observed_is_clear());
    assert!(take(&fired).is_empty());
    assert_eq!(rt.queued_slots(), vec![q.slot()]);
    rt.stabilize();
    assert!(take(&fired).is_empty());
    assert!(rt.queued_slots().is_empty());
    rt.set(x, 2);
    rt.stabilize();
    assert_eq!(take(&fired), vec![(0, 2)]);
}

/// A Failed observed node queued ahead of a healthy one, whether its
/// compute panics in this `stabilize` or it was Failed before: the healthy
/// observer fires in the same `stabilize`.
fn a_failed_observed_node_does_not_block_later_observers<C: Cells>() {
    for failed_before in [false, true] {
        let rt = Runtime::<C>::new();
        let x = rt.create_input(0_u64);
        let y = rt.create_input(10_u64);
        let q = rt.create_query(move |rt: &Runtime<C>| {
            let v = rt.get(x);
            assert!(v != 1, "boom");
            v
        });
        let p = rt.create_query(move |rt: &Runtime<C>| rt.get(y) + 1);
        let fired: Fired = Arc::default();
        observe_into(&rt, q, 0, &fired);
        observe_into(&rt, p, 1, &fired);
        rt.stabilize();
        assert_eq!(take(&fired), vec![(0, 0), (1, 11)]);
        rt.set(x, 1);
        if failed_before {
            assert!(catch_unwind(AssertUnwindSafe(|| rt.get(q))).is_err());
            assert_eq!(rt.state_of(q), NodeState::Failed);
        }
        rt.set(y, 20);
        rt.stabilize();
        assert_eq!(
            take(&fired),
            vec![(1, 21)],
            "failed before: {failed_before}"
        );
        // Only a landing in this `stabilize` reports the node again.
        let requeued = if failed_before {
            vec![]
        } else {
            vec![q.slot()]
        };
        assert_eq!(rt.queued_slots(), requeued);
        assert!(observed_is_clear());
    }
}

/// A watched, observed node lands Superseded after a drain took it: the
/// landing queues it again, so `stabilize` fires it with the settled
/// value.
#[test]
fn superseded_landing_fires_on_the_next_stabilize() {
    let rt = Arc::new(Rt::new());
    let late = rt.create_input(5_u64);
    let trigger = rt.create_input(0_u64);
    let q = rt.create_query(move |rt: &Rt| {
        if rt.get(trigger) != 0 {
            rt.get(late)
        } else {
            0
        }
    });
    assert_eq!(rt.get(q), 0);
    let fired: Fired = Arc::default();
    observe_into(&rt, q, 0, &fired);

    let set_gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterWalk, set_gate.clone())]);
    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(late, 6))
    };
    set_gate.wait_parked();

    rt.set(trigger, 1);
    let drained: Vec<u32> = rt
        .drain_dirtied_watched()
        .iter()
        .map(|h| h.slot())
        .collect();
    assert_eq!(drained, vec![q.slot()]);
    let _ = rt.get(q);
    assert_eq!(
        rt.state_of(q),
        NodeState::Dirty,
        "the landing was not Superseded"
    );
    set_gate.release();
    setter.join().unwrap();
    rt.stabilize();
    assert_eq!(take(&fired), vec![(0, 6)]);
    rt.assert_invariant_i();
}

/// `q` with its closure blocked on its first compute (`first`) or on a
/// changing recompute, observed while the other thread computes it: the
/// observer fires exactly once, with the landed value.
fn observe_in_flight(first: bool) {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let block = Arc::new(AtomicBool::new(false));
    let (entered_tx, entered_rx) = mpsc::channel::<()>();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let (entered_tx, go_rx) = (Mutex::new(entered_tx), Mutex::new(go_rx));
    let q = {
        let block = Arc::clone(&block);
        rt.create_query(move |rt: &Rt| {
            let v = rt.get(x);
            if block.swap(false, Ordering::SeqCst) {
                entered_tx.lock().unwrap().send(()).unwrap();
                go_rx.lock().unwrap().recv().unwrap();
            }
            v + 1
        })
    };
    if !first {
        assert_eq!(rt.get(q), 2);
        rt.set(x, 2);
    }
    block.store(true, Ordering::SeqCst);
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(q))
    };
    entered_rx.recv().unwrap();
    let fired: Fired = Arc::default();
    observe_into(&rt, q, 0, &fired);
    go_tx.send(()).unwrap();
    let landed = reader.join().unwrap();
    assert_eq!(rt.state_of(q), NodeState::Clean);
    rt.stabilize();
    rt.stabilize();
    assert_eq!(take(&fired), vec![(0, landed)]);
}

#[test]
fn observe_during_a_first_compute_fires_once() {
    observe_in_flight(true);
}

#[test]
fn observe_during_a_changing_recompute_fires_once() {
    observe_in_flight(false);
}

/// A node watched and drained before it is observed, Dirty or never
/// computed: the observe queues it, so the next `stabilize` fires it.
fn observing_a_drained_watched_node_queues_it<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    let dirty = rt.create_query(move |rt: &Runtime<C>| rt.get(x) + 1);
    let new = rt.create_query(move |rt: &Runtime<C>| rt.get(x) + 2);
    assert_eq!(rt.get(dirty), 2);
    rt.watch(dirty);
    rt.watch(new);
    rt.set(x, 5);
    assert_eq!(rt.drain_dirtied_watched().len(), 2);
    let fired: Fired = Arc::default();
    observe_into(&rt, dirty, 0, &fired);
    observe_into(&rt, new, 1, &fired);
    rt.stabilize();
    assert_eq!(take(&fired), vec![(0, 6), (1, 7)]);
}
