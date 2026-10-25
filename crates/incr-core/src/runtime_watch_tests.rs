//! Dirty-driven watching: `watch`, `unwatch` and `drain_dirtied_watched`.

use std::collections::HashSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use proptest::prelude::*;

use super::race_tests::{park_at, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::handle::{ErasedIncr, Incr};
use crate::node::NodeId;
use crate::state::NodeState;

const SETTLE: Duration = Duration::from_millis(100);

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
    walk_reports_a_watched_node_once_per_flip,
    set_reports_its_own_watched_input,
    drain_follows_the_walk_order,
    watch_reports_a_node_that_is_not_clean,
    unwatch_drops_the_entry,
    recycled_slot_reports_only_the_new_node,
    delete_drops_a_watched_entry,
    failed_landing_is_reported,
    broken_cycle_is_reported,
);

fn drained<C: Cells>(rt: &Runtime<C>) -> Vec<ErasedIncr> {
    let out = rt.drain_dirtied_watched();
    let distinct: HashSet<_> = out.iter().collect();
    assert_eq!(
        distinct.len(),
        out.len(),
        "a drain repeated a node: {out:?}"
    );
    out
}

fn walk_reports_a_watched_node_once_per_flip<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    let q = rt.create_query(move |rt| rt.get(x) + 1);
    assert_eq!(rt.get(q), 2);
    rt.watch(q);
    assert!(drained(&rt).is_empty(), "a Clean node was reported");

    rt.set(x, 2);
    assert_eq!(drained(&rt), vec![q.erase()]);
    assert!(drained(&rt).is_empty());

    // Left Dirty by the drain without a read, so the next walk stops at it.
    rt.set(x, 3);
    assert!(drained(&rt).is_empty());

    assert_eq!(rt.get(q), 4);
    rt.set(x, 4);
    assert_eq!(drained(&rt), vec![q.erase()]);
}

fn set_reports_its_own_watched_input<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    rt.watch(x);
    assert!(drained(&rt).is_empty(), "an input is Clean");
    rt.set(x, 2);
    rt.set(x, 3);
    assert_eq!(drained(&rt), vec![x.erase()]);
    rt.set(x, 3);
    assert!(drained(&rt).is_empty(), "an equal set reported its input");
}

fn drain_follows_the_walk_order<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    let a = rt.create_query(move |rt| rt.get(x) + 1);
    let b = rt.create_query(move |rt| rt.get(a) + 1);
    let c = rt.create_query(move |rt| rt.get(b) + 1);
    assert_eq!(rt.get(c), 4);
    rt.watch(c);
    rt.watch(a);
    rt.watch(x);
    rt.set(x, 2);
    assert_eq!(drained(&rt), vec![x.erase(), a.erase(), c.erase()]);
}

fn watch_reports_a_node_that_is_not_clean<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    let q = rt.create_query(move |rt| rt.get(x) + 1);
    rt.watch(q);
    assert_eq!(drained(&rt), vec![q.erase()], "a New node was not reported");
    assert_eq!(rt.get(q), 2);
    rt.set(x, 2);
    rt.unwatch(q);
    assert!(drained(&rt).is_empty());
    rt.watch(q);
    assert_eq!(
        drained(&rt),
        vec![q.erase()],
        "a Dirty node was not reported"
    );
    rt.watch(q);
    assert!(
        drained(&rt).is_empty(),
        "a watch of a watched node reported it"
    );
}

fn unwatch_drops_the_entry<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    let q = rt.create_query(move |rt| rt.get(x) + 1);
    assert_eq!(rt.get(q), 2);
    rt.watch(q);
    rt.set(x, 2);
    rt.unwatch(q);
    assert!(drained(&rt).is_empty());
    assert_eq!(rt.get(q), 3);
    rt.set(x, 3);
    assert!(drained(&rt).is_empty(), "an unwatched node was reported");
}

fn recycled_slot_reports_only_the_new_node<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    let old = rt.create_query(move |rt| rt.get(x) + 1);
    assert_eq!(rt.get(old), 2);
    rt.watch(old);
    rt.set(x, 2);
    rt.unwatch(old);
    rt.delete_node(old);
    let new = rt.create_query(move |rt| rt.get(x) * 10);
    assert_eq!(new.slot(), old.slot(), "the slot was recycled");
    assert_eq!(rt.get(new), 20);
    rt.watch(new);
    rt.set(x, 3);
    assert_eq!(drained(&rt), vec![new.erase()]);
    assert!(drained(&rt).is_empty());
}

fn delete_drops_a_watched_entry<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    let old = rt.create_query(move |rt| rt.get(x) + 1);
    assert_eq!(rt.get(old), 2);
    rt.watch(old);
    rt.set(x, 2);
    rt.delete_node(old);
    let new = rt.create_query(move |rt| rt.get(x) * 10);
    assert_eq!(new.slot(), old.slot(), "the slot was recycled");
    assert_eq!(rt.get(new), 20);
    rt.set(x, 3);
    assert!(
        drained(&rt).is_empty(),
        "the recycled slot inherited a watch"
    );
    rt.watch(new);
    rt.set(x, 4);
    assert_eq!(drained(&rt), vec![new.erase()]);
}

fn failed_landing_is_reported<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(2_u64);
    let q = rt.create_query(move |rt| {
        let v = rt.get(x);
        assert!(v % 2 == 0, "odd input");
        v
    });
    assert_eq!(rt.get(q), 2);
    rt.watch(q);
    rt.set(x, 3);
    assert_eq!(drained(&rt), vec![q.erase()]);
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(q))).is_err());
    assert_eq!(rt.state_of(q), NodeState::Failed);
    assert_eq!(drained(&rt), vec![q.erase()], "the Failed landing was lost");
}

fn broken_cycle_is_reported<C: Cells>() {
    let rt = Runtime::<C>::new();
    let g = rt.create_input(0_i64);
    let bc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let bc = bc.clone();
        rt.create_query(move |rt| {
            if rt.get(g) == 0 {
                rt.try_get(*bc.get().unwrap()).map_or(-1, |v| v + 1)
            } else {
                7
            }
        })
    };
    let b = rt.create_query(move |rt| rt.try_get(a).map_or(-1, |v| v * 2));
    bc.set(b).unwrap();
    let d = rt.create_query(move |rt| rt.try_get(b).unwrap_or(-100));
    assert_eq!(rt.get(d), -100);
    assert_eq!(rt.state_of(a), NodeState::Cyclic);
    rt.watch(a);
    rt.watch(d);
    assert!(drained(&rt).is_empty(), "a Cyclic node was reported");

    rt.set(g, 1);
    let mut got: Vec<u32> = drained(&rt).iter().map(|e| e.slot()).collect();
    got.sort_unstable();
    assert_eq!(got, vec![a.slot(), d.slot()]);
}

type Rt = Runtime<Shared>;

/// A watched node whose recompute reads a dep with an unrecorded walk
/// lands Dirty after the drain that took it: the landing reports it again.
#[test]
fn superseded_landing_is_reported_again() {
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
    rt.watch(q);

    let set_gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterWalk, set_gate.clone())]);
    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(late, 6))
    };
    set_gate.wait_parked();

    rt.set(trigger, 1);
    assert_eq!(drained(&rt), vec![q.erase()]);
    let _ = rt.get(q);
    assert_eq!(
        rt.state_of(q),
        NodeState::Dirty,
        "the landing was not Superseded"
    );
    assert_eq!(
        drained(&rt),
        vec![q.erase()],
        "the Superseded landing was lost"
    );

    set_gate.release();
    setter.join().unwrap();
    assert_eq!(rt.get(q), 6);
    assert!(drained(&rt).is_empty());
    rt.assert_invariant_i();
}

/// A watch during a recompute reports the node at once; a set during it
/// flags it ComputingDirty, and the flagged landing adds no second entry.
#[test]
fn computing_node_is_reported_once() {
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
    assert_eq!(rt.get(q), 2);
    rt.set(x, 2);
    block.store(true, Ordering::SeqCst);
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(q))
    };
    entered_rx.recv().unwrap();
    rt.watch(q);
    assert_eq!(
        drained(&rt),
        vec![q.erase()],
        "a Computing node was not reported"
    );
    rt.set(x, 3);
    assert_eq!(rt.state_of(q), NodeState::ComputingDirty);
    go_tx.send(()).unwrap();
    reader.join().unwrap();
    assert_eq!(rt.state_of(q), NodeState::Dirty);
    assert_eq!(drained(&rt), vec![q.erase()]);
    assert!(drained(&rt).is_empty());
}

/// `unwatch` parks with `WATCHED` clear and the entry still queued, under
/// its write hold. A delete, a recycle, a watch of the new node and a set
/// queue behind it, so its dequeue never meets the new node's entry.
#[test]
fn unwatch_dequeues_under_its_hold() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let old = rt.create_query(move |rt: &Rt| rt.get(x) + 1);
    assert_eq!(rt.get(old), 2);
    rt.watch(old);
    rt.set(x, 2);

    let gate = Gate::new();
    park_at(
        &rt,
        vec![(
            ParkPoint::UnwatchBeforeDequeue(NodeId(old.slot())),
            gate.clone(),
        )],
    );
    let unwatcher = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.unwatch(old))
    };
    gate.wait_parked();
    let recycler = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || {
            rt.delete_node(old);
            let new = rt.create_query(move |rt: &Rt| rt.get(x) * 10);
            rt.watch(new);
            rt.set(x, 3);
            new
        })
    };
    thread::sleep(SETTLE);
    gate.release();
    unwatcher.join().unwrap();
    let new = recycler.join().unwrap();
    assert_eq!(new.slot(), old.slot(), "the slot was recycled");
    assert_eq!(drained(&rt), vec![new.erase()]);
    assert!(drained(&rt).is_empty());
}

#[derive(Clone, Debug)]
enum Op {
    Set(usize, u64),
    Get(usize),
    Watch(usize),
    Unwatch(usize),
    Drain,
}

#[derive(Clone, Debug)]
struct Case {
    inputs: usize,
    /// Each query's deps, as indices of earlier nodes.
    queries: Vec<Vec<usize>>,
    ops: Vec<Op>,
}

fn case() -> impl Strategy<Value = Case> {
    (1usize..4, 1usize..24).prop_flat_map(|(inputs, queries)| {
        let deps = (0..queries)
            .map(|q| prop::collection::vec(0..inputs + q, 1..4))
            .collect::<Vec<_>>();
        let n = inputs + queries;
        let op = prop_oneof![
            3 => (0..inputs, 0u64..3).prop_map(|(i, v)| Op::Set(i, v)),
            2 => (0..n).prop_map(Op::Get),
            2 => (0..n).prop_map(Op::Watch),
            1 => (0..n).prop_map(Op::Unwatch),
            2 => Just(Op::Drain),
        ];
        (deps, prop::collection::vec(op, 1..60)).prop_map(move |(queries, ops)| Case {
            inputs,
            queries,
            ops,
        })
    })
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum M {
    New,
    Clean,
    Dirty,
}

/// The oracle: a drain reports the watched nodes the walk moved out of
/// Clean since the last drain, the watched inputs a changing set wrote,
/// and the nodes that were not Clean when watched. Queries never fail or
/// cycle here, and one thread lands every compute Clean.
struct Model {
    inputs: usize,
    deps: Vec<Vec<usize>>,
    values: Vec<u64>,
    state: Vec<M>,
    watched: Vec<bool>,
    queued: Vec<usize>,
}

impl Model {
    fn new(case: &Case) -> Self {
        let n = case.inputs + case.queries.len();
        let mut deps = vec![Vec::new(); case.inputs];
        deps.extend(case.queries.iter().cloned());
        let mut state = vec![M::Clean; case.inputs];
        state.resize(n, M::New);
        Self {
            inputs: case.inputs,
            deps,
            values: vec![0; case.inputs],
            state,
            watched: vec![false; n],
            queued: Vec::new(),
        }
    }

    fn enqueue(&mut self, i: usize) {
        if self.watched[i] && !self.queued.contains(&i) {
            self.queued.push(i);
        }
    }

    fn dependents(&self, i: usize) -> Vec<usize> {
        (self.inputs..self.deps.len())
            .filter(|&q| self.state[q] != M::New && self.deps[q].contains(&i))
            .collect()
    }

    fn set(&mut self, i: usize, v: u64) {
        if self.values[i] == v {
            return;
        }
        self.values[i] = v;
        self.enqueue(i);
        let mut stack = self.dependents(i);
        while let Some(q) = stack.pop() {
            if self.state[q] == M::Clean {
                self.state[q] = M::Dirty;
                self.enqueue(q);
                stack.extend(self.dependents(q));
            }
        }
    }

    fn get(&mut self, i: usize) {
        self.state[i] = M::Clean;
        for d in self.deps[i].clone() {
            self.get(d);
        }
    }

    fn watch(&mut self, i: usize) {
        if std::mem::replace(&mut self.watched[i], true) {
            return;
        }
        if self.state[i] != M::Clean {
            self.enqueue(i);
        }
    }

    fn unwatch(&mut self, i: usize) {
        self.watched[i] = false;
        self.queued.retain(|&q| q != i);
    }
}

fn watch_against_model<C: Cells>(case: &Case) {
    let rt = Runtime::<C>::new();
    let mut handles: Vec<Incr<u64>> = (0..case.inputs).map(|_| rt.create_input(0)).collect();
    for deps in &case.queries {
        let ds: Vec<Incr<u64>> = deps.iter().map(|&d| handles[d]).collect();
        handles.push(rt.create_query(move |rt| {
            ds.iter().fold(1_u64, |acc, &d| {
                acc.wrapping_mul(31).wrapping_add(rt.get(d))
            })
        }));
    }
    let index_of = |e: &ErasedIncr| {
        handles
            .iter()
            .position(|h| h.erase() == *e)
            .expect("a drain returned a node the case never created")
    };
    let mut model = Model::new(case);
    for op in &case.ops {
        match *op {
            Op::Set(i, v) => {
                rt.set(handles[i], v);
                model.set(i, v);
            }
            Op::Get(i) => {
                let _ = rt.get(handles[i]);
                model.get(i);
            }
            Op::Watch(i) => {
                rt.watch(handles[i]);
                model.watch(i);
            }
            Op::Unwatch(i) => {
                rt.unwatch(handles[i]);
                model.unwatch(i);
            }
            Op::Drain => {
                let mut got: Vec<usize> = drained(&rt).iter().map(index_of).collect();
                let mut want = std::mem::take(&mut model.queued);
                got.sort_unstable();
                want.sort_unstable();
                assert_eq!(got, want, "drain after {:?}", case.ops);
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: if cfg!(miri) { 6 } else { 64 },
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn drain_matches_the_model_local(case in case()) {
        watch_against_model::<Local>(&case);
    }

    #[test]
    fn drain_matches_the_model_shared(case in case()) {
        watch_against_model::<Shared>(&case);
    }
}
