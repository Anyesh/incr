//! `get_traced` owns the runtime's single trace log for its whole call, so
//! a second one, nested in a compute or on another thread, is refused.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::node::NodeId;
use crate::trace::{Direction, GraphQuery, TraceAction};

const NESTED: &str = "incr-core: get_traced called while another get_traced is in progress";

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(s) => *s,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(s) => (*s).to_string(),
            Err(_) => "<non-string panic payload>".to_string(),
        },
    }
}

fn recomputed(trace: &crate::trace::PropagationTrace) -> usize {
    trace
        .node_traces
        .iter()
        .filter(|t| matches!(t.action, TraceAction::Recomputed { .. }))
        .count()
}

fn nested_caught_keeps_outer_trace<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let a = rt.create_input(1_i64);
    let x = rt.create_input(7_i64);
    let b = rt.create_query(move |r| r.get(a) + 1);
    let refused = Arc::new(Mutex::new(None::<String>));
    let seen = Arc::clone(&refused);
    let q = rt.create_query(move |r| {
        let v = r.get(b);
        let nested = catch_unwind(AssertUnwindSafe(|| r.get_traced(x)));
        *seen.lock().unwrap() = nested.err().map(panic_message);
        v * 2
    });

    let (value, trace) = rt.get_traced(q);
    assert_eq!(value, 4);
    assert_eq!(trace.nodes_recomputed, 2, "{:?}", trace.node_traces);
    assert_eq!(recomputed(&trace), 2);
    let message = refused.lock().unwrap().take();
    assert!(
        message.as_deref().is_some_and(|m| m.contains(NESTED)),
        "the nested get_traced was allowed: {message:?}"
    );
    assert!(!rt.tracing_is_armed());
}

fn nested_uncaught_unwinds_outer<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let a = rt.create_input(1_i64);
    let x = rt.create_input(7_i64);
    let b = rt.create_query(move |r| r.get(a) + 1);
    let q = rt.create_query(move |r| r.get(b) + r.get_traced(x).0);
    let c = rt.create_query(move |r| r.get(a) * 3);
    assert_eq!(rt.get(c), 3);

    let outer = catch_unwind(AssertUnwindSafe(|| rt.get_traced(q)));
    let message = match outer {
        Ok((_, trace)) => panic!(
            "the nested get_traced was allowed; outer trace {:?}",
            trace.node_traces
        ),
        Err(payload) => panic_message(payload),
    };
    assert!(message.contains(NESTED), "got: {message}");
    assert!(!rt.tracing_is_armed());

    rt.set(a, 5);
    let (value, trace) = rt.get_traced(c);
    assert_eq!(value, 15);
    assert_eq!(trace.nodes_recomputed, 1, "{:?}", trace.node_traces);
    assert!(!rt.tracing_is_armed());
}

#[test]
fn local_nested_get_traced_caught_keeps_outer_trace() {
    nested_caught_keeps_outer_trace::<Local>();
}

#[test]
fn shared_nested_get_traced_caught_keeps_outer_trace() {
    nested_caught_keeps_outer_trace::<Shared>();
}

#[test]
fn local_nested_get_traced_unwinds_outer_and_runtime_stays_usable() {
    nested_uncaught_unwinds_outer::<Local>();
}

#[test]
fn shared_nested_get_traced_unwinds_outer_and_runtime_stays_usable() {
    nested_uncaught_unwinds_outer::<Shared>();
}

#[test]
fn shared_concurrent_get_traced_is_refused_and_keeps_owner_trace() {
    let rt: Runtime<Shared> = Runtime::new();
    let a = rt.create_input(1_i64);
    let x = rt.create_input(7_i64);
    let b = rt.create_query(move |r| r.get(a) + 1);
    let armed = Arc::new(Barrier::new(2));
    let done = Arc::new(Barrier::new(2));
    let first = Arc::new(AtomicBool::new(true));
    let q = {
        let (armed, done) = (Arc::clone(&armed), Arc::clone(&done));
        rt.create_query(move |r| {
            let v = r.get(b);
            if first.swap(false, Ordering::SeqCst) {
                armed.wait();
                done.wait();
            }
            v * 2
        })
    };

    let (owner, intruder) = thread::scope(|s| {
        let owner = s.spawn(|| rt.get_traced(q));
        let intruder = s.spawn(|| {
            armed.wait();
            let result = catch_unwind(AssertUnwindSafe(|| rt.get_traced(x)));
            done.wait();
            result.err().map(panic_message)
        });
        (owner.join().unwrap(), intruder.join().unwrap())
    });

    let (value, trace) = owner;
    assert_eq!(value, 4);
    assert_eq!(trace.nodes_recomputed, 2, "{:?}", trace.node_traces);
    assert!(
        intruder.as_deref().is_some_and(|m| m.contains(NESTED)),
        "the concurrent get_traced was allowed: {intruder:?}"
    );
    assert!(!rt.tracing_is_armed());
    let (value, _) = rt.get_traced(x);
    assert_eq!(value, 7);
}

#[test]
fn shared_get_traced_started_during_the_owners_hand_off_keeps_the_owner_trace() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let a = rt.create_input(1_i64);
    let x = rt.create_input(7_i64);
    let q = rt.create_query(move |r| r.get(a) + 1);
    let y = rt.create_query(move |r| r.get(x) * 2);
    rt.get(q);
    rt.get(y);
    rt.set(a, 2);
    rt.set(x, 8);
    let intruder = Arc::new(Mutex::new(None::<Result<usize, String>>));
    let first = Arc::new(AtomicBool::new(true));
    {
        let weak = Arc::downgrade(&rt);
        let (intruder, first) = (Arc::clone(&intruder), Arc::clone(&first));
        let installed = rt.park_hook.set(Box::new(move |point| {
            if !matches!(point, super::ParkPoint::TraceHandOff)
                || !first.swap(false, Ordering::SeqCst)
            {
                return;
            }
            let rt = weak.upgrade().expect("the runtime outlives its get_traced");
            let result = thread::spawn(move || {
                catch_unwind(AssertUnwindSafe(|| recomputed(&rt.get_traced(y).1)))
                    .map_err(panic_message)
            })
            .join()
            .expect("the intruder thread joins");
            *intruder.lock().unwrap() = Some(result);
        }));
        assert!(installed.is_ok());
    }
    let (v, trace) = rt.get_traced(q);
    assert_eq!(v, 3);
    assert!(
        trace
            .node_traces
            .iter()
            .any(|t| t.id.0 == q.slot() && matches!(t.action, TraceAction::Recomputed { .. })),
        "the owner lost q's event: {:?}",
        trace.node_traces
    );
    assert!(
        trace.node_traces.iter().all(|t| t.id.0 != y.slot()),
        "the owner's trace holds the intruder's event: {:?}",
        trace.node_traces
    );
    let intruded = intruder
        .lock()
        .unwrap()
        .take()
        .expect("the hand-off park ran");
    assert_eq!(intruded, Err(NESTED.to_string()));
    assert_eq!(recomputed(&rt.get_traced(y).1), 1);
}

fn ids(snap: &crate::trace::BoundedSnapshot) -> Vec<u32> {
    snap.nodes.iter().map(|n| n.id.0).collect()
}

fn query<'a>(roots: &'a [NodeId], direction: Direction) -> GraphQuery<'a> {
    GraphQuery {
        roots,
        direction,
        depth: None,
        label_prefix: None,
        limit: None,
        max_visited: None,
    }
}

/// a0 <- q1 <- q2 <- q3 <- q4: each query reads the one before it.
fn chain<C: Cells>(rt: &Runtime<C>, len: usize) -> Vec<u32> {
    let a = rt.create_input(1_i64);
    let mut slots = vec![a.slot()];
    let mut prev = a;
    for _ in 1..len {
        let p = prev;
        let q = rt.create_query(move |r| r.get(p) + 1);
        let _ = rt.get(q);
        slots.push(q.slot());
        prev = q;
    }
    slots
}

fn bounded_depth_counts_edges_from_the_root<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let s = chain(&rt, 5);
    let roots = [NodeId(s[3])];
    let mut q = query(&roots, Direction::Dependencies);
    q.depth = Some(0);
    assert_eq!(ids(&rt.graph_snapshot_bounded(&q)), vec![s[3]]);
    q.depth = Some(1);
    let snap = rt.graph_snapshot_bounded(&q);
    assert_eq!(ids(&snap), vec![s[3], s[2]]);
    assert!(!snap.truncated);
    assert_eq!(snap.nodes[1].dependencies, vec![NodeId(s[1])]);
    assert_eq!(snap.nodes[1].dependents, vec![NodeId(s[3])]);
    q.depth = None;
    assert_eq!(
        ids(&rt.graph_snapshot_bounded(&q)),
        vec![s[3], s[2], s[1], s[0]]
    );
}

fn bounded_limit_caps_and_sets_truncated<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let s = chain(&rt, 6);
    let roots = [NodeId(s[5])];
    let mut q = query(&roots, Direction::Dependencies);
    q.limit = Some(3);
    let snap = rt.graph_snapshot_bounded(&q);
    assert_eq!(ids(&snap), vec![s[5], s[4], s[3]]);
    assert!(snap.truncated);
    q.limit = Some(6);
    let snap = rt.graph_snapshot_bounded(&q);
    assert_eq!(snap.nodes.len(), 6);
    assert!(!snap.truncated);
}

fn bounded_label_prefix_filters_but_walks_through<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let s = chain(&rt, 3);
    rt.set_label(s[0], "a/in".to_string());
    rt.set_label(s[1], "x/mid".to_string());
    rt.set_label(s[2], "a/out".to_string());
    let roots = [NodeId(s[2])];
    let mut q = query(&roots, Direction::Dependencies);
    q.label_prefix = Some("a/");
    let snap = rt.graph_snapshot_bounded(&q);
    assert_eq!(ids(&snap), vec![s[2], s[0]]);
    assert!(!snap.truncated);
    assert_eq!(snap.nodes[0].dependencies, vec![NodeId(s[1])]);
    rt.clear_label(s[2]);
    assert_eq!(ids(&rt.graph_snapshot_bounded(&q)), vec![s[0]]);
}

fn bounded_direction_dependents_and_both<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let s = chain(&rt, 4);
    let roots = [NodeId(s[1])];
    let mut q = query(&roots, Direction::Dependents);
    assert_eq!(ids(&rt.graph_snapshot_bounded(&q)), vec![s[1], s[2], s[3]]);
    q.direction = Direction::Both;
    q.depth = Some(1);
    let got = ids(&rt.graph_snapshot_bounded(&q));
    assert_eq!(got.len(), 3);
    assert_eq!(got[0], s[1]);
    assert!(got.contains(&s[0]) && got.contains(&s[2]));
    q.depth = None;
    assert_eq!(ids(&rt.graph_snapshot_bounded(&q)).len(), 4);
}

fn bounded_max_visited_truncates_a_filtered_walk<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let s = chain(&rt, 8);
    rt.set_label(s[0], "keep".to_string());
    let roots = [NodeId(s[7])];
    let mut q = query(&roots, Direction::Dependencies);
    q.label_prefix = Some("keep");
    q.max_visited = Some(4);
    let snap = rt.graph_snapshot_bounded(&q);
    assert!(snap.nodes.is_empty());
    assert!(snap.truncated);
    q.max_visited = Some(8);
    let snap = rt.graph_snapshot_bounded(&q);
    assert_eq!(ids(&snap), vec![s[0]]);
    assert!(!snap.truncated);
    let rt: Runtime<C> = Runtime::new();
    let s = chain(&rt, 70);
    rt.set_label(s[0], "keep".to_string());
    let roots = [NodeId(s[69])];
    let mut q = query(&roots, Direction::Dependencies);
    q.label_prefix = Some("keep");
    q.limit = Some(1);
    let snap = rt.graph_snapshot_bounded(&q);
    assert!(snap.nodes.is_empty(), "limit 1 defaults max_visited to 64");
    assert!(snap.truncated);
    q.max_visited = Some(70);
    assert_eq!(ids(&rt.graph_snapshot_bounded(&q)), vec![s[0]]);
}

fn bounded_ignores_dead_roots_and_dedups<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let a = rt.create_input(1_i64);
    let b = rt.create_input(2_i64);
    let q1 = rt.create_query(move |r| r.get(a) + r.get(b));
    let _ = rt.get(q1);
    let roots = [NodeId(q1.slot()), NodeId(q1.slot())];
    let q = query(&roots, Direction::Dependencies);
    let snap = rt.graph_snapshot_bounded(&q);
    assert_eq!(snap.nodes.len(), 3);
    let dead = rt.create_input(3_i64);
    let dead_slot = dead.slot();
    rt.delete_node(dead);
    let roots = [NodeId(dead_slot)];
    let q = query(&roots, Direction::Both);
    assert!(rt.graph_snapshot_bounded(&q).nodes.is_empty());
}

fn bounded_keeps_edges_outside_the_returned_set<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let s = chain(&rt, 4);
    let roots = [NodeId(s[2])];
    let mut q = query(&roots, Direction::Dependencies);
    q.depth = Some(0);
    let snap = rt.graph_snapshot_bounded(&q);
    assert_eq!(snap.nodes[0].dependencies, vec![NodeId(s[1])]);
    assert_eq!(snap.nodes[0].dependents, vec![NodeId(s[3])]);
}

#[test]
fn local_bounded_snapshot_cases() {
    bounded_depth_counts_edges_from_the_root::<Local>();
    bounded_limit_caps_and_sets_truncated::<Local>();
    bounded_label_prefix_filters_but_walks_through::<Local>();
    bounded_direction_dependents_and_both::<Local>();
    bounded_max_visited_truncates_a_filtered_walk::<Local>();
    bounded_ignores_dead_roots_and_dedups::<Local>();
    bounded_keeps_edges_outside_the_returned_set::<Local>();
}

#[test]
fn shared_bounded_snapshot_cases() {
    bounded_depth_counts_edges_from_the_root::<Shared>();
    bounded_limit_caps_and_sets_truncated::<Shared>();
    bounded_label_prefix_filters_but_walks_through::<Shared>();
    bounded_direction_dependents_and_both::<Shared>();
    bounded_max_visited_truncates_a_filtered_walk::<Shared>();
    bounded_ignores_dead_roots_and_dedups::<Shared>();
    bounded_keeps_edges_outside_the_returned_set::<Shared>();
}
