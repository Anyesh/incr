//! A `get_traced` root keeps the payload its wheel rendered from the trace
//! while the root stays Clean at the stamps that trace left it with.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::handle::Incr;
use crate::node::NodeId;
use crate::state::NodeState;
use crate::trace::{Retain, RootEntries, StampPair, TraceAction, TraceCapture};
use crate::value::Value;

fn traced<C: Cells, T: Value>(rt: &Runtime<C>, q: Incr<T>, tag: &'static str) -> T {
    let (value, _, capture) = rt.get_traced_retaining(q);
    rt.retain_trace(capture, Arc::new(tag));
    value
}

fn last<C: Cells, T: Value>(rt: &Runtime<C>, q: Incr<T>) -> (NodeState, Option<&'static str>) {
    let found = rt
        .last_trace(NodeId(q.slot()), q.generation())
        .expect("a live node has a last-trace answer");
    let tag = found.payload.map(|p| {
        *p.downcast_ref::<&'static str>()
            .expect("the payload's type")
    });
    (found.state, tag)
}

fn counted<C: Cells>(rt: &Runtime<C>) -> (Incr<i64>, Incr<i64>, Arc<AtomicUsize>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let a = rt.create_input(1_i64);
    let seen = Arc::clone(&runs);
    let q = rt.create_query(move |r: &Runtime<C>| {
        seen.fetch_add(1, Ordering::SeqCst);
        r.get(a) * 2
    });
    (a, q, runs)
}

fn parity_chain<C: Cells>(rt: &Runtime<C>) -> (Incr<i64>, Incr<i64>) {
    let a = rt.create_input(1_i64);
    let parity = rt.create_query(move |r: &Runtime<C>| r.get(a) % 2);
    let q = rt.create_query(move |r: &Runtime<C>| r.get(parity) + 10);
    (a, q)
}

fn recomputed_root_keeps_its_trace_until_dirty<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let (a, q, runs) = counted(&rt);
    assert_eq!(last(&rt, q), (NodeState::New, None));
    assert_eq!(traced(&rt, q, "first"), 2);
    assert_eq!(last(&rt, q), (NodeState::Clean, Some("first")));

    rt.set(a, 5);
    let before = runs.load(Ordering::SeqCst);
    assert_eq!(last(&rt, q), (NodeState::Dirty, None));
    assert_eq!(
        runs.load(Ordering::SeqCst),
        before,
        "last_trace ran a compute"
    );

    assert_eq!(traced(&rt, q, "second"), 10);
    assert_eq!(last(&rt, q), (NodeState::Clean, Some("second")));
}

fn traced_read_of_a_clean_root_leaves_the_trace<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let (_, q, _) = counted(&rt);
    traced(&rt, q, "first");
    let (_, trace, capture) = rt.get_traced_retaining(q);
    assert!(trace.node_traces.iter().all(|t| t.id.0 != q.slot()));
    rt.retain_trace(capture, Arc::new("second"));
    assert_eq!(last(&rt, q), (NodeState::Clean, Some("first")));
}

fn traced_verify_clean_restamps_the_trace<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let (a, q) = parity_chain(&rt);
    traced(&rt, q, "first");
    rt.set(a, 3);
    let (value, trace, capture) = rt.get_traced_retaining(q);
    assert_eq!(value, 11);
    let root: Vec<_> = trace
        .node_traces
        .iter()
        .filter(|t| t.id.0 == q.slot())
        .map(|t| t.action.clone())
        .collect();
    assert_eq!(root, vec![TraceAction::VerifiedClean]);
    rt.retain_trace(capture, Arc::new("second"));
    assert_eq!(last(&rt, q), (NodeState::Clean, Some("first")));
}

fn plain_verify_clean_hides_the_trace_until_a_traced_recompute<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let (a, q) = parity_chain(&rt);
    traced(&rt, q, "first");
    rt.set(a, 3);
    assert_eq!(rt.get(q), 11);
    assert_eq!(last(&rt, q), (NodeState::Clean, None));

    rt.set(a, 5);
    traced(&rt, q, "verified");
    assert_eq!(
        last(&rt, q),
        (NodeState::Clean, None),
        "a verify over a hidden trace must not bring it back"
    );

    rt.set(a, 4);
    assert_eq!(traced(&rt, q, "third"), 10);
    assert_eq!(last(&rt, q), (NodeState::Clean, Some("third")));
}

fn plain_get_traced_and_get_store_nothing<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let (a, q, _) = counted(&rt);
    rt.get(q);
    rt.set(a, 2);
    rt.get_traced(q);
    rt.set(a, 3);
    let (_, _, capture) = rt.get_traced_retaining(q);
    drop(capture);
    assert_eq!(rt.inner.read().last_traces.capacity(), 0);
    assert_eq!(last(&rt, q), (NodeState::Clean, None));
}

struct CountDrop(Arc<AtomicUsize>);

impl Drop for CountDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn delete_drops_the_payload_and_a_reused_slot_starts_empty<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let (_, q, _) = counted(&rt);
    let dropped = Arc::new(AtomicUsize::new(0));
    let (_, _, capture) = rt.get_traced_retaining(q);
    rt.retain_trace(capture, Arc::new(CountDrop(Arc::clone(&dropped))));
    rt.delete_node(q);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert!(rt.last_trace(NodeId(q.slot()), q.generation()).is_none());

    let reused = rt.create_input(9_i64);
    assert_eq!(reused.slot(), q.slot());
    assert_eq!(last(&rt, reused), (NodeState::Clean, None));
    assert!(rt.last_trace(NodeId(u32::MAX - 1), 0).is_none());
}

struct RtPtr<C: Cells>(*const Runtime<C>);
// SAFETY: test-only. The runtime outlives every payload holding its
// pointer, and a Local runtime's pointer never leaves its thread.
unsafe impl<C: Cells> Send for RtPtr<C> {}
unsafe impl<C: Cells> Sync for RtPtr<C> {}

/// Calls back into the runtime from its Drop, as a Python `__del__` can.
struct TouchOnDrop<C: Cells> {
    rt: RtPtr<C>,
    dropped: Arc<AtomicUsize>,
}

impl<C: Cells> Drop for TouchOnDrop<C> {
    fn drop(&mut self) {
        // SAFETY: see RtPtr.
        let rt = unsafe { &*self.rt.0 };
        let probe = rt.create_input(0_i64);
        rt.delete_node(probe);
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

fn payloads_drop_outside_the_lock<C: Cells>(rt: &Runtime<C>) {
    let (a, q, _) = counted(rt);
    let dropped = Arc::new(AtomicUsize::new(0));
    let touch = || {
        Arc::new(TouchOnDrop {
            rt: RtPtr(rt as *const Runtime<C>),
            dropped: Arc::clone(&dropped),
        })
    };
    let (_, _, capture) = rt.get_traced_retaining(q);
    rt.retain_trace(capture, touch());
    rt.set(a, 2);
    let (_, _, capture) = rt.get_traced_retaining(q);
    rt.retain_trace(capture, touch());
    assert_eq!(dropped.load(Ordering::SeqCst), 1, "the replaced payload");
    rt.delete_node(q);
    assert_eq!(
        dropped.load(Ordering::SeqCst),
        2,
        "the deleted root's payload"
    );
}

fn node_info_reads_one_node_by_id<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let (a, q, runs) = counted(&rt);
    let info = rt.node_info(NodeId(q.slot()), q.generation()).unwrap();
    assert_eq!(info.state, NodeState::New);
    rt.get(q);
    rt.set(a, 4);
    let info = rt.node_info(NodeId(q.slot()), q.generation()).unwrap();
    assert_eq!((info.state, info.generation), (NodeState::Dirty, 0));
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert!(rt.node_info(NodeId(q.slot()), 1).is_none());
    assert!(rt.node_info(NodeId(u32::MAX - 1), 0).is_none());
    rt.delete_node(q);
    assert!(rt.node_info(NodeId(q.slot()), 0).is_none());
}

fn pair(verified_at: u64, changed_at: u64) -> StampPair {
    StampPair {
        verified_at,
        changed_at,
    }
}

fn capture(before: StampPair, entries: &[(TraceAction, StampPair)]) -> TraceCapture {
    let mut root = RootEntries::default();
    for (action, stamps) in entries {
        root.see(action, *stamps);
    }
    TraceCapture {
        slot: 0,
        generation: 0,
        before,
        root,
    }
}

#[test]
fn a_capture_keeps_only_what_its_root_entries_vouch_for() {
    let verified = TraceAction::VerifiedClean;
    let recomputed = TraceAction::Recomputed {
        value_changed: true,
    };
    assert_eq!(
        capture(pair(4, 3), &[(verified.clone(), pair(6, 3))]).decide(),
        Retain::Restamp {
            from: pair(4, 3),
            to: pair(6, 3)
        }
    );
    assert_eq!(
        capture(pair(4, 3), &[(verified.clone(), pair(6, 5))]).decide(),
        Retain::Nothing,
        "a verify never moves changed_at, so another run did"
    );
    assert_eq!(
        capture(
            pair(4, 3),
            &[
                (recomputed.clone(), pair(6, 6)),
                (verified.clone(), pair(7, 6))
            ]
        )
        .decide(),
        Retain::Nothing
    );
    assert_eq!(
        capture(pair(4, 3), &[(recomputed.clone(), pair(0, 6))]).decide(),
        Retain::Nothing,
        "a poisoned or memo landing reads verified_at 0"
    );
    assert_eq!(
        capture(pair(4, 3), &[(recomputed, pair(6, 6))]).decide(),
        Retain::Keep(pair(6, 6))
    );
    assert_eq!(capture(pair(4, 3), &[]).decide(), Retain::Nothing);
}

fn cases<C: Cells>() {
    recomputed_root_keeps_its_trace_until_dirty::<C>();
    traced_read_of_a_clean_root_leaves_the_trace::<C>();
    traced_verify_clean_restamps_the_trace::<C>();
    plain_verify_clean_hides_the_trace_until_a_traced_recompute::<C>();
    plain_get_traced_and_get_store_nothing::<C>();
    delete_drops_the_payload_and_a_reused_slot_starts_empty::<C>();
    node_info_reads_one_node_by_id::<C>();
}

#[test]
fn local_last_trace_cases() {
    cases::<Local>();
}

#[test]
fn shared_last_trace_cases() {
    cases::<Shared>();
}

#[test]
fn local_last_trace_payloads_drop_outside_the_lock() {
    payloads_drop_outside_the_lock(&Runtime::<Local>::new());
}

#[test]
fn shared_last_trace_payloads_drop_outside_the_lock() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        payloads_drop_outside_the_lock(&Runtime::<Shared>::new());
        tx.send(()).unwrap();
    });
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .expect("a payload drop deadlocked or panicked under the runtime lock");
}
