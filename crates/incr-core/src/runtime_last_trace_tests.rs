//! A `get_traced` root keeps the compact form of its trace while the root
//! stays Clean at the stamps that trace left it with.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::handle::Incr;
use crate::kept_trace::{KeptTrace, KeptValue, StepValue};
use crate::node::NodeId;
use crate::state::NodeState;
use crate::trace::{PropagationTrace, Retain, RootEntries, StampPair, TraceAction, TraceCapture};
use crate::value::Value;

type Tag = Box<&'static str>;

/// Keeps `trace` with `held` as its first step's value.
fn keep<C: Cells, V: Send + Sync + 'static>(
    rt: &Runtime<C>,
    trace: &PropagationTrace,
    capture: TraceCapture,
    held: V,
) {
    let mut held = Some(held);
    let kept = rt.kept_trace(&capture, trace, |_, _| {
        held.take().map_or(StepValue::None, StepValue::Ref)
    });
    rt.retain_trace(capture, kept);
}

fn traced<C: Cells, T: Value>(rt: &Runtime<C>, q: Incr<T>, tag: &'static str) -> T {
    let (value, trace, capture) = rt.get_traced_retaining(q);
    keep(rt, &trace, capture, Tag::new(tag));
    value
}

fn kept<C: Cells, T: Value>(rt: &Runtime<C>, q: Incr<T>) -> Option<KeptTrace> {
    rt.last_trace(NodeId(q.slot()), q.generation())
        .expect("a live node has a last-trace answer")
        .trace
}

fn last<C: Cells, T: Value>(rt: &Runtime<C>, q: Incr<T>) -> (NodeState, Option<&'static str>) {
    let found = rt
        .last_trace(NodeId(q.slot()), q.generation())
        .expect("a live node has a last-trace answer");
    let tag = found.trace.and_then(|kept| {
        kept.steps().find_map(|step| match step.value {
            KeptValue::Ref(held) => Some(**held.get::<Tag>()),
            _ => None,
        })
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
    keep(&rt, &trace, capture, Tag::new("second"));
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
    keep(&rt, &trace, capture, Tag::new("second"));
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
    let (_, trace, capture) = rt.get_traced_retaining(q);
    keep(
        &rt,
        &trace,
        capture,
        Arc::new(CountDrop(Arc::clone(&dropped))),
    );
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
    let (_, trace, capture) = rt.get_traced_retaining(q);
    keep(rt, &trace, capture, touch());
    rt.set(a, 2);
    let (_, trace, capture) = rt.get_traced_retaining(q);
    keep(rt, &trace, capture, touch());
    assert_eq!(dropped.load(Ordering::SeqCst), 1, "the replaced payload");
    rt.delete_node(q);
    assert_eq!(
        dropped.load(Ordering::SeqCst),
        2,
        "the deleted root's payload"
    );
}

fn kept_totals_follow_keep_replace_and_delete<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    assert_eq!(rt.kept_trace_memory(), Default::default());
    let (a, q, _) = counted(&rt);
    let (_, p, _) = counted(&rt);
    traced(&rt, q, "q");
    traced(&rt, p, "p");
    let two = rt.kept_trace_memory();
    let body = kept(&rt, q).expect("q's trace").heap_bytes();
    let table = two.bytes - 2 * body;
    assert_eq!((two.count, two.refs), (2, 2));
    assert_eq!(
        table,
        crate::memory::table_bytes::<(u32, KeptTrace)>(rt.inner.read().last_traces.capacity())
    );
    assert!(table > 0);

    rt.set(a, 3);
    traced(&rt, q, "replaced");
    assert_eq!(
        rt.kept_trace_memory(),
        two,
        "a replacement of the same shape"
    );

    rt.delete_node(q);
    let one = rt.kept_trace_memory();
    assert_eq!((one.count, one.bytes, one.refs), (1, table + body, 1));
    rt.delete_node(p);
    let none = rt.kept_trace_memory();
    assert_eq!((none.count, none.bytes, none.refs), (0, table, 0));
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
        runtime: crate::handle::RuntimeId::PENDING,
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

fn a_capture_from_another_runtime_is_refused<C: Cells>() {
    let traced: Runtime<C> = Runtime::new();
    let (_, q, _) = counted(&traced);
    for _ in 0..3 {
        traced.create_input(0_i64);
    }
    let other: Runtime<C> = Runtime::new();
    let (_, trace, capture) = traced.get_traced_retaining(q);
    let foreign = traced.kept_trace(&capture, &trace, |_, _| StepValue::<Tag>::None);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        other.retain_trace(capture, foreign);
    }));
    let payload = refused.expect_err("a foreign capture was accepted");
    let message = payload
        .downcast_ref::<&str>()
        .map(|m| m.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(message.contains("another runtime"), "got: {message}");
    assert_eq!(other.inner.read().last_traces.capacity(), 0);
}

fn a_trace_for_another_capture_or_shared_is_refused<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let (a, q, _) = counted(&rt);
    let (_, p, _) = counted(&rt);
    rt.delete_node(p);
    let (_, trace, first) = rt.get_traced_retaining(q);
    let kept = rt.kept_trace(&first, &trace, |_, _| StepValue::<Tag>::None);
    let clone = kept.clone();
    let shared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rt.retain_trace(first, kept);
    }));
    assert!(panic_text(shared.expect_err("a shared trace was kept")).contains("shared"));
    drop(clone);

    let reused = rt.create_query(move |r: &Runtime<C>| r.get(a) + 1);
    assert_eq!(
        (reused.slot(), reused.generation()),
        (p.slot(), p.generation() + 1)
    );
    let (_, trace, other) = rt.get_traced_retaining(q);
    let (_, _, wrong) = rt.get_traced_retaining(reused);
    let kept = rt.kept_trace(&other, &trace, |_, _| StepValue::<Tag>::None);
    let foreign = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rt.retain_trace(wrong, kept);
    }));
    assert!(
        panic_text(foreign.expect_err("a trace of another capture was kept"))
            .contains("another capture")
    );
    assert_eq!(rt.kept_trace_memory().count, 0);
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|m| m.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

fn only_this_runtimes_handles_arm_a_trace_root<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    rt.set_placeholder::<i64>(-1);
    let own = rt.create_input(1_i64);
    assert_eq!(
        rt.trace_root_for(own).map(|(slot, _)| slot),
        Some(own.slot())
    );
    assert!(rt.trace_root_for(rt.pending_handle::<i64>()).is_none());

    let other: Runtime<C> = Runtime::new();
    let foreign = other.create_input(2_i64);
    assert!(rt.trace_root_for(foreign).is_none());
    let by_get = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(foreign)));
    let by_trace = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rt.get_traced_retaining(foreign).0
    }));
    assert_eq!(
        panic_text(by_trace.expect_err("a foreign handle was traced")),
        panic_text(by_get.expect_err("a foreign handle was read"))
    );
    assert!(!rt.tracing_is_armed());
}

fn cases<C: Cells>() {
    recomputed_root_keeps_its_trace_until_dirty::<C>();
    traced_read_of_a_clean_root_leaves_the_trace::<C>();
    traced_verify_clean_restamps_the_trace::<C>();
    plain_verify_clean_hides_the_trace_until_a_traced_recompute::<C>();
    plain_get_traced_and_get_store_nothing::<C>();
    delete_drops_the_payload_and_a_reused_slot_starts_empty::<C>();
    node_info_reads_one_node_by_id::<C>();
    kept_totals_follow_keep_replace_and_delete::<C>();
    a_trace_for_another_capture_or_shared_is_refused::<C>();
    a_capture_from_another_runtime_is_refused::<C>();
    only_this_runtimes_handles_arm_a_trace_root::<C>();
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
