//! `get_traced` owns the runtime's single trace log for its whole call, so
//! a second one, nested in a compute or on another thread, is refused.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::trace::TraceAction;

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
