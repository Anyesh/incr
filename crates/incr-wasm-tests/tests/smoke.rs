#![cfg(target_arch = "wasm32")]

use incr_core::{Local, Runtime};
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn local_input_query_set_get() {
    let rt: Runtime<Local> = Runtime::new();
    let a = rt.create_input(2_u64);
    let doubled = rt.create_query(move |rt| rt.get(a) * 2);
    assert_eq!(rt.get(doubled), 4);
    rt.set(a, 5);
    assert_eq!(rt.get(doubled), 10);
}

#[wasm_bindgen_test]
fn get_traced_runs_without_a_clock() {
    let rt: Runtime<Local> = Runtime::new();
    let a = rt.create_input(1_u64);
    let inc = rt.create_query(move |rt| rt.get(a) + 1);
    let (value, trace) = rt.get_traced(inc);
    assert_eq!(value, 2);
    assert_eq!(trace.nodes_recomputed, 1);
    assert_eq!(trace.elapsed_ns, 0);
}

/// A ring of 64 Fallback queries resolves to its fallbacks through R2
/// reads alone, so it never unwinds, which `panic = abort` would turn
/// into an abort.
#[wasm_bindgen_test]
fn fallback_ring_of_64_resolves_without_unwinding() {
    use std::sync::{Arc, OnceLock};

    let rt: Runtime<Local> = Runtime::new();
    let first: Arc<OnceLock<incr_core::Incr<u64>>> = Arc::new(OnceLock::new());
    let mut ring = Vec::new();
    let mut next: Option<incr_core::Incr<u64>> = None;
    for i in (0..64_u64).rev() {
        let first = first.clone();
        let q = rt.create_query_with_fallback(
            move |rt| match next {
                Some(n) => rt.get(n) + 1,
                None => rt.get(*first.get().unwrap()) + 1,
            },
            move |_c: &incr_core::Cycle| 1000 + i,
        );
        next = Some(q);
        ring.push(q);
    }
    ring.reverse();
    first.set(ring[0]).ok().unwrap();
    for (i, &q) in ring.iter().enumerate() {
        assert_eq!(rt.get(q), 1000 + i as u64);
    }
    assert_eq!(rt.cycle_of(ring[5]).unwrap().len(), 64);
}
