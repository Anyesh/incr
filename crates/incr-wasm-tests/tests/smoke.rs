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
