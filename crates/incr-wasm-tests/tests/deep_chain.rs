#![cfg(target_arch = "wasm32")]

use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
use std::sync::Arc;

use incr_core::{Local, QueryOpts, Replay, Runtime, DEFAULT_MAX_FRAMES};
use wasm_bindgen_test::wasm_bindgen_test;

/// A first compute of a 100k chain of replayable queries at the default
/// budget nests at most 256 frames, inside the harness's shadow stack and
/// V8 limit, and runs each node at most twice.
#[wasm_bindgen_test]
fn chain_of_100k_replays_within_the_frame_budget() {
    const N: usize = 100_000;
    let rt: Runtime<Local> = Runtime::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(DEFAULT_MAX_FRAMES);
    let input = rt.create_input(0_u64);
    let runs: Arc<Vec<AtomicU32>> = Arc::new((0..N).map(|_| AtomicU32::new(0)).collect());
    let mut top = input;
    for i in (0..N).rev() {
        let runs = runs.clone();
        let below = top;
        top = rt.create_query_with(
            move |rt| {
                runs[i].fetch_add(1, SeqCst);
                rt.get(below) + 1
            },
            QueryOpts {
                replay: Replay::Allowed,
                ..Default::default()
            },
        );
    }
    assert_eq!(rt.get(top), N as u64);
    let max = runs.iter().map(|r| r.load(SeqCst)).max().unwrap();
    assert!(max <= 2, "a node ran {max} times");
}
