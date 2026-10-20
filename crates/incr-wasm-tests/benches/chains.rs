//! First computes of deep chains on Node. Run with
//! `cargo bench -p incr-wasm-tests --target wasm32-unknown-unknown`, adding
//! `--profile dev` for the debug build. Each sample builds a fresh runtime
//! outside the timed part, so only the first `get`s are measured.
#![cfg(target_arch = "wasm32")]

use std::hint::black_box;
use std::time::Duration;

use incr_core::{Incr, Local, QueryOpts, Replay, Runtime, DEFAULT_MAX_FRAMES};
use wasm_bindgen_test::{wasm_bindgen_bench, Criterion, Instant};

/// Times `compute` on a fresh `build()` for each iteration, and checks that
/// it returns `expected`.
fn first_compute<S>(
    c: &mut Criterion,
    name: &str,
    build: impl Fn() -> S,
    compute: impl Fn(&S) -> u64,
    expected: u64,
) {
    // A sample builds a whole graph, so the default 100 samples would spend
    // most of the run building.
    *c = std::mem::take(c)
        .sample_size(10)
        .with_location(file!(), module_path!());
    c.bench_function(name, |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let graph = build();
                let start = Instant::now();
                let value = black_box(compute(&graph));
                total += start.elapsed();
                assert_eq!(value, expected);
            }
            total
        })
    });
}

/// T8's chain: 100k replayable queries over one input, with a placeholder
/// registered for the type they read, at the default frame budget.
fn replayable_chain_100k() -> (Runtime<Local>, Incr<u64>) {
    let rt: Runtime<Local> = Runtime::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(DEFAULT_MAX_FRAMES);
    let mut top = rt.create_input(0_u64);
    for _ in 0..100_000 {
        let below = top;
        top = rt.create_query_with(
            move |rt| rt.get(below) + 1,
            QueryOpts {
                replay: Replay::Allowed,
                ..Default::default()
            },
        );
    }
    (rt, top)
}

#[wasm_bindgen_bench]
fn chain_100k_first_compute(c: &mut Criterion) {
    first_compute(
        c,
        "chain_100k_first_compute",
        replayable_chain_100k,
        |(rt, top)| rt.get(*top),
        100_000,
    );
}

/// 1000 inputs, each under a chain of 100 queries, so reading a chain's
/// end nests 100 levels.
fn chains_1000x100() -> (Runtime<Local>, Vec<Incr<u64>>) {
    let rt: Runtime<Local> = Runtime::new();
    let ends = (0..1000_u64)
        .map(|head| {
            let mut end = rt.create_input(head);
            for _ in 0..100 {
                let below = end;
                end = rt.create_query(move |rt| rt.get(below) + 1);
            }
            end
        })
        .collect();
    (rt, ends)
}

#[wasm_bindgen_bench]
fn chains_1000x100_first_compute(c: &mut Criterion) {
    first_compute(
        c,
        "chains_1000x100_first_compute",
        chains_1000x100,
        |(rt, ends)| ends.iter().map(|end| rt.get(*end)).sum(),
        // each chain's end is its head plus 100
        (0..1000).sum::<u64>() + 1000 * 100,
    );
}
