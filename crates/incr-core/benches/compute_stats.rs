//! R5: what compute stats and tracing cost a propagation. Each workload
//! (a chain of `INCR_STATS_CHAIN` levels, default 1,000, and a fan-out of
//! `INCR_STATS_FANOUT` queries over one input, default 1,000) is timed per
//! `set` plus read of the output with instrumentation off, with
//! `enable_compute_stats(true)`, and with the read made through
//! `get_traced`, on `Local` and `Shared`.
//!
//! A small run that checks the bench works:
//! `INCR_STATS_CHAIN=50 INCR_STATS_FANOUT=50 cargo bench -p incr-core --bench compute_stats -- --quick`.

use criterion::{criterion_group, criterion_main, Criterion};
use incr_core::{Cells, Incr, Local, Runtime, Shared};
use std::hint::black_box;

fn env_or(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a whole number, got {v:?}")),
        Err(_) => default,
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Off,
    Stats,
    Trace,
}

impl Mode {
    const ALL: [(&'static str, Mode); 3] = [
        ("off", Mode::Off),
        ("stats", Mode::Stats),
        ("trace", Mode::Trace),
    ];
}

struct Graph<C: Cells> {
    rt: Runtime<C>,
    input: Incr<i64>,
    output: Incr<i64>,
}

fn chain<C: Cells>(levels: usize) -> Graph<C> {
    let rt: Runtime<C> = Runtime::new();
    let input = rt.create_input(1_i64);
    let mut prev = input;
    for _ in 0..levels {
        let dep = prev;
        prev = rt.create_query(move |rt| rt.get(dep).wrapping_add(1));
    }
    black_box(rt.get(prev));
    Graph {
        rt,
        input,
        output: prev,
    }
}

fn fanout<C: Cells>(width: usize) -> Graph<C> {
    let rt: Runtime<C> = Runtime::new();
    let input = rt.create_input(1_i64);
    let leaves: Vec<Incr<i64>> = (0..width as i64)
        .map(|i| rt.create_query(move |rt| rt.get(input).wrapping_add(i)))
        .collect();
    let output = rt.create_query(move |rt| {
        leaves
            .iter()
            .fold(0_i64, |acc, &leaf| acc.wrapping_add(rt.get(leaf)))
    });
    black_box(rt.get(output));
    Graph { rt, input, output }
}

fn run<C: Cells>(c: &mut Criterion, shape: &str, build: impl Fn() -> Graph<C>, suffix: &str) {
    let mut group = c.benchmark_group(format!("compute_stats_{shape}_{suffix}"));
    for (name, mode) in Mode::ALL {
        let g = build();
        g.rt.enable_compute_stats(matches!(mode, Mode::Stats));
        let mut val = 1_i64;
        group.bench_function(name, |b| {
            b.iter(|| {
                val = val.wrapping_add(1);
                g.rt.set(g.input, val);
                match mode {
                    Mode::Trace => black_box(g.rt.get_traced(g.output).0),
                    Mode::Off | Mode::Stats => black_box(g.rt.get(g.output)),
                }
            })
        });
        if matches!(mode, Mode::Stats) {
            assert!(!g.rt.compute_stats(true).is_empty());
        }
    }
    group.finish();
}

fn benches(c: &mut Criterion) {
    let levels = env_or("INCR_STATS_CHAIN", 1_000);
    let width = env_or("INCR_STATS_FANOUT", 1_000);
    run::<Local>(c, "chain", || chain(levels), "local");
    run::<Shared>(c, "chain", || chain(levels), "shared");
    run::<Local>(c, "fanout", || fanout(width), "local");
    run::<Shared>(c, "fanout", || fanout(width), "shared");
}

criterion_group!(group, benches);
criterion_main!(group);
