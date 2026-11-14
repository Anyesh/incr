//! R7: `Runtime::graph_summary` over a large graph. `INCR_SUMMARY_NODES`
//! nodes (default 1,000,000) are built as labelled groups: one input, then
//! queries in `INCR_SUMMARY_GROUPS` label groups (default 20), each query
//! reading the previous node of its group, so the walk sees every node and
//! edge. Reports build time once on stderr and times each summary call.
//!
//! A small run that checks the bench works:
//! `INCR_SUMMARY_NODES=20000 cargo bench -p incr-core --bench graph_summary -- --quick`.

use criterion::{criterion_group, criterion_main, Criterion};
use incr_core::{Cells, Incr, Local, Runtime, Shared};
use std::hint::black_box;
use std::time::Instant;

fn env_or(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a whole number, got {v:?}")),
        Err(_) => default,
    }
}

fn build<C: Cells>(nodes: usize, groups: usize) -> Runtime<C> {
    let rt: Runtime<C> = Runtime::new();
    let input = rt.create_input(1_i64);
    rt.set_label(input.slot(), "input".to_string());
    let mut tails: Vec<Incr<i64>> = vec![input; groups];
    for i in 0..nodes.saturating_sub(1) {
        let g = i % groups;
        let dep = tails[g];
        let node = rt.create_query(move |rt| rt.get(dep).wrapping_add(1));
        rt.set_label(node.slot(), format!("group{g}/{i}"));
        tails[g] = node;
    }
    rt
}

fn bench<C: Cells>(c: &mut Criterion, suffix: &str) {
    let nodes = env_or("INCR_SUMMARY_NODES", 1_000_000);
    let groups = env_or("INCR_SUMMARY_GROUPS", 20);
    let started = Instant::now();
    let rt = build::<C>(nodes, groups);
    eprintln!(
        "graph_summary_{suffix}: {nodes} nodes in {groups} groups built in {:?}",
        started.elapsed()
    );
    let first = rt.graph_summary();
    assert!(!first.groups.is_empty());
    let mut group = c.benchmark_group("graph_summary");
    group.sample_size(10);
    group.bench_function(suffix, |b| b.iter(|| black_box(rt.graph_summary())));
    group.finish();
}

fn benches(c: &mut Criterion) {
    bench::<Local>(c, "local");
    bench::<Shared>(c, "shared");
}

criterion_group!(group, benches);
criterion_main!(group);
