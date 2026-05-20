//! Collection operator benches. Measures the per-insert cost on a
//! steady-state collection (size N pre-populated) for each operator
//! pipeline, comparing against a from-scratch HashSet/Vec baseline.
//!
//! The README claims ~14x speedup at 10K elements and ~186x at 100K
//! for incremental vs batch on a filter+map+count pipeline. This bench
//! validates that the v0.2 incr-core matches those numbers through the
//! type-aliased Runtime<Local> path.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use incr_core::{Cells, IncrCollection, Local, Runtime, Shared};
use std::collections::HashSet;

fn build_pipeline<C: Cells>(
    size: usize,
) -> (Runtime<C>, IncrCollection<i64, C>, incr_core::Incr<u64>)
where
    Runtime<C>: Default,
{
    let rt: Runtime<C> = Runtime::new();
    let col: IncrCollection<i64, C> = rt.create_collection();
    let evens = col.filter(&rt, |x| x % 2 == 0);
    let doubled = evens.map(&rt, |x| x * 2);
    let count = doubled.count(&rt);
    for i in 0..size as i64 {
        col.insert(&rt, i);
    }
    let _ = rt.get(count);
    (rt, col, count)
}

fn bench_collection_incremental(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_insert_then_read");
    for size in [1_000_usize, 10_000, 100_000] {
        // incr-core Local: incremental delta-log path
        group.bench_with_input(
            BenchmarkId::new("local_incremental", size),
            &size,
            |b, &size| {
                let (rt, col, count) = build_pipeline::<Local>(size);
                let mut next = size as i64;
                b.iter(|| {
                    col.insert(&rt, next);
                    next += 1;
                    black_box(rt.get(count));
                });
            },
        );

        // incr-core Shared: same pipeline, atomic strategy
        group.bench_with_input(
            BenchmarkId::new("shared_incremental", size),
            &size,
            |b, &size| {
                let (rt, col, count) = build_pipeline::<Shared>(size);
                let mut next = size as i64;
                b.iter(|| {
                    col.insert(&rt, next);
                    next += 1;
                    black_box(rt.get(count));
                });
            },
        );

        // Batch baseline: full HashSet rebuild + filter + map + count per
        // insert. The pessimistic comparison the README uses.
        group.bench_with_input(BenchmarkId::new("batch", size), &size, |b, &size| {
            let mut elements: HashSet<i64> = (0..size as i64).collect();
            let mut next = size as i64;
            b.iter(|| {
                elements.insert(next);
                next += 1;
                let result: usize = elements
                    .iter()
                    .filter(|x| *x % 2 == 0)
                    .map(|x| x * 2)
                    .count();
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_simple_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_count");
    for size in [1_000_usize, 10_000] {
        group.bench_with_input(BenchmarkId::new("local", size), &size, |b, &size| {
            let rt: Runtime<Local> = Runtime::new();
            let col: IncrCollection<i64, Local> = rt.create_collection();
            let n = col.count(&rt);
            for i in 0..size as i64 {
                col.insert(&rt, i);
            }
            let _ = rt.get(n);
            let mut next = size as i64;
            b.iter(|| {
                col.insert(&rt, next);
                next += 1;
                black_box(rt.get(n));
            });
        });

        group.bench_with_input(BenchmarkId::new("shared", size), &size, |b, &size| {
            let rt: Runtime<Shared> = Runtime::new();
            let col: IncrCollection<i64, Shared> = rt.create_collection();
            let n = col.count(&rt);
            for i in 0..size as i64 {
                col.insert(&rt, i);
            }
            let _ = rt.get(n);
            let mut next = size as i64;
            b.iter(|| {
                col.insert(&rt, next);
                next += 1;
                black_box(rt.get(n));
            });
        });
    }
    group.finish();
}

criterion_group!(
    operator_benches,
    bench_collection_incremental,
    bench_simple_count
);
criterion_main!(operator_benches);
