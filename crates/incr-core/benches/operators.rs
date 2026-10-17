//! Collection operator benches. Measures the per-insert cost on a
//! steady-state collection (size N pre-populated) for each operator
//! pipeline, comparing against a from-scratch HashSet/Vec baseline.
//!
//! The README claims ~14x speedup at 10K elements and ~186x at 100K
//! for incremental vs batch on a filter+map+count pipeline. This bench
//! validates that the v0.2 incr-core matches those numbers through the
//! type-aliased Runtime<Local> path.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use incr_core::{Local, Shared};
use workloads::{BatchRebuild, InsertThenRead};

// Allowed because each bench target uses only its own slice of the
// shared workloads.
#[allow(dead_code)]
mod workloads;

fn bench_collection_incremental(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_insert_then_read");
    for size in [1_000_usize, 10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::new("local_incremental", size),
            &size,
            |b, &size| workloads::steady(b, &InsertThenRead::<Local>::pipeline(size)),
        );
        group.bench_with_input(
            BenchmarkId::new("shared_incremental", size),
            &size,
            |b, &size| workloads::steady(b, &InsertThenRead::<Shared>::pipeline(size)),
        );
        // Full HashSet rebuild plus filter, map and count per insert: the
        // pessimistic comparison the README uses.
        group.bench_with_input(BenchmarkId::new("batch", size), &size, |b, &size| {
            workloads::steady(b, &BatchRebuild::new(size))
        });
    }
    group.finish();
}

fn bench_simple_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_count");
    for size in [1_000_usize, 10_000] {
        group.bench_with_input(BenchmarkId::new("local", size), &size, |b, &size| {
            workloads::steady(b, &InsertThenRead::<Local>::simple_count(size))
        });
        group.bench_with_input(BenchmarkId::new("shared", size), &size, |b, &size| {
            workloads::steady(b, &InsertThenRead::<Shared>::simple_count(size))
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
