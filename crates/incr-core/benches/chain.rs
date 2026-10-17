//! Chain-propagation bench mirroring the comparison harness in
//! `incr-concurrent/benches/comparison.rs`. The point is to confirm the
//! consolidated `incr-core` runtime matches (or beats) the production
//! crates' per-node propagation cost.
//!
//! Workload: build a chain `input → f_1 → f_2 → ... → f_n` where each
//! `f_i` adds 1 to its predecessor. On each iteration, set a new input
//! value, then read the chain head. Criterion reports the total time
//! per iteration; dividing by `n` gives the per-node propagation cost.
//!
//! The production target was 175 ns per node propagation; the
//! consolidated `incr-core` should land within noise of that under
//! `Shared` and faster (no atomic-fence cost) under `Local`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use incr_core::{Local, Shared};
use workloads::{Chain, Diamond, EarlyCutoff};

// Allowed because each bench target uses only its own slice of the
// shared workloads.
#[allow(dead_code)]
mod workloads;

fn bench_chain_local(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_local");
    for size in [4_usize, 10, 100] {
        group.bench_with_input(BenchmarkId::new("propagate", size), &size, |b, &size| {
            workloads::steady(b, &Chain::<Local>::new(size));
        });
    }
    group.finish();
}

fn bench_chain_shared(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_shared");
    for size in [4_usize, 10, 100] {
        group.bench_with_input(BenchmarkId::new("propagate", size), &size, |b, &size| {
            workloads::steady(b, &Chain::<Shared>::new(size));
        });
    }
    group.finish();
}

fn bench_diamond_local(c: &mut Criterion) {
    let w = Diamond::<Local>::new();
    c.bench_function("diamond_local", |b| workloads::steady(b, &w));
}

fn bench_diamond_shared(c: &mut Criterion) {
    let w = Diamond::<Shared>::new();
    c.bench_function("diamond_shared", |b| workloads::steady(b, &w));
}

fn bench_early_cutoff_local(c: &mut Criterion) {
    let w = EarlyCutoff::<Local>::new();
    c.bench_function("early_cutoff_local", |b| workloads::steady(b, &w));
}

fn bench_early_cutoff_shared(c: &mut Criterion) {
    let w = EarlyCutoff::<Shared>::new();
    c.bench_function("early_cutoff_shared", |b| workloads::steady(b, &w));
}

criterion_group!(
    chain_benches,
    bench_chain_local,
    bench_chain_shared,
    bench_diamond_local,
    bench_diamond_shared,
    bench_early_cutoff_local,
    bench_early_cutoff_shared,
);
criterion_main!(chain_benches);
