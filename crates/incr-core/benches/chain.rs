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

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use incr_core::{Incr, Local, Runtime, Shared};

fn build_chain_local(n: usize) -> (Runtime<Local>, Incr<i64>, Incr<i64>) {
    let rt: Runtime<Local> = Runtime::new();
    let input = rt.create_input(1_i64);
    let mut prev = input;
    for _ in 0..n {
        let dep = prev;
        prev = rt.create_query(move |rt| rt.get(dep).wrapping_add(1));
    }
    let _ = rt.get(prev);
    (rt, input, prev)
}

fn build_chain_shared(n: usize) -> (Runtime<Shared>, Incr<i64>, Incr<i64>) {
    let rt: Runtime<Shared> = Runtime::new();
    let input = rt.create_input(1_i64);
    let mut prev = input;
    for _ in 0..n {
        let dep = prev;
        prev = rt.create_query(move |rt| rt.get(dep).wrapping_add(1));
    }
    let _ = rt.get(prev);
    (rt, input, prev)
}

fn bench_chain_local(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_local");
    for size in [4_usize, 10, 100] {
        group.bench_with_input(BenchmarkId::new("propagate", size), &size, |b, &size| {
            let (rt, input, output) = build_chain_local(size);
            let mut val = 1_i64;
            b.iter(|| {
                val = val.wrapping_add(1);
                rt.set(input, val);
                black_box(rt.get(output));
            });
        });
    }
    group.finish();
}

fn bench_chain_shared(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_shared");
    for size in [4_usize, 10, 100] {
        group.bench_with_input(BenchmarkId::new("propagate", size), &size, |b, &size| {
            let (rt, input, output) = build_chain_shared(size);
            let mut val = 1_i64;
            b.iter(|| {
                val = val.wrapping_add(1);
                rt.set(input, val);
                black_box(rt.get(output));
            });
        });
    }
    group.finish();
}

fn bench_diamond_local(c: &mut Criterion) {
    let rt: Runtime<Local> = Runtime::new();
    let input = rt.create_input(1_i64);
    let a = {
        let dep = input;
        rt.create_query(move |rt| rt.get(dep).wrapping_add(10))
    };
    let b = {
        let dep = input;
        rt.create_query(move |rt| rt.get(dep).wrapping_add(100))
    };
    let out = rt.create_query(move |rt| rt.get(a).wrapping_add(rt.get(b)));
    let _ = rt.get(out);

    c.bench_function("diamond_local", |bencher| {
        let mut val = 1_i64;
        bencher.iter(|| {
            val = val.wrapping_add(1);
            rt.set(input, val);
            black_box(rt.get(out));
        });
    });
}

fn bench_diamond_shared(c: &mut Criterion) {
    let rt: Runtime<Shared> = Runtime::new();
    let input = rt.create_input(1_i64);
    let a = {
        let dep = input;
        rt.create_query(move |rt| rt.get(dep).wrapping_add(10))
    };
    let b = {
        let dep = input;
        rt.create_query(move |rt| rt.get(dep).wrapping_add(100))
    };
    let out = rt.create_query(move |rt| rt.get(a).wrapping_add(rt.get(b)));
    let _ = rt.get(out);

    c.bench_function("diamond_shared", |bencher| {
        let mut val = 1_i64;
        bencher.iter(|| {
            val = val.wrapping_add(1);
            rt.set(input, val);
            black_box(rt.get(out));
        });
    });
}

fn bench_early_cutoff_local(c: &mut Criterion) {
    let rt: Runtime<Local> = Runtime::new();
    let input = rt.create_input(200_i64);
    let clamped = rt.create_query(move |rt| rt.get(input).min(100));
    let after = rt.create_query(move |rt| rt.get(clamped).wrapping_add(1));
    let _ = rt.get(after);

    c.bench_function("early_cutoff_local", |bencher| {
        let mut val = 200_i64;
        bencher.iter(|| {
            val = val.wrapping_add(1);
            rt.set(input, val); // always > 100, clamp produces 100, early cutoff
            black_box(rt.get(after));
        });
    });
}

fn bench_early_cutoff_shared(c: &mut Criterion) {
    let rt: Runtime<Shared> = Runtime::new();
    let input = rt.create_input(200_i64);
    let clamped = rt.create_query(move |rt| rt.get(input).min(100));
    let after = rt.create_query(move |rt| rt.get(clamped).wrapping_add(1));
    let _ = rt.get(after);

    c.bench_function("early_cutoff_shared", |bencher| {
        let mut val = 200_i64;
        bencher.iter(|| {
            val = val.wrapping_add(1);
            rt.set(input, val);
            black_box(rt.get(after));
        });
    });
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
