//! Chain-propagation bench through the `incr-compute` v0.2 wrapper.
//! Confirms the thin re-export adds no measurable cost beyond the
//! `incr-core` bench numbers.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use incr_compute::{Incr, Runtime};

fn build_chain(n: usize) -> (Runtime, Incr<i64>, Incr<i64>) {
    let rt = Runtime::new();
    let input = rt.create_input(1_i64);
    let mut prev = input;
    for _ in 0..n {
        let dep = prev;
        prev = rt.create_query(move |rt| rt.get(dep).wrapping_add(1));
    }
    let _ = rt.get(prev);
    (rt, input, prev)
}

fn bench_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("incr_compute_chain");
    for size in [4_usize, 10, 100] {
        group.bench_with_input(BenchmarkId::new("propagate", size), &size, |b, &size| {
            let (rt, input, output) = build_chain(size);
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

fn bench_diamond(c: &mut Criterion) {
    let rt = Runtime::new();
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

    c.bench_function("incr_compute_diamond", |bencher| {
        let mut val = 1_i64;
        bencher.iter(|| {
            val = val.wrapping_add(1);
            rt.set(input, val);
            black_box(rt.get(out));
        });
    });
}

criterion_group!(benches, bench_chain, bench_diamond);
criterion_main!(benches);
