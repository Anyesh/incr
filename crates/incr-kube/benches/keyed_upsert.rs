//! `KeyedCollection::upsert` throughput. Steady-state (collection
//! pre-populated to N keys, then repeated same-key replaces) mirrors
//! `incr-core`'s `benches/operators.rs` convention. The contended
//! variant has no equivalent there (that file has no multi-threaded
//! case); it follows `tests/stress.rs`'s pattern instead: reader
//! threads spawned with a shared stop flag before the timed loop.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use incr_concurrent::Runtime;
use incr_kube::KeyedCollection;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn build(size: usize) -> (Runtime, KeyedCollection<u64, u64>, u64) {
    let rt = Runtime::new();
    let kc: KeyedCollection<u64, u64> = KeyedCollection::new(&rt);
    for key in 0..size as u64 {
        kc.upsert(&rt, key, 1, key);
    }
    (rt, kc, size as u64)
}

fn bench_upsert_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("keyed_upsert_baseline");
    for size in [1_000_usize, 10_000, 100_000] {
        group.bench_with_input(BenchmarkId::new("upsert", size), &size, |b, &size| {
            let (rt, kc, _) = build(size);
            let key = 0_u64;
            let mut version = 2_u64;
            b.iter(|| {
                kc.upsert(&rt, key, version, version);
                version += 1;
                black_box(kc.get(&key));
            });
        });
    }
    group.finish();
}

fn bench_upsert_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("keyed_upsert_contended");
    for size in [1_000_usize, 10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::new("upsert_4_readers", size),
            &size,
            |b, &size| {
                let (rt, kc, key_count) = build(size);
                let rt = Arc::new(rt);
                let kc = Arc::new(kc);
                let count = kc.collection().count(&rt);

                let stop = Arc::new(AtomicBool::new(false));
                let readers: Vec<_> = (0..4)
                    .map(|_| {
                        let rt = Arc::clone(&rt);
                        let stop = Arc::clone(&stop);
                        std::thread::spawn(move || {
                            while !stop.load(Ordering::Relaxed) {
                                black_box(rt.get(count));
                            }
                        })
                    })
                    .collect();

                let key = 0_u64;
                let mut version = 2_u64;
                b.iter(|| {
                    kc.upsert(&rt, key, version, version);
                    version += 1;
                });

                stop.store(true, Ordering::Relaxed);
                for r in readers {
                    r.join().unwrap();
                }
                assert_eq!(kc.len() as u64, key_count);
            },
        );
    }
    group.finish();
}

criterion_group!(
    keyed_upsert_benches,
    bench_upsert_baseline,
    bench_upsert_contended
);
criterion_main!(keyed_upsert_benches);
