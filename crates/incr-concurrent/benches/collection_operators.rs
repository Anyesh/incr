use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use incr_concurrent::Runtime;

/// Batch: sort N timestamps, compute pairwise gaps, sum them.
fn batch_travel_premium(timestamps: &[i64]) -> i64 {
    let mut sorted = timestamps.to_vec();
    sorted.sort();
    sorted.windows(2).map(|w| w[1] - w[0]).sum()
}

/// Set up an incremental pipeline with N elements already inserted.
/// Returns (runtime, collection, reduce_node) ready for mutation benchmarks.
fn setup_incremental(
    n: usize,
) -> (
    Runtime,
    incr_concurrent::IncrCollection<i64>,
    incr_concurrent::Incr<i64>,
) {
    let rt = Runtime::new();
    let col = rt.create_collection::<i64>();
    let sorted = col.sort_by_key(&rt, |t: &i64| *t);
    let pairs = sorted.pairwise(&rt);
    let gaps = pairs.map(&rt, |(a, b): &(i64, i64)| b - a);
    let total = gaps.reduce(&rt, |elements| -> i64 { elements.iter().sum() });

    // Insert N elements with gaps of 10 between them
    for i in 0..n {
        col.insert(&rt, (i as i64) * 10);
    }
    // Warmup: stabilize the graph
    let _ = rt.get(total);

    (rt, col, total)
}

fn bench_incremental_vs_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("travel_premium");

    for &n in &[10, 20, 40, 100, 500, 1000, 5000] {
        // Batch benchmark
        let timestamps: Vec<i64> = (0..n).map(|i| (i as i64) * 10).collect();
        group.bench_with_input(BenchmarkId::new("batch", n), &timestamps, |b, ts| {
            b.iter(|| black_box(batch_travel_premium(ts)));
        });

        // Incremental benchmark: measure cost of changing one element and reading result
        group.bench_with_input(BenchmarkId::new("incremental_change", n), &n, |b, &n| {
            let (rt, col, total) = setup_incremental(n);
            // Change the middle element back and forth
            let mid = (n / 2) as i64 * 10;
            let mut toggle = true;
            b.iter(|| {
                if toggle {
                    col.delete(&rt, &mid);
                    col.insert(&rt, mid + 1); // shift by 1
                } else {
                    col.delete(&rt, &(mid + 1));
                    col.insert(&rt, mid); // shift back
                }
                let result = rt.get(total);
                toggle = !toggle;
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_incremental_vs_batch);
criterion_main!(benches);
