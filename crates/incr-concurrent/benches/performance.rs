// crates/incr-concurrent/benches/performance.rs
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use incr_concurrent::{Incr, Runtime};

/// Build a linear chain: input -> n1 -> n2 -> ... -> output
fn build_chain(size: usize) -> (Runtime, Incr<i64>, Incr<i64>) {
    let rt = Runtime::new();
    let input = rt.create_input(1_i64);
    let mut prev: Incr<i64> = input;
    for _ in 0..size {
        let dep = prev;
        prev = rt.create_query(move |rt| rt.get(dep).wrapping_add(1));
    }
    let _ = rt.get(prev);
    (rt, input, prev)
}

/// Build a wide fan-out: input -> [n1, n2, ..., n_width] -> output
fn build_fanout(width: usize) -> (Runtime, Incr<i64>, Incr<i64>) {
    let rt = Runtime::new();
    let input = rt.create_input(1_i64);
    let mut intermediates: Vec<Incr<i64>> = Vec::new();
    for i in 0..width {
        let dep = input;
        let offset = i as i64;
        intermediates.push(rt.create_query(move |rt| rt.get(dep).wrapping_add(offset)));
    }
    // Sum all intermediates
    let first = intermediates[0];
    let output = if intermediates.len() == 1 {
        first
    } else {
        let nodes = intermediates.clone();
        rt.create_query(move |rt| nodes.iter().map(|n| rt.get(*n)).sum::<i64>())
    };
    let _ = rt.get(output);
    (rt, input, output)
}

fn build_layered(
    num_inputs: usize,
    nodes_per_layer: usize,
    num_layers: usize,
) -> (Runtime, Vec<Incr<i64>>, Incr<i64>) {
    let rt = Runtime::new();
    let mut inputs = Vec::new();
    let mut all_nodes: Vec<Incr<i64>> = Vec::new();

    for i in 0..num_inputs {
        let node = rt.create_input(i as i64);
        inputs.push(node);
        all_nodes.push(node);
    }

    for _ in 0..num_layers {
        let available = all_nodes.len();
        for j in 0..nodes_per_layer {
            let a = all_nodes[j % available];
            let b = all_nodes[(j + 1) % available];
            let node = rt.create_query(move |rt| rt.get(a).wrapping_add(rt.get(b)));
            all_nodes.push(node);
        }
    }

    let last = *all_nodes.last().unwrap();
    let _ = rt.get(last);
    (rt, inputs, last)
}

fn bench_propagate_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("propagate_single_change");

    for size in [100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let (rt, input, output) = build_chain(size);
            let mut val = 1_i64;
            b.iter(|| {
                val += 1;
                rt.set(input, val);
                black_box(rt.get(output));
            });
        });
    }

    group.finish();
}

fn bench_early_cutoff(c: &mut Criterion) {
    c.bench_function("early_cutoff_chain_1000", |b| {
        let rt = Runtime::new();
        let input = rt.create_input(1_i64);
        let clamped = {
            let dep = input;
            rt.create_query(move |rt| rt.get(dep).min(100))
        };
        let mut prev: Incr<i64> = clamped;
        for _ in 0..999 {
            let dep = prev;
            prev = rt.create_query(move |rt| rt.get(dep).wrapping_add(1));
        }
        let output = prev;
        let _ = rt.get(output);

        // Set input to >100 so clamp activates
        rt.set(input, 200);
        let _ = rt.get(output);

        let mut val = 200_i64;
        b.iter(|| {
            val += 1;
            rt.set(input, val); // Clamped to 100, same as before
            black_box(rt.get(output));
        });
    });
}

fn bench_overhead_vs_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("overhead_vs_batch");

    for size in [100, 1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("incremental_initial", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let (rt, _, output) = build_chain(size);
                    black_box(rt.get(output));
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("batch_plain", size), &size, |b, &size| {
            b.iter(|| {
                let mut val = 1_i64;
                for _ in 0..size {
                    val = val.wrapping_add(1);
                }
                black_box(val);
            });
        });
    }

    group.finish();
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_with_graph_size");

    for &(inputs, per_layer, layers) in &[
        (10, 10, 1),    // ~20 nodes
        (10, 10, 10),   // ~110 nodes
        (10, 10, 100),  // ~1010 nodes
        (50, 50, 20),   // ~1050 nodes
        (100, 100, 10), // ~1100 nodes
    ] {
        let total = inputs + per_layer * layers;
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}n", total)),
            &(inputs, per_layer, layers),
            |b, &(inputs, per_layer, layers)| {
                let (rt, input_nodes, output) = build_layered(inputs, per_layer, layers);
                let mut val = 100_i64;
                b.iter(|| {
                    val += 1;
                    rt.set(input_nodes[0], val);
                    black_box(rt.get(output));
                });
            },
        );
    }

    group.finish();
}

fn bench_collection_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_insert_throughput");

    for size in [1_000, 10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}elem", size)),
            &size,
            |b, &size| {
                let rt = Runtime::new();
                let col = rt.create_collection::<i64>();
                let filtered = col.filter(&rt, |x| x % 2 == 0);
                let mapped = filtered.map(&rt, |x| x * 2);
                let count = mapped.count(&rt);

                for i in 0..size {
                    col.insert(&rt, i);
                }
                let _ = rt.get(count);

                let mut next = size;
                b.iter(|| {
                    col.insert(&rt, next);
                    next += 1;
                    black_box(rt.get(count));
                });
            },
        );
    }

    group.finish();
}

fn bench_collection_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_delete_throughput");

    for size in [1_000, 10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}elem", size)),
            &size,
            |b, &size| {
                let rt = Runtime::new();
                let col = rt.create_collection::<i64>();
                let filtered = col.filter(&rt, |x| x % 2 == 0);
                let count = filtered.count(&rt);

                for i in 0..size {
                    col.insert(&rt, i);
                }
                let _ = rt.get(count);

                let mut idx = 0_i64;
                b.iter(|| {
                    let val = idx % size;
                    col.delete(&rt, &val);
                    black_box(rt.get(count));
                    col.insert(&rt, val);
                    let _ = rt.get(count);
                    idx += 1;
                });
            },
        );
    }

    group.finish();
}

fn bench_collection_pipeline_depth(c: &mut Criterion) {
    c.bench_function("5_stage_pipeline_insert", |b| {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let stage1 = col.filter(&rt, |x| *x > 0);
        let stage2 = stage1.filter(&rt, |x| *x < 1_000_000);
        let stage3 = stage2.map(&rt, |x| x * 2);
        let stage4 = stage3.filter(&rt, |x| *x < 500_000);
        let count = stage4.count(&rt);

        for i in 1..10_001_i64 {
            col.insert(&rt, i);
        }
        let _ = rt.get(count);

        let mut next = 10_001_i64;
        b.iter(|| {
            col.insert(&rt, next);
            next += 1;
            black_box(rt.get(count));
        });
    });
}

criterion_group!(
    benches,
    bench_propagate_single,
    bench_early_cutoff,
    bench_overhead_vs_batch,
    bench_scaling,
    bench_collection_insert,
    bench_collection_delete,
    bench_collection_pipeline_depth,
);
criterion_main!(benches);
