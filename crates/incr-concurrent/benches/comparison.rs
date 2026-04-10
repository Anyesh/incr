use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use incr_concurrent::Runtime;
use salsa::Setter;

// Workload 1: Linear chain (input → f1 → f2 → ... → fn). Measures per-node
// propagation cost.

fn incr_chain_propagate(
    size: usize,
) -> (
    Runtime,
    incr_concurrent::Incr<i64>,
    incr_concurrent::Incr<i64>,
) {
    let rt = Runtime::new();
    let input = rt.create_input(1_i64);
    let mut prev = input;
    for _ in 0..size {
        let dep = prev;
        prev = rt.create_query(move |rt| rt.get(dep).wrapping_add(1));
    }
    let _ = rt.get(prev);
    (rt, input, prev)
}

fn salsa_chain_propagate(size: usize) -> (salsa::DatabaseImpl, SalsaInput, usize) {
    let db = salsa::DatabaseImpl::new();
    let input = SalsaInput::new(&db, 1_i64);
    // Salsa doesn't support dynamic chains via closures.
    // We measure the overhead of a single tracked function call instead,
    // and multiply conceptually. The real benchmark is the per-query cost.
    (db, input, size)
}

// Salsa types for comparison
#[salsa::input]
struct SalsaInput {
    value: i64,
}

#[salsa::tracked]
fn salsa_add_one(db: &dyn salsa::Database, input: SalsaInput) -> i64 {
    input.value(db).wrapping_add(1)
}

#[salsa::tracked]
fn salsa_chain_2(db: &dyn salsa::Database, input: SalsaInput) -> i64 {
    salsa_add_one(db, input).wrapping_add(1)
}

#[salsa::tracked]
fn salsa_chain_4(db: &dyn salsa::Database, input: SalsaInput) -> i64 {
    let v = salsa_add_one(db, input);
    let v = v.wrapping_add(1);
    let v = v.wrapping_add(1);
    v.wrapping_add(1)
}

// Workload 2: Diamond (input → [A, B] → output). Measures handling of
// shared dependencies.

fn incr_diamond_propagate() -> (
    Runtime,
    incr_concurrent::Incr<i64>,
    incr_concurrent::Incr<i64>,
) {
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
    let output = rt.create_query(move |rt| rt.get(a).wrapping_add(rt.get(b)));
    let _ = rt.get(output);
    (rt, input, output)
}

#[salsa::tracked]
fn salsa_diamond_a(db: &dyn salsa::Database, input: SalsaInput) -> i64 {
    input.value(db).wrapping_add(10)
}

#[salsa::tracked]
fn salsa_diamond_b(db: &dyn salsa::Database, input: SalsaInput) -> i64 {
    input.value(db).wrapping_add(100)
}

#[salsa::tracked]
fn salsa_diamond_out(db: &dyn salsa::Database, input: SalsaInput) -> i64 {
    salsa_diamond_a(db, input).wrapping_add(salsa_diamond_b(db, input))
}

// Workload 3: Early cutoff (input → clamp → downstream). Measures whether
// early cutoff prevents unnecessary work.

#[salsa::tracked]
fn salsa_clamp(db: &dyn salsa::Database, input: SalsaInput) -> i64 {
    input.value(db).min(100)
}

#[salsa::tracked]
fn salsa_after_clamp(db: &dyn salsa::Database, input: SalsaInput) -> i64 {
    salsa_clamp(db, input).wrapping_add(1)
}

// Workload 4: Collection pipeline — insert into filter → map → count
// Batch baseline: compute from scratch each time.

fn batch_collection_insert(elements: &mut std::collections::HashSet<i64>, new_val: i64) -> usize {
    elements.insert(new_val);
    elements
        .iter()
        .filter(|x| *x % 2 == 0)
        .map(|x| x * 2)
        .count()
}

// Benchmarks

fn bench_chain_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_incr_vs_salsa");

    // incr: propagate through chain
    for size in [10, 100] {
        group.bench_with_input(BenchmarkId::new("incr", size), &size, |b, &size| {
            let (rt, input, output) = incr_chain_propagate(size);
            let mut val = 1_i64;
            b.iter(|| {
                val += 1;
                rt.set(input, val);
                black_box(rt.get(output));
            });
        });
    }

    // salsa: single query re-evaluation (the comparable unit of work)
    group.bench_function("salsa_single_query", |b| {
        let mut db = salsa::DatabaseImpl::new();
        let input = SalsaInput::new(&db, 1_i64);
        let _ = salsa_add_one(&db, input);
        let mut val = 1_i64;
        b.iter(|| {
            val += 1;
            input.set_value(&mut db).to(val);
            black_box(salsa_add_one(&db, input));
        });
    });

    // salsa: 2-deep chain
    group.bench_function("salsa_chain_2", |b| {
        let mut db = salsa::DatabaseImpl::new();
        let input = SalsaInput::new(&db, 1_i64);
        let _ = salsa_chain_2(&db, input);
        let mut val = 1_i64;
        b.iter(|| {
            val += 1;
            input.set_value(&mut db).to(val);
            black_box(salsa_chain_2(&db, input));
        });
    });

    // salsa: 4-deep chain
    group.bench_function("salsa_chain_4", |b| {
        let mut db = salsa::DatabaseImpl::new();
        let input = SalsaInput::new(&db, 1_i64);
        let _ = salsa_chain_4(&db, input);
        let mut val = 1_i64;
        b.iter(|| {
            val += 1;
            input.set_value(&mut db).to(val);
            black_box(salsa_chain_4(&db, input));
        });
    });

    group.finish();
}

fn bench_diamond_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("diamond_incr_vs_salsa");

    group.bench_function("incr", |b| {
        let (rt, input, output) = incr_diamond_propagate();
        let mut val = 1_i64;
        b.iter(|| {
            val += 1;
            rt.set(input, val);
            black_box(rt.get(output));
        });
    });

    group.bench_function("salsa", |b| {
        let mut db = salsa::DatabaseImpl::new();
        let input = SalsaInput::new(&db, 1_i64);
        let _ = salsa_diamond_out(&db, input);
        let mut val = 1_i64;
        b.iter(|| {
            val += 1;
            input.set_value(&mut db).to(val);
            black_box(salsa_diamond_out(&db, input));
        });
    });

    group.finish();
}

fn bench_early_cutoff_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("early_cutoff_incr_vs_salsa");

    group.bench_function("incr", |b| {
        let rt = Runtime::new();
        let input = rt.create_input(200_i64);
        let clamped = rt.create_query(move |rt| rt.get(input).min(100));
        let after = rt.create_query(move |rt| rt.get(clamped).wrapping_add(1));
        let _ = rt.get(after);

        let mut val = 200_i64;
        b.iter(|| {
            val += 1;
            rt.set(input, val); // Always > 100, clamp produces 100, early cutoff
            black_box(rt.get(after));
        });
    });

    group.bench_function("salsa", |b| {
        let mut db = salsa::DatabaseImpl::new();
        let input = SalsaInput::new(&db, 200_i64);
        let _ = salsa_after_clamp(&db, input);

        let mut val = 200_i64;
        b.iter(|| {
            val += 1;
            input.set_value(&mut db).to(val);
            black_box(salsa_after_clamp(&db, input));
        });
    });

    group.finish();
}

fn bench_collection_vs_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_incr_vs_batch");

    for size in [1_000, 10_000, 100_000] {
        // incr: delta-based pipeline
        group.bench_with_input(BenchmarkId::new("incr", size), &size, |b, &size| {
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
        });

        // batch: recompute from scratch
        group.bench_with_input(BenchmarkId::new("batch", size), &size, |b, &size| {
            let mut elements = std::collections::HashSet::new();
            for i in 0..size {
                elements.insert(i);
            }

            let mut next = size;
            b.iter(|| {
                let result = batch_collection_insert(&mut elements, next);
                next += 1;
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    comparison_benches,
    bench_chain_comparison,
    bench_diamond_comparison,
    bench_early_cutoff_comparison,
    bench_collection_vs_batch,
);
criterion_main!(comparison_benches);
