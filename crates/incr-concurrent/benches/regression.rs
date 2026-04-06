use criterion::{black_box, criterion_group, criterion_main, Criterion};
use incr_concurrent::Runtime;

fn hot_read_input(c: &mut Criterion) {
    let rt = Runtime::new();
    let input = rt.create_input(42_u64);
    let _ = rt.get(input);

    c.bench_function("hot_read_input", |b| b.iter(|| black_box(rt.get(input))));
}

fn hot_read_query(c: &mut Criterion) {
    let rt = Runtime::new();
    let input = rt.create_input(42_u64);
    let query = rt.create_query(move |rt| rt.get(input) * 2);
    let _ = rt.get(query);

    c.bench_function("hot_read_query", |b| b.iter(|| black_box(rt.get(query))));
}

fn set_input_no_deps(c: &mut Criterion) {
    let rt = Runtime::new();
    let input = rt.create_input(0_u64);

    c.bench_function("set_input_no_deps", |b| {
        let mut val = 0u64;
        b.iter(|| {
            val += 1;
            rt.set(input, val);
        })
    });
}

fn propagate_chain_100(c: &mut Criterion) {
    let rt = Runtime::new();
    let input = rt.create_input(0_u64);
    let mut prev = input;
    for _ in 0..100 {
        let dep = prev;
        prev = rt.create_query(move |rt| rt.get(dep) + 1);
    }
    let tail = prev;
    let _ = rt.get(tail);

    c.bench_function("propagate_chain_100", |b| {
        let mut val = 0u64;
        b.iter(|| {
            val += 1;
            rt.set(input, val);
            black_box(rt.get(tail))
        })
    });
}

fn collection_pipeline(c: &mut Criterion) {
    let rt = Runtime::new();
    let col = rt.create_collection::<i64>();
    let evens = col.filter(&rt, |x| x % 2 == 0);
    let doubled = evens.map(&rt, |x| x * 2);
    let sum = doubled.reduce(&rt, |elems| -> i64 { elems.iter().sum() });

    for i in 0..50 {
        col.insert(&rt, i);
    }
    let _ = rt.get(sum);

    c.bench_function("collection_pipeline", |b| {
        let mut next = 50i64;
        b.iter(|| {
            col.insert(&rt, next);
            next += 1;
            black_box(rt.get(sum))
        })
    });
}

criterion_group!(
    benches,
    hot_read_input,
    hot_read_query,
    set_input_no_deps,
    propagate_chain_100,
    collection_pipeline
);
criterion_main!(benches);
