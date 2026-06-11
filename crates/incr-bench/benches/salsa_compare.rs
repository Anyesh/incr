//! Reproducible comparison against Salsa on the README's workloads.
//!
//! Same shapes as incr-core's `chain.rs` bench: per-iteration set + read
//! on a chain (per-node propagation), a diamond (sharing), and an
//! early-cutoff pair. The Salsa side models the chain as a recursive
//! tracked function memoized per depth, which is the idiomatic Salsa
//! encoding of a dynamic chain (one memo per level, automatic argument
//! interning); diamond and cutoff are plain tracked functions. Salsa
//! mutation requires `&mut db`, which is itself the design difference
//! the incr-concurrent column exists to highlight.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use incr_core::{Incr, Local, Runtime, Shared};
use salsa::Setter;

#[salsa::input]
struct BenchInput {
    value: i64,
}

#[salsa::tracked]
fn chain_level(db: &dyn salsa::Database, input: BenchInput, n: u32) -> i64 {
    if n == 0 {
        input.value(db).wrapping_add(1)
    } else {
        chain_level(db, input, n - 1).wrapping_add(1)
    }
}

#[salsa::tracked]
fn diamond_a(db: &dyn salsa::Database, input: BenchInput) -> i64 {
    input.value(db).wrapping_add(10)
}

#[salsa::tracked]
fn diamond_b(db: &dyn salsa::Database, input: BenchInput) -> i64 {
    input.value(db).wrapping_add(100)
}

#[salsa::tracked]
fn diamond_out(db: &dyn salsa::Database, input: BenchInput) -> i64 {
    diamond_a(db, input).wrapping_add(diamond_b(db, input))
}

#[salsa::tracked]
fn clamped(db: &dyn salsa::Database, input: BenchInput) -> i64 {
    input.value(db).min(100)
}

#[salsa::tracked]
fn after_clamp(db: &dyn salsa::Database, input: BenchInput) -> i64 {
    clamped(db, input).wrapping_add(1)
}

fn build_chain_incr<C: incr_core::Cells>(n: usize) -> (Runtime<C>, Incr<i64>, Incr<i64>) {
    let rt: Runtime<C> = Runtime::new();
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
    let mut group = c.benchmark_group("chain");
    for size in [4_usize, 10, 100] {
        group.bench_with_input(BenchmarkId::new("incr_local", size), &size, |b, &size| {
            let (rt, input, output) = build_chain_incr::<Local>(size);
            let mut val = 1_i64;
            b.iter(|| {
                val = val.wrapping_add(1);
                rt.set(input, val);
                black_box(rt.get(output));
            });
        });
        group.bench_with_input(BenchmarkId::new("incr_shared", size), &size, |b, &size| {
            let (rt, input, output) = build_chain_incr::<Shared>(size);
            let mut val = 1_i64;
            b.iter(|| {
                val = val.wrapping_add(1);
                rt.set(input, val);
                black_box(rt.get(output));
            });
        });
        group.bench_with_input(BenchmarkId::new("salsa", size), &size, |b, &size| {
            let mut db = salsa::DatabaseImpl::new();
            let input = BenchInput::new(&db, 1);
            let _ = chain_level(&db, input, size as u32 - 1);
            let mut val = 1_i64;
            b.iter(|| {
                val = val.wrapping_add(1);
                input.set_value(&mut db).to(val);
                black_box(chain_level(&db, input, size as u32 - 1));
            });
        });
    }
    group.finish();
}

fn bench_diamond(c: &mut Criterion) {
    {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(1_i64);
        let a = rt.create_query(move |rt| rt.get(input).wrapping_add(10));
        let b = rt.create_query(move |rt| rt.get(input).wrapping_add(100));
        let out = rt.create_query(move |rt| rt.get(a).wrapping_add(rt.get(b)));
        let _ = rt.get(out);
        c.bench_function("diamond/incr_local", |bencher| {
            let mut val = 1_i64;
            bencher.iter(|| {
                val = val.wrapping_add(1);
                rt.set(input, val);
                black_box(rt.get(out));
            });
        });
    }
    {
        let rt: Runtime<Shared> = Runtime::new();
        let input = rt.create_input(1_i64);
        let a = rt.create_query(move |rt| rt.get(input).wrapping_add(10));
        let b = rt.create_query(move |rt| rt.get(input).wrapping_add(100));
        let out = rt.create_query(move |rt| rt.get(a).wrapping_add(rt.get(b)));
        let _ = rt.get(out);
        c.bench_function("diamond/incr_shared", |bencher| {
            let mut val = 1_i64;
            bencher.iter(|| {
                val = val.wrapping_add(1);
                rt.set(input, val);
                black_box(rt.get(out));
            });
        });
    }
    {
        let mut db = salsa::DatabaseImpl::new();
        let input = BenchInput::new(&db, 1);
        let _ = diamond_out(&db, input);
        c.bench_function("diamond/salsa", |bencher| {
            let mut val = 1_i64;
            bencher.iter(|| {
                val = val.wrapping_add(1);
                input.set_value(&mut db).to(val);
                black_box(diamond_out(&db, input));
            });
        });
    }
}

fn bench_early_cutoff(c: &mut Criterion) {
    {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(200_i64);
        let clamp = rt.create_query(move |rt| rt.get(input).min(100));
        let after = rt.create_query(move |rt| rt.get(clamp).wrapping_add(1));
        let _ = rt.get(after);
        c.bench_function("early_cutoff/incr_local", |bencher| {
            let mut val = 200_i64;
            bencher.iter(|| {
                val = val.wrapping_add(1);
                rt.set(input, val);
                black_box(rt.get(after));
            });
        });
    }
    {
        let rt: Runtime<Shared> = Runtime::new();
        let input = rt.create_input(200_i64);
        let clamp = rt.create_query(move |rt| rt.get(input).min(100));
        let after = rt.create_query(move |rt| rt.get(clamp).wrapping_add(1));
        let _ = rt.get(after);
        c.bench_function("early_cutoff/incr_shared", |bencher| {
            let mut val = 200_i64;
            bencher.iter(|| {
                val = val.wrapping_add(1);
                rt.set(input, val);
                black_box(rt.get(after));
            });
        });
    }
    {
        let mut db = salsa::DatabaseImpl::new();
        let input = BenchInput::new(&db, 200);
        let _ = after_clamp(&db, input);
        c.bench_function("early_cutoff/salsa", |bencher| {
            let mut val = 200_i64;
            bencher.iter(|| {
                val = val.wrapping_add(1);
                input.set_value(&mut db).to(val);
                black_box(after_clamp(&db, input));
            });
        });
    }
}

criterion_group!(
    salsa_compare,
    bench_chain,
    bench_diamond,
    bench_early_cutoff
);
criterion_main!(salsa_compare);
