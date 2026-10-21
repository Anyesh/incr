//! Set throughput of a Store on each backend against a plain `rt.set`.
//!
//! Each case writes one `u64` input over and over. Logs live in a fresh
//! directory under the workspace's `target/` (or `INCR_PERSIST_BENCH_DIR`),
//! never under a tmpfs `/tmp`, where fsync would cost nothing. The
//! two-writer SQLite case needs two processes and lives in
//! `examples/sqlite_writers.rs`. Every case runs on Local in the `set`
//! group and on Shared in `set_shared`.

mod common;

use std::time::Duration;

use common::{bench_dir, BenchCells};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput,
};
use incr_core::Runtime;
use incr_persist::{
    Backend, Durability, FileBackend, Local, MemoryBackend, Options, Shared, Store,
};
#[cfg(feature = "sqlite")]
use incr_persist::{SqliteBackend, SqliteDurability, SqliteOptions};
#[cfg(feature = "sqlite")]
use tempfile::TempDir;

const BATCH: u64 = 100;

fn options(durability: Durability) -> Options {
    Options {
        durability,
        ..Options::default()
    }
}

fn set_group<C: BenchCells>(c: &mut Criterion) -> BenchmarkGroup<'_, WallTime> {
    c.benchmark_group(format!("set{}", C::SUFFIX))
}

fn bench_store<C: BenchCells, B: Backend>(
    c: &mut Criterion,
    name: &str,
    open: impl FnOnce() -> Store<C, B>,
) {
    let store = open();
    let input = store.input("x", 0u64).unwrap();
    let mut group = set_group::<C>(c);
    group.throughput(Throughput::Elements(1));
    let mut value = 0u64;
    group.bench_function(name, |b| {
        b.iter(|| {
            value += 1;
            black_box(store.set(input, value).unwrap())
        })
    });
    group.finish();
}

fn plain<C: BenchCells>(c: &mut Criterion) {
    let rt: Runtime<C> = Runtime::new();
    let input = rt.create_input(0u64);
    let mut group = set_group::<C>(c);
    group.throughput(Throughput::Elements(1));
    let mut value = 0u64;
    group.bench_function("runtime", |b| {
        b.iter(|| {
            value += 1;
            rt.set(input, black_box(value));
        })
    });
    group.finish();
}

fn memory<C: BenchCells>(c: &mut Criterion) {
    bench_store::<C, _>(c, "memory", || {
        Store::open(MemoryBackend::new(), options(Durability::Sync)).unwrap()
    });
}

fn file_buffered<C: BenchCells>(c: &mut Criterion) {
    let dir = bench_dir();
    bench_store::<C, _>(c, "file_buffered", || {
        Store::open(
            FileBackend::open(dir.path()).unwrap(),
            options(Durability::Buffered),
        )
        .unwrap()
    });
}

fn file_sync<C: BenchCells>(c: &mut Criterion) {
    let dir = bench_dir();
    bench_store::<C, _>(c, "file_sync", || {
        Store::open(
            FileBackend::open(dir.path()).unwrap(),
            options(Durability::Sync),
        )
        .unwrap()
    });
}

fn file_sync_batched<C: BenchCells>(c: &mut Criterion) {
    let dir = bench_dir();
    let store = Store::<C, _>::open(
        FileBackend::open(dir.path()).unwrap(),
        options(Durability::Sync),
    )
    .unwrap();
    let input = store.input("x", 0u64).unwrap();
    let mut group = set_group::<C>(c);
    group.throughput(Throughput::Elements(BATCH));
    let mut value = 0u64;
    group.bench_function("file_sync_batch100", |b| {
        b.iter(|| {
            let position = store
                .batch(|batch| {
                    for _ in 0..BATCH {
                        value += 1;
                        batch.set(input, value);
                    }
                })
                .unwrap();
            black_box(position)
        })
    });
    group.finish();
}

// SqliteBackend::sync is a no-op, so the store's durability option changes
// nothing for it; what an append survives is SQLite's `synchronous`
// setting, which each case pairs with the matching store option.
#[cfg(feature = "sqlite")]
fn sqlite_store<C: BenchCells>(
    durability: SqliteDurability,
    dir: &TempDir,
) -> Store<C, SqliteBackend> {
    let backend = SqliteBackend::open_with(
        dir.path().join("log.db"),
        SqliteOptions {
            durability,
            ..SqliteOptions::default()
        },
    )
    .unwrap();
    let store_durability = match durability {
        SqliteDurability::Full => Durability::Sync,
        SqliteDurability::Buffered => Durability::Buffered,
    };
    Store::open(backend, options(store_durability)).unwrap()
}

#[cfg(feature = "sqlite")]
fn sqlite_full<C: BenchCells>(c: &mut Criterion) {
    let dir = bench_dir();
    bench_store::<C, _>(c, "sqlite_full", || {
        sqlite_store(SqliteDurability::Full, &dir)
    });
}

#[cfg(feature = "sqlite")]
fn sqlite_buffered<C: BenchCells>(c: &mut Criterion) {
    let dir = bench_dir();
    bench_store::<C, _>(c, "sqlite_buffered", || {
        sqlite_store(SqliteDurability::Buffered, &dir)
    });
}

#[cfg(not(feature = "sqlite"))]
fn sqlite_full<C: BenchCells>(_: &mut Criterion) {}

#[cfg(not(feature = "sqlite"))]
fn sqlite_buffered<C: BenchCells>(_: &mut Criterion) {}

fn cpu_bound() -> Criterion {
    Criterion::default().measurement_time(Duration::from_secs(5))
}

// One fsync per sample iteration on a disk costs milliseconds, so the
// default hundred samples would take minutes per case.
fn fsync_bound() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(15))
}

criterion_group! {
    name = cpu;
    config = cpu_bound();
    targets = plain::<Local>, memory::<Local>, file_buffered::<Local>, sqlite_buffered::<Local>,
        plain::<Shared>, memory::<Shared>, file_buffered::<Shared>, sqlite_buffered::<Shared>
}

criterion_group! {
    name = fsync;
    config = fsync_bound();
    targets = file_sync::<Local>, file_sync_batched::<Local>, sqlite_full::<Local>,
        file_sync::<Shared>, file_sync_batched::<Shared>, sqlite_full::<Shared>
}

criterion_main!(cpu, fsync);
