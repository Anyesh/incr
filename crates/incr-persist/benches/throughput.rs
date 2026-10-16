//! Set throughput of a Store on each backend against a plain `rt.set`.
//!
//! Each case writes one `u64` input over and over. Logs live in a fresh
//! directory under the workspace's `target/` (or `INCR_PERSIST_BENCH_DIR`),
//! never under a tmpfs `/tmp`, where fsync would cost nothing. The
//! two-writer SQLite case needs two processes and lives in
//! `examples/sqlite_writers.rs`.

use std::path::PathBuf;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use incr_core::Runtime;
use incr_persist::{Backend, Durability, FileBackend, Local, MemoryBackend, Options, Store};
#[cfg(feature = "sqlite")]
use incr_persist::{SqliteBackend, SqliteDurability, SqliteOptions};
use tempfile::TempDir;

const BATCH: u64 = 100;

fn bench_dir() -> TempDir {
    let base = std::env::var_os("INCR_PERSIST_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/incr-persist-bench")
        });
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(base).unwrap()
}

fn options(durability: Durability) -> Options {
    Options {
        durability,
        ..Options::default()
    }
}

fn bench_store<B: Backend>(c: &mut Criterion, name: &str, open: impl FnOnce() -> Store<Local, B>) {
    let store = open();
    let input = store.input("x", 0u64).unwrap();
    let mut group = c.benchmark_group("set");
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

fn plain(c: &mut Criterion) {
    let rt: Runtime<Local> = Runtime::new();
    let input = rt.create_input(0u64);
    let mut group = c.benchmark_group("set");
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

fn memory(c: &mut Criterion) {
    bench_store(c, "memory", || {
        Store::open(MemoryBackend::new(), options(Durability::Sync)).unwrap()
    });
}

fn file_buffered(c: &mut Criterion) {
    let dir = bench_dir();
    bench_store(c, "file_buffered", || {
        Store::open(
            FileBackend::open(dir.path()).unwrap(),
            options(Durability::Buffered),
        )
        .unwrap()
    });
}

fn file_sync(c: &mut Criterion) {
    let dir = bench_dir();
    bench_store(c, "file_sync", || {
        Store::open(
            FileBackend::open(dir.path()).unwrap(),
            options(Durability::Sync),
        )
        .unwrap()
    });
}

fn file_sync_batched(c: &mut Criterion) {
    let dir = bench_dir();
    let store = Store::<Local, _>::open(
        FileBackend::open(dir.path()).unwrap(),
        options(Durability::Sync),
    )
    .unwrap();
    let input = store.input("x", 0u64).unwrap();
    let mut group = c.benchmark_group("set");
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
fn sqlite_store(durability: SqliteDurability, dir: &TempDir) -> Store<Local, SqliteBackend> {
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
fn sqlite_full(c: &mut Criterion) {
    let dir = bench_dir();
    bench_store(c, "sqlite_full", || {
        sqlite_store(SqliteDurability::Full, &dir)
    });
}

#[cfg(feature = "sqlite")]
fn sqlite_buffered(c: &mut Criterion) {
    let dir = bench_dir();
    bench_store(c, "sqlite_buffered", || {
        sqlite_store(SqliteDurability::Buffered, &dir)
    });
}

#[cfg(not(feature = "sqlite"))]
fn sqlite_full(_: &mut Criterion) {}

#[cfg(not(feature = "sqlite"))]
fn sqlite_buffered(_: &mut Criterion) {}

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
    targets = plain, memory, file_buffered, sqlite_buffered
}

criterion_group! {
    name = fsync;
    config = fsync_bound();
    targets = file_sync, file_sync_batched, sqlite_full
}

criterion_main!(cpu, fsync);
