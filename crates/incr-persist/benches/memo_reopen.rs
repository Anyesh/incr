//! R10: reopening a memo whose entries depend on keyed rows.
//!
//! A store on `MemoryBackend` holds `INCR_REOPEN_ENTRIES` rows (default
//! 240,000) in a persisted keyed collection. A memo table `price` has one
//! entry per key, each reading its row through `get_key`, so each is stored
//! with a `StoredDep::Row`. The log is checkpointed and the memo
//! checkpointed once, untimed. Each timed iteration reopens from those
//! backends and reports one step: `store_open` (the log's snapshot loaded),
//! `memo_open` (the generation loaded and the row dependencies mapped),
//! `first_reads` (every entry read once, verified against its row; the
//! bench asserts no entry ran), and `total`.
//!
//! A small run that checks the bench works:
//! `INCR_REOPEN_ENTRIES=5000 cargo bench -p incr-persist --bench memo_reopen -- --quick`.

// The shared module also holds `bench_dir`, which this bench has no use for.
#[allow(dead_code)]
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::BenchCells;
use criterion::{criterion_group, criterion_main, Criterion};
use incr_persist::memo::{BuildId, Memo, MemoBuilder, MemoOptions, MemoTable, MemoryMemo};
use incr_persist::{Local, MemoryBackend, Options, PersistedKeyed, Shared, Store};

const BUILD: u128 = 0x7e0be;

type Kc<C> = PersistedKeyed<u32, String, C>;

fn entries() -> u32 {
    match std::env::var("INCR_REOPEN_ENTRIES") {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("INCR_REOPEN_ENTRIES must be a whole number, got {v:?}")),
        Err(_) => 240_000,
    }
}

/// Lets a query closure, which must be `Send + Sync`, capture a keyed
/// collection of a `Local` store.
struct Captured<T>(T);

// SAFETY: a `Local` store and its handles stay on the one thread that opened
// them; `Shared` handles are `Send + Sync` anyway.
unsafe impl<T> Send for Captured<T> {}
// SAFETY: see `Send`.
unsafe impl<T> Sync for Captured<T> {}

impl<T> Captured<T> {
    fn get(&self) -> &T {
        &self.0
    }
}

struct Reopened<C: BenchCells> {
    memo: Memo<C, MemoryBackend>,
    price: MemoTable<u32, Option<String>>,
    runs: Arc<AtomicUsize>,
}

struct Steps {
    store_open: Duration,
    memo_open: Duration,
}

fn open_memo<C: BenchCells>(
    store: &Arc<Store<C, MemoryBackend>>,
    memo_backend: &MemoryMemo,
) -> Reopened<C> {
    let kc: Kc<C> = store.keyed_collection("rows").unwrap();
    let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(BUILD)).max_bytes(u64::MAX));
    let price = builder.table::<u32, Option<String>>("price", 1);
    let runs = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&runs);
    let captured = Captured(kc);
    builder.define(price, move |ctx, k| {
        counted.fetch_add(1, Ordering::Relaxed);
        captured.get().get_key(ctx.rt(), k)
    });
    let memo = builder.open(store, memo_backend.clone()).unwrap();
    Reopened { memo, price, runs }
}

impl<C: BenchCells> Reopened<C> {
    fn read_all(&self, n: u32) {
        for k in 0..n {
            let q = self.memo.get_or_insert(self.price, k);
            criterion::black_box(self.memo.runtime().get(q));
        }
    }
}

fn fill<C: BenchCells>(log: &MemoryBackend, memo_backend: &MemoryMemo, n: u32) {
    let started = Instant::now();
    let store = Arc::new(Store::<C, _>::open(log.clone(), Options::default()).unwrap());
    let kc: Kc<C> = store.keyed_collection("rows").unwrap();
    store
        .reconcile(&kc, (0..n).map(|k| (k, format!("row-{k}"))))
        .unwrap();
    let opened = open_memo::<C>(&store, memo_backend);
    opened.read_all(n);
    opened.memo.checkpoint().unwrap();
    store.checkpoint().unwrap();
    eprintln!(
        "memo_reopen: {n} entries with Row deps filled in {:?}",
        started.elapsed()
    );
}

fn reopen_once<C: BenchCells>(
    log: &MemoryBackend,
    memo_backend: &MemoryMemo,
    n: u32,
) -> (Steps, Duration) {
    let started = Instant::now();
    let store = Arc::new(Store::<C, _>::open(log.clone(), Options::default()).unwrap());
    let store_open = started.elapsed();
    let memo_started = Instant::now();
    let opened = open_memo::<C>(&store, memo_backend);
    let memo_open = memo_started.elapsed();
    let reads_started = Instant::now();
    opened.read_all(n);
    let first_reads = reads_started.elapsed();
    assert_eq!(opened.runs.load(Ordering::Relaxed), 0, "an entry ran");
    (
        Steps {
            store_open,
            memo_open,
        },
        first_reads,
    )
}

fn reopen<C: BenchCells>(c: &mut Criterion) {
    let n = entries();
    let (log, memo_backend) = (MemoryBackend::new(), MemoryMemo::new());
    fill::<C>(&log, &memo_backend, n);
    let mut group = c.benchmark_group(format!("memo_reopen{}", C::SUFFIX));
    group.sample_size(10);
    type Pick = fn(&Steps, Duration) -> Duration;
    let picks: [(&str, Pick); 4] = [
        ("store_open", |s, _| s.store_open),
        ("memo_open", |s, _| s.memo_open),
        ("first_reads", |_, reads| reads),
        ("total", |s, reads| s.store_open + s.memo_open + reads),
    ];
    for (name, pick) in picks {
        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                (0..iters)
                    .map(|_| {
                        let (steps, reads) = reopen_once::<C>(&log, &memo_backend, n);
                        pick(&steps, reads)
                    })
                    .sum()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, reopen::<Local>, reopen::<Shared>);
criterion_main!(benches);
