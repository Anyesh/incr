//! The stall a memo checkpoint costs its caller when a few entries are
//! dirty in a large memo: the checkpoint encodes only the dirty entries
//! but merges and rewrites the whole generation.
//!
//! A memo on `Local` of `INCR_MEMO_STALL_BYTES` of entry frames (default
//! 1 GiB), each value `INCR_MEMO_STALL_VALUE` bytes (default 1000), is
//! filled and checkpointed once, untimed. Each timed checkpoint first
//! recomputes `INCR_MEMO_STALL_DIRTY` entries (default 1000), untimed,
//! then reports one of the steps from `CheckpointStats::times`:
//! `snapshot` (dirty entries found and encoded), `merge` (base read and
//! decoded), `write` (frames, eviction and the durable write), and
//! `total`, the whole `Memo::checkpoint` call. Groups are named by the
//! backend, `file` and, with the `sqlite` feature, `sqlite`, each in a
//! fresh directory under the workspace's `target/`.
//!
//! A small run that checks the bench works:
//! `INCR_MEMO_STALL_BYTES=4194304 cargo bench -p incr-persist --bench memo_stall -- --quick`.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{bench_dir, BenchCells};
use criterion::{criterion_group, criterion_main, Criterion};
use incr_persist::memo::{
    BuildId, CheckpointTimes, FileMemo, Memo, MemoBackend, MemoBuilder, MemoOptions, MemoTable,
};
use incr_persist::{Input, Local, MemoryBackend, Options, Store};
use tempfile::TempDir;

const BUILD: u128 = 0x57a11;

fn env_or(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a whole number, got {v:?}")),
        Err(_) => default,
    }
}

struct Sizes {
    entries: u32,
    dirty: u32,
    value: usize,
}

impl Sizes {
    fn from_env() -> Self {
        let bytes = env_or("INCR_MEMO_STALL_BYTES", 1 << 30);
        let value = env_or("INCR_MEMO_STALL_VALUE", 1000);
        let dirty = env_or("INCR_MEMO_STALL_DIRTY", 1000);
        let entries = u32::try_from((bytes / value.max(1)).max(1)).expect("under 2^32 entries");
        Sizes {
            entries,
            dirty: u32::try_from(dirty).unwrap().min(entries),
            value: usize::try_from(value).unwrap(),
        }
    }
}

/// `blob(k)` is `value` bytes; the first `dirty` keys read `epoch`, so
/// moving it recomputes exactly those.
struct Filled<C: BenchCells> {
    store: Arc<Store<C, MemoryBackend>>,
    memo: Memo<C, MemoryBackend>,
    epoch: Input<u64>,
    blob: MemoTable<u32, Vec<u8>>,
    sizes: Sizes,
    /// Last, so the directory goes after the memo that writes into it.
    _dir: TempDir,
}

impl<C: BenchCells> Filled<C> {
    fn open(backend: impl FnOnce(&TempDir) -> Box<dyn MemoBackendBox<C>>) -> Self {
        let sizes = Sizes::from_env();
        let dir = bench_dir();
        let store = Arc::new(Store::open(MemoryBackend::new(), Options::default()).unwrap());
        let epoch = store.input("epoch", 0_u64).unwrap();
        let eh = epoch.handle();
        let (dirty, value) = (sizes.dirty, sizes.value);
        let mut builder =
            MemoBuilder::new(MemoOptions::new(BuildId::Fixed(BUILD)).max_bytes(u64::MAX));
        let blob = builder.table::<u32, Vec<u8>>("blob", 1);
        builder.define(blob, move |ctx, &k| {
            let seed = if k < dirty { ctx.rt().get(eh) } else { 0 };
            let mut out = vec![0_u8; value];
            for (i, b) in out.iter_mut().enumerate() {
                *b = (u64::from(k) ^ seed ^ i as u64) as u8;
            }
            out
        });
        let memo = backend(&dir).open(builder, &store);
        let filled = Filled {
            store,
            memo,
            epoch,
            blob,
            sizes,
            _dir: dir,
        };
        for k in 0..filled.sizes.entries {
            filled.read(k);
        }
        filled.memo.checkpoint().unwrap();
        let stats = filled.memo.stats();
        eprintln!(
            "memo_stall: {} entries, {} bytes, {} dirty per checkpoint",
            stats.entries, stats.bytes, filled.sizes.dirty
        );
        filled
    }

    fn read(&self, k: u32) {
        let q = self.memo.get_or_insert(self.blob, k);
        criterion::black_box(self.memo.runtime().get(q).len());
    }

    /// Dirties the first `dirty` entries, then checkpoints.
    fn checkpoint(&self) -> (CheckpointTimes, Duration) {
        let next = self.store.runtime().get(self.epoch.handle()) + 1;
        self.store.set(self.epoch, next).unwrap();
        for k in 0..self.sizes.dirty {
            self.read(k);
        }
        let started = Instant::now();
        let stats = self.memo.checkpoint().unwrap();
        let total = started.elapsed();
        assert_eq!(stats.evicted, 0);
        assert_eq!(stats.written as u32, self.sizes.entries);
        (stats.times, total)
    }
}

/// Opens a memo over one backend type, erased so groups share the code.
trait MemoBackendBox<C: BenchCells> {
    fn open(
        self: Box<Self>,
        builder: MemoBuilder<C>,
        store: &Arc<Store<C, MemoryBackend>>,
    ) -> Memo<C, MemoryBackend>;
}

impl<C: BenchCells, M: MemoBackend + 'static> MemoBackendBox<C> for M {
    fn open(
        self: Box<Self>,
        builder: MemoBuilder<C>,
        store: &Arc<Store<C, MemoryBackend>>,
    ) -> Memo<C, MemoryBackend> {
        builder.open(store, *self).unwrap()
    }
}

type Step = fn(&CheckpointTimes, Duration) -> Duration;

const STEPS: [(&str, Step); 4] = [
    ("snapshot", |t, _| t.snapshot),
    ("merge", |t, _| t.merge),
    ("write", |t, _| t.write),
    ("total", |_, total| total),
];

fn stall<C: BenchCells>(
    c: &mut Criterion,
    name: &str,
    backend: fn(&TempDir) -> Box<dyn MemoBackendBox<C>>,
) {
    let filled = Filled::<C>::open(backend);
    let mut group = c.benchmark_group(format!("memo_stall_{name}{}", C::SUFFIX));
    group.sample_size(10);
    for (step, pick) in STEPS {
        group.bench_function(step, |b| {
            b.iter_custom(|iters| {
                (0..iters)
                    .map(|_| {
                        let (times, total) = filled.checkpoint();
                        pick(&times, total)
                    })
                    .sum()
            })
        });
    }
    group.finish();
}

fn file(c: &mut Criterion) {
    stall::<Local>(c, "file", |dir| {
        Box::new(FileMemo::open(dir.path()).unwrap())
    });
}

#[cfg(feature = "sqlite")]
fn sqlite(c: &mut Criterion) {
    stall::<Local>(c, "sqlite", |dir| {
        Box::new(incr_persist::memo::SqliteMemo::open(dir.path().join("memo.db")).unwrap())
    });
}

#[cfg(feature = "sqlite")]
criterion_group!(benches, file, sqlite);
#[cfg(not(feature = "sqlite"))]
criterion_group!(benches, file);
criterion_main!(benches);
