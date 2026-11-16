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
//! With `INCR_MEMO_STALL_CYCLE` set, the timed groups are skipped and each
//! backend instead runs one chain cycle, `MAX_CHAIN_DEPTH` delta
//! checkpoints and the compaction after them, printing per checkpoint its
//! kind, step times and `VmRSS`, then the delta, compaction and amortized
//! times (the cycle's total over its checkpoints) and `VmHWM`.
//! `INCR_MEMO_STALL_CHECKPOINTS` sets the checkpoints run (default one
//! cycle). A memo of 600000 entries with 6000 dirty:
//! `INCR_MEMO_STALL_BYTES=600000000 INCR_MEMO_STALL_DIRTY=6000 INCR_MEMO_STALL_CYCLE=1
//! cargo bench -p incr-persist --bench memo_stall`.
//!
//! A small run that checks the bench works:
//! `INCR_MEMO_STALL_BYTES=4194304 cargo bench -p incr-persist --bench memo_stall -- --quick`.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{bench_dir, BenchCells};
use criterion::{criterion_group, criterion_main, Criterion};
use incr_persist::memo::{
    BuildId, CheckpointStats, CheckpointTimes, FileMemo, Memo, MemoBackend, MemoBuilder,
    MemoOptions, MemoTable,
};
use incr_persist::{Input, Local, MemoryBackend, Options, Store};
use tempfile::TempDir;

const BUILD: u128 = 0x57a11;

fn proc_status_kib(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with(field))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn proc_read_mib() -> u64 {
    let io = std::fs::read_to_string("/proc/self/io").unwrap_or_default();
    let line = io.lines().find(|l| l.starts_with("rchar:"));
    line.and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        .unwrap_or(0)
        >> 20
}

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
        let (stats, total) = self.checkpoint_stats();
        (stats.times, total)
    }

    fn checkpoint_stats(&self) -> (CheckpointStats, Duration) {
        let next = self.store.runtime().get(self.epoch.handle()) + 1;
        self.store.set(self.epoch, next).unwrap();
        for k in 0..self.sizes.dirty {
            self.read(k);
        }
        let started = Instant::now();
        let stats = self.memo.checkpoint().unwrap();
        let total = started.elapsed();
        assert_eq!(stats.evicted, 0);
        shape::assert_written(&stats, self.sizes.entries, self.sizes.dirty);
        (stats, total)
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
    if std::env::var_os("INCR_MEMO_STALL_CYCLE").is_some() {
        return;
    }
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

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn cycle<C: BenchCells>(name: &str, backend: fn(&TempDir) -> Box<dyn MemoBackendBox<C>>) {
    if std::env::var_os("INCR_MEMO_STALL_CYCLE").is_none() {
        return;
    }
    let checkpoints = env_or("INCR_MEMO_STALL_CHECKPOINTS", shape::CYCLE as u64);
    let filled = Filled::<C>::open(backend);
    let rss_before = proc_status_kib("VmRSS:").unwrap_or(0);
    eprintln!(
        "memo_stall_{name}{}: VmRSS before {rss_before} KiB",
        C::SUFFIX
    );
    let (mut deltas, mut fulls, mut all) = (Vec::new(), Vec::new(), Duration::ZERO);
    for n in 1..=checkpoints {
        let (stats, total) = filled.checkpoint_stats();
        let kind = shape::kind(&stats);
        eprintln!(
            "memo_stall_{name}{}: #{n} {kind} total {:.1} ms (snapshot {:.1}, merge {:.1}, write {:.1}) VmRSS {} KiB, read {} MiB",
            C::SUFFIX,
            ms(total),
            ms(stats.times.snapshot),
            ms(stats.times.merge),
            ms(stats.times.write),
            proc_status_kib("VmRSS:").unwrap_or(0),
            proc_read_mib(),
        );
        all += total;
        if shape::is_delta(&stats) {
            deltas.push(total);
        } else {
            fulls.push(total);
        }
    }
    let each = |v: &[Duration]| {
        if v.is_empty() {
            "n/a".to_string()
        } else {
            format!("{:.1} ms", ms(median(v.to_vec())))
        }
    };
    eprintln!(
        "memo_stall_{name}{}: delta median {}, full median {}, amortized {:.1} ms over {checkpoints} checkpoints, VmHWM {} KiB, VmRSS after {} KiB",
        C::SUFFIX,
        each(&deltas),
        each(&fulls),
        ms(all) / checkpoints as f64,
        proc_status_kib("VmHWM:").unwrap_or(0),
        proc_status_kib("VmRSS:").unwrap_or(0),
    );
}

fn file(c: &mut Criterion) {
    let open: fn(&TempDir) -> Box<dyn MemoBackendBox<Local>> =
        |dir| Box::new(FileMemo::open(dir.path()).unwrap());
    stall::<Local>(c, "file", open);
    cycle::<Local>("file", open);
}

#[cfg(feature = "sqlite")]
fn sqlite(c: &mut Criterion) {
    let open: fn(&TempDir) -> Box<dyn MemoBackendBox<Local>> =
        |dir| Box::new(incr_persist::memo::SqliteMemo::open(dir.path().join("memo.db")).unwrap());
    stall::<Local>(c, "sqlite", open);
    cycle::<Local>("sqlite", open);
}

/// What a checkpoint's stats say about its shape; the one place the bench
/// reads the delta fields.
mod shape {
    use incr_persist::memo::{CheckpointStats, MAX_CHAIN_DEPTH};

    pub const CYCLE: usize = MAX_CHAIN_DEPTH + 1;

    pub fn is_delta(stats: &CheckpointStats) -> bool {
        stats.delta
    }

    pub fn kind(stats: &CheckpointStats) -> &'static str {
        match (stats.delta, stats.compacted) {
            (true, _) => "delta",
            (false, true) => "compaction",
            (false, false) => "full",
        }
    }

    pub fn assert_written(stats: &CheckpointStats, entries: u32, dirty: u32) {
        if stats.delta {
            assert_eq!(stats.fresh_written as u32, dirty);
        } else {
            assert_eq!(stats.written as u32, entries);
        }
    }
}

#[cfg(feature = "sqlite")]
criterion_group!(benches, file, sqlite);
#[cfg(not(feature = "sqlite"))]
criterion_group!(benches, file);
criterion_main!(benches);
