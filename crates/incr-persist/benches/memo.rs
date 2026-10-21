//! Restoring memo entries after a restart against running them cold.
//!
//! Every case is a fresh `Store` and memo over a `FileMemo` in a directory
//! under the workspace's `target/`, as a restarted process would open them; restores read a generation an earlier checkpoint wrote, with
//! the page cache warm. Opening the memo is untimed except in `open`.
//!
//! - `chain`: 100k entries, each reading the one below it and entry 0 an
//!   input. Cold reads ascend, because memo-table queries can't be
//!   replayable and a cold read of the top would recurse 100k levels. A
//!   restore reads only the top and restores the chain under it.
//! - `wide`: 100k independent entries over one input, each read once.
//! - `sweep`: 1000 independent entries whose closure spins for 100 ns to
//!   1 ms; where `restore` stops being faster than `cold` per entry is the
//!   break-even closure cost. `spin` times the closure's spin alone.
//! - `index`: a 100k-item `Generation` alone: `build` indexes it as a
//!   load does, and `find` looks up every item once in key order, which
//!   is hash order scattered over the index.
//!
//! `chain`, `wide` and `sweep` run on Local and, in groups named with a
//! `_shared` suffix, on Shared; `index` uses no runtime.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{bench_dir, BenchCells};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkGroup, Criterion, Throughput,
};
use incr_core::Incr;
use incr_persist::memo::{
    key_hash, BuildId, FileMemo, GenHeader, GenId, Generation, IndexItem, Memo, MemoBuilder,
    MemoOptions, MemoTable,
};
use incr_persist::{Local, MemoryBackend, Options, Shared, Store};
use tempfile::TempDir;

const BIG: u32 = 100_000;
const SWEEP: u32 = 1_000;
const BUILD: u128 = 0xbe9c4;

type Define<C> = dyn Fn(&mut MemoBuilder<C>, Incr<u64>) -> MemoTable<u32, u64>;

/// One shape: how its table is defined and which keys a cold run and a
/// restore read.
struct Shape<C: BenchCells> {
    define: Box<Define<C>>,
    keys: u32,
    restore_reads_top_only: bool,
}

struct Opened<C: BenchCells> {
    memo: Memo<C, MemoryBackend>,
    table: MemoTable<u32, u64>,
}

impl<C: BenchCells> Opened<C> {
    fn read(&self, key: u32) -> u64 {
        let q = self.memo.get_or_insert(self.table, key);
        self.memo.runtime().get(q)
    }

    fn read_ascending(&self, keys: u32) {
        for k in 0..keys {
            black_box(self.read(k));
        }
    }
}

impl<C: BenchCells> Shape<C> {
    fn chain() -> Self {
        Shape {
            define: Box::new(|builder, x| {
                let chain = builder.table::<u32, u64>("chain", 1);
                builder.define(chain, move |ctx, &k| {
                    if k == 0 {
                        ctx.rt().get(x)
                    } else {
                        ctx.get(chain, k - 1) + 1
                    }
                });
                chain
            }),
            keys: BIG,
            restore_reads_top_only: true,
        }
    }

    fn wide(keys: u32, cost: Duration) -> Self {
        Shape {
            define: Box::new(move |builder, x| {
                let wide = builder.table::<u32, u64>("wide", 1);
                builder.define(wide, move |ctx, &k| {
                    if !cost.is_zero() {
                        spin(cost);
                    }
                    ctx.rt().get(x).wrapping_mul(u64::from(k))
                });
                wide
            }),
            keys,
            restore_reads_top_only: false,
        }
    }

    /// The log is shared by every open of one shape, so each restart
    /// replays the same input.
    // `MemoBuilder::open` takes the store as an `Arc` whatever its cells.
    #[allow(clippy::arc_with_non_send_sync)]
    fn open(&self, log: &MemoryBackend, dir: &Path) -> Opened<C> {
        let store = Arc::new(Store::<C, _>::open(log.clone(), Options::default()).unwrap());
        let x = store.input("x", 3_u64).unwrap().handle();
        let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(BUILD)));
        let table = (self.define)(&mut builder, x);
        let memo = builder.open(&store, FileMemo::open(dir).unwrap()).unwrap();
        Opened { memo, table }
    }

    fn cold(&self, opened: &Opened<C>) {
        opened.read_ascending(self.keys);
    }

    fn restore(&self, opened: &Opened<C>) {
        if self.restore_reads_top_only {
            black_box(opened.read(self.keys - 1));
        } else {
            opened.read_ascending(self.keys);
        }
    }

    /// A directory holding a checkpoint of every key, checked to restore
    /// whole: a bench that silently ran cold would measure the wrong thing.
    fn checkpointed(&self, log: &MemoryBackend) -> TempDir {
        let dir = bench_dir();
        {
            let opened = self.open(log, dir.path());
            self.cold(&opened);
            let written = opened.memo.checkpoint().unwrap().written;
            assert_eq!(written, self.keys as usize);
        }
        let opened = self.open(log, dir.path());
        self.restore(&opened);
        let stats = opened.memo.stats();
        assert_eq!((stats.reused, stats.cold), (u64::from(self.keys), 0));
        dir
    }
}

fn spin(d: Duration) {
    let start = Instant::now();
    while start.elapsed() < d {
        std::hint::spin_loop();
    }
}

fn bench_cold<C: BenchCells>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    id: &str,
    shape: &Shape<C>,
    log: &MemoryBackend,
) {
    let empty = bench_dir();
    group.bench_function(id, |b| {
        b.iter_batched(
            || shape.open(log, empty.path()),
            |opened| {
                shape.cold(&opened);
                opened
            },
            BatchSize::PerIteration,
        )
    });
}

fn bench_restore<C: BenchCells>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    id: &str,
    shape: &Shape<C>,
    log: &MemoryBackend,
    dir: &Path,
) {
    group.bench_function(id, |b| {
        b.iter_batched(
            || shape.open(log, dir),
            |opened| {
                shape.restore(&opened);
                opened
            },
            BatchSize::PerIteration,
        )
    });
}

/// Cold, restore, open and checkpoint for one 100k-entry shape.
fn big<C: BenchCells>(c: &mut Criterion, name: &str, shape: Shape<C>) {
    let log = MemoryBackend::new();
    let stored = shape.checkpointed(&log);
    let mut group = c.benchmark_group(format!("{name}{}", C::SUFFIX));
    group.sample_size(10);
    group.throughput(Throughput::Elements(u64::from(shape.keys)));
    bench_cold(&mut group, "cold", &shape, &log);
    bench_restore(&mut group, "restore", &shape, &log, stored.path());
    group.bench_function("open", |b| {
        b.iter_batched(
            || (),
            |()| shape.open(&log, stored.path()),
            BatchSize::PerIteration,
        )
    });
    group.bench_function("checkpoint", |b| {
        b.iter_batched(
            || {
                let dir = bench_dir();
                let opened = shape.open(&log, dir.path());
                shape.cold(&opened);
                (dir, opened)
            },
            |(dir, opened)| {
                black_box(opened.memo.checkpoint().unwrap());
                (dir, opened)
            },
            BatchSize::PerIteration,
        )
    });
    group.finish();
}

fn chain<C: BenchCells>(c: &mut Criterion) {
    big(c, "memo_chain_100k", Shape::<C>::chain());
}

fn wide<C: BenchCells>(c: &mut Criterion) {
    big(c, "memo_wide_100k", Shape::<C>::wide(BIG, Duration::ZERO));
}

const COSTS: [(&str, Duration); 9] = [
    ("100ns", Duration::from_nanos(100)),
    ("300ns", Duration::from_nanos(300)),
    ("1us", Duration::from_micros(1)),
    ("3us", Duration::from_micros(3)),
    ("10us", Duration::from_micros(10)),
    ("30us", Duration::from_micros(30)),
    ("100us", Duration::from_micros(100)),
    ("300us", Duration::from_micros(300)),
    ("1ms", Duration::from_millis(1)),
];

fn sweep<C: BenchCells>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("memo_sweep{}", C::SUFFIX));
    group.sample_size(10);
    for (label, cost) in COSTS {
        group.throughput(Throughput::Elements(1));
        group.bench_function(format!("spin/{label}"), |b| b.iter(|| spin(cost)));
        let shape = Shape::<C>::wide(SWEEP, cost);
        let log = MemoryBackend::new();
        let stored = shape.checkpointed(&log);
        group.throughput(Throughput::Elements(u64::from(SWEEP)));
        bench_cold(&mut group, &format!("cold/{label}"), &shape, &log);
        bench_restore(
            &mut group,
            &format!("restore/{label}"),
            &shape,
            &log,
            stored.path(),
        );
    }
    group.finish();
}

fn index(c: &mut Criterion) {
    let items: Vec<IndexItem> = (0..BIG)
        .map(|k| IndexItem {
            table: 0,
            key_hash: key_hash(&k.to_le_bytes()),
            loc: u64::from(k) * 64,
            len: 64,
        })
        .collect();
    let header = GenHeader {
        build: BUILD,
        seq: 1,
        tables: Vec::new(),
    };
    let mut group = c.benchmark_group("memo_index_100k");
    group.throughput(Throughput::Elements(u64::from(BIG)));
    group.bench_function("build", |b| {
        b.iter_batched(
            || items.clone(),
            |items| Generation::new(GenId(1), header.clone(), items),
            BatchSize::LargeInput,
        )
    });
    let generation = Generation::new(GenId(1), header.clone(), items.clone());
    group.bench_function("find", |b| {
        b.iter(|| {
            for item in &items {
                black_box(generation.find(item.table, item.key_hash));
            }
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    chain::<Local>,
    wide::<Local>,
    sweep::<Local>,
    index,
    chain::<Shared>,
    wide::<Shared>,
    sweep::<Shared>
);
criterion_main!(benches);
