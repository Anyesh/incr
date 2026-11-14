//! R11: the stall of a log checkpoint (`Store::checkpoint`) over a large
//! store, which folds the whole state into one snapshot and compacts the
//! backend to it.
//!
//! The store holds `INCR_LOG_CKPT_BYTES` of input values (default 1 GiB),
//! each input `INCR_LOG_CKPT_VALUE` bytes (default 1000). It is
//! checkpointed once, untimed. Each timed iteration first rewrites
//! `INCR_LOG_CKPT_DIRTY` inputs (default 1000), untimed, then times
//! `Store::checkpoint`. The API reports no per-step times for it, so the
//! groups give the whole call, `checkpoint`, and, on the memory backend,
//! `reopen`: a fresh `Store::open` over the checkpointed backend. Peak
//! resident memory (`VmHWM`, Linux) is printed after the fill and after the
//! timed groups. Groups are named by backend, `memory` and `file`, the file
//! backend in a fresh directory under the workspace's `target/`.
//!
//! A small run that checks the bench works:
//! `INCR_LOG_CKPT_BYTES=4194304 cargo bench -p incr-persist --bench log_checkpoint -- --quick`.

// The shared module also holds `BenchCells`, which this bench has no use for.
#[allow(dead_code)]
mod common;

use std::time::{Duration, Instant};

use common::bench_dir;
use criterion::{criterion_group, criterion_main, Criterion};
use incr_persist::{
    Backend, Cells, Durability, FileBackend, Input, Local, MemoryBackend, Options, Store,
};

const FILL_CHUNK: usize = 1000;

fn env_or(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a whole number, got {v:?}")),
        Err(_) => default,
    }
}

fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

struct Sizes {
    inputs: usize,
    dirty: usize,
    value: usize,
}

impl Sizes {
    fn from_env() -> Self {
        let bytes = env_or("INCR_LOG_CKPT_BYTES", 1 << 30);
        let value = env_or("INCR_LOG_CKPT_VALUE", 1000).max(1);
        let inputs = usize::try_from((bytes / value).max(1)).expect("fits usize");
        Sizes {
            inputs,
            dirty: usize::try_from(env_or("INCR_LOG_CKPT_DIRTY", 1000))
                .unwrap()
                .min(inputs),
            value: usize::try_from(value).unwrap(),
        }
    }
}

struct Filled<C: Cells, B: Backend> {
    store: Store<C, B>,
    inputs: Vec<Input<Vec<u8>>>,
    sizes: Sizes,
    epoch: u8,
}

impl<C: Cells, B: Backend> Filled<C, B> {
    fn open(backend: B) -> Self {
        let sizes = Sizes::from_env();
        let options = Options {
            durability: Durability::Buffered,
            ..Options::default()
        };
        let store = Store::<C, B>::open(backend, options).unwrap();
        let started = Instant::now();
        let inputs: Vec<Input<Vec<u8>>> = (0..sizes.inputs)
            .map(|i| store.input(&format!("in/{i}"), Vec::new()).unwrap())
            .collect();
        for chunk in inputs.chunks(FILL_CHUNK) {
            let value = vec![0_u8; sizes.value];
            store
                .batch(|b| {
                    for &input in chunk {
                        b.set(input, value.clone());
                    }
                })
                .unwrap();
        }
        let filled_in = started.elapsed();
        store.checkpoint().unwrap();
        eprintln!(
            "log_checkpoint: {} inputs of {} bytes filled in {:?}, {} dirty per checkpoint, peak RSS {:?} KiB",
            sizes.inputs,
            sizes.value,
            filled_in,
            sizes.dirty,
            peak_rss_kib()
        );
        Filled {
            store,
            inputs,
            sizes,
            epoch: 0,
        }
    }

    fn checkpoint(&mut self) -> Duration {
        self.epoch = self.epoch.wrapping_add(1);
        let value = vec![self.epoch; self.sizes.value];
        let dirty = &self.inputs[..self.sizes.dirty];
        self.store
            .batch(|b| {
                for &input in dirty {
                    b.set(input, value.clone());
                }
            })
            .unwrap();
        let started = Instant::now();
        self.store.checkpoint().unwrap();
        started.elapsed()
    }
}

fn memory(c: &mut Criterion) {
    let backend = MemoryBackend::new();
    let mut filled = Filled::<Local, _>::open(backend.clone());
    let mut group = c.benchmark_group("log_checkpoint_memory");
    group.sample_size(10);
    group.bench_function("checkpoint", |b| {
        b.iter_custom(|iters| (0..iters).map(|_| filled.checkpoint()).sum())
    });
    group.bench_function("reopen", |b| {
        b.iter_custom(|iters| {
            (0..iters)
                .map(|_| {
                    let started = Instant::now();
                    let store = Store::<Local, _>::open(backend.clone(), Options::default());
                    let took = started.elapsed();
                    criterion::black_box(store.unwrap());
                    took
                })
                .sum()
        })
    });
    group.finish();
    eprintln!("log_checkpoint: peak RSS {:?} KiB", peak_rss_kib());
}

fn file(c: &mut Criterion) {
    let dir = bench_dir();
    let mut filled = Filled::<Local, _>::open(FileBackend::open(dir.path()).unwrap());
    let mut group = c.benchmark_group("log_checkpoint_file");
    group.sample_size(10);
    group.bench_function("checkpoint", |b| {
        b.iter_custom(|iters| (0..iters).map(|_| filled.checkpoint()).sum())
    });
    group.finish();
    eprintln!("log_checkpoint: peak RSS {:?} KiB", peak_rss_kib());
}

criterion_group!(benches, memory, file);
criterion_main!(benches);
