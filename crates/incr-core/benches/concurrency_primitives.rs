// crates/incr-core/benches/concurrency_primitives.rs
//
// Microbenchmark: the cost of node access under different concurrency primitives.
//
// This benchmark is load-bearing for the architectural decision described in
// devlogs/2026-04-05-core-design-values.md. The central question is whether a
// unified Runtime whose uncontended single-threaded path pays near-zero overhead
// for its concurrency primitives is feasible. If it is, incr gets one API for
// both single-threaded and concurrent use, and the 175ns per-node budget survives.
// If it is not, we split the runtime into local and shared variants, sharing
// internals via generics over a concurrency strategy trait, so that neither side
// subsidizes the other.
//
// The benchmark isolates the primitive cost by using a fixed u64 value field
// (no Box<dyn Any>) and a representative 64-byte node layout, so the numbers
// reflect concurrency-primitive cost rather than value-storage cost. The
// Box<dyn Any> issue is a separate atom-level perf gap tracked in memory.
//
// Primitives compared (single-threaded, uncontended):
//   1. Baseline      - Vec<Node>, direct field access, no sync. Theoretical floor.
//   2. RefCell       - RefCell<Vec<Node>>, matches current engine.
//   3. Atomic fields - each scalar is AtomicU64 with Relaxed ordering.
//   4. Seqlock       - per-node version counter + relaxed atomic payload.
//   5. Epoch         - crossbeam_epoch Atomic<Box<Node>>, pin-and-load reads.
//
// Workloads:
//   - sequential_read : walk in order, read (state, value, verified, changed)
//   - random_read     : read in a precomputed shuffled order
//   - traversal       : follow a precomputed next-index chain, simulating ensure_clean
//   - write_burst     : update state + verified_at for every node, simulating mark_dirty
//
// Sizes: 64 (L1), 1024 (L1 edge), 16384 (L2), 262144 (L3). Four sizes times four
// workloads times five primitives is eighty measurements.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use crossbeam_epoch::{self as epoch, Atomic, Owned};
use rand::prelude::SliceRandom;
use rand::SeedableRng;
use std::cell::RefCell;
use std::sync::atomic::{fence, AtomicU64, Ordering};

// ============================================================================
// Node shapes. All variants target ~64 bytes (one cache line) so that cache
// behavior is comparable across variants.
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy)]
struct BaselineNode {
    state: u64, // u8 widened to u64 to match atomic variant word size
    value: u64,
    verified_at: u64,
    changed_at: u64,
    _pad: [u64; 4], // 32 bytes of padding to simulate deps storage, total 64 bytes
}

impl BaselineNode {
    fn new(i: u32) -> Self {
        Self {
            state: 0,
            value: i as u64,
            verified_at: i as u64,
            changed_at: i as u64,
            _pad: [0; 4],
        }
    }
}

#[repr(C)]
struct AtomicNode {
    state: AtomicU64,
    value: AtomicU64,
    verified_at: AtomicU64,
    changed_at: AtomicU64,
    _pad: [AtomicU64; 4],
}

impl AtomicNode {
    fn new(i: u32) -> Self {
        Self {
            state: AtomicU64::new(0),
            value: AtomicU64::new(i as u64),
            verified_at: AtomicU64::new(i as u64),
            changed_at: AtomicU64::new(i as u64),
            _pad: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        }
    }
}

#[repr(C)]
struct SeqlockNode {
    version: AtomicU64,
    state: AtomicU64,
    value: AtomicU64,
    verified_at: AtomicU64,
    changed_at: AtomicU64,
    _pad: [AtomicU64; 3],
}

impl SeqlockNode {
    fn new(i: u32) -> Self {
        Self {
            version: AtomicU64::new(0),
            state: AtomicU64::new(0),
            value: AtomicU64::new(i as u64),
            verified_at: AtomicU64::new(i as u64),
            changed_at: AtomicU64::new(i as u64),
            _pad: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        }
    }

    /// Seqlock read: returns (state, value, verified_at, changed_at) consistent
    /// as-of a single version. Retries if a concurrent writer is mid-update.
    /// On an uncontended single-threaded path, the retry never fires.
    #[inline(always)]
    fn read_all(&self) -> (u64, u64, u64, u64) {
        loop {
            let v1 = self.version.load(Ordering::Acquire);
            if v1 & 1 != 0 {
                // Writer in progress
                std::hint::spin_loop();
                continue;
            }
            let state = self.state.load(Ordering::Relaxed);
            let value = self.value.load(Ordering::Relaxed);
            let verified_at = self.verified_at.load(Ordering::Relaxed);
            let changed_at = self.changed_at.load(Ordering::Relaxed);
            fence(Ordering::Acquire);
            let v2 = self.version.load(Ordering::Relaxed);
            if v1 == v2 {
                return (state, value, verified_at, changed_at);
            }
        }
    }

    /// Seqlock write: bumps version odd, writes fields, bumps version even.
    #[inline(always)]
    fn write_state_verified(&self, state: u64, verified_at: u64) {
        let v = self.version.load(Ordering::Relaxed);
        self.version.store(v.wrapping_add(1), Ordering::Release);
        self.state.store(state, Ordering::Relaxed);
        self.verified_at.store(verified_at, Ordering::Relaxed);
        self.version.store(v.wrapping_add(2), Ordering::Release);
    }
}

/// Epoch variant. The payload lives in a heap allocation behind an Atomic<T>.
/// Readers pin an epoch and load the pointer. Writers allocate a new payload
/// and CAS it in, deferring destruction of the old one until all readers have
/// advanced past the current epoch. This is the crossbeam_epoch pattern.
#[repr(C)]
struct EpochSlot {
    payload: Atomic<BaselineNode>,
}

impl EpochSlot {
    fn new(i: u32) -> Self {
        Self {
            payload: Atomic::new(BaselineNode::new(i)),
        }
    }
}

// ============================================================================
// Storage builders
// ============================================================================

fn build_baseline(n: usize) -> Vec<BaselineNode> {
    (0..n as u32).map(BaselineNode::new).collect()
}

fn build_refcell(n: usize) -> RefCell<Vec<BaselineNode>> {
    RefCell::new(build_baseline(n))
}

fn build_atomic(n: usize) -> Vec<AtomicNode> {
    (0..n as u32).map(AtomicNode::new).collect()
}

fn build_seqlock(n: usize) -> Vec<SeqlockNode> {
    (0..n as u32).map(SeqlockNode::new).collect()
}

fn build_epoch(n: usize) -> Vec<EpochSlot> {
    (0..n as u32).map(EpochSlot::new).collect()
}

// ============================================================================
// Precomputed access orders. Built once outside the bench loop so that RNG
// cost and chain-building cost do not pollute the measurement.
// ============================================================================

fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    idx
}

/// Build a "next index" chain: traversal_chain[i] gives the index to visit after i.
/// Forms a single cycle touching every element exactly once. This simulates
/// the linked traversal pattern of ensure_clean walking dependencies.
fn traversal_chain(n: usize, seed: u64) -> Vec<usize> {
    let shuffled = shuffled_indices(n, seed);
    let mut chain = vec![0usize; n];
    for i in 0..n {
        chain[shuffled[i]] = shuffled[(i + 1) % n];
    }
    chain
}

// ============================================================================
// Workloads - Sequential Read
// ============================================================================

#[inline(always)]
fn seq_read_baseline(nodes: &[BaselineNode]) -> u64 {
    let mut acc: u64 = 0;
    for n in nodes {
        acc = acc
            .wrapping_add(n.state)
            .wrapping_add(n.value)
            .wrapping_add(n.verified_at)
            .wrapping_add(n.changed_at);
    }
    acc
}

#[inline(always)]
fn seq_read_refcell(nodes: &RefCell<Vec<BaselineNode>>) -> u64 {
    let borrowed = nodes.borrow();
    let mut acc: u64 = 0;
    for n in borrowed.iter() {
        acc = acc
            .wrapping_add(n.state)
            .wrapping_add(n.value)
            .wrapping_add(n.verified_at)
            .wrapping_add(n.changed_at);
    }
    acc
}

#[inline(always)]
fn seq_read_atomic(nodes: &[AtomicNode]) -> u64 {
    let mut acc: u64 = 0;
    for n in nodes {
        acc = acc
            .wrapping_add(n.state.load(Ordering::Relaxed))
            .wrapping_add(n.value.load(Ordering::Relaxed))
            .wrapping_add(n.verified_at.load(Ordering::Relaxed))
            .wrapping_add(n.changed_at.load(Ordering::Relaxed));
    }
    acc
}

#[inline(always)]
fn seq_read_seqlock(nodes: &[SeqlockNode]) -> u64 {
    let mut acc: u64 = 0;
    for n in nodes {
        let (s, v, ver, ch) = n.read_all();
        acc = acc
            .wrapping_add(s)
            .wrapping_add(v)
            .wrapping_add(ver)
            .wrapping_add(ch);
    }
    acc
}

#[inline(always)]
fn seq_read_epoch(nodes: &[EpochSlot]) -> u64 {
    let guard = &epoch::pin();
    let mut acc: u64 = 0;
    for slot in nodes {
        // Safety: the epoch pin guarantees the pointee is valid for the read
        let shared = slot.payload.load(Ordering::Acquire, guard);
        let node = unsafe { shared.deref() };
        acc = acc
            .wrapping_add(node.state)
            .wrapping_add(node.value)
            .wrapping_add(node.verified_at)
            .wrapping_add(node.changed_at);
    }
    acc
}

// ============================================================================
// Workloads - Random Read
// ============================================================================

#[inline(always)]
fn rand_read_baseline(nodes: &[BaselineNode], order: &[usize]) -> u64 {
    let mut acc: u64 = 0;
    for &i in order {
        let n = &nodes[i];
        acc = acc
            .wrapping_add(n.state)
            .wrapping_add(n.value)
            .wrapping_add(n.verified_at)
            .wrapping_add(n.changed_at);
    }
    acc
}

#[inline(always)]
fn rand_read_refcell(nodes: &RefCell<Vec<BaselineNode>>, order: &[usize]) -> u64 {
    let borrowed = nodes.borrow();
    let mut acc: u64 = 0;
    for &i in order {
        let n = &borrowed[i];
        acc = acc
            .wrapping_add(n.state)
            .wrapping_add(n.value)
            .wrapping_add(n.verified_at)
            .wrapping_add(n.changed_at);
    }
    acc
}

#[inline(always)]
fn rand_read_atomic(nodes: &[AtomicNode], order: &[usize]) -> u64 {
    let mut acc: u64 = 0;
    for &i in order {
        let n = &nodes[i];
        acc = acc
            .wrapping_add(n.state.load(Ordering::Relaxed))
            .wrapping_add(n.value.load(Ordering::Relaxed))
            .wrapping_add(n.verified_at.load(Ordering::Relaxed))
            .wrapping_add(n.changed_at.load(Ordering::Relaxed));
    }
    acc
}

#[inline(always)]
fn rand_read_seqlock(nodes: &[SeqlockNode], order: &[usize]) -> u64 {
    let mut acc: u64 = 0;
    for &i in order {
        let (s, v, ver, ch) = nodes[i].read_all();
        acc = acc
            .wrapping_add(s)
            .wrapping_add(v)
            .wrapping_add(ver)
            .wrapping_add(ch);
    }
    acc
}

#[inline(always)]
fn rand_read_epoch(nodes: &[EpochSlot], order: &[usize]) -> u64 {
    let guard = &epoch::pin();
    let mut acc: u64 = 0;
    for &i in order {
        let shared = nodes[i].payload.load(Ordering::Acquire, guard);
        let node = unsafe { shared.deref() };
        acc = acc
            .wrapping_add(node.state)
            .wrapping_add(node.value)
            .wrapping_add(node.verified_at)
            .wrapping_add(node.changed_at);
    }
    acc
}

// ============================================================================
// Workloads - Traversal. Walks a precomputed chain of next-indices, simulating
// the ensure_clean pattern where each node read leads to another. The chain
// visits every node exactly once.
// ============================================================================

#[inline(always)]
fn traversal_baseline(nodes: &[BaselineNode], chain: &[usize], start: usize) -> u64 {
    let mut i = start;
    let mut acc: u64 = 0;
    for _ in 0..nodes.len() {
        let n = &nodes[i];
        acc = acc.wrapping_add(n.state).wrapping_add(n.value);
        i = chain[i];
    }
    acc
}

#[inline(always)]
fn traversal_refcell(nodes: &RefCell<Vec<BaselineNode>>, chain: &[usize], start: usize) -> u64 {
    let borrowed = nodes.borrow();
    let mut i = start;
    let mut acc: u64 = 0;
    for _ in 0..borrowed.len() {
        let n = &borrowed[i];
        acc = acc.wrapping_add(n.state).wrapping_add(n.value);
        i = chain[i];
    }
    acc
}

#[inline(always)]
fn traversal_atomic(nodes: &[AtomicNode], chain: &[usize], start: usize) -> u64 {
    let mut i = start;
    let mut acc: u64 = 0;
    for _ in 0..nodes.len() {
        let n = &nodes[i];
        acc = acc
            .wrapping_add(n.state.load(Ordering::Relaxed))
            .wrapping_add(n.value.load(Ordering::Relaxed));
        i = chain[i];
    }
    acc
}

#[inline(always)]
fn traversal_seqlock(nodes: &[SeqlockNode], chain: &[usize], start: usize) -> u64 {
    let mut i = start;
    let mut acc: u64 = 0;
    for _ in 0..nodes.len() {
        let (s, v, _, _) = nodes[i].read_all();
        acc = acc.wrapping_add(s).wrapping_add(v);
        i = chain[i];
    }
    acc
}

#[inline(always)]
fn traversal_epoch(nodes: &[EpochSlot], chain: &[usize], start: usize) -> u64 {
    let guard = &epoch::pin();
    let mut i = start;
    let mut acc: u64 = 0;
    for _ in 0..nodes.len() {
        let shared = nodes[i].payload.load(Ordering::Acquire, guard);
        let node = unsafe { shared.deref() };
        acc = acc.wrapping_add(node.state).wrapping_add(node.value);
        i = chain[i];
    }
    acc
}

// ============================================================================
// Workloads - Write Burst. Simulates mark_dirty: iterate and update the state
// and verified_at fields of every node.
// ============================================================================

#[inline(always)]
fn write_burst_baseline(nodes: &mut [BaselineNode], rev: u64) {
    for n in nodes {
        n.state = 1;
        n.verified_at = rev;
    }
}

#[inline(always)]
fn write_burst_refcell(nodes: &RefCell<Vec<BaselineNode>>, rev: u64) {
    let mut borrowed = nodes.borrow_mut();
    for n in borrowed.iter_mut() {
        n.state = 1;
        n.verified_at = rev;
    }
}

#[inline(always)]
fn write_burst_atomic(nodes: &[AtomicNode], rev: u64) {
    for n in nodes {
        n.state.store(1, Ordering::Relaxed);
        n.verified_at.store(rev, Ordering::Relaxed);
    }
}

#[inline(always)]
fn write_burst_seqlock(nodes: &[SeqlockNode], rev: u64) {
    for n in nodes {
        n.write_state_verified(1, rev);
    }
}

#[inline(always)]
fn write_burst_epoch(nodes: &[EpochSlot], rev: u64) {
    // Pin per-node rather than per-call. A single long-held pin accumulates
    // deferred destroys in the thread-local garbage bag without allowing
    // reclamation, which causes runaway memory growth under sustained writes.
    // Per-node pinning lets the global epoch advance between operations so
    // reclamation keeps up, and it reflects the realistic cost of using
    // crossbeam_epoch for a write-heavy node store.
    for slot in nodes {
        let guard = &epoch::pin();
        let current = slot.payload.load(Ordering::Acquire, guard);
        let current_ref = unsafe { current.deref() };
        let mut new_node = *current_ref;
        new_node.state = 1;
        new_node.verified_at = rev;
        let new_owned = Owned::new(new_node);
        match slot.payload.compare_exchange(
            current,
            new_owned,
            Ordering::AcqRel,
            Ordering::Acquire,
            guard,
        ) {
            Ok(_) => unsafe {
                guard.defer_destroy(current);
            },
            Err(_) => {
                // Under contention the CAS would fail; uncontended it will not.
                // If it does fail, we skip rather than retry for benchmark
                // consistency (we are measuring uncontended cost).
            }
        }
    }
}

// ============================================================================
// Criterion harness
// ============================================================================

// Sizes chosen to span cache hierarchy: 64 nodes (~4KB, L1), 1024 (~64KB, L1 edge),
// 16384 (~1MB, L2), 65536 (~4MB, L3). The top size was originally 262144 (~16MB)
// but the epoch write variant produced runaway garbage at that scale even with
// per-node pinning, so 65536 is the honest "beyond L2" signal without distorting
// other measurements through memory pressure.
const SIZES: &[usize] = &[64, 1024, 16384, 65536];

fn bench_sequential_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_read");

    for &size in SIZES {
        let baseline = build_baseline(size);
        let refcell = build_refcell(size);
        let atomic = build_atomic(size);
        let seqlock = build_seqlock(size);
        let epoch_nodes = build_epoch(size);

        group.bench_with_input(BenchmarkId::new("baseline", size), &size, |b, _| {
            b.iter(|| black_box(seq_read_baseline(&baseline)));
        });
        group.bench_with_input(BenchmarkId::new("refcell", size), &size, |b, _| {
            b.iter(|| black_box(seq_read_refcell(&refcell)));
        });
        group.bench_with_input(BenchmarkId::new("atomic", size), &size, |b, _| {
            b.iter(|| black_box(seq_read_atomic(&atomic)));
        });
        group.bench_with_input(BenchmarkId::new("seqlock", size), &size, |b, _| {
            b.iter(|| black_box(seq_read_seqlock(&seqlock)));
        });
        group.bench_with_input(BenchmarkId::new("epoch", size), &size, |b, _| {
            b.iter(|| black_box(seq_read_epoch(&epoch_nodes)));
        });
    }

    group.finish();
}

fn bench_random_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_read");

    for &size in SIZES {
        let baseline = build_baseline(size);
        let refcell = build_refcell(size);
        let atomic = build_atomic(size);
        let seqlock = build_seqlock(size);
        let epoch_nodes = build_epoch(size);
        let order = shuffled_indices(size, 0xdeadbeef);

        group.bench_with_input(BenchmarkId::new("baseline", size), &size, |b, _| {
            b.iter(|| black_box(rand_read_baseline(&baseline, &order)));
        });
        group.bench_with_input(BenchmarkId::new("refcell", size), &size, |b, _| {
            b.iter(|| black_box(rand_read_refcell(&refcell, &order)));
        });
        group.bench_with_input(BenchmarkId::new("atomic", size), &size, |b, _| {
            b.iter(|| black_box(rand_read_atomic(&atomic, &order)));
        });
        group.bench_with_input(BenchmarkId::new("seqlock", size), &size, |b, _| {
            b.iter(|| black_box(rand_read_seqlock(&seqlock, &order)));
        });
        group.bench_with_input(BenchmarkId::new("epoch", size), &size, |b, _| {
            b.iter(|| black_box(rand_read_epoch(&epoch_nodes, &order)));
        });
    }

    group.finish();
}

fn bench_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("traversal");

    for &size in SIZES {
        let baseline = build_baseline(size);
        let refcell = build_refcell(size);
        let atomic = build_atomic(size);
        let seqlock = build_seqlock(size);
        let epoch_nodes = build_epoch(size);
        let chain = traversal_chain(size, 0xcafef00d);

        group.bench_with_input(BenchmarkId::new("baseline", size), &size, |b, _| {
            b.iter(|| black_box(traversal_baseline(&baseline, &chain, 0)));
        });
        group.bench_with_input(BenchmarkId::new("refcell", size), &size, |b, _| {
            b.iter(|| black_box(traversal_refcell(&refcell, &chain, 0)));
        });
        group.bench_with_input(BenchmarkId::new("atomic", size), &size, |b, _| {
            b.iter(|| black_box(traversal_atomic(&atomic, &chain, 0)));
        });
        group.bench_with_input(BenchmarkId::new("seqlock", size), &size, |b, _| {
            b.iter(|| black_box(traversal_seqlock(&seqlock, &chain, 0)));
        });
        group.bench_with_input(BenchmarkId::new("epoch", size), &size, |b, _| {
            b.iter(|| black_box(traversal_epoch(&epoch_nodes, &chain, 0)));
        });
    }

    group.finish();
}

fn bench_write_burst(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_burst");

    for &size in SIZES {
        let mut baseline = build_baseline(size);
        let refcell = build_refcell(size);
        let atomic = build_atomic(size);
        let seqlock = build_seqlock(size);
        let epoch_nodes = build_epoch(size);
        let mut rev: u64 = 1;

        group.bench_with_input(BenchmarkId::new("baseline", size), &size, |b, _| {
            b.iter(|| {
                rev = rev.wrapping_add(1);
                write_burst_baseline(&mut baseline, rev);
                black_box(&baseline);
            });
        });
        group.bench_with_input(BenchmarkId::new("refcell", size), &size, |b, _| {
            b.iter(|| {
                rev = rev.wrapping_add(1);
                write_burst_refcell(&refcell, rev);
                black_box(&refcell);
            });
        });
        group.bench_with_input(BenchmarkId::new("atomic", size), &size, |b, _| {
            b.iter(|| {
                rev = rev.wrapping_add(1);
                write_burst_atomic(&atomic, rev);
                black_box(&atomic);
            });
        });
        group.bench_with_input(BenchmarkId::new("seqlock", size), &size, |b, _| {
            b.iter(|| {
                rev = rev.wrapping_add(1);
                write_burst_seqlock(&seqlock, rev);
                black_box(&seqlock);
            });
        });
        group.bench_with_input(BenchmarkId::new("epoch", size), &size, |b, _| {
            b.iter(|| {
                rev = rev.wrapping_add(1);
                write_burst_epoch(&epoch_nodes, rev);
                black_box(&epoch_nodes);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sequential_read,
    bench_random_read,
    bench_traversal,
    bench_write_burst
);
criterion_main!(benches);
