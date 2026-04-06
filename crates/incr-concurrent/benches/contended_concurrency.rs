// crates/incr-concurrent/benches/contended_concurrency.rs
//
// Multi-threaded contended benchmark for atomic-field node access.
//
// The single-threaded benchmark (concurrency_primitives.rs) established that
// atomic fields with Relaxed ordering match baseline direct access within 1%
// on the uncontended path. This benchmark tests whether that design holds up
// under real contention: whether readers scale linearly, whether partitioned
// writers stay cheap, and whether the Release/Acquire publish pattern used
// for completing a compute and publishing a value stays cheap as readers and
// writers multiply.
//
// Workloads:
//   read_scaling       - N threads reading all nodes. Should be near-linear.
//   partitioned_write  - N threads writing disjoint partitions. Should also be
//                        near-linear because no cache line is shared.
//   overlapping_write  - N threads writing the same nodes. Pessimistic case;
//                        measures cache-coherence cost under contention.
//   acquire_release    - All threads read via Acquire loads and occasionally
//                        publish via Release stores. Realistic pattern for
//                        completing a compute and publishing new value.
//
// Timing approach: criterion iter_custom with std::thread::scope. The scope
// blocks until all spawned threads complete, which means we time the full
// "spawn + barrier + iters of work + join" sequence. For iter counts large
// enough to give criterion its ~50ms sample window, the thread-spawn overhead
// (~50 microseconds for 8 threads) is under 0.2% of sample time and does not
// distort the measurement.
//
// Node shape: AtomicNode (64 bytes, AtomicU64 fields). Size: 16384 nodes
// (~1 MB, L2 resident on the dev machine i7-9750H). Thread counts: 1, 2, 4, 8.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

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

fn build_nodes(n: usize) -> Arc<Vec<AtomicNode>> {
    Arc::new((0..n as u32).map(AtomicNode::new).collect())
}

// ============================================================================
// Workload functions. All operate on a shared &[AtomicNode] and return an
// aggregated u64 (or nothing for writes) to prevent the compiler from
// eliminating the work.
// ============================================================================

#[inline(always)]
fn read_all_relaxed(nodes: &[AtomicNode]) -> u64 {
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
fn write_partition_relaxed(
    nodes: &[AtomicNode],
    partition: usize,
    num_partitions: usize,
    rev: u64,
) {
    let start = (nodes.len() * partition) / num_partitions;
    let end = (nodes.len() * (partition + 1)) / num_partitions;
    for n in &nodes[start..end] {
        n.state.store(1, Ordering::Relaxed);
        n.verified_at.store(rev, Ordering::Relaxed);
    }
}

#[inline(always)]
fn write_all_relaxed(nodes: &[AtomicNode], rev: u64) {
    for n in nodes {
        n.state.store(1, Ordering::Relaxed);
        n.verified_at.store(rev, Ordering::Relaxed);
    }
}

#[inline(always)]
fn consume_value_acquire(nodes: &[AtomicNode]) -> u64 {
    let mut acc: u64 = 0;
    for n in nodes {
        let state = n.state.load(Ordering::Acquire);
        if state != 0 {
            acc = acc.wrapping_add(n.value.load(Ordering::Relaxed));
        }
    }
    acc
}

#[inline(always)]
fn publish_value_release(nodes: &[AtomicNode], rev: u64) {
    for n in nodes {
        n.value.store(rev, Ordering::Relaxed);
        n.state.store(1, Ordering::Release);
    }
}

// ============================================================================
// Generic parallel-timing harness. Takes a closure that receives the thread
// index and does one iteration of work. Returns the total wall-clock duration
// for `iters` iterations across `threads` threads, all starting at a barrier.
// ============================================================================

fn run_parallel<F>(iters: u64, threads: usize, work: F) -> Duration
where
    F: Fn(usize) + Sync,
{
    let barrier = Barrier::new(threads);
    let work_ref = &work;
    let barrier_ref = &barrier;
    let start = Instant::now();
    thread::scope(|s| {
        for t in 0..threads {
            s.spawn(move || {
                barrier_ref.wait();
                for _ in 0..iters {
                    work_ref(t);
                }
            });
        }
    });
    start.elapsed()
}

// ============================================================================
// Benchmarks
// ============================================================================

const SIZE: usize = 16384;
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8];

fn bench_read_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_scaling");
    group.sample_size(30);

    for &threads in THREAD_COUNTS {
        let nodes = build_nodes(SIZE);
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let nodes_ref: &Vec<AtomicNode> = &nodes;
                    run_parallel(iters, threads, |_t| {
                        black_box(read_all_relaxed(nodes_ref));
                    })
                });
            },
        );
    }
    group.finish();
}

fn bench_partitioned_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("partitioned_write");
    group.sample_size(30);

    for &threads in THREAD_COUNTS {
        let nodes = build_nodes(SIZE);
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |b, &threads| {
                let mut rev: u64 = 1;
                b.iter_custom(|iters| {
                    let nodes_ref: &Vec<AtomicNode> = &nodes;
                    rev = rev.wrapping_add(1);
                    let current_rev = rev;
                    run_parallel(iters, threads, move |t| {
                        write_partition_relaxed(nodes_ref, t, threads, current_rev);
                    })
                });
            },
        );
    }
    group.finish();
}

fn bench_overlapping_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("overlapping_write");
    group.sample_size(30);

    for &threads in THREAD_COUNTS {
        let nodes = build_nodes(SIZE);
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |b, &threads| {
                let mut rev: u64 = 1;
                b.iter_custom(|iters| {
                    let nodes_ref: &Vec<AtomicNode> = &nodes;
                    rev = rev.wrapping_add(1);
                    let current_rev = rev;
                    run_parallel(iters, threads, move |_t| {
                        write_all_relaxed(nodes_ref, current_rev);
                    })
                });
            },
        );
    }
    group.finish();
}

/// Acquire/Release publish pattern. One thread (index 0) acts as the publisher,
/// calling publish_value_release on every iteration. The remaining N-1 threads
/// act as consumers, calling consume_value_acquire. This measures the realistic
/// cost of the pattern incr will use for completing a compute and publishing
/// the new value to downstream readers. When threads==1 we measure the writer
/// alone; when threads>1 we measure the writer plus N-1 readers.
fn bench_acquire_release(c: &mut Criterion) {
    let mut group = c.benchmark_group("acquire_release");
    group.sample_size(30);

    for &threads in THREAD_COUNTS {
        let nodes = build_nodes(SIZE);
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |b, &threads| {
                let mut rev: u64 = 1;
                b.iter_custom(|iters| {
                    let nodes_ref: &Vec<AtomicNode> = &nodes;
                    rev = rev.wrapping_add(1);
                    let current_rev = rev;
                    run_parallel(iters, threads, move |t| {
                        if t == 0 {
                            publish_value_release(nodes_ref, current_rev);
                        } else {
                            black_box(consume_value_acquire(nodes_ref));
                        }
                    })
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_read_scaling,
    bench_partitioned_write,
    bench_overlapping_write,
    bench_acquire_release
);
criterion_main!(benches);
