//! Head-to-head performance comparison between v1 and v2.
//!
//! Four workloads from `benches/performance.rs` ported to run
//! against both engines in the same binary, with the same iteration
//! counts, on the same machine. This is deliberately in-crate
//! (rather than using criterion in `benches/`) because v2 is
//! `pub(crate)` until Gate 5 and criterion benches live outside the
//! crate. The tradeoff: we lose criterion's statistical warm-up and
//! confidence intervals, but we gain direct A/B comparison of v1
//! and v2 on every run.
//!
//! Run with:
//!
//! ```text
//! cargo test --release -p incr-core v2::runtime_vs_v1_bench \
//!   -- --ignored --nocapture
//! ```
//!
//! The `#[ignore]` gate keeps the benchmarks out of the normal
//! test run so `cargo test` stays fast. Each benchmark prints
//! ns-per-op numbers for both engines and the v2/v1 ratio. A
//! ratio of 1.0 means parity; above 1.0 means v2 is slower than
//! v1 (a regression); below 1.0 means v2 is faster (a win).

use std::hint::black_box;
use std::time::Instant;

use crate::v2::runtime::Runtime as V2Runtime;
use crate::Runtime as V1Runtime;

/// How many iterations to run each benchmark inner loop. Large
/// enough that timer resolution (~1 µs on Linux) is well below
/// the measurement noise.
const ITERS_HOT: usize = 1_000_000;
const ITERS_PROPAGATE: usize = 50_000;

/// Print a result line with ns-per-op for both engines and the
/// v2/v1 ratio. Lead with the label so `grep` can pull out
/// individual results from the benchmark output.
fn report(label: &str, v1_ns: f64, v2_ns: f64) {
    let ratio = v2_ns / v1_ns;
    let marker = if ratio < 1.0 {
        "WIN "
    } else if ratio < 1.1 {
        "par "
    } else if ratio < 2.0 {
        "SLOW"
    } else {
        "REGR"
    };
    eprintln!(
        "[{marker}] {label:<40} v1={v1_ns:>8.2} ns/op  v2={v2_ns:>8.2} ns/op  ratio={ratio:.2}x"
    );
}

#[test]
#[ignore = "benchmark, run with --release"]
fn bench_hot_clean_read_input() {
    // Read an input whose value has already been set, in a tight
    // loop. This isolates the single-threaded read path cost:
    // handle verification, state check, arena read. No compute
    // runs, no dirty walk, no locks contested.

    let v1_ns = {
        let rt = V1Runtime::new();
        let input = rt.create_input::<u64>(42);
        let start = Instant::now();
        let mut acc = 0u64;
        for _ in 0..ITERS_HOT {
            acc = acc.wrapping_add(rt.get(input));
        }
        let elapsed = start.elapsed();
        black_box(acc);
        elapsed.as_nanos() as f64 / ITERS_HOT as f64
    };

    let v2_ns = {
        let rt = V2Runtime::new();
        let input = rt.create_input::<u64>(42);
        let start = Instant::now();
        let mut acc = 0u64;
        for _ in 0..ITERS_HOT {
            acc = acc.wrapping_add(rt.get(input));
        }
        let elapsed = start.elapsed();
        black_box(acc);
        elapsed.as_nanos() as f64 / ITERS_HOT as f64
    };

    report("hot_clean_read_input_u64", v1_ns, v2_ns);
}

#[test]
#[ignore = "benchmark, run with --release"]
fn bench_hot_clean_read_query() {
    // Read a query whose value is already Clean. This path adds
    // the compute_fns access over the input-read path, though
    // the compute closure does not run (Clean fast path).

    let v1_ns = {
        let rt = V1Runtime::new();
        let input = rt.create_input::<u64>(7);
        let query = rt.create_query::<u64, _>(move |rt| rt.get(input) * 2);
        let _ = rt.get(query);
        let start = Instant::now();
        let mut acc = 0u64;
        for _ in 0..ITERS_HOT {
            acc = acc.wrapping_add(rt.get(query));
        }
        let elapsed = start.elapsed();
        black_box(acc);
        elapsed.as_nanos() as f64 / ITERS_HOT as f64
    };

    let v2_ns = {
        let rt = V2Runtime::new();
        let input = rt.create_input::<u64>(7);
        let query = rt.create_query::<u64, _>(move |rt| rt.get(input) * 2);
        let _ = rt.get(query);
        let start = Instant::now();
        let mut acc = 0u64;
        for _ in 0..ITERS_HOT {
            acc = acc.wrapping_add(rt.get(query));
        }
        let elapsed = start.elapsed();
        black_box(acc);
        elapsed.as_nanos() as f64 / ITERS_HOT as f64
    };

    report("hot_clean_read_query_u64", v1_ns, v2_ns);
}

#[test]
#[ignore = "benchmark, run with --release"]
fn bench_propagate_chain_1000() {
    // Build a chain of 1000 queries where each adds 1 to its
    // predecessor. Set the input to a new value and read the tail.
    // Measures end-to-end propagation cost: dirty walk over 1000
    // nodes plus 1000 recomputes plus early-cutoff comparisons.
    //
    // This is the headline spec section 14 benchmark:
    //   "performance/propagate_single_change/1000: currently
    //    ~175 µs. Target: ≤175 µs."

    fn build_v1(size: usize) -> (V1Runtime, crate::Incr<i64>, crate::Incr<i64>) {
        let rt = V1Runtime::new();
        let input = rt.create_input::<i64>(1);
        let mut prev = input;
        for _ in 0..size {
            let dep = prev;
            prev = rt.create_query::<i64, _>(move |rt| rt.get(dep).wrapping_add(1));
        }
        let _ = rt.get(prev);
        (rt, input, prev)
    }

    fn build_v2(
        size: usize,
    ) -> (
        V2Runtime,
        crate::v2::handle::Incr<i64>,
        crate::v2::handle::Incr<i64>,
    ) {
        let rt = V2Runtime::new();
        let input = rt.create_input::<i64>(1);
        let mut prev = input;
        for _ in 0..size {
            let dep = prev;
            prev = rt.create_query::<i64, _>(move |rt| rt.get(dep).wrapping_add(1));
        }
        let _ = rt.get(prev);
        (rt, input, prev)
    }

    let v1_ns = {
        let (rt, input, output) = build_v1(1000);
        let mut val: i64 = 100;
        let start = Instant::now();
        for _ in 0..ITERS_PROPAGATE {
            val = val.wrapping_add(1);
            rt.set(input, val);
            black_box(rt.get(output));
        }
        let elapsed = start.elapsed();
        elapsed.as_nanos() as f64 / ITERS_PROPAGATE as f64
    };

    let v2_ns = {
        let (rt, input, output) = build_v2(1000);
        let mut val: i64 = 100;
        let start = Instant::now();
        for _ in 0..ITERS_PROPAGATE {
            val = val.wrapping_add(1);
            rt.set(input, val);
            black_box(rt.get(output));
        }
        let elapsed = start.elapsed();
        elapsed.as_nanos() as f64 / ITERS_PROPAGATE as f64
    };

    report("propagate_chain_1000", v1_ns, v2_ns);
}

#[test]
#[ignore = "benchmark, run with --release"]
fn bench_propagate_chain_1000_stable_intermediate() {
    // Red-green transitive early cutoff showcase. A "stabilizer" q0
    // reads the input but returns a constant; every downstream query
    // is a `+ 1` of its predecessor. On set(input), red-green should
    // short-circuit every level past q0 because q0's changed_at
    // never moves. v1 has the same mechanism; v2 should match it.

    fn build_v1(size: usize) -> (V1Runtime, crate::Incr<i64>, crate::Incr<i64>) {
        let rt = V1Runtime::new();
        let input = rt.create_input::<i64>(1);
        let q0 = rt.create_query::<i64, _>(move |rt| {
            let _ = rt.get(input);
            0
        });
        let mut prev = q0;
        for _ in 0..size {
            let dep = prev;
            prev = rt.create_query::<i64, _>(move |rt| rt.get(dep).wrapping_add(1));
        }
        let _ = rt.get(prev);
        (rt, input, prev)
    }

    fn build_v2(
        size: usize,
    ) -> (
        V2Runtime,
        crate::v2::handle::Incr<i64>,
        crate::v2::handle::Incr<i64>,
    ) {
        let rt = V2Runtime::new();
        let input = rt.create_input::<i64>(1);
        let q0 = rt.create_query::<i64, _>(move |rt| {
            let _ = rt.get(input);
            0
        });
        let mut prev = q0;
        for _ in 0..size {
            let dep = prev;
            prev = rt.create_query::<i64, _>(move |rt| rt.get(dep).wrapping_add(1));
        }
        let _ = rt.get(prev);
        (rt, input, prev)
    }

    let v1_ns = {
        let (rt, input, output) = build_v1(1000);
        let mut val: i64 = 100;
        let start = Instant::now();
        for _ in 0..ITERS_PROPAGATE {
            val = val.wrapping_add(1);
            rt.set(input, val);
            black_box(rt.get(output));
        }
        let elapsed = start.elapsed();
        elapsed.as_nanos() as f64 / ITERS_PROPAGATE as f64
    };

    let v2_ns = {
        let (rt, input, output) = build_v2(1000);
        let mut val: i64 = 100;
        let start = Instant::now();
        for _ in 0..ITERS_PROPAGATE {
            val = val.wrapping_add(1);
            rt.set(input, val);
            black_box(rt.get(output));
        }
        let elapsed = start.elapsed();
        elapsed.as_nanos() as f64 / ITERS_PROPAGATE as f64
    };

    report("propagate_chain_1000_stable_intermediate", v1_ns, v2_ns);
}

#[test]
#[ignore = "benchmark, run with --release"]
fn bench_set_input_only() {
    // Set an input with no dependents in a tight loop. Measures
    // set() cost without any dirty walk work: write mutex, arena
    // write, state Release, revision bump.

    let v1_ns = {
        let rt = V1Runtime::new();
        let input = rt.create_input::<u64>(0);
        let start = Instant::now();
        for i in 0..ITERS_HOT as u64 {
            rt.set(input, i);
        }
        let elapsed = start.elapsed();
        elapsed.as_nanos() as f64 / ITERS_HOT as f64
    };

    let v2_ns = {
        let rt = V2Runtime::new();
        let input = rt.create_input::<u64>(0);
        let start = Instant::now();
        for i in 0..ITERS_HOT as u64 {
            rt.set(input, i);
        }
        let elapsed = start.elapsed();
        elapsed.as_nanos() as f64 / ITERS_HOT as f64
    };

    report("set_input_no_dependents", v1_ns, v2_ns);
}

#[test]
#[ignore = "benchmark, run with --release"]
fn bench_cold_build_chain_1000() {
    // Build a chain of 1000 queries from scratch and read the
    // tail. Measures per-node creation cost plus the first
    // compute cascade.

    let v1_ns = {
        let start = Instant::now();
        let iters = 100;
        for _ in 0..iters {
            let rt = V1Runtime::new();
            let input = rt.create_input::<i64>(1);
            let mut prev = input;
            for _ in 0..1000 {
                let dep = prev;
                prev = rt.create_query::<i64, _>(move |rt| rt.get(dep).wrapping_add(1));
            }
            black_box(rt.get(prev));
        }
        let elapsed = start.elapsed();
        elapsed.as_nanos() as f64 / iters as f64
    };

    let v2_ns = {
        let start = Instant::now();
        let iters = 100;
        for _ in 0..iters {
            let rt = V2Runtime::new();
            let input = rt.create_input::<i64>(1);
            let mut prev = input;
            for _ in 0..1000 {
                let dep = prev;
                prev = rt.create_query::<i64, _>(move |rt| rt.get(dep).wrapping_add(1));
            }
            black_box(rt.get(prev));
        }
        let elapsed = start.elapsed();
        elapsed.as_nanos() as f64 / iters as f64
    };

    report("cold_build_chain_1000_total_ns", v1_ns, v2_ns);
    // Also report per-node to make it comparable with the hot
    // read numbers. 1000 nodes + 1 input per iteration.
    report(
        "cold_build_chain_1000_per_node",
        v1_ns / 1001.0,
        v2_ns / 1001.0,
    );
}
