# incr

 [![crates.io badge for incr-compute](https://img.shields.io/crates/v/incr-compute?label=incr-compute&logo=rust&color=blue)](https://crates.io/crates/incr-compute)
 [![crates.io badge for incr-concurrent](https://img.shields.io/crates/v/incr-concurrent?label=incr-concurrent&logo=rust&color=orange)](https://crates.io/crates/incr-concurrent)
 [![PyPI badge for incr-compute](https://img.shields.io/pypi/v/incr-compute?label=incr-compute&logo=python&color=blue)](https://pypi.org/project/incr-compute/)
 [![PyPI badge for incr-concurrent](https://img.shields.io/pypi/v/incr-concurrent?label=incr-concurrent&logo=python&color=orange)](https://pypi.org/project/incr-concurrent/)
[![CI](https://github.com/Anyesh/incr/workflows/CI/badge.svg)](https://github.com/Anyesh/incr/actions?query=workflow%3ACI)


Most software recomputes everything from scratch whenever anything changes. Your CI rebuilds the whole project when you edit one file, your dashboard re-queries the whole database when one row updates. There are domain-specific fixes for this (React diffs the DOM, Salsa caches compiler queries, Materialize does incremental SQL) but if you just want to make your own code incremental, theres nothing to reach for.

incr is a crack at solving that. It's a Rust library (with Python bindings on the roadmap) that tracks dependencies between computations automatically and only reruns what's actually affected by a change. The engine lives in `incr-core` and is parameterized over a concurrency strategy; two surface crates expose it: `incr-compute` (`Local` strategy, single-threaded, zero atomic-fence cost) and `incr-concurrent` (`Shared` strategy, `Send + Sync`, lock-free reads). Same public API, one-line dependency swap.

![Live spreadsheet demo showing formula cells updating incrementally as values change, powered by incr-concurrent with real-time WebSocket sync](examples/spreadsheet/demo.gif)

## Quick look

You've got two ways to use it. Function graphs let you wire up computations that depend on each other:

```rust
use incr_compute::Runtime;

let rt = Runtime::new();
let width = rt.create_input(10.0_f64);
let height = rt.create_input(5.0_f64);
let area = rt.create_query(move |rt| rt.get(width) * rt.get(height));

rt.get(area);          // 50.0
rt.set(width, 12.0);
rt.get(area);          // 60.0 — height wasn't touched, only area reran
```

And then theres incremental collections, which is where it gets more interesting. You set up a pipeline of operators, and when you insert or delete a row, only that row flows through. The engine doesnt re-examine existing data.

```rust
use incr_compute::{Runtime, IncrCollection};

let rt = Runtime::new();
let visits: IncrCollection<Visit> = rt.create_collection();
let sorted = visits.sort_by_key(&rt, |v| v.time);
let pairs = sorted.pairwise(&rt);
let gaps = pairs.map(&rt, |pair| distance(&pair.0, &pair.1));
let total = gaps.reduce(&rt, |xs| xs.iter().sum::<f64>());

visits.insert(&rt, visit_at_9am);
visits.insert(&rt, visit_at_2pm);
visits.insert(&rt, visit_at_11am);
rt.get(total);   // computes all distances

// Move one visit: only the two affected segments recompute.
visits.delete(&rt, &visit_at_11am);
visits.insert(&rt, visit_at_11am_moved_to_noon);
rt.get(total);
```

Nine operators ship: `filter`, `map`, `count`, `reduce`, `sort_by_key`, `pairwise`, `window`, `group_by`, `join`. The function-DAG API and the collection API share the same dependency graph, so a function query can read a collection's `reduce` and stay incremental end to end.

## Three crates, one engine

| | `incr-compute` | `incr-concurrent` | `incr-core` |
|---|---|---|---|
| Role | User-facing surface | User-facing surface | Shared engine |
| Thread safety | Single-threaded (`!Send + !Sync`) | `Send + Sync`, shareable across threads | Strategy-parameterized |
| Backing | `Cell`/`RefCell` (no atomics) | `Atomic*` + `RwLock` | `Cells` trait |
| Rust | `cargo add incr-compute` | `cargo add incr-concurrent` | (used via the wrappers) |

Both surface crates re-export `incr_core::Runtime<Local>` and `incr_core::Runtime<Shared>` respectively. The compiler monomorphizes both paths from the same source, so neither crate subsidizes the other: single-threaded users pay no atomic-fence cost; concurrent users pay no extra indirection for the lock-free read path. The asm of `Local`'s hot path is byte-identical to direct field access (validated on the spike branch and preserved through the type alias).

If you start with `incr-compute` and later need thread safety, swap the dependency. The `Value` bound (`Clone + PartialEq + Send + Sync + 'static`) is identical between crates, so user types do not need a per-crate impl.

## Benchmarks

Measured on this branch with `criterion --quick` against `Salsa` (the incremental engine in rust-analyzer). Not cherry-picked.

| Workload | `incr-compute` | `incr-concurrent` | Salsa |
|----------|----------------|---------------------|-------|
| Diamond graph, propagate input through 4 nodes | 647 ns | 764 ns | 1,066 ns |
| Early cutoff (input changes but clamped output doesnt) | 314 ns | 404 ns | 469 ns |
| Per-node propagation cost (chain) | ~135 ns/node | ~169 ns/node | ~387 ns/query |

Collection pipeline (`filter` → `map` → `count`) vs from-scratch batch:

| Collection size | `incr-compute` insert | From scratch | Speedup |
|----------------|----------------------|--------------|---------|
| 1K elements    | 673 ns               | 102 µs       | 152x    |
| 10K elements   | 657 ns               | 67 µs        | 102x    |
| 100K elements  | 661 ns               | 156 µs       | 236x    |

The interesting thing in that second table is the `incr-compute` column barely moves as the collection grows. 661 ns at 100K is essentially the same as 673 ns at 1K because we're only touching the new row, not scanning the existing ones.

## How it works internally

Calling `rt.set()` on an input eagerly marks downstream nodes as potentially dirty (just flipping bits, no recomputation). Then when you `rt.get()` a result, the engine walks backwards from what you asked for, checks if each dirty node's dependencies actually changed, and only reruns the ones that need it. If a node reruns but produces the same value it had before, propagation stops there, and thats the "early cutoff" you see in the benchmarks.

For collections each pipeline stage keeps a read offset into the upstream's change log. When triggered, it just reads entries past that offset, processes them, and advances the pointer. Inserting one row into a 100K collection means each stage does O(1) work regardless of collection size.

The engine itself lives in [`incr-core`](crates/incr-core/) under a `Cells` strategy trait. `Local` backs every cell with `std::cell::Cell` and gives you a `!Send + !Sync` runtime with no atomic ops. `Shared` backs every cell with the matching atomic type and uses Acquire/Release for state-visibility transitions. The trait inlines through every call site (`#[inline(always)]`), so the compiler emits the same code for `Runtime<Local>` operations as it would for direct `Cell::get()` calls. The 64-byte cache-line layout for `NodeData` is preserved under both strategies by const-time assertions.

## Getting started

**Rust:**

```toml
[dependencies]
incr-compute = "0.2"      # single-threaded
# or
incr-concurrent = "0.2"   # multi-threaded (Send + Sync)
```

**Running the tests:**

```bash
cargo test -p incr-core             # full engine: 100+ tests with proptest
cargo test -p incr-compute          # single-threaded wrapper integration
cargo test -p incr-concurrent       # concurrent wrapper integration

cargo bench -p incr-core            # full engine benches
cargo bench -p incr-compute         # bench through the wrapper
```

## Testing

100+ unit/integration tests across the engine and wrappers, plus a proptest suite that runs **the same generator + verifier under both strategies**: 1000 random function graphs and 3000 random collection op sequences per `cargo test` run. The core correctness contract is that incremental evaluation produces the same final result as recomputing everything from scratch; if those two ever disagree on any random input, proptest shrinks to a minimal failing case.

A concurrent stress test runs 4 reader threads against 1 writer thread for 1000 iterations and asserts no torn reads on derived values.

The unsafe code (segmented store, raw-pointer dep reclamation, hazard-free state machine CAS) is exercised under `cargo +nightly miri test -p incr-core --lib`. 79 unit tests pass under miri including 16-thread CAS races (3,200 concurrent attempts) and 50-iteration dynamic-dep-set churn through the overflow path with runtime drop. Zero undefined behavior detected.

## Demos

- [`examples/concurrent-server/`](examples/concurrent-server/) — multi-threaded HTTP server (Rust) where one writer thread feeds live market data into the graph while many HTTP handler threads read derived portfolio values concurrently without blocking.
- [`examples/spreadsheet/`](examples/spreadsheet/) — live spreadsheet engine driving formula cells through the incremental graph with WebSocket sync.

A Python `travel-premium` demo and a `dashboard` demo with real per-node tracing live on the v0.1 line; they are scheduled to land on v0.2 alongside the Python re-implementation in 0.3.

## CI

GitHub Actions runs on every push to `main` and on pull requests: tests for all three Rust crates, benchmarks, clippy, and fmt. Tagging a release (`v*`) triggers automatic publishing to crates.io. Python wheels return in 0.3.

## Background and references

The core theory goes back to Umut Acar's PhD thesis on [self-adjusting computation](https://www.cs.cmu.edu/~rwh/students/acar.pdf) at Carnegie Mellon around 2005. He showed you can take arbitrary functional programs, track their dependencies at runtime, and replay only the parts affected by a change. The problem was overhead: the initial run was 2-30x slower, and memory usage exploded because you had to keep the whole dependency graph around. Nobody really figured out how to make it practical at the time.

A few years later, [Adapton](https://dl.acm.org/doi/10.1145/2594291.2594324) (PLDI 2014) introduced demand-driven incremental computation, where the key idea is that you dont recompute eagerly when inputs change, you just mark things dirty and only recompute when someone asks for a result. That's the approach we use for the function DAG side.

On the collections side, the big influence is Frank McSherry's [Differential Dataflow](https://www.cidrdb.org/cidr2013/Papers/CIDR13_Paper111.pdf) (CIDR 2013), which represents collections as streams of differences (+1/-1 for inserts/deletes) and propagates those differences through operators. Materialize built a whole company around this for SQL. Our delta-log approach is a simplified version of the same idea.

The systems we benchmark against and learned from:

- [Salsa](https://salsa-rs.github.io/salsa/) powers rust-analyzer's incremental analysis. It uses a "red-green" algorithm with dual timestamps for early cutoff, which we borrowed; our `verified_at` / `changed_at` design comes directly from studying how Salsa works.
- [Jane Street's Incremental](https://blog.janestreet.com/introducing-incremental/) is an OCaml library that went through [seven implementations](https://www.janestreet.com/tech-talks/seven-implementations-of-incremental/) before they got it right. Their ~30ns per-node firing cost was our original performance target.
- [Build Systems a la Carte](https://www.cambridge.org/core/journals/journal-of-functional-programming/article/build-systems-a-la-carte-theory-and-practice/097CE52C750E69BD16B78C318754C7A4) (Mokhov, Mitchell, Peyton Jones, JFP 2020) provided the theoretical framework showing that build systems and incremental computation are the same problem viewed from different angles.

Y. Annie Liu's 2024 survey [Incremental Computation: What Is the Essence?](https://arxiv.org/abs/2312.07946) is probably the best current overview of the whole field if you want to understand where all these approaches fit relative to each other. One of her key findings is that fully general incrementalization is provably undecidable, which is why every practical system (including ours) picks a restricted but useful subset of computations to handle.

None of the existing systems combine function DAGs with incremental collections in a single engine across single-threaded and concurrent topologies the way incr does. Early results across the function DAG and the operator pipeline both beat the published numbers for Salsa and the v0.1 line; the architecture is what made that possible.

## License

Apache-2.0
