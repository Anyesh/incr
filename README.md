# incr
 [![crates.io badge for incr-compute](https://img.shields.io/crates/v/incr-compute?label=incr-compute&logo=rust&color=blue)](https://crates.io/crates/incr-compute)
 [![crates.io badge for incr-concurrent](https://img.shields.io/crates/v/incr-concurrent?label=incr-concurrent&logo=rust&color=orange)](https://crates.io/crates/incr-concurrent)
 [![PyPI badge for incr-compute](https://img.shields.io/pypi/v/incr-compute?label=incr-compute&logo=python&color=blue)](https://pypi.org/project/incr-compute/)
 [![PyPI badge for incr-concurrent](https://img.shields.io/pypi/v/incr-concurrent?label=incr-concurrent&logo=python&color=orange)](https://pypi.org/project/incr-concurrent/)
[![CI](https://github.com/Anyesh/incr/workflows/CI/badge.svg)](https://github.com/Anyesh/incr/actions?query=workflow%3ACI)


Most software recomputes everything from scratch whenever anything changes. Your CI rebuilds the whole project when you edit one file, your dashboard re-queries the whole database when one row updates. There are domain-specific fixes for this (React diffs the DOM, Salsa caches compiler queries, Materialize does incremental SQL) but if you just want to make your own code incremental, theres nothing to reach for.

incr is a crack at solving that. Its a Rust library (with Python bindings) that tracks dependencies between computations automatically and only reruns what's actually affected by a change. It ships as two crates: `incr-compute` for single-threaded, zero-overhead use, and `incr-concurrent` for multi-threaded programs where the runtime needs to be `Send + Sync`. Both are published on crates.io and PyPI, and they expose the same API surface, so switching between them is a one-line dependency swap.

![Live spreadsheet demo showing formula cells updating incrementally as values change, powered by incr-concurrent with real-time WebSocket sync](examples/spreadsheet/demo.gif)

## Quick look

You've got two ways to use it. Function graphs let you wire up computations that depend on each other:

```python
from incr import Runtime

rt = Runtime()
width = rt.create_input(10.0)
height = rt.create_input(5.0)
area = rt.create_query(lambda rt: rt.get(width) * rt.get(height))

rt.get(area)  # 50.0
rt.set(width, 12.0)
rt.get(area)  # 60.0, height wasnt touched, only area reran
```

And then theres incremental collections, which is where it gets more interesting. You set up a pipeline of operators, and when you insert or delete a row, only that row flows through. The engine doesnt re-examine existing data.

```python
# Travel premium calculation: sort visits by time, compute gaps
# between consecutive visits, sum the premiums
visits = rt.create_collection()
sorted_visits = visits.sort_by_key(lambda v: v.time)
pairs = sorted_visits.pairwise()
gaps = pairs.map(lambda pair: distance(pair[0], pair[1]))
total = gaps.reduce(lambda elements: sum(elements))

visits.insert(visit_at_9am)
visits.insert(visit_at_2pm)
visits.insert(visit_at_11am)
rt.get(total)  # computes all distances

# Move one visit: only the two affected segments recompute
visits.delete(visit_at_11am)
visits.insert(visit_at_11am_moved_to_noon)
rt.get(total)  # only recomputes 2 of 3 distances
```

The pipeline supports filter, map, count, reduce, sort_by_key, pairwise, group_by, join, and window. The two APIs (function DAG and collections) share the same dependency graph under the hood so you can have a function query that reads from a collection's reduce and it all stays incremental.

## Two crates, one API

| | `incr-compute` | `incr-concurrent` |
|---|---|---|
| Thread safety | Single-threaded (`!Send`, `!Sync`) | `Send + Sync`, safe to share across threads |
| Overhead | Zero runtime cost for thread safety | Atomic operations for node state transitions |
| When to use | Scripts, CLI tools, single-threaded services | HTTP servers, background workers, anything multi-threaded |
| Rust | `cargo add incr-compute` | `cargo add incr-concurrent` |
| Python | `pip install incr-compute` | `pip install incr-concurrent` |
| Python import | `from incr import Runtime` | `from incr_concurrent import Runtime` |

The API is identical between the two. If you start with `incr-compute` and later need thread safety, swap the dependency and everything compiles without changes.

## Benchmarks

We run these head-to-head against Salsa (the incremental engine in rust-analyzer) on the same machine, same workloads. Not cherry-picked.

| Workload | incr | Salsa |
|----------|------|-------|
| Diamond graph, change input and propagate through 4 nodes | 752 ns | 1,066 ns |
| Early cutoff (input changes but clamped output doesnt) | 445 ns | 469 ns |
| Per-node propagation cost in a chain | ~175 ns/node | ~387 ns/query |

Collection insert vs just recomputing the whole pipeline from scratch:

| Collection size | Incremental | From scratch | Speedup |
|----------------|-------------|-------------|---------|
| 1K elements | 798 ns | 2.5 us | 3x |
| 10K elements | 1.0 us | 14.2 us | 14x |
| 100K elements | 818 ns | 152 us | 186x |

The interesting thing in that second table is the incremental column barely moves as the collection grows. 818 ns for 100K is almost the same as 798 ns for 1K because we're only touching the new row, not scanning the existing ones.

## How it works internally

Calling `rt.set()` on an input eagerly marks downstream nodes as potentially dirty (just flipping bits, no recomputation). Then when you `rt.get()` a result, the engine walks backwards from what you asked for, checks if each dirty node's dependencies actually changed, and only reruns the ones that need it. If a node reruns but produces the same value it had before, propagation stops there, and thats the "early cutoff" you see in the benchmarks.

For collections its a bit different. Each pipeline stage keeps a read offset into the upstream's change log. When triggered, it just reads entries past that offset, processes them, and advances the pointer. Inserting one row into a 100K collection means each stage does O(1) work regardless of collection size.

## Getting started

**Rust:**

```toml
[dependencies]
incr-compute = "0.1"      # single-threaded
# or
incr-concurrent = "0.1"   # multi-threaded (Send + Sync)
```

**Python:**

```bash
pip install incr-compute        # single-threaded
# or
pip install incr-concurrent     # multi-threaded
```

```python
from incr import Runtime              # incr-compute
# or
from incr_concurrent import Runtime   # incr-concurrent
```

**Running the tests:**

```bash
cargo test -p incr-compute         # single-threaded crate
cargo test -p incr-concurrent      # concurrent crate
pytest ./examples/tests/python/    # python bindings
cargo bench -p incr-compute        # benchmarks (single-threaded)
cargo bench -p incr-concurrent     # benchmarks (concurrent)
```

## Testing

300+ tests across both Rust crates (unit, property, and integration), plus a Python test suite for the bindings. We use proptest to generate thousands of random computation graphs, apply random mutations, and check that the incremental result matches what you'd get by recomputing everything from scratch. Thats the core correctness guarantee: if those two ever disagree on any random input, proptest shrinks it down to a minimal failing case.

The property test suites cover every operator (filter, map, count, reduce, sort_by_key, pairwise, group_by, join, window) in both crates, verifying that incremental evaluation produces the same result as full recomputation across thousands of randomly generated scenarios.

## Demos

Three demos show different aspects of the library:

- **`examples/travel-premium/`** is a mobile worker scheduling demo (Python) that computes travel premiums incrementally using the full operator pipeline (sort, pairwise, map, reduce). It's backed by SQLite for persistence, with a distance cache that survives server restarts, and shows 5-8x speedup over batch recomputation when the map step involves expensive operations like distance lookups.
- **`examples/dashboard/`** is a live API monitoring dashboard (Python) with dependency graph visualization and real-time tracing of which nodes recompute vs get skipped.
- **`examples/concurrent-server/`** is a multi-threaded HTTP server (Rust) that proves the concurrent access model: one writer thread feeds live market data into an incr graph while multiple HTTP handler threads read derived portfolio values simultaneously without blocking.

## CI

GitHub Actions runs on every push to main and on pull requests: tests for both crates, Python binding builds, benchmarks, clippy, and fmt. Tagging a release (`v*`) triggers automatic publishing to both crates.io and PyPI.

## Background and references

The core theory goes back to Umut Acar's PhD thesis on [self-adjusting computation](https://www.cs.cmu.edu/~rwh/students/acar.pdf) at Carnegie Mellon around 2005. He showed you can take arbitrary functional programs, track their dependencies at runtime, and replay only the parts affected by a change. The problem was overhead: the initial run was 2-30x slower, and memory usage exploded because you had to keep the whole dependency graph around. Nobody really figured out how to make it practical at the time.

A few years later, [Adapton](https://dl.acm.org/doi/10.1145/2594291.2594324) (PLDI 2014) introduced demand-driven incremental computation, where the key idea is that you dont recompute eagerly when inputs change, you just mark things dirty and only recompute when someone asks for a result. That's the approach we use for the function DAG side.

On the collections side, the big influence is Frank McSherry's [Differential Dataflow](https://www.cidrdb.org/cidr2013/Papers/CIDR13_Paper111.pdf) (CIDR 2013), which represents collections as streams of differences (+1/-1 for inserts/deletes) and propagates those differences through operators. Materialize built a whole company around this for SQL. Our delta-log approach is a simplified version of the same idea.

The systems we benchmark against and learned from:

- [Salsa](https://salsa-rs.github.io/salsa/) powers rust-analyzer's incremental analysis. It uses a "red-green" algorithm with dual timestamps for early cutoff, which we borrowed; our `verified_at` / `changed_at` design comes directly from studying how Salsa works.
- [Jane Street's Incremental](https://blog.janestreet.com/introducing-incremental/) is an OCaml library that went through [seven implementations](https://www.janestreet.com/tech-talks/seven-implementations-of-incremental/) before they got it right. Their ~30ns per-node firing cost was our original performance target.
- [Build Systems a la Carte](https://www.cambridge.org/core/journals/journal-of-functional-programming/article/build-systems-a-la-carte-theory-and-practice/097CE52C750E69BD16B78C318754C7A4) (Mokhov, Mitchell, Peyton Jones, JFP 2020) provided the theoretical framework showing that build systems and incremental computation are the same problem viewed from different angles.

Y. Annie Liu's 2024 survey [Incremental Computation: What Is the Essence?](https://arxiv.org/abs/2312.07946) is probably the best current overview of the whole field if you want to understand where all these approaches fit relative to each other. One of her key findings is that fully general incrementalization is provably undecidable, which is why every practical system (including ours) picks a restricted but useful subset of computations to handle.

None of the existing systems combine function DAGs with incremental collections in a single engine, which is what incr tries to do. Whether that actually works out as a general purpose tool is still an open question, but the early results are encouraging.

## License

Apache-2.0
