# incr

A general-purpose incremental computation engine. You write normal computations, and incr figures out what to recompute when inputs change. Only the parts that are actually affected get rerun.

Works from both Rust and Python.

## Why this exists

Every software system recomputes too much. When one cell changes in a spreadsheet, the whole sheet recalculates. When one file changes in CI, the whole project rebuilds. There are domain-specific solutions to this (React for UI, Salsa for compilers, Materialize for SQL) but nothing general purpose that you can just drop into your code.

incr is an attempt at that general purpose thing. Its built on the same academic foundations as those systems (Acar's self-adjusting computation, Adapton's demand-driven evaluation) but packages them into a library that works with arbitrary computations, not just one domain.

## What it does

Two APIs that share one engine:

**Function graphs** -- define inputs and derived computations. When an input changes, only affected downstream nodes recompute.

```python
from incr import Runtime

rt = Runtime()
width = rt.create_input(10.0)
height = rt.create_input(5.0)
area = rt.create_query(lambda rt: rt.get(width) * rt.get(height))

rt.get(area)  # 50.0
rt.set(width, 12.0)
rt.get(area)  # 60.0 -- only area recomputed, not height
```

**Incremental collections** -- filter, map, count over datasets. When you insert or delete a row, only the new row flows through the pipeline.

```python
scores = rt.create_collection()
high_scores = scores.filter(lambda x: x >= 90)
count = high_scores.count()

scores.insert(85)
scores.insert(92)
scores.insert(95)
rt.get(count)  # 2

scores.insert(91)
rt.get(count)  # 3 -- only the new row was checked
```

These compose. A function query can depend on a collection count, and the whole thing stays incremental end to end.

## Performance

Benchmarked on the same machine, same workloads, head to head against Salsa (the incremental engine used by rust-analyzer):

| Workload | incr | Salsa |
|----------|------|-------|
| Diamond graph (4 nodes), change input and propagate | 752 ns | 1,066 ns |
| Early cutoff (input changes but output doesnt) | 445 ns | 469 ns |
| Single tracked query, change and recompute | ~175 ns/node | ~387 ns/query |

For collections, compared against recomputing the whole pipeline from scratch:

| Collection size | Incremental insert | Batch recompute | Speedup |
|----------------|-------------------|----------------|---------|
| 1,000 elements | 798 ns | 2.5 us | 3x |
| 10,000 elements | 1.0 us | 14.2 us | 14x |
| 100,000 elements | 818 ns | 152 us | 186x |

The insert time stays roughly constant regardless of how big the collection is. Thats the whole point.

## How it works

When you call `rt.set()` on an input, incr marks all downstream nodes as "maybe dirty". When you call `rt.get()` on a result, it walks backwards from what you asked for, only recomputing nodes whose inputs actually changed. If a node recomputes but produces the same value as before (early cutoff), propagation stops there.

Collections use delta-based propagation. Each pipeline stage (filter, map, count) tracks an index into the upstream change log and only processes new entries since last time. So inserting one row into a 100K-element collection touches maybe 3-4 pipeline stages, each doing O(1) work.

## Correctness

Every release runs 4000+ property-based test cases (proptest) that generate random computation graphs, apply random mutations, and verify the incremental result always matches computing everything from scratch. If these ever disagree, proptest gives you a minimal failing example.

50 Rust tests, 15 Python tests, covering function graphs, collections, early cutoff, dynamic dependencies, and cross-API composition.

## Install

**Python:**
```
pip install incr
```
(not on PyPI yet -- build from source with `maturin develop`)

**Rust:**
```toml
[dependencies]
incr-core = { path = "crates/incr-core" }
```

## Building from source

You need Rust and (for Python bindings) uv or any Python 3.9+.

```bash
cd incr
cargo test -p incr-core          # run rust tests
cargo bench -p incr-core          # run benchmarks

# for python
uv venv ../.venv
uv pip install maturin pytest
source ../.venv/Scripts/activate  # or bin/activate on linux/mac
maturin develop
pytest ../tests/python/
```

## Status

This is early stage. The core engine works and is fast, but the API will probably change. No stability guarantees yet.

What works:
- Function DAG with automatic dependency tracking
- Early cutoff when values dont change
- Dynamic dependencies (a node can read different inputs depending on values)
- Incremental collections with delta-based filter, map, count
- Python bindings via PyO3
- Competitive with Salsa on equivalent workloads

Whats missing:
- Collection joins and group-by (planned)
- Memory management / garbage collection for long-running processes
- Multi-threaded propagation
- Published to crates.io / PyPI

## Prior art

The theory behind this comes from Umut Acar's self-adjusting computation work at CMU (~2005). The practical design borrows from Salsa (rust-analyzer's incremental engine), Adapton (demand-driven evaluation), and Differential Dataflow (delta propagation for collections). None of those systems combine function DAGs with incremental collections in a single engine, which is what incr tries to do.
