# incr-compute

Single-threaded, zero-overhead incremental computation. Since 0.2, this crate is a thin re-export of [`incr-core`](https://crates.io/crates/incr-core) with the `Local` strategy; the algorithm and operators live in the shared engine and monomorphize through this wrapper without adding any runtime cost.

`incr-compute` builds a reactive computation graph where derived values automatically recompute when their inputs change. It only recomputes what actually needs to change: if an intermediate result stays the same after an input mutation, everything downstream is skipped entirely (early cutoff). The `Runtime` is `!Send + !Sync` and pays no atomic-fence cost on its hot path; under the hood every cell is `std::cell::Cell`.

## Install

```
cargo add incr-compute
```

## Quick start

```rust
use incr_compute::Runtime;

let rt = Runtime::new();

let width = rt.create_input(3);
let height = rt.create_input(7);
let area = rt.create_query(move |rt| rt.get(width) * rt.get(height));

assert_eq!(rt.get(area), 21);

rt.set(width, 10);
assert_eq!(rt.get(area), 70);
```

Dependencies are tracked automatically. When you call `rt.get(width)` inside a query's closure, the runtime records that the query depends on `width`. No manual wiring needed.

## Collections

```rust
use incr_compute::{IncrCollection, Runtime};

let rt = Runtime::new();
let scores: IncrCollection<i64> = rt.create_collection();

scores.insert(&rt, 80);
scores.insert(&rt, 95);
scores.insert(&rt, 60);
scores.insert(&rt, 42);

let passing = scores.filter(&rt, |s| *s >= 50);
let curved = passing.map(&rt, |s| s + 10);
let total = curved.reduce(&rt, |vals| vals.iter().sum::<i64>());

assert_eq!(rt.get(total), 265); // (80+10) + (95+10) + (60+10)

scores.insert(&rt, 30); // filtered out, total unchanged
assert_eq!(rt.get(total), 265);
```

## All operators

- **filter** keeps elements matching a predicate
- **map** transforms each element
- **count** tracks the number of elements (incremental, O(1) per insert/delete)
- **aggregate** incrementally maintained monoid fold over a balanced tree (O(log n) per change, correct for non-invertible folds like max)
- **reduce** folds all elements via an arbitrary function over a snapshot (O(n) per change by construction; prefer `aggregate` for associative folds)
- **sort_by_key** produces a sorted view with positional deltas
- **pairwise** emits consecutive pairs from a sorted collection (O(1) output deltas per change)
- **window** emits sliding windows of a given size from a sorted collection
- **group_by** partitions into per-key sub-collections
- **join** pairs two collections on a shared key

Beyond operators: `delete_node` removes nodes and recycles their slots (stale handles panic with a clear message), and `observe`/`stabilize` deliver batch change notifications without polling.

## When to use incr-compute vs incr-concurrent

If your computation lives on a single thread, use `incr-compute`. It has zero synchronization overhead and is the fastest option.

If you need to share one computation graph across multiple threads (for example, a writer thread updating inputs while reader threads query derived values), use [`incr-concurrent`](https://crates.io/crates/incr-concurrent) instead. The API is identical: switching is a one-line import change.

## Value bound

User types stored in the runtime must implement `Value`, which is `Clone + PartialEq + Send + Sync + 'static`. A blanket impl auto-derives `Value` for every qualifying type, so most user types need no explicit impl. The same bound applies in `incr-concurrent`, so types compile cleanly under both crates.

## Python

`pip install incr-compute` ships the same engine as an abi3 wheel for CPython 3.10+ (`from incr import Runtime`).
