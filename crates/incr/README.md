# incr

Single-threaded, zero-overhead incremental computation.

`incr` builds a reactive computation graph where derived values automatically recompute when their inputs change. It only recomputes what actually needs to change: if an intermediate result stays the same after an input mutation, everything downstream is skipped entirely (early cutoff). This makes it fast enough to sit in a hot loop without thinking about it.

## Install

```
cargo add incr
```

## Quick start

```rust
use incr::Runtime;

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

Incremental collections let you build data pipelines that update incrementally as elements are inserted or removed.

```rust
use incr::{Runtime, IncrCollection};

let rt = Runtime::new();
let scores = rt.create_collection::<i64>();

scores.insert(&rt, 80);
scores.insert(&rt, 95);
scores.insert(&rt, 60);
scores.insert(&rt, 42);

let passing = scores.filter(&rt, |s| *s >= 50);
let curved = passing.map(&rt, |s| s + 10);
let total = curved.reduce(&rt, |vals| vals.iter().sum::<i64>());

assert_eq!(rt.get(total), 255); // (80+10) + (95+10) + (60+10)

scores.insert(&rt, 30); // filtered out, total unchanged
assert_eq!(rt.get(total), 255);
```

## All operators

- **filter** keeps elements matching a predicate
- **map** transforms each element
- **count** tracks the number of elements
- **reduce** folds all elements into a single value
- **sort_by_key** produces a sorted view with positional deltas
- **pairwise** emits consecutive pairs from a sorted collection
- **group_by** partitions into keyed sub-collections
- **join** pairs two collections on a shared key
- **window** emits sliding windows of a given size from a sorted collection

## When to use incr vs incr-concurrent

If your computation lives on a single thread, use `incr`. It has zero synchronization overhead and is the fastest option.

If you need to share one computation graph across multiple threads (for example, a writer thread updating inputs while reader threads query derived values), use [`incr-concurrent`](https://crates.io/crates/incr-concurrent) instead. The API is identical, so switching is a one-line import change.

## Python

```
pip install incr
```
