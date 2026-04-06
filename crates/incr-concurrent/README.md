# incr-concurrent

Thread-safe incremental computation with `Send + Sync` runtime.

`incr-concurrent` builds a reactive computation graph that can be shared across threads. One thread mutates inputs while any number of reader threads query derived values concurrently, with no contention on the reader path. Like `incr`, it only recomputes what actually changed and applies early cutoff to skip unnecessary downstream work. The tradeoff is roughly 1.6x slower single-threaded throughput in exchange for safe concurrent access.

## Install

```
cargo add incr-concurrent
```

## Quick start

```rust
use incr_concurrent::Runtime;

let rt = Runtime::new();

let width = rt.create_input(3);
let height = rt.create_input(7);
let area = rt.create_query(move |rt| rt.get(width) * rt.get(height));

assert_eq!(rt.get(area), 21);

rt.set(width, 10);
assert_eq!(rt.get(area), 70);
```

The API is identical to `incr`. Dependencies are tracked automatically when your query closure calls `rt.get`.

## Concurrent access

Wrap the runtime in an `Arc` and share it across threads. Writers call `rt.set`, readers call `rt.get`, and the runtime handles synchronization internally.

```rust
use incr_concurrent::Runtime;
use std::sync::Arc;
use std::thread;

let rt = Arc::new(Runtime::new());

let counter = rt.create_input(0_i64);
let doubled = rt.create_query(move |rt| rt.get(counter) * 2);

let writer = {
    let rt = Arc::clone(&rt);
    thread::spawn(move || {
        for i in 1..=100 {
            rt.set(counter, i);
        }
    })
};

let reader = {
    let rt = Arc::clone(&rt);
    thread::spawn(move || {
        for _ in 0..200 {
            let val = rt.get(doubled);
            assert!(val % 2 == 0); // always even, never a torn read
        }
    })
};

writer.join().unwrap();
reader.join().unwrap();
```

## Collections

Incremental collections work the same way as in `incr`, and the entire pipeline is `Send + Sync`.

```rust
use incr_concurrent::{Runtime, IncrCollection};

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

## When to use

Use `incr-concurrent` when you need to share a computation graph across threads. If everything runs on a single thread, use [`incr`](https://crates.io/crates/incr) instead for better raw throughput.

## Python

```
pip install incr-concurrent
```
