# incr-concurrent

Thread-safe incremental computation with `Send + Sync` runtime. Since 0.2, this crate is a thin re-export of [`incr-core`](https://crates.io/crates/incr-core) with the `Shared` strategy; the algorithm and operators live in the shared engine.

`incr-concurrent` builds a reactive computation graph that can be shared across threads. One thread mutates inputs while any number of reader threads query derived values concurrently. Under the hood every cell is the matching atomic type and state transitions use explicit Acquire/Release for visibility. On x86 (TSO) Acquire compiles to a plain `mov` with no fences, so the lock-free read path costs essentially nothing over the single-threaded variant. ARM/Apple Silicon pays one `dmb ld` per Acquire load, which is the unavoidable cost of cross-thread synchronization on a weak memory model.

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

The API is identical to `incr-compute`. Dependencies are tracked automatically when your query closure calls `rt.get`.

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

Incremental collections work the same way as in `incr-compute`, and the entire pipeline is `Send + Sync`.

```rust
use incr_concurrent::{IncrCollection, Runtime};

let rt = Runtime::new();
let scores: IncrCollection<i64> = rt.create_collection();

scores.insert(&rt, 80);
scores.insert(&rt, 95);
scores.insert(&rt, 60);
scores.insert(&rt, 42);

let passing = scores.filter(&rt, |s| *s >= 50);
let curved = passing.map(&rt, |s| s + 10);
let total = curved.reduce(&rt, |vals| vals.iter().sum::<i64>());

assert_eq!(rt.get(total), 265);
```

## All operators

Same nine as `incr-compute`: filter, map, count, reduce, sort_by_key, pairwise, window, group_by, join. The `count` operator is incremental (O(1) per delta); `reduce` is snapshot-based; everything else is incremental on the delta log.

## When to use

Use `incr-concurrent` when you need to share a computation graph across threads. If everything runs on a single thread, use [`incr-compute`](https://crates.io/crates/incr-compute) instead for the slightly faster uncontended path.

## Python

Python bindings re-implement against the v0.2 engine in 0.3.
