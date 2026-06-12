# incr-concurrent

Thread-safe incremental computation with `Send + Sync` runtime. Since 0.2, this crate is a thin re-export of [`incr-core`](https://crates.io/crates/incr-core) with the `Shared` strategy; the algorithm and operators live in the shared engine.

`incr-concurrent` builds a reactive computation graph that can be shared across threads. One thread mutates inputs while any number of reader threads query derived values concurrently. State transitions are CAS-based with Acquire/Release ordering, and values live behind hazard-pointer-protected pointer swaps, so a read can never observe a torn value and the displaced value cannot be freed under a reader: this holds for heap values (String, Vec, your structs), not just machine words. The price relative to `incr-compute` is a pair of hazard-pointer atomics per value read and one allocation plus a deferred free per changed value write; the in-repo benchmarks quantify it (roughly 3x `incr-compute` on chain propagation, at parity with Salsa, which does not allow concurrent readers at all).

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

Same ten as `incr-compute`: filter, map, count, aggregate, reduce, sort_by_key, pairwise, window, group_by, join. `count` and `aggregate` are incremental (O(1) and O(log n) per delta); `reduce` is snapshot-based for arbitrary folds; everything else is incremental on the delta log. `delete_node` and `observe`/`stabilize` are available here too.

## When to use

Use `incr-concurrent` when you need to share a computation graph across threads. If everything runs on a single thread, use [`incr-compute`](https://crates.io/crates/incr-compute) instead for the slightly faster uncontended path.

## Python

`pip install incr-concurrent` ships the same engine as an abi3 wheel for CPython 3.10+ (`from incr_concurrent import Runtime`); its classes are shareable across Python threads and the GIL is released around computing calls.
