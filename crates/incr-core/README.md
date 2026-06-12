# incr-core

Shared engine behind [`incr-compute`](https://crates.io/crates/incr-compute) and [`incr-concurrent`](https://crates.io/crates/incr-concurrent). Strategy-parameterized: the same `Runtime<C: Cells>` monomorphizes into the single-threaded variant (`Cell`-backed) when `C = Local` and the concurrent variant (atomic-backed) when `C = Shared`.

Most users should depend on one of the surface crates, not this one. Use `incr-core` directly only if:

- You want to build your own concurrency strategy on top of the `Cells` trait, or
- You're embedding the engine in a place where the wrapper crates' default choices don't fit (e.g., a custom `no_std` strategy).

## Architecture

The engine is built around the `Cells` strategy trait:

```rust
pub trait Cells: 'static + Sized {
    type U8;
    type U16;
    type U32;
    type U64;
    type State;
    type Ptr<T: 'static>: PtrCell<T>;
    type Lock<T: 'static>: Lock<T>;
    type DepStack: DepStack;
    type ValueSlot<T: Value>: ValueSlot<T>;

    // ... constructors and inline-only load/store/CAS/fetch helpers
}
```

All trait methods are `#[inline(always)]` and take `&Self::Cell` references, so the compiler can see through every call site. The validation that this carries zero overhead on the single-threaded path lives in the spike branch's RESULTS.md (`spike/incr-core-monomorphization`): `walk_local` and a hand-written non-trait baseline produce **byte-identical assembly**.

Two strategy impls ship in this crate:

- `Local`: `Cell`-backed scalars, `RefCell` locks, in-place `UnsafeCell` value slots. `!Send + !Sync` (correct for the single-threaded variant), no atomics anywhere on its paths.
- `Shared`: atomic scalars, `RwLock` locks, hazard-pointer-protected pointer-swap value slots (via [`haphazard`](https://crates.io/crates/haphazard)). `Send + Sync`; every state transition is a CAS, so concurrent reads can never tear and a node is only ever computed by one thread at a time.

## What's exposed

- `Runtime<C: Cells>` with `create_input`, `create_query`, `get`, `set`, `delete_node`, `observe` / `unobserve` / `stabilize`, `node_count`, `graph_snapshot`, `get_traced`, `set_label` / `label`.
- `Incr<T>`: 16-byte `Copy` handle with embedded `RuntimeId` and a generation counter, so handles from foreign runtimes and handles to deleted nodes are rejected with a clear panic instead of corrupting state.
- `IncrCollection<T, C>`, `GroupedCollection<K, T, C>`, `SortedCollection<T, K, C>` with the full operator suite (filter, map, count, aggregate, reduce, sort_by_key, pairwise, window, group_by, join). Operator outputs are derived collections and reject direct mutation.
- `Value` blanket trait (`Clone + PartialEq + Send + Sync + 'static`) — auto-derived for every qualifying type.
- Tracing types: `NodeInfo`, `NodeKindInfo`, `NodeTrace`, `TraceAction`, `PropagationTrace`.

## Layout invariants

`NodeData<C>` is exactly 64 bytes and 64-byte aligned under both strategies. `const _: () = assert!(...)` blocks enforce this at compile time; layout drift breaks the build immediately.

The segmented node store supports up to 1M nodes per runtime (1024 segments × 1024 slots). Segments are lazily allocated, never moved, and live until the runtime drops.

## Known limitations

- **`get_traced` per-node trace**: records compute, verified-clean, and cutoff events for the current `get` call's compute path. Cross-thread events are not aggregated.
- **Cross-thread dependency cycles livelock**: a cycle confined to one thread's compute stack panics with a diagnostic, but a cycle spanning two threads' in-flight computes spins in the wait loop; detecting that needs a waits-for graph, which is not implemented.
- **`Hash`/`Eq`/`Ord` panics during operator apply phases** are outside the exactly-once replay guarantee that covers user closures (predicates, mappers, key extractors, folds).

## Memory reclamation

Two structures go through the [`haphazard`](https://crates.io/crates/haphazard) global hazard-pointer domain: overflow-dep lists (the heap allocation a node holds when it has more than 7 dependencies) and, under `Shared`, every node's value slot, which writers replace by atomic pointer swap. Concurrent readers hold a `HazardPointer` while dereferencing either, so a writer's retire is deferred until no protecting reader remains. Deleted nodes recycle their slot and arena storage under a bumped generation, and collection delta logs compact behind their consumers' cursors, so long-lived runtimes are bounded by live state, not by history.

## Soundness

All unsafe code in this crate (the value slots, the segmented node store's `UnsafeCell + MaybeUninit` slots, hazard-pointer reclamation, `ArenaRegistry`'s `Arc` downcast via raw-pointer rewrap, the `SharedDepStack` thread-local) is exercised under `cargo +nightly miri test -p incr-core --lib --test stress`, and Miri's race detector runs the threaded suites: reader threads cloning heap values while writers swap them, claim races on first computes, concurrent setters, and a mixed workload with node churn through slot recycling. The same suites run under ThreadSanitizer in CI. A panic-injection suite asserts that a panicking compute closure lands its node in `Failed`, the runtime stays usable, and recovery replays exactly once.

No undefined behavior detected. (Miri's race detector is also the tool that flagged the pre-0.2.0-beta.2 implementation's value-read data race, which is what motivated the current protocol.)

## Stability

`0.2.x` is the consolidation milestone. The `Runtime<C>` and `Cells` API is intentionally usable but minimal; user-facing API stability commitments live on the wrapper crates.

## License

Apache-2.0
