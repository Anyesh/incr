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
    type U32;
    type U64;
    type State;
    type Ptr<T: 'static>: PtrCell<T>;
    type Lock<T: 'static>: Lock<T>;
    type DepStack: DepStack;

    // ... constructors and inline-only load/store/CAS helpers
}
```

All trait methods are `#[inline(always)]` and take `&Self::Cell` references, so the compiler can see through every call site. The validation that this carries zero overhead on the single-threaded path lives in the spike branch's RESULTS.md (`spike/incr-core-monomorphization`): `walk_local` and a hand-written non-trait baseline produce **byte-identical assembly**.

Two strategy impls ship in this crate:

- `Local`: `Cell<u8>`, `Cell<u32>`, `Cell<u64>`, `Cell<*mut T>`, `RefCell<T>`. `!Send + !Sync` (correct for the single-threaded variant).
- `Shared`: `AtomicU8`, `AtomicU32`, `AtomicU64`, `AtomicPtr<T>`, `RwLock<T>`. `Send + Sync` with Acquire/Release ordering on state-machine transitions.

## What's exposed

- `Runtime<C: Cells>` with `create_input`, `create_query`, `get`, `set`, `node_count`, `graph_snapshot`, `get_traced`, `set_label` / `label`.
- `Incr<T>`: 16-byte `Copy` handle with embedded `RuntimeId` for cross-runtime detection.
- `IncrCollection<T, C>`, `GroupedCollection<K, T, C>`, `SortedCollection<T, K, C>` with the full operator suite (filter, map, count, reduce, sort_by_key, pairwise, window, group_by, join).
- `Value` blanket trait (`Clone + PartialEq + Send + Sync + 'static`) — auto-derived for every qualifying type.
- Tracing types: `NodeInfo`, `NodeKindInfo`, `NodeTrace`, `TraceAction`, `PropagationTrace`.

## Layout invariants

`NodeData<C>` is exactly 64 bytes and 64-byte aligned under both strategies. `const _: () = assert!(...)` blocks enforce this at compile time; layout drift breaks the build immediately.

The segmented node store supports up to 1M nodes per runtime (1024 segments × 1024 slots). Segments are lazily allocated, never moved, and live until the runtime drops.

## Known limitations

- **`get_traced` per-node trace**: records compute, verified-clean, and cutoff events for the current `get` call's compute path. Cross-thread events are not aggregated.

## Memory reclamation

Overflow-dep lists (the heap allocation a node holds when it has more than 7 dependencies) are retired through the [`haphazard`](https://crates.io/crates/haphazard) global hazard-pointer domain. Concurrent readers in `for_each_dep` hold a `HazardPointer` while dereferencing the slot, so an `install_deps` writer's retire is deferred until no protecting reader remains. Memory is reclaimed during normal operation, not just at runtime drop.

## Soundness

All unsafe code in this crate (segmented node store's `UnsafeCell + MaybeUninit` slots, `NodeData::Drop`'s `Box::from_raw` reclamation, `ArenaRegistry`'s `Arc` downcast via raw-pointer rewrap, the graveyard's deferred reclamation, the `SharedDepStack` thread-local) is exercised under `cargo +nightly miri test -p incr-core --lib`. 79 unit tests pass under miri including:

- 50 dynamic dep-set transitions through the overflow path with runtime drop (`local_dynamic_overflow_deps_retirement`)
- 16 threads × 200 rounds racing on the state machine's `try_claim_compute` CAS (`shared_concurrent_claim_one_winner`)
- Cross-segment growth on the segmented node store with reference validity preserved across pushes

No undefined behavior detected.

## Stability

`0.2.x` is the consolidation milestone. The `Runtime<C>` and `Cells` API is intentionally usable but minimal; user-facing API stability commitments live on the wrapper crates.

## License

Apache-2.0
