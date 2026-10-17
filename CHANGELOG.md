# Changelog

All notable changes to this project are documented here. Format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-beta.6] - unreleased

### Changed

- **Contract change: value and key methods must not call into the runtime.** A `Clone`, `PartialEq`, `Hash`, `Eq` or `Ord` that the runtime calls on a value, a collection element or a key now panics with "value and key methods must not call into the runtime" if it reads or writes any node or collection, under `Local` and `Shared` alike. Clean nodes included: beta.5 returned the value there, but under `Shared` that read held a type arena's read lock across the user `clone`, so a `clone` that read another node of the same type could deadlock behind a queued writer. A later release restores reads of Clean nodes, and of inputs another thread is setting, from value code. `Drop` is under the same contract but is not checked, because a displaced value may be reclaimed on any thread; a `Drop` that calls into the runtime may panic or deadlock, never corrupt the graph. Operator closures are unaffected.
- **Python: an exception from a value's `__eq__`, `__hash__` or `__lt__` now propagates.** Beta.5 read a raising `__eq__` or `__lt__` as false and a raising `__hash__` as 0, so a collection could store an equal element twice and the contract error above would have been read as "not equal". Now the exception is raised from the call that ran it (`insert`, `delete`, `set`, `get`, `get_group`) as itself, and the collection is left unchanged. Inserting an unhashable value such as a list into a collection now raises `TypeError`. An exception raised by `__eq__` itself propagates, but when `==` succeeds and only the truth value of its result raises `ValueError` (the NumPy and pandas convention for an ambiguous truth value, as with an array), the values count as not equal, as in beta.5. Any other exception from the truth value propagates.
- **`set` claims its input.** A value-changing `set` now holds the input's `Computing` claim across its compare and its write, so readers of that input block until it lands (measured on `get_contended_shared`), a delete of it waits for it, and a `set` holder never waits while claimed. A `set` whose value is unchanged takes no claim; one equal to a change another thread has written but not yet walked returns only after that walk. The displaced value's `Drop` runs after the claim is released, outside every lock `set` took.

### Fixed

- A `set` that raced a delete and recycle of its input could write its value into the node that reused the slot, stamp that node and walk dependents it never had a handle to. It now panics as stale and leaves the new occupant untouched.
- A `set` of an owned input (a `group_by` group's version node) no longer loses its claim to a concurrent delete of the owner: the delete waits for the `set` to land. Before, the delete stole the claim (a debug assertion in debug builds).
- Under `Local`, a `Drop` of the value a `set` displaced that panicked left the slot holding the dropped value; the slot now holds the new value before the old one is dropped.

## [0.2.0-beta.5] - unreleased

### Added

- **`incr-persist`, a new opt-in crate for durable, shareable state.** A `Store` keeps a runtime's inputs and collections in sync with an ordered operation log held by a pluggable `Backend`, and incr-core gains no serde dependency. It ships three backends: memory (several stores in one process, for tests), file (single writer and many readers, torn-tail recovery, crash-safe compaction) and SQLite behind the `sqlite` feature (many writers across processes, WAL mode). A conformance suite behind `backend-tests` holds any backend, including your own, to the same contract.
  - `store.input(key, default)` and `store.collection(key)` declare persisted keys; the first declaration in the log wins, so every process agrees on defaults. `restored_keys(prefix)` lists keys known from the log that this process has not declared yet.
  - `set`, collection `insert`/`delete`/`replace` and `batch` append to the log before applying, with `Durability::Sync` (the default) or `Buffered`. An error after the append poisons the store until it is reopened, so a retry can never double-apply.
  - `checkpoint` compacts the log, forgetting keys that are back at their default; `refresh` applies other writers' records, and a store that fell behind a compaction catches up from the snapshot while user queries over its keys survive.
  - `store.delete(key)` resets a key for every process, `store.undeclare(key)` removes it from this process's graph only, and `subscribe` reports every applied record in log order with its origin.
  - Debug builds report a write that bypassed the store through a raw handle.
- **Cascading delete and erased handles in incr-core.** `ErasedIncr` names a node of any value type; `delete_with_dependents` removes a node and everything that computed against it or is owned by it, returning the count; `is_alive` checks a handle without panicking; `try_delete_node` is a strict delete that returns `DeleteError::{HasDependents, NotAlive}` instead of panicking. All three are in the Python bindings too.
- An example, `examples/persist-config`, whose test kills a writer with SIGKILL mid-write sixteen times and checks that every acknowledged write survives each restart.

### Changed

- **Contract change: `group_by` owns its group nodes.** Deleting a `group_by` router now releases its groups too, and panics with "still has dependents" if any group has consumers (delete those first, or use `delete_with_dependents`). Before, the delete succeeded and leaked the groups. After a router is deleted, every `IncrCollection` in its `GroupedCollection` is stale.
- `get` and `set` find a node's value arena by the type tag the node already stores instead of hashing its `TypeId`, so propagation runs 9 to 12% fewer instructions under `Local` and 5 to 8% fewer under `Shared`.
- Deleting a node drops its compute closure and observers after releasing the runtime lock, so a `Drop` that touches the runtime no longer deadlocks `Shared` or double-borrows `Local`.

### Fixed

- `delete_node` could claim the next occupant of a slot that another thread deleted and recycled between its state check and its claim.
- In Python, each read of a collection's `.version_node` created a new node that was never freed; every collection now shares one, and `NodeId` compares and hashes by node identity.

## [0.2.0-beta.4] - 2026-10-16

### Soundness

Seven races in the `Shared` protocol, all present in 0.2.0-beta.3, are fixed. Each one has a deterministic test that parks a thread at a named point in the protocol, and loom models now run the real runtime over every interleaving up to a preemption bound, in CI on every push.

- **A delete could remove a node that a compute had just started depending on (B1).** `delete_node` checked for dependents and claimed the node in two steps, so a publish could land between them and its edge was dropped. Both now happen under one lock hold.
- **A compute that raced a delete could add a reverse edge to the recycled slot's next occupant (B2).** Recorded deps now carry the slot's generation and are validated under the publish lock, and a compute that read a deleted dep discards its value and runs again.
- **A dep read for the first time could miss a concurrent `set` (B3).** Its reverse edge did not exist until publish, so the set's walk could not reach the reader. A publish that finds a new dep changed since the read now lands the node `Dirty`.
- **An early-cutoff compute flagged mid-flight could strand its dependents (B4), and red/green verification could verify a node `Clean` past a concurrent `set` (B5).** Both left a node `Clean` over a `Dirty` dep, returning a stale value forever. Verification now claims the node and treats a dep that is not `Clean` as changed, and a compute that read a dep landing non-`Clean` lands `Dirty` itself.
- **`verified_at` could name a revision whose dirty walk had not run yet (B6).** It now uses a walked-revision watermark that only passes revisions whose walks have finished.
- **A compute could stamp a change below the `set` that caused it (B7).** Its revisions were captured before its claim, and the claim can succeed on a later `Dirty` than the one it observed. The loom models found this one.
- The dirty walk passes through `New` nodes, so a `Failed` dependent behind a node recovering from a stale dep still recovers, and a panicking compute whose new dep had a change in flight lands `Dirty` rather than staying `Failed` with no walk left to wake it.

### Changed

- Readers of a `Dirty` node now wait while another thread verifies it, as they already did during a recompute.
- Propagation runs fewer instructions per `set` and `get` than 0.2.0-beta.3, 8 to 11% fewer on `Local` chains and 2 to 6% fewer on `Shared`, because a compute whose deps did not change now publishes without allocating. `Shared` also gains atomic read-modify-writes in verification and in the dirty walk, whose cost shows up in cycles rather than instructions.
- **Breaking for custom strategies and dep stacks.** `Cells` gains `DepListCell`, `Watermark`, `new_watermark`, `walked`, `complete_walk`, `fence_seqcst` and a defaulted `backoff`. `DepStack::record_dep` takes the dep's generation, `pop_frame` returns a `PoppedFrame` carrying a poison flag, and `DepStack` gains `poison_current_frame`. Code that only uses `Local` or `Shared` is unaffected.

## [0.2.0-beta.2] — 2026-06-12

### Soundness

A full audit of the Shared protocol found and fixed undefined behavior that Miri's race detector confirms on beta.1:

- **Value storage is now tear-free under concurrency.** Shared value slots are hazard-pointer-protected pointer swaps instead of in-place `UnsafeCell` writes; a `get` racing a `set` (or a recompute) on the same node was a data race on heap values, with use-after-free potential. Local keeps in-place storage and got faster (RefCell + inline slots replace an RwLock + Box indirection).
- **Every state transition is a CAS.** The verified-clean path could clobber another thread's `Computing` claim and run two computes on one node; claim losers now wait for the owner instead of reading stale or empty slots (two threads racing the first `get` of a query could read an uninitialized slot).
- **A `set` racing an in-flight compute can no longer be swallowed.** A new `ComputingDirty` state plus pending-marker `changed_at` stamping and compute-start-revision `fetch_max` stamps close several stale-forever windows; revision bumps are atomic (two concurrent `set` calls could collapse into one revision).
- **Compute closures have a panic boundary.** A panicking closure lands its node in `Failed` (previously: wedged in `Computing` forever, deadlocking all future readers), keeps the runtime usable, and recovers when a dependency changes. Lock poisoning is no longer fatal; operators stage user-closure results outside locks so an unwind cannot half-apply a batch.
- **Dependency cycles panic with a diagnostic** instead of hanging or overflowing the stack. (Known limitation: a cycle spanning two threads' in-flight computes livelocks; cross-thread cycle detection needs a waits-for graph.)
- **Foreign-runtime and stale handles are rejected in release builds** (was: debug_assert only; release silently corrupted whichever node shared the slot).
- **Python collections no longer dangle.** They held a raw pointer into the runtime object; dropping the runtime first left use-after-free on `insert`. They now hold strong references.

### Added

- **`delete_node`**: nodes can be deleted (leaves first); slots and arena storage recycle under a bumped generation, so long-lived dynamic graphs no longer grow without bound. Stale handles panic with a clear message.
- **`aggregate` operator**: incrementally maintained monoid fold over a balanced aggregation tree, O(log n) per change, correct for non-invertible folds (max, min). `reduce` keeps snapshot semantics for arbitrary folds and says so in its docs.
- **Incremental `pairwise`/`window`**: consume positional sort deltas; one upstream row produces O(1) / O(size) output deltas instead of a full rebuild.
- **Observers**: `observe(handle, callback)` + `stabilize()` for batch change notifications on the pull model; `unobserve` to detach.
- **Delta-log compaction**: collection logs drop the prefix all registered consumers have passed (everything, if no consumer is attached); operators attached to populated collections bootstrap from a snapshot. Log memory is bounded by consumer lag, not collection lifetime.
- **Queries may read any number of inputs** (was: panic above 255 dependencies).
- **In-repo Salsa comparison harness** (`crates/incr-bench`): the README's numbers are reproducible with `cargo bench -p incr-bench --bench salsa_compare`.
- **ThreadSanitizer CI job** and a mixed-workload concurrent stress suite; Miri runs the threaded hammer tests.
- **Python**: real exception propagation (callbacks re-raise the original Python exception), `delete_node`/`observe`/`stabilize`/`aggregate` bindings, abi3 wheels for CPython 3.10+, pytest suite wired into CI on 3.10 and 3.14. The `incr_concurrent` classes are genuinely shareable across Python threads and release the GIL around computing calls (was: everything `unsendable`, defeating the crate's purpose).

### Changed

- Collections use the strategy lock: single-threaded users no longer pay RwLock acquires per collection op.
- Shared `get` costs two hazard-pointer atomics per value read and Shared writes allocate (the price of tear-free lock-free reads); chain propagation is ~108 ns/node Local, ~348 ns/node Shared on the bench machine. beta.1's faster Shared numbers were measured on the racy code.
- pyo3 0.23 → 0.26; Python floor is now 3.10.
- Workspace version is centralized; release publishing verifies crates (no `--no-verify`) and ships sdists + abi3 wheels.

## [0.2.0-beta.1] — 2026-05-20

### Architecture

The big break: `incr-compute` and `incr-concurrent` are now thin re-export wrappers over a shared engine crate, `incr-core`. The engine is parameterized over a `Cells` strategy trait (`Local` for single-threaded, `Shared` for `Send + Sync`); the compiler monomorphizes each surface crate into the appropriate variant. The full algorithm — dependency tracking, ensure_clean's iterative post-order walker, red-green early cutoff, the segmented node store, the typed value arenas, all nine operators — lives in one place. v0.1's parallel implementations are deleted.

### Breaking changes

- **`Value` bound** is now `Clone + PartialEq + Send + Sync + 'static` in **both** crates (was `Any + Clone + PartialEq + 'static` in `incr-compute` v0.1). Most user types already meet the bound; types that don't will need wrapping (e.g., `Arc<Mutex<T>>` instead of bare `Rc<...>`).
- **Single `Runtime` per crate** rather than the v0.1 split. `incr_compute::Runtime` is `Runtime<Local>`; `incr_concurrent::Runtime` is `Runtime<Shared>`. The public method names match v0.1.
- **`NodeId::raw()` → `NodeId.0`**. The struct is `pub struct NodeId(pub u32)`; the field is accessed directly.
- **`Incr<T>::node_id()` → `Incr<T>::slot()`**. The handle returns its u32 slot index.
- **`IncrCollection::version_node_id()` removed**. Use `version_node()` which returns `Incr<u64>`.
- **`count()` returns `Incr<u64>`** (was `Incr<usize>` in `incr-concurrent` v0.1). Sized to the network-portable type.
- **`Runtime::set_label`** takes a `u32` slot directly (was `NodeId` in v0.1).
- **`Runtime::set_tracing` removed**. `get_traced` now arms tracing internally for the duration of the call.
- **`SortedCollection::entries()` → `snapshot()`**.
- **`IncrCollection::delete` returns `bool`** indicating whether a delete was actually recorded (was: silently dropped the inner result in production v0.1).
- **`Runtime::set` on a query node panics** with a clear message. This was undefined behavior in v0.1 (would overwrite the arena slot and corrupt the state machine).
- **`Runtime` `!Send + !Sync` under Local**; `Send + Sync` under Shared (was: mixed in v0.1).

### Added

- **`incr-core` published crate** as the shared engine. Re-exported types include `Cells`, `Local`, `Shared`, `PtrCell`, `Lock`, `DepStack`, `LocalDepStack`, `SharedDepStack`, `LocalLock`. Users who want to build a custom concurrency strategy on top of the engine can do so.
- **Overflow-dep storage**: queries with more than 7 dependencies are now supported (was: hard limit of 7 in v0.1's inline-only path). Overflow lists live in a heap-allocated `DepList`, reclaimed via the [`haphazard`](https://crates.io/crates/haphazard) global hazard-pointer domain. Concurrent readers hold a hazard pointer during traversal; writers retire displaced lists for deferred free.
- **Real per-node tracing** in `get_traced`: every node visit during a get records a `NodeTrace` (`VerifiedClean` or `Recomputed { value_changed }`). Aggregates (`nodes_recomputed`, `nodes_cutoff`) populated from the trace. Hot-path cost: one Relaxed u8 load per compute when disarmed (~1 ns).
- **Property tests under both strategies**: the same generator + verifier (`verify_incremental_matches_batch<C: Cells>`) runs against `Local` and `Shared`. 1000 random function graphs × 2 strategies + 500 random collection op sequences × 6 tests = ~5000 random scenarios per `cargo test` run.
- **Concurrent stress test** for `incr-concurrent`: 4 reader threads + 1 writer thread × 1000 iterations with torn-read detection.
- **Miri validation**: `cargo +nightly miri test -p incr-core --lib` covers all unsafe paths (segmented store, hazard-pointer reclamation, state machine CAS races). Zero undefined behavior reported across 79 unit tests.
- **`Runtime::graph_snapshot`** returns real per-node `NodeInfo` with dependencies (read from inline-7 + overflow storage) and dependents (from inner state).

### Performance

Per-node propagation cost on this machine (criterion --quick):

| Workload | `incr-compute` | `incr-concurrent` | Salsa |
|---|---|---|---|
| Diamond (4 nodes, propagate input through) | 647 ns | 764 ns | 1,066 ns |
| Early cutoff (input changes, clamped output doesn't) | 314 ns | 404 ns | 469 ns |
| Per-node propagation (chain) | ~135 ns | ~169 ns | ~387 ns |

Collection insert through `filter → map → count`:

| Size | `incr-compute` insert | From-scratch batch | Speedup |
|---|---|---|---|
| 1K | 673 ns | 102 µs | **152x** |
| 10K | 657 ns | 67 µs | **102x** |
| 100K | 661 ns | 156 µs | **236x** |

The "incremental cost is constant in collection size" property holds. Production v0.1 README claimed 186x at 100K; v0.2 beats that by 27%. Lab notes in the wiki devlog.

### Fixed

- `count()` operator is now O(new deltas) per get rather than O(N) (was: summed over the entire multiset on every get in v0.1).
- `publish_deps` static-dep fast path (was: O(N) churn on `dependents` lists in v0.1 due to a bug that grew the lists unbounded across iterations).

### Removed

- `incr-python` and `incr-concurrent-python` crates have been **re-implemented** against the v0.2 engine; their public Python API matches v0.1 but they internally use the new types. PyPI publish is gated on the next 0.2.x patch alongside the wheel-build job in CI.

### Architecture decisions

- The architectural reset and migration plan are documented in the project's internal wiki (`wiki/projects/incr/decisions/unification-into-incr-core.md` and `wiki/projects/incr/plans/incr-core-consolidation.md`).
- 21 commits on the `v0.2-rewrite` branch (cut from main 2026-05-20).

## [0.1.x]

The v0.1 line shipped two independent crates (`incr-compute` and `incr-concurrent`) with shared API names but separate implementations. See git history for the per-release notes.
