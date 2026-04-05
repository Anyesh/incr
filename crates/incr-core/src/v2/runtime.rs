//! Skeletal v2 Runtime.
//!
//! This is the commit-F scaffolding: it ties the state machine, typed
//! arenas, arena registry, NodeData, and Incr<T> handle together into
//! a runnable `Runtime` type with minimal behavior. The goal is to
//! prove the architecture holds together end-to-end, not to reach
//! feature parity with v1.
//!
//! # What is implemented
//!
//! - `Runtime::new()` constructs a fresh runtime with a unique
//!   `RuntimeId`.
//! - `create_input::<T>(initial)` registers an input node whose value
//!   is immediately available via `get`.
//! - `create_query::<T, F>(compute)` registers a lazy query node whose
//!   compute closure runs on first `get` and whose result is memoized.
//! - `get::<T>(handle)` reads an input or forces a compute on a query.
//!   Handle verification rejects cross-runtime and stale-generation
//!   handles with clear panics.
//! - `set::<T>(handle, value)` updates an input. Taking `set` on a
//!   query node panics with a diagnostic message.
//!
//! # What is deliberately deferred
//!
//! - **Dependency tracking.** A compute closure can call `rt.get(dep)`
//!   but the runtime does not record which deps the compute touched.
//!   Queries are memoized forever after their first compute.
//! - **Dirty propagation.** `set` updates an input's arena slot and
//!   bumps the revision but does NOT mark dependent queries as dirty.
//!   No dependent queries exist in the runtime's bookkeeping anyway.
//! - **Early cutoff.** No value comparison on recompute (there is no
//!   recompute).
//! - **Cycle detection.** No `COMPUTE_STACK` thread-local yet. A query
//!   whose compute closure transitively calls its own handle will
//!   deadlock-spin on the Computing state indefinitely; this is
//!   temporary.
//! - **Panic catching.** A panic inside a compute closure leaves the
//!   node in `Computing` state permanently. A future commit adds
//!   `catch_unwind` and the `Failed` transition per spec section 8.
//! - **AtomicPrimitiveArena dispatch.** Every value type uses
//!   `GenericArena<T>`, even primitives. The primitive arena from
//!   commit A is unused here because specialization is unstable and
//!   a sealed `Value` trait with manual primitive impls adds complexity
//!   this commit does not need. A future commit adds the Value trait
//!   and wires primitives to their faster arena.
//!
//! # Node storage
//!
//! Nodes live in a `RwLock<Vec<Box<NodeData>>>` indexed by slot. The
//! `Box` keeps each NodeData at a stable heap address across Vec
//! resizes, so pointers into NodeData remain valid after the vec
//! grows. This is simpler than a segmented store and suffices for the
//! scaffolding. A later commit upgrades nodes to a segmented store
//! (sharing the segment machinery with the arenas via an extracted
//! helper) so that reader traversal is fully lock-free.
//!
//! Compute closures live in a parallel
//! `RwLock<Vec<Option<Arc<ComputeFn>>>>` indexed the same way.
//! Input nodes have `None`; query nodes have `Some`. `Arc` lets
//! `run_compute` take a cheap clone under the read lock and release
//! the lock before invoking the closure, so nested `get` calls inside
//! compute do not reenter the same lock.

use std::any::Any;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use super::arena::ErasedArena;
use super::handle::{Incr, RuntimeId};
use super::node::{NodeData, NodeId};
use super::nodes_store::SegmentedNodes;
use super::registry::ArenaRegistry;
use super::state::NodeState;
use super::value::Value;

// ---------------------------------------------------------------------------
// COMPUTE_STACK: per-thread stack of active compute frames.
// ---------------------------------------------------------------------------
//
// When the Runtime enters `run_compute` for a node, it pushes a frame onto
// this thread's compute stack. Every `rt.get` call checks the stack top:
// if there is an active frame for the same runtime, the handle being read
// is recorded as a dependency of the frame's node. On compute exit, the
// frame is popped and its recorded deps are published to the node.
//
// The stack is per-thread, not per-runtime, so a compute closure running
// on thread T recording deps sees only the frames that T is responsible
// for. Cross-thread dep tracking is a non-concept: a compute closure runs
// on exactly one thread from start to finish. Cross-runtime dep tracking
// is explicitly skipped: a compute closure for runtime A that happens to
// call into runtime B does not record B's nodes as deps of A's node. The
// frame's `runtime_id` field is how we make that distinction.
//
// Cycle detection will plug into this stack in a later commit (L per the
// rewrite sequencing): before pushing a frame for node X, walk the stack
// to see whether X is already present on the current thread. If yes,
// panic with CycleError. This commit does not implement that check; a
// cyclic query will simply self-spin on its own Computing state and the
// test suite stays away from that case.

/// A single frame in the compute stack. Created when `run_compute` begins
/// for a node and destroyed when the compute completes (or panics, once
/// panic catching lands).
struct ComputeFrame {
    /// Identity of the runtime that pushed this frame. A cross-runtime
    /// `rt.get` call whose runtime id does not match this field skips
    /// dep recording. This keeps cross-runtime compute closures honest:
    /// they do not contaminate each other's dep graphs.
    runtime_id: RuntimeId,
    /// Slot index of the node whose compute this frame is tracking.
    /// Recorded deps become this node's dependencies on compute exit.
    node_slot: u32,
    /// Dependencies recorded so far. Appended to on every `rt.get` inside
    /// the compute closure that matches this frame's runtime. Deduplicated
    /// on compute exit before publishing to the node.
    deps: Vec<NodeId>,
}

thread_local! {
    /// Per-thread stack of active compute frames. Nested computes push
    /// and pop in LIFO order. `RefCell` suffices because a single thread
    /// cannot have two overlapping borrows on its own stack (nested
    /// operations are strictly sequential), and the stack is never
    /// shared across threads.
    static COMPUTE_STACK: RefCell<Vec<ComputeFrame>> = const { RefCell::new(Vec::new()) };
}

/// Type-erased compute closure. A query node's compute fn takes the
/// runtime (so it can call `rt.get(...)` for dependencies) and returns
/// a `Box<dyn Any + Send + Sync>` which the runtime downcasts to the
/// concrete value type at call time. The double boxing (Arc + Box)
/// lets `run_compute` clone the Arc under a read lock and release the
/// lock before calling the closure, avoiding reentrant lock issues.
type ComputeFn = dyn Fn(&Runtime) -> Box<dyn Any + Send + Sync> + Send + Sync;

/// The v2 incremental computation runtime.
pub struct Runtime {
    /// Stable identity for this runtime. Used by `Incr<T>` handles to
    /// detect cross-runtime misuse. Adopted from the arena registry's
    /// id so the two systems share one identity concept.
    id: RuntimeId,

    /// Node storage. `SegmentedNodes` is a lock-free segmented store
    /// that gives readers direct `&NodeData` access without any
    /// RwLock acquisition (commit U replaced the prior
    /// `RwLock<Vec<Box<NodeData>>>`). Segments are 1024 slots each,
    /// allocated lazily and never moved or deallocated until the
    /// store drops, so references obtained during reads remain
    /// valid for the runtime's lifetime.
    ///
    /// Writers serialize through the runtime's `write_mutex`; no
    /// writer contention on the `len` counter or segment allocation.
    /// Readers Acquire-load `len` and then `&` the appropriate slot,
    /// with the Release-Acquire pair on `len` synchronizing the
    /// reader's view of the initialized NodeData.
    nodes: SegmentedNodes,

    /// Compute closures for query nodes. Indexed the same way as
    /// `nodes`. `None` for input nodes. `Arc` lets `run_compute` extract
    /// the closure under a short-lived read guard and invoke it after
    /// the guard is dropped, so nested `get` calls do not reenter the
    /// same lock.
    compute_fns: RwLock<Vec<Option<Arc<ComputeFn>>>>,

    /// Forward edges. Indexed by node slot; `dependents[slot]` holds
    /// the list of nodes that depend on the node at `slot`. Populated
    /// by `run_compute` after a query publishes its recorded deps: for
    /// each dep, the computing node is appended to that dep's
    /// dependents list. A later commit's dirty walk reads this vector
    /// to find every query that needs invalidation when an input
    /// changes.
    ///
    /// This is the spec's "parallel dependents vec" per section 5.1
    /// (the NodeData docs also call it out as the resolved design for
    /// the inline-vs-parallel ambiguity in the spec).
    dependents: RwLock<Vec<Vec<NodeId>>>,

    /// Failure messages for nodes in the `Failed` state. Parallel to
    /// `nodes`; `Some(msg)` when the node's most recent compute
    /// panicked, `None` otherwise. Populated in the panic-catching
    /// path of `run_compute` and read when `get` encounters a Failed
    /// node. Cleared when the dirty walk transitions a node out of
    /// Failed (Failed → Dirty on an upstream change, allowing a
    /// retry).
    failure_messages: RwLock<Vec<Option<String>>>,

    /// Arena registry; owns the typed storage for node values.
    registry: ArenaRegistry,

    /// Monotonic revision counter. Bumped on every `set`. Not yet used
    /// for dirty propagation or early cutoff in this commit, but kept
    /// in the struct so subsequent commits can plug in without
    /// breaking the public shape.
    revision: AtomicU64,

    /// Serializes all writer operations (`create_*`, `set`). The spec's
    /// single-writer-many-readers model holds this mutex briefly for
    /// the duration of an input update; here it also covers node
    /// creation because we have a simpler store.
    write_mutex: Mutex<()>,
}

impl Runtime {
    /// Construct a new runtime with a fresh identity.
    pub fn new() -> Self {
        let registry = ArenaRegistry::new();
        Self {
            id: registry.id(),
            nodes: SegmentedNodes::new(),
            compute_fns: RwLock::new(Vec::new()),
            dependents: RwLock::new(Vec::new()),
            failure_messages: RwLock::new(Vec::new()),
            registry,
            revision: AtomicU64::new(0),
            write_mutex: Mutex::new(()),
        }
    }

    /// Return this runtime's identity.
    pub fn id(&self) -> RuntimeId {
        self.id
    }

    /// Current revision counter. Bumped on every `set`. Not user-facing
    /// yet; exposed for tests.
    #[cfg(test)]
    pub(crate) fn revision(&self) -> u64 {
        self.revision.load(Ordering::Relaxed)
    }

    /// Create an input node holding the given initial value.
    ///
    /// The `T: PartialEq` bound enables early cutoff: a `set` with the
    /// same value as the current one is a no-op (no revision bump, no
    /// dirty walk), and a recompute that produces a value equal to the
    /// previous one skips the arena write. The bound is uniform across
    /// all v2 value types so the API surface stays consistent.
    pub fn create_input<T>(&self, initial: T) -> Incr<T>
    where
        T: Value,
    {
        let _guard = self
            .write_mutex
            .lock()
            .expect("runtime write mutex poisoned");
        let revision = self.revision.load(Ordering::Relaxed);

        let arena_slot = T::reserve_with(self.arena_for::<T>(), initial);

        let node = NodeData::new_input(0, arena_slot, revision);

        let slot = self.append_node(node, None);
        Incr::new(slot, 0, self.id)
    }

    /// Create a query node whose value is produced by running `compute`.
    ///
    /// Reactivity: after the initial compute on first `get`, the value
    /// is memoized until an input this query transitively depends on
    /// changes, at which point the query is marked Dirty and the next
    /// read triggers a recompute. Early cutoff applies: if a recompute
    /// produces a value equal to the previous one, the arena write is
    /// skipped, which saves a clone for expensive value types. Full
    /// transitive early cutoff (where an unchanged recompute also
    /// prevents downstream queries from recomputing) requires red-green
    /// verification via `verified_at` / `changed_at` and is deferred
    /// to a later commit.
    pub fn create_query<T, F>(&self, compute: F) -> Incr<T>
    where
        T: Value,
        F: Fn(&Runtime) -> T + Send + Sync + 'static,
    {
        let _guard = self
            .write_mutex
            .lock()
            .expect("runtime write mutex poisoned");

        // Reserve an empty slot in the arena; first compute will
        // populate it. For primitive types this zero-initializes the
        // slot; for generic types it leaves the Option None.
        let arena_slot = T::reserve_empty(self.arena_for::<T>());

        let node = NodeData::new_query(0, arena_slot);

        // Type-erase the closure: return a Box<dyn Any + Send + Sync>
        // that the runtime downcasts to T at call time.
        let erased: Arc<ComputeFn> = Arc::new(move |rt: &Runtime| -> Box<dyn Any + Send + Sync> {
            let value = compute(rt);
            Box::new(value) as Box<dyn Any + Send + Sync>
        });

        let slot = self.append_node(node, Some(erased));
        Incr::new(slot, 0, self.id)
    }

    /// Read the value of a node.
    ///
    /// The fast path (observing `Clean`) holds a `nodes.read()` guard
    /// across the arena read so that no writer can mutate the slot
    /// concurrently: `set` takes `nodes.write()`, which is mutually
    /// exclusive with any outstanding `nodes.read()`. This is the
    /// commit-F correctness fallback for the reader-writer race on
    /// `GenericArena<T>` slots that was discovered in review finding C1.
    /// A later commit replaces nodes with a segmented store and uses a
    /// proper hazard / RCU scheme so reads do not need the RwLock gate.
    pub fn get<T>(&self, handle: Incr<T>) -> T
    where
        T: Value,
    {
        self.check_runtime(handle);
        // Cycle detection: if the handle's slot is already on this
        // thread's compute stack, running compute on it would either
        // spin forever on the Computing state or recurse infinitely.
        // Panic with a clear diagnostic instead. This runs before
        // record_dep so a cycling `rt.get` never contaminates the
        // parent frame's dep list with a self-edge.
        self.check_for_cycle(handle.slot());
        // If this get is happening inside a compute closure, record
        // the handle as a dependency of the currently-computing node.
        // This runs before the actual read so the dep is captured
        // even if the read panics or the node is still computing.
        self.record_dep(handle.slot());
        loop {
            // Fast path (Clean): direct lock-free access via the
            // segmented nodes store. No RwLock. Reader/writer
            // synchronization on the arena slot is delegated to the
            // Value trait implementation (atomic loads for primitive
            // types, per-slot mutex for generic types).
            {
                let node = self.nodes.get(handle.slot());
                node.verify_handle(handle, self.id)
                    .unwrap_or_else(|e| panic!("{}", e));
                if node.state() == NodeState::Clean {
                    let arena_slot = node.arena_slot();
                    let value: T = T::read(self.arena_for::<T>(), arena_slot);
                    return value;
                }
            }

            // Slow path: not Clean. Check the state and dispatch.
            let state = self.nodes.get(handle.slot()).state();
            match state {
                NodeState::Clean => {
                    // Raced with a compute completion between our fast
                    // path check and here. Loop to re-enter the fast path.
                }
                NodeState::New => {
                    // First-time compute. CAS New or Dirty to Computing
                    // via try_claim_compute (which accepts both source
                    // states). If we win we own the compute; if we lose,
                    // another thread won and will transition to Clean.
                    let claimed = self
                        .nodes
                        .get(handle.slot())
                        .state_cell()
                        .try_claim_compute()
                        .is_ok();
                    if claimed {
                        self.run_compute::<T>(handle.slot(), false);
                    } else {
                        std::hint::spin_loop();
                    }
                }
                NodeState::Dirty => {
                    // Recompute. Same CAS path as New (try_claim_compute
                    // handles both source states), but `is_recompute` is
                    // true so run_compute uses the dep-diff path.
                    let claimed = self
                        .nodes
                        .get(handle.slot())
                        .state_cell()
                        .try_claim_compute()
                        .is_ok();
                    if claimed {
                        self.run_compute::<T>(handle.slot(), true);
                    } else {
                        std::hint::spin_loop();
                    }
                }
                NodeState::Computing => {
                    // Another thread (or a prior CAS from this thread
                    // in a nested call) is running the compute. Spin.
                    std::hint::spin_loop();
                }
                NodeState::Failed => {
                    // A prior compute panicked and the node was
                    // transitioned to Failed. Look up the stored
                    // failure message and panic with it so the
                    // caller sees the original diagnostic. Retry
                    // is only possible via an upstream change that
                    // transitions Failed → Dirty through the walk.
                    let msg = {
                        let fails = self
                            .failure_messages
                            .read()
                            .expect("failure_messages lock poisoned");
                        fails[handle.slot() as usize].clone()
                    };
                    panic!(
                        "v2 runtime: node at slot {} is Failed: {}",
                        handle.slot(),
                        msg.as_deref().unwrap_or("unknown failure")
                    );
                }
            }
        }
    }

    /// Update an input node's value. Panics if the handle refers to a
    /// query node.
    ///
    /// Takes `nodes.write()` rather than `nodes.read()` for the
    /// duration of the arena mutation. This excludes all concurrent
    /// `get` callers (which hold `nodes.read()` on their fast path)
    /// so the writer has exclusive access to the arena slot while
    /// calling `arena.write(slot, value)`, which for `GenericArena<T>`
    /// is a plain non-atomic store that drops the old `T` and
    /// installs a new one. Without this exclusion a concurrent reader
    /// could observe a torn value (review finding C1).
    ///
    /// Commit U replaces the nodes RwLock with a lock-free segmented
    /// store. C1 synchronization between readers and writers of the
    /// same arena slot now lives at the Value trait layer: primitive
    /// types route to `AtomicPrimitiveArena` where reads and writes
    /// are tear-free atomic operations, and non-primitive types
    /// route to `GenericArena` which (in commit U) gains per-slot
    /// mutex synchronization around the `Option<T>` cell.
    pub fn set<T>(&self, handle: Incr<T>, value: T)
    where
        T: Value,
    {
        self.check_runtime(handle);
        let _guard = self
            .write_mutex
            .lock()
            .expect("runtime write mutex poisoned");

        // Direct lock-free access to the node; write_mutex serializes
        // us against other writers and the Value trait handles
        // reader/writer exclusion on the arena slot.
        let node = self.nodes.get(handle.slot());
        node.verify_handle(handle, self.id)
            .unwrap_or_else(|e| panic!("{}", e));
        assert!(
            self.compute_fns.read().expect("compute lock poisoned")[handle.slot() as usize]
                .is_none(),
            "set() called on a query node; only input nodes may be set"
        );
        let arena_slot = node.arena_slot();

        // Early cutoff: if the new value equals the current value,
        // treat this set as a no-op. Skip the arena write, skip the
        // revision bump, and skip the dirty walk.
        let current: T = T::read(self.arena_for::<T>(), arena_slot);
        if current == value {
            return;
        }

        // Arena write. For primitive types this is a tear-free
        // atomic store via AtomicPrimitiveArena and safe against
        // concurrent readers. For generic types this is a
        // mutex-guarded replacement via GenericArena (commit U) and
        // also safe against concurrent readers.
        T::write(self.arena_for::<T>(), arena_slot, value);

        // State was and remains Clean. This Release store anchors
        // the memory ordering contract for any reader that observes
        // the Clean state via Acquire: it synchronizes with the
        // arena write above for the non-atomic generic path.
        node.state_cell().store_release(NodeState::Clean);

        self.revision.fetch_add(1, Ordering::Relaxed);

        // Transitively mark every query reachable from this input as
        // Dirty. Done after the input's new value is published (both
        // the arena write and the state Release above have completed)
        // so that any reader that observes a dependent Dirty and then
        // recomputes is guaranteed to see the new input value via the
        // Release-Acquire chain on the dependent's state. The walk
        // runs outside the nodes.write() guard because (a) it only
        // needs read access to nodes and (b) it takes its own brief
        // locks on dependents and state cells for each visited node.
        self.mark_dependents_dirty(handle.slot());
    }

    // -- internal helpers --------------------------------------------------

    /// Append a new node to the store and the parallel compute_fns,
    /// dependents, and failure_messages vecs, returning the slot.
    /// Called from the write-mutex-guarded paths.
    ///
    /// The nodes store is lock-free (commit U) and manages its own
    /// synchronization via Release on `len`; only the parallel vecs
    /// need explicit write-lock acquisition.
    fn append_node(&self, node: NodeData, compute: Option<Arc<ComputeFn>>) -> u32 {
        let slot = self.nodes.push(node);

        let mut computes = self.compute_fns.write().expect("compute lock poisoned");
        let mut dependents = self.dependents.write().expect("dependents lock poisoned");
        let mut failures = self
            .failure_messages
            .write()
            .expect("failure_messages lock poisoned");
        debug_assert_eq!(slot as usize, computes.len());
        debug_assert_eq!(slot as usize, dependents.len());
        debug_assert_eq!(slot as usize, failures.len());
        computes.push(compute);
        dependents.push(Vec::new());
        failures.push(None);
        slot
    }

    /// Check that a handle's runtime id matches this runtime. Must be
    /// called before any code that dereferences `handle.slot()`, since
    /// a cross-runtime handle's slot may be out of bounds in this
    /// runtime's nodes vec. Runs before any index operation so the
    /// user sees the actual cross-runtime diagnostic rather than an
    /// opaque index-out-of-bounds panic.
    fn check_runtime<T: 'static>(&self, handle: Incr<T>) {
        if handle.runtime_id() != self.id {
            panic!(
                "Incr handle from runtime {:?} used with runtime {:?}",
                handle.runtime_id(),
                self.id
            );
        }
    }

    /// Walk the current thread's compute stack looking for a frame
    /// belonging to this runtime whose `node_slot` equals `slot`. If
    /// found, the caller is trying to `get` a node that is already
    /// computing on the same thread: a dependency cycle. Panic with
    /// a clear diagnostic. Cross-runtime frames are ignored, because
    /// a cycle inside runtime A cannot pass through runtime B's
    /// dep graph.
    ///
    /// This is the spec's section 9 cycle detection. Called from
    /// `get` before `record_dep` so a cycling read never contaminates
    /// a parent frame's dep list with a self- or back-edge.
    fn check_for_cycle(&self, slot: u32) {
        COMPUTE_STACK.with(|stack| {
            let stack = stack.borrow();
            for frame in stack.iter() {
                if frame.runtime_id == self.id && frame.node_slot == slot {
                    panic!(
                        "CycleError: dependency cycle detected: node at slot {} \
                         is already computing on this thread",
                        slot
                    );
                }
            }
        });
    }

    /// Record `slot` as a dependency of the currently-computing node on
    /// this thread, if any. Called at the start of every `get`.
    ///
    /// Dep recording is silently skipped in three cases, and the skip
    /// is intentional rather than an oversight:
    ///
    /// 1. No active compute frame on this thread. Top-level `rt.get`
    ///    calls from user code are not deps of anything.
    /// 2. The active frame belongs to a different runtime. A compute
    ///    closure for runtime A that happens to call into runtime B
    ///    does not record B's nodes as deps of A's node.
    /// 3. The `slot` equals the frame's own `node_slot`. A query whose
    ///    compute reads its own handle is a self-cycle; recording it
    ///    would create a self-loop in the dep graph. Cycle detection
    ///    proper arrives in a later commit (L) and will panic instead
    ///    of silently skipping; for now the self-read simply does not
    ///    create a dep edge and the node's Computing state will cause
    ///    the self-read to spin (caller's problem, not ours).
    fn record_dep(&self, slot: u32) {
        COMPUTE_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some(frame) = stack.last_mut() {
                if frame.runtime_id == self.id && frame.node_slot != slot {
                    frame.deps.push(NodeId(slot));
                }
            }
        });
    }

    /// Push a new compute frame onto this thread's stack. Called at
    /// the start of `run_compute`.
    fn push_compute_frame(&self, node_slot: u32) {
        COMPUTE_STACK.with(|stack| {
            stack.borrow_mut().push(ComputeFrame {
                runtime_id: self.id,
                node_slot,
                deps: Vec::new(),
            });
        });
    }

    /// Pop the top compute frame from this thread's stack and return
    /// its recorded (and deduplicated, order-preserving) deps. Called
    /// at the end of `run_compute`. Panics if the stack is empty or
    /// the top frame does not match the expected node slot, either of
    /// which would indicate a bug in the push/pop pairing.
    fn pop_compute_frame(&self, expected_node_slot: u32) -> Vec<NodeId> {
        /// Threshold below which linear dedup beats HashSet. Nearly
        /// every compute in realistic workloads has 1-4 deps, and
        /// linear scan over a small list (~1-2 ns per probe) is
        /// dramatically cheaper than building a HashSet (~15-20 ns
        /// hash + insert per element plus allocation). 8 is a
        /// conservative cutoff: at 8 elements linear dedup does 28
        /// compares worst case, well under the constant cost of a
        /// single HashSet operation.
        const LINEAR_DEDUP_THRESHOLD: usize = 8;

        COMPUTE_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            let frame = stack.pop().expect("compute stack underflow");
            debug_assert_eq!(
                frame.runtime_id, self.id,
                "compute frame belongs to a different runtime"
            );
            debug_assert_eq!(
                frame.node_slot, expected_node_slot,
                "compute frame node_slot mismatch (expected {}, got {})",
                expected_node_slot, frame.node_slot
            );
            // Deduplicate while preserving the order of first occurrence.
            // Linear scan for small lists (the common case by far);
            // HashSet for wide fan-in nodes.
            if frame.deps.len() <= LINEAR_DEDUP_THRESHOLD {
                let mut out: Vec<NodeId> = Vec::with_capacity(frame.deps.len());
                for id in frame.deps {
                    if !out.contains(&id) {
                        out.push(id);
                    }
                }
                out
            } else {
                let mut seen: std::collections::HashSet<NodeId> =
                    std::collections::HashSet::with_capacity(frame.deps.len());
                frame
                    .deps
                    .into_iter()
                    .filter(|id| seen.insert(*id))
                    .collect()
            }
        })
    }

    /// Run the compute closure for a query node and transition its
    /// state to Clean. The caller must have already CAS'd the state
    /// from New or Dirty to Computing.
    ///
    /// `is_recompute = false` on the first compute (source state was
    /// New) publishes recorded deps and reverse edges.
    /// `is_recompute = true` on a recompute (source state was Dirty)
    /// skips both, assuming the dep set is unchanged between runs.
    /// Dynamic dependencies (where the dep set differs between runs)
    /// are not supported in this commit: the recompute path will
    /// silently use the stale dep list and produce a correct value
    /// for the original deps but miss any new deps that would have
    /// been discovered on this run. A follow-up commit adds dep
    /// diff + epoch reclamation for the dynamic case.
    ///
    /// Dep tracking still happens on every run (the COMPUTE_STACK
    /// frame is pushed and popped) because it is cheap and symmetric,
    /// and because skipping it would leave stale state on the stack
    /// if a future change starts using the recorded deps on recompute.
    fn run_compute<T>(&self, slot: u32, is_recompute: bool)
    where
        T: Value,
    {
        // Extract the compute closure under a short-lived read guard.
        // Clone the Arc so we can drop the guard before calling, which
        // means the closure may freely recurse into self.get without
        // self-deadlocking on compute_fns.
        let compute = {
            let computes = self.compute_fns.read().expect("compute lock poisoned");
            computes[slot as usize]
                .as_ref()
                .expect("query node has no compute closure")
                .clone()
        };

        // Snapshot the revision BEFORE the compute runs. After the
        // compute returns, we compare this to the current revision.
        // If any writer called `set` during our compute (bumping the
        // revision), our result may be based on stale input values
        // that the writer's dirty walk could not mark us Dirty for
        // because we were Computing. In that case we transition to
        // Dirty instead of Clean so the next reader retries the
        // compute with fresh inputs. This closes the "Computing-
        // during-dirty-walk" race the commit J docs flagged as a
        // known limitation.
        let revision_at_start = self.revision.load(Ordering::Relaxed);

        // Push a compute frame so that any `rt.get` calls made by the
        // closure record their handles as dependencies of this node.
        self.push_compute_frame(slot);

        // Run the closure outside any lock, inside a catch_unwind
        // boundary so a panic inside the closure does not leave the
        // COMPUTE_STACK in an inconsistent state or poison the whole
        // runtime. `AssertUnwindSafe` is safe here because the runtime
        // state we touch on unwind (stack, failure_messages, node
        // state cell) is explicitly restored in the error arm below.
        //
        // Per spec section 8: a caught panic transitions the node to
        // Failed, stashes a string representation of the panic payload
        // so subsequent `get` calls can report it, and re-raises the
        // panic so the original `rt.get` caller sees the diagnostic
        // rather than a silently-swallowed failure. The re-raise is
        // important: a compute panic is a programmer error and needs
        // to be visible, not absorbed.
        let value_box_result: std::thread::Result<Box<dyn Any + Send + Sync>> =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (compute)(self)));

        // Pop the frame regardless of compute outcome. In the Ok
        // branch this restores the parent frame for continued dep
        // tracking; in the Err branch it prevents a leaked frame
        // from corrupting subsequent computes on the same thread.
        let recorded_deps = self.pop_compute_frame(slot);

        // Publish dependencies and reverse edges. On first compute
        // we install them fresh; on recompute we diff against the
        // previous set and update only the differences. Both paths
        // run regardless of whether the compute succeeded or
        // panicked, because a partial dep record from a panicked
        // compute is still the most accurate information we have
        // and it lets the dirty walk reach a Failed node on a
        // subsequent upstream change (enabling retry).
        if !is_recompute {
            self.nodes.get(slot).publish_initial_deps(&recorded_deps);
            if !recorded_deps.is_empty() {
                let mut dependents = self.dependents.write().expect("dependents lock poisoned");
                for dep in &recorded_deps {
                    dependents[dep.0 as usize].push(NodeId(slot));
                }
            }
        } else {
            // Recompute: diff the new recorded deps against the
            // node's previous dep list and update only the
            // differences. Handles the dynamic-deps case where a
            // compute reads a different set of deps on different
            // runs (e.g., a conditional that branches on an input
            // value).
            self.update_deps_on_recompute(slot, &recorded_deps);
        }

        // Handle the compute result. The panic arm runs AFTER the
        // dep publish above, so a Failed node has enough reverse-
        // edge bookkeeping in place for the dirty walk to find it
        // on the next upstream change.
        let value_box = match value_box_result {
            Ok(v) => v,
            Err(panic_payload) => {
                // Extract a readable message from the panic payload.
                // Rust panic payloads are `Box<dyn Any + Send>`; the
                // common cases are `&'static str` (from `panic!("...")`
                // with a string literal) and `String` (from
                // `panic!("{}", ...)`).
                let msg = extract_panic_message(&panic_payload);

                // Stash the message before transitioning state so a
                // reader that observes Failed can immediately retrieve
                // it. The lock is held briefly.
                {
                    let mut fails = self
                        .failure_messages
                        .write()
                        .expect("failure_messages lock poisoned");
                    fails[slot as usize] = Some(msg);
                }

                // Transition the node to Failed with Release so that
                // subsequent readers observing Failed also observe
                // the stored failure message via the happens-before
                // chain (read lock on failure_messages + Release on
                // state gives the necessary ordering).
                self.nodes
                    .get(slot)
                    .state_cell()
                    .store_release(NodeState::Failed);

                // Re-raise the panic so the `rt.get` caller that
                // triggered this compute sees it. This unwinds past
                // the caller's frame too; a caller who wants to
                // handle the error without dying needs to wrap the
                // `rt.get` in `catch_unwind` themselves (a future
                // `try_get` method provides the ergonomic version).
                std::panic::resume_unwind(panic_payload);
            }
        };

        // Downcast to the caller's expected type. A mismatch here is a
        // bug in how `create_query` packed the closure; it should be
        // impossible from safe user code because `create_query<T, F>`
        // binds T at the call site.
        let new_value: T = *value_box
            .downcast::<T>()
            .expect("compute function returned wrong type");

        // Look up the arena slot for this node.
        let arena_slot = self.nodes.get(slot).arena_slot();

        // Post-compute revision check: if the revision bumped during
        // our compute, at least one writer ran `set` while we were
        // computing, and the dirty walk from that set could not mark
        // us Dirty because our state was Computing. Our result may
        // be stale. Transition to Dirty instead of Clean so the next
        // reader retries the compute against the fresh input values.
        // Skip the arena write in this case: writing the stale value
        // would pollute the slot with wrong data if an early-cutoff-
        // style optimization later decided to believe it.
        let revision_at_end = self.revision.load(Ordering::Relaxed);
        let stale_due_to_concurrent_write = revision_at_end != revision_at_start;

        if stale_due_to_concurrent_write {
            // Do not write the stale value. Transition to Dirty via
            // Release so the next reader observes Dirty and retries.
            // The recorded_deps are still published above, so the
            // retry has an accurate dep list for the dirty walk to
            // reach it on subsequent sets.
            self.nodes
                .get(slot)
                .state_cell()
                .store_release(NodeState::Dirty);
            return;
        }

        // Early cutoff on recompute: if the new value equals the value
        // currently in the arena (from a previous compute), skip the
        // arena write. This saves the clone inside `arena.write` for
        // large types. Only applies to recomputes, because a first
        // compute starts from an uninitialized slot and must always
        // write. Full transitive early cutoff (where an unchanged
        // recompute prevents downstream queries from recomputing)
        // requires red-green verification via `verified_at` /
        // `changed_at` and is deferred to a later commit.
        //
        // Uses `try_read` rather than `read` because a Failed → Dirty
        // retry (commit L) may call `run_compute` with is_recompute
        // true even though the slot has never been written: the
        // previous compute panicked before writing, so the cell is
        // still `None`. In that case there is nothing to compare
        // against and we must always write.
        let should_write = if is_recompute {
            match T::try_read(self.arena_for::<T>(), arena_slot) {
                Some(old_value) => old_value != new_value,
                None => true,
            }
        } else {
            true
        };

        if should_write {
            T::write(self.arena_for::<T>(), arena_slot, new_value);
        }

        // Release-store state to Clean. This publishes the arena write
        // (and any dep list changes) together with the state
        // transition to readers that Acquire-load state afterward.
        self.nodes
            .get(slot)
            .state_cell()
            .store_release(NodeState::Clean);
    }

    /// Diff the new recorded deps against the node's previous dep
    /// list and update both the node's forward dep list and the
    /// runtime's reverse-edge (dependents) vec to reflect the
    /// difference.
    ///
    /// Called from `run_compute` on recompute (when `is_recompute`
    /// is true). The common case is that the dep set is unchanged
    /// and this function is a fast compare-and-return; the
    /// interesting case is dynamic dependencies where a conditional
    /// inside the compute closure caused it to read a different
    /// set of deps than the previous run.
    ///
    /// After commit U (SegmentedNodes), mutual exclusion for
    /// `NodeData::replace_deps`'s overflow-pointer swap comes from
    /// the Runtime's `write_mutex` being held by the caller
    /// (run_compute is only invoked from get, which does not hold
    /// any node lock, BUT a concurrent writer calling set would
    /// take write_mutex and be blocked from running another
    /// recompute). Since run_compute runs outside write_mutex,
    /// there can be concurrent recompute on different nodes but
    /// not on the same node (state machine's Computing CAS
    /// guarantees at most one). The same-node exclusion is what
    /// replace_deps actually needs.
    fn update_deps_on_recompute(&self, slot: u32, new_deps: &[NodeId]) {
        use std::collections::HashSet;

        // Fast path: check for exact match without allocating a
        // snapshot Vec. The vast majority of recomputes have
        // unchanged dep sets (static deps), and allocating a Vec
        // on every recompute just to compare and throw away is a
        // measurable chunk of propagate_chain_1000's cost. Walk
        // the existing deps via for_each_dep and compare
        // element-by-element against new_deps; short-circuit on
        // the first mismatch.
        let node = self.nodes.get(slot);
        let matches: bool = if node.dep_count() as usize != new_deps.len() {
            false
        } else {
            let mut iter = new_deps.iter();
            let mut all_matched = true;
            node.for_each_dep(|existing| {
                if all_matched {
                    match iter.next() {
                        Some(expected) if *expected == existing => {}
                        _ => all_matched = false,
                    }
                }
            });
            all_matched
        };
        if matches {
            return;
        }

        // Slow path: dep set changed. Snapshot the old deps into
        // a Vec so we can compute diff sets against new_deps.
        let old_deps: Vec<NodeId> = node.collect_deps();

        // Compute the added and removed sets for reverse-edge
        // bookkeeping.
        let old_set: HashSet<NodeId> = old_deps.iter().copied().collect();
        let new_set: HashSet<NodeId> = new_deps.iter().copied().collect();
        let added: Vec<NodeId> = new_deps
            .iter()
            .filter(|d| !old_set.contains(*d))
            .copied()
            .collect();
        let removed: Vec<NodeId> = old_deps
            .iter()
            .filter(|d| !new_set.contains(*d))
            .copied()
            .collect();

        // Replace the node's forward dep list. The state machine's
        // Computing ownership guarantees no other thread is
        // computing this same node, but concurrent readers may be
        // observing `overflow_deps` if they already loaded it. For
        // commit U we leak the old overflow list (skip the
        // Box::from_raw in replace_deps) and defer proper
        // reclamation to a follow-up epoch-based commit. Leaking is
        // a memory cost only, not a correctness cost.
        node.replace_deps_leaking_old_overflow(new_deps);

        // Update reverse edges. Added deps gain an incoming edge
        // from `slot`; removed deps lose theirs.
        if !added.is_empty() || !removed.is_empty() {
            let mut dependents = self.dependents.write().expect("dependents lock poisoned");
            for dep in &added {
                dependents[dep.0 as usize].push(NodeId(slot));
            }
            for dep in &removed {
                dependents[dep.0 as usize].retain(|d| d.0 != slot);
            }
        }
    }

    /// Transitively mark every query reachable from `changed_slot`
    /// (via forward edges in `dependents`) as Dirty.
    ///
    /// Called from `set` after the input's new value is written and
    /// published. Walks the dependents graph breadth-first from the
    /// changed input, using `try_transition(Clean, Dirty)` on each
    /// visited query. Nodes in states other than Clean are skipped:
    ///
    /// - New: never computed, nothing to invalidate.
    /// - Dirty: already marked by a previous walk.
    /// - Computing: concurrent recompute in progress on another
    ///   thread. The walk's `try_transition(Clean, Dirty)` fails
    ///   silently for this node, which would leak a stale result
    ///   if the compute finished with a pre-set value and
    ///   Release-stored Clean. Commit P's post-compute revision
    ///   check in `run_compute` handles this: before transitioning
    ///   out of Computing, the compute thread compares the current
    ///   revision to the revision it recorded at compute start, and
    ///   transitions to Dirty (not Clean) if they differ. Any set
    ///   that raced with Computing bumped the revision, so the
    ///   mismatch is detected and the next reader retries.
    /// - Failed: the node's last compute panicked. The walk attempts
    ///   Failed → Dirty so an upstream change gives the failed node
    ///   a chance to retry on the next read (per spec section 8).
    ///   If the transition succeeds, the stashed failure message is
    ///   cleared.
    ///
    /// The walk does not mark the changed node itself (it is an input
    /// and stays Clean by convention; only its dependents are
    /// potentially stale).
    fn mark_dependents_dirty(&self, changed_slot: u32) {
        use std::collections::HashSet;
        let mut visited: HashSet<u32> = HashSet::new();
        let mut queue: Vec<u32> = Vec::new();
        // Track nodes whose Failed state we transitioned to Dirty
        // so we can clear their stashed failure messages at the end
        // of the walk (one batched write lock on failure_messages
        // instead of one per node).
        let mut cleared_failures: Vec<u32> = Vec::new();

        // Seed the queue with the changed node's direct dependents.
        {
            let dependents = self.dependents.read().expect("dependents lock poisoned");
            for dep in &dependents[changed_slot as usize] {
                if visited.insert(dep.0) {
                    queue.push(dep.0);
                }
            }
        }

        while let Some(slot) = queue.pop() {
            // Try Clean → Dirty first (the common case). If that
            // fails, try Failed → Dirty to allow retry of a failed
            // compute after an upstream change. Other source states
            // (New, Computing, Dirty) are skipped silently.
            let cell = self.nodes.get(slot).state_cell();
            if cell
                .try_transition(NodeState::Clean, NodeState::Dirty)
                .is_err()
                && cell
                    .try_transition(NodeState::Failed, NodeState::Dirty)
                    .is_ok()
            {
                cleared_failures.push(slot);
            }

            // Walk forward from this node regardless of whether the
            // transition succeeded. Even if we did not transition this
            // node (e.g., it was already Dirty), its dependents may
            // not yet have been visited on a previous walk, and they
            // might still be Clean and need marking. The visited set
            // prevents revisiting a node we have already queued.
            let children: Vec<u32> = {
                let dependents = self.dependents.read().expect("dependents lock poisoned");
                dependents[slot as usize].iter().map(|id| id.0).collect()
            };
            for child in children {
                if visited.insert(child) {
                    queue.push(child);
                }
            }
        }

        // Clear the stashed failure messages for nodes that
        // transitioned Failed → Dirty. One write lock for the whole
        // batch keeps this cheap even when many nodes retry at once.
        if !cleared_failures.is_empty() {
            let mut fails = self
                .failure_messages
                .write()
                .expect("failure_messages lock poisoned");
            for slot in cleared_failures {
                fails[slot as usize] = None;
            }
        }
    }

    /// Look up (creating if necessary) the arena for value type `T`
    /// via the Value trait. Returns a `&dyn ErasedArena` that the
    /// Value trait's methods downcast to the concrete arena type.
    ///
    /// Per commit T: T's Value impl decides whether to route to
    /// `AtomicPrimitiveArena<T>` (for primitives — tear-free reads)
    /// or `GenericArena<T>` (for non-primitives — Option-gated).
    /// The registry caches the arena per type so the factory runs
    /// at most once per T per runtime.
    fn arena_for<T: Value>(&self) -> &dyn ErasedArena {
        let arena_ptr = self.registry.ensure_arena::<T, _>(|| T::create_arena());
        // SAFETY: `arena_ptr` was returned by the registry and is
        // stable for the registry's lifetime (arenas are never
        // removed and each arena lives at a fixed heap address via
        // Box). The returned reference's lifetime is tied to &self,
        // which outlives the registry.
        unsafe { &*arena_ptr }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract a readable message from a caught panic payload.
///
/// Rust panic payloads are `Box<dyn Any + Send>` with no enforced
/// type; the common producers are `panic!("literal")` which yields a
/// `&'static str` and `panic!("fmt {}", x)` which yields a `String`.
/// Other types (user-constructed panics via `panic_any`) fall back to
/// a generic message so failures are never silently swallowed.
fn extract_panic_message(payload: &Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "compute function panicked with a non-string payload".to_string()
    }
}

// SAFETY: Runtime's fields are all Send + Sync already (RwLock, Mutex,
// ArenaRegistry, AtomicU64, RuntimeId). The compute closures are bound
// to `Fn + Send + Sync + 'static`, so the `Arc<ComputeFn>` values inside
// the compute_fns vec are Send + Sync.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_runtime_has_unique_id_and_empty_store() {
        let rt_a = Runtime::new();
        let rt_b = Runtime::new();
        assert_ne!(rt_a.id(), rt_b.id());
        assert_eq!(rt_a.revision(), 0);
    }

    #[test]
    fn create_input_and_get_returns_initial_value() {
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(42);
        assert_eq!(rt.get(input), 42);
    }

    #[test]
    fn set_updates_the_input_value() {
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(1);
        assert_eq!(rt.get(input), 1);
        rt.set(input, 99);
        assert_eq!(rt.get(input), 99);
        rt.set(input, 7);
        assert_eq!(rt.get(input), 7);
    }

    #[test]
    fn set_bumps_revision() {
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(0);
        assert_eq!(rt.revision(), 0);
        rt.set(input, 1);
        assert_eq!(rt.revision(), 1);
        rt.set(input, 2);
        assert_eq!(rt.revision(), 2);
    }

    #[test]
    fn multiple_inputs_of_same_type_are_independent() {
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(1);
        let b = rt.create_input::<u64>(2);
        let c = rt.create_input::<u64>(3);
        assert_eq!(rt.get(a), 1);
        assert_eq!(rt.get(b), 2);
        assert_eq!(rt.get(c), 3);
        rt.set(b, 20);
        assert_eq!(rt.get(a), 1);
        assert_eq!(rt.get(b), 20);
        assert_eq!(rt.get(c), 3);
    }

    #[test]
    fn inputs_of_different_types_coexist() {
        let rt = Runtime::new();
        let int_in = rt.create_input::<u64>(10);
        let str_in = rt.create_input::<String>("hello".to_string());
        let vec_in = rt.create_input::<Vec<i32>>(vec![1, 2, 3]);
        assert_eq!(rt.get(int_in), 10);
        assert_eq!(rt.get(str_in), "hello");
        assert_eq!(rt.get(vec_in), vec![1, 2, 3]);
        rt.set(str_in, "world".to_string());
        rt.set(vec_in, vec![4, 5]);
        assert_eq!(rt.get(int_in), 10);
        assert_eq!(rt.get(str_in), "world");
        assert_eq!(rt.get(vec_in), vec![4, 5]);
    }

    #[test]
    fn query_computes_on_first_get_and_memoizes() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let rt = Runtime::new();
        let counter = Arc::new(AtomicUsize::new(0));

        let q = {
            let counter = counter.clone();
            rt.create_query::<u64, _>(move |_rt| {
                counter.fetch_add(1, Ordering::SeqCst);
                42
            })
        };

        assert_eq!(counter.load(Ordering::SeqCst), 0, "compute should be lazy");
        assert_eq!(rt.get(q), 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        // Subsequent gets return the memoized value without re-running.
        assert_eq!(rt.get(q), 42);
        assert_eq!(rt.get(q), 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn query_compute_can_call_get_on_another_node() {
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(10);
        let b = rt.create_input::<u64>(32);
        let sum = rt.create_query::<u64, _>(move |rt| rt.get(a) + rt.get(b));
        assert_eq!(rt.get(sum), 42);
    }

    #[test]
    fn query_with_string_value() {
        let rt = Runtime::new();
        let name = rt.create_input::<String>("Anish".to_string());
        let greeting = rt.create_query::<String, _>(move |rt| format!("hi, {}", rt.get(name)));
        assert_eq!(rt.get(greeting), "hi, Anish");
    }

    #[test]
    fn query_recomputes_when_its_input_changes() {
        // Commit J makes v2 reactive. Previous commits shipped a
        // failing-intent test named
        // query_memoization_is_NOT_reactive_in_this_commit that
        // asserted the opposite of this behavior. That test was
        // renamed and its assertion flipped here. A future commit
        // touching reactivity that breaks this test is breaking
        // something load-bearing.
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(1);
        let q = rt.create_query::<u64, _>(move |rt| rt.get(a) * 10);
        assert_eq!(rt.get(q), 10);
        rt.set(a, 7);
        assert_eq!(
            rt.get(q),
            70,
            "query should reflect the new input value after set + get"
        );
        rt.set(a, 100);
        assert_eq!(rt.get(q), 1000);
    }

    #[test]
    #[should_panic(expected = "Incr handle from runtime")]
    fn cross_runtime_handle_panics() {
        let rt_a = Runtime::new();
        let rt_b = Runtime::new();
        let h = rt_a.create_input::<u64>(1);
        let _ = rt_b.get(h);
    }

    #[test]
    #[should_panic(expected = "set() called on a query node")]
    fn set_on_query_node_panics() {
        let rt = Runtime::new();
        let q = rt.create_query::<u64, _>(|_| 42);
        rt.set(q, 99);
    }

    #[test]
    fn runtime_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Runtime>();
    }

    #[test]
    fn get_is_callable_from_multiple_threads_after_set_completes() {
        use std::thread;
        let rt = Arc::new(Runtime::new());
        let input = rt.create_input::<u64>(100);
        rt.set(input, 200);

        // Spawn several readers, each of which observes the final value.
        // This only tests that the runtime's Send+Sync contract holds
        // and that a single-writer-many-readers handoff works. Real
        // concurrent correctness is verified by a future commit's
        // loom/property tests.
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let rt = rt.clone();
                thread::spawn(move || rt.get(input))
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), 200);
        }
    }

    #[test]
    fn concurrent_get_and_set_on_non_copy_input_is_race_free() {
        // Regression test for the reader-writer data race on GenericArena
        // slots identified in review finding C1. Before the fix, concurrent
        // rt.get and rt.set on the same String input would race on the
        // UnsafeCell<Option<String>> slot: the writer's plain-non-atomic
        // `*slot = Some(new)` drops the old String and installs a new one
        // while the reader is mid-clone, yielding torn data or a segfault.
        //
        // The fix uses the nodes RwLock as a synchronization gate: readers
        // hold nodes.read() across the arena read, writers hold nodes.write()
        // across the arena write, so reader and writer never touch the slot
        // simultaneously.
        //
        // This test exists to prove the fix works. Under miri, the broken
        // version of the code flags a data race; the fixed version passes.
        // Under regular cargo test, the broken version may corrupt strings
        // or segfault; the fixed version returns valid strings reliably.

        use std::thread;

        // A small set of valid values. Every observation must match one
        // of these exactly; anything else indicates a torn read.
        let valid_values: Vec<String> = (0..4)
            .map(|i| format!("value-{}-with-padding-to-force-heap-allocation", i))
            .collect();

        let rt = Arc::new(Runtime::new());
        let input = rt.create_input::<String>(valid_values[0].clone());

        // Spawn readers that loop on get and verify each observed value
        // is in the valid set. Any torn string will mismatch.
        const READER_ITERS: usize = 2_000;
        const READERS: usize = 4;
        const WRITER_ITERS: usize = 2_000;

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let reader_handles: Vec<_> = (0..READERS)
            .map(|_| {
                let rt = rt.clone();
                let valid = valid_values.clone();
                let stop = stop.clone();
                thread::spawn(move || {
                    let mut seen = 0usize;
                    while !stop.load(Ordering::Relaxed) && seen < READER_ITERS {
                        let v = rt.get(input);
                        assert!(
                            valid.contains(&v),
                            "observed torn or corrupt value: {:?}",
                            v
                        );
                        seen += 1;
                    }
                })
            })
            .collect();

        // Writer loop: rotate through the valid values.
        for i in 0..WRITER_ITERS {
            let v = &valid_values[i % valid_values.len()];
            rt.set(input, v.clone());
        }
        stop.store(true, Ordering::Relaxed);

        for h in reader_handles {
            h.join()
                .expect("reader thread panicked; data race detected");
        }
    }

    #[test]
    fn concurrent_get_and_set_on_vec_input_is_race_free() {
        // Second shape of the C1 regression: Vec<u64> has both a length
        // and a pointer in its slot, so a torn read can observe a
        // length from one Vec and a data pointer from another, leading
        // to an out-of-bounds read when the cloned Vec is used.

        use std::thread;

        let values: Vec<Vec<u64>> = vec![
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![100, 200, 300, 400, 500, 600, 700, 800],
            vec![9999; 16],
        ];

        let rt = Arc::new(Runtime::new());
        let input = rt.create_input::<Vec<u64>>(values[0].clone());

        const ITERS: usize = 2_000;
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let handles: Vec<_> = (0..3)
            .map(|_| {
                let rt = rt.clone();
                let valid = values.clone();
                let stop = stop.clone();
                thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let v = rt.get(input);
                        // Every observed vec must match one of the valids
                        // exactly. A torn vec would fail this check OR
                        // would have faulted in the clone itself.
                        assert!(
                            valid.iter().any(|expected| expected == &v),
                            "observed torn vec: {:?}",
                            v
                        );
                    }
                })
            })
            .collect();

        for i in 0..ITERS {
            rt.set(input, values[i % values.len()].clone());
        }
        stop.store(true, Ordering::Relaxed);

        for h in handles {
            h.join()
                .expect("reader thread panicked; data race detected");
        }
    }

    // -------------------------------------------------------------------
    // Dependency tracking tests (commit H).
    // -------------------------------------------------------------------
    //
    // These tests verify that `rt.get` calls inside a compute closure
    // record their handles as dependencies of the currently-computing
    // node, and that the recorded deps are correctly deduplicated and
    // published to the node. They do NOT test reactivity: commit H
    // records deps but does not yet propagate changes through them.
    // Reactivity is commit J.

    /// Read a node's published dependencies. Test-only helper.
    /// After commit U the nodes store is lock-free; this helper
    /// just calls through to the direct accessor.
    fn collect_deps_for_slot(rt: &Runtime, slot: u32) -> Vec<super::super::node::NodeId> {
        rt.nodes.get(slot).collect_deps()
    }

    /// Read a node's published dependents (forward edges). Test-only
    /// helper for commit I's reverse-edge bookkeeping.
    fn collect_dependents_for_slot(rt: &Runtime, slot: u32) -> Vec<super::super::node::NodeId> {
        let dependents = rt.dependents.read().unwrap();
        dependents[slot as usize].clone()
    }

    #[test]
    fn query_records_its_input_dependency() {
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(7);
        let q = rt.create_query::<u64, _>(move |rt| rt.get(input) + 1);
        assert_eq!(rt.get(q), 8);
        let deps = collect_deps_for_slot(&rt, q.slot());
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].0, input.slot());
    }

    #[test]
    fn query_records_multiple_input_dependencies_in_get_order() {
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(1);
        let b = rt.create_input::<u64>(2);
        let c = rt.create_input::<u64>(3);
        let sum = rt.create_query::<u64, _>(move |rt| rt.get(a) + rt.get(b) + rt.get(c));
        assert_eq!(rt.get(sum), 6);
        let deps = collect_deps_for_slot(&rt, sum.slot());
        assert_eq!(deps.len(), 3);
        assert_eq!(deps[0].0, a.slot());
        assert_eq!(deps[1].0, b.slot());
        assert_eq!(deps[2].0, c.slot());
    }

    #[test]
    fn duplicate_reads_dedup_to_single_dep_in_first_occurrence_order() {
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(10);
        let b = rt.create_input::<u64>(20);
        // Compute reads a, b, a, b, a. The dedup should preserve the
        // order of first occurrence: [a, b], not [b, a] or duplicates.
        let q = rt.create_query::<u64, _>(move |rt| {
            let _ = rt.get(a);
            let _ = rt.get(b);
            let _ = rt.get(a);
            let _ = rt.get(b);
            rt.get(a)
        });
        let _ = rt.get(q);
        let deps = collect_deps_for_slot(&rt, q.slot());
        assert_eq!(deps.len(), 2, "expected 2 unique deps, got {:?}", deps);
        assert_eq!(deps[0].0, a.slot(), "first unique dep should be a");
        assert_eq!(deps[1].0, b.slot(), "second unique dep should be b");
    }

    #[test]
    fn nested_queries_each_get_their_own_dep_list() {
        // Q1 reads input I.
        // Q2 reads Q1 only.
        // Verify Q1's deps are [I] and Q2's deps are [Q1], not that
        // Q2 transitively inherits I. Each compute frame has its own
        // recorded deps, and reads to Q1 from inside Q2's compute
        // run in their own (newly pushed) frame rather than appending
        // to Q2's.
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(5);
        let q1 = rt.create_query::<u64, _>(move |rt| rt.get(input) * 2);
        let q2 = rt.create_query::<u64, _>(move |rt| rt.get(q1) + 3);
        assert_eq!(rt.get(q2), 13);

        let q1_deps = collect_deps_for_slot(&rt, q1.slot());
        assert_eq!(q1_deps.len(), 1);
        assert_eq!(q1_deps[0].0, input.slot());

        let q2_deps = collect_deps_for_slot(&rt, q2.slot());
        assert_eq!(q2_deps.len(), 1);
        assert_eq!(
            q2_deps[0].0,
            q1.slot(),
            "q2's deps should contain q1, not input transitively"
        );
    }

    #[test]
    fn top_level_get_outside_compute_records_nothing() {
        // A plain `rt.get(input)` from the test body (no active
        // compute frame on this thread) must not panic and must not
        // leave stale state in the thread-local stack. After the
        // call, any subsequent compute should still be able to push
        // a fresh frame cleanly.
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(42);
        // Top-level read: no compute is running, no frame to record on.
        assert_eq!(rt.get(input), 42);
        // Now create a query and verify its first compute works
        // normally (i.e., the stack was left in a clean state after
        // the top-level read).
        let q = rt.create_query::<u64, _>(move |rt| rt.get(input) * 10);
        assert_eq!(rt.get(q), 420);
        let deps = collect_deps_for_slot(&rt, q.slot());
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].0, input.slot());
    }

    #[test]
    fn compute_stack_is_clean_between_queries() {
        // After one query's compute runs, the stack should be empty
        // again. If it isn't, the next query would see the previous
        // query's frame as its "parent" and misattribute deps.
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(1);
        let b = rt.create_input::<u64>(2);
        let q_a = rt.create_query::<u64, _>(move |rt| rt.get(a));
        let q_b = rt.create_query::<u64, _>(move |rt| rt.get(b));
        // Trigger q_a first, then q_b, and verify each has only its
        // own dep, not the union. If the stack leaked between
        // compute invocations, q_b's deps would include `a`.
        let _ = rt.get(q_a);
        let _ = rt.get(q_b);
        let q_a_deps = collect_deps_for_slot(&rt, q_a.slot());
        let q_b_deps = collect_deps_for_slot(&rt, q_b.slot());
        assert_eq!(q_a_deps.len(), 1);
        assert_eq!(q_a_deps[0].0, a.slot());
        assert_eq!(q_b_deps.len(), 1);
        assert_eq!(q_b_deps[0].0, b.slot());
    }

    #[test]
    fn cross_runtime_get_inside_compute_does_not_record_on_current_frame() {
        // A compute closure for runtime A that captures a handle from
        // runtime B and reads it should not record B's slot on A's
        // frame. Runtime identity is the gate.
        let rt_a = Arc::new(Runtime::new());
        let rt_b = Arc::new(Runtime::new());
        let b_input = rt_b.create_input::<u64>(99);
        let a_input = rt_a.create_input::<u64>(1);

        // Build a compute closure that captures rt_b and b_input by
        // Arc + Copy and reads both a_input (own runtime) and b_input
        // (other runtime).
        let q = {
            let rt_b_inner = rt_b.clone();
            rt_a.create_query::<u64, _>(move |rt| {
                let other = rt_b_inner.get(b_input); // cross-runtime, must not record on rt_a's frame
                rt.get(a_input) + other
            })
        };
        assert_eq!(rt_a.get(q), 100);
        let deps = collect_deps_for_slot(&rt_a, q.slot());
        assert_eq!(
            deps.len(),
            1,
            "cross-runtime reads should not record; expected only a_input dep, got {:?}",
            deps
        );
        assert_eq!(deps[0].0, a_input.slot());
    }

    #[test]
    fn query_reading_nothing_records_empty_deps() {
        let rt = Runtime::new();
        let q = rt.create_query::<u64, _>(|_rt| 42);
        assert_eq!(rt.get(q), 42);
        let deps = collect_deps_for_slot(&rt, q.slot());
        assert!(deps.is_empty(), "got unexpected deps: {:?}", deps);
    }

    // -------------------------------------------------------------------
    // Forward-edge (dependents) tests (commit I).
    // -------------------------------------------------------------------

    #[test]
    fn fresh_input_has_no_dependents() {
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(1);
        let dependents = collect_dependents_for_slot(&rt, input.slot());
        assert!(dependents.is_empty());
    }

    #[test]
    fn fresh_query_has_no_dependents() {
        let rt = Runtime::new();
        let q = rt.create_query::<u64, _>(|_| 1);
        // Dependents are populated for a node when OTHER queries
        // depend on it, not when this query runs its own compute.
        let dependents = collect_dependents_for_slot(&rt, q.slot());
        assert!(dependents.is_empty());
    }

    #[test]
    fn input_gains_dependent_after_query_first_reads_it() {
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(10);
        let q = rt.create_query::<u64, _>(move |rt| rt.get(input) + 1);

        // Before the query is ever run, input has no dependents.
        assert!(collect_dependents_for_slot(&rt, input.slot()).is_empty());

        // Running the query triggers its first compute, which
        // records input as a dep and writes the reverse edge.
        let _ = rt.get(q);

        let dependents = collect_dependents_for_slot(&rt, input.slot());
        assert_eq!(dependents.len(), 1);
        assert_eq!(dependents[0].0, q.slot());
    }

    #[test]
    fn input_with_multiple_dependents_collects_all_of_them() {
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(5);
        let q1 = rt.create_query::<u64, _>(move |rt| rt.get(input) * 2);
        let q2 = rt.create_query::<u64, _>(move |rt| rt.get(input) * 3);
        let q3 = rt.create_query::<u64, _>(move |rt| rt.get(input) + 100);

        let _ = rt.get(q1);
        let _ = rt.get(q2);
        let _ = rt.get(q3);

        let dependents = collect_dependents_for_slot(&rt, input.slot());
        assert_eq!(dependents.len(), 3);
        // Order reflects the order in which queries were first computed.
        assert_eq!(dependents[0].0, q1.slot());
        assert_eq!(dependents[1].0, q2.slot());
        assert_eq!(dependents[2].0, q3.slot());
    }

    #[test]
    fn intermediate_query_has_its_downstream_as_dependent() {
        // input → q1 → q2
        // After running q2, q1's dependents should be [q2] and
        // input's dependents should be [q1]. This exercises the
        // multi-level dep chain.
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(3);
        let q1 = rt.create_query::<u64, _>(move |rt| rt.get(input) + 10);
        let q2 = rt.create_query::<u64, _>(move |rt| rt.get(q1) * 2);

        let _ = rt.get(q2); // triggers q1's compute as a side effect

        let input_deps = collect_dependents_for_slot(&rt, input.slot());
        assert_eq!(input_deps.len(), 1);
        assert_eq!(input_deps[0].0, q1.slot());

        let q1_deps = collect_dependents_for_slot(&rt, q1.slot());
        assert_eq!(q1_deps.len(), 1);
        assert_eq!(q1_deps[0].0, q2.slot());

        // q2 has no dependents yet; nothing reads it.
        let q2_deps = collect_dependents_for_slot(&rt, q2.slot());
        assert!(q2_deps.is_empty());
    }

    #[test]
    fn query_with_multiple_distinct_deps_writes_reverse_edge_to_each() {
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(1);
        let b = rt.create_input::<u64>(2);
        let c = rt.create_input::<u64>(3);
        let sum = rt.create_query::<u64, _>(move |rt| rt.get(a) + rt.get(b) + rt.get(c));

        let _ = rt.get(sum);

        let a_deps = collect_dependents_for_slot(&rt, a.slot());
        let b_deps = collect_dependents_for_slot(&rt, b.slot());
        let c_deps = collect_dependents_for_slot(&rt, c.slot());

        assert_eq!(a_deps.len(), 1);
        assert_eq!(a_deps[0].0, sum.slot());
        assert_eq!(b_deps.len(), 1);
        assert_eq!(b_deps[0].0, sum.slot());
        assert_eq!(c_deps.len(), 1);
        assert_eq!(c_deps[0].0, sum.slot());
    }

    #[test]
    fn reverse_edges_are_written_once_per_dep_not_once_per_read() {
        // A query that reads the same input three times should still
        // add exactly one reverse edge. The dep recording dedup
        // happens inside the compute frame, so publish_initial_deps
        // sees a single entry, and the reverse-edge loop runs once
        // per unique dep.
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(7);
        let q = rt.create_query::<u64, _>(move |rt| {
            let a = rt.get(input);
            let b = rt.get(input);
            let c = rt.get(input);
            a + b + c
        });

        let _ = rt.get(q);

        let dependents = collect_dependents_for_slot(&rt, input.slot());
        assert_eq!(
            dependents.len(),
            1,
            "expected exactly one reverse edge, got {:?}",
            dependents
        );
        assert_eq!(dependents[0].0, q.slot());
    }

    // -------------------------------------------------------------------
    // Reactivity tests (commit J).
    // -------------------------------------------------------------------

    fn state_of(rt: &Runtime, slot: u32) -> NodeState {
        rt.nodes.get(slot).state()
    }

    #[test]
    fn set_marks_single_direct_dependent_dirty() {
        // After set, the dependent query should be in Dirty state
        // (before it has been re-read). This proves the dirty walk
        // visited the query and transitioned it.
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(1);
        let q = rt.create_query::<u64, _>(move |rt| rt.get(input) + 100);

        // First compute leaves q in Clean.
        let _ = rt.get(q);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Clean);

        // Set the input; the dirty walk should mark q Dirty.
        rt.set(input, 50);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Dirty);

        // Reading q triggers the recompute and observes the new value.
        assert_eq!(rt.get(q), 150);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Clean);
    }

    #[test]
    fn set_marks_transitive_dependents_dirty() {
        // input -> q1 -> q2 -> q3. Setting input should mark all
        // three queries Dirty via the transitive walk.
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(1);
        let q1 = rt.create_query::<u64, _>(move |rt| rt.get(input) + 1);
        let q2 = rt.create_query::<u64, _>(move |rt| rt.get(q1) * 2);
        let q3 = rt.create_query::<u64, _>(move |rt| rt.get(q2) + 100);

        assert_eq!(rt.get(q3), ((1 + 1) * 2) + 100); // 104
        assert_eq!(state_of(&rt, q1.slot()), NodeState::Clean);
        assert_eq!(state_of(&rt, q2.slot()), NodeState::Clean);
        assert_eq!(state_of(&rt, q3.slot()), NodeState::Clean);

        rt.set(input, 10);
        // All three should be Dirty after the walk.
        assert_eq!(state_of(&rt, q1.slot()), NodeState::Dirty);
        assert_eq!(state_of(&rt, q2.slot()), NodeState::Dirty);
        assert_eq!(state_of(&rt, q3.slot()), NodeState::Dirty);

        // Reading q3 cascades recomputes through q2 and q1.
        assert_eq!(rt.get(q3), ((10 + 1) * 2) + 100); // 122
    }

    #[test]
    fn set_leaves_unrelated_queries_clean() {
        // Two inputs, two queries. Setting one input should only
        // dirty the query that reads it, not the other.
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(1);
        let b = rt.create_input::<u64>(10);
        let q_a = rt.create_query::<u64, _>(move |rt| rt.get(a) * 100);
        let q_b = rt.create_query::<u64, _>(move |rt| rt.get(b) * 100);

        assert_eq!(rt.get(q_a), 100);
        assert_eq!(rt.get(q_b), 1000);

        rt.set(a, 5);
        assert_eq!(state_of(&rt, q_a.slot()), NodeState::Dirty);
        assert_eq!(
            state_of(&rt, q_b.slot()),
            NodeState::Clean,
            "q_b reads b, not a; should not be invalidated"
        );

        assert_eq!(rt.get(q_a), 500);
        assert_eq!(rt.get(q_b), 1000); // unchanged
    }

    #[test]
    fn multiple_dependents_of_one_input_all_dirtied() {
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(1);
        let q1 = rt.create_query::<u64, _>(move |rt| rt.get(input) + 1);
        let q2 = rt.create_query::<u64, _>(move |rt| rt.get(input) + 2);
        let q3 = rt.create_query::<u64, _>(move |rt| rt.get(input) + 3);

        let _ = rt.get(q1);
        let _ = rt.get(q2);
        let _ = rt.get(q3);

        rt.set(input, 100);
        assert_eq!(state_of(&rt, q1.slot()), NodeState::Dirty);
        assert_eq!(state_of(&rt, q2.slot()), NodeState::Dirty);
        assert_eq!(state_of(&rt, q3.slot()), NodeState::Dirty);

        assert_eq!(rt.get(q1), 101);
        assert_eq!(rt.get(q2), 102);
        assert_eq!(rt.get(q3), 103);
    }

    #[test]
    fn diamond_dependency_each_node_visited_once() {
        // q1 depends on input; q2a and q2b both depend on q1; q3
        // depends on both q2a and q2b. This is a diamond: q1 is
        // reached via two paths in the dirty walk. The visited set
        // ensures it's only processed once.
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(1);
        let q1 = rt.create_query::<u64, _>(move |rt| rt.get(input) * 2);
        let q2a = rt.create_query::<u64, _>(move |rt| rt.get(q1) + 10);
        let q2b = rt.create_query::<u64, _>(move |rt| rt.get(q1) + 20);
        let q3 = rt.create_query::<u64, _>(move |rt| rt.get(q2a) + rt.get(q2b));

        // Initial: input=1 → q1=2 → q2a=12, q2b=22 → q3=34.
        assert_eq!(rt.get(q3), 34);

        rt.set(input, 5);
        // input=5 → q1=10 → q2a=20, q2b=30 → q3=50.
        assert_eq!(rt.get(q3), 50);
    }

    #[test]
    fn multiple_sets_in_sequence_propagate_correctly() {
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(1);
        let q = rt.create_query::<u64, _>(move |rt| rt.get(input) * 1000);

        assert_eq!(rt.get(q), 1000);
        rt.set(input, 2);
        assert_eq!(rt.get(q), 2000);
        rt.set(input, 3);
        assert_eq!(rt.get(q), 3000);
        rt.set(input, 4);
        assert_eq!(rt.get(q), 4000);
        rt.set(input, 5);
        assert_eq!(rt.get(q), 5000);
    }

    #[test]
    fn set_on_input_with_no_dependents_is_a_noop_walk() {
        // An input that no query depends on still has its value
        // updated correctly on set, and the (empty) dirty walk
        // should not do anything observable.
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(10);
        assert_eq!(rt.get(input), 10);
        rt.set(input, 99);
        assert_eq!(rt.get(input), 99);
    }

    #[test]
    fn query_never_computed_is_unaffected_by_set() {
        // A query in New state (never computed) should not be
        // transitioned to Dirty by a set. The dirty walk's
        // try_transition(Clean, Dirty) should fail silently on New.
        // The dependents edge doesn't exist yet either (it's written
        // on first compute), so the walk simply doesn't reach it.
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(1);
        let q = rt.create_query::<u64, _>(move |rt| rt.get(input) * 10);

        // Do NOT read q. Its state is New, and input has no
        // dependents recorded yet.
        assert_eq!(state_of(&rt, q.slot()), NodeState::New);
        assert!(collect_dependents_for_slot(&rt, input.slot()).is_empty());

        rt.set(input, 5);

        // q is still New; the first get after this set will compute
        // with the latest value (5), not 1.
        assert_eq!(state_of(&rt, q.slot()), NodeState::New);
        assert_eq!(rt.get(q), 50);
    }

    #[test]
    fn query_reading_two_inputs_invalidated_by_either() {
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(1);
        let b = rt.create_input::<u64>(2);
        let q = rt.create_query::<u64, _>(move |rt| rt.get(a) * 10 + rt.get(b));

        assert_eq!(rt.get(q), 12);

        rt.set(a, 5);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Dirty);
        assert_eq!(rt.get(q), 52);

        rt.set(b, 99);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Dirty);
        assert_eq!(rt.get(q), 149);
    }

    #[test]
    fn string_query_reactivity() {
        // Reactivity with a non-primitive value type. Exercises the
        // same code path as the u64 tests but through GenericArena's
        // UnsafeCell<Option<String>> storage.
        let rt = Runtime::new();
        let name = rt.create_input::<String>("Anish".to_string());
        let greeting = rt.create_query::<String, _>(move |rt| format!("hi, {}", rt.get(name)));

        assert_eq!(rt.get(greeting), "hi, Anish");
        rt.set(name, "world".to_string());
        assert_eq!(rt.get(greeting), "hi, world");
    }

    // -------------------------------------------------------------------
    // Early cutoff tests (commit K).
    // -------------------------------------------------------------------

    #[test]
    fn set_with_same_value_is_a_noop() {
        // Setting an input to its current value should not bump the
        // revision counter, because the early cutoff short-circuits
        // before the arena write. Verifies the no-op path at the input
        // level, which is the cheapest and most common case.
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(42);
        assert_eq!(rt.revision(), 0);

        rt.set(input, 42); // same value
        assert_eq!(rt.revision(), 0, "no-op set must not bump revision");

        rt.set(input, 100); // different value
        assert_eq!(rt.revision(), 1, "real set should bump revision");

        rt.set(input, 100); // same as current
        assert_eq!(rt.revision(), 1, "second no-op set must not bump");
    }

    #[test]
    fn set_with_same_value_does_not_dirty_dependents() {
        // The dirty walk should be skipped on a no-op set. Dependents
        // stay Clean because the walk never runs.
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(1);
        let q = rt.create_query::<u64, _>(move |rt| rt.get(input) + 100);

        assert_eq!(rt.get(q), 101);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Clean);

        rt.set(input, 1); // no-op
        assert_eq!(
            state_of(&rt, q.slot()),
            NodeState::Clean,
            "no-op set must not mark dependent Dirty"
        );
        // Reading q returns the cached value without recomputing.
        assert_eq!(rt.get(q), 101);
    }

    #[test]
    fn set_with_same_string_is_a_noop() {
        // Early cutoff for non-primitive types. PartialEq on String
        // drives the check; the saved work is the String clone inside
        // arena.write, which is the whole point of cutoff for large
        // value types.
        let rt = Runtime::new();
        let s = rt.create_input::<String>("hello".to_string());
        assert_eq!(rt.revision(), 0);
        rt.set(s, "hello".to_string());
        assert_eq!(rt.revision(), 0, "no-op String set must not bump revision");
        rt.set(s, "world".to_string());
        assert_eq!(rt.revision(), 1);
    }

    #[test]
    fn recompute_returning_same_value_transitions_clean_without_panic() {
        // When a recompute produces the same value as before, the
        // arena write is skipped (saves a clone for large types) and
        // the state transitions back to Clean. This test verifies
        // the path runs without panicking or leaving the node in a
        // weird state; the value-level cutoff's effect is hard to
        // observe directly without red-green verification, which is
        // a follow-up commit.
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(5);
        // Query whose value depends on the input but happens to be
        // constant over the range of inputs we use: sign of input.
        let q = rt.create_query::<u64, _>(move |rt| if rt.get(input) > 0 { 1 } else { 0 });

        assert_eq!(rt.get(q), 1);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Clean);

        // Change input to another positive value. q is marked Dirty.
        rt.set(input, 100);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Dirty);

        // Recompute produces 1 again (same value). The code path
        // skips the arena write. q transitions to Clean.
        assert_eq!(rt.get(q), 1);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Clean);

        // Changing to a value that would flip the output actually
        // does flip it.
        rt.set(input, 0);
        assert_eq!(rt.get(q), 0);
    }

    #[test]
    fn input_with_dependents_noop_set_does_not_trigger_recompute() {
        // Count compute invocations via a shared counter. A no-op
        // set must not cause the dependent query to recompute, which
        // the counter directly proves.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let rt = Runtime::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let input = rt.create_input::<u64>(10);
        let q = {
            let counter = counter.clone();
            rt.create_query::<u64, _>(move |rt| {
                counter.fetch_add(1, Ordering::SeqCst);
                rt.get(input) * 2
            })
        };

        // First read triggers the initial compute (count = 1).
        assert_eq!(rt.get(q), 20);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // No-op set: no dirty walk, no recompute.
        rt.set(input, 10);
        assert_eq!(rt.get(q), 20);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "no-op set should not recompute"
        );

        // Real set: dirty walk runs, recompute happens.
        rt.set(input, 20);
        assert_eq!(rt.get(q), 40);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "real set should recompute"
        );

        // Another no-op set: no recompute.
        rt.set(input, 20);
        assert_eq!(rt.get(q), 40);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    // -------------------------------------------------------------------
    // Cycle detection and panic catching tests (commit L).
    // -------------------------------------------------------------------

    /// Read a node's stashed failure message for test assertions.
    fn failure_message_for(rt: &Runtime, slot: u32) -> Option<String> {
        let fails = rt.failure_messages.read().unwrap();
        fails[slot as usize].clone()
    }

    #[test]
    fn self_cycle_panics_and_leaves_node_failed() {
        // A query whose compute reads its own handle. Build the
        // handle first (via a Mutex<Option<_>>) so the closure can
        // reach back to it after create_query returns.
        use std::sync::Mutex;
        let rt = Runtime::new();
        let me: Arc<Mutex<Option<Incr<u64>>>> = Arc::new(Mutex::new(None));
        let q = {
            let me = me.clone();
            rt.create_query::<u64, _>(move |rt| {
                let h = me.lock().unwrap().expect("self handle not set");
                rt.get(h) // cycles
            })
        };
        *me.lock().unwrap() = Some(q);

        // Reading q triggers its compute, which attempts to read q,
        // which trips the cycle detector. The panic unwinds through
        // the compute closure, is caught by run_compute, the node is
        // transitioned to Failed, and the panic is re-raised to our
        // caller frame here.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(q)));
        assert!(result.is_err(), "expected cycle to panic");

        assert_eq!(state_of(&rt, q.slot()), NodeState::Failed);
        let msg = failure_message_for(&rt, q.slot()).expect("failure stashed");
        assert!(
            msg.contains("CycleError"),
            "expected CycleError in failure message, got: {}",
            msg
        );
    }

    #[test]
    fn mutual_cycle_between_two_queries_panics() {
        use std::sync::Mutex;
        let rt = Runtime::new();
        let q1_handle: Arc<Mutex<Option<Incr<u64>>>> = Arc::new(Mutex::new(None));
        let q2_handle: Arc<Mutex<Option<Incr<u64>>>> = Arc::new(Mutex::new(None));

        let q1 = {
            let q2h = q2_handle.clone();
            rt.create_query::<u64, _>(move |rt| {
                let h = q2h.lock().unwrap().expect("q2 handle not set");
                rt.get(h)
            })
        };
        let q2 = {
            let q1h = q1_handle.clone();
            rt.create_query::<u64, _>(move |rt| {
                let h = q1h.lock().unwrap().expect("q1 handle not set");
                rt.get(h)
            })
        };
        *q1_handle.lock().unwrap() = Some(q1);
        *q2_handle.lock().unwrap() = Some(q2);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(q1)));
        assert!(result.is_err(), "expected mutual cycle to panic");
    }

    #[test]
    fn compute_panic_is_caught_and_node_transitions_to_failed() {
        let rt = Runtime::new();
        let q = rt.create_query::<u64, _>(|_| panic!("oops, compute failed"));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(q)));
        assert!(result.is_err());

        assert_eq!(state_of(&rt, q.slot()), NodeState::Failed);
        let msg = failure_message_for(&rt, q.slot()).expect("failure stashed");
        assert!(
            msg.contains("oops, compute failed"),
            "expected panic message in failure, got: {}",
            msg
        );
    }

    #[test]
    fn subsequent_get_on_failed_node_panics_with_stored_message() {
        let rt = Runtime::new();
        let q = rt.create_query::<u64, _>(|_| panic!("original failure text"));

        // First get triggers the panic path.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(q)));
        assert_eq!(state_of(&rt, q.slot()), NodeState::Failed);

        // Second get on the Failed node should panic again with the
        // stored message (not re-run the compute).
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(q)));
        let err = result.unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&'static str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains("Failed") && msg.contains("original failure text"),
            "expected Failed + original message, got: {}",
            msg
        );
    }

    #[test]
    fn panic_preserves_compute_stack_for_subsequent_operations() {
        // After a panicking compute, the thread's COMPUTE_STACK must
        // be empty again so subsequent computes can push fresh
        // frames. If the stack leaked, the next query's deps would
        // be misattributed to the dead frame.
        let rt = Runtime::new();
        let panicking = rt.create_query::<u64, _>(|_| panic!("this compute fails"));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(panicking)));

        // Now create and run a query that should work fine.
        let input = rt.create_input::<u64>(5);
        let q = rt.create_query::<u64, _>(move |rt| rt.get(input) * 2);
        assert_eq!(rt.get(q), 10);

        // And its deps should be recorded correctly.
        let deps = collect_deps_for_slot(&rt, q.slot());
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].0, input.slot());
    }

    #[test]
    fn failed_node_retries_after_upstream_set() {
        // A query that panics only if its input is below some
        // threshold. Set the input to a value that panics, observe
        // Failed, set the input to a safe value, observe the node
        // transitions Failed → Dirty → Clean on next read.
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(0);
        let q = rt.create_query::<u64, _>(move |rt| {
            let v = rt.get(input);
            if v == 0 {
                panic!("input is zero");
            }
            v * 10
        });

        let r1 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(q)));
        assert!(r1.is_err());
        assert_eq!(state_of(&rt, q.slot()), NodeState::Failed);
        assert!(failure_message_for(&rt, q.slot()).is_some());

        // Set input to a non-panicking value. The dirty walk should
        // transition Failed → Dirty and clear the stashed message.
        rt.set(input, 7);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Dirty);
        assert!(failure_message_for(&rt, q.slot()).is_none());

        // Next get retries the compute, which now succeeds.
        assert_eq!(rt.get(q), 70);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Clean);
    }

    #[test]
    fn panic_inside_nested_compute_propagates_to_outer() {
        // q_outer reads q_inner; q_inner panics. The panic should
        // leave both nodes in Failed state (q_inner directly from
        // its own compute, q_outer because its compute panicked
        // while propagating q_inner's panic).
        let rt = Runtime::new();
        let q_inner = rt.create_query::<u64, _>(|_| panic!("inner failure"));
        let q_outer = rt.create_query::<u64, _>(move |rt| rt.get(q_inner) + 1);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(q_outer)));
        assert!(result.is_err());

        assert_eq!(state_of(&rt, q_inner.slot()), NodeState::Failed);
        assert_eq!(state_of(&rt, q_outer.slot()), NodeState::Failed);
    }

    #[test]
    fn non_cycling_read_does_not_trigger_cycle_check() {
        // Sanity: a compute that reads an unrelated node should NOT
        // trip the cycle check, even if its slot index happens to be
        // low or near the computing node's slot.
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(1);
        let b = rt.create_input::<u64>(2);
        let q = rt.create_query::<u64, _>(move |rt| rt.get(a) + rt.get(b));
        assert_eq!(rt.get(q), 3); // no panic
    }

    // -------------------------------------------------------------------
    // Dynamic dependency tests (commit M).
    // -------------------------------------------------------------------

    #[test]
    fn query_with_conditional_deps_tracks_only_read_branch() {
        // A classic dynamic-dep query: read `flag`, then read one of
        // two inputs depending on the flag. Initial flag true, reads
        // `a`. Query's deps should be [flag, a].
        let rt = Runtime::new();
        let flag = rt.create_input::<bool>(true);
        let a = rt.create_input::<u64>(10);
        let b = rt.create_input::<u64>(20);

        let q =
            rt.create_query::<u64, _>(move |rt| if rt.get(flag) { rt.get(a) } else { rt.get(b) });

        assert_eq!(rt.get(q), 10);
        let deps = collect_deps_for_slot(&rt, q.slot());
        // Deps should be [flag, a] — only the branch that was
        // actually taken.
        assert_eq!(deps.len(), 2);
        assert!(deps.iter().any(|d| d.0 == flag.slot()));
        assert!(deps.iter().any(|d| d.0 == a.slot()));
        assert!(!deps.iter().any(|d| d.0 == b.slot()));
    }

    #[test]
    fn flipping_conditional_dep_updates_dep_list_and_reverse_edges() {
        // Start with flag=true (reads a). Change flag to false, read
        // q again (triggers recompute via dirty walk from flag). The
        // new recompute reads b, not a. Assert that:
        //   1. q's new deps are [flag, b]
        //   2. a's dependents list no longer contains q (stale
        //      reverse edge removed)
        //   3. b's dependents list now contains q (new reverse edge
        //      added)
        let rt = Runtime::new();
        let flag = rt.create_input::<bool>(true);
        let a = rt.create_input::<u64>(100);
        let b = rt.create_input::<u64>(200);

        let q =
            rt.create_query::<u64, _>(move |rt| if rt.get(flag) { rt.get(a) } else { rt.get(b) });

        // First compute: reads flag + a.
        assert_eq!(rt.get(q), 100);
        assert!(collect_dependents_for_slot(&rt, a.slot())
            .iter()
            .any(|d| d.0 == q.slot()));
        assert!(!collect_dependents_for_slot(&rt, b.slot())
            .iter()
            .any(|d| d.0 == q.slot()));

        // Flip the flag. Dirty walk marks q Dirty via flag's
        // dependents list.
        rt.set(flag, false);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Dirty);

        // Recompute. Now reads flag + b instead of flag + a.
        assert_eq!(rt.get(q), 200);
        let deps = collect_deps_for_slot(&rt, q.slot());
        assert_eq!(deps.len(), 2);
        assert!(deps.iter().any(|d| d.0 == flag.slot()));
        assert!(deps.iter().any(|d| d.0 == b.slot()));
        assert!(!deps.iter().any(|d| d.0 == a.slot()));

        // Reverse edges: a should no longer point to q, b should.
        let a_deps = collect_dependents_for_slot(&rt, a.slot());
        let b_deps = collect_dependents_for_slot(&rt, b.slot());
        assert!(
            !a_deps.iter().any(|d| d.0 == q.slot()),
            "stale reverse edge a -> q not removed: {:?}",
            a_deps
        );
        assert!(
            b_deps.iter().any(|d| d.0 == q.slot()),
            "new reverse edge b -> q not added: {:?}",
            b_deps
        );
    }

    #[test]
    fn removed_dep_no_longer_invalidates_query() {
        // After a recompute changes the dep set to drop `a`, setting
        // `a` should NOT mark q dirty (because q no longer reads a).
        // Setting the dep that's now in the set (`b`) SHOULD mark q
        // dirty.
        let rt = Runtime::new();
        let flag = rt.create_input::<bool>(true);
        let a = rt.create_input::<u64>(1);
        let b = rt.create_input::<u64>(2);
        let q =
            rt.create_query::<u64, _>(move |rt| if rt.get(flag) { rt.get(a) } else { rt.get(b) });

        // First compute reads a. Second compute (after flag flip)
        // reads b.
        assert_eq!(rt.get(q), 1);
        rt.set(flag, false);
        assert_eq!(rt.get(q), 2);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Clean);

        // Now setting a should NOT dirty q.
        rt.set(a, 999);
        assert_eq!(
            state_of(&rt, q.slot()),
            NodeState::Clean,
            "q should not be dirtied by changes to a (no longer a dep)"
        );

        // But setting b should.
        rt.set(b, 888);
        assert_eq!(state_of(&rt, q.slot()), NodeState::Dirty);
        assert_eq!(rt.get(q), 888);
    }

    #[test]
    fn static_recompute_dep_list_does_not_trigger_reverse_edge_rewrite() {
        // A recompute whose dep set is identical to the previous
        // run's should take the fast path and not touch the
        // dependents vec. Hard to observe directly, but we can
        // verify the dependents list stays exactly as it was (no
        // duplicates introduced by a redundant push).
        let rt = Runtime::new();
        let input = rt.create_input::<u64>(1);
        let q = rt.create_query::<u64, _>(move |rt| rt.get(input) + 10);

        assert_eq!(rt.get(q), 11);
        let dependents_before = collect_dependents_for_slot(&rt, input.slot());
        assert_eq!(dependents_before.len(), 1);
        assert_eq!(dependents_before[0].0, q.slot());

        // Change the input to a different value. Same deps on
        // recompute, but this exercises the dep-diff fast path.
        rt.set(input, 5);
        assert_eq!(rt.get(q), 15);

        let dependents_after = collect_dependents_for_slot(&rt, input.slot());
        assert_eq!(
            dependents_after.len(),
            1,
            "static recompute must not duplicate reverse edges: {:?}",
            dependents_after
        );
        assert_eq!(dependents_after[0].0, q.slot());
    }

    #[test]
    fn adding_a_dep_on_recompute_creates_new_reverse_edge() {
        // Query starts reading just `a`. After flag flip it reads
        // both `a` and `b`. Verify b's dependents gains q.
        let rt = Runtime::new();
        let flag = rt.create_input::<bool>(false);
        let a = rt.create_input::<u64>(10);
        let b = rt.create_input::<u64>(20);
        let q = rt.create_query::<u64, _>(move |rt| {
            let av = rt.get(a);
            if rt.get(flag) {
                av + rt.get(b)
            } else {
                av
            }
        });

        assert_eq!(rt.get(q), 10);
        // Only a (and flag) are deps; b is not yet.
        assert!(!collect_dependents_for_slot(&rt, b.slot())
            .iter()
            .any(|d| d.0 == q.slot()));

        rt.set(flag, true);
        assert_eq!(rt.get(q), 30);
        // Now b should be a dep.
        assert!(collect_dependents_for_slot(&rt, b.slot())
            .iter()
            .any(|d| d.0 == q.slot()));
    }

    #[test]
    fn dep_list_shrinking_across_inline_overflow_boundary() {
        // Exercise the inline→overflow and overflow→inline
        // transitions in replace_deps. First compute reads 10
        // inputs (spills to overflow). Recompute after a conditional
        // flip reads only 3 inputs (fits in inline). The old
        // overflow box must be freed.
        let rt = Runtime::new();
        let flag = rt.create_input::<bool>(true);
        let mut inputs: Vec<Incr<u64>> = Vec::new();
        for i in 0..10 {
            inputs.push(rt.create_input::<u64>(i));
        }
        let inputs_for_closure = inputs.clone();
        let q = rt.create_query::<u64, _>(move |rt| {
            if rt.get(flag) {
                // Wide read: 10 inputs spill to overflow dep list.
                let mut sum = 0;
                for inp in &inputs_for_closure {
                    sum += rt.get(*inp);
                }
                sum
            } else {
                // Narrow read: 3 inputs fit inline.
                rt.get(inputs_for_closure[0])
                    + rt.get(inputs_for_closure[1])
                    + rt.get(inputs_for_closure[2])
            }
        });

        // First compute: 10 deps + flag = 11 deps, overflow.
        let expected_sum: u64 = (0..10u64).sum();
        assert_eq!(rt.get(q), expected_sum);
        assert_eq!(collect_deps_for_slot(&rt, q.slot()).len(), 11);

        // Flip to narrow. Recompute reads flag + 3 deps = 4 deps,
        // fits inline. The old overflow DepList is reclaimed.
        rt.set(flag, false);
        assert_eq!(rt.get(q), 0 + 1 + 2);
        assert_eq!(collect_deps_for_slot(&rt, q.slot()).len(), 4);

        // Go back to wide. Reallocate overflow.
        rt.set(flag, true);
        assert_eq!(rt.get(q), expected_sum);
        assert_eq!(collect_deps_for_slot(&rt, q.slot()).len(), 11);
    }
}
