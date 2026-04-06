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
use std::collections::HashMap;
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
// Introspection types.
// ---------------------------------------------------------------------------

/// Whether a node is an input or a computed value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeKindInfo {
    Input,
    Compute,
}

/// Structural metadata about a single node, for visualization/debugging.
#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub slot: u32,
    pub kind: NodeKindInfo,
    pub label: String,
    pub dependencies: Vec<u32>,
    pub dependents: Vec<u32>,
}

/// What happened to a node during a traced get() call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceAction {
    /// Node was verified clean without recomputing.
    VerifiedClean,
    /// Node was recomputed; `value_changed` is false when early cutoff applied.
    Recomputed { value_changed: bool },
}

/// Trace entry for a single node during propagation.
#[derive(Clone, Debug)]
pub struct NodeTrace {
    pub slot: u32,
    pub action: TraceAction,
}

/// Summary of what happened during a single get() call.
#[derive(Clone, Debug)]
pub struct PropagationTrace {
    pub target: u32,
    pub total_nodes: usize,
    pub nodes_recomputed: usize,
    pub nodes_cutoff: usize,
    pub elapsed_ns: u64,
    pub node_traces: Vec<NodeTrace>,
}

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

/// Type-erased compute closure for a query node. Returns `true` if the
/// new value differs from the previous arena value (so the runtime
/// should bump `changed_at`), `false` if local early cutoff applied.
/// The closure owns all T-specific work (user compute, early-cutoff
/// comparison, arena write) so `run_compute` can be non-generic.
type ComputeFn = dyn Fn(&Runtime, u32, bool) -> bool + Send + Sync;

/// Runtime state behind a single `RwLock`. Three parallel Vecs
/// indexed by node slot, grown together via `append_node`.
struct RuntimeInner {
    /// `None` for input nodes; `Some(arc)` for query nodes.
    compute_fns: Vec<Option<Arc<ComputeFn>>>,
    /// Forward edges: `dependents[slot]` lists nodes that depend on
    /// `slot`. Spec section 5.1's parallel dependents vec.
    dependents: Vec<Vec<NodeId>>,
    /// Stashed panic message for nodes in the Failed state, else
    /// None. Cleared by the dirty walk on Failed → Dirty.
    failure_messages: Vec<Option<String>>,
    /// Optional display labels for nodes, keyed by slot.
    labels: HashMap<u32, String>,
    /// When true, get_traced() can record trace events (stub for now).
    tracing: bool,
}

/// The v2 incremental computation runtime.
pub struct Runtime {
    id: RuntimeId,
    /// Lock-free segmented node store. Readers get direct `&NodeData`
    /// via Acquire on `len`; writers serialize through `write_mutex`.
    nodes: SegmentedNodes,
    inner: RwLock<RuntimeInner>,
    registry: ArenaRegistry,
    /// Monotonic revision counter. Bumped on every `set`; used by
    /// the post-compute revision check to detect writer races.
    revision: AtomicU64,
    /// Serializes all writers (`create_*`, `set`).
    write_mutex: Mutex<()>,
}

impl Runtime {
    /// Construct a new runtime with a fresh identity.
    pub fn new() -> Self {
        let registry = ArenaRegistry::new();
        Self {
            id: registry.id(),
            nodes: SegmentedNodes::new(),
            inner: RwLock::new(RuntimeInner {
                compute_fns: Vec::new(),
                dependents: Vec::new(),
                failure_messages: Vec::new(),
                labels: HashMap::new(),
                tracing: false,
            }),
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
    /// The value is memoized until an upstream input changes.
    pub fn create_query<T, F>(&self, compute: F) -> Incr<T>
    where
        T: Value,
        F: Fn(&Runtime) -> T + Send + Sync + 'static,
    {
        let _guard = self
            .write_mutex
            .lock()
            .expect("runtime write mutex poisoned");

        let arena_slot = T::reserve_empty(self.arena_for::<T>());
        let node = NodeData::new_query(0, arena_slot);

        // Wrap the user closure in a type-erased adapter that owns all
        // T-specific post-compute work so `run_compute` can be
        // non-generic. `try_read` handles the Failed→Dirty retry case
        // where the previous compute panicked before writing.
        let erased: Arc<ComputeFn> = Arc::new(
            move |rt: &Runtime, _slot: u32, is_recompute: bool| -> bool {
                let new_value: T = compute(rt);
                let arena = rt.arena_for::<T>();
                let value_changed = if is_recompute {
                    match T::try_read(arena, arena_slot) {
                        Some(old_value) => old_value != new_value,
                        None => true,
                    }
                } else {
                    true
                };
                if value_changed {
                    T::write(arena, arena_slot, new_value);
                }
                value_changed
            },
        );

        let slot = self.append_node(node, Some(erased));
        Incr::new(slot, 0, self.id)
    }

    /// Read the value of a node. Fast path is a lock-free Clean check;
    /// slow path delegates to the type-erased `ensure_clean` walker.
    pub fn get<T>(&self, handle: Incr<T>) -> T
    where
        T: Value,
    {
        self.check_runtime(handle);
        self.check_cycle_and_record_dep(handle.slot());

        let node = self.nodes.get(handle.slot());
        node.verify_handle(handle, self.id)
            .unwrap_or_else(|e| panic!("{}", e));
        if node.state() == NodeState::Clean {
            let arena_slot = node.arena_slot();
            return T::read(self.arena_for::<T>(), arena_slot);
        }

        self.ensure_clean(handle.slot());
        let arena_slot = self.nodes.get(handle.slot()).arena_slot();
        T::read(self.arena_for::<T>(), arena_slot)
    }

    /// Iterative post-order walker: drive the node at `target` to
    /// Clean. Each stack entry is `(slot, visited)`. Unvisited: push
    /// self back as visited, then push not-yet-Clean deps. Visited:
    /// deps are Clean, run the compute. Cycle detection in user dep
    /// graphs still flows through `COMPUTE_STACK` via
    /// `check_cycle_and_record_dep`; this walker can't hit a cycle
    /// because the dep graph is a DAG by construction.
    fn ensure_clean(&self, target: u32) {
        if self.nodes.get(target).state() == NodeState::Clean {
            return;
        }

        let mut stack: Vec<(u32, bool)> = Vec::with_capacity(16);
        stack.push((target, false));

        while let Some((slot, visited)) = stack.pop() {
            if visited {
                self.compute_slot_via_walker(slot);
                continue;
            }

            let node = self.nodes.get(slot);
            match node.state() {
                NodeState::Clean => {}
                NodeState::Failed => self.panic_with_failure(slot),
                NodeState::Computing => {
                    std::hint::spin_loop();
                    stack.push((slot, false));
                }
                NodeState::New | NodeState::Dirty => {
                    stack.push((slot, true));
                    node.for_each_dep(|dep| {
                        if self.nodes.get(dep.0).state() != NodeState::Clean {
                            stack.push((dep.0, false));
                        }
                    });
                }
            }
        }
    }

    /// Post-order-visit handler for `ensure_clean`. Loops until the
    /// slot is observably Clean: `run_compute` may transition back
    /// to Dirty when the commit-P revision check detects a
    /// concurrent writer race, in which case we retry.
    fn compute_slot_via_walker(&self, slot: u32) {
        loop {
            let state = self.nodes.get(slot).state();
            match state {
                NodeState::Clean => return,
                NodeState::Failed => self.panic_with_failure(slot),
                NodeState::Computing => std::hint::spin_loop(),
                NodeState::New | NodeState::Dirty => {
                    let claimed = self
                        .nodes
                        .get(slot)
                        .state_cell()
                        .try_claim_compute()
                        .is_ok();
                    if claimed {
                        self.run_compute(slot, state == NodeState::Dirty);
                        continue;
                    }
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Panic with a Failed node's stashed compute-closure message.
    #[cold]
    fn panic_with_failure(&self, slot: u32) -> ! {
        let msg = {
            let inner = self.inner.read().expect("runtime inner lock poisoned");
            inner.failure_messages[slot as usize].clone()
        };
        panic!(
            "v2 runtime: node at slot {} is Failed: {}",
            slot,
            msg.as_deref().unwrap_or("unknown failure")
        );
    }

    /// Update an input node's value. Panics if the handle refers to a
    /// query node. Writer/reader exclusion on the arena slot is
    /// handled by the Value trait (atomic for primitives, per-slot
    /// mutex for generics); `write_mutex` only serializes writers.
    pub fn set<T>(&self, handle: Incr<T>, value: T)
    where
        T: Value,
    {
        self.check_runtime(handle);
        let _guard = self
            .write_mutex
            .lock()
            .expect("runtime write mutex poisoned");

        let node = self.nodes.get(handle.slot());
        node.verify_handle(handle, self.id)
            .unwrap_or_else(|e| panic!("{}", e));
        assert!(
            self.inner
                .read()
                .expect("runtime inner lock poisoned")
                .compute_fns[handle.slot() as usize]
                .is_none(),
            "set() called on a query node; only input nodes may be set"
        );
        let arena_slot = node.arena_slot();

        // Early cutoff: same-value set is a no-op.
        let current: T = T::read(self.arena_for::<T>(), arena_slot);
        if current == value {
            return;
        }

        T::write(self.arena_for::<T>(), arena_slot, value);
        let new_revision = self.revision.fetch_add(1, Ordering::Relaxed) + 1;
        node.set_changed_at(new_revision);
        node.set_verified_at(new_revision);
        // State was and remains Clean; the Release store anchors the
        // arena write and revision bumps for readers that Acquire
        // Clean.
        node.state_cell().store_release(NodeState::Clean);
        self.mark_dependents_dirty(handle.slot());
    }

    // -- internal helpers --------------------------------------------------

    /// Append a new node to the store and the parallel vecs in
    /// `inner`, returning the slot. Caller must hold `write_mutex`.
    fn append_node(&self, node: NodeData, compute: Option<Arc<ComputeFn>>) -> u32 {
        let slot = self.nodes.push(node);

        let mut inner = self.inner.write().expect("runtime inner lock poisoned");
        debug_assert_eq!(slot as usize, inner.compute_fns.len());
        debug_assert_eq!(slot as usize, inner.dependents.len());
        debug_assert_eq!(slot as usize, inner.failure_messages.len());
        inner.compute_fns.push(compute);
        inner.dependents.push(Vec::new());
        inner.failure_messages.push(None);
        slot
    }

    /// Check that a handle's runtime id matches this runtime. Must be
    /// called before any code that dereferences `handle.slot()`, since
    /// a cross-runtime handle's slot may be out of bounds in this
    /// runtime's nodes vec. Runs before any index operation so the
    /// user sees the actual cross-runtime diagnostic rather than an
    /// opaque index-out-of-bounds panic.
    #[inline]
    fn check_runtime<T: 'static>(&self, handle: Incr<T>) {
        if handle.runtime_id() != self.id {
            panic!(
                "Incr handle from runtime {:?} used with runtime {:?}",
                handle.runtime_id(),
                self.id
            );
        }
    }

    /// Combined cycle detection and dep recording for the `get` hot
    /// path. Hot path is an empty-stack early return (one RefCell
    /// borrow). On a non-empty stack, walks all frames for cycles
    /// (spec section 9), then pushes `slot` onto the top frame's
    /// deps if it belongs to this runtime and isn't a self-read.
    #[inline]
    fn check_cycle_and_record_dep(&self, slot: u32) {
        COMPUTE_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if stack.is_empty() {
                return;
            }
            for frame in stack.iter() {
                if frame.runtime_id == self.id && frame.node_slot == slot {
                    panic!(
                        "CycleError: dependency cycle detected: node at slot {} \
                         is already computing on this thread",
                        slot
                    );
                }
            }
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
    /// state to Clean (or Dirty if a concurrent writer raced). The
    /// caller must have already CAS'd the state from New or Dirty to
    /// Computing via `try_claim_compute`, and `ensure_clean`'s
    /// iterative walker guarantees every dep is already Clean by the
    /// time we get here.
    ///
    /// On Dirty recomputes, red-green runs before anything else:
    /// load each dep's `changed_at` and compare against our
    /// `verified_at`. If nothing moved, skip the closure entirely,
    /// bump `verified_at`, Release Clean. This is the spec's section
    /// 14 transitive early cutoff. Red-green must precede the
    /// `inner.read()` Arc clone and frame push so short-circuits pay
    /// neither cost.
    fn run_compute(&self, slot: u32, is_recompute: bool) {
        let revision_at_start = self.revision.load(Ordering::Relaxed);

        // Red-green short-circuit: only on recompute, only when no
        // dep has moved since our last verification.
        if is_recompute {
            let node = self.nodes.get(slot);
            let my_verified = node.verified_at();
            let mut any_dep_changed = false;
            node.for_each_dep(|dep| {
                if any_dep_changed {
                    return;
                }
                if self.nodes.get(dep.0).changed_at() > my_verified {
                    any_dep_changed = true;
                }
            });

            if !any_dep_changed {
                let revision_at_end = self.revision.load(Ordering::Relaxed);
                if revision_at_end != revision_at_start {
                    // Writer raced during our red-green walk; go
                    // Dirty and let the next reader retry.
                    node.state_cell().store_release(NodeState::Dirty);
                    return;
                }
                // Do NOT touch `changed_at`: downstream red-green
                // checks need to see it unchanged so they too can
                // short-circuit.
                node.set_verified_at(revision_at_end);
                node.state_cell().store_release(NodeState::Clean);
                return;
            }
        }

        // Full compute path.
        let compute = {
            let inner = self.inner.read().expect("runtime inner lock poisoned");
            inner.compute_fns[slot as usize]
                .as_ref()
                .expect("query node has no compute closure")
                .clone()
        };
        self.push_compute_frame(slot);

        let value_changed_result: std::thread::Result<bool> =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (compute)(self, slot, is_recompute)
            }));
        let recorded_deps = self.pop_compute_frame(slot);

        if !is_recompute {
            self.nodes.get(slot).publish_initial_deps(&recorded_deps);
            if !recorded_deps.is_empty() {
                let mut inner = self.inner.write().expect("runtime inner lock poisoned");
                for dep in &recorded_deps {
                    inner.dependents[dep.0 as usize].push(NodeId(slot));
                }
            }
        } else {
            self.update_deps_on_recompute(slot, &recorded_deps);
        }

        let value_changed = match value_changed_result {
            Ok(c) => c,
            Err(panic_payload) => {
                // Stash the message and transition Failed before
                // re-raising. Dep bookkeeping above is preserved so
                // the next upstream change can Failed → Dirty retry.
                let msg = extract_panic_message(&panic_payload);
                {
                    let mut inner = self.inner.write().expect("runtime inner lock poisoned");
                    inner.failure_messages[slot as usize] = Some(msg);
                }
                self.nodes
                    .get(slot)
                    .state_cell()
                    .store_release(NodeState::Failed);
                std::panic::resume_unwind(panic_payload);
            }
        };

        // Post-compute revision check: if a writer raced during the
        // compute, our result may be based on stale inputs. Go Dirty
        // and let the next reader retry; skip the verified_at/
        // changed_at updates.
        let revision_at_end = self.revision.load(Ordering::Relaxed);
        if revision_at_end != revision_at_start {
            self.nodes
                .get(slot)
                .state_cell()
                .store_release(NodeState::Dirty);
            return;
        }

        // Update red-green revisions. `changed_at` only on actual
        // value change so local early cutoff propagates transitively.
        let node = self.nodes.get(slot);
        if value_changed {
            node.set_changed_at(revision_at_end);
        }
        node.set_verified_at(revision_at_end);
        node.state_cell().store_release(NodeState::Clean);
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
            let mut inner = self.inner.write().expect("runtime inner lock poisoned");
            for dep in &added {
                inner.dependents[dep.0 as usize].push(NodeId(slot));
            }
            for dep in &removed {
                inner.dependents[dep.0 as usize].retain(|d| d.0 != slot);
            }
        }
    }

    /// BFS from `changed_slot`'s dependents, transitioning each
    /// reachable query to Dirty (or Failed → Dirty, for retry on
    /// upstream change). Holds one `inner` read guard across the
    /// entire walk to avoid per-node lock acquires. Clean→Dirty is
    /// the common case; other source states (New, Dirty, Computing)
    /// are skipped — Computing races are handled by the
    /// post-compute revision check in `run_compute`. Failed nodes
    /// that transition back to Dirty have their stashed messages
    /// cleared at the end in one batched write.
    fn mark_dependents_dirty(&self, changed_slot: u32) {
        use std::collections::HashSet;
        let mut visited: HashSet<u32> = HashSet::new();
        let mut queue: Vec<u32> = Vec::new();
        let mut cleared_failures: Vec<u32> = Vec::new();

        {
            let inner = self.inner.read().expect("runtime inner lock poisoned");
            for dep in &inner.dependents[changed_slot as usize] {
                if visited.insert(dep.0) {
                    queue.push(dep.0);
                }
            }

            while let Some(slot) = queue.pop() {
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

                // Walk forward regardless of whether we transitioned
                // this node: dependents below may still be Clean and
                // need marking.
                for child in &inner.dependents[slot as usize] {
                    if visited.insert(child.0) {
                        queue.push(child.0);
                    }
                }
            }
        }

        if !cleared_failures.is_empty() {
            let mut inner = self.inner.write().expect("runtime inner lock poisoned");
            for slot in cleared_failures {
                inner.failure_messages[slot as usize] = None;
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

    // -- Introspection API ---------------------------------------------------

    /// Return the number of nodes currently registered in this runtime.
    pub fn node_count(&self) -> usize {
        self.nodes.len() as usize
    }

    /// Assign a human-readable label to a node slot for visualization/debugging.
    pub fn set_label(&self, slot: u32, label: String) {
        self.inner
            .write()
            .expect("runtime inner lock poisoned")
            .labels
            .insert(slot, label);
    }

    /// Enable or disable execution tracing. When enabled, `get_traced`
    /// can in principle record which nodes were visited; the current
    /// implementation is a stub that stores the flag but does not yet
    /// record per-node trace events.
    pub fn set_tracing(&self, enabled: bool) {
        self.inner
            .write()
            .expect("runtime inner lock poisoned")
            .tracing = enabled;
    }

    /// Like `get`, but also returns a `PropagationTrace` describing the
    /// propagation. The trace fields `nodes_recomputed`, `nodes_cutoff`,
    /// and `node_traces` are currently stubs (zero/empty); full trace
    /// recording is deferred until the dashboard demo requires it.
    pub fn get_traced<T: Value>(&self, handle: Incr<T>) -> (T, PropagationTrace) {
        let value = self.get(handle);
        let trace = PropagationTrace {
            target: handle.slot(),
            total_nodes: self.node_count(),
            nodes_recomputed: 0,
            nodes_cutoff: 0,
            elapsed_ns: 0,
            node_traces: Vec::new(),
        };
        (value, trace)
    }

    /// Return structural metadata about every node in the graph. Useful
    /// for visualizing the dependency graph in the dashboard demo.
    pub fn graph_snapshot(&self) -> Vec<NodeInfo> {
        let inner = self.inner.read().expect("runtime inner lock poisoned");
        let count = inner.compute_fns.len();
        let mut infos = Vec::with_capacity(count);
        for slot in 0..count {
            let is_compute = inner.compute_fns[slot].is_some();
            let label = inner
                .labels
                .get(&(slot as u32))
                .cloned()
                .unwrap_or_default();
            let node = self.nodes.get(slot as u32);
            let deps: Vec<u32> = node.collect_deps().iter().map(|d| d.0).collect();
            let dependents: Vec<u32> = inner.dependents[slot].iter().map(|d| d.0).collect();
            infos.push(NodeInfo {
                slot: slot as u32,
                kind: if is_compute {
                    NodeKindInfo::Compute
                } else {
                    NodeKindInfo::Input
                },
                label,
                dependencies: deps,
                dependents,
            });
        }
        infos
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

// Runtime is Send + Sync via its fields: compute closures are bound
// to `Fn + Send + Sync + 'static`, and everything else is either
// atomic or wrapped in RwLock/Mutex.

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

    /// Test-only: read a node's published dependents (forward edges).
    fn collect_dependents_for_slot(rt: &Runtime, slot: u32) -> Vec<super::super::node::NodeId> {
        let inner = rt.inner.read().unwrap();
        inner.dependents[slot as usize].clone()
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
        let inner = rt.inner.read().unwrap();
        inner.failure_messages[slot as usize].clone()
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
        assert_eq!(rt.get(q), 1 + 2);
        assert_eq!(collect_deps_for_slot(&rt, q.slot()).len(), 4);

        // Go back to wide. Reallocate overflow.
        rt.set(flag, true);
        assert_eq!(rt.get(q), expected_sum);
        assert_eq!(collect_deps_for_slot(&rt, q.slot()).len(), 11);
    }

    // -- Introspection API tests --------------------------------------------

    #[test]
    fn node_count_tracks_created_nodes() {
        let rt = Runtime::new();
        assert_eq!(rt.node_count(), 0);
        let _a = rt.create_input::<u64>(1);
        assert_eq!(rt.node_count(), 1);
        let _b = rt.create_input::<u64>(2);
        assert_eq!(rt.node_count(), 2);
        let _q = rt.create_query::<u64, _>(move |rt_inner| rt_inner.get(_a) + rt_inner.get(_b));
        assert_eq!(rt.node_count(), 3);
    }

    #[test]
    fn set_label_and_graph_snapshot_reflect_labels() {
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(10);
        let b = rt.create_query::<u64, _>(move |r| r.get(a) * 2);
        rt.set_label(a.slot(), "input_a".to_string());
        rt.set_label(b.slot(), "double_a".to_string());

        let snapshot = rt.graph_snapshot();
        assert_eq!(snapshot.len(), 2);

        let info_a = snapshot.iter().find(|n| n.slot == a.slot()).unwrap();
        assert_eq!(info_a.label, "input_a");
        assert_eq!(info_a.kind, NodeKindInfo::Input);

        let info_b = snapshot.iter().find(|n| n.slot == b.slot()).unwrap();
        assert_eq!(info_b.label, "double_a");
        assert_eq!(info_b.kind, NodeKindInfo::Compute);
    }

    #[test]
    fn graph_snapshot_includes_edges_after_compute() {
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(5);
        let q = rt.create_query::<u64, _>(move |r| r.get(a) + 1);
        // Force compute so dep edges are recorded.
        assert_eq!(rt.get(q), 6);

        let snapshot = rt.graph_snapshot();
        let info_q = snapshot.iter().find(|n| n.slot == q.slot()).unwrap();
        assert!(info_q.dependencies.contains(&a.slot()));

        let info_a = snapshot.iter().find(|n| n.slot == a.slot()).unwrap();
        assert!(info_a.dependents.contains(&q.slot()));
    }

    #[test]
    fn get_traced_returns_correct_value_and_stub_trace() {
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(7);
        let (val, trace) = rt.get_traced(a);
        assert_eq!(val, 7);
        assert_eq!(trace.target, a.slot());
        assert_eq!(trace.total_nodes, 1);
        assert!(trace.node_traces.is_empty());
    }

    #[test]
    fn set_tracing_does_not_panic() {
        let rt = Runtime::new();
        rt.set_tracing(true);
        rt.set_tracing(false);
    }
}
