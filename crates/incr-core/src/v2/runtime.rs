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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use super::arena::GenericArena;
use super::handle::{Incr, RuntimeId};
use super::node::NodeData;
use super::registry::ArenaRegistry;
use super::state::NodeState;

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

    /// Node storage. `Box<NodeData>` keeps each node at a stable heap
    /// address across vec resizes; the `RwLock` allows concurrent reads
    /// (one briefly per `get`) with exclusive access for the writer
    /// (one briefly per `create_*` append).
    ///
    /// `clippy::vec_box` suggests collapsing `Vec<Box<NodeData>>` into
    /// `Vec<NodeData>`, but that would invalidate pointers into nodes
    /// on every `Vec` reallocation, which this design relies on
    /// (`NodeData` carries atomics that cannot be relocated while
    /// another thread is observing them). The Box is load-bearing.
    #[allow(clippy::vec_box)]
    nodes: RwLock<Vec<Box<NodeData>>>,

    /// Compute closures for query nodes. Indexed the same way as
    /// `nodes`. `None` for input nodes. `Arc` lets `run_compute` extract
    /// the closure under a short-lived read guard and invoke it after
    /// the guard is dropped, so nested `get` calls do not reenter the
    /// same lock.
    compute_fns: RwLock<Vec<Option<Arc<ComputeFn>>>>,

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
            nodes: RwLock::new(Vec::new()),
            compute_fns: RwLock::new(Vec::new()),
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
    pub fn create_input<T>(&self, initial: T) -> Incr<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        let _guard = self
            .write_mutex
            .lock()
            .expect("runtime write mutex poisoned");
        let revision = self.revision.load(Ordering::Relaxed);

        let arena_slot = self.with_generic_arena::<T, _, _>(|arena| arena.reserve_with(initial));

        let node = Box::new(NodeData::new_input(0, arena_slot, revision));

        let slot = self.append_node(node, None);
        Incr::new(slot, 0, self.id)
    }

    /// Create a query node whose value is produced by running `compute`.
    ///
    /// The closure runs lazily on the first `get` and its result is
    /// memoized. In this commit there is no dependency tracking, so
    /// setting an input that a query "conceptually depends on" does
    /// not invalidate the query's cached value. Reactivity lands in a
    /// subsequent commit.
    pub fn create_query<T, F>(&self, compute: F) -> Incr<T>
    where
        T: Clone + Send + Sync + 'static,
        F: Fn(&Runtime) -> T + Send + Sync + 'static,
    {
        let _guard = self
            .write_mutex
            .lock()
            .expect("runtime write mutex poisoned");

        // Reserve an empty slot in the arena; first compute will
        // populate it.
        let arena_slot = self.with_generic_arena::<T, _, _>(|arena| arena.reserve());

        let node = Box::new(NodeData::new_query(0, arena_slot));

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
        T: Clone + Send + Sync + 'static,
    {
        self.check_runtime(handle);
        loop {
            // Fast path (Clean): hold nodes.read() through the arena
            // read so no writer can mutate the slot in between.
            {
                let nodes = self.nodes.read().expect("nodes lock poisoned");
                let node = &nodes[handle.slot() as usize];
                node.verify_handle(handle, self.id)
                    .unwrap_or_else(|e| panic!("{}", e));
                if node.state() == NodeState::Clean {
                    let arena_slot = node.arena_slot();
                    let value: T =
                        self.with_generic_arena::<T, _, _>(|arena| arena.read(arena_slot));
                    return value;
                }
            }

            // Slow path: not Clean. Release the nodes guard before
            // calling back into the runtime (CAS, compute, spin) so
            // std::sync::RwLock is not reentered on this thread (std
            // RwLock is not guaranteed reentrant).
            let state = {
                let nodes = self.nodes.read().expect("nodes lock poisoned");
                nodes[handle.slot() as usize].state()
            };
            match state {
                NodeState::Clean => {
                    // Raced with a compute completion between our fast
                    // path check and here. Loop to re-enter the fast path.
                }
                NodeState::New => {
                    // Try to CAS from New to Computing. If we win we
                    // own the compute; if we lose, another thread won
                    // and will transition to Clean shortly.
                    let claimed = {
                        let nodes = self.nodes.read().expect("nodes lock poisoned");
                        let node = &nodes[handle.slot() as usize];
                        node.state_cell().try_claim_compute().is_ok()
                    };
                    if claimed {
                        self.run_compute::<T>(handle.slot());
                        // Loop to re-read and return the Clean value.
                    } else {
                        std::hint::spin_loop();
                    }
                }
                NodeState::Computing => {
                    // Another thread (or a prior CAS from this thread
                    // in a nested call) is running the compute. Spin.
                    std::hint::spin_loop();
                }
                NodeState::Dirty | NodeState::Failed => {
                    panic!(
                        "v2 skeletal runtime: encountered {:?} state, but dirty/failed \
                         are not supported in this commit. Reactivity and error \
                         handling land in subsequent commits.",
                        state
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
    /// The RwLock gate is a commit-F fallback: the spec's intended
    /// model is lock-free reads via a segmented nodes store, where
    /// reader/writer synchronization on an input slot is carried by
    /// the dirty walk (which marks dependent queries Dirty with
    /// Release). That model assumes input values are reached through
    /// dependent queries, not via direct `rt.get(input_handle)` from
    /// a concurrent reader, but the public API permits direct reads
    /// and therefore the runtime has to handle them. The segmented
    /// store + hazard / RCU approach that replaces this gate lands
    /// in a later commit; until then, nodes RwLock is the honest
    /// correctness mechanism.
    pub fn set<T>(&self, handle: Incr<T>, value: T)
    where
        T: Clone + Send + Sync + 'static,
    {
        self.check_runtime(handle);
        let _guard = self
            .write_mutex
            .lock()
            .expect("runtime write mutex poisoned");

        // Take nodes.write() to exclude all concurrent readers for
        // the duration of the arena mutation below. See the method
        // docs for the reasoning.
        let nodes = self.nodes.write().expect("nodes lock poisoned");
        let node = &nodes[handle.slot() as usize];
        node.verify_handle(handle, self.id)
            .unwrap_or_else(|e| panic!("{}", e));
        assert!(
            self.compute_fns.read().expect("compute lock poisoned")[handle.slot() as usize]
                .is_none(),
            "set() called on a query node; only input nodes may be set"
        );
        let arena_slot = node.arena_slot();

        // Arena write happens while holding nodes.write(). No reader
        // can be inside a concurrent arena.read on the same slot,
        // because readers require nodes.read() which is mutually
        // exclusive with nodes.write().
        self.with_generic_arena::<T, _, _>(|arena| arena.write(arena_slot, value));

        // State was already Clean and stays Clean; this Release store
        // is a no-op semantically but anchors the memory ordering
        // contract for the future lock-free variant. When the RwLock
        // gate is removed, readers will Acquire-load state to gate
        // their arena access and this Release is the publish side.
        node.state_cell().store_release(NodeState::Clean);

        drop(nodes);
        self.revision.fetch_add(1, Ordering::Relaxed);
    }

    // -- internal helpers --------------------------------------------------

    /// Append a new node to the store and parallel compute_fns vec,
    /// returning the slot. Called from the write-mutex-guarded paths.
    fn append_node(&self, node: Box<NodeData>, compute: Option<Arc<ComputeFn>>) -> u32 {
        let mut nodes = self.nodes.write().expect("nodes lock poisoned");
        let mut computes = self.compute_fns.write().expect("compute lock poisoned");
        debug_assert_eq!(nodes.len(), computes.len());
        let slot = nodes.len() as u32;
        nodes.push(node);
        computes.push(compute);
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

    /// Run the compute closure for a query node and transition its
    /// state to Clean. The caller must have already CAS'd the state
    /// from New to Computing.
    fn run_compute<T>(&self, slot: u32)
    where
        T: Clone + Send + Sync + 'static,
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

        // Run the closure outside any lock.
        let value_box: Box<dyn Any + Send + Sync> = (compute)(self);

        // Downcast to the caller's expected type. A mismatch here is a
        // bug in how `create_query` packed the closure; it should be
        // impossible from safe user code because `create_query<T, F>`
        // binds T at the call site.
        let value: T = *value_box
            .downcast::<T>()
            .expect("compute function returned wrong type");

        // Look up the arena slot for this node, then write and publish.
        let arena_slot = {
            let nodes = self.nodes.read().expect("nodes lock poisoned");
            nodes[slot as usize].arena_slot()
        };

        self.with_generic_arena::<T, _, _>(|arena| arena.write(arena_slot, value));

        // Release-store state to Clean. This publishes the arena write
        // (Relaxed) together with the state transition (Release) to
        // readers that Acquire-load state afterward.
        {
            let nodes = self.nodes.read().expect("nodes lock poisoned");
            nodes[slot as usize]
                .state_cell()
                .store_release(NodeState::Clean);
        }
    }

    /// Look up (creating if necessary) the `GenericArena<T>` for T and
    /// pass it to the closure. The closure runs with a reference to the
    /// arena that is valid for the registry's lifetime.
    fn with_generic_arena<T, F, R>(&self, f: F) -> R
    where
        T: Clone + Send + Sync + 'static,
        F: FnOnce(&GenericArena<T>) -> R,
    {
        let arena_ptr = self
            .registry
            .ensure_arena::<T, _>(|| Box::new(GenericArena::<T>::new()));
        // SAFETY: `arena_ptr` was returned by the registry and is
        // stable for the registry's lifetime (arenas are never removed
        // and each arena lives at a fixed heap address via Box).
        // The type assertion via downcast_ref is enforced by matching
        // TypeId, so we cannot accidentally pull a differently-typed
        // arena out from under the T parameter.
        let arena: &GenericArena<T> = unsafe {
            (*arena_ptr)
                .as_any()
                .downcast_ref::<GenericArena<T>>()
                .expect("arena type mismatch; registry invariant violated")
        };
        f(arena)
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
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
    #[allow(non_snake_case)]
    fn query_memoization_is_NOT_reactive_in_this_commit() {
        // Document the explicit scope limit: setting an input does not
        // invalidate queries that "depend on" it, because there is no
        // dependency tracking in the skeletal runtime. This test
        // exists so that a future commit introducing reactivity will
        // trip it and force an author to decide whether to update or
        // delete this test.
        let rt = Runtime::new();
        let a = rt.create_input::<u64>(1);
        let q = rt.create_query::<u64, _>(move |rt| rt.get(a) * 10);
        assert_eq!(rt.get(q), 10);
        rt.set(a, 7);
        // No dirty walk: q still memoizes the first result.
        assert_eq!(
            rt.get(q),
            10,
            "skeletal runtime is non-reactive by design; this assertion flips when dep tracking lands"
        );
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
}
