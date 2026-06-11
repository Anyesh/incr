//! `Runtime<C>`: the strategy-parameterized incremental computation engine.
//!
//! Single struct, single `impl` block, monomorphized at compile time into
//! the single-threaded variant (`Runtime<Local>`) and the concurrent
//! variant (`Runtime<Shared>`). The user-facing crates `incr-compute` and
//! `incr-concurrent` re-export the appropriate alias.
//!
//! This first slice ships the core algorithm:
//! - `create_input<T>` / `create_query<T, F>`: node construction.
//! - `get<T>(handle)` / `set<T>(handle, value)`: the user-facing API.
//! - `ensure_clean`: iterative post-order walker that recomputes dirty
//!   nodes in topological order.
//! - `run_compute`: claim Computing, run the closure, observe new deps,
//!   update edges, Release Clean. Includes red/green early cutoff.
//! - `mark_dependents_dirty`: BFS dirty walk from a mutated input.
//!
//! Deferred to follow-ups: handle validation (runtime_id + generation
//! checks), introspection (graph_snapshot, labels), real tracing,
//! collection operators, soundness fixes (race-detection ordering with
//! AcqRel, overflow-dep reclamation).

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Arc;

use crate::arena_registry::ArenaRegistry;
use crate::cells::Cells;
use crate::dep_stack::DepStack;
use crate::generic_arena::GenericArena;
use crate::handle::{Incr, RuntimeId};
use crate::locks::Lock;
use crate::node::{NodeData, NodeId};
use crate::segmented_nodes::SegmentedNodes;
use crate::state::{self, NodeState};
use crate::value::Value;

/// Compute closure: takes a borrow of the runtime, the node's slot, and
/// whether this is a recompute (true) or first compute (false). Returns
/// `true` if the value actually changed (for early-cutoff propagation),
/// `false` if it was the same as before.
type ComputeFn<C> = Arc<dyn Fn(&Runtime<C>, u32, bool) -> bool + Send + Sync + 'static>;

/// Per-runtime mutable state guarded by the inner lock. Holds everything
/// that's not on the per-node `NodeData` and not in an arena.
pub(crate) struct Inner<C: Cells> {
    pub(crate) compute_fns: Vec<Option<ComputeFn<C>>>,
    pub(crate) dependents: Vec<Vec<NodeId>>,
    pub(crate) arenas: ArenaRegistry<C>,
    pub(crate) type_tags: HashMap<TypeId, u16>,
    pub(crate) next_type_tag: u16,
    pub(crate) labels: HashMap<u32, String>,
    pub(crate) trace_log: Vec<crate::trace::NodeTrace>,
}

impl<C: Cells> Inner<C> {
    fn new() -> Self {
        Self {
            compute_fns: Vec::new(),
            dependents: Vec::new(),
            arenas: ArenaRegistry::new(),
            type_tags: HashMap::new(),
            next_type_tag: 0,
            labels: HashMap::new(),
            trace_log: Vec::new(),
        }
    }

    fn type_tag_for<T: Value>(&mut self) -> u16 {
        let id = TypeId::of::<T>();
        if let Some(&tag) = self.type_tags.get(&id) {
            return tag;
        }
        let tag = self.next_type_tag;
        self.next_type_tag = self
            .next_type_tag
            .checked_add(1)
            .expect("incr-core: more than u16::MAX distinct value types in one runtime");
        self.type_tags.insert(id, tag);
        tag
    }
}

/// The runtime.
pub struct Runtime<C: Cells> {
    pub(crate) nodes: SegmentedNodes<C>,
    pub(crate) inner: <C as Cells>::Lock<Inner<C>>,
    pub(crate) revision: <C as Cells>::U64,
    pub(crate) dep_stack: <C as Cells>::DepStack,
    pub(crate) runtime_id: RuntimeId,
    /// `1` when `get_traced` is actively recording. Checked on every
    /// `compute_one` via a Relaxed load (~1 ns when disarmed) so the
    /// non-tracing hot path pays no measurable cost.
    pub(crate) tracing_armed: <C as Cells>::U8,
}

impl<C: Cells> Default for Runtime<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Cells> Runtime<C> {
    pub fn new() -> Self {
        Self {
            nodes: SegmentedNodes::new(),
            inner: <<C as Cells>::Lock<Inner<C>> as Lock<Inner<C>>>::new(Inner::new()),
            revision: C::new_u64(1),
            dep_stack: <C::DepStack as DepStack>::new(),
            runtime_id: RuntimeId::allocate(),
            tracing_armed: C::new_u8(0),
        }
    }

    #[inline(always)]
    fn tracing_is_armed(&self) -> bool {
        C::u8_load_relaxed(&self.tracing_armed) == 1
    }

    fn record_trace(&self, id: NodeId, action: crate::trace::TraceAction) {
        if self.tracing_is_armed() {
            self.inner
                .write()
                .trace_log
                .push(crate::trace::NodeTrace { id, action });
        }
    }

    #[inline]
    fn current_revision(&self) -> u64 {
        C::u64_load_acquire(&self.revision)
    }

    fn bump_revision(&self) -> u64 {
        // fetch_add, not load+store: two concurrent set() calls must not
        // collapse into one revision or the cutoff protocol conflates
        // their changes.
        C::u64_fetch_add(&self.revision, 1) + 1
    }

    /// Create an input node with an initial value.
    pub fn create_input<T: Value>(&self, value: T) -> Incr<T> {
        assert!(
            !self.dep_stack.current_frame_active(),
            "create_input called during compute; not permitted",
        );
        self.create_input_unchecked(value)
    }

    /// Internal: create an input without the dep-stack-empty check.
    /// Used by operators like `group_by` that need to allocate
    /// sub-collection version nodes lazily from inside a compute closure.
    /// The caller is responsible for ensuring the new node is not a dep
    /// of the currently-computing node (i.e., the new node is downstream
    /// of the operator, not upstream).
    pub(crate) fn create_input_unchecked<T: Value>(&self, value: T) -> Incr<T> {
        let revision = self.current_revision();
        let (slot, type_tag, generation) = {
            let mut inner = self.inner.write();
            let type_tag = inner.type_tag_for::<T>();
            let arena = inner.arenas.ensure_arena::<T>();
            let arena_slot = arena.reserve_with(value);
            let node = NodeData::<C>::new_input(type_tag, arena_slot, revision);
            let slot = self.nodes.push(node);
            inner.compute_fns.push(None);
            inner.dependents.push(Vec::new());
            let generation = self.nodes.get(slot).generation();
            (slot, type_tag, generation)
        };
        let _ = type_tag;
        Incr::new(slot, generation, self.runtime_id)
    }

    /// Create a compute (query) node. Dependencies are tracked
    /// automatically: every `rt.get(other)` call inside `f` records
    /// `other` as a dep.
    pub fn create_query<T, F>(&self, f: F) -> Incr<T>
    where
        T: Value,
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        assert!(
            !self.dep_stack.current_frame_active(),
            "create_query called during compute; not permitted",
        );

        let (slot, generation, _type_tag) = {
            let mut inner = self.inner.write();
            let type_tag = inner.type_tag_for::<T>();
            let arena = inner.arenas.ensure_arena::<T>();
            let arena_slot = arena.reserve();
            let node = NodeData::<C>::new_query(type_tag, arena_slot);
            let slot = self.nodes.push(node);

            // Compute closure: invokes f, writes value, returns whether
            // the value changed (for early cutoff). `is_recompute=false`
            // means there's no prior value, so we always treat it as
            // changed; `is_recompute=true` compares against the stored
            // value via T's PartialEq.
            let arena_inner = arena.clone();
            let compute: ComputeFn<C> = Arc::new(
                move |rt: &Runtime<C>, slot: u32, is_recompute: bool| -> bool {
                    let new_value = f(rt);
                    let node = rt.nodes.get(slot);
                    if is_recompute {
                        // Compare-and-publish in one slot session; the
                        // false return is the early cutoff.
                        arena_inner.write_if_changed(node.arena_slot(), new_value)
                    } else {
                        arena_inner.write(node.arena_slot(), new_value);
                        true
                    }
                },
            );

            inner.compute_fns.push(Some(compute));
            inner.dependents.push(Vec::new());
            let generation = self.nodes.get(slot).generation();
            (slot, generation, type_tag)
        };
        Incr::new(slot, generation, self.runtime_id)
    }

    /// Reject handles minted by a different runtime. A real branch, not
    /// a debug_assert: in release a foreign handle would silently read
    /// or write another node's slot in the same type arena, which is
    /// silent data corruption. Cost is one always-predicted compare on
    /// the hot path.
    #[inline(always)]
    fn check_handle<T: Value>(&self, handle: Incr<T>) {
        if handle.runtime_id() != self.runtime_id {
            panic!(
                "incr-core: Incr<{}> handle from a foreign runtime (handle was minted by \
                 runtime {}, used on runtime {})",
                std::any::type_name::<T>(),
                handle.runtime_id().get(),
                self.runtime_id.get(),
            );
        }
    }

    /// Read the current value of a node. Triggers recomputation of the
    /// minimum necessary subgraph if anything is dirty.
    pub fn get<T: Value>(&self, handle: Incr<T>) -> T {
        self.check_handle(handle);
        let slot = handle.slot();

        // Record dep if we're inside a compute closure.
        self.dep_stack.record_dep(NodeId(slot));

        // Ensure clean, then read.
        self.ensure_clean(NodeId(slot));

        let arena = {
            let inner = self.inner.read();
            inner
                .arenas
                .try_arena::<T>()
                .expect("incr-core: arena missing for handle's type; this should be impossible")
        };
        let node = self.nodes.get(slot);
        arena.read(node.arena_slot())
    }

    /// Read the current value and return a propagation trace alongside.
    /// Records per-node events (Recomputed { value_changed } or
    /// VerifiedClean) for every compute or short-circuit that happens
    /// during this `get`.
    pub fn get_traced<T: Value>(&self, handle: Incr<T>) -> (T, crate::trace::PropagationTrace) {
        use crate::trace::TraceAction;

        // Arm tracing: clear any prior log, then flip the gate so
        // compute_one starts appending events.
        {
            let mut inner = self.inner.write();
            inner.trace_log.clear();
        }
        C::u8_store_release(&self.tracing_armed, 1);

        // Disarm on every exit path: a panicking compute inside the
        // traced get must not leave tracing armed forever.
        struct Disarm<'a, C2: Cells>(&'a Runtime<C2>);
        impl<C2: Cells> Drop for Disarm<'_, C2> {
            fn drop(&mut self) {
                C2::u8_store_release(&self.0.tracing_armed, 0);
            }
        }
        let disarm = Disarm(self);

        let start = std::time::Instant::now();
        let value = self.get(handle);
        let elapsed_ns = start.elapsed().as_nanos() as u64;

        drop(disarm);
        let node_traces: Vec<crate::trace::NodeTrace> = {
            let mut inner = self.inner.write();
            std::mem::take(&mut inner.trace_log)
        };

        let nodes_recomputed = node_traces
            .iter()
            .filter(|t| matches!(t.action, TraceAction::Recomputed { .. }))
            .count();
        let nodes_cutoff = node_traces
            .iter()
            .filter(|t| {
                matches!(
                    t.action,
                    TraceAction::Recomputed {
                        value_changed: false
                    }
                )
            })
            .count();

        let trace = crate::trace::PropagationTrace {
            target: NodeId(handle.slot()),
            node_traces,
            total_nodes: self.node_count(),
            nodes_recomputed,
            nodes_cutoff,
            elapsed_ns,
        };
        (value, trace)
    }

    /// Number of nodes in the runtime.
    pub fn node_count(&self) -> usize {
        self.nodes.len() as usize
    }

    /// Assign a human-readable label to a node slot. Surfaces in
    /// `graph_snapshot()` and trace output. Re-assigning replaces.
    pub fn set_label(&self, slot: u32, label: String) {
        self.inner.write().labels.insert(slot, label);
    }

    /// Retrieve the label for a node slot, if any.
    pub fn label(&self, slot: u32) -> Option<String> {
        self.inner.read().labels.get(&slot).cloned()
    }

    /// Structural snapshot of every node. Returns `NodeInfo` with each
    /// node's dependencies (read from inline-7 storage) and dependents
    /// (read from the inner state).
    pub fn graph_snapshot(&self) -> Vec<crate::trace::NodeInfo> {
        use crate::trace::{NodeInfo, NodeKindInfo};
        let inner = self.inner.read();
        let count = self.nodes.len();
        let mut out = Vec::with_capacity(count as usize);
        for slot in 0..count {
            let node = self.nodes.get(slot);
            let kind = if inner
                .compute_fns
                .get(slot as usize)
                .is_some_and(|f| f.is_some())
            {
                NodeKindInfo::Compute
            } else {
                NodeKindInfo::Input
            };
            let mut dependencies = Vec::new();
            node.for_each_dep(|d| dependencies.push(d));
            let dependents = inner
                .dependents
                .get(slot as usize)
                .cloned()
                .unwrap_or_default();
            out.push(NodeInfo {
                id: NodeId(slot),
                kind,
                label: inner.labels.get(&slot).cloned(),
                dependencies,
                dependents,
            });
        }
        out
    }

    /// Set a new value on an input node. Bumps revision and marks all
    /// transitive dependents dirty.
    ///
    /// Panics if the handle refers to a query (compute) node. Setting a
    /// query node would overwrite its computed value and bypass the
    /// state machine; the only valid setter is the compute closure itself.
    pub fn set<T: Value>(&self, handle: Incr<T>, value: T) {
        self.check_handle(handle);
        let slot = handle.slot();

        let (arena, is_query) = {
            let inner = self.inner.read();
            let arena = inner
                .arenas
                .try_arena::<T>()
                .expect("incr-core: arena missing for input handle's type");
            let is_query = inner
                .compute_fns
                .get(slot as usize)
                .map(|f| f.is_some())
                .unwrap_or(false);
            (arena, is_query)
        };

        assert!(
            !is_query,
            "Runtime::set called on a query (compute) node at slot {}; only input nodes can be set",
            slot,
        );

        // No-op if the value is unchanged. Racy against a concurrent
        // set(), but benignly: the no-op linearizes before the other
        // setter.
        let node = self.nodes.get(slot);
        if arena.eq_current(node.arena_slot(), &value) {
            return;
        }

        // Publication order is the protocol's backbone:
        // 1. pending marker, so no verifier can stamp past this change
        //    while the concrete revision is still unknown;
        // 2. value swap, so any thread that later observes the bumped
        //    revision also observes the new value (swap is sequenced
        //    before the AcqRel fetch_add);
        // 3. revision bump;
        // 4. settle changed_at to the concrete revision;
        // 5. dirty walk, whose Release stores publish all of the above
        //    to claimants that Acquire the Dirty state.
        node.mark_changed_pending();
        arena.write(node.arena_slot(), value);
        let new_rev = self.bump_revision();
        node.settle_changed_at(new_rev);
        node.max_verified_at(new_rev);

        self.mark_dependents_dirty(NodeId(slot));
    }

    /// BFS forward walk from `start`'s dependents, marking each Clean
    /// node as Dirty. Every transition is a CAS so the walk can never
    /// clobber a `Computing` claim it raced with.
    fn mark_dependents_dirty(&self, start: NodeId) {
        let mut queue: Vec<NodeId> = {
            let inner = self.inner.read();
            inner.dependents[start.0 as usize].clone()
        };

        while let Some(id) = queue.pop() {
            let node = self.nodes.get(id.0);
            loop {
                let cur = state::load::<C>(node.state_cell());
                match cur {
                    NodeState::Clean | NodeState::Failed => {
                        if state::try_transition::<C>(node.state_cell(), cur, NodeState::Dirty)
                            .is_ok()
                        {
                            let inner = self.inner.read();
                            queue.extend(inner.dependents[id.0 as usize].iter().copied());
                            break;
                        }
                        // The state moved under us (a claim or another
                        // walk); re-examine.
                    }
                    NodeState::Computing => {
                        // An in-flight compute may have read the old
                        // input value and will stamp timestamps that
                        // predate this set. Flag it so its finishing CAS
                        // fails and it lands Dirty, and walk through to
                        // its dependents, which would otherwise stay
                        // Clean against a stale parent.
                        if state::try_transition::<C>(
                            node.state_cell(),
                            NodeState::Computing,
                            NodeState::ComputingDirty,
                        )
                        .is_ok()
                        {
                            let inner = self.inner.read();
                            queue.extend(inner.dependents[id.0 as usize].iter().copied());
                            break;
                        }
                    }
                    NodeState::New | NodeState::Dirty | NodeState::ComputingDirty => {
                        // Already flagged; the walk that flagged it also
                        // covered its dependents.
                        break;
                    }
                }
            }
        }
    }

    /// Ensure the node at `id` is Clean, recomputing the minimum
    /// necessary subgraph.
    fn ensure_clean(&self, id: NodeId) {
        // Fast path: already clean.
        if state::load::<C>(self.nodes.get(id.0).state_cell()) == NodeState::Clean {
            return;
        }

        // Iterative post-order walk. Each stack entry is (node, visited).
        // visited=false: first visit, push self and push dirty deps.
        // visited=true: all deps clean now, run this node's compute.
        let mut work: Vec<(NodeId, bool)> = vec![(id, false)];

        // Cycle backstop for the walk itself. A DAG bounds first-visit
        // expansions by the dirty subgraph's edge count, which dep_count
        // caps at 255 per node plus one self-push each, so a DAG can
        // never exhaust this budget; an all-Dirty cycle re-expands
        // forever and always will. Cycles that pass through a compute
        // are caught exactly in compute_one via the frame stack; this
        // catches the remaining walk-only case.
        let mut budget: u64 = 1024 + 256 * u64::from(self.nodes.len());

        while let Some((cur, visited)) = work.pop() {
            if visited {
                self.compute_one(cur);
                continue;
            }

            budget -= 1;
            if budget == 0 {
                panic!(
                    "incr-core: dependency cycle detected: ensure_clean({}) exceeded its \
                     traversal budget, which no acyclic graph can",
                    id.0,
                );
            }

            let node = self.nodes.get(cur.0);
            match state::load::<C>(node.state_cell()) {
                NodeState::Clean => continue,
                NodeState::Computing | NodeState::ComputingDirty => {
                    // Another thread owns this subtree and is cleaning
                    // its deps itself; expanding them here would tear
                    // through a dep list mid-rewrite. Just wait for the
                    // owner in compute_one.
                    work.push((cur, true));
                }
                _ => {
                    // First visit: push self (to process after deps)
                    // then push any non-clean deps.
                    work.push((cur, true));
                    node.for_each_dep(|dep| {
                        let dep_node = self.nodes.get(dep.0);
                        let dep_state = state::load::<C>(dep_node.state_cell());
                        if dep_state != NodeState::Clean {
                            work.push((dep, false));
                        }
                    });
                }
            }
        }
    }

    /// Compute (or verify) a single node, assuming all its known deps
    /// are already clean. Handles state-machine transitions and red/green
    /// early cutoff.
    ///
    /// Protocol invariants (see the shared-read-protocol-v2 decision):
    /// - every transition is a CAS; nothing ever clobbers a Computing
    ///   claim;
    /// - a thread observing Computing waits for the owner, unless the
    ///   node is on its own frame stack, which is a dependency cycle;
    /// - stamps use the revision captured BEFORE verify/compute via
    ///   fetch_max, so a concurrent set() invalidates rather than masks.
    fn compute_one(&self, id: NodeId) {
        let node = self.nodes.get(id.0);
        let mut spins: u32 = 0;

        loop {
            let observed = state::load::<C>(node.state_cell());
            match observed {
                NodeState::Clean => return,
                NodeState::Computing | NodeState::ComputingDirty => {
                    if self.dep_stack.is_computing(id) {
                        panic!(
                            "incr-core: dependency cycle detected: node {} was read while \
                             its own compute was in progress",
                            id.0,
                        );
                    }
                    // Another thread owns the compute; wait for it.
                    // Bounded spin first (computes are usually short),
                    // then yield. Local never reaches this arm: single-
                    // threaded Computing always means a cycle.
                    //
                    // Known limitation: a cyclic graph whose cycle spans
                    // two threads' in-flight computes spins here forever
                    // instead of panicking; cross-thread cycle detection
                    // needs a waits-for graph and is not implemented.
                    spins += 1;
                    if spins < 64 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                    continue;
                }
                NodeState::Failed => {
                    panic!(
                        "incr-core: node {} read but its last compute panicked; it stays \
                         Failed until a dependency changes (set an upstream input to retry)",
                        id.0,
                    );
                }
                NodeState::New | NodeState::Dirty => {}
            }

            // Distinguish input vs query: inputs don't compute, they
            // just need their state stamped clean.
            let compute = {
                let inner = self.inner.read();
                inner.compute_fns.get(id.0 as usize).and_then(|f| f.clone())
            };
            let compute = match compute {
                Some(f) => f,
                None => {
                    node.max_verified_at(self.current_revision());
                    if state::try_transition::<C>(node.state_cell(), observed, NodeState::Clean)
                        .is_ok()
                    {
                        return;
                    }
                    continue;
                }
            };

            // Capture the revision BEFORE verifying or computing. If a
            // set() bumps it mid-flight, our stamps stay below the new
            // changed_at and the next reader re-verifies; stamping the
            // post-compute revision would mask the change forever.
            let start_rev = self.current_revision();

            // Red/green check: if no dep's changed_at exceeds our
            // verified_at, we can skip the closure entirely. Only valid
            // from Dirty: a New node has nothing to verify against.
            if observed == NodeState::Dirty {
                let my_verified = node.verified_at();
                let mut any_changed = false;
                node.for_each_dep(|dep| {
                    if any_changed {
                        return;
                    }
                    if self.nodes.get(dep.0).changed_at() > my_verified {
                        any_changed = true;
                    }
                });
                if !any_changed {
                    node.max_verified_at(start_rev);
                    if state::try_transition::<C>(
                        node.state_cell(),
                        NodeState::Dirty,
                        NodeState::Clean,
                    )
                    .is_ok()
                    {
                        self.record_trace(id, crate::trace::TraceAction::VerifiedClean);
                        return;
                    }
                    continue;
                }
            }

            // Full compute path. Claim Computing first; the claim source
            // tells us exactly whether this is a recompute.
            let claimed_from = match state::try_claim_compute::<C>(node.state_cell()) {
                Ok(src) => src,
                Err(_) => continue,
            };
            let is_recompute = claimed_from == NodeState::Dirty;

            // Track deps via the strategy's dep stack. The closure runs
            // under a panic boundary: a panicking user closure must not
            // strand the node in Computing (wedging every future reader)
            // or leak its dep frame. AssertUnwindSafe is justified
            // because the only state the closure can reach is the
            // runtime's, and the Failed transition plus the frame pop
            // below restore every invariant the unwind could have
            // interrupted; no lock is held while user code runs.
            self.dep_stack.push_frame(id);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (compute)(self, id.0, is_recompute)
            }));
            let recorded_deps = self.dep_stack.pop_frame();

            let value_changed = match result {
                Ok(changed) => changed,
                Err(payload) => {
                    // Publish the union of the old deps and whatever was
                    // recorded before the panic. Without the recorded
                    // subset, a first-compute panic leaves no reverse
                    // edges at all and no set() can ever flip the node
                    // out of Failed; without the old set, a recompute
                    // panic would lose recovery paths through deps the
                    // failed run had not reached yet. Extra edges only
                    // cause spurious re-verification, never staleness.
                    let mut union_deps: Vec<NodeId> = Vec::with_capacity(recorded_deps.len() + 8);
                    node.for_each_dep(|d| union_deps.push(d));
                    for d in &recorded_deps {
                        if !union_deps.contains(d) {
                            union_deps.push(*d);
                        }
                    }
                    self.publish_deps(id, &union_deps);

                    // verified_at was not raised, so once a dependency
                    // changes, the dirty walk flips Failed -> Dirty and
                    // red/green sees the change and recomputes for real.
                    // If a set() already flagged us mid-compute, land
                    // Dirty now so the next reader retries immediately.
                    if state::try_transition::<C>(
                        node.state_cell(),
                        NodeState::Computing,
                        NodeState::Failed,
                    )
                    .is_err()
                    {
                        state::store::<C>(node.state_cell(), NodeState::Dirty);
                    }
                    std::panic::resume_unwind(payload);
                }
            };

            self.publish_deps(id, &recorded_deps);

            if value_changed || !is_recompute {
                node.max_changed_at(start_rev);
            }
            node.max_verified_at(start_rev);

            // Finish. If a set() flagged us ComputingDirty mid-compute,
            // land Dirty so the next reader recomputes against the newer
            // input; the value we just produced is still a consistent
            // pre-set snapshot, so returning it is linearizable.
            if state::try_transition::<C>(node.state_cell(), NodeState::Computing, NodeState::Clean)
                .is_err()
            {
                state::store::<C>(node.state_cell(), NodeState::Dirty);
            }
            self.record_trace(id, crate::trace::TraceAction::Recomputed { value_changed });
            return;
        }
    }

    /// Record dependencies on the node and update reverse edges in the
    /// inner state. Diffs old vs new deps so static-dep queries (the
    /// common case) skip the inner.write() acquire and the dependents
    /// vector edits on recompute.
    ///
    /// Up to 7 deps live inline; beyond that, they live in a heap-allocated
    /// `DepList`. Old overflow lists are leaked under `Shared` (no
    /// hazard pointers yet); `NodeData::Drop` reclaims the final one.
    fn publish_deps(&self, id: NodeId, new_deps: &[NodeId]) {
        let node = self.nodes.get(id.0);

        // Read old deps before overwriting (the comparison uses the same
        // backing storage we're about to write into, so we MUST collect
        // first). Uses for_each_dep which handles inline and overflow.
        let mut old_deps: Vec<NodeId> = Vec::with_capacity(8);
        node.for_each_dep(|d| old_deps.push(d));
        let old_slice = old_deps.as_slice();

        // Fast path: static deps. The common case for both inputs and
        // long-lived queries is that the dep set does not change between
        // computes. Skip every write if we detect equality.
        if old_slice.len() == new_deps.len()
            && old_slice.iter().zip(new_deps.iter()).all(|(a, b)| a == b)
        {
            return;
        }

        // Slow path: deps changed. Install the new dep list (handles
        // inline + overflow). Any displaced overflow DepList is retired
        // internally through the haphazard global domain so concurrent
        // readers finish their traversal safely before the actual free.
        node.install_deps(new_deps);

        // Reverse-edge diff under the inner write lock. Linear scans
        // for small dep sets are faster than HashSet construction
        // below ~16 items.
        let mut inner = self.inner.write();
        for old_dep in old_slice {
            if !new_deps.contains(old_dep) {
                inner.dependents[old_dep.0 as usize].retain(|&d| d != id);
            }
        }
        for new_dep in new_deps {
            if !old_slice.contains(new_dep) {
                inner.dependents[new_dep.0 as usize].push(id);
            }
        }
    }

    /// Borrow the arena for `T`, panicking if none exists.
    #[allow(dead_code)]
    pub(crate) fn arena<T: Value>(&self) -> Arc<GenericArena<T, C>> {
        let inner = self.inner.read();
        inner
            .arenas
            .try_arena::<T>()
            .expect("incr-core: arena lookup failed for T")
    }
}

// SAFETY: Runtime<Shared> is Send + Sync by composition (SegmentedNodes,
// RwLock, AtomicU64, SharedDepStack, RuntimeId all are). Runtime<Local>
// uses Cell/RefCell-backed cells through the Local strategy and is
// !Send + !Sync by auto-derive. We rely on auto traits here; no manual
// impls needed.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    #[test]
    fn local_create_and_get_input() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(42_u64);
        assert_eq!(rt.get(a), 42);
    }

    #[test]
    fn shared_create_and_get_input() {
        let rt: Runtime<Shared> = Runtime::new();
        let a = rt.create_input(42_u64);
        assert_eq!(rt.get(a), 42);
    }

    #[test]
    fn local_simple_query() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(10_i64);
        let b = rt.create_query(move |rt| rt.get(a) * 2);
        assert_eq!(rt.get(b), 20);
    }

    #[test]
    fn shared_simple_query() {
        let rt: Runtime<Shared> = Runtime::new();
        let a = rt.create_input(10_i64);
        let b = rt.create_query(move |rt| rt.get(a) * 2);
        assert_eq!(rt.get(b), 20);
    }

    #[test]
    fn local_set_propagates() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(10_i64);
        let b = rt.create_query(move |rt| rt.get(a) * 2);
        assert_eq!(rt.get(b), 20);
        rt.set(a, 15);
        assert_eq!(rt.get(b), 30);
    }

    #[test]
    fn shared_set_propagates() {
        let rt: Runtime<Shared> = Runtime::new();
        let a = rt.create_input(10_i64);
        let b = rt.create_query(move |rt| rt.get(a) * 2);
        assert_eq!(rt.get(b), 20);
        rt.set(a, 15);
        assert_eq!(rt.get(b), 30);
    }

    #[test]
    fn local_chain() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(5_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 1);
        let c = rt.create_query(move |rt| rt.get(b) * 2);
        assert_eq!(rt.get(c), 12);
        rt.set(a, 10);
        assert_eq!(rt.get(c), 22);
    }

    #[test]
    fn shared_chain() {
        let rt: Runtime<Shared> = Runtime::new();
        let a = rt.create_input(5_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 1);
        let c = rt.create_query(move |rt| rt.get(b) * 2);
        assert_eq!(rt.get(c), 12);
        rt.set(a, 10);
        assert_eq!(rt.get(c), 22);
    }

    #[test]
    fn local_diamond() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 10);
        let c = rt.create_query(move |rt| rt.get(a) + 100);
        let d = rt.create_query(move |rt| rt.get(b) + rt.get(c));
        assert_eq!(rt.get(d), 112);
        rt.set(a, 2);
        assert_eq!(rt.get(d), 114);
    }

    #[test]
    fn shared_diamond() {
        let rt: Runtime<Shared> = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 10);
        let c = rt.create_query(move |rt| rt.get(a) + 100);
        let d = rt.create_query(move |rt| rt.get(b) + rt.get(c));
        assert_eq!(rt.get(d), 112);
        rt.set(a, 2);
        assert_eq!(rt.get(d), 114);
    }

    #[test]
    fn local_only_affected_recompute() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_input(2_i64);

        let ca = Arc::new(AtomicU32::new(0));
        let cb = Arc::new(AtomicU32::new(0));

        let ca_clone = ca.clone();
        let derived_a = rt.create_query(move |rt| {
            ca_clone.fetch_add(1, Ordering::Relaxed);
            rt.get(a) * 10
        });
        let cb_clone = cb.clone();
        let derived_b = rt.create_query(move |rt| {
            cb_clone.fetch_add(1, Ordering::Relaxed);
            rt.get(b) * 10
        });

        assert_eq!(rt.get(derived_a), 10);
        assert_eq!(rt.get(derived_b), 20);
        assert_eq!(ca.load(Ordering::Relaxed), 1);
        assert_eq!(cb.load(Ordering::Relaxed), 1);

        rt.set(a, 5);
        assert_eq!(rt.get(derived_a), 50);
        assert_eq!(rt.get(derived_b), 20);
        assert_eq!(ca.load(Ordering::Relaxed), 2);
        assert_eq!(cb.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn shared_early_cutoff_stops_propagation() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let rt: Runtime<Shared> = Runtime::new();
        let a = rt.create_input(50_i64);

        let c_count = Arc::new(AtomicU32::new(0));
        let cc = c_count.clone();
        let b = rt.create_query(move |rt| rt.get(a).min(100));
        let c = rt.create_query(move |rt| {
            cc.fetch_add(1, Ordering::Relaxed);
            rt.get(b) + 1
        });

        assert_eq!(rt.get(c), 51);
        assert_eq!(c_count.load(Ordering::Relaxed), 1);

        rt.set(a, 200);
        assert_eq!(rt.get(c), 101);
        assert_eq!(c_count.load(Ordering::Relaxed), 2);

        // a=300 → b still clamps to 100 → c skipped via early cutoff.
        rt.set(a, 300);
        assert_eq!(rt.get(c), 101);
        assert_eq!(c_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn local_query_with_more_than_seven_deps() {
        // Exercises the overflow path on NodeData::install_deps and
        // for_each_dep.
        let rt: Runtime<Local> = Runtime::new();
        let inputs: Vec<_> = (0..12_i64).map(|v| rt.create_input(v)).collect();
        let captured = inputs.clone();
        let sum = rt.create_query(move |rt| {
            let mut total = 0_i64;
            for i in &captured {
                total += rt.get(*i);
            }
            total
        });
        // 0+1+2+...+11 = 66
        assert_eq!(rt.get(sum), 66);
        // Mutate one input and verify it propagates.
        rt.set(inputs[5], 100);
        // 0+1+2+3+4+100+6+7+8+9+10+11 = 161
        assert_eq!(rt.get(sum), 161);
    }

    #[test]
    fn shared_query_with_more_than_seven_deps() {
        let rt: Runtime<Shared> = Runtime::new();
        let inputs: Vec<_> = (0..15_i64).map(|v| rt.create_input(v)).collect();
        let captured = inputs.clone();
        let sum = rt.create_query(move |rt| {
            let mut total = 0_i64;
            for i in &captured {
                total += rt.get(*i);
            }
            total
        });
        // sum 0..15 = 105
        assert_eq!(rt.get(sum), 105);
        rt.set(inputs[10], 1000);
        // 0+1+...+9 + 1000 + 11+12+13+14 = 45 + 1000 + 50 = 1095
        assert_eq!(rt.get(sum), 1095);
    }

    #[test]
    fn local_get_traced_records_recompute_events() {
        use crate::trace::TraceAction;
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 10);
        let c = rt.create_query(move |rt| rt.get(b) * 2);
        let _ = rt.get(c);

        // Set then traced read: every dirty node should appear in the trace.
        rt.set(a, 5);
        let (value, trace) = rt.get_traced(c);
        assert_eq!(value, 30); // (5 + 10) * 2
        assert_eq!(trace.target, NodeId(c.slot()));
        assert_eq!(trace.nodes_recomputed, 2); // b and c both recomputed
        assert_eq!(trace.nodes_cutoff, 0);
        // Verify the trace has Recomputed events with value_changed=true
        let recomputed_count = trace
            .node_traces
            .iter()
            .filter(|t| {
                matches!(
                    t.action,
                    TraceAction::Recomputed {
                        value_changed: true
                    }
                )
            })
            .count();
        assert_eq!(recomputed_count, 2);
    }

    #[test]
    fn local_get_traced_records_early_cutoff() {
        use crate::trace::TraceAction;
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(200_i64);
        let clamped = rt.create_query(move |rt| rt.get(input).min(100));
        let downstream = rt.create_query(move |rt| rt.get(clamped) + 1);
        let _ = rt.get(downstream);

        // Set input > 100 again; clamped still produces 100, so downstream
        // gets early-cutoff (Recomputed with value_changed=false on clamped,
        // VerifiedClean on downstream because its dep didn't change_at).
        rt.set(input, 300);
        let (value, trace) = rt.get_traced(downstream);
        assert_eq!(value, 101);

        // clamped should have a Recomputed event with value_changed=false
        // (the cutoff).
        let cutoffs = trace
            .node_traces
            .iter()
            .filter(|t| {
                matches!(
                    t.action,
                    TraceAction::Recomputed {
                        value_changed: false
                    }
                )
            })
            .count();
        assert!(
            cutoffs >= 1,
            "expected at least one cutoff event, got trace {:?}",
            trace.node_traces
        );
        assert!(trace.nodes_cutoff >= 1);
    }

    #[test]
    #[should_panic(expected = "dependency cycle")]
    fn local_direct_cycle_panics() {
        use std::sync::OnceLock;
        let rt: Runtime<Local> = Runtime::new();
        let own: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let captured = own.clone();
        let q = rt.create_query(move |rt| rt.get(*captured.get().unwrap()) + 1);
        own.set(q).unwrap();
        let _ = rt.get(q);
    }

    #[test]
    #[should_panic(expected = "dependency cycle")]
    fn shared_direct_cycle_panics() {
        use std::sync::OnceLock;
        let rt: Runtime<Shared> = Runtime::new();
        let own: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let captured = own.clone();
        let q = rt.create_query(move |rt| rt.get(*captured.get().unwrap()) + 1);
        own.set(q).unwrap();
        let _ = rt.get(q);
    }

    #[test]
    #[should_panic(expected = "dependency cycle")]
    fn local_transitive_cycle_panics() {
        use std::sync::OnceLock;
        let rt: Runtime<Local> = Runtime::new();
        let q1_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let q1_for_q2 = q1_cell.clone();
        let q2 = rt.create_query(move |rt| rt.get(*q1_for_q2.get().unwrap()) * 2);
        let q1 = rt.create_query(move |rt| rt.get(q2) + 1);
        q1_cell.set(q1).unwrap();
        let _ = rt.get(q1);
    }

    /// A cycle that only forms after a set() makes a query start reading
    /// a node that transitively reads it back. The cycle is reached
    /// mid-compute, so the frame-stack check fires.
    #[test]
    #[should_panic(expected = "dependency cycle")]
    fn local_cycle_formed_by_dynamic_deps_panics() {
        use std::sync::OnceLock;
        let rt: Runtime<Local> = Runtime::new();
        let s = rt.create_input(0_i64);
        let q1_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let q1_for_q2 = q1_cell.clone();
        let q2 = rt.create_query(move |rt| {
            if rt.get(s) == 1 {
                rt.get(*q1_for_q2.get().unwrap())
            } else {
                0
            }
        });
        let q1 = rt.create_query(move |rt| if rt.get(s) == 1 { rt.get(q2) } else { 0 });
        q1_cell.set(q1).unwrap();
        assert_eq!(rt.get(q1), 0);
        rt.set(s, 1);
        let _ = rt.get(q1);
    }

    /// Many threads race the FIRST get of the same query chain. Before
    /// the wait-on-Computing protocol, claim losers fell through to the
    /// arena read and hit an uninitialized slot.
    #[test]
    fn shared_first_get_claim_race_returns_correct_value() {
        use std::sync::Barrier;
        for _ in 0..50 {
            let rt: Arc<Runtime<Shared>> = Arc::new(Runtime::new());
            let a = rt.create_input(3_i64);
            let b = rt.create_query(move |rt| rt.get(a) * 7);
            let c = rt.create_query(move |rt| rt.get(b) + 1);
            let barrier = Arc::new(Barrier::new(8));
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let rt = Arc::clone(&rt);
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        barrier.wait();
                        rt.get(c)
                    })
                })
                .collect();
            for h in handles {
                assert_eq!(h.join().unwrap(), 22);
            }
        }
    }

    /// Concurrent setters on distinct inputs must not lose revision
    /// increments or strand stale cutoffs: the sum must be exact after
    /// all writers join.
    #[test]
    fn shared_concurrent_setters_converge() {
        let rt: Arc<Runtime<Shared>> = Arc::new(Runtime::new());
        let inputs: Vec<_> = (0..8).map(|_| rt.create_input(0_i64)).collect();
        let captured = inputs.clone();
        let sum = rt.create_query(move |rt| captured.iter().map(|i| rt.get(*i)).sum::<i64>());
        assert_eq!(rt.get(sum), 0);

        let handles: Vec<_> = inputs
            .iter()
            .map(|&input| {
                let rt = Arc::clone(&rt);
                std::thread::spawn(move || {
                    for v in 1..=500_i64 {
                        rt.set(input, v);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(rt.get(sum), 500 * 8);
    }

    /// Readers hammer a query while a writer mutates its input. Every
    /// observed value must be a consistent function of SOME input the
    /// writer published, and reads must never go backwards once the
    /// writer is done.
    #[test]
    fn shared_reader_writer_hammer_observes_consistent_values() {
        let rt: Arc<Runtime<Shared>> = Arc::new(Runtime::new());
        let input = rt.create_input(0_i64);
        let doubled = rt.create_query(move |rt| rt.get(input) * 2);
        assert_eq!(rt.get(doubled), 0);

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let rt = Arc::clone(&rt);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        let v = rt.get(doubled);
                        assert!(v % 2 == 0 && (0..=2000).contains(&v), "torn read: {}", v);
                    }
                })
            })
            .collect();

        for v in 1..=1000_i64 {
            rt.set(input, v);
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
        assert_eq!(rt.get(doubled), 2000);
    }

    #[test]
    #[should_panic(expected = "foreign runtime")]
    fn get_with_foreign_handle_panics() {
        let rt1: Runtime<Local> = Runtime::new();
        let rt2: Runtime<Local> = Runtime::new();
        let a = rt1.create_input(1_i64);
        let _ = rt2.get(a);
    }

    #[test]
    #[should_panic(expected = "foreign runtime")]
    fn set_with_foreign_handle_panics() {
        let rt1: Runtime<Shared> = Runtime::new();
        let rt2: Runtime<Shared> = Runtime::new();
        let a = rt1.create_input(1_i64);
        rt2.set(a, 2);
    }

    /// A panicking compute closure must mark the node Failed, propagate
    /// the panic, keep the runtime usable, and recover once an upstream
    /// input changes.
    #[test]
    fn local_compute_panic_marks_failed_then_recovers() {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(0_i64);
        let q = rt.create_query(move |rt| {
            let v = rt.get(input);
            assert!(v != 0, "intentional test panic");
            v * 10
        });
        let other = rt.create_query(move |rt| rt.get(input) + 1);

        let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(q)));
        assert!(first.is_err());

        // Unrelated nodes still work; the dep stack was unwound cleanly.
        assert_eq!(rt.get(other), 1);

        // Reading the failed node again is a deliberate panic with a
        // recognizable message, not a deadlock.
        let again = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(q)));
        let msg = *again.unwrap_err().downcast::<String>().unwrap();
        assert!(msg.contains("last compute panicked"), "got: {}", msg);

        // A dependency change flips Failed -> Dirty and the recompute
        // succeeds.
        rt.set(input, 7);
        assert_eq!(rt.get(q), 70);
        assert_eq!(rt.get(other), 8);
    }

    #[test]
    fn shared_compute_panic_marks_failed_then_recovers() {
        let rt: Runtime<Shared> = Runtime::new();
        let input = rt.create_input(0_i64);
        let q = rt.create_query(move |rt| {
            let v = rt.get(input);
            assert!(v != 0, "intentional test panic");
            v * 10
        });

        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(q))).is_err());
        rt.set(input, 3);
        assert_eq!(rt.get(q), 30);
    }

    /// A panic mid-chain must fail both the panicking node and let its
    /// parent's frame unwind; recovery recomputes the whole chain.
    #[test]
    fn local_nested_compute_panic_unwinds_both_frames() {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(0_i64);
        let child = rt.create_query(move |rt| {
            let v = rt.get(input);
            assert!(v != 0, "intentional test panic");
            v + 1
        });
        let parent = rt.create_query(move |rt| rt.get(child) * 2);

        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(parent))).is_err());
        assert!(!rt.dep_stack.current_frame_active());

        rt.set(input, 5);
        assert_eq!(rt.get(parent), 12);
    }

    /// A query may read more than 255 inputs; the u8 dep-count cell uses
    /// an overflow marker and the DepList length is authoritative.
    #[test]
    fn local_query_with_300_deps_propagates() {
        let rt: Runtime<Local> = Runtime::new();
        let inputs: Vec<_> = (0..300_i64).map(|v| rt.create_input(v)).collect();
        let captured = inputs.clone();
        let sum = rt.create_query(move |rt| captured.iter().map(|i| rt.get(*i)).sum::<i64>());
        let expected: i64 = (0..300).sum();
        assert_eq!(rt.get(sum), expected);
        rt.set(inputs[123], 10_000);
        assert_eq!(rt.get(sum), expected - 123 + 10_000);
    }

    /// Stress test: many dynamic-dep transitions through the
    /// overflow path. Each iteration the dynamic query selects a
    /// different subset of inputs, forcing publish_deps to allocate
    /// a fresh overflow DepList and retire the old one through the
    /// haphazard global domain. Drop must complete cleanly with no
    /// UAF on the retired lists; miri / ASan would catch any leak.
    #[test]
    fn local_dynamic_overflow_deps_retirement() {
        use std::cell::Cell as StdCell;
        let rt: Runtime<Local> = Runtime::new();
        let switch = rt.create_input(0_u8);
        let inputs: Vec<_> = (0..16_i64).map(|v| rt.create_input(v)).collect();

        let captured = inputs.clone();
        let dynamic = rt.create_query(move |rt| -> i64 {
            let s = rt.get(switch) as usize;
            let start = s % 8;
            let mut total = 0;
            let extra = StdCell::new(s % 4);
            let end = (start + 8 + extra.get()).min(captured.len());
            for i in start..end {
                total += rt.get(captured[i]);
            }
            total
        });

        for s in 1..=50_u8 {
            rt.set(switch, s);
            let _ = rt.get(dynamic);
        }
        drop(rt);
    }
}
