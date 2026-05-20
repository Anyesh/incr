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
}

impl<C: Cells> Inner<C> {
    fn new() -> Self {
        Self {
            compute_fns: Vec::new(),
            dependents: Vec::new(),
            arenas: ArenaRegistry::new(),
            type_tags: HashMap::new(),
            next_type_tag: 0,
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
        }
    }

    #[inline]
    fn current_revision(&self) -> u64 {
        C::u64_load_acquire(&self.revision)
    }

    fn bump_revision(&self) -> u64 {
        let cur = C::u64_load_acquire(&self.revision);
        let next = cur
            .checked_add(1)
            .expect("incr-core: revision counter overflow");
        C::u64_store_release(&self.revision, next);
        next
    }

    /// Create an input node with an initial value.
    pub fn create_input<T: Value>(&self, value: T) -> Incr<T> {
        assert!(
            !self.dep_stack.current_frame_active(),
            "create_input called during compute; not permitted",
        );

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
                        if let Some(old) = arena_inner.try_read(node.arena_slot()) {
                            if old == new_value {
                                return false; // early cutoff
                            }
                        }
                    }
                    arena_inner.write(node.arena_slot(), new_value);
                    true
                },
            );

            inner.compute_fns.push(Some(compute));
            inner.dependents.push(Vec::new());
            let generation = self.nodes.get(slot).generation();
            (slot, generation, type_tag)
        };
        Incr::new(slot, generation, self.runtime_id)
    }

    /// Read the current value of a node. Triggers recomputation of the
    /// minimum necessary subgraph if anything is dirty.
    pub fn get<T: Value>(&self, handle: Incr<T>) -> T {
        debug_assert_eq!(
            handle.runtime_id(),
            self.runtime_id,
            "Incr<T> handle from a foreign runtime",
        );
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

    /// Set a new value on an input node. Bumps revision and marks all
    /// transitive dependents dirty.
    pub fn set<T: Value>(&self, handle: Incr<T>, value: T) {
        debug_assert_eq!(
            handle.runtime_id(),
            self.runtime_id,
            "Incr<T> handle from a foreign runtime",
        );
        let slot = handle.slot();

        let arena = {
            let inner = self.inner.read();
            inner
                .arenas
                .try_arena::<T>()
                .expect("incr-core: arena missing for input handle's type")
        };

        // No-op if the value is unchanged.
        let node = self.nodes.get(slot);
        if let Some(old) = arena.try_read(node.arena_slot()) {
            if old == value {
                return;
            }
        }

        let new_rev = self.bump_revision();
        arena.write(node.arena_slot(), value);
        node.set_changed_at(new_rev);
        node.set_verified_at(new_rev);

        self.mark_dependents_dirty(NodeId(slot));
    }

    /// BFS forward walk from `start`'s dependents, marking each Clean
    /// node as Dirty. Stops at already-Dirty/New nodes (they're already
    /// in the dirty set).
    fn mark_dependents_dirty(&self, start: NodeId) {
        let mut queue: Vec<NodeId> = {
            let inner = self.inner.read();
            inner.dependents[start.0 as usize].clone()
        };

        while let Some(id) = queue.pop() {
            let node = self.nodes.get(id.0);
            let cur = state::load::<C>(node.state_cell());
            match cur {
                NodeState::Clean | NodeState::Failed => {
                    // Transition to Dirty so the next reader recomputes.
                    state::store::<C>(node.state_cell(), NodeState::Dirty);
                    let inner = self.inner.read();
                    for &dep in &inner.dependents[id.0 as usize] {
                        queue.push(dep);
                    }
                }
                NodeState::New | NodeState::Dirty | NodeState::Computing => {
                    // Already dirty (or being computed); don't re-enqueue.
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

        while let Some((cur, visited)) = work.pop() {
            if visited {
                self.compute_one(cur);
                continue;
            }

            let node = self.nodes.get(cur.0);
            let cur_state = state::load::<C>(node.state_cell());
            if cur_state == NodeState::Clean {
                continue;
            }

            // First visit: push self (to process after deps) then push
            // any non-clean deps.
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

    /// Compute (or verify) a single node, assuming all its known deps
    /// are already clean. Handles state-machine transitions and red/green
    /// early cutoff.
    fn compute_one(&self, id: NodeId) {
        let node = self.nodes.get(id.0);

        // If something else cleaned us in the meantime, we're done.
        if state::load::<C>(node.state_cell()) == NodeState::Clean {
            return;
        }

        // Distinguish input vs query: inputs don't compute, they just
        // need their state stamped clean.
        let compute = {
            let inner = self.inner.read();
            inner.compute_fns.get(id.0 as usize).and_then(|f| f.clone())
        };
        let compute = match compute {
            Some(f) => f,
            None => {
                // Input node: state machine bookkeeping only.
                let rev = self.current_revision();
                node.set_verified_at(rev);
                state::store::<C>(node.state_cell(), NodeState::Clean);
                return;
            }
        };

        let is_recompute = !matches!(state::load::<C>(node.state_cell()), NodeState::New,);

        // Red/green check: if no dep's changed_at exceeds our verified_at,
        // we can skip the closure entirely.
        if is_recompute {
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
                // Verified clean: bump verified_at, leave changed_at alone
                // so downstream cutoffs also work.
                let rev = self.current_revision();
                node.set_verified_at(rev);
                state::store::<C>(node.state_cell(), NodeState::Clean);
                return;
            }
        }

        // Full compute path. Claim Computing first.
        if state::try_claim_compute::<C>(node.state_cell()).is_err() {
            // Lost the race (Shared) or already cleaned (Local). Re-check
            // and bail; the caller's ensure_clean loop will see Clean and
            // move on.
            return;
        }

        // Track deps via the strategy's dep stack.
        self.dep_stack.push_frame();
        let value_changed = (compute)(self, id.0, is_recompute);
        let recorded_deps = self.dep_stack.pop_frame();

        // Update dep edges. For the first compute (is_recompute=false)
        // there are no old deps; for recompute we diff against the old
        // set. NodeData stores deps via publish_initial_deps; for now
        // we always treat deps as initial (the leaky overflow-replace
        // path lands in the next commit alongside hazard-pointer
        // reclamation).
        self.publish_deps(id, &recorded_deps);

        // Update timestamps and transition to Clean.
        let rev = self.current_revision();
        if value_changed || !is_recompute {
            node.set_changed_at(rev);
        }
        node.set_verified_at(rev);
        state::store::<C>(node.state_cell(), NodeState::Clean);
    }

    /// Record dependencies on the node. First-compute path: install via
    /// inline-7 + overflow ptr; record reverse edges in the inner map.
    /// Recompute path with changed dep set: same shape for now (the
    /// hazard-pointer-reclaimed swap lands next).
    fn publish_deps(&self, id: NodeId, deps: &[NodeId]) {
        let node = self.nodes.get(id.0);
        // For the first cut we only support up-to-7 inline deps via
        // direct field stores. Overflow handling lands in the next commit.
        assert!(
            deps.len() <= 7,
            "incr-core: more than 7 deps not yet supported (lands with hazard-pointer overflow path)",
        );

        // Store inline.
        for (i, dep) in deps.iter().enumerate() {
            C::u32_store_relaxed(&node.inline_deps[i], dep.0);
        }
        C::u8_store_release(&node.dep_count, deps.len() as u8);

        // Add reverse edges: for each dep, append self to dep's dependents.
        if !deps.is_empty() {
            let mut inner = self.inner.write();
            for dep in deps {
                inner.dependents[dep.0 as usize].push(id);
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
}
