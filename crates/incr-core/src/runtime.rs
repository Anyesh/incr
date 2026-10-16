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

use std::collections::HashMap;
use std::sync::Arc;

use crate::arena_registry::ArenaRegistry;
use crate::cells::Cells;
use crate::dep_stack::DepStack;
use crate::generic_arena::GenericArena;
use crate::handle::{ErasedIncr, Incr, RuntimeId};
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
    pub(crate) labels: HashMap<u32, String>,
    pub(crate) trace_log: Vec<crate::trace::NodeTrace>,
    /// Node slots released by delete_node, awaiting recycling.
    pub(crate) free_nodes: Vec<u32>,
    pub(crate) observers: HashMap<u64, ObserverEntry<C>>,
    pub(crate) next_observer_id: u64,
    /// Nodes an operator created for its own output and adopted: it sets
    /// them but never reads them, so no reverse edge records them.
    /// Sparse, because only operators like `group_by` own nodes. Kept in
    /// step with `owner_of` and unhooked by the child's delete under the
    /// same write hold as its generation bump, so an entry always names
    /// the node that was adopted.
    pub(crate) owned: HashMap<u32, Vec<NodeId>>,
    pub(crate) owner_of: HashMap<u32, NodeId>,
}

/// Type-erased observer runner: takes the runtime and the observer's
/// last-seen stamp, returns the new stamp if the callback fired.
type ObserverRunFn<C> = Arc<dyn Fn(&Runtime<C>, u64) -> Option<u64> + Send + Sync>;

/// One registered observer: which slot it watches, the changed_at stamp
/// it last fired for, and a type-erased runner that ensures the node is
/// clean and invokes the user callback if the value changed.
pub(crate) struct ObserverEntry<C: Cells> {
    pub(crate) slot: u32,
    pub(crate) last_seen: u64,
    pub(crate) run: ObserverRunFn<C>,
}

impl<C: Cells> Inner<C> {
    fn new() -> Self {
        Self {
            compute_fns: Vec::new(),
            dependents: Vec::new(),
            arenas: ArenaRegistry::new(),
            labels: HashMap::new(),
            trace_log: Vec::new(),
            free_nodes: Vec::new(),
            observers: HashMap::new(),
            next_observer_id: 1,
            owned: HashMap::new(),
            owner_of: HashMap::new(),
        }
    }
}

/// Token returned by [`Runtime::observe`]; pass to [`Runtime::unobserve`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ObserverId(u64);

/// Named points in the protocol where a race test can park one thread
/// while another acts. Outside `cfg(test)` [`Runtime::park`] is an empty
/// inline function, so these compile away.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ParkPoint {
    /// In `ensure_clean`, just before `compute_one` on a visited node.
    BeforeComputeOne(NodeId),
    /// In `compute_one`, after the state load and before the claim CAS.
    BeforeClaim(NodeId),
    /// In red/green verification, after the dep scan found no change and
    /// before the landing CAS.
    VerifyAfterScan(NodeId),
    /// In `set`, after `bump_revision` and before the dirty walk.
    SetAfterBump,
    /// In `set`, after the dirty walk and before the revision is recorded
    /// as walked.
    SetAfterWalk,
    /// In `delete_erased`, after the dependents check and the `Dead` claim.
    DeleteAfterCheck(NodeId),
    /// In `delete_erased`, on finding the node claimed `Dead` by another
    /// delete that has not bumped its generation yet.
    DeleteWaitsOnClaim(NodeId),
}

/// What [`Runtime::delete_erased`] did with a node.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum DeleteOutcome {
    Deleted,
    /// The handle's node was already deleted, by this caller earlier or
    /// by a concurrent delete that has now finished.
    AlreadyGone,
    /// The node has a dependent. Nothing changed.
    HasDependents,
}

/// Records `rev` as walked when dropped. A drop guard rather than a call
/// at the end of `set`, because an unwind that skipped the completion
/// would stall the watermark, and with it every later `set`, for good.
struct WalkDone<'a, C: Cells> {
    rt: &'a Runtime<C>,
    rev: u64,
}

impl<C: Cells> Drop for WalkDone<'_, C> {
    fn drop(&mut self) {
        C::complete_walk(&self.rt.walk, self.rev);
    }
}

/// What `publish_deps` found when it validated a compute's recorded deps.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Publish {
    /// Edges are current; the compute may land `Clean`.
    Published,
    /// A newly added dep was not observed `Clean` at or below the
    /// compute's stamp revision, so a concurrent change may have missed
    /// this node, which had no reverse edge yet (B3). Edges are
    /// published; the compute must land `Dirty`.
    Superseded,
    /// A recorded dep was deleted, and possibly recycled, after the read
    /// (B2). Nothing was published.
    Stale,
}

#[cfg(test)]
pub(crate) type ParkHook = Box<dyn Fn(ParkPoint) + Send + Sync>;

/// The runtime.
pub struct Runtime<C: Cells> {
    pub(crate) nodes: SegmentedNodes<C>,
    pub(crate) inner: <C as Cells>::Lock<Inner<C>>,
    pub(crate) revision: <C as Cells>::U64,
    pub(crate) walk: <C as Cells>::Watermark,
    pub(crate) dep_stack: <C as Cells>::DepStack,
    pub(crate) runtime_id: RuntimeId,
    /// `1` when `get_traced` is actively recording. Checked on every
    /// `compute_one` via a Relaxed load (~1 ns when disarmed) so the
    /// non-tracing hot path pays no measurable cost.
    pub(crate) tracing_armed: <C as Cells>::U8,
    #[cfg(test)]
    pub(crate) park_hook: std::sync::OnceLock<ParkHook>,
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
            walk: C::new_watermark(1),
            dep_stack: <C::DepStack as DepStack>::new(),
            runtime_id: RuntimeId::allocate(),
            tracing_armed: C::new_u8(0),
            #[cfg(test)]
            park_hook: std::sync::OnceLock::new(),
        }
    }

    #[cfg(test)]
    fn park(&self, point: ParkPoint) {
        if let Some(hook) = self.park_hook.get() {
            hook(point);
        }
    }

    #[cfg(not(test))]
    #[inline(always)]
    fn park(&self, _point: ParkPoint) {}

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
        let (slot, generation) = {
            let mut inner = self.inner.write();
            let (type_tag, arena) = inner.arenas.ensure_tagged::<T>();
            let arena_slot = arena.reserve_with(value);
            let slot = if let Some(slot) = inner.free_nodes.pop() {
                self.nodes
                    .get(slot)
                    .reinit_recycled(type_tag, arena_slot, revision, true);
                slot
            } else {
                let node = NodeData::<C>::new_input(type_tag, arena_slot, revision);
                let slot = self.nodes.push(node);
                inner.compute_fns.push(None);
                inner.dependents.push(Vec::new());
                slot
            };
            (slot, self.nodes.get(slot).generation())
        };
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

        let (slot, generation) = {
            let mut inner = self.inner.write();
            let (type_tag, arena) = inner.arenas.ensure_tagged::<T>();
            let arena_slot = arena.reserve();
            let slot = if let Some(slot) = inner.free_nodes.pop() {
                self.nodes
                    .get(slot)
                    .reinit_recycled(type_tag, arena_slot, 0, false);
                slot
            } else {
                let node = NodeData::<C>::new_query(type_tag, arena_slot);
                let slot = self.nodes.push(node);
                inner.compute_fns.push(None);
                inner.dependents.push(Vec::new());
                slot
            };

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

            inner.compute_fns[slot as usize] = Some(compute);
            (slot, self.nodes.get(slot).generation())
        };
        Incr::new(slot, generation, self.runtime_id)
    }

    /// Reject handles minted by a different runtime and handles whose
    /// node was deleted (the slot's generation moved past the handle's).
    /// Real branches, not debug_asserts: in release a foreign or stale
    /// handle would silently read or write another node's slot in the
    /// same type arena, which is silent data corruption. Cost is two
    /// always-predicted compares on the hot path.
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
        let node = self.nodes.get(handle.slot());
        if node.generation() != handle.generation() {
            panic!(
                "incr-core: stale Incr<{}> handle: the node at slot {} was deleted (handle \
                 generation {}, slot generation {})",
                std::any::type_name::<T>(),
                handle.slot(),
                handle.generation(),
                node.generation(),
            );
        }
    }

    /// Register a callback fired by [`Self::stabilize`] whenever the
    /// node's value has changed since the last firing. Observation does
    /// not change the engine's pull model: nothing runs until someone
    /// calls `stabilize` (or `get`); the observer registry is a batch
    /// poll with last-seen bookkeeping, the same shape as Jane Street
    /// Incremental's observers.
    ///
    /// A node that has never computed fires on the first stabilize. A
    /// stabilize racing a concurrent `set` may fire twice for one
    /// change; it can never miss one.
    pub fn observe<T: Value, F>(&self, handle: Incr<T>, callback: F) -> ObserverId
    where
        F: Fn(&T) + Send + Sync + 'static,
    {
        self.check_handle(handle);
        let slot = handle.slot();
        let node = self.nodes.get(slot);
        // Fire-baseline: if the node already has a clean value, only
        // future changes fire; a never-computed node starts at 0 so the
        // first stabilize computes and fires.
        let last_seen = match state::load::<C>(node.state_cell()) {
            NodeState::New => 0,
            _ => node.changed_at().min(self.current_revision()),
        };

        let run = Arc::new(move |rt: &Runtime<C>, last_seen: u64| -> Option<u64> {
            rt.ensure_clean(NodeId(handle.slot()));
            let node = rt.nodes.get(handle.slot());
            // A pending set() marker reads as u64::MAX; clamp to the
            // current revision so last_seen stays passable and the
            // settled stamp can still exceed it (duplicate, not lost).
            let stamp = node.changed_at().min(rt.current_revision());
            if stamp > last_seen {
                let value = rt.get(handle);
                callback(&value);
                Some(stamp)
            } else {
                None
            }
        });

        let mut inner = self.inner.write();
        let id = inner.next_observer_id;
        inner.next_observer_id += 1;
        inner.observers.insert(
            id,
            ObserverEntry {
                slot,
                last_seen,
                run,
            },
        );
        ObserverId(id)
    }

    /// Remove an observer. Safe to call with an already-removed id.
    pub fn unobserve(&self, id: ObserverId) {
        // Bound to a local so the entry, whose callback is user code,
        // drops after the write guard does.
        let removed = self.inner.write().observers.remove(&id.0);
        drop(removed);
    }

    /// Bring every observed node up to date and fire callbacks for the
    /// ones whose values changed since they last fired. Callbacks run
    /// with no engine lock held; they may read from the runtime but must
    /// not call `stabilize` reentrantly from another thread if duplicate
    /// notifications matter.
    pub fn stabilize(&self) {
        let entries: Vec<(u64, u64, ObserverRunFn<C>)> = {
            let inner = self.inner.read();
            inner
                .observers
                .iter()
                .map(|(&id, e)| (id, e.last_seen, Arc::clone(&e.run)))
                .collect()
        };
        for (id, last_seen, run) in entries {
            if let Some(new_stamp) = run(self, last_seen) {
                let mut inner = self.inner.write();
                if let Some(e) = inner.observers.get_mut(&id) {
                    e.last_seen = e.last_seen.max(new_stamp);
                }
            }
        }
    }

    /// Delete a node and recycle its slot. The node must have no
    /// dependents (delete downstream nodes first) and must not be
    /// deleted from inside its own compute. Any in-flight compute on the
    /// node is waited out, exactly like a reader. After deletion every
    /// surviving copy of the handle panics with a stale-handle message
    /// on use; the slot and its arena storage are reused by future
    /// create calls under a bumped generation.
    pub fn delete_node<T: Value>(&self, handle: Incr<T>) {
        self.check_handle(handle);
        let root = handle.erase();
        self.assert_not_own_compute(NodeId(root.slot()));
        let owned = match self.owned_for_strict_delete(root) {
            Ok(owned) => owned,
            Err(slot) if slot == root.slot() => Self::panic_has_dependents(root.slot()),
            Err(child) => panic!(
                "incr-core: delete_node on node {} whose owned node {} still has \
                 dependents; delete downstream nodes first",
                root.slot(),
                child,
            ),
        };
        for child in owned {
            if self.delete_erased(child) == DeleteOutcome::HasDependents {
                Self::panic_has_dependents(child.slot());
            }
        }
        match self.delete_erased(root) {
            DeleteOutcome::Deleted => {}
            // check_handle passed, so a concurrent delete got there first.
            DeleteOutcome::AlreadyGone => {
                panic!("incr-core: node {} deleted twice concurrently", root.slot())
            }
            DeleteOutcome::HasDependents => Self::panic_has_dependents(root.slot()),
        }
    }

    fn panic_has_dependents(slot: u32) -> ! {
        panic!(
            "incr-core: delete_node on node {} which still has dependents; \
             delete downstream nodes first",
            slot,
        )
    }

    fn assert_not_own_compute(&self, id: NodeId) {
        assert!(
            !self.dep_stack.is_computing(id),
            "incr-core: delete_node called from inside node {}'s own compute",
            id.0,
        );
    }

    /// Everything `root` owns, transitively, ordered so each node comes
    /// before its owner. `Err(slot)` names the first node found, `root`
    /// included, that has a dependent. Checking everything up front
    /// means a strict delete that panics has released nothing, unless a
    /// dependent attaches concurrently after this check.
    fn owned_for_strict_delete(&self, root: ErasedIncr) -> Result<Vec<ErasedIncr>, u32> {
        let inner = self.inner.read();
        let mut out = Vec::new();
        let mut stack = vec![root.slot()];
        while let Some(slot) = stack.pop() {
            if !inner.dependents[slot as usize].is_empty() {
                return Err(slot);
            }
            if slot != root.slot() {
                out.push(self.erased_at(slot));
            }
            if let Some(children) = inner.owned.get(&slot) {
                stack.extend(children.iter().map(|c| c.0));
            }
        }
        out.reverse();
        Ok(out)
    }

    /// The handle of the node currently at `slot`. Only meaningful under
    /// an `inner` hold, where the dependents and owned invariants make
    /// the slot's generation the linked node's.
    fn erased_at(&self, slot: u32) -> ErasedIncr {
        ErasedIncr::new(slot, self.nodes.get(slot).generation(), self.runtime_id)
    }

    /// Record `child` as owned by `owner`: it is deleted with the owner,
    /// by strict delete and by the cascade. Both must be live.
    pub(crate) fn adopt(&self, owner: ErasedIncr, child: ErasedIncr) {
        let mut inner = self.inner.write();
        for h in [owner, child] {
            assert!(
                self.is_alive(h),
                "incr-core: adopt of deleted node {}",
                h.slot()
            );
        }
        let previous = inner.owner_of.insert(child.slot(), NodeId(owner.slot()));
        assert!(
            previous.is_none(),
            "incr-core: node {} adopted twice",
            child.slot()
        );
        inner
            .owned
            .entry(owner.slot())
            .or_default()
            .push(NodeId(child.slot()));
    }

    /// True iff `handle` names a node of this runtime that has not been
    /// deleted. A handle from another runtime is simply not alive here.
    /// The answer is a snapshot: a concurrent delete can end the node
    /// right after it returns.
    pub fn is_alive(&self, handle: impl Into<ErasedIncr>) -> bool {
        let handle = handle.into();
        if handle.runtime_id() != self.runtime_id {
            return false;
        }
        let node = self.nodes.get(handle.slot());
        // State before generation: a recycle bumps the generation before
        // it Release-stores the new occupant's state, so if the Acquire
        // load below sees that occupant, the generation load sees the bump.
        let state = node.state();
        state != NodeState::Dead && node.generation() == handle.generation()
    }

    /// The deletion path behind every public delete. Never panics on a
    /// dependent or on a node that is already gone; it reports them. An
    /// owned node counts as a dependent: callers delete owned nodes first.
    ///
    /// Invariant: a slot listed in any `dependents` or `owned` list
    /// carries the generation of the node linked there. That is why the
    /// generation is bumped under `inner.write()`, together with the
    /// unhooking, and not at the claim.
    pub(crate) fn delete_erased(&self, handle: ErasedIncr) -> DeleteOutcome {
        assert_eq!(
            handle.runtime_id(),
            self.runtime_id,
            "incr-core: delete of a handle from a foreign runtime",
        );
        let slot = handle.slot();
        let id = NodeId(slot);
        let node = self.nodes.get(slot);
        if node.generation() != handle.generation() {
            return DeleteOutcome::AlreadyGone;
        }
        self.assert_not_own_compute(id);

        let mut spins: u32 = 0;
        loop {
            let cur = state::load::<C>(node.state_cell());
            if node.generation() != handle.generation() {
                return DeleteOutcome::AlreadyGone;
            }
            match cur {
                NodeState::Computing | NodeState::ComputingDirty => {
                    spins += 1;
                    C::backoff(spins);
                }
                NodeState::Dead => {
                    // Another delete claimed the node and will bump the
                    // generation under its next write lock. Returning now
                    // would let a cascade re-find this slot in its
                    // parent's dependents and spin on it instead.
                    self.park(ParkPoint::DeleteWaitsOnClaim(id));
                    while node.generation() == handle.generation() {
                        spins += 1;
                        C::backoff(spins);
                    }
                    return DeleteOutcome::AlreadyGone;
                }
                _ => {
                    // The check and the claim share one read hold so no
                    // publish (which pushes reverse edges under the write
                    // lock) can land between them; clearing dependents
                    // below would otherwise drop its edge silently. The
                    // generation is rechecked under the hold because a
                    // delete and recycle between the load of `cur` and
                    // the CAS would otherwise let the CAS claim the
                    // slot's next occupant.
                    let claimed = {
                        let inner = self.inner.read();
                        if node.generation() != handle.generation() {
                            return DeleteOutcome::AlreadyGone;
                        }
                        if !inner.dependents[slot as usize].is_empty()
                            || inner.owned.contains_key(&slot)
                        {
                            return DeleteOutcome::HasDependents;
                        }
                        state::try_transition::<C>(node.state_cell(), cur, NodeState::Dead).is_ok()
                    };
                    if claimed {
                        self.park(ParkPoint::DeleteAfterCheck(id));
                        break;
                    }
                }
            }
        }

        // We own the Dead node now. Capture deps before clearing them,
        // then unhook reverse edges and recycle storage.
        let mut old_deps: Vec<NodeId> = Vec::new();
        node.for_each_dep(|d| old_deps.push(d));
        let arena_slot = node.arena_slot();
        let type_tag = node.type_tag();
        node.retire_for_delete();

        let mut inner = self.inner.write();
        for d in &old_deps {
            inner.dependents[d.0 as usize].retain(|&x| x != id);
        }
        inner.dependents[slot as usize].clear();
        if let Some(owner) = inner.owner_of.remove(&slot) {
            let siblings = inner
                .owned
                .get_mut(&owner.0)
                .expect("incr-core: owner_of names an owner with no owned list");
            siblings.retain(|&c| c != id);
            if siblings.is_empty() {
                inner.owned.remove(&owner.0);
            }
        }
        node.bump_generation();
        // The closures are dropped only after the lock is released,
        // because their captures' Drop impls are user code that may call
        // back into the runtime.
        let compute = inner.compute_fns[slot as usize].take();
        let observers: Vec<ObserverEntry<C>> = inner
            .observers
            .extract_if(|_, e| e.slot == slot)
            .map(|(_, e)| e)
            .collect();
        inner.labels.remove(&slot);
        let arena = Arc::clone(inner.arenas.erased(type_tag));
        inner.free_nodes.push(slot);
        drop(inner);
        drop(compute);
        drop(observers);
        arena.release(arena_slot);
        DeleteOutcome::Deleted
    }

    /// Read the current value of a node. Triggers recomputation of the
    /// minimum necessary subgraph if anything is dirty.
    pub fn get<T: Value>(&self, handle: Incr<T>) -> T {
        self.check_handle(handle);
        let slot = handle.slot();

        // Record dep if we're inside a compute closure.
        self.dep_stack.record_dep(NodeId(slot), handle.generation());

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

    /// Number of live nodes in the runtime (deleted-and-not-yet-recycled
    /// slots excluded).
    pub fn node_count(&self) -> usize {
        let free = self.inner.read().free_nodes.len();
        self.nodes.len() as usize - free
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
            if state::load::<C>(node.state_cell()) == NodeState::Dead {
                continue;
            }
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
        let walk_done = WalkDone {
            rt: self,
            rev: new_rev,
        };
        self.park(ParkPoint::SetAfterBump);
        node.settle_changed_at(new_rev);
        node.max_verified_at(new_rev);

        self.mark_dependents_dirty(NodeId(slot));
        self.park(ParkPoint::SetAfterWalk);
        drop(walk_done);
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
                    NodeState::New => {
                        // A node that recovered from a stale dep lands New
                        // but keeps its old reverse edges, so a Failed
                        // dependent can only be reached through it.
                        // Never-run queries have no dependents. The visit
                        // is an RMW for the reason given on the Dirty arm.
                        if state::try_transition::<C>(node.state_cell(), cur, cur).is_ok() {
                            let inner = self.inner.read();
                            queue.extend(inner.dependents[id.0 as usize].iter().copied());
                            break;
                        }
                    }
                    NodeState::Dirty | NodeState::ComputingDirty => {
                        // Already flagged; the walk that flagged it also
                        // covered its dependents. The visit must still be
                        // an RMW, not a plain load: this walk stores a
                        // dep's Dirty and then reaches the node, while a
                        // verifier claims the node and then loads its deps'
                        // states. With a plain load here that is the Dekker
                        // pattern, and both sides may read the old value.
                        // As an RMW the visit sits in the node's
                        // modification order, so the verifier's claim either
                        // reads from it (and sees the dep Dirty) or this CAS
                        // fails on the claim and the next pass flags it.
                        if state::try_transition::<C>(node.state_cell(), cur, cur).is_ok() {
                            break;
                        }
                    }
                    NodeState::Dead => {
                        // Raced a delete through a not-yet-unhooked
                        // edge; a deleted node has no dependents to
                        // propagate to.
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
                self.park(ParkPoint::BeforeComputeOne(cur));
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
                    C::backoff(spins);
                    continue;
                }
                NodeState::Failed => {
                    panic!(
                        "incr-core: node {} read but its last compute panicked; it stays \
                         Failed until a dependency changes (set an upstream input to retry)",
                        id.0,
                    );
                }
                NodeState::Dead => {
                    panic!(
                        "incr-core: node {} was deleted while a read was reaching it (a \
                         compute raced delete_node on one of its recorded dependencies)",
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

            self.park(ParkPoint::BeforeClaim(id));

            // Verification claims the node too, so a walk that arrives
            // meanwhile flags ComputingDirty and blocks the Clean landing.
            // The claim source tells us exactly whether this is a
            // recompute: the node may have gone Dirty since `observed`.
            let verifying = observed == NodeState::Dirty;
            let claimed_dirty = if verifying {
                if state::try_transition::<C>(
                    node.state_cell(),
                    NodeState::Dirty,
                    NodeState::Computing,
                )
                .is_err()
                {
                    continue;
                }
                true
            } else {
                match state::try_claim_compute::<C>(node.state_cell()) {
                    Ok(src) => src == NodeState::Dirty,
                    Err(_) => continue,
                }
            };

            // Capture the revisions after the claim and BEFORE verifying or
            // computing. If a set() bumps the revision mid-flight, our
            // stamps stay below its changed_at and the next reader
            // re-verifies; stamping the post-compute revision would mask
            // the change forever. Before the claim is too early: the claim
            // can succeed on a later Dirty than `observed` (another reader
            // cleans the node, a second set dirties it again), and a change
            // stamped below that set hides it from dependents verified in
            // between. The claim reads from the walk that made the node
            // Dirty, so the loads below see that walk's revision bump.
            //
            // changed_at uses the current revision, so a value changed by
            // an in-flight set is never under-stamped. verified_at uses the
            // walked watermark instead: a set bumps the revision before its
            // walk, and a verified_at naming a revision whose walk had not
            // dirtied our deps yet would let us verify Clean past it (B6).
            let stamp_rev = self.current_revision();
            let verify_rev = C::walked(&self.walk, &self.revision);

            // Red/green verification, only from Dirty: a New node has
            // nothing to verify against. It counts a dep that is not Clean
            // as changed, which catches a walk that passed this node before
            // the claim, and a dep that landed Dirty without a new stamp.
            if verifying {
                let my_verified = node.verified_at();
                let mut any_changed = false;
                node.for_each_dep(|dep| {
                    if !any_changed && !self.observed_clean(dep, my_verified) {
                        any_changed = true;
                    }
                });
                if !any_changed {
                    self.park(ParkPoint::VerifyAfterScan(id));
                    node.max_verified_at(verify_rev);
                    if self.land(node, NodeState::Clean, NodeState::Dirty) == NodeState::Clean {
                        self.record_trace(id, crate::trace::TraceAction::VerifiedClean);
                    } else {
                        self.dep_stack.poison_current_frame();
                    }
                    return;
                }
                // Keep the claim and recompute; the closure's nested reads
                // clean the changed deps.
            }
            let is_recompute = claimed_dirty;

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
            let frame = self.dep_stack.pop_frame();

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
                    let landing = match self.publish_deps(id, &frame.deps, true, stamp_rev) {
                        // A stale dep's slot may already belong to another
                        // node, so no union is published. The old deps keep
                        // their reverse edges.
                        Publish::Stale => NodeState::New,
                        // A new dep had a change in flight when we published,
                        // and that change's walk may have read the dep's
                        // dependents before our edge existed. Nothing would
                        // ever flip us out of Failed, so land Dirty and let
                        // the next reader retry.
                        Publish::Superseded => NodeState::Dirty,
                        // verified_at was not raised, so once a dependency
                        // changes, the dirty walk flips Failed -> Dirty and
                        // red/green sees the change and recomputes for real.
                        // If a set() already flagged us mid-compute, land
                        // Dirty now so the next reader retries immediately.
                        Publish::Published => NodeState::Failed,
                    };
                    let flagged = if landing == NodeState::New {
                        NodeState::New
                    } else {
                        NodeState::Dirty
                    };
                    self.land(node, landing, flagged);
                    // An enclosing closure that catches the unwind has
                    // still read a dep that did not land Clean.
                    self.dep_stack.poison_current_frame();
                    std::panic::resume_unwind(payload);
                }
            };

            let outcome = self.publish_deps(id, &frame.deps, false, stamp_rev);

            // The closure has already written its value into the slot, and
            // the next recompute's early cutoff compares against it, so a
            // changed value is stamped even when it is never returned.
            if value_changed || !is_recompute {
                node.max_changed_at(stamp_rev);
            }

            if outcome == Publish::Stale {
                // The value may have been read from a dep slot that was
                // released or recycled under the closure, so it is never
                // returned: land New and recompute. The retry either stops
                // reading the dep or panics on its stale handle.
                self.land(node, NodeState::New, NodeState::New);
                continue;
            }

            node.max_verified_at(verify_rev);

            // Finish. Land Dirty instead of Clean if a set() flagged us
            // ComputingDirty mid-compute, if a new dep changed after we
            // read it, or if a dep we read did not itself land Clean, so
            // the next reader recomputes against the newer input.
            let target = if outcome == Publish::Superseded || frame.poisoned {
                NodeState::Dirty
            } else {
                NodeState::Clean
            };
            if self.land(node, target, NodeState::Dirty) != NodeState::Clean {
                // The enclosing compute read this node through a non-Clean
                // landing. Its own publish may take the fast path, where
                // nothing else would catch that.
                self.dep_stack.poison_current_frame();
            }
            self.record_trace(id, crate::trace::TraceAction::Recomputed { value_changed });
            return;
        }
    }

    /// Leave this thread's `Computing` claim for `target`, or for `flagged`
    /// if a walk raised `ComputingDirty` meanwhile, and return the state
    /// landed. Both moves are CASes, never a plain store: a failed CAS is
    /// only a load and need not read the newest walker's flag, and a
    /// plain store would end the walkers' release sequence. The second
    /// CAS, as an RMW, reads the newest value, so every walker's earlier
    /// writes happen-before the next claim. Loom and Miri both treat a
    /// failed CAS as reading the newest value, so neither would catch a
    /// regression here.
    fn land(&self, node: &NodeData<C>, target: NodeState, flagged: NodeState) -> NodeState {
        if state::try_transition::<C>(node.state_cell(), NodeState::Computing, target).is_ok() {
            return target;
        }
        let owned =
            state::try_transition::<C>(node.state_cell(), NodeState::ComputingDirty, flagged);
        debug_assert!(
            owned.is_ok(),
            "only the claim's owner leaves ComputingDirty"
        );
        flagged
    }

    /// Validate a compute's recorded deps, then install them and update
    /// reverse edges. With `panicked` the installed list is the union of
    /// the old deps and the recorded ones, and a new dep that is `Failed`
    /// does not supersede: the compute that read it fails with it, and the
    /// dep's own recovery walk reaches this node through the new edge. Diffs old vs
    /// new deps so static-dep queries (the common case) skip the
    /// inner.write() acquire and the dependents vector edits on
    /// recompute.
    ///
    /// The fast path needs no validation: its deps already have reverse
    /// edges, so a strict delete refuses them and a dirty walk reaches
    /// this node. The slow path validates, installs and diffs inside one
    /// `inner.write()` hold. That ordering is part of the contract: a
    /// delete claims `Dead` under `inner.read()`, and a set's walk flips a
    /// dep before reading its dependents under `inner.read()`, so each
    /// either finishes before validation sees it or sees this node's new
    /// edge.
    fn publish_deps(
        &self,
        id: NodeId,
        recorded: &[(NodeId, u32)],
        panicked: bool,
        stamp_rev: u64,
    ) -> Publish {
        let node = self.nodes.get(id.0);

        let mut old_deps: Vec<NodeId> = Vec::with_capacity(8);
        node.for_each_dep(|d| old_deps.push(d));
        let new_deps: Vec<NodeId> = if panicked {
            let mut union = old_deps.clone();
            for &(d, _) in recorded {
                if !union.contains(&d) {
                    union.push(d);
                }
            }
            union
        } else {
            recorded.iter().map(|&(d, _)| d).collect()
        };

        if old_deps == new_deps {
            return Publish::Published;
        }

        let mut inner = self.inner.write();
        let stale = recorded.iter().any(|&(d, generation)| {
            let dep = self.nodes.get(d.0);
            dep.generation() != generation || dep.state() == NodeState::Dead
        });
        if stale {
            return Publish::Stale;
        }
        let settled = |d: &NodeId| {
            old_deps.contains(d)
                || self.observed_clean(*d, stamp_rev)
                || (panicked && self.nodes.get(d.0).state() == NodeState::Failed)
        };
        let superseded = !new_deps.iter().all(settled);

        node.install_deps(&new_deps);

        // Linear scans for small dep sets are faster than HashSet
        // construction below ~16 items.
        for old_dep in &old_deps {
            if !new_deps.contains(old_dep) {
                inner.dependents[old_dep.0 as usize].retain(|&d| d != id);
            }
        }
        for new_dep in &new_deps {
            if !old_deps.contains(new_dep) {
                inner.dependents[new_dep.0 as usize].push(id);
            }
        }
        if superseded {
            Publish::Superseded
        } else {
            Publish::Published
        }
    }

    /// True iff `dep` is `Clean` with `changed_at <= rev`. The state is
    /// loaded first because the Release that publishes a recompute's
    /// `changed_at` stamp is its finishing state CAS: reading the stamp
    /// first could pair an old stamp with a newer `Clean`.
    fn observed_clean(&self, dep: NodeId, rev: u64) -> bool {
        let node = self.nodes.get(dep.0);
        node.state() == NodeState::Clean && node.changed_at() <= rev
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
impl<C: Cells> Runtime<C> {
    /// Invariant (I) of the shared-protocol plan, checked at quiescence:
    /// every installed dep S of a `Clean` node X is alive, has X in
    /// `dependents[S]`, and is itself `Clean`. Also checks that no freed
    /// slot still carries reverse edges.
    pub(crate) fn assert_invariant_i(&self) {
        let inner = self.inner.read();
        for x in 0..self.nodes.len() {
            let node = self.nodes.get(x);
            if node.state() != NodeState::Clean {
                continue;
            }
            node.for_each_dep(|s| {
                let dep = self.nodes.get(s.0);
                assert!(
                    dep.state() != NodeState::Dead && !inner.free_nodes.contains(&s.0),
                    "(I) violated: Clean node {x} has dead dep {}",
                    s.0
                );
                assert!(
                    inner.dependents[s.0 as usize].contains(&NodeId(x)),
                    "(I) violated: Clean node {x} is missing from dependents[{}]",
                    s.0
                );
                assert_eq!(
                    dep.state(),
                    NodeState::Clean,
                    "(I) violated: Clean node {x} has non-Clean dep {}",
                    s.0
                );
            });
        }
        for &slot in &inner.free_nodes {
            assert!(
                inner.dependents[slot as usize].is_empty(),
                "freed slot {slot} still has reverse edges {:?}",
                inner.dependents[slot as usize]
            );
        }
    }

    pub(crate) fn has_reverse_edge<T, U>(&self, dep: Incr<T>, dependent: Incr<U>) -> bool {
        self.inner.read().dependents[dep.slot() as usize].contains(&NodeId(dependent.slot()))
    }

    pub(crate) fn state_of<T>(&self, handle: Incr<T>) -> NodeState {
        self.nodes.get(handle.slot()).state()
    }

    /// Every installed dep of a node that is not `Dead` is alive and has
    /// the node in its dependents. Covers `Dirty`, `Failed` and recovered
    /// `New` nodes, where a missing edge would otherwise surface only
    /// after the node's next fast-path publish.
    pub(crate) fn assert_edges_complete(&self) {
        let inner = self.inner.read();
        for x in 0..self.nodes.len() {
            let node = self.nodes.get(x);
            if node.state() == NodeState::Dead {
                continue;
            }
            node.for_each_dep(|s| {
                assert!(
                    self.nodes.get(s.0).state() != NodeState::Dead
                        && !inner.free_nodes.contains(&s.0),
                    "node {x} ({:?}) has dead dep {}",
                    node.state(),
                    s.0
                );
                assert!(
                    inner.dependents[s.0 as usize].contains(&NodeId(x)),
                    "node {x} ({:?}) is missing from dependents[{}]",
                    node.state(),
                    s.0
                );
            });
        }
    }

    pub(crate) fn assert_walks_complete(&self) {
        assert_eq!(
            C::walked(&self.walk, &self.revision),
            self.current_revision(),
            "the walked watermark lags the revision at quiescence"
        );
    }
}

#[cfg(all(test, not(loom)))]
#[path = "runtime_race_tests.rs"]
mod race_tests;

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
    fn observe_and_stabilize_fire_on_change_only() {
        use std::sync::Mutex;
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(1_i64);
        let doubled = rt.create_query(move |rt| rt.get(input) * 2);

        let seen: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_cb = Arc::clone(&seen);
        let id = rt.observe(doubled, move |v| seen_cb.lock().unwrap().push(*v));

        // Never-computed node fires on first stabilize.
        rt.stabilize();
        assert_eq!(*seen.lock().unwrap(), vec![2]);

        // No change, no fire.
        rt.stabilize();
        assert_eq!(*seen.lock().unwrap(), vec![2]);

        // Change fires once per stabilize, with the new value.
        rt.set(input, 5);
        rt.stabilize();
        rt.stabilize();
        assert_eq!(*seen.lock().unwrap(), vec![2, 10]);

        // Unobserve stops notifications.
        rt.unobserve(id);
        rt.set(input, 9);
        rt.stabilize();
        assert_eq!(*seen.lock().unwrap(), vec![2, 10]);
    }

    #[test]
    fn observe_respects_early_cutoff() {
        use std::sync::Mutex;
        let rt: Runtime<Shared> = Runtime::new();
        let input = rt.create_input(200_i64);
        let clamped = rt.create_query(move |rt| rt.get(input).min(100));
        let _ = rt.get(clamped);

        let seen: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_cb = Arc::clone(&seen);
        rt.observe(clamped, move |v| seen_cb.lock().unwrap().push(*v));

        // Input changes but the clamped value does not: cutoff, no fire.
        rt.set(input, 300);
        rt.stabilize();
        assert_eq!(seen.lock().unwrap().len(), 0);

        rt.set(input, 50);
        rt.stabilize();
        assert_eq!(*seen.lock().unwrap(), vec![50]);
    }

    #[test]
    fn delete_node_drops_its_observers() {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(1_i64);
        let fired = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let fired_cb = Arc::clone(&fired);
        rt.observe(input, move |_| {
            fired_cb.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        rt.delete_node(input);
        // The observer is gone; stabilize must not touch the dead slot.
        rt.stabilize();
        assert_eq!(fired.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn delete_node_recycles_slot_and_rejects_stale_handles() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_input(2_i64);
        assert_eq!(rt.node_count(), 2);

        rt.delete_node(a);
        assert_eq!(rt.node_count(), 1);

        // The freed slot is recycled, under a new generation.
        let c = rt.create_input(30_i64);
        assert_eq!(c.slot(), a.slot());
        assert_ne!(c, a);
        assert_eq!(rt.node_count(), 2);
        assert_eq!(rt.get(c), 30);
        assert_eq!(rt.get(b), 2);

        let stale = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(a)));
        let msg = *stale.unwrap_err().downcast::<String>().unwrap();
        assert!(msg.contains("stale"), "got: {}", msg);
    }

    #[test]
    fn delete_query_then_input_unhooks_edges() {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(5_i64);
        let q = rt.create_query(move |rt| rt.get(input) * 2);
        assert_eq!(rt.get(q), 10);

        rt.delete_node(q);
        // The input no longer has dependents, so it can go too.
        rt.delete_node(input);
        assert_eq!(rt.node_count(), 0);

        // set on the deleted input must fail, not corrupt a recycled slot.
        let fresh = rt.create_input("hello".to_string());
        assert_eq!(rt.get(fresh), "hello");
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.set(input, 9))).is_err()
        );
    }

    #[test]
    #[should_panic(expected = "still has dependents")]
    fn delete_node_with_dependents_panics() {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(5_i64);
        let _q = rt.create_query(move |rt| rt.get(input) * 2);
        let _ = rt.get(_q);
        rt.delete_node(input);
    }

    struct RtPtr<C: Cells>(*const Runtime<C>);
    // SAFETY: test-only. Every runtime outlives the closures holding its
    // pointer, and a Local runtime's pointer never leaves its thread.
    unsafe impl<C: Cells> Send for RtPtr<C> {}
    unsafe impl<C: Cells> Sync for RtPtr<C> {}

    /// Runs runtime calls from its Drop, as a user capture might.
    struct TouchOnDrop<C: Cells> {
        rt: RtPtr<C>,
        probe: Incr<i64>,
        victim: Incr<i64>,
        dropped: Arc<std::sync::atomic::AtomicU32>,
    }

    impl<C: Cells> Drop for TouchOnDrop<C> {
        fn drop(&mut self) {
            // SAFETY: see RtPtr.
            let rt = unsafe { &*self.rt.0 };
            assert!(rt.is_alive(self.probe));
            rt.delete_node(self.victim);
            self.dropped
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn closures_drop_outside_the_lock<C: Cells>(rt: &Runtime<C>) {
        use std::sync::atomic::{AtomicU32, Ordering};
        let probe = rt.create_input(1_i64);
        let victims = [2, 3, 4].map(|v| rt.create_input(v as i64));
        let dropped = Arc::new(AtomicU32::new(0));
        let guard = |victim| TouchOnDrop {
            rt: RtPtr(rt as *const Runtime<C>),
            probe,
            victim,
            dropped: Arc::clone(&dropped),
        };

        let g = guard(victims[0]);
        let q = rt.create_query(move |rt| {
            let _keep = &g;
            rt.get(probe)
        });
        assert_eq!(rt.get(q), 1);
        let g = guard(victims[1]);
        rt.observe(q, move |_| {
            let _keep = &g;
        });
        let g = guard(victims[2]);
        let watcher = rt.observe(probe, move |_| {
            let _keep = &g;
        });

        rt.unobserve(watcher);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        rt.delete_node(q);
        assert_eq!(dropped.load(Ordering::SeqCst), 3);
        assert!(victims.iter().all(|&v| !rt.is_alive(v)));
        assert!(rt.is_alive(probe));
    }

    #[test]
    fn local_closures_drop_outside_the_lock() {
        closures_drop_outside_the_lock(&Runtime::<Local>::new());
    }

    #[test]
    fn shared_closures_drop_outside_the_lock() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            closures_drop_outside_the_lock(&Runtime::<Shared>::new());
            tx.send(()).unwrap();
        });
        rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("a closure drop deadlocked or panicked under the runtime lock");
    }

    #[test]
    fn is_alive_tracks_delete_and_recycle() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(1_i64);
        assert!(rt.is_alive(a));
        assert!(rt.is_alive(a.erase()));

        rt.delete_node(a);
        assert!(!rt.is_alive(a));

        let b = rt.create_input("recycled".to_string());
        assert_eq!(b.slot(), a.slot());
        assert!(!rt.is_alive(a), "a stale handle matches a recycled slot");
        assert!(rt.is_alive(b));
    }

    #[test]
    fn is_alive_is_false_for_a_foreign_handle() {
        let rt1: Runtime<Shared> = Runtime::new();
        let rt2: Runtime<Shared> = Runtime::new();
        let a = rt1.create_input(1_i64);
        let _same_slot = rt2.create_input(2_i64);
        assert!(!rt2.is_alive(a));
    }

    #[test]
    #[should_panic(expected = "stale")]
    fn delete_node_twice_panics_stale() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(1_i64);
        rt.delete_node(a);
        rt.delete_node(a);
    }

    #[test]
    #[should_panic(expected = "foreign runtime")]
    fn delete_node_with_foreign_handle_panics() {
        let rt1: Runtime<Local> = Runtime::new();
        let rt2: Runtime<Local> = Runtime::new();
        let a = rt1.create_input(1_i64);
        let _ = rt2.create_input(2_i64);
        rt2.delete_node(a);
    }

    #[test]
    #[should_panic(expected = "own compute")]
    fn delete_node_from_its_own_compute_panics() {
        let rt: Runtime<Local> = Runtime::new();
        let me: Arc<std::sync::OnceLock<Incr<i64>>> = Arc::new(std::sync::OnceLock::new());
        let me_in = Arc::clone(&me);
        let q = rt.create_query(move |rt| {
            rt.delete_node(*me_in.get().unwrap());
            0_i64
        });
        me.set(q).unwrap();
        let _ = rt.get(q);
    }

    #[test]
    fn delete_erased_reports_dependents_without_panicking() {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(5_i64);
        let q = rt.create_query(move |rt| rt.get(input) * 2);
        assert_eq!(rt.get(q), 10);

        assert_eq!(
            rt.delete_erased(input.erase()),
            DeleteOutcome::HasDependents
        );
        assert!(rt.is_alive(input));
        assert_eq!(rt.get(q), 10);

        assert_eq!(rt.delete_erased(q.erase()), DeleteOutcome::Deleted);
        assert_eq!(rt.delete_erased(q.erase()), DeleteOutcome::AlreadyGone);
        assert_eq!(rt.delete_erased(input.erase()), DeleteOutcome::Deleted);
        assert_eq!(rt.node_count(), 0);
    }

    #[test]
    fn deleted_slot_recycles_across_value_types() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(7_i64);
        rt.delete_node(a);
        let s = rt.create_input("typed".to_string());
        assert_eq!(s.slot(), a.slot());
        assert_eq!(rt.get(s), "typed");
        rt.set(s, "retyped".to_string());
        assert_eq!(rt.get(s), "retyped");
    }

    #[test]
    fn shared_delete_and_recycle_under_reads() {
        let rt: Arc<Runtime<Shared>> = Arc::new(Runtime::new());
        let stable = rt.create_input(100_i64);
        let q = rt.create_query(move |rt| rt.get(stable) + 1);
        assert_eq!(rt.get(q), 101);

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..2)
            .map(|_| {
                let rt = Arc::clone(&rt);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        assert_eq!(rt.get(q), 101);
                    }
                })
            })
            .collect();

        // Churn unrelated nodes through delete/recreate while readers run.
        for i in 0..500_i64 {
            let tmp = rt.create_input(i);
            assert_eq!(rt.get(tmp), i);
            rt.delete_node(tmp);
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
        assert_eq!(rt.node_count(), 2);
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
