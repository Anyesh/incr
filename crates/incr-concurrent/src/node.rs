//! Node data layout.
//!
//! Per section 5.1 of the concurrent core rewrite spec, a node's read-hot
//! fields live in a single 64-byte cache-line-aligned `NodeData` struct.
//! Write-hot fields (dependents, per-node labels) live in parallel vectors
//! on the `Runtime` so that reader traversal touches exactly one cache line
//! per visited node.
//!
//! ## Layout
//!
//! Fields are ordered in alignment-descending order so that the struct is
//! exactly 64 bytes with no internal padding:
//!
//! ```text
//! offset  size   field
//! ------  ----   -----
//!    0     8     verified_at   AtomicU64
//!    8     8     changed_at    AtomicU64
//!   16     8     overflow_deps AtomicPtr<DepList>
//!   24    28     inline_deps   [AtomicU32; 7]
//!   52     4     arena_slot    u32           (write-once)
//!   56     2     type_tag      u16           (write-once)
//!   58     1     state         AtomicNodeState
//!   59     1     dep_count     AtomicU8
//!   60     4     generation    AtomicU32     (bumped on slot recycle)
//! ```
//!
//! `#[repr(C, align(64))]` forces both the layout (C-style, no field
//! reordering) and the 64-byte alignment. The `const _: () = assert!(...)`
//! at the bottom of this module is load-bearing and will trip the build if
//! a future edit perturbs the size or alignment.
//!
//! ## Resolved spec ambiguity
//!
//! Section 5.1 first sketches a `NodeData` with `dependents` fields
//! inline in the struct, then three paragraphs later says "the final
//! layout separates read-hot fields from write-hot fields ... this is
//! better and is what the implementation will do." We implement the
//! parallel-vector version: this `NodeData` carries dependencies (the
//! reader traversal path) but not dependents (the writer's dirty walk
//! path). The `Runtime` holds a `Vec<DependentsSlice>` indexed by node id.
//! The label, similarly, lives on the `Runtime` side rather than inline,
//! since the spec's sketch did not account for its 8 bytes in the 64-byte
//! budget.
//!
//! ## Dependency list
//!
//! Dependencies use an inline-7 + overflow-pointer layout. Up to seven
//! deps live directly in `inline_deps`; beyond that, `overflow_deps`
//! points at a heap-allocated `DepList` containing *all* the deps.
//! The `inline_deps` array is ignored when `dep_count > 7`.
//!
//! Inline-7 is chosen because most function queries have 1-3 deps and
//! almost all have under 8. The occasional wide fan-in node pays one
//! pointer dereference via the overflow path, which is acceptable.
//!
//! ## Dependency mutation (deferred)
//!
//! This commit establishes the NodeData struct, construction, and
//! read-only dep access. Mutation of deps (during a recompute that
//! discovers a different set of dependencies than the previous run)
//! requires coordination with the state machine and an epoch-based
//! reclamation story for the old overflow list. Both land in commit F
//! alongside the Runtime's compute path. In the meantime, a node's
//! deps are write-once at construction.
//!
//! ## Memory ordering in this module
//!
//! Constructors use `Relaxed` stores for every field. Visibility to
//! other threads is established later by the caller (the Runtime)
//! when it Release-stores the final state on the node, or when it
//! publishes the node's segment pointer to readers. This module does
//! not attempt to be self-synchronizing; it only provides the right
//! atomic primitives and correct load orderings on the read side.

use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicU8, Ordering};

use super::handle::{HandleError, Incr, RuntimeId};
use super::state::{AtomicNodeState, NodeState};

/// Stable identifier for a node within a `Runtime`. The `u32` is an index
/// into the runtime's segmented nodes store.
///
/// `NodeId` is a newtype rather than a bare `u32` so that mixing up node
/// ids with arena slot indices (also `u32`) produces a type error.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct NodeId(pub u32);

impl NodeId {
    /// Sentinel value used for uninitialized inline dep slots. Real nodes
    /// start at index 0; using `u32::MAX` as the sentinel keeps the common
    /// case (small index) fitting in a smaller integer for debug output.
    pub(crate) const SENTINEL: NodeId = NodeId(u32::MAX);
}

/// Heap-allocated overflow dependency list. Used when a node has more than
/// seven dependencies. Immutable once published.
///
/// The `Box<[NodeId]>` layout gives us a length prefix for free (fat
/// pointer) without a separate length field. Readers iterate via
/// `.deps.iter()`.
pub(crate) struct DepList {
    pub(crate) deps: Box<[NodeId]>,
}

/// Read-hot per-node state. Exactly one 64-byte cache line.
///
/// See the module docs for the field layout rationale and the
/// coordination contract with the state machine and the runtime.
#[repr(C, align(64))]
pub(crate) struct NodeData {
    /// Revision at which this node was last verified against its
    /// dependencies. In the red-green algorithm, a node is known clean
    /// at `verified_at` even if `changed_at < verified_at`.
    verified_at: AtomicU64,

    /// Revision at which this node's value last changed. Set once per
    /// successful compute to the current runtime revision.
    changed_at: AtomicU64,

    /// Pointer to overflow dep list, null when `dep_count <= 7`.
    /// Allocated via `Box::into_raw` in the constructor; reclaimed by
    /// this struct's `Drop`. Mutation (via compute) is out of scope for
    /// commit D; the pointer is effectively write-once here.
    overflow_deps: AtomicPtr<DepList>,

    /// Inline dependency storage, valid for slots `0..dep_count.min(7)`.
    /// Slots at or beyond `dep_count` are stale; treat them as
    /// uninitialized padding. When `dep_count > 7`, the entire inline
    /// array is ignored and `overflow_deps` holds the authoritative list.
    inline_deps: [AtomicU32; 7],

    /// Index into the typed arena for this node's value. Immutable after
    /// construction. Not atomic; synchronized with readers via the state
    /// machine's initial Release store.
    arena_slot: u32,

    /// Tag identifying which typed arena holds this node's value. The
    /// runtime maps `type_tag` to a concrete `AtomicPrimitiveArena<T>` or
    /// `GenericArena<T>` via the arena registry. Immutable after
    /// construction.
    type_tag: u16,

    /// The node's lifecycle state. See `v2::state` for the transition
    /// table and memory ordering contract.
    state: AtomicNodeState,

    /// Current number of dependencies. Used to decide whether to read
    /// from `inline_deps` or `overflow_deps`.
    dep_count: AtomicU8,

    /// Generation counter for detecting use-after-recycle of this slot.
    /// Matched against `Incr<T>::generation` on every handle access.
    /// Bumped by `Runtime::delete_node` (a future capability; commit E
    /// reserves the field but does not yet recycle slots). Lives in the
    /// four bytes that would otherwise be trailing padding, so adding
    /// this field does not change the struct size. The const assertion
    /// at the bottom of this file enforces size == 64.
    generation: AtomicU32,
}

// These assertions are load-bearing. A mismatch means a later edit
// perturbed the layout in a way that breaks the one-cache-line-per-node
// invariant. If an edit intentionally grows the struct, the fix is not
// to relax the assertion; it is to revisit the spec's 64-byte budget.
const _: () = assert!(
    std::mem::size_of::<NodeData>() == 64,
    "NodeData must be exactly one 64-byte cache line"
);
const _: () = assert!(
    std::mem::align_of::<NodeData>() == 64,
    "NodeData must be 64-byte aligned"
);

impl NodeData {
    /// Construct a new input node. Input nodes start in `Clean` state
    /// because their value is provided directly at creation time by the
    /// runtime's `create_input`. They have no dependencies.
    ///
    /// `revision` is the runtime's current revision at the time of input
    /// creation, used for both `verified_at` and `changed_at`.
    pub(crate) fn new_input(type_tag: u16, arena_slot: u32, revision: u64) -> Self {
        Self {
            verified_at: AtomicU64::new(revision),
            changed_at: AtomicU64::new(revision),
            overflow_deps: AtomicPtr::new(std::ptr::null_mut()),
            inline_deps: Self::empty_inline_deps(),
            arena_slot,
            type_tag,
            state: AtomicNodeState::new(NodeState::Clean),
            dep_count: AtomicU8::new(0),
            generation: AtomicU32::new(0),
        }
    }

    /// Construct a new query node. Query nodes start in `New` state
    /// because their value has not yet been computed; the first reader
    /// will CAS to `Computing` and run the compute closure.
    ///
    /// Query nodes have no known dependencies at construction time:
    /// dependencies are discovered during compute by the dep tracker.
    pub(crate) fn new_query(type_tag: u16, arena_slot: u32) -> Self {
        Self {
            verified_at: AtomicU64::new(0),
            changed_at: AtomicU64::new(0),
            overflow_deps: AtomicPtr::new(std::ptr::null_mut()),
            inline_deps: Self::empty_inline_deps(),
            arena_slot,
            type_tag,
            state: AtomicNodeState::new(NodeState::New),
            dep_count: AtomicU8::new(0),
            generation: AtomicU32::new(0),
        }
    }

    /// Internal: construct an inline-deps array with every slot set to
    /// the sentinel (u32::MAX). Used by the constructors; real deps
    /// overwrite these sentinels.
    fn empty_inline_deps() -> [AtomicU32; 7] {
        [
            AtomicU32::new(NodeId::SENTINEL.0),
            AtomicU32::new(NodeId::SENTINEL.0),
            AtomicU32::new(NodeId::SENTINEL.0),
            AtomicU32::new(NodeId::SENTINEL.0),
            AtomicU32::new(NodeId::SENTINEL.0),
            AtomicU32::new(NodeId::SENTINEL.0),
            AtomicU32::new(NodeId::SENTINEL.0),
        ]
    }

    /// Replace this node's dependency list with a new set.
    ///
    /// Called by the runtime on recompute when the compute closure
    /// recorded a different set of dependencies than the previous run
    /// (dynamic dependencies). Handles all four inline/overflow
    /// transitions:
    ///
    /// - inline → inline: overwrite the inline slots in place, clear
    ///   the overflow pointer if it was somehow non-null (shouldn't
    ///   be, but harmless).
    /// - inline → overflow: allocate a new DepList, install it in the
    ///   overflow pointer.
    /// - overflow → inline: copy into inline slots, free the old
    ///   overflow box.
    /// - overflow → overflow: allocate a new DepList, swap it into
    ///   the overflow pointer, free the old one.
    ///
    /// # Safety of overflow reclamation
    ///
    /// The caller must guarantee that no concurrent reader holds a
    /// pointer to the old overflow list at the time of this call.
    /// The runtime enforces this by taking `nodes.write()` for the
    /// duration of the call, which is mutually exclusive with any
    /// reader's `nodes.read()` guard. A later commit replaces this
    /// with epoch reclamation so recompute can run without the write
    /// lock; for now the write lock is the correctness mechanism.
    ///
    /// Stores use `Relaxed` ordering because the caller will issue a
    /// `Release` store on the node's state after this call (the
    /// Computing → Clean transition at the end of `run_compute`),
    /// which publishes these writes to subsequent readers via the
    /// standard Release-Acquire chain on state.
    #[allow(dead_code)]
    pub(crate) fn replace_deps(&self, new_deps: &[NodeId]) {
        let count = new_deps.len();
        assert!(
            count <= u8::MAX as usize,
            "deps overflow u8 count: {}",
            count
        );

        // Snapshot the old overflow pointer so we can reclaim it
        // after installing the new dep list. Load is Relaxed because
        // the caller already owns exclusive access via nodes.write()
        // and no other thread can be racing on this slot.
        let old_overflow = self.overflow_deps.load(Ordering::Relaxed);

        if count <= 7 {
            // New deps fit inline. Overwrite inline slots and clear
            // the overflow pointer. Slots beyond `count` are stale
            // but ignored by for_each_dep (which loops to count).
            for (i, dep) in new_deps.iter().enumerate() {
                self.inline_deps[i].store(dep.0, Ordering::Relaxed);
            }
            self.overflow_deps
                .store(std::ptr::null_mut(), Ordering::Relaxed);
        } else {
            // New deps spill to overflow. Allocate a fresh DepList
            // and install it. The inline slots are ignored when
            // dep_count > 7 so we do not touch them.
            let list = Box::new(DepList {
                deps: new_deps.to_vec().into_boxed_slice(),
            });
            let new_ptr = Box::into_raw(list);
            self.overflow_deps.store(new_ptr, Ordering::Relaxed);
        }

        self.dep_count.store(count as u8, Ordering::Relaxed);

        // Reclaim the old overflow box if present. The caller's
        // nodes.write() guarantees no concurrent reader can be
        // dereferencing this pointer right now.
        if !old_overflow.is_null() {
            // SAFETY: the pointer came from `Box::into_raw` in a
            // previous call to `publish_initial_deps` or
            // `replace_deps`; the caller holds nodes.write() so no
            // concurrent reader can still be using it.
            unsafe {
                drop(Box::from_raw(old_overflow));
            }
        }
    }

    /// Variant of `replace_deps` that leaks the old overflow list
    /// instead of reclaiming it.
    ///
    /// Required because SegmentedNodes has no reader/writer
    /// exclusion on node state, so freeing the old overflow pointer
    /// while a walker is mid-traversal would UAF. The leak is
    /// bounded: `NodeData::Drop` reclaims the currently-installed
    /// overflow list, and this method is only called on the rare
    /// dynamic-dep path where the dep set changes AND the node has
    /// more than 7 deps. Static-dep workloads never call it.
    ///
    /// Spec section 5.3 calls for epoch-based reclamation here. The
    /// planned fix (commit X of Gate 4) used `crossbeam-epoch 0.9`,
    /// which turns out not to be miri-clean due to integer-to-pointer
    /// casts in its internal thread-local list init. Rather than
    /// regress the miri-clean invariant for a bounded leak on a
    /// rarely-hit path, X was dropped from Gate 4 and the leak is
    /// kept as the permanent post-Gate-4 state. Proper reclamation
    /// is queued as a dedicated later chunk that will evaluate
    /// `seize`, `haphazard`, or a custom strict-provenance
    /// implementation.
    pub(crate) fn replace_deps_leaking_old_overflow(&self, new_deps: &[NodeId]) {
        let count = new_deps.len();
        assert!(
            count <= u8::MAX as usize,
            "deps overflow u8 count: {}",
            count
        );

        if count <= 7 {
            for (i, dep) in new_deps.iter().enumerate() {
                self.inline_deps[i].store(dep.0, Ordering::Relaxed);
            }
            // We do NOT clear overflow_deps here. If we did and the
            // previous deps were in overflow, we would lose the
            // pointer without freeing it AND Drop would fail to
            // reclaim the final overflow allocation. Leaving the
            // old overflow pointer in place means Drop still has
            // something to free; it just frees a list that is not
            // the current dep list. Accepted tradeoff for the
            // commit U stopgap.
            //
            // A reader that loads overflow_deps and sees the old
            // pointer will dereference the old list, but since we
            // also updated dep_count to `count` <= 7, the reader's
            // for_each_dep takes the inline branch (`if count <= 7`)
            // and never touches overflow_deps. So the stale pointer
            // is never read as deps; it is only used at Drop time.
        } else {
            let list = Box::new(DepList {
                deps: new_deps.to_vec().into_boxed_slice(),
            });
            let new_ptr = Box::into_raw(list);
            // Swap the pointer, leaking the old one (intentional
            // per the method contract). Drop will reclaim `new_ptr`
            // when the node itself drops, but any prior overflow
            // lists this slot held are leaked for the node's
            // lifetime.
            self.overflow_deps.store(new_ptr, Ordering::Relaxed);
        }

        self.dep_count.store(count as u8, Ordering::Relaxed);
    }

    /// Publish an initial dependency list on a `New` query node.
    ///
    /// This is the commit-D placeholder for the full compute-path dep
    /// publish that lands in commit F. It may be called exactly once,
    /// by the runtime, before the node has been read by any other
    /// thread. It does not coordinate with the state machine or with
    /// epoch reclamation; it assumes the caller owns the node's
    /// `Computing` state (or has equivalent exclusive access).
    ///
    /// Takes `&self` rather than `&mut self` because all underlying
    /// stores are on atomic fields that do not require exclusive
    /// reference. The caller accesses the node through a shared
    /// `nodes.read()` guard. The exclusivity guarantee needed for
    /// correctness comes from the state machine (Computing state is
    /// owned by exactly one thread at a time), not from Rust's
    /// aliasing rules.
    ///
    /// # Panics
    /// Panics in debug builds if called on a node that already has
    /// dependencies, or on a node whose state has already transitioned
    /// out of `New`. Safe but meaningless in release in those cases.
    pub(crate) fn publish_initial_deps(&self, deps: &[NodeId]) {
        debug_assert_eq!(
            self.dep_count.load(Ordering::Relaxed),
            0,
            "publish_initial_deps on a node with existing deps"
        );
        let count = deps.len();
        assert!(
            count <= u8::MAX as usize,
            "deps overflow u8 count: {}",
            count
        );
        if count <= 7 {
            for (i, dep) in deps.iter().enumerate() {
                self.inline_deps[i].store(dep.0, Ordering::Relaxed);
            }
        } else {
            let list = Box::new(DepList {
                deps: deps.to_vec().into_boxed_slice(),
            });
            let ptr = Box::into_raw(list);
            // Store uses Relaxed because the caller owns exclusive
            // access via the state machine (Computing state) and the
            // eventual publish step (state transition to Clean) will
            // Release-synchronize this write with any future reader.
            self.overflow_deps.store(ptr, Ordering::Relaxed);
        }
        self.dep_count.store(count as u8, Ordering::Relaxed);
    }

    /// Current state. Acquire-loaded: synchronizes with the Release
    /// store that published this state.
    #[inline]
    pub(crate) fn state(&self) -> NodeState {
        self.state.load_acquire()
    }

    /// The node's state cell. Exposes the AtomicNodeState helpers
    /// (CAS, try_claim_compute, etc.) to the runtime.
    #[inline]
    pub(crate) fn state_cell(&self) -> &AtomicNodeState {
        &self.state
    }

    /// Type tag for the arena holding this node's value. Immutable.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn type_tag(&self) -> u16 {
        self.type_tag
    }

    /// Arena slot index for this node's value. Immutable.
    #[inline]
    pub(crate) fn arena_slot(&self) -> u32 {
        self.arena_slot
    }

    /// Last-verified revision.
    #[inline]
    pub(crate) fn verified_at(&self) -> u64 {
        self.verified_at.load(Ordering::Relaxed)
    }

    /// Last-changed revision.
    #[inline]
    pub(crate) fn changed_at(&self) -> u64 {
        self.changed_at.load(Ordering::Relaxed)
    }

    /// Update this node's last-verified revision. Relaxed store;
    /// visibility to other threads is established by the subsequent
    /// Release on the node's state cell.
    #[inline]
    pub(crate) fn set_verified_at(&self, revision: u64) {
        self.verified_at.store(revision, Ordering::Relaxed);
    }

    /// Update this node's last-changed revision. Same ordering
    /// argument as `set_verified_at`.
    #[inline]
    pub(crate) fn set_changed_at(&self, revision: u64) {
        self.changed_at.store(revision, Ordering::Relaxed);
    }

    /// Current dependency count.
    #[inline]
    pub(crate) fn dep_count(&self) -> u8 {
        self.dep_count.load(Ordering::Relaxed)
    }

    /// Current generation counter. Handles carry an expected generation
    /// and verify it against this value on every access.
    ///
    /// Uses `Acquire` ordering so that a bump from another thread via
    /// `bump_generation` (which uses `Release`) establishes the required
    /// happens-before edge: a reader verifying a handle after a slot
    /// recycle observes the bumped counter and rejects the stale
    /// handle. Without `Acquire` here, the Release on the bump side
    /// pairs with nothing and a reader might see the pre-bump value
    /// indefinitely (addressed review finding C3).
    #[inline]
    pub(crate) fn generation(&self) -> u32 {
        self.generation.load(Ordering::Acquire)
    }

    /// Verify that a handle is valid for this node in a given runtime.
    ///
    /// Returns `Ok(())` if the handle's runtime id matches `runtime_id`
    /// AND the handle's generation matches this node's current
    /// generation. Returns a descriptive error otherwise. The runtime's
    /// public `get` / `set` methods turn these errors into panics; tests
    /// observe the `Result` directly.
    ///
    /// The caller is responsible for establishing happens-before with
    /// the most recent writer of this node (typically via an Acquire
    /// load on state before calling `verify_handle`).
    pub(crate) fn verify_handle<T: 'static>(
        &self,
        handle: Incr<T>,
        runtime_id: RuntimeId,
    ) -> Result<(), HandleError> {
        if handle.runtime_id() != runtime_id {
            return Err(HandleError::WrongRuntime {
                handle_runtime: handle.runtime_id(),
                current_runtime: runtime_id,
            });
        }
        let current = self.generation();
        if handle.generation() != current {
            return Err(HandleError::StaleGeneration {
                handle_generation: handle.generation(),
                current_generation: current,
            });
        }
        Ok(())
    }

    /// Bump this node's generation counter, invalidating all outstanding
    /// handles to the slot. Reserved for future use by `Runtime::delete_node`;
    /// not exercised in commit E because node deletion is not yet a
    /// capability of the runtime. Exposed now to lock in the Release
    /// ordering contract: any subsequent reader that verifies a handle
    /// with an Acquire-adjacent load sees the bumped generation and
    /// rejects the handle.
    #[cfg(test)]
    pub(crate) fn bump_generation(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Iterate over this node's dependencies.
    ///
    /// Reads the count, then reads the appropriate source (inline or
    /// overflow). The caller must have established happens-before with
    /// any writer of the deps via an Acquire load on the node state
    /// first; see the module docs.
    pub(crate) fn for_each_dep(&self, mut f: impl FnMut(NodeId)) {
        let count = self.dep_count.load(Ordering::Relaxed);
        if count <= 7 {
            for i in 0..(count as usize) {
                let raw = self.inline_deps[i].load(Ordering::Relaxed);
                f(NodeId(raw));
            }
        } else {
            let overflow = self.overflow_deps.load(Ordering::Relaxed);
            debug_assert!(
                !overflow.is_null(),
                "dep_count > 7 but overflow_deps is null"
            );
            // SAFETY: `overflow` is non-null when `count > 7` by the
            // invariant maintained in `publish_initial_deps`, and points
            // at a `DepList` allocated via `Box::into_raw` that lives
            // until this node's `Drop`. No mutation path in commit D
            // can swap or free this pointer between load and use.
            let list = unsafe { &*overflow };
            for &id in list.deps.iter() {
                f(id);
            }
        }
    }

    /// Collect dependencies into a `Vec<NodeId>`. Convenience for tests
    /// and diagnostics; production code uses `for_each_dep` to avoid
    /// the allocation where possible. The runtime's dep-diff path on
    /// recompute calls `collect_deps` to materialize the previous
    /// dep list for comparison against the newly-recorded set.
    pub(crate) fn collect_deps(&self) -> Vec<NodeId> {
        let mut out = Vec::with_capacity(self.dep_count() as usize);
        self.for_each_dep(|id| out.push(id));
        out
    }
}

impl Drop for NodeData {
    fn drop(&mut self) {
        // Reclaim the overflow dep list if one was allocated.
        let overflow = *self.overflow_deps.get_mut();
        if !overflow.is_null() {
            // SAFETY: `overflow` came from `Box::into_raw` in
            // `publish_initial_deps`; this Drop holds `&mut self`, so
            // no other thread can observe or mutate the pointer.
            unsafe {
                drop(Box::from_raw(overflow));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nodedata_is_one_cache_line() {
        // Redundant with the const assertion but lets a quick
        // `cargo test` surface the invariant with a readable failure.
        assert_eq!(std::mem::size_of::<NodeData>(), 64);
        assert_eq!(std::mem::align_of::<NodeData>(), 64);
    }

    #[test]
    fn new_input_starts_clean_with_no_deps() {
        let node = NodeData::new_input(3, 42, 17);
        assert_eq!(node.state(), NodeState::Clean);
        assert_eq!(node.type_tag(), 3);
        assert_eq!(node.arena_slot(), 42);
        assert_eq!(node.verified_at(), 17);
        assert_eq!(node.changed_at(), 17);
        assert_eq!(node.dep_count(), 0);
        assert!(node.collect_deps().is_empty());
    }

    #[test]
    fn new_query_starts_new_with_no_deps() {
        let node = NodeData::new_query(5, 99);
        assert_eq!(node.state(), NodeState::New);
        assert_eq!(node.type_tag(), 5);
        assert_eq!(node.arena_slot(), 99);
        assert_eq!(node.verified_at(), 0);
        assert_eq!(node.changed_at(), 0);
        assert_eq!(node.dep_count(), 0);
        assert!(node.collect_deps().is_empty());
    }

    #[test]
    fn publish_zero_inline_deps() {
        let node = NodeData::new_query(0, 0);
        node.publish_initial_deps(&[]);
        assert_eq!(node.dep_count(), 0);
        assert!(node.collect_deps().is_empty());
    }

    #[test]
    fn publish_one_inline_dep() {
        let node = NodeData::new_query(0, 0);
        node.publish_initial_deps(&[NodeId(42)]);
        assert_eq!(node.dep_count(), 1);
        assert_eq!(node.collect_deps(), vec![NodeId(42)]);
    }

    #[test]
    fn publish_seven_inline_deps_exactly() {
        let node = NodeData::new_query(0, 0);
        let deps: Vec<NodeId> = (0..7u32).map(NodeId).collect();
        node.publish_initial_deps(&deps);
        assert_eq!(node.dep_count(), 7);
        assert_eq!(node.collect_deps(), deps);
        // Overflow must still be null when we're inside the inline limit.
        assert!(node.overflow_deps.load(Ordering::Relaxed).is_null());
    }

    #[test]
    fn publish_eight_deps_spills_to_overflow() {
        let node = NodeData::new_query(0, 0);
        let deps: Vec<NodeId> = (100..108u32).map(NodeId).collect();
        node.publish_initial_deps(&deps);
        assert_eq!(node.dep_count(), 8);
        assert_eq!(node.collect_deps(), deps);
        // Overflow must be non-null when we exceed the inline limit.
        assert!(!node.overflow_deps.load(Ordering::Relaxed).is_null());
    }

    #[test]
    fn publish_many_deps_via_overflow() {
        let node = NodeData::new_query(0, 0);
        let deps: Vec<NodeId> = (1000..1100u32).map(NodeId).collect();
        node.publish_initial_deps(&deps);
        assert_eq!(node.dep_count(), 100);
        assert_eq!(node.collect_deps(), deps);
    }

    #[test]
    fn for_each_dep_visits_inline_deps_in_order() {
        let node = NodeData::new_query(0, 0);
        let expected = [NodeId(3), NodeId(1), NodeId(4), NodeId(1), NodeId(5)];
        node.publish_initial_deps(&expected);
        let mut visited = Vec::new();
        node.for_each_dep(|id| visited.push(id));
        assert_eq!(visited, expected);
    }

    #[test]
    fn for_each_dep_visits_overflow_deps_in_order() {
        let node = NodeData::new_query(0, 0);
        let expected: Vec<NodeId> = (0..12u32).rev().map(NodeId).collect();
        node.publish_initial_deps(&expected);
        let mut visited = Vec::new();
        node.for_each_dep(|id| visited.push(id));
        assert_eq!(visited, expected);
    }

    #[test]
    fn dropping_node_with_overflow_deps_is_leak_free() {
        // Use Miri to really verify; here we just exercise the Drop path.
        let node = NodeData::new_query(0, 0);
        let deps: Vec<NodeId> = (0..50u32).map(NodeId).collect();
        node.publish_initial_deps(&deps);
        drop(node);
    }

    #[test]
    fn dropping_node_without_overflow_is_trivial() {
        let node = NodeData::new_input(0, 0, 0);
        drop(node);
    }

    #[test]
    fn field_offsets_match_design() {
        // Cross-check the commented layout against the actual offsets
        // the compiler chose. If this test ever fails, the module's
        // layout comment is wrong and needs updating.
        let node = NodeData::new_input(0, 0, 0);
        let base = &node as *const NodeData as usize;
        let verified_at = &node.verified_at as *const _ as usize - base;
        let changed_at = &node.changed_at as *const _ as usize - base;
        let overflow_deps = &node.overflow_deps as *const _ as usize - base;
        let inline_deps = &node.inline_deps as *const _ as usize - base;
        let arena_slot = &node.arena_slot as *const _ as usize - base;
        let type_tag = &node.type_tag as *const _ as usize - base;
        let state = &node.state as *const _ as usize - base;
        let dep_count = &node.dep_count as *const _ as usize - base;
        let generation = &node.generation as *const _ as usize - base;

        assert_eq!(verified_at, 0, "verified_at at offset 0");
        assert_eq!(changed_at, 8, "changed_at at offset 8");
        assert_eq!(overflow_deps, 16, "overflow_deps at offset 16");
        assert_eq!(inline_deps, 24, "inline_deps at offset 24");
        assert_eq!(arena_slot, 52, "arena_slot at offset 52");
        assert_eq!(type_tag, 56, "type_tag at offset 56");
        assert_eq!(state, 58, "state at offset 58");
        assert_eq!(dep_count, 59, "dep_count at offset 59");
        assert_eq!(generation, 60, "generation at offset 60");
    }

    #[test]
    fn nodedata_implements_send_and_sync() {
        // Compile-time check: if NodeData accidentally loses Send/Sync
        // (e.g., someone adds a raw pointer field without wrapping it),
        // this fn will fail to compile.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NodeData>();
    }

    #[test]
    fn new_node_starts_at_generation_zero() {
        let input = NodeData::new_input(0, 0, 0);
        assert_eq!(input.generation(), 0);
        let query = NodeData::new_query(0, 0);
        assert_eq!(query.generation(), 0);
    }

    #[test]
    fn verify_handle_succeeds_on_match() {
        let node = NodeData::new_input(0, 0, 0);
        let rid = RuntimeId::from_raw(42);
        let h: Incr<u64> = Incr::new(7, 0, rid);
        assert!(node.verify_handle(h, rid).is_ok());
    }

    #[test]
    fn verify_handle_rejects_wrong_runtime() {
        let node = NodeData::new_input(0, 0, 0);
        let rid_a = RuntimeId::from_raw(42);
        let rid_b = RuntimeId::from_raw(43);
        let h: Incr<u64> = Incr::new(7, 0, rid_a);
        let err = node.verify_handle(h, rid_b).unwrap_err();
        assert_eq!(
            err,
            HandleError::WrongRuntime {
                handle_runtime: rid_a,
                current_runtime: rid_b,
            }
        );
    }

    #[test]
    fn verify_handle_rejects_stale_generation() {
        let node = NodeData::new_input(0, 0, 0);
        let rid = RuntimeId::from_raw(42);
        // Handle with an out-of-date generation.
        let h: Incr<u64> = Incr::new(7, 5, rid);
        let err = node.verify_handle(h, rid).unwrap_err();
        assert_eq!(
            err,
            HandleError::StaleGeneration {
                handle_generation: 5,
                current_generation: 0,
            }
        );
    }

    #[test]
    fn bumping_generation_invalidates_outstanding_handles() {
        let node = NodeData::new_input(0, 0, 0);
        let rid = RuntimeId::from_raw(42);
        let h: Incr<u64> = Incr::new(7, 0, rid);
        // Fresh handle works.
        assert!(node.verify_handle(h, rid).is_ok());
        // Bump the generation (simulating a slot recycle).
        node.bump_generation();
        assert_eq!(node.generation(), 1);
        // Old handle no longer works.
        let err = node.verify_handle(h, rid).unwrap_err();
        assert!(matches!(err, HandleError::StaleGeneration { .. }));
        // A handle with the new generation would work.
        let h2: Incr<u64> = Incr::new(7, 1, rid);
        assert!(node.verify_handle(h2, rid).is_ok());
    }

    #[test]
    fn verify_handle_checks_runtime_before_generation() {
        // If both runtime and generation are wrong, the runtime error
        // should win because it is the more specific failure (cross-
        // runtime handles are a hard bug; stale generations are a
        // legitimate state after a node has been recycled).
        let node = NodeData::new_input(0, 0, 0);
        node.bump_generation(); // current generation = 1
        let rid_other = RuntimeId::from_raw(99);
        let rid_this = RuntimeId::from_raw(42);
        let h: Incr<u64> = Incr::new(7, 0, rid_other);
        let err = node.verify_handle(h, rid_this).unwrap_err();
        assert!(
            matches!(err, HandleError::WrongRuntime { .. }),
            "runtime mismatch should be reported before generation mismatch"
        );
    }
}
