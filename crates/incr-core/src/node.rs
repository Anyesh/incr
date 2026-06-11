//! `NodeData<C: Cells>`: the per-node read-hot struct, parameterized over
//! the [`Cells`] strategy. Production `incr-concurrent` uses a 64-byte
//! cache-line-aligned layout to keep reader traversal at one cache line
//! per node; this design carries forward unchanged into `incr-core`.
//!
//! The const-time size and alignment assertions are load-bearing under
//! both strategies. The spike validated that field-by-field the `Local`
//! cells (Cell-backed) and `Shared` cells (atomic-backed) produce the
//! same layout, so the same 64-byte total holds.
//!
//! ## Layout (both strategies)
//!
//! ```text
//! offset  size   field
//! ------  ----   -----
//!    0     8     verified_at   Cells::U64
//!    8     8     changed_at    Cells::U64
//!   16     8     overflow_deps Cells::U64   (raw pointer stored as u64 for the spike;
//!                                            full DepList-pointer machinery lands in
//!                                            the next step of the consolidation)
//!   24    28     inline_deps   [Cells::U32; 7]
//!   52     4     arena_slot    u32
//!   56     2     type_tag      u16
//!   58     1     state         Cells::State
//!   59     1     dep_count     Cells::U8
//!   60     4     generation    Cells::U32
//! ```
//!
//! Total: 64 bytes, 64-byte aligned. Asserted at compile time below.
//!
//! ## What lands later
//!
//! The first incr-core slice covers the layout and the basic accessors.
//! The next slice ports the inline-7 + heap-overflow dep storage with
//! proper Drop reclamation (replacing the leaky `replace_deps_leaking_old_overflow`
//! from production with a hazard-pointer reclaimed path). The slice
//! after that lifts the segmented node store. Tracking in the
//! consolidation plan.

use crate::cells::Cells;
use crate::state::NodeState;
use haphazard::{AtomicPtr as HzAtomicPtr, HazardPointer};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const SENTINEL: NodeId = NodeId(u32::MAX);
}

/// Heap-allocated overflow dependency list. Used when a node has more
/// than seven dependencies.
///
/// Reclamation policy: when a node's dep set changes and the new list
/// requires re-allocation, the OLD overflow list is retired through
/// the `haphazard` global domain. Hazard-pointer protection in
/// [`NodeData::for_each_dep`] guarantees concurrent readers can finish
/// their traversal before the retired list is freed. Memory is
/// reclaimed during normal operation (not just at runtime drop), so
/// long-lived runtimes with churning dynamic deps no longer accumulate
/// retired lists.
pub struct DepList {
    pub(crate) deps: Box<[NodeId]>,
}

#[repr(C, align(64))]
pub struct NodeData<C: Cells> {
    pub(crate) verified_at: C::U64,
    pub(crate) changed_at: C::U64,
    pub(crate) overflow_deps: HzAtomicPtr<DepList>,
    pub(crate) inline_deps: [C::U32; 7],
    pub(crate) arena_slot: u32,
    pub(crate) type_tag: u16,
    pub(crate) state: C::State,
    pub(crate) dep_count: C::U8,
    pub(crate) generation: C::U32,
}

impl<C: Cells> NodeData<C> {
    /// Construct a new input node. Input nodes start `Clean` because their
    /// value is provided at creation. `revision` seeds both `verified_at`
    /// and `changed_at`.
    pub fn new_input(type_tag: u16, arena_slot: u32, revision: u64) -> Self {
        Self {
            verified_at: C::new_u64(revision),
            changed_at: C::new_u64(revision),
            overflow_deps: unsafe { HzAtomicPtr::new(std::ptr::null_mut()) },
            inline_deps: Self::empty_inline_deps(),
            arena_slot,
            type_tag,
            state: C::new_state(NodeState::Clean.as_u8()),
            dep_count: C::new_u8(0),
            generation: C::new_u32(0),
        }
    }

    /// Construct a new query node. Query nodes start `New` because their
    /// value has not been computed; the first reader CASes to `Computing`
    /// and runs the compute closure.
    pub fn new_query(type_tag: u16, arena_slot: u32) -> Self {
        Self {
            verified_at: C::new_u64(0),
            changed_at: C::new_u64(0),
            overflow_deps: unsafe { HzAtomicPtr::new(std::ptr::null_mut()) },
            inline_deps: Self::empty_inline_deps(),
            arena_slot,
            type_tag,
            state: C::new_state(NodeState::New.as_u8()),
            dep_count: C::new_u8(0),
            generation: C::new_u32(0),
        }
    }

    #[inline(always)]
    pub fn arena_slot(&self) -> u32 {
        self.arena_slot
    }

    #[inline(always)]
    pub fn type_tag(&self) -> u16 {
        self.type_tag
    }

    #[inline(always)]
    pub fn state(&self) -> NodeState {
        NodeState::from_u8(C::state_load_acquire(&self.state))
    }

    #[inline(always)]
    pub fn state_cell(&self) -> &C::State {
        &self.state
    }

    #[inline(always)]
    pub fn verified_at(&self) -> u64 {
        C::u64_load_acquire(&self.verified_at)
    }

    #[inline(always)]
    pub fn changed_at(&self) -> u64 {
        C::u64_load_acquire(&self.changed_at)
    }

    #[inline(always)]
    pub fn set_verified_at(&self, v: u64) {
        C::u64_store_release(&self.verified_at, v);
    }

    #[inline(always)]
    pub fn set_changed_at(&self, v: u64) {
        C::u64_store_release(&self.changed_at, v);
    }

    /// Raise `verified_at` to `v` if `v` is larger. Protocol stamps must
    /// use max, not store: two threads can race to stamp (a verifier and
    /// a computer) and a plain store could regress the timestamp, masking
    /// a change from a later verifier.
    #[inline(always)]
    pub fn max_verified_at(&self, v: u64) {
        C::u64_fetch_max(&self.verified_at, v);
    }

    /// Raise `changed_at` to `v` if `v` is larger. See [`Self::max_verified_at`].
    #[inline(always)]
    pub fn max_changed_at(&self, v: u64) {
        C::u64_fetch_max(&self.changed_at, v);
    }

    /// Sentinel meaning "a set() is in flight on this input". Any
    /// verifier comparing `changed_at > my_verified` sees the marker as
    /// infinitely new and recomputes instead of verifying clean. Without
    /// it there is a mask window: a setter that has bumped the global
    /// revision but not yet stamped `changed_at` lets a concurrent
    /// verifier stamp `verified_at` at or above the setter's revision
    /// while having read the old stamp, after which the change is
    /// invisible forever.
    pub const CHANGED_PENDING: u64 = u64::MAX;

    /// Enter the pending-marker state. Must happen BEFORE the value swap
    /// and revision bump in `set()`; see [`Self::CHANGED_PENDING`].
    #[inline(always)]
    pub fn mark_changed_pending(&self) {
        C::u64_fetch_max(&self.changed_at, Self::CHANGED_PENDING);
    }

    /// Replace the pending marker with the setter's concrete revision,
    /// without regressing a concurrent setter's higher revision and
    /// without erasing a concurrent setter's fresh marker more than
    /// necessary (max semantics among concrete revisions).
    pub fn settle_changed_at(&self, r: u64) {
        let mut cur = C::u64_load_acquire(&self.changed_at);
        loop {
            let target = if cur == Self::CHANGED_PENDING || r > cur {
                r
            } else {
                return;
            };
            match C::u64_compare_exchange(&self.changed_at, cur, target) {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Raw dep-count cell value: exact count for 0..=7 (inline storage),
    /// [`Self::OVERFLOW_MARKER`] when deps live in the overflow list
    /// (whose length is authoritative).
    #[inline(always)]
    pub fn dep_count(&self) -> u8 {
        C::u8_load_relaxed(&self.dep_count)
    }

    #[inline(always)]
    pub fn generation(&self) -> u32 {
        C::u32_load_relaxed(&self.generation)
    }

    #[inline(always)]
    pub fn set_state(&self, v: u8) {
        C::state_store_release(&self.state, v);
    }

    /// Install a new dep list. Inline-7 path stores into the inline
    /// array; overflow path heap-allocates a `DepList` and Release-stores
    /// the pointer.
    ///
    /// Reclamation: any displaced overflow pointer is retired through
    /// the `haphazard` global domain. Concurrent readers in
    /// [`Self::for_each_dep`] hold a `HazardPointer` while
    /// dereferencing the slot, so the retired list is not freed until
    /// every protecting reader has finished. Free-during-runtime; no
    /// graveyard build-up; no leak past process scope.
    pub(crate) fn install_deps(&self, new_deps: &[NodeId]) {
        let count = new_deps.len();
        if count <= 7 {
            for (i, dep) in new_deps.iter().enumerate() {
                C::u32_store_relaxed(&self.inline_deps[i], dep.0);
            }
            // Going inline: swap null in to displace any stale overflow
            // pointer, then retire it. for_each_dep's count<=7 path
            // never reads overflow_deps, but a subsequent overflow
            // install would clobber the stale pointer without retiring
            // it, leaking memory. Retiring here closes that gap.
            // SAFETY: swap_ptr with a null target is safe.
            let displaced = unsafe { self.overflow_deps.swap_ptr(std::ptr::null_mut()) };
            if let Some(old) = displaced {
                // SAFETY: `old` came from a Box::into_raw via a previous
                // install_deps; it is not aliased for writes (the state
                // machine guarantees single-writer-at-a-time on this
                // node). Hazard pointers ensure the actual free is
                // deferred until no reader still references this list.
                unsafe { old.retire() };
            }
            C::u8_store_release(&self.dep_count, count as u8);
        } else {
            let list = Box::new(DepList {
                deps: new_deps.to_vec().into_boxed_slice(),
            });
            // `swap` takes ownership of the box; the old box (if any)
            // is wrapped in a `Replaced` whose `retire` defers free
            // through the global hazard-pointer domain.
            let replaced = self.overflow_deps.swap(list);
            // The cell is one byte, so counts above 7 are encoded as a
            // marker and the DepList length is authoritative. This is
            // what lifts the old 255-dep ceiling: a query may read any
            // number of inputs.
            C::u8_store_release(&self.dep_count, Self::OVERFLOW_MARKER);
            if let Some(old) = replaced {
                // SAFETY: same as the inline-path retire above.
                unsafe { old.retire() };
            }
        }
    }

    /// Sentinel stored in `dep_count` when the dep list lives in
    /// `overflow_deps`. Any value above 7 would do; 8 is the smallest.
    pub const OVERFLOW_MARKER: u8 = 8;

    /// Iterate over the node's recorded dependencies. The caller must
    /// have observed the node's state via an Acquire load (e.g., through
    /// the state machine) to synchronize with the writer of these deps.
    ///
    /// Up to 7 deps live inline; beyond that, `overflow_deps` points at
    /// a heap-allocated `DepList` whose load is protected by a
    /// `HazardPointer`. A concurrent retire by an `install_deps`
    /// writer will defer the actual free until this reader's hazard
    /// is released.
    ///
    /// The dispatch is intentionally split: the inline fast path is
    /// `#[inline]`-friendly and stays small enough to be inlined into
    /// the caller. The cold overflow path is `#[inline(never)]` so
    /// `HazardPointer::new` (thread_local lookup + potential allocation)
    /// is not duplicated into every call site.
    #[inline]
    pub fn for_each_dep(&self, mut f: impl FnMut(NodeId)) {
        // Acquire pairs with install_deps's Release store on dep_count:
        // a reader that observes the new count must also observe the
        // inline entries (or the overflow pointer) written before it.
        // Relaxed here could yield a count ahead of the entries and index
        // sentinel garbage.
        let count = C::u8_load_acquire(&self.dep_count);
        if count <= 7 {
            for i in 0..(count as usize) {
                let raw = C::u32_load_relaxed(&self.inline_deps[i]);
                f(NodeId(raw));
            }
        } else {
            self.for_each_overflow_dep(&mut f);
        }
    }

    #[inline(never)]
    fn for_each_overflow_dep(&self, f: &mut dyn FnMut(NodeId)) {
        let mut hazard = HazardPointer::new();
        // SAFETY: the AtomicPtr is populated by install_deps with
        // Box-allocated DepLists; retirements go through the global
        // haphazard domain so safe_load returns a reference that
        // remains valid for the lifetime of `hazard`.
        let list_ref: Option<&DepList> = unsafe { self.overflow_deps.load(&mut hazard) };
        let list = list_ref.expect("overflow_deps null with dep_count > 7");
        for &id in list.deps.iter() {
            f(id);
        }
    }

    fn empty_inline_deps() -> [C::U32; 7] {
        [
            C::new_u32(NodeId::SENTINEL.0),
            C::new_u32(NodeId::SENTINEL.0),
            C::new_u32(NodeId::SENTINEL.0),
            C::new_u32(NodeId::SENTINEL.0),
            C::new_u32(NodeId::SENTINEL.0),
            C::new_u32(NodeId::SENTINEL.0),
            C::new_u32(NodeId::SENTINEL.0),
        ]
    }
}

impl<C: Cells> Drop for NodeData<C> {
    fn drop(&mut self) {
        // Swap null in and retire whatever was installed. The actual
        // free is deferred through the haphazard global domain, which
        // reclaims it the next time a domain pass detects no protecting
        // hazard pointers. For the runtime-drop case all hazards are
        // already gone, so reclamation is immediate.
        // SAFETY: swap_ptr to null is safe; the displaced pointer (if
        // any) came from install_deps's Box::into_raw and goes through
        // haphazard's retire path.
        let displaced = unsafe { self.overflow_deps.swap_ptr(std::ptr::null_mut()) };
        if let Some(old) = displaced {
            unsafe { old.retire() };
        }
    }
}

const _: () = assert!(
    std::mem::size_of::<NodeData<crate::cells::Local>>() == 64,
    "NodeData<Local> must be exactly one 64-byte cache line"
);
const _: () = assert!(
    std::mem::align_of::<NodeData<crate::cells::Local>>() == 64,
    "NodeData<Local> must be 64-byte aligned"
);
const _: () = assert!(
    std::mem::size_of::<NodeData<crate::cells::Shared>>() == 64,
    "NodeData<Shared> must be exactly one 64-byte cache line"
);
const _: () = assert!(
    std::mem::align_of::<NodeData<crate::cells::Shared>>() == 64,
    "NodeData<Shared> must be 64-byte aligned"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    #[test]
    fn local_node_is_64_bytes() {
        assert_eq!(std::mem::size_of::<NodeData<Local>>(), 64);
        assert_eq!(std::mem::align_of::<NodeData<Local>>(), 64);
    }

    #[test]
    fn shared_node_is_64_bytes() {
        assert_eq!(std::mem::size_of::<NodeData<Shared>>(), 64);
        assert_eq!(std::mem::align_of::<NodeData<Shared>>(), 64);
    }

    #[test]
    fn local_input_roundtrip() {
        let n: NodeData<Local> = NodeData::new_input(0, 42, 7);
        assert_eq!(n.arena_slot(), 42);
        assert_eq!(n.verified_at(), 7);
        assert_eq!(n.changed_at(), 7);
        assert_eq!(n.state(), NodeState::Clean);
        n.set_verified_at(11);
        assert_eq!(n.verified_at(), 11);
    }

    #[test]
    fn shared_input_roundtrip() {
        let n: NodeData<Shared> = NodeData::new_input(0, 42, 7);
        assert_eq!(n.arena_slot(), 42);
        assert_eq!(n.verified_at(), 7);
        assert_eq!(n.changed_at(), 7);
        assert_eq!(n.state(), NodeState::Clean);
        n.set_verified_at(11);
        assert_eq!(n.verified_at(), 11);
    }

    #[test]
    fn local_query_starts_new() {
        let n: NodeData<Local> = NodeData::new_query(0, 99);
        assert_eq!(n.state(), NodeState::New);
        assert_eq!(n.dep_count(), 0);
    }

    #[test]
    fn shared_query_starts_new() {
        let n: NodeData<Shared> = NodeData::new_query(0, 99);
        assert_eq!(n.state(), NodeState::New);
        assert_eq!(n.dep_count(), 0);
    }

    #[test]
    fn max_stamps_are_monotonic() {
        let n: NodeData<Shared> = NodeData::new_input(0, 0, 10);
        n.max_verified_at(7);
        assert_eq!(n.verified_at(), 10);
        n.max_verified_at(12);
        assert_eq!(n.verified_at(), 12);
        n.max_changed_at(11);
        assert_eq!(n.changed_at(), 11);
        n.max_changed_at(5);
        assert_eq!(n.changed_at(), 11);
    }

    #[test]
    fn deps_beyond_255_roundtrip() {
        for_both_strategies_deps(300);
        for_both_strategies_deps(8);
        for_both_strategies_deps(7);
    }

    fn for_both_strategies_deps(n: u32) {
        let local: NodeData<Local> = NodeData::new_query(0, 0);
        let shared: NodeData<Shared> = NodeData::new_query(0, 0);
        let deps: Vec<NodeId> = (0..n).map(NodeId).collect();
        local.install_deps(&deps);
        shared.install_deps(&deps);
        for node_deps in [collect_deps(&local) as Vec<NodeId>, collect_deps(&shared)] {
            assert_eq!(node_deps, deps, "roundtrip failed for n={}", n);
        }
        if n > 7 {
            assert_eq!(local.dep_count(), NodeData::<Local>::OVERFLOW_MARKER);
        }
    }

    fn collect_deps<C: crate::cells::Cells>(node: &NodeData<C>) -> Vec<NodeId> {
        let mut out = Vec::new();
        node.for_each_dep(|d| out.push(d));
        out
    }

    #[test]
    fn shrink_from_overflow_back_to_inline() {
        let n: NodeData<Local> = NodeData::new_query(0, 0);
        let big: Vec<NodeId> = (0..20).map(NodeId).collect();
        n.install_deps(&big);
        assert_eq!(collect_deps(&n), big);
        let small: Vec<NodeId> = (0..3).map(NodeId).collect();
        n.install_deps(&small);
        assert_eq!(collect_deps(&n), small);
        assert_eq!(n.dep_count(), 3);
    }
}
