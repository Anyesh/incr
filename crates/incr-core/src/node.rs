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
        assert!(count <= u8::MAX as usize, "dep count exceeds u8");
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
            C::u8_store_release(&self.dep_count, count as u8);
            if let Some(old) = replaced {
                // SAFETY: same as the inline-path retire above.
                unsafe { old.retire() };
            }
        }
    }

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
        let count = C::u8_load_relaxed(&self.dep_count);
        if count <= 7 {
            for i in 0..(count as usize) {
                let raw = C::u32_load_relaxed(&self.inline_deps[i]);
                f(NodeId(raw));
            }
        } else {
            self.for_each_overflow_dep(&mut f);
            return;
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
}
