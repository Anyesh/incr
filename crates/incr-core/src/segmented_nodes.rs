//! Segmented lock-free-on-read store for [`NodeData<C>`].
//!
//! Nodes live in never-moving segments of `SEGMENT_SIZE` behind a
//! `Directory` that doubles as the store grows, so a slot index splits
//! into a segment index and an offset by a shift and a mask. A read loads
//! the directory with Acquire, bounds-checks the segment index, loads the
//! segment pointer with Relaxed and tests it for null.
//!
//! The Relaxed segment load is enough because a reader only ever holds a
//! slot it reached through a happens-before chain from that slot's
//! `push`: a handle passed between threads by a synchronizing channel, a
//! dep id Release-stored and Acquire-loaded or read from the
//! hazard-protected overflow list, a dependent or group list read under
//! the runtime's lock, a table bucket published by its Release. That
//! `push` stored the segment pointer, so the reader sees it, in the
//! directory it went into or in a later one copied from it (see the
//! `directory` module). A null segment means a path without that chain,
//! which is reported as an internal invariant violation.
//!
//! Segments are freed only when the store drops, so a `&NodeData<C>`
//! stays valid for the store's lifetime. Pushes are serialized by the
//! runtime's write-side lock; the store does not order writers itself.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

use crate::cells::{Cells, PtrCell};
use crate::directory::Directory;
use crate::node::{NodeData, NodeId};

// Four-node segments under loom and Miri, so a model or a Miri test
// allocates segments and grows the directory without creating thousands
// of nodes.
#[cfg(any(loom, miri))]
const SEGMENT_SHIFT: u32 = 2;
#[cfg(not(any(loom, miri)))]
const SEGMENT_SHIFT: u32 = 10;
const SEGMENT_SIZE: usize = 1 << SEGMENT_SHIFT;
const SEGMENT_MASK: u32 = (SEGMENT_SIZE as u32) - 1;
// One entry under loom, so a model doubles the directory on its second
// segment.
#[cfg(loom)]
const FIRST_DIRECTORY_LEN: usize = 1;
#[cfg(not(loom))]
const FIRST_DIRECTORY_LEN: usize = 16;

/// Maximum total nodes per runtime. Slots run from 0 to
/// `MAX_NODES - 1 = u32::MAX - 1025`.
pub const MAX_NODES: u32 = u32::MAX - 1024;

const _: () = assert!(
    MAX_NODES - 1 < NodeId::SENTINEL.0,
    "NodeId::SENTINEL must never be a valid slot"
);

/// `NodeFlags` bit: the node owns nodes (`Inner::owned` has an entry
/// for it), so deletes look the entry up only when it is set.
pub(crate) const HAS_CHILDREN: u8 = 1;

/// `NodeFlags` bit: the node is a member of a landed cycle group
/// (`Inner::member_recs` has an entry for it).
pub(crate) const IN_GROUP: u8 = 2;

/// `NodeFlags` bit: the node has a cycle strategy (`Inner::cycle_strategies`
/// has an entry for it, or else `Inner::type_fallbacks` has one for its
/// type).
pub(crate) const HAS_STRATEGY: u8 = 4;

/// `NodeFlags` bit: the node has created owned children through
/// `child_input` or `child_query` (`Inner::children` has its index).
/// Unlike [`HAS_CHILDREN`] it stays set when every child is gone, so a
/// later run reuses the index instead of starting a second one.
pub(crate) const HAS_INDEX: u8 = 8;

/// `NodeFlags` bit: the node is owned (`Inner::owner_of` has an entry
/// for it), so a delete looks the entry up only when it is set.
pub(crate) const OWNED: u8 = 16;

/// `NodeFlags` bit: the query was created with `Replay::Allowed`.
pub(crate) const REPLAYABLE: u8 = 32;

/// `NodeFlags` bit: the query must never be replayed, whatever
/// `INCR_TEST_ALL_REPLAYABLE` says: a cursor-driven collection operator,
/// whose run consumes its upstream's log.
pub(crate) const NOT_REPLAYABLE: u8 = 64;

/// `NodeFlags` bit: the node is watched. Unlike the other bits it can
/// change at any time, so it is decided on only inside an `inner`
/// section (`runtime_watch.rs`).
pub(crate) const WATCHED: u8 = 128;

const _: () = assert!(crate::bits::disjoint(&[
    HAS_CHILDREN as u64,
    IN_GROUP as u64,
    HAS_STRATEGY as u64,
    HAS_INDEX as u64,
    OWNED as u64,
    REPLAYABLE as u64,
    NOT_REPLAYABLE as u64,
    WATCHED as u64,
]));

/// A segment of nodes and, beside them, one flags byte per node. The
/// bytes are zeroed when the segment is allocated and stored only under
/// the runtime's write lock, so a Relaxed load anywhere reads a value
/// some write-lock holder stored.
struct Segment<C: Cells> {
    slots: [UnsafeCell<MaybeUninit<NodeData<C>>>; SEGMENT_SIZE],
    flags: [C::U8; SEGMENT_SIZE],
}

/// Strategy-parameterized segmented node store.
pub struct SegmentedNodes<C: Cells> {
    segments: Directory<Segment<C>, C>,
    len: C::U32,
}

impl<C: Cells> SegmentedNodes<C> {
    /// Construct an empty store. No segments are allocated until the
    /// first push.
    pub fn new() -> Self {
        Self {
            segments: Directory::with_len(FIRST_DIRECTORY_LEN),
            len: C::new_u32(0),
        }
    }

    /// Append `node` and return its slot index. Caller must hold the
    /// runtime's write-side lock (or be single-threaded under Local) so
    /// no concurrent writer races on `len` or segment allocation.
    pub fn push(&self, node: NodeData<C>) -> u32 {
        let slot = C::u32_load_relaxed(&self.len);
        assert!(
            slot < MAX_NODES,
            "SegmentedNodes exhausted at {} slots",
            MAX_NODES
        );
        let seg_idx = (slot >> SEGMENT_SHIFT) as usize;
        if slot & SEGMENT_MASK == 0 {
            self.alloc_segment(seg_idx);
        }
        let seg_ptr = self
            .segments
            .get(seg_idx)
            .expect("incr-core: a pushed slot's segment has a directory entry")
            .load_relaxed();

        // SAFETY: the segment was allocated by this push or an earlier
        // one and is freed only on drop. `slot` is past `len`, so no
        // reader holds it, and the caller serializes writers.
        unsafe {
            (*(*seg_ptr).slots[(slot & SEGMENT_MASK) as usize].get()).write(node);
        }

        // Readers never synchronize through `len`: they reach a slot
        // through a happens-before chain from its push (module doc).
        C::u32_store_relaxed(&self.len, slot + 1);
        slot
    }

    #[cold]
    #[inline(never)]
    fn alloc_segment(&self, seg_idx: usize) {
        self.segments.grow_to(seg_idx);
        let mut segment = Box::<Segment<C>>::new_uninit();
        let p = segment.as_mut_ptr();
        // SAFETY: every flags byte is written before `assume_init`; the
        // slots are `MaybeUninit` and need no initialization.
        let segment = unsafe {
            for i in 0..SEGMENT_SIZE {
                std::ptr::addr_of_mut!((*p).flags[i]).write(C::new_u8(0));
            }
            segment.assume_init()
        };
        self.segments
            .get(seg_idx)
            .expect("incr-core: grow_to left the segment's entry out")
            .store_release(Box::into_raw(segment));
    }

    /// Read the node at `slot`. The returned reference is valid for the
    /// store's lifetime.
    ///
    /// Caller must have obtained `slot` from `push` on this store.
    /// Debug builds assert `slot < len`; release builds skip the check
    /// and rely on the caller's invariant.
    pub fn get(&self, slot: u32) -> &NodeData<C> {
        self.get_with_flags(slot).0
    }

    /// The node at `slot` and its flags byte, through one directory walk.
    #[inline(always)]
    pub(crate) fn get_with_flags(&self, slot: u32) -> (&NodeData<C>, &C::U8) {
        debug_assert!(
            slot < C::u32_load_relaxed(&self.len),
            "SegmentedNodes::get slot {} out of range (len {})",
            slot,
            C::u32_load_relaxed(&self.len),
        );

        let seg_idx = (slot >> SEGMENT_SHIFT) as usize;
        let seg_ptr = match self.segments.get(seg_idx) {
            Some(entry) => entry.load_relaxed(),
            None => segment_not_visible(seg_idx),
        };
        if seg_ptr.is_null() {
            segment_not_visible(seg_idx);
        }

        // SAFETY: the caller reached `slot` through a happens-before
        // chain from its push (module doc), so the slot is initialized
        // and the non-null segment is live until drop.
        unsafe {
            let within = (slot & SEGMENT_MASK) as usize;
            (
                (*(*seg_ptr).slots[within].get()).assume_init_ref(),
                &(*seg_ptr).flags[within],
            )
        }
    }

    /// Number of initialized slots.
    pub fn len(&self) -> u32 {
        C::u32_load_relaxed(&self.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cold]
#[inline(never)]
fn segment_not_visible(seg_idx: usize) -> ! {
    panic!(
        "incr-core: internal invariant violated: segment {} read before its allocation \
         was published to this thread",
        seg_idx
    );
}

impl<C: Cells> Default for SegmentedNodes<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Cells> Drop for SegmentedNodes<C> {
    fn drop(&mut self) {
        let final_len = C::u32_load_relaxed(&self.len);
        for slot in 0..final_len {
            let seg_ptr = self.segments.get((slot >> SEGMENT_SHIFT) as usize);
            let seg_ptr = seg_ptr.map_or(std::ptr::null_mut(), |e| e.load_relaxed());
            if !seg_ptr.is_null() {
                // SAFETY: slot < final_len, so push initialized it; `&mut
                // self` rules out readers.
                unsafe {
                    (*(*seg_ptr).slots[(slot & SEGMENT_MASK) as usize].get()).assume_init_drop()
                };
            }
        }
        for index in 0..self.segments.len() {
            let Some(entry) = self.segments.get(index) else {
                break;
            };
            let segment = entry.load_relaxed();
            if !segment.is_null() {
                // SAFETY: came from `Box::into_raw` in `alloc_segment`, and
                // `&mut self` rules out readers.
                drop(unsafe { Box::from_raw(segment) });
            }
        }
    }
}

// SAFETY (Shared only): `NodeData<Shared>` is `Send + Sync` because all
// its fields are atomic, and the directory's entries are `AtomicPtr`s.
// The directory is `Send + Sync` whatever it points at, so the assert
// below must name `NodeData<Shared>` itself.
// Under Local, `LocalPtrCell` is `!Sync` (via `Cell`), so the resulting
// `SegmentedNodes<Local>` is also `!Sync`, which is the correct property
// for the single-threaded variant. We rely on auto-derived Send/Sync
// here rather than manual unsafe impls; the per-strategy auto traits do
// the right thing without our intervention.

const _: fn() = || {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<NodeData<crate::cells::Shared>>();
    send_sync::<SegmentedNodes<crate::cells::Shared>>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    #[test]
    fn local_push_get() {
        let store: SegmentedNodes<Local> = SegmentedNodes::new();
        let slot = store.push(NodeData::<Local>::new_input(0, 42, 2));
        assert_eq!(store.get(slot).arena_slot(), 42);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn shared_push_get() {
        let store: SegmentedNodes<Shared> = SegmentedNodes::new();
        let slot = store.push(NodeData::<Shared>::new_input(0, 42, 2));
        assert_eq!(store.get(slot).arena_slot(), 42);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn local_many_pushes_cross_segment_boundary() {
        let store: SegmentedNodes<Local> = SegmentedNodes::new();
        let count = SEGMENT_SIZE + 100;
        let mut slots = Vec::with_capacity(count);
        for i in 0..count {
            slots.push(store.push(NodeData::<Local>::new_input(0, i as u32, 2)));
        }
        for (i, slot) in slots.into_iter().enumerate() {
            assert_eq!(store.get(slot).arena_slot(), i as u32);
        }
        assert_eq!(store.len(), count as u32);
    }

    #[test]
    fn shared_many_pushes_cross_segment_boundary() {
        let store: SegmentedNodes<Shared> = SegmentedNodes::new();
        let count = SEGMENT_SIZE + 100;
        let mut slots = Vec::with_capacity(count);
        for i in 0..count {
            slots.push(store.push(NodeData::<Shared>::new_input(0, i as u32, 2)));
        }
        for (i, slot) in slots.into_iter().enumerate() {
            assert_eq!(store.get(slot).arena_slot(), i as u32);
        }
        assert_eq!(store.len(), count as u32);
    }

    #[test]
    fn local_references_stay_valid_across_growth() {
        let store: SegmentedNodes<Local> = SegmentedNodes::new();
        let slot_a = store.push(NodeData::<Local>::new_input(0, 111, 2));
        let ref_a = store.get(slot_a);
        for i in 0..(SEGMENT_SIZE as u32 + 10) {
            store.push(NodeData::<Local>::new_input(0, 1000 + i, 2));
        }
        assert_eq!(ref_a.arena_slot(), 111);
    }

    #[test]
    fn shared_references_stay_valid_across_growth() {
        let store: SegmentedNodes<Shared> = SegmentedNodes::new();
        let slot_a = store.push(NodeData::<Shared>::new_input(0, 111, 2));
        let ref_a = store.get(slot_a);
        for i in 0..(SEGMENT_SIZE as u32 + 10) {
            store.push(NodeData::<Shared>::new_input(0, 1000 + i, 2));
        }
        assert_eq!(ref_a.arena_slot(), 111);
    }
}
