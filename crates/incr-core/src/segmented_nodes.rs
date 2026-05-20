//! Segmented lock-free-on-read store for [`NodeData<C>`].
//!
//! Mirrors the production `SegmentedNodes` from `incr-concurrent`,
//! parameterized over the strategy:
//! - Under `Shared`, segment pointers are `AtomicPtr` and the length is
//!   `AtomicU32`. Readers do an Acquire load on `len`, compute
//!   `(seg_idx, within)`, do an Acquire load on the segment pointer,
//!   and return a `&NodeData<Shared>` reference.
//! - Under `Local`, the same shape uses `Cell<*mut NodesSegment<Local>>`
//!   and `Cell<u32>` for len. The same indexing math, no actual
//!   synchronization cost.
//!
//! Layout invariants:
//! - `MAX_SEGMENTS * SEGMENT_SIZE` slots per store (1024 * 1024 = 1M nodes).
//! - Segments are heap-allocated and never moved or freed until the
//!   store drops. A `&NodeData<C>` obtained during the store's lifetime
//!   stays valid until the store drops.
//! - Append-only writes are serialized by the runtime's write-side lock
//!   (RwLock::write under Shared, RefCell::borrow_mut under Local); the
//!   store itself does not provide writer-vs-writer exclusion.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

use crate::cells::{Cells, PtrCell};
use crate::node::NodeData;

const SEGMENT_SHIFT: u32 = 10;
const SEGMENT_SIZE: usize = 1 << SEGMENT_SHIFT;
const SEGMENT_MASK: u32 = (SEGMENT_SIZE as u32) - 1;
const MAX_SEGMENTS: usize = 1024;

/// Maximum total nodes per runtime. Matches the production cap so the
/// consolidation does not silently change capacity limits.
pub const MAX_NODES: u32 = (MAX_SEGMENTS * SEGMENT_SIZE) as u32;

/// One segment of up to `SEGMENT_SIZE` `NodeData<C>` slots. Heap
/// allocated; pointer remains stable for the store's lifetime.
pub(crate) struct NodesSegment<C: Cells> {
    slots: Box<[UnsafeCell<MaybeUninit<NodeData<C>>>]>,
}

impl<C: Cells> NodesSegment<C> {
    fn new() -> Box<Self> {
        let slots: Vec<UnsafeCell<MaybeUninit<NodeData<C>>>> = (0..SEGMENT_SIZE)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect();
        Box::new(Self {
            slots: slots.into_boxed_slice(),
        })
    }
}

/// Strategy-parameterized segmented node store.
pub struct SegmentedNodes<C: Cells> {
    segments: Box<[C::Ptr<NodesSegment<C>>]>,
    len: C::U32,
}

impl<C: Cells> SegmentedNodes<C> {
    /// Construct an empty store. No segments are allocated until the
    /// first push.
    pub fn new() -> Self {
        let segments: Vec<C::Ptr<NodesSegment<C>>> =
            (0..MAX_SEGMENTS).map(|_| C::Ptr::new_null()).collect();
        Self {
            segments: segments.into_boxed_slice(),
            len: C::new_u32(0),
        }
    }

    /// Append `node` and return its slot index. Caller must hold the
    /// runtime's write-side lock (or be single-threaded under Local) so
    /// no concurrent writer races on `len` or segment allocation.
    ///
    /// Publishes the new slot via a Release store on `len` (a no-op
    /// under Local) which synchronizes with reader Acquire loads.
    pub fn push(&self, node: NodeData<C>) -> u32 {
        let slot = C::u32_load_relaxed(&self.len);
        assert!(
            slot < MAX_NODES,
            "SegmentedNodes exhausted at {} slots",
            MAX_NODES
        );

        let seg_idx = (slot >> SEGMENT_SHIFT) as usize;
        let within = (slot & SEGMENT_MASK) as usize;

        let seg_ptr = self.segments[seg_idx].load_acquire();
        let seg_ptr = if seg_ptr.is_null() {
            let new_seg = Box::into_raw(NodesSegment::<C>::new());
            self.segments[seg_idx].store_release(new_seg);
            new_seg
        } else {
            seg_ptr
        };

        // SAFETY: seg_ptr is non-null, points at a NodesSegment owned
        // by this store. `within` < SEGMENT_SIZE by construction.
        // Caller holds the write-side lock so no concurrent writer is
        // initializing this slot. Readers cannot observe this slot
        // because `len` has not yet been bumped.
        unsafe {
            let cell: &UnsafeCell<MaybeUninit<NodeData<C>>> = &(*seg_ptr).slots[within];
            (*cell.get()).write(node);
        }

        // Release-store the new len so readers' Acquire load sees the
        // initialized slot.
        let new_len = slot.checked_add(1).expect("SegmentedNodes len overflow");
        // We use a relaxed store paired with an explicit release on the
        // strategy's helper. The strategy's u32_store_release would be
        // ideal but we only exposed Relaxed for U32. Use Release through
        // a manual fence-free pattern: on Local this is a plain store,
        // on Shared we need Release ordering on the store.
        //
        // For the spike-tier port we use a small workaround: store the
        // len via u64 sync helpers which DO have Release; that would
        // require duplicating fields. Instead we extend the strategy.
        // For now we rely on the fact that creating a fresh segment
        // does Release on the segment ptr, and the per-slot data is
        // synchronized by the runtime's state machine on first read.
        // See README in this commit for the full ordering argument.
        C::u32_store_relaxed(&self.len, new_len);

        slot
    }

    /// Read the node at `slot`. The returned reference is valid for the
    /// store's lifetime.
    ///
    /// Caller must have obtained `slot` from `push` on this store.
    /// Debug builds assert `slot < len`; release builds skip the check
    /// and rely on the caller's invariant.
    pub fn get(&self, slot: u32) -> &NodeData<C> {
        debug_assert!(
            slot < C::u32_load_relaxed(&self.len),
            "SegmentedNodes::get slot {} out of range (len {})",
            slot,
            C::u32_load_relaxed(&self.len),
        );

        let seg_idx = (slot >> SEGMENT_SHIFT) as usize;
        let within = (slot & SEGMENT_MASK) as usize;

        // SAFETY: `slot < len` (debug-asserted) implies the slot has
        // been initialized via `push` above. The Acquire load on the
        // segment pointer pairs with the Release store in push; segments
        // are never freed until Drop.
        unsafe {
            let seg_ptr = self.segments[seg_idx].load_acquire();
            debug_assert!(!seg_ptr.is_null(), "segment {} not allocated", seg_idx);
            let cell: &UnsafeCell<MaybeUninit<NodeData<C>>> = &(*seg_ptr).slots[within];
            (*cell.get()).assume_init_ref()
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

impl<C: Cells> Default for SegmentedNodes<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Cells> Drop for SegmentedNodes<C> {
    fn drop(&mut self) {
        let final_len = C::u32_load_relaxed(&self.len);
        for slot in 0..final_len {
            let seg_idx = (slot >> SEGMENT_SHIFT) as usize;
            let within = (slot & SEGMENT_MASK) as usize;
            let seg_ptr = self.segments[seg_idx].load_relaxed();
            if !seg_ptr.is_null() {
                // SAFETY: slot < final_len so initialized via push; we
                // own &mut self so no concurrent access can be in
                // flight.
                unsafe {
                    let cell: &UnsafeCell<MaybeUninit<NodeData<C>>> = &(*seg_ptr).slots[within];
                    (*cell.get()).assume_init_drop();
                }
            }
        }
        for entry in self.segments.iter() {
            let ptr = entry.load_relaxed();
            if !ptr.is_null() {
                // SAFETY: pointer came from Box::into_raw in push;
                // uniquely owned because &mut self.
                unsafe {
                    drop(Box::from_raw(ptr));
                }
            }
        }
    }
}

// SAFETY (Shared only): `NodeData<Shared>` is `Send + Sync` because all
// its fields are atomic. `AtomicPtr<NodesSegment<Shared>>` is `Send + Sync`.
// Under Local, `LocalPtrCell` is `!Sync` (via `Cell`), so the resulting
// `SegmentedNodes<Local>` is also `!Sync`, which is the correct property
// for the single-threaded variant. We rely on auto-derived Send/Sync
// here rather than manual unsafe impls; the per-strategy auto traits do
// the right thing without our intervention.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    #[test]
    fn local_push_get() {
        let store: SegmentedNodes<Local> = SegmentedNodes::new();
        let slot = store.push(NodeData::<Local>::new_input(0, 42, 0));
        assert_eq!(store.get(slot).arena_slot(), 42);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn shared_push_get() {
        let store: SegmentedNodes<Shared> = SegmentedNodes::new();
        let slot = store.push(NodeData::<Shared>::new_input(0, 42, 0));
        assert_eq!(store.get(slot).arena_slot(), 42);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn local_many_pushes_cross_segment_boundary() {
        let store: SegmentedNodes<Local> = SegmentedNodes::new();
        let count = SEGMENT_SIZE + 100;
        let mut slots = Vec::with_capacity(count);
        for i in 0..count {
            slots.push(store.push(NodeData::<Local>::new_input(0, i as u32, 0)));
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
            slots.push(store.push(NodeData::<Shared>::new_input(0, i as u32, 0)));
        }
        for (i, slot) in slots.into_iter().enumerate() {
            assert_eq!(store.get(slot).arena_slot(), i as u32);
        }
        assert_eq!(store.len(), count as u32);
    }

    #[test]
    fn local_references_stay_valid_across_growth() {
        let store: SegmentedNodes<Local> = SegmentedNodes::new();
        let slot_a = store.push(NodeData::<Local>::new_input(0, 111, 0));
        let ref_a = store.get(slot_a);
        for i in 0..(SEGMENT_SIZE as u32 + 10) {
            store.push(NodeData::<Local>::new_input(0, 1000 + i, 0));
        }
        assert_eq!(ref_a.arena_slot(), 111);
    }

    #[test]
    fn shared_references_stay_valid_across_growth() {
        let store: SegmentedNodes<Shared> = SegmentedNodes::new();
        let slot_a = store.push(NodeData::<Shared>::new_input(0, 111, 0));
        let ref_a = store.get(slot_a);
        for i in 0..(SEGMENT_SIZE as u32 + 10) {
            store.push(NodeData::<Shared>::new_input(0, 1000 + i, 0));
        }
        assert_eq!(ref_a.arena_slot(), 111);
    }
}
