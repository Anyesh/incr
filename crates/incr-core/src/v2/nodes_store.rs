//! Segmented lock-free storage for `NodeData`.
//!
//! Replaces the commit F scaffolding's `RwLock<Vec<Box<NodeData>>>`
//! with a segmented store modeled on the arenas from commits A and B.
//! Reads are lock-free: a handler computes `(seg_idx, within_idx)` from
//! the slot, does an Acquire load on the segment pointer, and returns
//! a direct `&NodeData` reference. No lock acquire, no allocation, no
//! indirection beyond the two atomic loads.
//!
//! Writes (append-only, via `create_input` / `create_query`) run under
//! the Runtime's `write_mutex`, so there is no concurrent writer race
//! on the `len` counter or on segment allocation.
//!
//! ## Layout
//!
//! - `segments`: fixed-size array of `AtomicPtr<NodesSegment>`. Size
//!   is `MAX_SEGMENTS`. Entries start null; non-null entries point at
//!   a heap-allocated `NodesSegment`. Lazy allocation.
//! - `len`: `AtomicU32`. Number of initialized slots. Monotonically
//!   increasing via write-mutex-guarded increments. Readers
//!   Acquire-load to check that their slot is valid.
//!
//! - `NodesSegment::slots`: boxed fixed-size slice of
//!   `UnsafeCell<MaybeUninit<NodeData>>`. Slots are uninitialized at
//!   segment creation; the first write to a slot initializes it via
//!   `MaybeUninit::write`. Slots at or beyond `len` are still
//!   uninitialized and must not be dereferenced.
//!
//! ## Safety invariants
//!
//! 1. Only the write-mutex-holding thread can mutate `len` and
//!    initialize slots.
//! 2. A slot at index `i` is initialized iff `i < len`. The ordering
//!    is: (a) initialize the slot, (b) Release-store the new `len`.
//!    Readers Acquire-load `len` and then dereference the slot; the
//!    Acquire pairs with the Release so the reader sees the
//!    initialized slot.
//! 3. Segments are never deallocated until the store is dropped, so
//!    a `&NodeData` obtained from a slot remains valid for the
//!    store's lifetime.
//! 4. On Drop, the store iterates slots `0..len` and drops them
//!    in place via `assume_init_drop`. Slots beyond `len` are
//!    untouched (they were never initialized).

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

use super::node::NodeData;

const SEGMENT_SHIFT: u32 = 10;
const SEGMENT_SIZE: usize = 1 << SEGMENT_SHIFT;
const SEGMENT_MASK: u32 = (SEGMENT_SIZE as u32) - 1;
const MAX_SEGMENTS: usize = 1024;

/// Total node capacity per runtime. Matches the arena capacity so a
/// single Runtime can hold at most the same number of nodes as its
/// arenas can hold values.
pub(crate) const MAX_NODES: u32 = (MAX_SEGMENTS * SEGMENT_SIZE) as u32;

/// One segment of up to `SEGMENT_SIZE` `NodeData` slots. Heap
/// allocated and never moved; a `*const NodesSegment` obtained
/// during the store's lifetime stays valid until the store drops.
struct NodesSegment {
    slots: Box<[UnsafeCell<MaybeUninit<NodeData>>]>,
}

impl NodesSegment {
    fn new() -> Box<Self> {
        let slots: Vec<UnsafeCell<MaybeUninit<NodeData>>> = (0..SEGMENT_SIZE)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect();
        Box::new(Self {
            slots: slots.into_boxed_slice(),
        })
    }
}

/// Segmented lock-free store for `NodeData`. Owned by `Runtime`.
pub(crate) struct SegmentedNodes {
    segments: Box<[AtomicPtr<NodesSegment>]>,
    len: AtomicU32,
}

impl SegmentedNodes {
    /// Construct an empty store. No segments are allocated until the
    /// first `push`.
    pub(crate) fn new() -> Self {
        let segments: Vec<AtomicPtr<NodesSegment>> = (0..MAX_SEGMENTS)
            .map(|_| AtomicPtr::new(std::ptr::null_mut()))
            .collect();
        Self {
            segments: segments.into_boxed_slice(),
            len: AtomicU32::new(0),
        }
    }

    /// Append a new `NodeData` to the store, returning its slot
    /// index. Caller must hold the Runtime's `write_mutex` so no
    /// other writer is racing on `len` or segment allocation.
    ///
    /// Publishes the new slot via a Release store on `len`, which
    /// synchronizes with the reader's Acquire load in `get`.
    pub(crate) fn push(&self, node: NodeData) -> u32 {
        let slot = self.len.load(Ordering::Relaxed);
        if slot >= MAX_NODES {
            panic!("SegmentedNodes exhausted at {} slots", MAX_NODES);
        }

        let seg_idx = (slot >> SEGMENT_SHIFT) as usize;
        let within = (slot & SEGMENT_MASK) as usize;

        // Ensure the target segment exists. Under write_mutex this
        // is race-free; we just check and allocate if null.
        let seg_ptr = self.segments[seg_idx].load(Ordering::Acquire);
        let seg_ptr = if seg_ptr.is_null() {
            let new_seg = Box::into_raw(NodesSegment::new());
            // Release-store so a concurrent reader's Acquire load
            // observes the fresh segment with all slots still
            // uninitialized (none are in the caller's reachable
            // range until `len` is bumped below).
            self.segments[seg_idx].store(new_seg, Ordering::Release);
            new_seg
        } else {
            seg_ptr
        };

        // Initialize the slot. SAFETY: `seg_ptr` is non-null and
        // points at a NodesSegment owned by this store. `within` is
        // in-range because `slot < MAX_NODES` and the segment has
        // `SEGMENT_SIZE` slots. The caller holds write_mutex so no
        // other thread is initializing this or nearby slots.
        // Readers cannot observe this slot yet because `len` has
        // not been bumped.
        unsafe {
            let cell: &UnsafeCell<MaybeUninit<NodeData>> = &(*seg_ptr).slots[within];
            (*cell.get()).write(node);
        }

        // Release-store the new len. Synchronizes with reader Acquire
        // loads in `get` to publish the initialized slot.
        self.len.store(slot + 1, Ordering::Release);
        slot
    }

    /// Read the node at `slot`. Returns a reference valid for the
    /// store's lifetime (tied to `&self`).
    ///
    /// Caller must have obtained `slot` from a handle returned by
    /// `push` on the same store (so `slot < len`). Debug builds
    /// assert this; release builds dereference unchecked and will
    /// hit undefined behavior for out-of-range slots.
    pub(crate) fn get(&self, slot: u32) -> &NodeData {
        debug_assert!(
            slot < self.len.load(Ordering::Acquire),
            "SegmentedNodes::get slot {} out of range (len {})",
            slot,
            self.len.load(Ordering::Acquire)
        );

        let seg_idx = (slot >> SEGMENT_SHIFT) as usize;
        let within = (slot & SEGMENT_MASK) as usize;

        // SAFETY: `slot < len` (debug asserted above) implies the
        // slot has been initialized and the segment has been
        // allocated. The Acquire load pairs with the Release store
        // in `push` to establish happens-before with the
        // initialization. Segments are never freed until `Drop`, so
        // the returned reference is valid for `&self`'s lifetime.
        unsafe {
            let seg_ptr = self.segments[seg_idx].load(Ordering::Acquire);
            debug_assert!(!seg_ptr.is_null(), "segment {} not allocated", seg_idx);
            let cell: &UnsafeCell<MaybeUninit<NodeData>> = &(*seg_ptr).slots[within];
            (*cell.get()).assume_init_ref()
        }
    }

    /// Current number of initialized slots. Used by tests and by
    /// debug assertions elsewhere in the runtime.
    pub(crate) fn len(&self) -> u32 {
        self.len.load(Ordering::Acquire)
    }
}

impl Drop for SegmentedNodes {
    fn drop(&mut self) {
        // Drop every initialized slot in place, then drop the
        // segments themselves. Slots beyond `len` are still
        // uninitialized and must not be dropped.
        let final_len = *self.len.get_mut();
        for slot in 0..final_len {
            let seg_idx = (slot >> SEGMENT_SHIFT) as usize;
            let within = (slot & SEGMENT_MASK) as usize;
            let seg_ptr = *self.segments[seg_idx].get_mut();
            if !seg_ptr.is_null() {
                // SAFETY: `slot < final_len` so this slot was
                // initialized via `MaybeUninit::write` in `push`.
                // `&mut self` guarantees no concurrent access.
                unsafe {
                    let cell: &UnsafeCell<MaybeUninit<NodeData>> = &(*seg_ptr).slots[within];
                    (*cell.get()).assume_init_drop();
                }
            }
        }

        // Reclaim the segment boxes themselves.
        for entry in self.segments.iter_mut() {
            let ptr = *entry.get_mut();
            if !ptr.is_null() {
                // SAFETY: pointer came from `Box::into_raw` in
                // `push`; uniquely owned at this point because
                // `&mut self`.
                unsafe {
                    drop(Box::from_raw(ptr));
                }
            }
        }
    }
}

// SAFETY: `SegmentedNodes` holds `NodeData` values which are themselves
// `Send + Sync` (atomics). The raw pointers in `segments` point at
// `NodesSegment`s which are heap-allocated and owned by this store.
// Concurrent access is coordinated by the Runtime's write_mutex for
// writers and by the `len` Release/Acquire pair for reader visibility.
unsafe impl Send for SegmentedNodes {}
unsafe impl Sync for SegmentedNodes {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::node::NodeData;

    #[test]
    fn push_then_get_returns_stable_reference() {
        let store = SegmentedNodes::new();
        let slot = store.push(NodeData::new_input(0, 42, 0));
        let node = store.get(slot);
        assert_eq!(node.arena_slot(), 42);
    }

    #[test]
    fn many_pushes_cross_segment_boundary() {
        let store = SegmentedNodes::new();
        let count = SEGMENT_SIZE + 100;
        let mut slots = Vec::with_capacity(count);
        for i in 0..count {
            slots.push(store.push(NodeData::new_input(0, i as u32, 0)));
        }
        for (i, slot) in slots.into_iter().enumerate() {
            assert_eq!(store.get(slot).arena_slot(), i as u32);
        }
        assert_eq!(store.len(), count as u32);
    }

    #[test]
    fn drop_frees_overflow_deps_from_every_node() {
        use crate::v2::node::NodeId;
        // Push nodes, give some of them overflow deps, drop the
        // store, and rely on miri / valgrind to confirm no leaks.
        let store = SegmentedNodes::new();
        for i in 0..20 {
            let slot = store.push(NodeData::new_query(0, i));
            let deps: Vec<NodeId> = (0..(i + 5)).map(NodeId).collect();
            store.get(slot).publish_initial_deps(&deps);
        }
        drop(store);
    }

    #[test]
    fn references_from_get_remain_valid_across_more_pushes() {
        let store = SegmentedNodes::new();
        let slot_a = store.push(NodeData::new_input(0, 111, 0));
        let ref_a = store.get(slot_a);
        // Trigger segment growth by pushing enough nodes to cross
        // a segment boundary. ref_a must still be valid.
        for i in 0..(SEGMENT_SIZE as u32 + 10) {
            store.push(NodeData::new_input(0, 1000 + i, 0));
        }
        assert_eq!(ref_a.arena_slot(), 111);
    }
}
