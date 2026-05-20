//! `GenericArena<T, C>`: typed value storage parameterized over both the
//! value type and the [`Cells`] strategy.
//!
//! Slot layout: `UnsafeCell<Option<T>>`. The `Option` allows two states:
//! - `None`: slot reserved but never written (e.g., a query node whose
//!   compute hasn't run yet).
//! - `Some(value)`: slot holds the current value.
//!
//! Exclusive access to a slot is gated by the node state machine, NOT by
//! Rust's borrow checker: the slot's `Computing` state is held by
//! exactly one thread (CAS-claimed on Shared, single-threaded on Local).
//! Readers reach a slot only when the corresponding node is `Clean`, so
//! they observe the writer's data through the Acquire load on state.
//!
//! Reads clone `T` rather than returning a reference because the runtime
//! may need to drop the slot (or recompute through it) after the read
//! returns; tying a reference's lifetime to the read call would prevent
//! that. Clone cost is part of the user's `T` impl.
//!
//! The segmented production primitive arenas (`AtomicPrimitiveArena<T>`
//! for u64/f64/etc.) are deferred. Primitives go through the generic
//! arena for now; the specialization that gives 5-10 ns per-get on
//! primitives lands in a follow-up commit once the rest of the engine
//! is in place.

use crate::cells::Cells;
use crate::value::Value;
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::sync::RwLock;

/// Typed arena for `T` values, parameterized over the strategy.
///
/// Under `Shared`, the slots vector is behind an `RwLock` (the runtime's
/// write-side lock guards all arena growth). Under `Local`, the same
/// RwLock is morally a `RefCell`; we use `RwLock` uniformly for the
/// first cut to avoid duplicating arena code per strategy. The cost on
/// Local is one uncontended lock acquire per arena op, which is
/// significant on the hot path. The follow-up commit replaces this with
/// a `C`-parameterized inner-lock primitive (`Cells::RwLock<Vec<...>>`)
/// to remove the cost on Local.
pub struct GenericArena<T: Value, C: Cells> {
    slots: RwLock<Vec<Box<UnsafeCell<Option<T>>>>>,
    _phantom: PhantomData<C>,
}

impl<T: Value, C: Cells> Default for GenericArena<T, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Value, C: Cells> GenericArena<T, C> {
    pub fn new() -> Self {
        Self {
            slots: RwLock::new(Vec::new()),
            _phantom: PhantomData,
        }
    }

    /// Append a new slot initialized to `Some(initial)`. Caller holds
    /// the runtime's write lock.
    pub fn reserve_with(&self, initial: T) -> u32 {
        let mut slots = self.slots.write().expect("arena slots lock poisoned");
        let id = slots.len() as u32;
        slots.push(Box::new(UnsafeCell::new(Some(initial))));
        id
    }

    /// Append an uninitialized slot (`None`). Used by query nodes whose
    /// compute will populate the slot on first run.
    pub fn reserve(&self) -> u32 {
        let mut slots = self.slots.write().expect("arena slots lock poisoned");
        let id = slots.len() as u32;
        slots.push(Box::new(UnsafeCell::new(None)));
        id
    }

    /// Read the value at `slot`. Panics if the slot is `None` (caller
    /// should use [`try_read`](Self::try_read) if they need to handle
    /// uninitialized slots).
    pub fn read(&self, slot: u32) -> T {
        let slots = self.slots.read().expect("arena slots lock poisoned");
        let cell = &slots[slot as usize];
        // SAFETY: exclusive access to this slot is governed by the
        // node state machine: a reader only reaches here when the
        // node is Clean (Acquire-synchronized with the writer's
        // Release store on state). No mutable alias is in flight.
        unsafe {
            (*cell.get())
                .as_ref()
                .expect("GenericArena::read on uninitialized slot")
                .clone()
        }
    }

    pub fn try_read(&self, slot: u32) -> Option<T> {
        let slots = self.slots.read().expect("arena slots lock poisoned");
        let cell = &slots[slot as usize];
        unsafe { (*cell.get()).as_ref().cloned() }
    }

    /// Overwrite the value at `slot`. Caller must own exclusive access
    /// via the Computing state.
    pub fn write(&self, slot: u32, value: T) {
        let slots = self.slots.read().expect("arena slots lock poisoned");
        let cell = &slots[slot as usize];
        unsafe {
            *cell.get() = Some(value);
        }
    }

    pub fn len(&self) -> usize {
        self.slots.read().expect("arena slots lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// SAFETY: `T: Send + Sync` (from Value bound), `Box<UnsafeCell<Option<T>>>`
// is Send when `T: Send`. Sync is the question: UnsafeCell is !Sync, but
// access to the cell is governed by the runtime's state machine (which
// provides exclusive access via the Computing CAS) and the RwLock around
// the vector (which prevents concurrent push during reads). The
// combination is sound when used as documented; we assert Send + Sync
// manually because UnsafeCell blocks the auto-derive.
unsafe impl<T: Value, C: Cells> Send for GenericArena<T, C> {}
unsafe impl<T: Value, C: Cells> Sync for GenericArena<T, C> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    #[test]
    fn local_roundtrip_string() {
        let a: GenericArena<String, Local> = GenericArena::new();
        let s = a.reserve_with("hello".to_string());
        assert_eq!(a.read(s), "hello");
        a.write(s, "world".to_string());
        assert_eq!(a.read(s), "world");
    }

    #[test]
    fn shared_roundtrip_string() {
        let a: GenericArena<String, Shared> = GenericArena::new();
        let s = a.reserve_with("hello".to_string());
        assert_eq!(a.read(s), "hello");
        a.write(s, "world".to_string());
        assert_eq!(a.read(s), "world");
    }

    #[test]
    fn local_uninitialized_try_read_is_none() {
        let a: GenericArena<u64, Local> = GenericArena::new();
        let s = a.reserve();
        assert_eq!(a.try_read(s), None);
        a.write(s, 42);
        assert_eq!(a.try_read(s), Some(42));
        assert_eq!(a.read(s), 42);
    }

    #[test]
    fn shared_uninitialized_try_read_is_none() {
        let a: GenericArena<u64, Shared> = GenericArena::new();
        let s = a.reserve();
        assert_eq!(a.try_read(s), None);
        a.write(s, 42);
        assert_eq!(a.try_read(s), Some(42));
        assert_eq!(a.read(s), 42);
    }

    #[test]
    fn shared_arena_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GenericArena<u64, Shared>>();
        assert_send_sync::<GenericArena<String, Shared>>();
        assert_send_sync::<GenericArena<Vec<u8>, Shared>>();
    }
}
