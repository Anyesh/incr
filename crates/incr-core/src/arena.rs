//! Primitive arena for u64 values, parameterized over the [`Cells`]
//! strategy.
//!
//! The first incr-core slice ships a flat `Vec`-backed arena because the
//! question that needed answering ("does the strategy abstraction add
//! cost over direct access?") is answerable without segmenting. The
//! next slice lifts the production segmented store from incr-concurrent
//! and parameterizes it the same way. The Vec arena stays as the
//! reference implementation for cross-checking.
//!
//! Slot indexing is `u32` because (a) it matches `NodeData::arena_slot`
//! and (b) the production `MAX_NODES` cap is 1M which fits.

use crate::cells::Cells;

pub struct PrimitiveArena<C: Cells> {
    slots: Vec<C::U64>,
}

impl<C: Cells> Default for PrimitiveArena<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Cells> PrimitiveArena<C> {
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            slots: Vec::with_capacity(cap),
        }
    }

    /// Append a new slot initialized to `initial` and return its index.
    /// Caller must hold the write side of the runtime's lock (or be
    /// single-threaded under `Local`); concurrent appends are not safe.
    pub fn reserve(&mut self, initial: u64) -> u32 {
        let slot = self.slots.len() as u32;
        self.slots.push(C::new_u64(initial));
        slot
    }

    /// Read the value at `slot` with Acquire ordering on `Shared`. Caller
    /// must have established happens-before with the most recent writer
    /// through the node state machine.
    #[inline(always)]
    pub fn read(&self, slot: u32) -> u64 {
        C::u64_load_acquire(&self.slots[slot as usize])
    }

    /// Write `value` to `slot` with Release ordering on `Shared`. Caller
    /// must own exclusive access to the slot via the Computing state.
    #[inline(always)]
    pub fn write(&self, slot: u32, value: u64) {
        C::u64_store_release(&self.slots[slot as usize], value);
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    #[test]
    fn local_arena_roundtrip() {
        let mut a: PrimitiveArena<Local> = PrimitiveArena::new();
        let s = a.reserve(7);
        assert_eq!(a.read(s), 7);
        a.write(s, 11);
        assert_eq!(a.read(s), 11);
    }

    #[test]
    fn shared_arena_roundtrip() {
        let mut a: PrimitiveArena<Shared> = PrimitiveArena::new();
        let s = a.reserve(7);
        assert_eq!(a.read(s), 7);
        a.write(s, 11);
        assert_eq!(a.read(s), 11);
    }

    #[test]
    fn local_arena_grows() {
        let mut a: PrimitiveArena<Local> = PrimitiveArena::new();
        let slots: Vec<u32> = (0..100).map(|i| a.reserve(i as u64)).collect();
        assert_eq!(a.len(), 100);
        for (i, s) in slots.into_iter().enumerate() {
            assert_eq!(a.read(s), i as u64);
        }
    }
}
