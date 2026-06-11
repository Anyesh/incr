//! `GenericArena<T, C>`: typed value storage for every node of value
//! type `T`, parameterized over the [`Cells`] strategy.
//!
//! The arena is a growable vector of [`ValueSlot`]s. The vector itself
//! is guarded by the strategy lock (`C::Lock`): readers take the read
//! side to index, growth takes the write side. Under `Local` that is a
//! `RefCell` borrow (no atomics); under `Shared` it is an uncontended
//! `RwLock` read.
//!
//! Per-slot synchronization lives entirely in the slot type
//! (in-place cell under `Local`, hazard-protected pointer swap under
//! `Shared`); see the `value_slot` module for the soundness argument.
//! The arena therefore needs no unsafe code of its own.

use crate::cells::Cells;
use crate::locks::Lock;
use crate::value::Value;
use crate::value_slot::ValueSlot;

struct ArenaInner<T: Value, C: Cells> {
    slots: Vec<C::ValueSlot<T>>,
    free: Vec<u32>,
}

pub struct GenericArena<T: Value, C: Cells> {
    inner: C::Lock<ArenaInner<T, C>>,
}

impl<T: Value, C: Cells> Default for GenericArena<T, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Value, C: Cells> GenericArena<T, C> {
    pub fn new() -> Self {
        Self {
            inner: <C::Lock<ArenaInner<T, C>> as Lock<_>>::new(ArenaInner {
                slots: Vec::new(),
                free: Vec::new(),
            }),
        }
    }

    /// Allocate a slot initialized to `Some(initial)`, reusing a
    /// released slot when one is free. Growth holds the write side of
    /// the strategy lock, so it cannot race a reader mid-index.
    pub fn reserve_with(&self, initial: T) -> u32 {
        let mut inner = self.inner.write();
        if let Some(id) = inner.free.pop() {
            inner.slots[id as usize].write(initial);
            return id;
        }
        let id = inner.slots.len() as u32;
        inner
            .slots
            .push(<C::ValueSlot<T> as ValueSlot<T>>::new_with(initial));
        id
    }

    /// Allocate an uninitialized slot. Used by query nodes whose compute
    /// will populate the slot on first run.
    pub fn reserve(&self) -> u32 {
        let mut inner = self.inner.write();
        if let Some(id) = inner.free.pop() {
            // Released slots were cleared on release.
            return id;
        }
        let id = inner.slots.len() as u32;
        inner
            .slots
            .push(<C::ValueSlot<T> as ValueSlot<T>>::new_empty());
        id
    }

    /// Return a slot to the free list, dropping its value (deferred via
    /// hazard retire under Shared, so concurrent readers finish first).
    pub fn release(&self, slot: u32) {
        let mut inner = self.inner.write();
        inner.slots[slot as usize].clear();
        inner.free.push(slot);
    }

    /// Clone the value at `slot` out. Panics if the slot was never
    /// written (callers reach values only through the state machine,
    /// which guarantees the first compute completed).
    pub fn read(&self, slot: u32) -> T {
        self.inner.read().slots[slot as usize].read()
    }

    pub fn try_read(&self, slot: u32) -> Option<T> {
        self.inner.read().slots[slot as usize].try_read()
    }

    /// Publish a new value at `slot`. Under `Shared` this is an atomic
    /// pointer swap; concurrent readers finish their clone against the
    /// displaced value, which is hazard-protected until they do.
    pub fn write(&self, slot: u32, value: T) {
        self.inner.read().slots[slot as usize].write(value);
    }

    /// Compare the current value at `slot` against `v` without cloning.
    /// False for never-written slots.
    pub fn eq_current(&self, slot: u32, v: &T) -> bool {
        self.inner.read().slots[slot as usize].eq_current(v)
    }

    /// Write `value` unless the slot already holds an equal value.
    /// Returns true iff the value changed. One slot session for the
    /// recompute hot path's compare+publish.
    pub fn write_if_changed(&self, slot: u32, value: T) -> bool {
        self.inner.read().slots[slot as usize].write_if_changed(value)
    }

    pub fn len(&self) -> usize {
        self.inner.read().slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// SAFETY: the blanket impls exist because the registry bound
// (`ErasedArena<C>: Send + Sync`) and the compute-closure bound
// (`ComputeFn<C>: Send + Sync`) are uniform across strategies, and in a
// generic `C` context the compiler cannot resolve the per-strategy auto
// traits. The claim is sound per strategy:
// - Shared: vacuously true. `RwLock<Vec<SharedValueSlot<T>>>` is
//   genuinely Send + Sync for `T: Value` (the slot holds a
//   haphazard::AtomicPtr); the test below asserts it without these
//   impls being load-bearing.
// - Local: the marker is never exercised across threads. Every path to
//   a Local arena runs through `Runtime<Local>`, which is !Send + !Sync
//   by composition (RefCell dep stack, Cell-backed segment pointers,
//   RefCell inner lock), so no two threads can ever alias this arena.
//   Code outside the runtime must not export `Arc<GenericArena<T,
//   Local>>` to user-held types that are Send.
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
    fn eq_current_compares_without_cloning() {
        let a: GenericArena<String, Shared> = GenericArena::new();
        let s = a.reserve();
        assert!(!a.eq_current(s, &"x".to_string()));
        a.write(s, "x".to_string());
        assert!(a.eq_current(s, &"x".to_string()));
        assert!(!a.eq_current(s, &"y".to_string()));
    }

    #[test]
    fn shared_arena_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GenericArena<u64, Shared>>();
        assert_send_sync::<GenericArena<String, Shared>>();
        assert_send_sync::<GenericArena<Vec<u8>, Shared>>();
    }

    /// Concurrent set-style writes and reads on the same slot with a
    /// heap value. This is the exact shape Miri flagged as UB against
    /// the old in-place arena.
    #[test]
    fn shared_concurrent_write_read_same_slot() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let iters = if cfg!(miri) { 30 } else { 1000 };
        let a: Arc<GenericArena<String, Shared>> = Arc::new(GenericArena::new());
        let slot = a.reserve_with("s0000".to_string());
        let stop = Arc::new(AtomicBool::new(false));

        let readers: Vec<_> = (0..2)
            .map(|_| {
                let a = Arc::clone(&a);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let v = a.read(slot);
                        assert!(v.starts_with('s') && v.len() == 5, "torn value {:?}", v);
                    }
                })
            })
            .collect();

        for i in 0..iters {
            a.write(slot, format!("s{:04}", i % 10_000));
        }
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
    }
}
