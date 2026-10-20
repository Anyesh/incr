//! `GenericArena<T, C>`: typed value storage for every node of value
//! type `T`, parameterized over the [`Cells`] strategy.
//!
//! The slots live in never-moving segments of `SEGMENT_SIZE` behind a
//! [`Directory`], so reads, writes and compares index without a lock:
//! one Acquire load of the directory, a bounds compare, one Acquire load
//! of the segment and a null test. A segment is allocated on the first
//! `reserve` that reaches it, under the arena lock, which also guards the
//! free list. Reads never touch that lock. A type's first segment comes
//! with its first node: 1024 slots, so 16 KiB for `f64` under `Local`.
//!
//! Per-slot synchronization lives entirely in the slot type
//! (in-place cell under `Local`, hazard-protected pointer swap under
//! `Shared`); see the `value_slot` module for the soundness argument.
//! Under `Local` a slot refuses a write, a clear and a reserve from value
//! code, so a `Clone` or `PartialEq` running against a slot can never
//! see it written under it.

use std::cell::Cell;

use crate::cells::{Cells, PtrCell};
use crate::directory::Directory;
use crate::lock_order::LockKind;
use crate::locks::OrderedLock;
use crate::value::Value;
use crate::value_slot::{Compared, Displaced, ValueSlot};

// Four-slot segments under loom and Miri, so their tests cross segments
// and grow the directory without allocating thousands of values.
#[cfg(any(loom, miri))]
const SEGMENT_SHIFT: u32 = 2;
#[cfg(not(any(loom, miri)))]
const SEGMENT_SHIFT: u32 = 10;
const SEGMENT_SIZE: usize = 1 << SEGMENT_SHIFT;
const SEGMENT_MASK: u32 = (SEGMENT_SIZE as u32) - 1;
// One entry under loom, so a model doubles the directory within a few
// reserves.
#[cfg(loom)]
const FIRST_DIRECTORY_LEN: usize = 1;
#[cfg(not(loom))]
const FIRST_DIRECTORY_LEN: usize = 16;

type Segment<T, C> = [<C as Cells>::ValueSlot<T>; SEGMENT_SIZE];

struct ArenaInner {
    /// Slots handed out so far, free ones included.
    len: u32,
    free: Vec<u32>,
}

pub struct GenericArena<T: Value, C: Cells> {
    segments: Directory<Segment<T, C>, C>,
    inner: OrderedLock<ArenaInner, C>,
    /// What a pending handle of this type reads; set at most once.
    placeholder: std::sync::OnceLock<T>,
}

impl<T: Value, C: Cells> Default for GenericArena<T, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Value, C: Cells> GenericArena<T, C> {
    pub fn new() -> Self {
        Self {
            segments: Directory::with_len(FIRST_DIRECTORY_LEN),
            inner: OrderedLock::new(
                LockKind::Arena,
                ArenaInner {
                    len: 0,
                    free: Vec::new(),
                },
            ),
            placeholder: std::sync::OnceLock::new(),
        }
    }

    /// Stores the placeholder; false if one was set already.
    pub(crate) fn set_placeholder(&self, value: T) -> bool {
        self.placeholder.set(value).is_ok()
    }

    pub(crate) fn placeholder(&self) -> Option<&T> {
        self.placeholder.get()
    }

    /// Allocate a slot initialized to `Some(initial)`, reusing a
    /// released slot when one is free.
    pub fn reserve_with(&self, initial: T) -> u32 {
        let id = self.reserve();
        // A fresh or released slot is empty, so nothing is displaced and
        // no user code runs.
        let stale = self.slot_or_panic(id).write(initial);
        debug_assert!(stale.is_none(), "a reserved arena slot still held a value");
        id
    }

    /// Allocate an uninitialized slot. Used by query nodes whose compute
    /// will populate the slot on first run.
    pub fn reserve(&self) -> u32 {
        <C::ValueSlot<T> as ValueSlot<T>>::refuse_value_code();
        self.reserve_slot()
    }

    #[inline(always)]
    fn reserve_slot(&self) -> u32 {
        let mut inner = self.inner.write();
        match inner.free.pop() {
            // Released slots were cleared before they were freed.
            Some(id) => id,
            None => self.fresh_slot(&mut inner),
        }
    }

    #[cold]
    #[inline(never)]
    fn fresh_slot(&self, inner: &mut ArenaInner) -> u32 {
        let id = inner.len;
        if id & SEGMENT_MASK == 0 {
            let index = (id >> SEGMENT_SHIFT) as usize;
            self.segments.grow_to(index);
            let segment: Box<[C::ValueSlot<T>]> = (0..SEGMENT_SIZE)
                .map(|_| <C::ValueSlot<T> as ValueSlot<T>>::new_empty())
                .collect();
            let segment: Box<Segment<T, C>> = segment
                .try_into()
                .unwrap_or_else(|_| unreachable!("a segment has SEGMENT_SIZE slots"));
            self.segments
                .get(index)
                .expect("incr-core: grow_to left the segment's entry out")
                .store_release(Box::into_raw(segment));
        }
        inner.len = id
            .checked_add(1)
            .expect("incr-core: more than u32::MAX values of one type");
        id
    }

    /// Empty the slot and return it to the free list, handing back its
    /// value (a retire token under Shared, so concurrent readers finish
    /// first). The slot is emptied before the arena lock is taken and
    /// the value is handed back after it is released, so its `Drop` runs
    /// outside the lock.
    #[must_use]
    pub fn release(&self, slot: u32) -> Option<Displaced<T, C>> {
        let displaced = self.slot_or_panic(slot).clear();
        self.inner.write().free.push(slot);
        displaced
    }

    /// Empty the slot and hand back its value, keeping the slot reserved
    /// for the node that owns it.
    #[must_use]
    pub(crate) fn clear(&self, slot: u32) -> Option<Displaced<T, C>> {
        self.slot_or_panic(slot).clear()
    }

    /// The slot at `slot`, or `None` if no reserve ever reached it.
    #[inline(always)]
    fn slot(&self, slot: u32) -> Option<&C::ValueSlot<T>> {
        let segment = self
            .segments
            .get((slot >> SEGMENT_SHIFT) as usize)?
            .load_acquire();
        if segment.is_null() {
            return None;
        }
        // SAFETY: a non-null entry came from `Box::into_raw` in `reserve`
        // and segments are freed only when the arena drops, so the slot
        // outlives the borrow of `self`.
        Some(unsafe { &(*segment)[(slot & SEGMENT_MASK) as usize] })
    }

    #[inline(always)]
    fn slot_or_panic(&self, slot: u32) -> &C::ValueSlot<T> {
        match self.slot(slot) {
            Some(s) => s,
            None => slot_never_reserved(slot),
        }
    }

    /// Clone the value at `slot` out. Panics if the slot was never
    /// written (callers reach values only through the state machine,
    /// which guarantees the first compute completed).
    pub fn read(&self, slot: u32) -> T {
        self.slot_or_panic(slot).read()
    }

    /// Clone the value at `slot` out, or `None` if no reserve reached the
    /// slot or it was never written.
    #[inline(always)]
    pub fn try_read(&self, slot: u32) -> Option<T> {
        self.slot(slot)?.try_read()
    }

    /// Publish a new value at `slot` and hand back the displaced one.
    /// Under `Shared` this is an atomic pointer swap; concurrent readers
    /// finish their clone against the displaced value, which is
    /// hazard-protected until they do. The caller drops the displaced
    /// value after this returns, at a point its protocol allows.
    #[must_use]
    pub fn write(&self, slot: u32, value: T) -> Option<Displaced<T, C>> {
        self.slot_or_panic(slot).write(value)
    }

    /// Compare the current value at `slot` against `v` without cloning.
    /// False for never-written slots.
    pub fn eq_current(&self, slot: u32, v: &T) -> bool {
        self.slot_or_panic(slot).eq_current(v)
    }

    /// Write `value` unless the slot already holds an equal value, in
    /// one slot session; `on_write` runs between the comparison and the
    /// store (see [`ValueSlot::write_if_changed`]). The displaced value,
    /// or the rejected equal one, is handed back for the caller to drop
    /// where its protocol allows.
    #[must_use]
    pub fn compare_and_write(
        &self,
        slot: u32,
        value: T,
        on_write: impl FnOnce(),
    ) -> Compared<T, Displaced<T, C>> {
        self.slot_or_panic(slot).write_if_changed(value, on_write)
    }

    /// [`Self::compare_and_write`] for the recompute hot path. Returns
    /// true iff the value changed. The displaced value, or the rejected
    /// equal one, is dropped here: this is the compute wrapper's write,
    /// which drops before its landing by design, with `wrote` set so the
    /// landing stamps it.
    pub fn write_if_changed(&self, slot: u32, value: T, wrote: &Cell<u8>) -> bool {
        match self.compare_and_write(slot, value, || wrote.set(1)) {
            Compared::Equal(rejected) => {
                drop(rejected);
                false
            }
            Compared::Replaced(displaced) => {
                drop(displaced);
                true
            }
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cold]
#[inline(never)]
fn slot_never_reserved(slot: u32) -> ! {
    panic!(
        "incr-core: internal invariant violated: arena slot {} was never reserved",
        slot
    );
}

impl<T: Value, C: Cells> Drop for GenericArena<T, C> {
    fn drop(&mut self) {
        for index in 0..self.segments.len() {
            let Some(entry) = self.segments.get(index) else {
                break;
            };
            let segment = entry.load_relaxed();
            if !segment.is_null() {
                // SAFETY: came from `Box::into_raw` in `reserve`, and
                // `&mut self` rules out readers.
                drop(unsafe { Box::from_raw(segment) });
            }
        }
    }
}

// SAFETY: the blanket impls exist because the registry bound
// (`ErasedArena<C>: Send + Sync`) and the compute-closure bound
// (`ComputeFn<C>: Send + Sync`) are uniform across strategies, and in a
// generic `C` context the compiler cannot resolve the per-strategy auto
// traits. The claim is sound per strategy:
// - Shared: the segment directory is `AtomicPtr`s to boxed arrays of
//   `SharedValueSlot<T>` (each a haphazard::AtomicPtr) plus an `RwLock`
//   free list, all Send + Sync for `T: Value`, and every slot access is
//   synchronized by the slot itself.
// - Local: the marker is never exercised across threads. Every path to
//   a Local arena runs through `Runtime<Local>`, which is !Send + !Sync
//   by composition (RefCell dep stack, Cell-backed segment pointers,
//   RefCell inner lock), so no two threads can ever alias this arena.
//   Reads index the segments with no borrow flag, so two threads
//   aliasing a Local arena would race on its slots.
//   Code outside the runtime must not export `Arc<GenericArena<T,
//   Local>>` to user-held types that are Send.
unsafe impl<T: Value, C: Cells> Send for GenericArena<T, C> {}
unsafe impl<T: Value, C: Cells> Sync for GenericArena<T, C> {}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    #[test]
    fn local_roundtrip_string() {
        let a: GenericArena<String, Local> = GenericArena::new();
        let s = a.reserve_with("hello".to_string());
        assert_eq!(a.read(s), "hello");
        drop(a.write(s, "world".to_string()));
        assert_eq!(a.read(s), "world");
    }

    #[test]
    fn shared_roundtrip_string() {
        let a: GenericArena<String, Shared> = GenericArena::new();
        let s = a.reserve_with("hello".to_string());
        assert_eq!(a.read(s), "hello");
        drop(a.write(s, "world".to_string()));
        assert_eq!(a.read(s), "world");
    }

    #[test]
    fn local_uninitialized_try_read_is_none() {
        let a: GenericArena<u64, Local> = GenericArena::new();
        let s = a.reserve();
        assert_eq!(a.try_read(s), None);
        let _ = a.write(s, 42);
        assert_eq!(a.try_read(s), Some(42));
        assert_eq!(a.read(s), 42);
    }

    #[test]
    fn shared_uninitialized_try_read_is_none() {
        let a: GenericArena<u64, Shared> = GenericArena::new();
        let s = a.reserve();
        assert_eq!(a.try_read(s), None);
        let _ = a.write(s, 42);
        assert_eq!(a.try_read(s), Some(42));
        assert_eq!(a.read(s), 42);
    }

    #[test]
    fn eq_current_compares_without_cloning() {
        let a: GenericArena<String, Shared> = GenericArena::new();
        let s = a.reserve();
        assert!(!a.eq_current(s, &"x".to_string()));
        drop(a.write(s, "x".to_string()));
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
            drop(a.write(slot, format!("s{:04}", i % 10_000)));
        }
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
    }

    fn crosses_segments_and_reuses_released_slots<C: Cells>() {
        let a: GenericArena<String, C> = GenericArena::new();
        let count = SEGMENT_SIZE * FIRST_DIRECTORY_LEN + 3;
        let slots: Vec<u32> = (0..count).map(|i| a.reserve_with(i.to_string())).collect();
        assert_eq!(a.len(), count);
        for (i, &s) in slots.iter().enumerate() {
            assert_eq!(a.read(s), i.to_string());
        }
        assert_eq!(a.try_read(count as u32), None);
        assert_eq!(a.try_read(u32::MAX), None);
        let freed = slots[SEGMENT_SIZE + 1];
        assert!(a.release(freed).is_some());
        assert_eq!(a.try_read(freed), None);
        assert_eq!(a.reserve_with("again".to_string()), freed);
        assert_eq!(a.read(freed), "again");
        assert_eq!(a.len(), count);
    }

    #[test]
    fn local_crosses_segments_and_reuses_released_slots() {
        crosses_segments_and_reuses_released_slots::<Local>();
    }

    #[test]
    fn shared_crosses_segments_and_reuses_released_slots() {
        crosses_segments_and_reuses_released_slots::<Shared>();
    }

    /// Readers of early slots keep reading while a writer reserves far
    /// enough to allocate segments and double the directory.
    #[test]
    fn shared_reads_during_growth() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let a: Arc<GenericArena<String, Shared>> = Arc::new(GenericArena::new());
        let first = a.reserve_with("first".to_string());
        let stop = Arc::new(AtomicBool::new(false));
        let readers: Vec<_> = (0..2)
            .map(|_| {
                let a = Arc::clone(&a);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        assert_eq!(a.read(first), "first");
                    }
                })
            })
            .collect();
        let count = SEGMENT_SIZE * FIRST_DIRECTORY_LEN * if cfg!(miri) { 2 } else { 4 };
        for i in 0..count {
            let s = a.reserve_with(i.to_string());
            assert_eq!(a.read(s), i.to_string());
        }
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
    }

    /// Under Local, value code that reaches a slot write, clear or reserve
    /// panics instead of writing under a reference the caller holds.
    #[test]
    fn local_slot_writes_refuse_value_code() {
        use crate::value_code;
        let a: GenericArena<u32, Local> = GenericArena::new();
        let s = a.reserve_with(1);
        for (what, op) in [
            (
                "write",
                Box::new(|| {
                    let _ = a.write(s, 2);
                }) as Box<dyn Fn()>,
            ),
            (
                "write_if_changed",
                Box::new(|| {
                    a.write_if_changed(s, 2, &Cell::new(0));
                }),
            ),
            (
                "release",
                Box::new(|| {
                    let _ = a.release(s);
                }),
            ),
            (
                "reserve",
                Box::new(|| {
                    a.reserve();
                }),
            ),
            (
                "reserve_with",
                Box::new(|| {
                    a.reserve_with(3);
                }),
            ),
        ] {
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                value_code::with_value_code(&op)
            }))
            .expect_err(what);
            let message = err
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| err.downcast_ref::<&str>().map(|m| m.to_string()))
                .unwrap_or_default();
            assert!(
                message.contains(value_code::CONTRACT_MESSAGE),
                "{what}: {message}"
            );
        }
        assert_eq!(a.read(s), 1);
        assert_eq!(a.len(), 1);
    }
}
