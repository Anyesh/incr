//! `Loom`: a third `Cells` strategy, compiled only under `cfg(loom)`, that
//! runs the real runtime protocol on loom's modeled atomics and locks so
//! `loom::model` can explore its interleavings.
//!
//! Orderings mirror [`Shared`](crate::cells::Shared) exactly; a model is
//! only evidence about `Shared` if every ordering here matches it. A
//! value slot is a loom `AtomicPtr` with `SharedValueSlot`'s orderings but
//! no hazard pointers, whose SeqCst fence on every protect would multiply
//! the explored state space; the boxes it displaces stay allocated until
//! the slot drops, in place of the hazard-deferred retire. Overflow dep
//! lists are loom `RwLock`s, because the models keep every dep list
//! inline.
//!
//! Every value of a loom type has to be created inside `loom::model`, so
//! no non-loom test can run in a `--cfg loom` build. The models are
//! named `loom_*` and CI filters on that prefix.

use crate::cells::{Cells, PtrCell};
use crate::dep_stack::LoomDepStack;
use crate::locks::Lock;
use crate::node::{DepList, DepListCell};
use crate::table::{Array, ArrayCell, DataCell, TableKey};
use crate::value::Value;
use crate::value_code;
use crate::value_slot::{Compared, ValueSlot};
use crate::walk_ring::WalkRing;
use loom::sync::atomic::{AtomicPtr, AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering};
use loom::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub struct Loom;

/// Newtype so the foreign loom `RwLock` can implement the crate's `Lock`.
pub struct LoomLock<T>(RwLock<T>);

impl<T: 'static> Lock<T> for LoomLock<T> {
    type ReadGuard<'a>
        = RwLockReadGuard<'a, T>
    where
        T: 'a;
    type WriteGuard<'a>
        = RwLockWriteGuard<'a, T>
    where
        T: 'a;

    fn new(val: T) -> Self {
        Self(RwLock::new(val))
    }

    // Poison is ignored for the same reason as the `Shared` impl in
    // `locks.rs`: a panicking user closure must not brick the runtime.
    fn read(&self) -> Self::ReadGuard<'_> {
        self.0
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn write(&self) -> Self::WriteGuard<'_> {
        self.0
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

pub struct LoomPtrCell<T>(AtomicPtr<T>);

impl<T: 'static> PtrCell<T> for LoomPtrCell<T> {
    fn new_null() -> Self {
        Self(AtomicPtr::new(std::ptr::null_mut()))
    }
    fn new(p: *mut T) -> Self {
        Self(AtomicPtr::new(p))
    }
    fn load_acquire(&self) -> *mut T {
        self.0.load(Ordering::Acquire)
    }
    fn store_release(&self, p: *mut T) {
        self.0.store(p, Ordering::Release);
    }
    fn load_relaxed(&self) -> *mut T {
        self.0.load(Ordering::Relaxed)
    }
    fn store_relaxed(&self, p: *mut T) {
        self.0.store(p, Ordering::Relaxed);
    }
}

/// Null means never written. Reads load with Acquire, the ordering of
/// the validating load in a hazard protect, and writes swap with Release,
/// as `haphazard::AtomicPtr::swap` does.
pub struct LoomValueSlot<T> {
    ptr: AtomicPtr<T>,
    /// Displaced boxes, freed with the slot so that a reader still cloning
    /// one never sees it freed. A std lock, because a loom one would add
    /// branch points; loom runs one modeled thread at a time and nothing
    /// under this lock yields to it.
    parked: std::sync::Mutex<Vec<Box<T>>>,
}

impl<T: Value> LoomValueSlot<T> {
    fn from_raw(ptr: *mut T) -> Self {
        Self {
            ptr: AtomicPtr::new(ptr),
            parked: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn current(&self) -> Option<&T> {
        // SAFETY: every non-null pointer in this slot came from
        // Box::into_raw, and a displaced one is parked until the slot
        // drops, so the reference stays valid for the borrow of `self`.
        unsafe { self.ptr.load(Ordering::Acquire).as_ref() }
    }

    fn swap(&self, ptr: *mut T) -> Option<()> {
        let old = self.ptr.swap(ptr, Ordering::Release);
        if old.is_null() {
            return None;
        }
        // SAFETY: `old` came from Box::into_raw and the swap took it out
        // of the slot, so this is its only owner.
        let old = unsafe { Box::from_raw(old) };
        self.parked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(old);
        Some(())
    }
}

impl<T: Value> ValueSlot<T> for LoomValueSlot<T> {
    type Displaced = ();
    fn new_empty() -> Self {
        Self::from_raw(std::ptr::null_mut())
    }
    fn new_with(v: T) -> Self {
        Self::from_raw(Box::into_raw(Box::new(v)))
    }
    fn read(&self) -> T {
        self.current()
            .expect("ValueSlot::read on a slot that was never written")
            .clone()
    }
    fn try_read(&self) -> Option<T> {
        self.current().cloned()
    }
    fn write(&self, v: T) -> Option<()> {
        self.swap(Box::into_raw(Box::new(v)))
    }
    fn eq_current(&self, v: &T) -> bool {
        self.current() == Some(v)
    }
    fn write_if_changed(&self, v: T, on_write: impl FnOnce()) -> Compared<T, ()> {
        if value_code::run(|| self.current() == Some(&v)) {
            return Compared::Equal(v);
        }
        on_write();
        Compared::Replaced(self.write(v))
    }
    fn clear(&self) -> Option<()> {
        self.swap(std::ptr::null_mut())
    }
}

impl<T> Drop for LoomValueSlot<T> {
    fn drop(&mut self) {
        let ptr = self.ptr.with_mut(|p| *p);
        if !ptr.is_null() {
            // SAFETY: the slot owns its current pointer, which came from
            // Box::into_raw, and nothing can load it past the slot's drop.
            drop(unsafe { Box::from_raw(ptr) });
        }
    }
}

pub struct LoomDepListCell(RwLock<Option<Box<DepList>>>);

impl DepListCell for LoomDepListCell {
    type Retired = Box<DepList>;
    fn new_null() -> Self {
        Self(RwLock::new(None))
    }
    fn replace(&self, list: Option<Box<DepList>>) -> Option<Box<DepList>> {
        std::mem::replace(
            &mut *self
                .0
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            list,
        )
    }
    fn with_list(&self, f: &mut dyn FnMut(&DepList)) -> bool {
        let guard = self
            .0
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match guard.as_deref() {
            Some(list) => {
                f(list);
                true
            }
            None => false,
        }
    }
}

/// A node table's array under loom: readers clone the `Arc` under the
/// read lock, so loom sees every publish and replace.
pub struct LoomArrayCell<A>(RwLock<loom::sync::Arc<A>>);

impl<A: 'static> ArrayCell<A> for LoomArrayCell<A> {
    type Retired = loom::sync::Arc<A>;

    fn new(array: A) -> Self {
        Self(RwLock::new(loom::sync::Arc::new(array)))
    }

    fn read<R>(&self, f: impl FnOnce(&A) -> R) -> R {
        let array = self.0.read().unwrap().clone();
        f(&array)
    }

    fn replace(&self, array: A) -> loom::sync::Arc<A> {
        std::mem::replace(&mut *self.0.write().unwrap(), loom::sync::Arc::new(array))
    }
}

/// A bucket's data under loom's `UnsafeCell`, so a model checks the key
/// and handle cells for races.
pub struct LoomDataCell<T>(loom::cell::UnsafeCell<T>);

// SAFETY: as for `StdDataCell`: written only under the miss mutex before
// its bucket is published, then only read.
unsafe impl<T: Send + Sync> Sync for LoomDataCell<T> {}

impl<T: 'static> DataCell<T> for LoomDataCell<T> {
    fn new(value: T) -> Self {
        Self(loom::cell::UnsafeCell::new(value))
    }

    fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R {
        self.0.with(f)
    }

    fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        self.0.with_mut(f)
    }
}

impl Cells for Loom {
    type U8 = AtomicU8;
    type U16 = AtomicU16;
    type U32 = AtomicU32;
    type U64 = AtomicU64;
    type State = AtomicU8;
    type Ptr<T: 'static> = LoomPtrCell<T>;
    type Lock<T: 'static> = LoomLock<T>;
    type DepStack = LoomDepStack;
    type ValueSlot<T: Value> = LoomValueSlot<T>;
    type DepListCell = LoomDepListCell;
    // Two slots, so a model with three revisions reaches slot reuse and
    // the wait path; a full-size ring would also put a loom atomic per
    // slot into every execution.
    type Watermark = WalkRing<Loom, 2>;
    type TableArray<K: TableKey> = LoomArrayCell<Array<K, Loom>>;
    type DataCell<T: Send + Sync + 'static> = LoomDataCell<T>;

    fn new_u8(v: u8) -> Self::U8 {
        AtomicU8::new(v)
    }
    fn new_u16(v: u16) -> Self::U16 {
        AtomicU16::new(v)
    }
    fn new_u32(v: u32) -> Self::U32 {
        AtomicU32::new(v)
    }
    fn new_u64(v: u64) -> Self::U64 {
        AtomicU64::new(v)
    }
    fn new_state(v: u8) -> Self::State {
        AtomicU8::new(v)
    }

    fn u16_load_relaxed(c: &Self::U16) -> u16 {
        c.load(Ordering::Relaxed)
    }
    fn u16_store_relaxed(c: &Self::U16, v: u16) {
        c.store(v, Ordering::Relaxed);
    }

    fn u8_load_acquire(c: &Self::U8) -> u8 {
        c.load(Ordering::Acquire)
    }
    fn u8_store_release(c: &Self::U8, v: u8) {
        c.store(v, Ordering::Release);
    }
    fn u8_load_relaxed(c: &Self::U8) -> u8 {
        c.load(Ordering::Relaxed)
    }
    fn u8_store_relaxed(c: &Self::U8, v: u8) {
        c.store(v, Ordering::Relaxed);
    }
    fn u8_compare_exchange(c: &Self::U8, current: u8, new: u8) -> Result<u8, u8> {
        c.compare_exchange(current, new, Ordering::AcqRel, Ordering::Acquire)
    }

    fn u32_load_acquire(c: &Self::U32) -> u32 {
        c.load(Ordering::Acquire)
    }
    fn u32_store_release(c: &Self::U32, v: u32) {
        c.store(v, Ordering::Release);
    }
    fn u32_load_relaxed(c: &Self::U32) -> u32 {
        c.load(Ordering::Relaxed)
    }
    fn u32_store_relaxed(c: &Self::U32, v: u32) {
        c.store(v, Ordering::Relaxed);
    }

    fn u64_load_acquire(c: &Self::U64) -> u64 {
        c.load(Ordering::Acquire)
    }
    fn u64_store_release(c: &Self::U64, v: u64) {
        c.store(v, Ordering::Release);
    }
    fn u64_load_relaxed(c: &Self::U64) -> u64 {
        c.load(Ordering::Relaxed)
    }
    fn u64_store_relaxed(c: &Self::U64, v: u64) {
        c.store(v, Ordering::Relaxed);
    }
    fn u64_fetch_add(c: &Self::U64, v: u64) -> u64 {
        c.fetch_add(v, Ordering::AcqRel)
    }
    fn u64_fetch_max(c: &Self::U64, v: u64) -> u64 {
        c.fetch_max(v, Ordering::AcqRel)
    }
    fn u64_fetch_or(c: &Self::U64, v: u64) -> u64 {
        c.fetch_or(v, Ordering::AcqRel)
    }
    fn u64_compare_exchange(c: &Self::U64, current: u64, new: u64) -> Result<u64, u64> {
        c.compare_exchange(current, new, Ordering::AcqRel, Ordering::Acquire)
    }

    fn new_watermark(initial_revision: u64) -> Self::Watermark {
        WalkRing::new(initial_revision)
    }
    fn walked(w: &Self::Watermark, _revision: &Self::U64) -> u64 {
        w.walked()
    }
    fn complete_walk(w: &Self::Watermark, rev: u64) {
        w.complete(rev);
    }
    fn fence_seqcst() {
        loom::sync::atomic::fence(Ordering::SeqCst);
    }
    fn fence_acquire() {
        loom::sync::atomic::fence(Ordering::Acquire);
    }
    fn fence_release() {
        loom::sync::atomic::fence(Ordering::Release);
    }

    fn state_load_acquire(c: &Self::State) -> u8 {
        c.load(Ordering::Acquire)
    }
    fn state_store_release(c: &Self::State, v: u8) {
        c.store(v, Ordering::Release);
    }
    fn state_try_transition(c: &Self::State, expected: u8, new: u8) -> Result<(), u8> {
        c.compare_exchange(expected, new, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
    }

    fn backoff(_spins: u32) {
        loom::thread::yield_now();
    }
}
