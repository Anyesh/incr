//! `Loom`: a third `Cells` strategy, compiled only under `cfg(loom)`, that
//! runs the real runtime protocol on loom's modeled atomics and locks so
//! `loom::model` can explore its interleavings.
//!
//! Orderings mirror [`Shared`](crate::cells::Shared) exactly; a model is
//! only evidence about `Shared` if every ordering here matches it. Value
//! slots and overflow dep lists are loom `RwLock`s instead of hazard
//! pointers: `SharedValueSlot` issues a SeqCst fence on every protect,
//! which would multiply the explored state space, and the models keep
//! every dep list inline.
//!
//! Every value of a loom type has to be created inside `loom::model`, so
//! no non-loom test can run in a `--cfg loom` build. The models are
//! named `loom_*` and CI filters on that prefix.

use crate::cells::{Cells, PtrCell};
use crate::dep_stack::LoomDepStack;
use crate::locks::Lock;
use crate::node::{DepList, DepListCell};
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

pub struct LoomValueSlot<T>(RwLock<Option<T>>);

impl<T: Value> LoomValueSlot<T> {
    fn read_guard(&self) -> RwLockReadGuard<'_, Option<T>> {
        self.0
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn write_guard(&self) -> RwLockWriteGuard<'_, Option<T>> {
        self.0
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl<T: Value> ValueSlot<T> for LoomValueSlot<T> {
    type Displaced = T;
    fn new_empty() -> Self {
        Self(RwLock::new(None))
    }
    fn new_with(v: T) -> Self {
        Self(RwLock::new(Some(v)))
    }
    fn read(&self) -> T {
        self.read_guard()
            .as_ref()
            .expect("ValueSlot::read on a slot that was never written")
            .clone()
    }
    fn try_read(&self) -> Option<T> {
        self.read_guard().clone()
    }
    fn write(&self, v: T) -> Option<T> {
        self.write_guard().replace(v)
    }
    fn eq_current(&self, v: &T) -> bool {
        self.read_guard().as_ref() == Some(v)
    }
    fn write_if_changed(&self, v: T, on_write: impl FnOnce()) -> Compared<T, T> {
        let mut slot = self.write_guard();
        if value_code::run(|| slot.as_ref() == Some(&v)) {
            return Compared::Equal(v);
        }
        on_write();
        Compared::Replaced(slot.replace(v))
    }
    fn clear(&self) -> Option<T> {
        self.write_guard().take()
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
