//! `ValueSlot<T>`: strategy-parameterized storage for one node's value.
//!
//! The slot is the synchronization point that makes `get`/`set`/recompute
//! sound under `Shared`: a reader cloning `T` must never observe a write
//! in progress, and no state-machine handshake can guarantee that for
//! inputs (a `set()` does not transition node state) or for readers that
//! observed `Clean` an instant before a recompute claimed the node.
//!
//! - [`LocalValueSlot`] backs the value with `UnsafeCell<Option<T>>` and
//!   reads/writes in place. Single-threaded execution makes accesses
//!   strictly sequential; this is byte-for-byte the pre-existing Local
//!   cost (one pointer chase removed, in fact, since slots now live
//!   inline in the arena's Vec).
//! - [`SharedValueSlot`] stores a `haphazard::AtomicPtr<T>`. Writers swap
//!   in a freshly boxed value and retire the old allocation through the
//!   global hazard-pointer domain; readers clone behind a hazard pointer.
//!   Reads can never tear (the pointer swap is atomic and the pointee is
//!   immutable once published) and the old value cannot be freed under a
//!   reader's feet. Pointer-swap was chosen over a per-slot seqlock
//!   because cloning a torn `&T` is UB for heap-backed values no matter
//!   how the read is retried, and over per-slot locks because a lock
//!   acquire on the uncontended read path violates the performance
//!   mandate. Reclamation reuses the same haphazard domain as the
//!   overflow dep lists, which is Miri-clean (crossbeam-epoch is not).
//!
//! Hazard pointers are pooled per thread: acquiring one from the global
//! domain walks a shared free list, which would put cross-thread
//! contention on every read. The pool makes steady-state acquisition two
//! thread-local operations. Nested reads (a `Clone`/`PartialEq` impl
//! that itself reads from the runtime) pop a second pointer from the
//! pool, so re-entrancy cannot alias a hazard slot.

use std::cell::{RefCell, UnsafeCell};

use haphazard::{AtomicPtr as HzAtomicPtr, HazardPointer};

use crate::value::Value;

/// Storage for one node's value, parameterized by the `Cells` strategy
/// through `Cells::ValueSlot<T>`.
pub trait ValueSlot<T: Value>: 'static {
    fn new_empty() -> Self;
    fn new_with(v: T) -> Self;
    /// Clone the current value out. Panics if the slot was never written
    /// (the runtime guarantees a query's first compute completes before
    /// any reader reaches its slot).
    fn read(&self) -> T;
    fn try_read(&self) -> Option<T>;
    fn write(&self, v: T);
    /// Compare the current value against `v` without cloning. False if
    /// the slot is empty. Used by the early-cutoff and set() no-op
    /// checks, where cloning heap values just to compare is waste.
    fn eq_current(&self, v: &T) -> bool;

    /// Write `v` unless the current value already equals it. Returns
    /// true iff the value changed (or the slot was empty). One slot
    /// session instead of an eq_current + write pair, which matters
    /// under Shared where each session costs a hazard acquire.
    fn write_if_changed(&self, v: T) -> bool;
}

/// Local strategy: in-place storage.
pub struct LocalValueSlot<T>(UnsafeCell<Option<T>>);

impl<T: Value> ValueSlot<T> for LocalValueSlot<T> {
    #[inline(always)]
    fn new_empty() -> Self {
        Self(UnsafeCell::new(None))
    }

    #[inline(always)]
    fn new_with(v: T) -> Self {
        Self(UnsafeCell::new(Some(v)))
    }

    #[inline(always)]
    fn read(&self) -> T {
        // SAFETY: the Local strategy is confined to one thread
        // (Runtime<Local> is !Send + !Sync), so accesses to this cell
        // are strictly sequential and the reference created here cannot
        // overlap a write: `read`, `write`, `try_read`, and `eq_current`
        // each drop their reference before returning, and user code
        // never receives a reference into the cell.
        unsafe {
            (*self.0.get())
                .as_ref()
                .expect("ValueSlot::read on a slot that was never written")
                .clone()
        }
    }

    #[inline(always)]
    fn try_read(&self) -> Option<T> {
        // SAFETY: see read().
        unsafe { (*self.0.get()).as_ref().cloned() }
    }

    #[inline(always)]
    fn write(&self, v: T) {
        // SAFETY: see read(); single-threaded sequential access, no
        // outstanding reference into the cell can exist here.
        unsafe {
            *self.0.get() = Some(v);
        }
    }

    #[inline(always)]
    fn eq_current(&self, v: &T) -> bool {
        // SAFETY: see read().
        unsafe {
            (*self.0.get())
                .as_ref()
                .map(|cur| cur == v)
                .unwrap_or(false)
        }
    }

    #[inline(always)]
    fn write_if_changed(&self, v: T) -> bool {
        // SAFETY: see read(); the comparison reference is dropped
        // before the write below.
        let unchanged = unsafe {
            (*self.0.get())
                .as_ref()
                .map(|cur| cur == &v)
                .unwrap_or(false)
        };
        if unchanged {
            return false;
        }
        // SAFETY: see write().
        unsafe {
            *self.0.get() = Some(v);
        }
        true
    }
}

thread_local! {
    static HAZARD_POOL: RefCell<Vec<HazardPointer<'static>>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` with a hazard pointer drawn from the thread-local pool,
/// returning the pointer afterwards. If `f` panics the pointer is
/// dropped instead of pooled, which releases its protection; nothing
/// leaks protection past this call.
fn with_pooled_hazard<R>(f: impl FnOnce(&mut HazardPointer<'static>) -> R) -> R {
    let mut hp = HAZARD_POOL
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_default();
    let out = f(&mut hp);
    hp.reset_protection();
    HAZARD_POOL.with(|pool| pool.borrow_mut().push(hp));
    out
}

/// Shared strategy: atomic pointer swap with hazard-protected reads.
/// Null means "never written".
pub struct SharedValueSlot<T: Value>(HzAtomicPtr<T>);

impl<T: Value> ValueSlot<T> for SharedValueSlot<T> {
    fn new_empty() -> Self {
        // SAFETY: null is the documented empty encoding; load() treats
        // it as None and never dereferences it.
        Self(unsafe { HzAtomicPtr::new(std::ptr::null_mut()) })
    }

    fn new_with(v: T) -> Self {
        Self(HzAtomicPtr::from(Box::new(v)))
    }

    fn read(&self) -> T {
        with_pooled_hazard(|hp| {
            // SAFETY: every non-null pointer in this slot came from
            // Box::into_raw via new_with/write, and every displaced
            // pointer is retired through the global haphazard domain,
            // so the protected reference stays valid for the hazard's
            // lifetime.
            let cur = unsafe { self.0.load(hp) };
            cur.expect("ValueSlot::read on a slot that was never written")
                .clone()
        })
    }

    fn try_read(&self) -> Option<T> {
        with_pooled_hazard(|hp| {
            // SAFETY: see read().
            unsafe { self.0.load(hp) }.cloned()
        })
    }

    fn write(&self, v: T) {
        let replaced = self.0.swap(Box::new(v));
        if let Some(old) = replaced {
            // SAFETY: `old` came from Box::into_raw via a previous
            // new_with/write on this slot and is displaced exactly once
            // (swap is atomic); hazard pointers defer the actual free
            // until no reader still references it.
            unsafe { old.retire() };
        }
    }

    fn eq_current(&self, v: &T) -> bool {
        with_pooled_hazard(|hp| {
            // SAFETY: see read().
            unsafe { self.0.load(hp) }
                .map(|cur| cur == v)
                .unwrap_or(false)
        })
    }

    fn write_if_changed(&self, v: T) -> bool {
        let unchanged = with_pooled_hazard(|hp| {
            // SAFETY: see read().
            unsafe { self.0.load(hp) }
                .map(|cur| cur == &v)
                .unwrap_or(false)
        });
        if unchanged {
            return false;
        }
        self.write(v);
        true
    }
}

impl<T: Value> Drop for SharedValueSlot<T> {
    fn drop(&mut self) {
        // SAFETY: swap_ptr with null is always safe; the displaced
        // pointer (if any) came from Box::into_raw and goes through the
        // haphazard retire path like every other displacement.
        let displaced = unsafe { self.0.swap_ptr(std::ptr::null_mut()) };
        if let Some(old) = displaced {
            // SAFETY: displaced exactly once; readers are hazard-protected.
            unsafe { old.retire() };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_roundtrip_and_eq() {
        let s: LocalValueSlot<String> = ValueSlot::new_with("a".to_string());
        assert_eq!(s.read(), "a");
        assert!(s.eq_current(&"a".to_string()));
        s.write("b".to_string());
        assert_eq!(s.try_read(), Some("b".to_string()));
        assert!(!s.eq_current(&"a".to_string()));
    }

    #[test]
    fn local_empty_slot() {
        let s: LocalValueSlot<u64> = ValueSlot::new_empty();
        assert_eq!(s.try_read(), None);
        assert!(!s.eq_current(&7));
        s.write(7);
        assert_eq!(s.read(), 7);
    }

    #[test]
    fn shared_roundtrip_and_eq() {
        let s: SharedValueSlot<String> = ValueSlot::new_with("a".to_string());
        assert_eq!(s.read(), "a");
        assert!(s.eq_current(&"a".to_string()));
        s.write("b".to_string());
        assert_eq!(s.try_read(), Some("b".to_string()));
        assert!(!s.eq_current(&"a".to_string()));
    }

    #[test]
    fn shared_empty_slot() {
        let s: SharedValueSlot<u64> = ValueSlot::new_empty();
        assert_eq!(s.try_read(), None);
        assert!(!s.eq_current(&7));
        s.write(7);
        assert_eq!(s.read(), 7);
    }

    #[test]
    fn shared_slot_is_send_sync() {
        fn assert_send_sync<X: Send + Sync>() {}
        assert_send_sync::<SharedValueSlot<String>>();
        assert_send_sync::<SharedValueSlot<Vec<u8>>>();
    }

    /// Readers clone heap values while writers swap them. Under the old
    /// in-place arena this was the UB Miri flagged; with pointer-swap
    /// slots every observed value is one of the published strings.
    #[test]
    fn shared_concurrent_heap_value_hammer() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let iters = if cfg!(miri) { 50 } else { 2000 };
        let slot: Arc<SharedValueSlot<String>> = Arc::new(ValueSlot::new_with("v0000".to_string()));
        let stop = Arc::new(AtomicBool::new(false));

        let readers: Vec<_> = (0..3)
            .map(|_| {
                let slot = Arc::clone(&slot);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let v = slot.read();
                        assert!(
                            v.starts_with('v') && v.len() == 5,
                            "torn or freed value observed: {:?}",
                            v
                        );
                    }
                })
            })
            .collect();

        for i in 0..iters {
            slot.write(format!("v{:04}", i % 10_000));
        }
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
    }
}
