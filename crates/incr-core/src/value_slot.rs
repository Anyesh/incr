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

use haphazard::{AtomicPtr as HzAtomicPtr, Global, HazardPointer, Replaced};

use crate::cells::Cells;
use crate::value::Value;
use crate::value_code;

/// Storage for one node's value, parameterized by the `Cells` strategy
/// through `Cells::ValueSlot<T>`.
pub trait ValueSlot<T: Value>: 'static {
    /// What a write hands back for the value it displaced: the value
    /// itself under `Local`, a retire token under `Shared`. Dropping it
    /// runs the value's `Drop`, or retires it so hazard-protected readers
    /// finish first, so the caller drops it outside every lock.
    type Displaced;

    fn new_empty() -> Self;
    fn new_with(v: T) -> Self;
    /// Clone the current value out. Panics if the slot was never written
    /// (the runtime guarantees a query's first compute completes before
    /// any reader reaches its slot).
    fn read(&self) -> T;
    fn try_read(&self) -> Option<T>;
    /// [`Self::try_read`] with `missing` run in place of `None`, so a
    /// slot that clones in place hands the value back with no `Option`
    /// to unwrap and copy.
    #[inline(always)]
    fn read_or(&self, missing: impl FnOnce() -> T) -> T {
        match self.try_read() {
            Some(v) => v,
            None => missing(),
        }
    }
    /// Store `v` and hand back the displaced value. The slot owns `v`
    /// before the displaced value can run any user code.
    #[must_use]
    fn write(&self, v: T) -> Option<Self::Displaced>;
    /// Compare the current value against `v` without cloning. False if
    /// the slot is empty. Used by the early-cutoff and set() no-op
    /// checks, where cloning heap values just to compare is waste.
    fn eq_current(&self, v: &T) -> bool;

    /// Write `v` unless the current value already equals it. One slot
    /// session instead of an eq_current + write pair, which matters
    /// under Shared where each session costs a hazard acquire.
    /// `on_write` runs after the comparison and immediately before the
    /// store, and not at all when the values are equal or the comparison
    /// panics, so whatever it records is true exactly when the slot holds
    /// `v`, whatever the displaced value's `Drop` does afterwards.
    #[must_use]
    fn write_if_changed(&self, v: T, on_write: impl FnOnce()) -> Compared<T, Self::Displaced>;

    /// Take the held value, returning the slot to the never-written
    /// state. Used when a node is deleted and its arena slot goes on
    /// the free list.
    #[must_use]
    fn clear(&self) -> Option<Self::Displaced>;

    /// Panic with the contract message if the thread is inside value
    /// code, where only a strategy whose readers hold a reference into
    /// the slot needs it. The arena calls it before a reserve, and the
    /// slot's own writes call it before they write.
    #[inline(always)]
    fn refuse_value_code() {}
}

/// What [`ValueSlot::write_if_changed`] did.
pub enum Compared<T, D> {
    /// The slot already held an equal value; `T` is the rejected one.
    Equal(T),
    /// The slot now holds the new value; the payload is what it displaced.
    Replaced(Option<D>),
}

/// The displaced type of `C`'s slot for `T`.
pub type Displaced<T, C> = <<C as Cells>::ValueSlot<T> as ValueSlot<T>>::Displaced;

/// Local strategy: in-place storage.
pub struct LocalValueSlot<T>(UnsafeCell<Option<T>>);

impl<T: Value> LocalValueSlot<T> {
    #[inline(always)]
    fn replace(&self, v: T) -> Option<T> {
        // SAFETY: see `ValueSlot::read`; the caller refused value code,
        // so no reference into the cell is outstanding.
        unsafe { (*self.0.get()).replace(v) }
    }
}

// A Local read clones through a reference into the cell with no borrow
// flag, and the clone is value code that may call into the runtime. So
// every write, `write_if_changed` and `clear` refuses value code first:
// otherwise a `Clone` that sets its own node would replace the value it
// is cloning from.
impl<T: Value> ValueSlot<T> for LocalValueSlot<T> {
    type Displaced = T;

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
        // are strictly sequential. Only shared references into the cell
        // exist while user code runs (the clone here, a compare), and
        // every path that writes the cell refuses value code first, so
        // no write can happen while one of them is live.
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
    fn read_or(&self, missing: impl FnOnce() -> T) -> T {
        // SAFETY: see read().
        match unsafe { (*self.0.get()).as_ref() } {
            Some(v) => v.clone(),
            None => missing(),
        }
    }

    #[inline(always)]
    fn write(&self, v: T) -> Option<T> {
        value_code::check();
        // The old value is moved out, so the slot owns `v` whatever its
        // Drop does later.
        self.replace(v)
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
    fn write_if_changed(&self, v: T, on_write: impl FnOnce()) -> Compared<T, T> {
        value_code::check();
        // SAFETY: see read(); the comparison reference is dropped
        // before the write below.
        let unchanged = unsafe {
            (*self.0.get())
                .as_ref()
                .map(|cur| value_code::run(|| cur == &v))
                .unwrap_or(false)
        };
        if unchanged {
            return Compared::Equal(v);
        }
        on_write();
        Compared::Replaced(self.replace(v))
    }

    #[inline(always)]
    fn clear(&self) -> Option<T> {
        value_code::check();
        // SAFETY: see read().
        unsafe { (*self.0.get()).take() }
    }

    #[inline(always)]
    fn refuse_value_code() {
        value_code::check();
    }
}

thread_local! {
    static HAZARD_POOL: RefCell<Vec<HazardPointer<'static>>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` with a hazard pointer drawn from the thread-local pool,
/// returning the pointer afterwards. If `f` panics the pointer is
/// dropped instead of pooled, which releases its protection; nothing
/// leaks protection past this call.
pub(crate) fn with_pooled_hazard<R>(f: impl FnOnce(&mut HazardPointer<'static>) -> R) -> R {
    // One `LocalKey::with` that only takes the address, so the call
    // inlines and the pop and push share its liveness check; with the
    // push inside a second one, the compiler leaves it out of line.
    let pool: *const RefCell<Vec<HazardPointer<'static>>> = HAZARD_POOL.with(std::ptr::from_ref);
    // SAFETY: a thread's locals are destroyed only as the thread exits,
    // never during a call on that thread, and the reference does not
    // outlive this call.
    let pool = unsafe { &*pool };
    let mut hp = pool.borrow_mut().pop().unwrap_or_default();
    let out = f(&mut hp);
    hp.reset_protection();
    pool.borrow_mut().push(hp);
    out
}

/// Shared strategy: atomic pointer swap with hazard-protected reads.
/// Null means "never written".
pub struct SharedValueSlot<T: Value>(HzAtomicPtr<T>);

impl<T: Value> ValueSlot<T> for SharedValueSlot<T> {
    type Displaced = RetiredValue<T>;

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

    fn write(&self, v: T) -> Option<RetiredValue<T>> {
        self.0.swap(Box::new(v)).map(RetiredValue)
    }

    fn eq_current(&self, v: &T) -> bool {
        with_pooled_hazard(|hp| {
            // SAFETY: see read().
            unsafe { self.0.load(hp) }
                .map(|cur| cur == v)
                .unwrap_or(false)
        })
    }

    fn write_if_changed(&self, v: T, on_write: impl FnOnce()) -> Compared<T, RetiredValue<T>> {
        let unchanged = with_pooled_hazard(|hp| {
            // SAFETY: see read().
            unsafe { self.0.load(hp) }
                .map(|cur| value_code::run(|| cur == &v))
                .unwrap_or(false)
        });
        if unchanged {
            return Compared::Equal(v);
        }
        on_write();
        Compared::Replaced(self.write(v))
    }

    fn clear(&self) -> Option<RetiredValue<T>> {
        // SAFETY: swap_ptr with null is always safe; the displaced
        // pointer goes through the standard retire path, so concurrent
        // readers finish their clone before the free.
        unsafe { self.0.swap_ptr(std::ptr::null_mut()) }.map(RetiredValue)
    }
}

impl<T: Value> Drop for SharedValueSlot<T> {
    fn drop(&mut self) {
        drop(self.clear());
    }
}

/// The retire token of a value displaced from a [`SharedValueSlot`].
/// Dropping it retires the value through the global haphazard domain,
/// which runs its `Drop` once no reader still references it.
pub struct RetiredValue<T: Value>(Replaced<T, Global, Box<T>>);

impl<T: Value> Drop for RetiredValue<T> {
    fn drop(&mut self) {
        // SAFETY: the pointer came from Box::into_raw via new_with/write
        // on its slot, was displaced exactly once by the atomic swap that
        // made this token, and is never loaded from the slot again;
        // hazard pointers defer the actual free until no reader still
        // references it.
        unsafe { self.0.retire() };
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
        drop(s.write("b".to_string()));
        assert_eq!(s.try_read(), Some("b".to_string()));
        assert!(!s.eq_current(&"a".to_string()));
    }

    #[test]
    fn local_empty_slot() {
        let s: LocalValueSlot<u64> = ValueSlot::new_empty();
        assert_eq!(s.try_read(), None);
        assert!(!s.eq_current(&7));
        let _ = s.write(7);
        assert_eq!(s.read(), 7);
    }

    #[test]
    fn shared_roundtrip_and_eq() {
        let s: SharedValueSlot<String> = ValueSlot::new_with("a".to_string());
        assert_eq!(s.read(), "a");
        assert!(s.eq_current(&"a".to_string()));
        drop(s.write("b".to_string()));
        assert_eq!(s.try_read(), Some("b".to_string()));
        assert!(!s.eq_current(&"a".to_string()));
    }

    #[test]
    fn shared_empty_slot() {
        let s: SharedValueSlot<u64> = ValueSlot::new_empty();
        assert_eq!(s.try_read(), None);
        assert!(!s.eq_current(&7));
        let _ = s.write(7);
        assert_eq!(s.read(), 7);
    }

    /// A value whose `Drop` panics once, for the value `ARMED` names.
    #[derive(Clone, PartialEq, Debug)]
    struct Boom(i64);

    static ARMED: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MIN);

    impl Drop for Boom {
        fn drop(&mut self) {
            use std::sync::atomic::Ordering::SeqCst;
            if ARMED
                .compare_exchange(self.0, i64::MIN, SeqCst, SeqCst)
                .is_ok()
            {
                panic!("drop of Boom({}) panicked on purpose", self.0);
            }
        }
    }

    /// The slot owns the new value before the displaced one's `Drop`
    /// runs, and that `Drop` runs where the caller drops the displaced
    /// value, never inside the slot operation.
    #[test]
    fn local_write_moves_the_old_value_out_before_dropping_it() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        use std::sync::atomic::Ordering::SeqCst;
        let s: LocalValueSlot<Boom> = ValueSlot::new_with(Boom(1));
        let wrote = std::cell::Cell::new(0_u8);
        ARMED.store(1, SeqCst);
        let Compared::Replaced(old) = s.write_if_changed(Boom(2), || wrote.set(1)) else {
            panic!("a different value must be written");
        };
        assert_eq!(wrote.get(), 1);
        assert_eq!(s.read(), Boom(2));
        assert!(catch_unwind(AssertUnwindSafe(|| drop(old))).is_err());

        ARMED.store(2, SeqCst);
        let old = s.write(Boom(3));
        assert_eq!(s.read(), Boom(3));
        assert!(catch_unwind(AssertUnwindSafe(|| drop(old))).is_err());

        ARMED.store(3, SeqCst);
        let old = s.clear();
        assert_eq!(s.try_read(), None);
        assert!(catch_unwind(AssertUnwindSafe(|| drop(old))).is_err());

        assert!(
            s.write(Boom(4)).is_none(),
            "a cleared slot displaces nothing"
        );
        let Compared::Equal(rejected) = s.write_if_changed(Boom(4), || wrote.set(1)) else {
            panic!("an equal value must be rejected");
        };
        assert_eq!(rejected, Boom(4));
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
            drop(slot.write(format!("v{:04}", i % 10_000)));
        }
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
    }
}
