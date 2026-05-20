//! `Lock<T>`: strategy-parameterized mutex-like primitive used for the
//! runtime's inner-state fields.
//!
//! Local backs the lock with `RefCell` (single-threaded, no synchronization
//! cost beyond a borrow-counter check). Shared backs it with `std::sync::RwLock`
//! (read-write lock with reader parallelism). The trait abstracts the
//! guard types via GATs so callers can write `let g = lock.read(); ... &*g ...`
//! identically across both strategies.
//!
//! Poisoning: under Shared, if a thread panics while holding the write
//! guard, the underlying RwLock becomes poisoned. We treat poisoning as a
//! fatal runtime invariant violation and `.expect()` it. The user-facing
//! API surfaces are panic-only after such a failure; no recovery path.

use std::cell::{Ref, RefCell, RefMut};
use std::ops::{Deref, DerefMut};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Mutex-like primitive abstracting RefCell (Local) and RwLock (Shared).
pub trait Lock<T: 'static>: 'static {
    type ReadGuard<'a>: Deref<Target = T>
    where
        Self: 'a;
    type WriteGuard<'a>: DerefMut<Target = T>
    where
        Self: 'a;

    fn new(val: T) -> Self;
    fn read(&self) -> Self::ReadGuard<'_>;
    fn write(&self) -> Self::WriteGuard<'_>;
}

/// Local strategy: `RefCell` wrapped in a newtype so we can implement the
/// `Lock` trait on it without conflicting with foreign-trait rules.
pub struct LocalLock<T>(RefCell<T>);

impl<T: 'static> Lock<T> for LocalLock<T> {
    type ReadGuard<'a>
        = Ref<'a, T>
    where
        T: 'a;
    type WriteGuard<'a>
        = RefMut<'a, T>
    where
        T: 'a;

    #[inline(always)]
    fn new(val: T) -> Self {
        Self(RefCell::new(val))
    }

    #[inline(always)]
    fn read(&self) -> Self::ReadGuard<'_> {
        self.0.borrow()
    }

    #[inline(always)]
    fn write(&self) -> Self::WriteGuard<'_> {
        self.0.borrow_mut()
    }
}

/// Shared strategy: `std::sync::RwLock`. The trait blanket impl below
/// uses the lock as-is; no newtype wrapper is needed because the foreign
/// impl is for `RwLock` directly (a foreign trait on a foreign type is
/// disallowed, but our local `Lock` trait on `RwLock` is fine).
impl<T: 'static> Lock<T> for RwLock<T> {
    type ReadGuard<'a>
        = RwLockReadGuard<'a, T>
    where
        T: 'a;
    type WriteGuard<'a>
        = RwLockWriteGuard<'a, T>
    where
        T: 'a;

    #[inline(always)]
    fn new(val: T) -> Self {
        RwLock::new(val)
    }

    #[inline(always)]
    fn read(&self) -> Self::ReadGuard<'_> {
        RwLock::read(self).expect("incr-core inner lock poisoned (Shared)")
    }

    #[inline(always)]
    fn write(&self) -> Self::WriteGuard<'_> {
        RwLock::write(self).expect("incr-core inner lock poisoned (Shared)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_lock_read_write() {
        let l: LocalLock<u64> = LocalLock::new(7);
        assert_eq!(*l.read(), 7);
        *l.write() = 11;
        assert_eq!(*l.read(), 11);
    }

    #[test]
    fn shared_lock_read_write() {
        let l: RwLock<u64> = <RwLock<u64> as Lock<u64>>::new(7);
        assert_eq!(*Lock::read(&l), 7);
        *Lock::write(&l) = 11;
        assert_eq!(*Lock::read(&l), 11);
    }
}
