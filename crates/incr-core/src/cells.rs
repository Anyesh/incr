//! `Cells`: the strategy trait that abstracts how scalar cells are
//! synchronized. [`Local`] backs every cell with `std::cell::Cell`;
//! [`Shared`] backs every cell with the matching atomic type and uses
//! Acquire/Release for visibility transitions.
//!
//! All trait methods are `#[inline(always)]` and take `&Self::Cell` so the
//! compiler can see through every call site and emit the same code it
//! would for a direct field access. This monomorphization is the
//! load-bearing invariant the spike validated: under `Local`, trait
//! method calls produce byte-identical assembly to direct `Cell::get()`
//! and `Cell::set()` operations; under `Shared` on x86, Acquire compiles
//! to a plain `mov` with no fences.
//!
//! `compare_exchange` is exposed only on the state cell because that is
//! the only place CAS is meaningful (one thread races to claim a node's
//! `Computing` slot). Other integer cells are written under exclusive
//! ownership granted by the state machine.

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};

/// Strategy trait selecting the synchronization primitives used by every
/// cell in the engine. Implemented by [`Local`] and [`Shared`].
///
/// `'static` so cell types can be embedded in trait-object closures
/// without lifetime gymnastics. `Sized` to allow associated-type
/// constructors.
pub trait Cells: 'static + Sized {
    type U8;
    type U32;
    type U64;
    type State;

    fn new_u8(v: u8) -> Self::U8;
    fn new_u32(v: u32) -> Self::U32;
    fn new_u64(v: u64) -> Self::U64;
    fn new_state(v: u8) -> Self::State;

    fn u8_load_acquire(c: &Self::U8) -> u8;
    fn u8_store_release(c: &Self::U8, v: u8);
    fn u8_load_relaxed(c: &Self::U8) -> u8;
    fn u8_store_relaxed(c: &Self::U8, v: u8);

    fn u32_load_relaxed(c: &Self::U32) -> u32;
    fn u32_store_relaxed(c: &Self::U32, v: u32);

    fn u64_load_acquire(c: &Self::U64) -> u64;
    fn u64_store_release(c: &Self::U64, v: u64);
    fn u64_load_relaxed(c: &Self::U64) -> u64;
    fn u64_store_relaxed(c: &Self::U64, v: u64);

    fn state_load_acquire(c: &Self::State) -> u8;
    fn state_store_release(c: &Self::State, v: u8);
    fn state_try_transition(c: &Self::State, expected: u8, new: u8) -> Result<(), u8>;
}

/// Single-threaded strategy. Backs every cell with `std::cell::Cell`.
/// The resulting types are `!Sync` and the runtime built on top of
/// `Local` is `!Send + !Sync` by composition.
pub struct Local;

/// Multi-threaded strategy. Backs every cell with the matching atomic
/// type and uses Acquire/Release for state-visibility transitions.
/// The resulting types are `Send + Sync`.
pub struct Shared;

impl Cells for Local {
    type U8 = Cell<u8>;
    type U32 = Cell<u32>;
    type U64 = Cell<u64>;
    type State = Cell<u8>;

    #[inline(always)]
    fn new_u8(v: u8) -> Self::U8 {
        Cell::new(v)
    }
    #[inline(always)]
    fn new_u32(v: u32) -> Self::U32 {
        Cell::new(v)
    }
    #[inline(always)]
    fn new_u64(v: u64) -> Self::U64 {
        Cell::new(v)
    }
    #[inline(always)]
    fn new_state(v: u8) -> Self::State {
        Cell::new(v)
    }

    #[inline(always)]
    fn u8_load_acquire(c: &Self::U8) -> u8 {
        c.get()
    }
    #[inline(always)]
    fn u8_store_release(c: &Self::U8, v: u8) {
        c.set(v);
    }
    #[inline(always)]
    fn u8_load_relaxed(c: &Self::U8) -> u8 {
        c.get()
    }
    #[inline(always)]
    fn u8_store_relaxed(c: &Self::U8, v: u8) {
        c.set(v);
    }

    #[inline(always)]
    fn u32_load_relaxed(c: &Self::U32) -> u32 {
        c.get()
    }
    #[inline(always)]
    fn u32_store_relaxed(c: &Self::U32, v: u32) {
        c.set(v);
    }

    #[inline(always)]
    fn u64_load_acquire(c: &Self::U64) -> u64 {
        c.get()
    }
    #[inline(always)]
    fn u64_store_release(c: &Self::U64, v: u64) {
        c.set(v);
    }
    #[inline(always)]
    fn u64_load_relaxed(c: &Self::U64) -> u64 {
        c.get()
    }
    #[inline(always)]
    fn u64_store_relaxed(c: &Self::U64, v: u64) {
        c.set(v);
    }

    #[inline(always)]
    fn state_load_acquire(c: &Self::State) -> u8 {
        c.get()
    }
    #[inline(always)]
    fn state_store_release(c: &Self::State, v: u8) {
        c.set(v);
    }
    #[inline(always)]
    fn state_try_transition(c: &Self::State, expected: u8, new: u8) -> Result<(), u8> {
        let cur = c.get();
        if cur == expected {
            c.set(new);
            Ok(())
        } else {
            Err(cur)
        }
    }
}

impl Cells for Shared {
    type U8 = AtomicU8;
    type U32 = AtomicU32;
    type U64 = AtomicU64;
    type State = AtomicU8;

    #[inline(always)]
    fn new_u8(v: u8) -> Self::U8 {
        AtomicU8::new(v)
    }
    #[inline(always)]
    fn new_u32(v: u32) -> Self::U32 {
        AtomicU32::new(v)
    }
    #[inline(always)]
    fn new_u64(v: u64) -> Self::U64 {
        AtomicU64::new(v)
    }
    #[inline(always)]
    fn new_state(v: u8) -> Self::State {
        AtomicU8::new(v)
    }

    #[inline(always)]
    fn u8_load_acquire(c: &Self::U8) -> u8 {
        c.load(Ordering::Acquire)
    }
    #[inline(always)]
    fn u8_store_release(c: &Self::U8, v: u8) {
        c.store(v, Ordering::Release);
    }
    #[inline(always)]
    fn u8_load_relaxed(c: &Self::U8) -> u8 {
        c.load(Ordering::Relaxed)
    }
    #[inline(always)]
    fn u8_store_relaxed(c: &Self::U8, v: u8) {
        c.store(v, Ordering::Relaxed);
    }

    #[inline(always)]
    fn u32_load_relaxed(c: &Self::U32) -> u32 {
        c.load(Ordering::Relaxed)
    }
    #[inline(always)]
    fn u32_store_relaxed(c: &Self::U32, v: u32) {
        c.store(v, Ordering::Relaxed);
    }

    #[inline(always)]
    fn u64_load_acquire(c: &Self::U64) -> u64 {
        c.load(Ordering::Acquire)
    }
    #[inline(always)]
    fn u64_store_release(c: &Self::U64, v: u64) {
        c.store(v, Ordering::Release);
    }
    #[inline(always)]
    fn u64_load_relaxed(c: &Self::U64) -> u64 {
        c.load(Ordering::Relaxed)
    }
    #[inline(always)]
    fn u64_store_relaxed(c: &Self::U64, v: u64) {
        c.store(v, Ordering::Relaxed);
    }

    #[inline(always)]
    fn state_load_acquire(c: &Self::State) -> u8 {
        c.load(Ordering::Acquire)
    }
    #[inline(always)]
    fn state_store_release(c: &Self::State, v: u8) {
        c.store(v, Ordering::Release);
    }
    #[inline(always)]
    fn state_try_transition(c: &Self::State, expected: u8, new: u8) -> Result<(), u8> {
        match c.compare_exchange(expected, new, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => Ok(()),
            Err(observed) => Err(observed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_roundtrips() {
        let c = Local::new_u64(7);
        assert_eq!(Local::u64_load_acquire(&c), 7);
        Local::u64_store_release(&c, 11);
        assert_eq!(Local::u64_load_acquire(&c), 11);
    }

    #[test]
    fn shared_roundtrips() {
        let c = Shared::new_u64(7);
        assert_eq!(Shared::u64_load_acquire(&c), 7);
        Shared::u64_store_release(&c, 11);
        assert_eq!(Shared::u64_load_acquire(&c), 11);
    }

    #[test]
    fn local_state_cas() {
        let s = Local::new_state(1);
        assert_eq!(Local::state_try_transition(&s, 1, 2), Ok(()));
        assert_eq!(Local::state_try_transition(&s, 1, 3), Err(2));
    }

    #[test]
    fn shared_state_cas() {
        let s = Shared::new_state(1);
        assert_eq!(Shared::state_try_transition(&s, 1, 2), Ok(()));
        assert_eq!(Shared::state_try_transition(&s, 1, 3), Err(2));
    }

    #[test]
    fn cell_sizes_match_atomic_sizes() {
        assert_eq!(
            std::mem::size_of::<<Local as Cells>::U64>(),
            std::mem::size_of::<<Shared as Cells>::U64>(),
        );
        assert_eq!(
            std::mem::size_of::<<Local as Cells>::U32>(),
            std::mem::size_of::<<Shared as Cells>::U32>(),
        );
        assert_eq!(
            std::mem::size_of::<<Local as Cells>::U8>(),
            std::mem::size_of::<<Shared as Cells>::U8>(),
        );
    }
}
