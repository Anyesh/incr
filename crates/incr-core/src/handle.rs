//! Public handle type `Incr<T>` and runtime identity.
//!
//! `Incr<T>` is a 16-byte `Copy` token returned by `Runtime::create_input`
//! and `Runtime::create_query`. It carries:
//!
//! - `slot: u32` — index into the runtime's segmented node store.
//! - `generation: u32` — the slot's generation counter, for detecting
//!   use-after-recycle (reserved; recycling lands with `delete_node` in a
//!   follow-up).
//! - `runtime_id: RuntimeId` (u64) — uniquely identifies the owning
//!   `Runtime` for the process lifetime. Used to reject handles from
//!   foreign runtimes with a clear error.
//! - `_phantom: PhantomData<fn() -> T>` — locks `T` at the type level
//!   without inheriting `T`'s auto traits. `Incr<T>` is always
//!   `Send + Sync + Copy + Unpin` regardless of `T`.
//!
//! Total: 16 bytes, 8-byte aligned. Asserted by tests.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

/// Unique identifier for a `Runtime` instance. Drawn from a process-wide
/// monotonic counter; never reused within a process lifetime.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct RuntimeId(u64);

impl RuntimeId {
    #[allow(dead_code)]
    pub(crate) const SENTINEL: RuntimeId = RuntimeId(0);

    /// Allocate a fresh runtime id. Called once per `Runtime::new`.
    pub(crate) fn allocate() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    #[inline]
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Typed handle to a node in a `Runtime<C>`.
#[repr(C)]
pub struct Incr<T: 'static> {
    slot: u32,
    generation: u32,
    runtime_id: RuntimeId,
    _phantom: PhantomData<fn() -> T>,
}

impl<T: 'static> Copy for Incr<T> {}
impl<T: 'static> Clone for Incr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: 'static> std::fmt::Debug for Incr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Incr")
            .field("slot", &self.slot)
            .field("generation", &self.generation)
            .field("runtime_id", &self.runtime_id)
            .field("type", &std::any::type_name::<T>())
            .finish()
    }
}

impl<T: 'static> PartialEq for Incr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot
            && self.generation == other.generation
            && self.runtime_id == other.runtime_id
    }
}

impl<T: 'static> Eq for Incr<T> {}

impl<T: 'static> std::hash::Hash for Incr<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.slot.hash(state);
        self.generation.hash(state);
        self.runtime_id.hash(state);
    }
}

impl<T: 'static> Incr<T> {
    pub(crate) fn new(slot: u32, generation: u32, runtime_id: RuntimeId) -> Self {
        Self {
            slot,
            generation,
            runtime_id,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn slot(self) -> u32 {
        self.slot
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn generation(self) -> u32 {
        self.generation
    }

    #[inline]
    pub(crate) fn runtime_id(self) -> RuntimeId {
        self.runtime_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incr_is_16_bytes_8_aligned() {
        assert_eq!(std::mem::size_of::<Incr<u64>>(), 16);
        assert_eq!(std::mem::align_of::<Incr<u64>>(), 8);
        assert_eq!(std::mem::size_of::<Incr<String>>(), 16);
        assert_eq!(std::mem::size_of::<Incr<Vec<u8>>>(), 16);
    }

    #[test]
    fn incr_is_send_sync_regardless_of_t() {
        fn assert_send_sync<T: Send + Sync>() {}
        fn assert_copy<T: Copy>() {}
        assert_send_sync::<Incr<u64>>();
        assert_copy::<Incr<u64>>();
        assert_send_sync::<Incr<String>>();
        assert_send_sync::<Incr<std::cell::RefCell<u64>>>();
        assert_send_sync::<Incr<std::rc::Rc<u64>>>();
    }

    #[test]
    fn runtime_id_sentinel_is_zero() {
        assert_eq!(RuntimeId::SENTINEL.get(), 0);
        let real = RuntimeId::allocate();
        assert_ne!(real, RuntimeId::SENTINEL);
        assert!(real.get() >= 1);
    }

    #[test]
    fn runtime_ids_are_unique() {
        let a = RuntimeId::allocate();
        let b = RuntimeId::allocate();
        assert_ne!(a, b);
    }
}
