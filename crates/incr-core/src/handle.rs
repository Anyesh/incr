//! Public handle type `Incr<T>` and runtime identity.
//!
//! `Incr<T>` is a 16-byte `Copy` token returned by `Runtime::create_input`
//! and `Runtime::create_query`. It carries:
//!
//! - `slot: u32` — index into the runtime's segmented node store.
//! - `generation: u32` — the slot's generation counter, bumped by each
//!   delete, so a handle to a deleted node never matches the slot's next
//!   occupant.
//! - `runtime_id: RuntimeId` (u64) — uniquely identifies the owning
//!   `Runtime` for the process lifetime. Used to reject handles from
//!   foreign runtimes with a clear error.
//! - `_phantom: PhantomData<fn() -> T>` — locks `T` at the type level
//!   without inheriting `T`'s auto traits. `Incr<T>` is always
//!   `Send + Sync + Copy + Unpin` regardless of `T`.
//!
//! Total: 16 bytes, 8-byte aligned. Asserted by tests.
//!
//! The first three fields form [`ErasedIncr`], the same identity without
//! `T`, which `Incr<T>::erase` returns.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

/// Unique identifier for a `Runtime` instance. Drawn from a process-wide
/// monotonic counter; never reused within a process lifetime.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct RuntimeId(u64);

impl RuntimeId {
    pub(crate) const SENTINEL: RuntimeId = RuntimeId(0);

    #[cfg(test)]
    pub(crate) fn for_test(n: u64) -> Self {
        Self(n)
    }

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

/// Untyped handle to a node: the identity part of an [`Incr<T>`]
/// without `T`. Lets code that walks the graph (whose nodes have mixed
/// value types) name nodes, compare them and ask whether they are alive.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct ErasedIncr {
    slot: u32,
    generation: u32,
    runtime_id: RuntimeId,
}

impl ErasedIncr {
    pub(crate) fn new(slot: u32, generation: u32, runtime_id: RuntimeId) -> Self {
        Self {
            slot,
            generation,
            runtime_id,
        }
    }

    #[inline]
    pub fn slot(self) -> u32 {
        self.slot
    }

    #[inline]
    pub(crate) fn generation(self) -> u32 {
        self.generation
    }

    #[inline]
    pub(crate) fn runtime_id(self) -> RuntimeId {
        self.runtime_id
    }
}

/// Typed handle to a node in a `Runtime<C>`.
#[repr(C)]
pub struct Incr<T: 'static> {
    id: ErasedIncr,
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
            .field("slot", &self.id.slot)
            .field("generation", &self.id.generation)
            .field("runtime_id", &self.id.runtime_id)
            .field("type", &std::any::type_name::<T>())
            .finish()
    }
}

impl<T: 'static> PartialEq for Incr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: 'static> Eq for Incr<T> {}

impl<T: 'static> std::hash::Hash for Incr<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T: 'static> Incr<T> {
    pub(crate) fn new(slot: u32, generation: u32, runtime_id: RuntimeId) -> Self {
        Self {
            id: ErasedIncr::new(slot, generation, runtime_id),
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn slot(self) -> u32 {
        self.id.slot
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn generation(self) -> u32 {
        self.id.generation
    }

    #[inline]
    pub(crate) fn runtime_id(self) -> RuntimeId {
        self.id.runtime_id
    }

    /// Drop the value type, keeping the node's identity.
    #[inline]
    pub fn erase(self) -> ErasedIncr {
        self.id
    }
}

impl<T: 'static> From<Incr<T>> for ErasedIncr {
    #[inline]
    fn from(handle: Incr<T>) -> Self {
        handle.id
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
    fn erased_incr_is_a_16_byte_copy_send_sync_token() {
        fn assert_send_sync_copy<T: Send + Sync + Copy>() {}
        assert_send_sync_copy::<ErasedIncr>();
        assert_eq!(std::mem::size_of::<ErasedIncr>(), 16);
    }

    #[test]
    fn erase_and_from_agree_and_keep_identity() {
        let rt = RuntimeId::allocate();
        let a: Incr<String> = Incr::new(3, 7, rt);
        let b: Incr<u64> = Incr::new(3, 8, rt);
        let ea = a.erase();
        assert_eq!(ea, ErasedIncr::from(a));
        assert_eq!(ea.slot(), 3);
        assert_eq!(ea.generation(), 7);
        assert_eq!(ea.runtime_id(), rt);
        assert_ne!(ea, b.erase(), "a different generation is a different node");
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
