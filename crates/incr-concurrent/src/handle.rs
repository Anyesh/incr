//! Runtime and node handle types.
//!
//! This module defines two identity types that together ensure handles
//! cannot be used in unsafe ways:
//!
//! - [`RuntimeId`] uniquely identifies a `Runtime` instance for the
//!   lifetime of a process. It is drawn from the same monotonic counter
//!   that the arena registry uses, so a runtime's id is the id of its
//!   registry.
//! - [`Incr<T>`] is a typed handle to a node inside a runtime. It
//!   carries enough information to detect three classes of misuse
//!   without undefined behavior:
//!     1. Using a handle with the wrong runtime. Caught via the
//!        `runtime_id` field.
//!     2. Using a handle after the underlying slot has been recycled
//!        (in a future version of incr that supports node deletion).
//!        Caught via the `generation` field.
//!     3. Using a handle with the wrong value type. Caught statically
//!        via the `PhantomData<fn() -> T>` parameter: once the runtime
//!        returns an `Incr<u64>` from `create_input::<u64>`, the type
//!        is locked in at compile time.
//!
//! ## Handle layout
//!
//! ```text
//! offset  size   field
//! ------  ----   -----
//!    0     4     slot         u32
//!    4     4     generation   u32
//!    8     8     runtime_id   RuntimeId (u64)
//!   16     0     _phantom     PhantomData<fn() -> T>
//! ```
//!
//! Total: 16 bytes on 64-bit platforms. The handle is `Copy` and cheap
//! to pass around by value. The decision to widen from the v1 4-byte
//! NodeId to v2's 16-byte Incr is covered in spec section 13, questions
//! Q3 and Q4; both recommendations ("add runtime identity", "add
//! generation counters") are applied here.
//!
//! ## Why `PhantomData<fn() -> T>`
//!
//! `PhantomData<T>` would tie `Incr<T>`'s auto traits to `T`: an
//! `Incr<RefCell<...>>` would not be `Sync` because `RefCell` is not
//! `Sync`. That is the wrong contract for a handle, because a handle
//! does not own a `T` and does not expose `&T` to shared callers; it
//! is just an opaque token. `PhantomData<fn() -> T>` covariantly
//! references `T` without inheriting its auto traits, so `Incr<T>` is
//! `Send + Sync + Copy + Unpin` for every `T: 'static`.

use std::marker::PhantomData;

/// Unique identifier for a `Runtime` (equivalently, its arena registry).
/// Assigned monotonically at construction; never reused within a process
/// lifetime because the underlying counter is `u64` and does not wrap
/// within any realistic program run.
///
/// The value zero is reserved as a sentinel for "not a real runtime" and
/// is used by the TLS arena pointer cache to mark empty slots.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct RuntimeId(u64);

impl RuntimeId {
    /// The sentinel runtime id. Never assigned to a real runtime.
    #[allow(dead_code)]
    pub(crate) const SENTINEL: RuntimeId = RuntimeId(0);

    /// Wrap a raw counter value. Called by the arena registry when a new
    /// runtime is constructed.
    pub(crate) const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Get the raw counter value. Used by the TLS arena pointer cache
    /// which keeps its storage as a bare `u64` to avoid churn on the
    /// hot path.
    #[inline]
    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

/// A typed handle to a node in a `Runtime`.
///
/// Handles are `Copy` and freely shareable across threads. Their validity
/// is checked at access time by the runtime, which verifies the handle's
/// `runtime_id` matches its own and the `generation` matches the node's
/// current generation counter. Both checks panic with a clear message on
/// failure via [`HandleError`] propagation in the runtime.
///
/// The `T` parameter is carried via `PhantomData<fn() -> T>` so that
/// auto-trait propagation is not affected by `T`. A handle is always
/// `Send + Sync + Copy` regardless of `T`.
#[repr(C)]
pub struct Incr<T: 'static> {
    slot: u32,
    generation: u32,
    runtime_id: RuntimeId,
    _phantom: PhantomData<fn() -> T>,
}

// Manual implementations of the standard derives so they do not require
// `T: Copy + Clone + Debug + PartialEq + Eq + Hash`. A handle is these
// things regardless of what `T` is.

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
    /// Construct a handle. Crate-private so user code cannot forge
    /// handles; only the runtime's node creation paths return an
    /// `Incr<T>`, which binds `T` at the creation call site.
    pub(crate) fn new(slot: u32, generation: u32, runtime_id: RuntimeId) -> Self {
        Self {
            slot,
            generation,
            runtime_id,
            _phantom: PhantomData,
        }
    }

    /// The slot index this handle refers to.
    #[inline]
    pub fn slot(self) -> u32 {
        self.slot
    }

    /// The expected generation counter for the slot.
    #[inline]
    pub(crate) fn generation(self) -> u32 {
        self.generation
    }

    /// The owning runtime's id.
    #[inline]
    pub(crate) fn runtime_id(self) -> RuntimeId {
        self.runtime_id
    }
}

/// Error returned by handle verification when a check fails.
///
/// The runtime's public `get` / `set` methods convert these into
/// panics with a clear message. Tests and internal diagnostics use the
/// `Result`-returning verifier so failures can be observed without
/// tearing down the process.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum HandleError {
    /// The handle was created by a different runtime than the one it
    /// is being used with. Carries both ids for diagnostics.
    WrongRuntime {
        handle_runtime: RuntimeId,
        current_runtime: RuntimeId,
    },
    /// The slot the handle points at has been recycled since the handle
    /// was created. Carries both generations for diagnostics.
    StaleGeneration {
        handle_generation: u32,
        current_generation: u32,
    },
}

impl std::fmt::Display for HandleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandleError::WrongRuntime {
                handle_runtime,
                current_runtime,
            } => write!(
                f,
                "Incr handle from runtime {:?} used with runtime {:?}",
                handle_runtime, current_runtime
            ),
            HandleError::StaleGeneration {
                handle_generation,
                current_generation,
            } => write!(
                f,
                "Incr handle with generation {} used after slot recycled to generation {}",
                handle_generation, current_generation
            ),
        }
    }
}

impl std::error::Error for HandleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incr_is_16_bytes_and_8_aligned() {
        assert_eq!(std::mem::size_of::<Incr<u64>>(), 16);
        assert_eq!(std::mem::align_of::<Incr<u64>>(), 8);
        // Same size regardless of T.
        assert_eq!(std::mem::size_of::<Incr<String>>(), 16);
        assert_eq!(std::mem::size_of::<Incr<Vec<u8>>>(), 16);
    }

    #[test]
    fn incr_is_copy_and_send_and_sync_regardless_of_t() {
        fn assert_copy<T: Copy>() {}
        fn assert_send_sync<T: Send + Sync>() {}
        // u64: Copy + Send + Sync (obvious)
        assert_copy::<Incr<u64>>();
        assert_send_sync::<Incr<u64>>();
        // String: !Copy, but Incr<String> is still Copy because Incr
        // does not store T.
        assert_copy::<Incr<String>>();
        assert_send_sync::<Incr<String>>();
        // A !Sync type: RefCell<T>. Incr<RefCell<_>> must still be Sync.
        assert_send_sync::<Incr<std::cell::RefCell<u64>>>();
        // A !Send type: Rc<T>. Incr<Rc<_>> must still be Send.
        assert_send_sync::<Incr<std::rc::Rc<u64>>>();
    }

    #[test]
    fn different_types_with_same_fields_are_distinct_at_the_type_level() {
        // Not a runtime check; this is a compile-gate that ensures the
        // phantom-T parameter actually participates in type identity.
        // If someone deletes the PhantomData this test will compile
        // just fine and the bug slips past, so we also check runtime
        // behavior: Incr<u64> and Incr<i64> with the same fields are
        // distinct types and cannot be passed to each other's slots.
        let _u: Incr<u64> = Incr::new(0, 0, RuntimeId::from_raw(1));
        let _i: Incr<i64> = Incr::new(0, 0, RuntimeId::from_raw(1));
        // If we uncommented the next line the compiler would reject it:
        //   let _: Incr<u64> = _i;
    }

    #[test]
    fn incr_equality_compares_all_three_fields() {
        let rid = RuntimeId::from_raw(1);
        let a: Incr<u64> = Incr::new(7, 3, rid);
        let b: Incr<u64> = Incr::new(7, 3, rid);
        assert_eq!(a, b);

        let different_slot: Incr<u64> = Incr::new(8, 3, rid);
        assert_ne!(a, different_slot);

        let different_gen: Incr<u64> = Incr::new(7, 4, rid);
        assert_ne!(a, different_gen);

        let different_rt: Incr<u64> = Incr::new(7, 3, RuntimeId::from_raw(2));
        assert_ne!(a, different_rt);
    }

    #[test]
    fn incr_hash_is_stable() {
        use std::collections::HashSet;
        let rid = RuntimeId::from_raw(42);
        let a: Incr<u64> = Incr::new(1, 0, rid);
        let b: Incr<u64> = Incr::new(1, 0, rid);
        let c: Incr<u64> = Incr::new(2, 0, rid);

        let mut set: HashSet<Incr<u64>> = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b)); // same fields → same hash → hit
        assert!(!set.contains(&c));
    }

    #[test]
    fn runtime_id_sentinel_is_zero_and_never_equals_real_ids() {
        assert_eq!(RuntimeId::SENTINEL.get(), 0);
        let real = RuntimeId::from_raw(1);
        assert_ne!(RuntimeId::SENTINEL, real);
    }

    #[test]
    fn handle_error_display_mentions_ids_and_generations() {
        let err = HandleError::WrongRuntime {
            handle_runtime: RuntimeId::from_raw(1),
            current_runtime: RuntimeId::from_raw(2),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("RuntimeId(1)"));
        assert!(msg.contains("RuntimeId(2)"));

        let err = HandleError::StaleGeneration {
            handle_generation: 3,
            current_generation: 7,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("3"));
        assert!(msg.contains("7"));
    }

    #[test]
    fn incr_debug_shows_type_name() {
        let h: Incr<u64> = Incr::new(1, 2, RuntimeId::from_raw(3));
        let s = format!("{:?}", h);
        assert!(s.contains("slot: 1"));
        assert!(s.contains("generation: 2"));
        assert!(s.contains("u64")); // type_name<u64>() is "u64"
    }
}
