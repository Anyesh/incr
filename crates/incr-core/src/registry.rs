//! Arena registry: the runtime's `TypeId → arena` lookup table.
//!
//! Per section 5.2 of the concurrent core rewrite spec, the runtime holds
//! one arena per value type, indexed by [`TypeId`]. This module is that
//! index.
//!
//! ## Design
//!
//! The registry is a `HashMap<TypeId, Box<dyn ErasedArena>>` behind a
//! `RwLock`. Readers take a short-lived read guard, look up their type's
//! arena, extract a raw pointer to the arena, and drop the guard. The
//! pointer is stable for the registry's lifetime because:
//!
//! 1. Arenas are never removed from the registry. The registry is
//!    append-only; adding a new value type inserts a new entry but no
//!    entry is ever deleted, even when nodes holding that type are
//!    destroyed. (Nodes are recycled via generation counters at the
//!    slot level; the arena itself lives on.)
//! 2. Each arena lives behind a `Box`, which pins it at a stable heap
//!    address. `HashMap` resizing moves the `Box` (two words) but not
//!    the arena interior the `Box` points to.
//!
//! Therefore `*const dyn ErasedArena` extracted from
//! `box.as_ref() as *const _` remains valid as long as the registry is
//! alive. Callers who cache pointers across operations get the same
//! correctness guarantee.
//!
//! ## Concurrency and the TLS pointer cache
//!
//! Benchmarks (see `mod bench` in this file) measured the naive
//! `RwLock<HashMap>` lookup at approximately 26 ns per call in release
//! mode. The spec's budget for a single-threaded Clean `get` is 5 ns,
//! and the registry is only one step of a full get. The naive path
//! blows the budget on its own, so the thread-local pointer cache
//! described in spec section 5.2 is load-bearing rather than optional.
//!
//! The cache lives in a `thread_local!` `RefCell<ArenaCache>` with four
//! slots. Each slot holds `(registry_id, type_id, arena_ptr)`. On a
//! lookup the cache does a linear scan over its entries; a hit avoids
//! both the `RwLock::read` and the `HashMap` lookup. The hit path is
//! read-only (no move-to-front) so that repeated hits do not dirty the
//! cache line. Misses fall through to the lock-backed lookup and
//! populate the cache via round-robin eviction.
//!
//! ## Registry identity and the ABA question
//!
//! The cache is keyed by `(registry_id, type_id)` where `registry_id`
//! is a monotonic `u64` drawn from a static counter at `ArenaRegistry`
//! construction. The counter never wraps in any realistic program
//! lifetime, so a cached entry from a dropped registry can never be
//! confused with a different registry that happens to be allocated at
//! the same address. Stale entries are inert: their `registry_id` will
//! never match a live registry's id, so lookups miss and eventually
//! overwrite the stale slot via round-robin eviction.
//!
//! When `RuntimeId` lands in commit E, the registry's id becomes the
//! runtime's id (they are the same quantity), and the `(RuntimeId,
//! TypeId)` pair the spec describes is exactly what the cache is
//! already keyed by.

use std::any::TypeId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use super::arena::ErasedArena;
use super::handle::RuntimeId;

/// Source of monotonic registry ids. Starts at 1; 0 is reserved as the
/// "never assigned" sentinel used by the TLS cache to mean "empty slot
/// that can never match a live registry."
static NEXT_REGISTRY_ID: AtomicU64 = AtomicU64::new(1);

/// Append-only registry of arenas keyed by value type.
pub(crate) struct ArenaRegistry {
    /// Monotonic id that uniquely identifies this registry for its
    /// lifetime and across all registries ever constructed in this
    /// process. Also serves as the owning runtime's `RuntimeId`: a
    /// runtime adopts its registry's id rather than tracking its own,
    /// so the TLS cache and `Incr<T>` handles can share a single
    /// identity concept. Used by the TLS cache to distinguish entries
    /// belonging to this registry from entries belonging to
    /// dropped-or-other registries; see the module docs.
    id: RuntimeId,
    arenas: RwLock<HashMap<TypeId, Box<dyn ErasedArena>>>,
}

impl ArenaRegistry {
    /// Create an empty registry with a fresh monotonic id.
    pub(crate) fn new() -> Self {
        Self {
            id: RuntimeId::from_raw(NEXT_REGISTRY_ID.fetch_add(1, Ordering::Relaxed)),
            arenas: RwLock::new(HashMap::new()),
        }
    }

    /// Return this registry's unique id. Exposed so the Runtime can
    /// adopt it as its own `RuntimeId` and so `Incr<T>` handles can
    /// carry it for cross-runtime detection.
    pub(crate) fn id(&self) -> RuntimeId {
        self.id
    }

    /// Ensure an arena exists for type `T`, constructing it via `factory`
    /// if this is the first time the registry has seen `T`.
    ///
    /// Returns a raw pointer to the arena, valid for the registry's
    /// lifetime. Callers downcast to the concrete arena type via
    /// [`ErasedArena::as_any`].
    ///
    /// This is the canonical entry point for node creation paths that
    /// need an arena but do not know whether one has been created for
    /// their type yet.
    pub(crate) fn ensure_arena<T: 'static, F>(&self, factory: F) -> *const dyn ErasedArena
    where
        F: FnOnce() -> Box<dyn ErasedArena>,
    {
        let tid = TypeId::of::<T>();

        // Hottest path: TLS cache hit.
        if let Some(ptr) = cache_lookup(self.id.get(), tid) {
            return ptr;
        }

        // Cache miss. Fall through to the read-locked registry lookup.
        {
            let guard = self
                .arenas
                .read()
                .expect("arena registry read lock poisoned");
            if let Some(entry) = guard.get(&tid) {
                let ptr = entry.as_ref() as *const dyn ErasedArena;
                cache_insert(self.id.get(), tid, ptr);
                return ptr;
            }
        }

        // Not in the registry either: write-locked insertion. Double-
        // checked under the write lock in case another thread inserted
        // between our read and write. `or_insert_with` does the check.
        let mut guard = self
            .arenas
            .write()
            .expect("arena registry write lock poisoned");
        let entry = guard.entry(tid).or_insert_with(factory);
        let ptr = entry.as_ref() as *const dyn ErasedArena;
        cache_insert(self.id.get(), tid, ptr);
        ptr
    }

    /// Look up an existing arena for `T`. Returns `None` if no arena
    /// for `T` has been created yet. Intended for read-only paths that
    /// should not trigger lazy creation (e.g., diagnostics, sanity
    /// checks); production get/set paths should use
    /// [`ArenaRegistry::ensure_arena`] so a missing arena is a bug,
    /// not a silent `None`.
    pub(crate) fn lookup<T: 'static>(&self) -> Option<*const dyn ErasedArena> {
        let tid = TypeId::of::<T>();

        // Hottest path: TLS cache hit.
        if let Some(ptr) = cache_lookup(self.id.get(), tid) {
            return Some(ptr);
        }

        // Cache miss: read-locked registry lookup.
        let ptr = {
            let guard = self
                .arenas
                .read()
                .expect("arena registry read lock poisoned");
            guard
                .get(&tid)
                .map(|entry| entry.as_ref() as *const dyn ErasedArena)
        }?;
        cache_insert(self.id.get(), tid, ptr);
        Some(ptr)
    }

    /// Number of distinct value types the registry currently holds.
    /// Used by tests and potential diagnostics.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.arenas
            .read()
            .expect("arena registry read lock poisoned")
            .len()
    }
}

impl Default for ArenaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: `ArenaRegistry` owns its `RwLock<HashMap<...>>` directly and
// does not expose internal state across thread boundaries in ways that
// would violate `Send`/`Sync`. `Box<dyn ErasedArena>` is `Send + Sync`
// because `ErasedArena: Send + Sync`. `RwLock` provides the necessary
// synchronization for the HashMap.
//
// The raw pointers returned by `ensure_arena` and `lookup` are not tied
// to the lock guard; callers observe arena contents directly. Concurrent
// readers of an arena coordinate via the node state machine, which is
// orthogonal to the registry-level locking.

// ---------------------------------------------------------------------------
// Thread-local arena pointer cache.
// ---------------------------------------------------------------------------

/// Number of slots in the TLS cache. Four is a sweet spot: almost all
/// workloads touch one to four value types on the hot path, and a linear
/// scan over four entries is ~1 ns even with the array-of-Option branch
/// overhead. Going wider (8, 16) pays more on every scan for a rare
/// benefit; going narrower (2) thrashes for workloads with three or four
/// hot types.
const CACHE_SLOTS: usize = 4;

/// One TLS cache entry. A registry id of zero means "empty slot" and
/// never matches any live registry (live registry ids start at 1).
#[derive(Copy, Clone)]
struct CacheEntry {
    registry_id: u64,
    type_id: TypeId,
    arena_ptr: *const dyn ErasedArena,
}

impl CacheEntry {
    const fn empty() -> Self {
        // SAFETY note: this sentinel pointer is never dereferenced. The
        // `registry_id: 0` acts as a guard: any real lookup compares
        // against a live registry id (which is nonzero), so this entry
        // cannot match and its pointer cannot be used.
        Self {
            registry_id: 0,
            type_id: TypeId::of::<()>(),
            arena_ptr: std::ptr::null::<EmptyErasedArena>() as *const dyn ErasedArena,
        }
    }
}

/// Placeholder type used only to give the empty cache entry's pointer a
/// concrete non-generic form for the `null::<_> as *const dyn Trait`
/// coercion. This type is never instantiated.
struct EmptyErasedArena;
impl ErasedArena for EmptyErasedArena {
    fn erased_type_id(&self) -> TypeId {
        TypeId::of::<()>()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The TLS cache itself: four slots plus a round-robin eviction cursor.
/// The hit path is read-only (no move-to-front) so repeated lookups of
/// the same type do not dirty the cache line.
struct ArenaCache {
    entries: [CacheEntry; CACHE_SLOTS],
    next_eviction: u32,
}

impl ArenaCache {
    const fn new() -> Self {
        Self {
            entries: [CacheEntry::empty(); CACHE_SLOTS],
            next_eviction: 0,
        }
    }
}

thread_local! {
    /// Per-thread arena pointer cache. `RefCell` lets the lookup path
    /// take a short `borrow_mut` only on cache insertion; the hit path
    /// uses `borrow` (counter increment only, no work on the hot path
    /// beyond the linear scan). In release builds the borrow counters
    /// are the dominant cost after the scan itself.
    static ARENA_CACHE: RefCell<ArenaCache> = const { RefCell::new(ArenaCache::new()) };
}

/// Look up `(registry_id, type_id)` in this thread's cache. Returns the
/// cached pointer on hit, `None` on miss. The hit path does not mutate
/// the cache (no move-to-front), so repeated hits stay cheap.
fn cache_lookup(registry_id: u64, type_id: TypeId) -> Option<*const dyn ErasedArena> {
    ARENA_CACHE.with(|cache| {
        let cache = cache.borrow();
        for entry in &cache.entries {
            if entry.registry_id == registry_id && entry.type_id == type_id {
                return Some(entry.arena_ptr);
            }
        }
        None
    })
}

/// Insert a `(registry_id, type_id) -> arena_ptr` mapping into this
/// thread's cache. Uses round-robin eviction: each insertion overwrites
/// the slot at `next_eviction` and advances the cursor.
fn cache_insert(registry_id: u64, type_id: TypeId, arena_ptr: *const dyn ErasedArena) {
    ARENA_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        // If an entry for this (registry, type) already exists, update
        // it in place rather than duplicating. Two copies would not
        // violate correctness but would waste a slot.
        for entry in cache.entries.iter_mut() {
            if entry.registry_id == registry_id && entry.type_id == type_id {
                entry.arena_ptr = arena_ptr;
                return;
            }
        }
        let idx = (cache.next_eviction as usize) % CACHE_SLOTS;
        cache.entries[idx] = CacheEntry {
            registry_id,
            type_id,
            arena_ptr,
        };
        cache.next_eviction = cache.next_eviction.wrapping_add(1);
    });
}

/// Clear the current thread's cache. Used only by tests that want to
/// observe the uncached fallback path in isolation.
#[cfg(test)]
fn cache_clear() {
    ARENA_CACHE.with(|cache| {
        *cache.borrow_mut() = ArenaCache::new();
    });
}

#[cfg(test)]
mod tests {
    use super::super::arena::{AtomicPrimitiveArena, GenericArena};
    use super::*;
    use std::sync::Arc;
    use std::thread;

    /// Downcast helper for tests: turn a `*const dyn ErasedArena` into
    /// a typed reference to a concrete arena.
    unsafe fn as_primitive<'a, T: super::super::arena::AtomicPrimitive>(
        ptr: *const dyn ErasedArena,
    ) -> &'a AtomicPrimitiveArena<T> {
        (*ptr)
            .as_any()
            .downcast_ref::<AtomicPrimitiveArena<T>>()
            .expect("arena type mismatch")
    }

    unsafe fn as_generic<'a, T: Clone + Send + Sync + 'static>(
        ptr: *const dyn ErasedArena,
    ) -> &'a GenericArena<T> {
        (*ptr)
            .as_any()
            .downcast_ref::<GenericArena<T>>()
            .expect("arena type mismatch")
    }

    #[test]
    fn new_registry_is_empty() {
        let registry = ArenaRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.lookup::<u64>().is_none());
    }

    #[test]
    fn ensure_arena_creates_once_per_type() {
        let registry = ArenaRegistry::new();
        let p1 = registry.ensure_arena::<u64, _>(|| Box::new(AtomicPrimitiveArena::<u64>::new()));
        let p2 = registry.ensure_arena::<u64, _>(|| {
            panic!("factory should not run on second call for same type")
        });
        // Fat pointer equality via pointer comparison: both should point
        // to the same arena instance.
        assert_eq!(
            p1 as *const (), p2 as *const (),
            "ensure_arena must return the same pointer for the same type"
        );
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn ensure_arena_creates_distinct_arenas_per_type() {
        let registry = ArenaRegistry::new();
        let p_u64 =
            registry.ensure_arena::<u64, _>(|| Box::new(AtomicPrimitiveArena::<u64>::new()));
        let p_i32 =
            registry.ensure_arena::<i32, _>(|| Box::new(AtomicPrimitiveArena::<i32>::new()));
        let p_string =
            registry.ensure_arena::<String, _>(|| Box::new(GenericArena::<String>::new()));
        assert_ne!(p_u64 as *const (), p_i32 as *const ());
        assert_ne!(p_u64 as *const (), p_string as *const ());
        assert_ne!(p_i32 as *const (), p_string as *const ());
        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn lookup_returns_none_before_creation_and_some_after() {
        let registry = ArenaRegistry::new();
        assert!(registry.lookup::<u64>().is_none());
        registry.ensure_arena::<u64, _>(|| Box::new(AtomicPrimitiveArena::<u64>::new()));
        assert!(registry.lookup::<u64>().is_some());
    }

    #[test]
    fn downcast_through_registry_returns_usable_arena() {
        let registry = ArenaRegistry::new();
        let ptr = registry.ensure_arena::<u64, _>(|| Box::new(AtomicPrimitiveArena::<u64>::new()));
        // SAFETY: we just created this arena as AtomicPrimitiveArena<u64>.
        let arena = unsafe { as_primitive::<u64>(ptr) };
        let slot = arena.reserve(777);
        assert_eq!(arena.read(slot), 777);

        // Confirm a second ensure_arena call returns a pointer to the
        // same arena and the slot is still there.
        let ptr2 = registry.ensure_arena::<u64, _>(|| panic!("must not recreate"));
        let arena2 = unsafe { as_primitive::<u64>(ptr2) };
        assert_eq!(arena2.read(slot), 777);
    }

    #[test]
    fn downcast_through_registry_for_generic_type() {
        let registry = ArenaRegistry::new();
        let ptr = registry.ensure_arena::<Vec<u8>, _>(|| Box::new(GenericArena::<Vec<u8>>::new()));
        // SAFETY: we just created this arena as GenericArena<Vec<u8>>.
        let arena = unsafe { as_generic::<Vec<u8>>(ptr) };
        let slot = arena.reserve_with(vec![1, 2, 3]);
        assert_eq!(arena.read(slot), vec![1, 2, 3]);
    }

    #[test]
    fn pointer_remains_valid_across_later_insertions() {
        // After inserting u64, we hold its pointer. Inserting more types
        // may cause HashMap resizing, but the u64 arena's Box lives on
        // the heap and does not move.
        let registry = ArenaRegistry::new();
        let p_u64 =
            registry.ensure_arena::<u64, _>(|| Box::new(AtomicPrimitiveArena::<u64>::new()));
        // SAFETY: valid because we just inserted and hold registry.
        let arena_u64 = unsafe { as_primitive::<u64>(p_u64) };
        let slot = arena_u64.reserve(42);
        assert_eq!(arena_u64.read(slot), 42);

        // Trigger many insertions to force HashMap rehashes. Each
        // "unique type" here uses a fresh marker struct via monomorphic
        // instantiation in a helper; here we use distinct primitive
        // arena types plus a generic one per iteration.
        registry.ensure_arena::<i32, _>(|| Box::new(AtomicPrimitiveArena::<i32>::new()));
        registry.ensure_arena::<u32, _>(|| Box::new(AtomicPrimitiveArena::<u32>::new()));
        registry.ensure_arena::<i64, _>(|| Box::new(AtomicPrimitiveArena::<i64>::new()));
        registry.ensure_arena::<f32, _>(|| Box::new(AtomicPrimitiveArena::<f32>::new()));
        registry.ensure_arena::<f64, _>(|| Box::new(AtomicPrimitiveArena::<f64>::new()));
        registry.ensure_arena::<bool, _>(|| Box::new(AtomicPrimitiveArena::<bool>::new()));
        registry.ensure_arena::<String, _>(|| Box::new(GenericArena::<String>::new()));
        registry.ensure_arena::<Vec<u8>, _>(|| Box::new(GenericArena::<Vec<u8>>::new()));

        // The original u64 pointer must still resolve to the same arena
        // holding the same slot value.
        // SAFETY: arenas are never removed; pointer is stable.
        let arena_u64_again = unsafe { as_primitive::<u64>(p_u64) };
        assert_eq!(arena_u64_again.read(slot), 42);
    }

    /// Collapse a fat `*const dyn ErasedArena` to its data-pointer
    /// address. We use this only for identity comparisons in tests; the
    /// vtable half is discarded so the result is a plain `usize` that
    /// can cross thread boundaries. Never dereferenced.
    fn data_addr(ptr: *const dyn ErasedArena) -> usize {
        ptr as *const () as usize
    }

    #[test]
    fn concurrent_ensure_arena_returns_a_single_arena() {
        // Many threads race to ensure an arena for the same type. Only
        // one factory invocation should succeed, and all threads should
        // receive pointers with the same data address.
        const THREADS: usize = 16;

        let registry = Arc::new(ArenaRegistry::new());
        let factory_invocations = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let registry = registry.clone();
                let counter = factory_invocations.clone();
                thread::spawn(move || {
                    let ptr = registry.ensure_arena::<u64, _>(|| {
                        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        Box::new(AtomicPrimitiveArena::<u64>::new())
                    });
                    data_addr(ptr)
                })
            })
            .collect();

        let addrs: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let first = addrs[0];
        for a in &addrs {
            assert_eq!(*a, first, "all threads must see the same arena pointer");
        }
        assert_eq!(
            factory_invocations.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "factory should run exactly once under concurrent ensure_arena"
        );
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn concurrent_ensure_arena_for_distinct_types_is_independent() {
        // Multiple threads each inserting a different type should all
        // succeed without interfering.
        let registry = Arc::new(ArenaRegistry::new());

        let h1 = {
            let registry = registry.clone();
            thread::spawn(move || {
                data_addr(
                    registry
                        .ensure_arena::<u64, _>(|| Box::new(AtomicPrimitiveArena::<u64>::new())),
                )
            })
        };
        let h2 = {
            let registry = registry.clone();
            thread::spawn(move || {
                data_addr(
                    registry
                        .ensure_arena::<i32, _>(|| Box::new(AtomicPrimitiveArena::<i32>::new())),
                )
            })
        };
        let h3 = {
            let registry = registry.clone();
            thread::spawn(move || {
                data_addr(
                    registry.ensure_arena::<String, _>(|| Box::new(GenericArena::<String>::new())),
                )
            })
        };

        let a1 = h1.join().unwrap();
        let a2 = h2.join().unwrap();
        let a3 = h3.join().unwrap();
        assert_ne!(a1, a2);
        assert_ne!(a1, a3);
        assert_ne!(a2, a3);
        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn tls_cache_hit_returns_same_pointer_as_uncached() {
        // The first lookup populates the cache; the second lookup must
        // return the same pointer. This is the happy-path correctness
        // check for the cache.
        super::cache_clear();
        let registry = ArenaRegistry::new();
        let p1 = registry.ensure_arena::<u64, _>(|| Box::new(AtomicPrimitiveArena::<u64>::new()));
        // Second call hits the cache (factory must not run).
        let p2 = registry.ensure_arena::<u64, _>(|| panic!("factory ran on cache hit"));
        assert_eq!(data_addr(p1), data_addr(p2));
        // And via lookup().
        let p3 = registry.lookup::<u64>().expect("arena exists");
        assert_eq!(data_addr(p1), data_addr(p3));
    }

    #[test]
    fn tls_cache_distinguishes_between_registries() {
        // Two registries each have their own u64 arena. After touching
        // both from the same thread, the cache should hold entries for
        // both and each lookup should return its own registry's arena.
        // This is the correctness check that keys the cache by
        // (registry_id, type_id), not just type_id.
        super::cache_clear();
        let reg_a = ArenaRegistry::new();
        let reg_b = ArenaRegistry::new();
        assert_ne!(reg_a.id(), reg_b.id());

        let a = reg_a.ensure_arena::<u64, _>(|| Box::new(AtomicPrimitiveArena::<u64>::new()));
        let b = reg_b.ensure_arena::<u64, _>(|| Box::new(AtomicPrimitiveArena::<u64>::new()));
        assert_ne!(data_addr(a), data_addr(b));

        // Populate the u64 arenas with distinct values via their typed
        // references, so subsequent lookups can be verified to return
        // the right arena and not the wrong one.
        let arena_a = unsafe { as_primitive::<u64>(a) };
        let arena_b = unsafe { as_primitive::<u64>(b) };
        let slot_a = arena_a.reserve(111);
        let slot_b = arena_b.reserve(222);

        // Interleaved lookups: both should still go to the right place.
        for _ in 0..10 {
            let got_a = reg_a.ensure_arena::<u64, _>(|| panic!("factory ran"));
            let got_b = reg_b.ensure_arena::<u64, _>(|| panic!("factory ran"));
            assert_eq!(data_addr(got_a), data_addr(a));
            assert_eq!(data_addr(got_b), data_addr(b));
            let arena_a_again = unsafe { as_primitive::<u64>(got_a) };
            let arena_b_again = unsafe { as_primitive::<u64>(got_b) };
            assert_eq!(arena_a_again.read(slot_a), 111);
            assert_eq!(arena_b_again.read(slot_b), 222);
        }
    }

    #[test]
    fn tls_cache_round_robin_eviction_over_five_types() {
        // With CACHE_SLOTS = 4, touching five distinct types evicts the
        // oldest on the fifth insertion. Subsequent lookups on all five
        // still succeed via the lock-backed fallback and re-enter the
        // cache. The test asserts correctness, not the specific
        // eviction victim, since round-robin order is an implementation
        // detail.
        super::cache_clear();
        let registry = ArenaRegistry::new();
        let a = registry.ensure_arena::<u64, _>(|| Box::new(AtomicPrimitiveArena::<u64>::new()));
        let b = registry.ensure_arena::<i32, _>(|| Box::new(AtomicPrimitiveArena::<i32>::new()));
        let c = registry.ensure_arena::<u32, _>(|| Box::new(AtomicPrimitiveArena::<u32>::new()));
        let d = registry.ensure_arena::<i64, _>(|| Box::new(AtomicPrimitiveArena::<i64>::new()));
        let e = registry.ensure_arena::<f64, _>(|| Box::new(AtomicPrimitiveArena::<f64>::new()));

        // All five should still lookup cleanly. None should trigger the
        // factory (they are all in the registry), and all pointers
        // should match the originals.
        assert_eq!(
            data_addr(registry.ensure_arena::<u64, _>(|| panic!("re-created u64"))),
            data_addr(a)
        );
        assert_eq!(
            data_addr(registry.ensure_arena::<i32, _>(|| panic!("re-created i32"))),
            data_addr(b)
        );
        assert_eq!(
            data_addr(registry.ensure_arena::<u32, _>(|| panic!("re-created u32"))),
            data_addr(c)
        );
        assert_eq!(
            data_addr(registry.ensure_arena::<i64, _>(|| panic!("re-created i64"))),
            data_addr(d)
        );
        assert_eq!(
            data_addr(registry.ensure_arena::<f64, _>(|| panic!("re-created f64"))),
            data_addr(e)
        );
    }
}

/// Microbenchmarks for the registry lookup cost.
///
/// These are `#[ignore]` tests rather than criterion benches because the
/// concrete arena types and the registry API are `pub(crate)` during the
/// v2 rewrite. External benches in `benches/` cannot see them. This
/// tradeoff keeps the v2 module visibility honest (nothing is exposed
/// publicly before Gate 5) while still letting us measure the lookup
/// cost against the spec's budget.
///
/// Run with:
///
/// ```text
/// cargo test --release -p incr-core v2::registry::bench -- --ignored --nocapture
/// ```
///
/// The output reports nanoseconds per call for each variant.
#[cfg(test)]
mod bench {
    use super::super::arena::AtomicPrimitiveArena;
    use super::ArenaRegistry;
    use std::hint::black_box;
    use std::time::Instant;

    /// How many iterations per measurement. Large enough to drive noise
    /// below a nanosecond when compiled with `--release`.
    const ITERS: usize = 20_000_000;

    #[test]
    #[ignore = "microbenchmark, run with --release"]
    fn bench_ensure_arena_hot_path() {
        // Hot path: `ensure_arena::<T>` after the first insertion. Takes
        // the read lock and hits the HashMap. This is the cost paid by
        // every `rt.get` for a type whose arena has been seen before.
        let registry = ArenaRegistry::new();
        registry.ensure_arena::<u64, _>(|| Box::new(AtomicPrimitiveArena::<u64>::new()));

        let start = Instant::now();
        let mut acc: usize = 0;
        for _ in 0..ITERS {
            let ptr = registry.ensure_arena::<u64, _>(|| unreachable!());
            acc = acc.wrapping_add(ptr as *const () as usize);
        }
        let elapsed = start.elapsed();
        black_box(acc);
        let per_call = elapsed.as_nanos() as f64 / ITERS as f64;
        eprintln!(
            "ensure_arena hot path (read lock + hashmap lookup): {:.2} ns/call",
            per_call
        );
    }

    #[test]
    #[ignore = "microbenchmark, run with --release"]
    fn bench_lookup_hot_path() {
        // Hot path via the simpler `lookup` method. Should match
        // `ensure_arena` hot path because both take the read lock and
        // both do a single hashmap lookup.
        let registry = ArenaRegistry::new();
        registry.ensure_arena::<u64, _>(|| Box::new(AtomicPrimitiveArena::<u64>::new()));

        let start = Instant::now();
        let mut acc: usize = 0;
        for _ in 0..ITERS {
            let ptr = registry.lookup::<u64>().expect("arena exists");
            acc = acc.wrapping_add(ptr as *const () as usize);
        }
        let elapsed = start.elapsed();
        black_box(acc);
        let per_call = elapsed.as_nanos() as f64 / ITERS as f64;
        eprintln!("lookup hot path: {:.2} ns/call", per_call);
    }

    #[test]
    #[ignore = "microbenchmark, run with --release"]
    fn bench_end_to_end_read() {
        // End-to-end: `ensure_arena::<u64>` + downcast + `arena.read`.
        // This is the full cost of a single `rt.get<u64>` at the
        // arena-access level. Excludes NodeData load, state.load_acquire,
        // and the fast-fail `state == Clean` check, which live in the
        // Runtime and will be measured separately when that code lands.
        let registry = ArenaRegistry::new();
        let ptr = registry.ensure_arena::<u64, _>(|| Box::new(AtomicPrimitiveArena::<u64>::new()));
        // SAFETY: pointer is valid for the registry's lifetime.
        let arena = unsafe {
            (*ptr)
                .as_any()
                .downcast_ref::<AtomicPrimitiveArena<u64>>()
                .expect("downcast")
        };
        let slot = arena.reserve(0xDEAD_BEEF);

        let start = Instant::now();
        let mut acc: u64 = 0;
        for _ in 0..ITERS {
            let ptr = registry.ensure_arena::<u64, _>(|| unreachable!());
            // SAFETY: valid for the registry's lifetime.
            let arena = unsafe {
                (*ptr)
                    .as_any()
                    .downcast_ref::<AtomicPrimitiveArena<u64>>()
                    .expect("downcast")
            };
            acc = acc.wrapping_add(arena.read(slot));
        }
        let elapsed = start.elapsed();
        black_box(acc);
        let per_call = elapsed.as_nanos() as f64 / ITERS as f64;
        eprintln!(
            "end-to-end (ensure_arena + downcast + read): {:.2} ns/call",
            per_call
        );
    }

    #[test]
    #[ignore = "microbenchmark, run with --release"]
    fn bench_direct_arena_read_baseline() {
        // Baseline: read from an arena we hold directly, bypassing the
        // registry entirely. Tells us the arena read cost in isolation,
        // so we can subtract it from the end-to-end number and see what
        // the registry itself is costing.
        let arena: AtomicPrimitiveArena<u64> = AtomicPrimitiveArena::new();
        let slot = arena.reserve(0xDEAD_BEEF);

        let start = Instant::now();
        let mut acc: u64 = 0;
        for _ in 0..ITERS {
            acc = acc.wrapping_add(arena.read(slot));
        }
        let elapsed = start.elapsed();
        black_box(acc);
        let per_call = elapsed.as_nanos() as f64 / ITERS as f64;
        eprintln!("arena.read direct (no registry): {:.2} ns/call", per_call);
    }
}
