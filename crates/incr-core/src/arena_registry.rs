//! Type-erased arena registry.
//!
//! The runtime needs to store arenas for arbitrary user types `T: Value`
//! under one lookup structure. We use the production pattern: a
//! `HashMap<TypeId, Arc<dyn ErasedArena<C>>>`. The trait `ErasedArena<C>`
//! is object-safe and provides downcast access to the concrete
//! `GenericArena<T, C>`.
//!
//! Per-T access pattern:
//! 1. Compute `TypeId::of::<T>()`.
//! 2. Look up or insert `Arc<GenericArena<T, C>>` in the registry.
//! 3. Clone the `Arc` out (cheap atomic refcount) and release the
//!    registry lock.
//! 4. Operate on the typed arena directly via the `Arc<GenericArena<T, C>>`.
//!
//! `Arc<dyn ErasedArena<C>>` requires `ErasedArena<C>: Send + Sync` so
//! the registry itself can be `Send + Sync` (under `Shared`'s `RwLock`).
//! Under `Local`, the `Send + Sync` bound is a zero-cost marker; the
//! actual arena access is single-threaded.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use crate::cells::Cells;
use crate::generic_arena::GenericArena;
use crate::value::Value;

/// Object-safe arena interface. The runtime stores arenas as
/// `Arc<dyn ErasedArena<C>>` and downcasts via `as_any` to the concrete
/// `GenericArena<T, C>`.
pub trait ErasedArena<C: Cells>: Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;
}

impl<T: Value, C: Cells> ErasedArena<C> for GenericArena<T, C> {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `HashMap<TypeId, Arc<dyn ErasedArena<C>>>` wrapped in a struct for
/// ergonomics. The runtime keeps one of these behind its inner lock.
pub struct ArenaRegistry<C: Cells> {
    arenas: HashMap<TypeId, Arc<dyn ErasedArena<C>>>,
}

impl<C: Cells> Default for ArenaRegistry<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Cells> ArenaRegistry<C> {
    pub fn new() -> Self {
        Self {
            arenas: HashMap::new(),
        }
    }

    /// Look up the arena for `T`, inserting a fresh one if missing.
    /// Returns an `Arc` to the typed arena; callers should clone the
    /// `Arc` out and operate on the typed reference.
    pub fn ensure_arena<T: Value>(&mut self) -> Arc<GenericArena<T, C>> {
        let type_id = TypeId::of::<T>();
        let erased = self
            .arenas
            .entry(type_id)
            .or_insert_with(|| Arc::new(GenericArena::<T, C>::new()) as Arc<dyn ErasedArena<C>>)
            .clone();
        // Downcast from `dyn ErasedArena<C>` to `Arc<GenericArena<T, C>>`.
        // Soundness: the entry was inserted with this exact T's TypeId,
        // so the downcast is guaranteed to succeed.
        downcast_arc::<T, C>(erased)
            .expect("ArenaRegistry::ensure_arena downcast failed; TypeId/arena mismatch")
    }

    /// Look up the arena for `T` without inserting. Returns `None` if
    /// no arena exists for this type yet.
    pub fn try_arena<T: Value>(&self) -> Option<Arc<GenericArena<T, C>>> {
        let type_id = TypeId::of::<T>();
        let erased = self.arenas.get(&type_id)?.clone();
        downcast_arc::<T, C>(erased)
    }

    pub fn arena_count(&self) -> usize {
        self.arenas.len()
    }
}

/// Downcast an `Arc<dyn ErasedArena<C>>` to `Arc<GenericArena<T, C>>`.
/// Returns `None` if the runtime type does not match `T`.
fn downcast_arc<T: Value, C: Cells>(
    arena: Arc<dyn ErasedArena<C>>,
) -> Option<Arc<GenericArena<T, C>>> {
    if arena.as_any().is::<GenericArena<T, C>>() {
        // SAFETY: we just checked the TypeId via `is::<GenericArena<T, C>>()`.
        // The Arc's referent is exactly `GenericArena<T, C>`. We transmute
        // the `Arc<dyn ErasedArena<C>>` into `Arc<GenericArena<T, C>>`
        // via raw-pointer rewrap. This is the standard pattern for
        // downcast-Arc; std::sync::Arc::downcast does the same.
        let raw = Arc::into_raw(arena) as *const GenericArena<T, C>;
        Some(unsafe { Arc::from_raw(raw) })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    #[test]
    fn local_registry_inserts_and_retrieves() {
        let mut r: ArenaRegistry<Local> = ArenaRegistry::new();
        assert_eq!(r.arena_count(), 0);
        let a1 = r.ensure_arena::<u64>();
        assert_eq!(r.arena_count(), 1);
        let a2 = r.ensure_arena::<u64>();
        assert_eq!(r.arena_count(), 1); // same TypeId, no new entry
                                        // The two Arcs point at the same arena.
        assert!(Arc::ptr_eq(&a1, &a2));

        let s = a1.reserve_with(7);
        assert_eq!(a2.read(s), 7);
    }

    #[test]
    fn shared_registry_inserts_and_retrieves() {
        let mut r: ArenaRegistry<Shared> = ArenaRegistry::new();
        let a = r.ensure_arena::<String>();
        let s = a.reserve_with("hello".to_string());
        assert_eq!(a.read(s), "hello");
    }

    #[test]
    fn different_types_get_different_arenas() {
        let mut r: ArenaRegistry<Shared> = ArenaRegistry::new();
        let _ = r.ensure_arena::<u64>();
        let _ = r.ensure_arena::<String>();
        assert_eq!(r.arena_count(), 2);
    }

    #[test]
    fn try_arena_returns_none_when_missing() {
        let r: ArenaRegistry<Shared> = ArenaRegistry::new();
        assert!(r.try_arena::<u64>().is_none());
    }
}
