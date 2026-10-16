//! Type-erased arena registry.
//!
//! The runtime needs to store arenas for arbitrary user types `T: Value`
//! under one lookup structure. Each type gets a dense `u16` tag on first
//! use, and the tag indexes a `Vec<Arc<dyn ErasedArena<C>>>`; a
//! `HashMap<TypeId, u16>` maps types to tags. Nodes store only the tag,
//! so code holding an untyped node reaches its arena through the
//! object-safe `ErasedArena<C>`, and typed code downcasts to the concrete
//! `GenericArena<T, C>`.
//!
//! Per-T access pattern:
//! 1. Compute `TypeId::of::<T>()`.
//! 2. Look up or insert the tag and `Arc<GenericArena<T, C>>`.
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

    /// [`GenericArena::release`] without knowing `T`.
    fn release(&self, slot: u32);
}

impl<T: Value, C: Cells> ErasedArena<C> for GenericArena<T, C> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn release(&self, slot: u32) {
        GenericArena::release(self, slot)
    }
}

/// Type tags and their arenas. The runtime keeps one of these behind its
/// inner lock.
pub struct ArenaRegistry<C: Cells> {
    tags: HashMap<TypeId, u16>,
    // Invariant: indexed by tag, and a tag is handed out only after its
    // arena is pushed here, so every tag has an arena.
    by_tag: Vec<Arc<dyn ErasedArena<C>>>,
}

impl<C: Cells> Default for ArenaRegistry<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Cells> ArenaRegistry<C> {
    pub fn new() -> Self {
        Self {
            tags: HashMap::new(),
            by_tag: Vec::new(),
        }
    }

    /// Look up the tag and arena for `T`, allocating both on `T`'s first
    /// use. Callers should clone the `Arc` out and operate on the typed
    /// reference.
    pub fn ensure_tagged<T: Value>(&mut self) -> (u16, Arc<GenericArena<T, C>>) {
        let type_id = TypeId::of::<T>();
        let tag = match self.tags.get(&type_id) {
            Some(&tag) => tag,
            None => {
                let tag = u16::try_from(self.by_tag.len())
                    .expect("incr-core: more than 65536 distinct value types in one runtime");
                self.by_tag
                    .push(Arc::new(GenericArena::<T, C>::new()) as Arc<dyn ErasedArena<C>>);
                self.tags.insert(type_id, tag);
                tag
            }
        };
        // The entry was inserted under this exact T's TypeId, so the
        // downcast cannot fail.
        let arena = downcast_arc::<T, C>(self.by_tag[tag as usize].clone())
            .expect("ArenaRegistry::ensure_tagged downcast failed; TypeId/arena mismatch");
        (tag, arena)
    }

    /// Look up the arena for `T`, inserting a fresh one if missing.
    pub fn ensure_arena<T: Value>(&mut self) -> Arc<GenericArena<T, C>> {
        self.ensure_tagged::<T>().1
    }

    /// Look up the arena for `T` without inserting. Returns `None` if
    /// no arena exists for this type yet.
    pub fn try_arena<T: Value>(&self) -> Option<Arc<GenericArena<T, C>>> {
        let tag = *self.tags.get(&TypeId::of::<T>())?;
        downcast_arc::<T, C>(self.by_tag[tag as usize].clone())
    }

    /// The arena behind `tag` if it holds `T`, `None` otherwise. Hot
    /// paths use this with the tag the node already stores, because
    /// hashing `TypeId` on every read cost more than the read itself.
    pub fn tagged_arena<T: Value>(&self, tag: u16) -> Option<Arc<GenericArena<T, C>>> {
        downcast_arc::<T, C>(self.by_tag.get(tag as usize)?.clone())
    }

    /// The arena behind `tag`. Panics on a tag this registry never
    /// handed out.
    pub fn erased(&self, tag: u16) -> &Arc<dyn ErasedArena<C>> {
        &self.by_tag[tag as usize]
    }

    pub fn arena_count(&self) -> usize {
        self.by_tag.len()
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
    fn tags_are_dense_and_index_their_erased_arena() {
        let mut r: ArenaRegistry<Local> = ArenaRegistry::new();
        let (t_u64, a) = r.ensure_tagged::<u64>();
        let (t_str, _) = r.ensure_tagged::<String>();
        assert_eq!((t_u64, t_str), (0, 1));
        let (again, a2) = r.ensure_tagged::<u64>();
        assert_eq!(again, t_u64);
        assert!(Arc::ptr_eq(&a, &a2));

        let s = a.reserve_with(7);
        r.erased(t_u64).release(s);
        assert_eq!(a.reserve_with(9), s, "erased release must free the slot");
        assert_eq!(a.read(s), 9);
    }

    #[test]
    fn tagged_arena_checks_the_type_behind_the_tag() {
        let mut r: ArenaRegistry<Shared> = ArenaRegistry::new();
        let (t_u64, a) = r.ensure_tagged::<u64>();
        let (t_str, _) = r.ensure_tagged::<String>();
        let s = a.reserve_with(7);
        assert_eq!(r.tagged_arena::<u64>(t_u64).unwrap().read(s), 7);
        assert!(r.tagged_arena::<u64>(t_str).is_none());
        assert!(r.tagged_arena::<u64>(t_str + 1).is_none());
    }

    #[test]
    fn try_arena_returns_none_when_missing() {
        let r: ArenaRegistry<Shared> = ArenaRegistry::new();
        assert!(r.try_arena::<u64>().is_none());
    }
}
