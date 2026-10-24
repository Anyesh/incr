//! Type-erased arena registry.
//!
//! The runtime stores arenas for arbitrary user types `T: Value` under
//! one lookup structure. Each type gets a dense `u16` tag on first use,
//! nodes store only the tag, and the tag indexes a [`Directory`] of
//! entries, each holding the arena and the `TypeId` of its `T`. Readers
//! index it with no lock: a typed reader checks the entry's `TypeId`
//! before it casts the arena to `GenericArena<T, C>`, and code holding
//! an untyped node uses the object-safe [`ErasedArena`].
//!
//! Writers are serialized by [`ArenaTags`], the `TypeId`-to-tag map: it
//! lives behind the runtime's inner lock, and only a `&mut ArenaTags`
//! can add an arena. Entries are published with a Release store and
//! freed, with their arenas, only when the registry drops.

use std::any::{Any, TypeId};
use std::cell::Cell;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use crate::cells::{Cells, PtrCell};
use crate::deferred::Unbuilt;
use crate::directory::Directory;
use crate::generic_arena::GenericArena;
use crate::value::Value;
use crate::value_slot::{Compared, ValueSlot};

/// Object-safe arena interface, for code that holds a node's tag but
/// not its value type.
pub trait ErasedArena<C: Cells>: Send + Sync + 'static {
    /// [`GenericArena::release`] without knowing `T`. The displaced value
    /// is dropped (retired under Shared) here, after the arena lock is
    /// released, so a delete that calls this at its drain point runs the
    /// value's `Drop` at that point and under no lock.
    fn release(&self, slot: u32);

    /// Writes `value`, which must be of the arena's type, into `slot`:
    /// unconditionally when `first`, and otherwise only if it differs
    /// from the slot's value. `wrote` is set immediately before the
    /// store. Returns whether the value changed, and what to drop once
    /// the caller's protocol allows: the displaced value or the rejected
    /// equal one.
    #[doc(hidden)]
    fn write_erased(
        &self,
        slot: u32,
        value: Box<dyn Any + Send>,
        first: bool,
        wrote: &Cell<u8>,
    ) -> (bool, Option<Box<dyn Any>>);

    /// Empties `slot` without returning it to the free list, handing back
    /// the value it held.
    #[doc(hidden)]
    fn clear_value(&self, slot: u32) -> Option<Box<dyn Any>>;

    /// [`Self::clear_value`] for a value no run the unbounded one makes
    /// built, dropped under isolation wherever its `Drop` runs.
    #[doc(hidden)]
    fn clear_unbuilt(&self, slot: u32) -> Option<Unbuilt>;

    /// Whether `slot` holds a value. Clones it to find out, so it is for
    /// debug assertions only.
    #[doc(hidden)]
    fn has_value(&self, slot: u32) -> bool;

    /// Whether the arena's type has a placeholder.
    #[doc(hidden)]
    fn has_placeholder(&self) -> bool;

    /// A clone of the value in `slot`, if it holds one.
    #[doc(hidden)]
    fn read_erased(&self, slot: u32) -> Option<Box<dyn Any + Send>>;

    /// Whether `slot` holds a value equal to `value`, which must be of the
    /// arena's type. Runs the type's `PartialEq`, which may panic.
    #[doc(hidden)]
    fn equals_erased(&self, slot: u32, value: &(dyn Any + Send)) -> bool;
}

impl<T: Value, C: Cells> ErasedArena<C> for GenericArena<T, C> {
    fn release(&self, slot: u32) {
        drop(GenericArena::release(self, slot));
    }

    fn write_erased(
        &self,
        slot: u32,
        value: Box<dyn Any + Send>,
        first: bool,
        wrote: &Cell<u8>,
    ) -> (bool, Option<Box<dyn Any>>) {
        let value = *value
            .downcast::<T>()
            .expect("incr-core: an episode value has its node's type");
        if first {
            wrote.set(1);
            let displaced = self.write(slot, value);
            return (true, displaced.map(|d| Box::new(d) as Box<dyn Any>));
        }
        match self.compare_and_write(slot, value, || wrote.set(1)) {
            Compared::Equal(rejected) => (false, Some(Box::new(rejected))),
            Compared::Replaced(displaced) => (true, displaced.map(|d| Box::new(d) as Box<dyn Any>)),
        }
    }

    fn clear_value(&self, slot: u32) -> Option<Box<dyn Any>> {
        self.clear(slot).map(|d| Box::new(d) as Box<dyn Any>)
    }

    fn clear_unbuilt(&self, slot: u32) -> Option<Unbuilt> {
        self.clear(slot)
            .map(|d| Unbuilt::new(<C::ValueSlot<T> as ValueSlot<T>>::isolating(d)))
    }

    fn has_value(&self, slot: u32) -> bool {
        crate::value_code::run(|| self.try_read(slot).is_some())
    }

    fn has_placeholder(&self) -> bool {
        self.placeholder().is_some()
    }

    fn read_erased(&self, slot: u32) -> Option<Box<dyn Any + Send>> {
        crate::value_code::run(|| self.try_read(slot)).map(|v| Box::new(v) as Box<dyn Any + Send>)
    }

    fn equals_erased(&self, slot: u32, value: &(dyn Any + Send)) -> bool {
        let value = value
            .downcast_ref::<T>()
            .expect("incr-core: a stashed value has its node's type");
        crate::value_code::run(|| self.try_read(slot).is_some_and(|v| v == *value))
    }
}

struct ArenaEntry<C: Cells> {
    type_id: TypeId,
    // An `Arc` because a query's compute closure keeps its typed arena.
    arena: Arc<dyn ErasedArena<C>>,
}

#[cfg(loom)]
const FIRST_DIRECTORY_LEN: usize = 2;
#[cfg(not(loom))]
const FIRST_DIRECTORY_LEN: usize = 16;

/// Arenas by tag, readable with no lock.
pub(crate) struct ArenaRegistry<C: Cells> {
    entries: Directory<ArenaEntry<C>, C>,
}

/// The writer side of an [`ArenaRegistry`]: which type has which tag.
/// The runtime keeps exactly one per registry, behind its inner lock, so
/// a `&mut ArenaTags` serializes every registry write.
#[derive(Default)]
pub(crate) struct ArenaTags {
    tags: HashMap<TypeId, u16, BuildHasherDefault<TypeIdHasher>>,
}

/// Uses a `TypeId`'s own bits as its hash. They are already a hash of
/// the type, and hashing them again with SipHash cost every create
/// about 40 instructions.
#[derive(Default)]
struct TypeIdHasher(u64);

impl Hasher for TypeIdHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 ^= n;
    }

    fn write_u128(&mut self, n: u128) {
        self.0 ^= n as u64 ^ (n >> 64) as u64;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

impl ArenaTags {
    pub(crate) fn tag<T: Value>(&self) -> Option<u16> {
        self.tags.get(&TypeId::of::<T>()).copied()
    }
}

impl<C: Cells> Default for ArenaRegistry<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Cells> ArenaRegistry<C> {
    pub(crate) fn new() -> Self {
        Self {
            entries: Directory::with_len(FIRST_DIRECTORY_LEN),
        }
    }

    #[inline(always)]
    fn entry(&self, tag: u16) -> Option<&ArenaEntry<C>> {
        let entry = self.entries.get(tag as usize)?.load_acquire();
        // SAFETY: a non-null entry came from `Box::into_raw` in
        // `ensure_tagged`, was published by a Release store after it was
        // built, and is freed only when the registry drops.
        unsafe { entry.as_ref() }
    }

    /// The arena behind `tag` if it holds `T`, `None` otherwise. Takes no
    /// lock; hot paths use the tag the node already stores, because
    /// hashing `TypeId` on every read cost more than the read itself.
    #[inline(always)]
    pub(crate) fn tagged_arena<T: Value>(&self, tag: u16) -> Option<&GenericArena<T, C>> {
        let entry = self.entry(tag)?;
        if entry.type_id != TypeId::of::<T>() {
            return None;
        }
        // SAFETY: the entry was built from a `GenericArena<T, C>` for
        // the `T` its `type_id` names, which the compare just matched,
        // so the erased pointer's data part points at one.
        Some(unsafe { &*(Arc::as_ptr(&entry.arena) as *const GenericArena<T, C>) })
    }

    /// The arena behind `tag`. Panics on a tag this registry never
    /// handed out.
    #[inline(always)]
    pub(crate) fn erased(&self, tag: u16) -> &dyn ErasedArena<C> {
        match self.entry(tag) {
            Some(entry) => &*entry.arena,
            None => no_arena(tag),
        }
    }

    /// The tag and arena for `T`, allocating both on `T`'s first use.
    #[inline(always)]
    pub(crate) fn ensure_tagged<T: Value>(
        &self,
        tags: &mut ArenaTags,
    ) -> (u16, &GenericArena<T, C>) {
        let tag = self.tag_of::<T>(tags);
        let arena = self
            .tagged_arena::<T>(tag)
            .expect("incr-core: an arena's entry does not match its tag's type");
        (tag, arena)
    }

    /// [`Self::ensure_tagged`], with a shared handle to the arena for a
    /// closure that outlives the borrow.
    pub(crate) fn ensure_tagged_shared<T: Value>(
        &self,
        tags: &mut ArenaTags,
    ) -> (u16, Arc<GenericArena<T, C>>) {
        let tag = self.tag_of::<T>(tags);
        let entry = self.entry(tag).expect("incr-core: a tag with no entry");
        assert!(
            entry.type_id == TypeId::of::<T>(),
            "incr-core: an arena's entry does not match its tag's type"
        );
        let raw = Arc::into_raw(Arc::clone(&entry.arena)) as *const GenericArena<T, C>;
        // SAFETY: the `type_id` compare above shows the referent is a
        // `GenericArena<T, C>`, so this is `Arc::downcast`'s rewrap.
        (tag, unsafe { Arc::from_raw(raw) })
    }

    #[inline(always)]
    fn tag_of<T: Value>(&self, tags: &mut ArenaTags) -> u16 {
        match tags.tags.get(&TypeId::of::<T>()) {
            Some(&tag) => tag,
            None => self.add::<T>(tags),
        }
    }

    #[cold]
    #[inline(never)]
    fn add<T: Value>(&self, tags: &mut ArenaTags) -> u16 {
        let type_id = TypeId::of::<T>();
        let tag = u16::try_from(tags.tags.len())
            .expect("incr-core: more than 65536 distinct value types in one runtime");
        self.entries.grow_to(tag as usize);
        let entry = Box::new(ArenaEntry {
            type_id,
            arena: Arc::new(GenericArena::<T, C>::new()) as Arc<dyn ErasedArena<C>>,
        });
        self.entries
            .get(tag as usize)
            .expect("incr-core: grow_to left the tag's entry out")
            .store_release(Box::into_raw(entry));
        tags.tags.insert(type_id, tag);
        tag
    }

    #[cfg(all(test, not(loom)))]
    pub(crate) fn arena_count(&self) -> usize {
        (0..self.entries.len())
            .take_while(|&tag| self.entry(tag as u16).is_some())
            .count()
    }
}

#[cold]
#[inline(never)]
fn no_arena(tag: u16) -> ! {
    panic!(
        "incr-core: internal invariant violated: no arena for tag {}",
        tag
    );
}

impl<C: Cells> Drop for ArenaRegistry<C> {
    fn drop(&mut self) {
        for tag in 0..self.entries.len() {
            let Some(cell) = self.entries.get(tag) else {
                break;
            };
            let entry = cell.load_relaxed();
            if !entry.is_null() {
                // SAFETY: came from `Box::into_raw` in `tag_of`, and
                // `&mut self` rules out readers.
                drop(unsafe { Box::from_raw(entry) });
            }
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    #[test]
    fn local_registry_inserts_and_retrieves() {
        let r: ArenaRegistry<Local> = ArenaRegistry::new();
        let mut tags = ArenaTags::default();
        assert_eq!(r.arena_count(), 0);
        let (t1, a1) = r.ensure_tagged::<u64>(&mut tags);
        assert_eq!(r.arena_count(), 1);
        let (t2, a2) = r.ensure_tagged::<u64>(&mut tags);
        assert_eq!(r.arena_count(), 1);
        assert_eq!(t1, t2);
        assert!(std::ptr::eq(a1, a2));
        let s = a1.reserve_with(7);
        assert_eq!(a2.read(s), 7);
    }

    #[test]
    fn shared_handle_is_the_registered_arena() {
        let r: ArenaRegistry<Shared> = ArenaRegistry::new();
        let mut tags = ArenaTags::default();
        let (tag, shared) = r.ensure_tagged_shared::<String>(&mut tags);
        let s = shared.reserve_with("hello".to_string());
        assert_eq!(r.tagged_arena::<String>(tag).unwrap().read(s), "hello");
        assert!(std::ptr::eq(
            &*shared,
            r.tagged_arena::<String>(tag).unwrap()
        ));
    }

    #[test]
    fn tags_are_dense_and_index_their_erased_arena() {
        let r: ArenaRegistry<Local> = ArenaRegistry::new();
        let mut tags = ArenaTags::default();
        let (t_u64, a) = r.ensure_tagged::<u64>(&mut tags);
        let (t_str, _) = r.ensure_tagged::<String>(&mut tags);
        assert_eq!((t_u64, t_str), (0, 1));
        let s = a.reserve_with(7);
        r.erased(t_u64).release(s);
        assert_eq!(a.reserve_with(9), s, "erased release must free the slot");
        assert_eq!(a.read(s), 9);
    }

    #[test]
    fn tagged_arena_checks_the_type_behind_the_tag() {
        let r: ArenaRegistry<Shared> = ArenaRegistry::new();
        let mut tags = ArenaTags::default();
        let (t_u64, a) = r.ensure_tagged::<u64>(&mut tags);
        let (t_str, _) = r.ensure_tagged::<String>(&mut tags);
        let s = a.reserve_with(7);
        assert_eq!(r.tagged_arena::<u64>(t_u64).unwrap().read(s), 7);
        assert!(r.tagged_arena::<u64>(t_str).is_none());
        assert!(r.tagged_arena::<u64>(t_str + 1).is_none());
        assert!(r.tagged_arena::<u64>(u16::MAX).is_none());
    }

    /// Enough distinct types to grow the directory past its first length.
    #[test]
    fn many_types_grow_the_directory() {
        struct Ty<const N: usize>;
        trait Add {
            fn add(r: &ArenaRegistry<Shared>, tags: &mut ArenaTags) -> u16;
        }
        macro_rules! types {
            ($($n:literal)*) => {
                $(
                    impl Add for Ty<$n> {
                        fn add(r: &ArenaRegistry<Shared>, tags: &mut ArenaTags) -> u16 {
                            let (tag, a) = r.ensure_tagged::<[u8; $n]>(tags);
                            let s = a.reserve_with([$n as u8; $n]);
                            assert_eq!(r.tagged_arena::<[u8; $n]>(tag).unwrap().read(s), [$n as u8; $n]);
                            tag
                        }
                    }
                )*
                let r: ArenaRegistry<Shared> = ArenaRegistry::new();
                let mut tags = ArenaTags::default();
                let got: Vec<u16> = vec![$(Ty::<$n>::add(&r, &mut tags)),*];
                assert_eq!(got, (0..got.len() as u16).collect::<Vec<_>>());
                assert_eq!(r.arena_count(), got.len());
            };
        }
        types!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20);
    }
}
