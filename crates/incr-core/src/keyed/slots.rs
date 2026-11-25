//! The slot array of a `KeyTable`: open addressing with linear probing over
//! slots that each hold a stored hash and, when full, a key's pairs.
//!
//! A slot's hash doubles as its state, so a zeroed allocation is an array
//! of empty slots and comes from the allocator without a write per slot.

use std::alloc::Layout;
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::ptr::NonNull;
use std::slice;

use crate::few::Few;
use crate::grow::Storage;
use crate::pages;

const EMPTY: u64 = 0;
const TOMBSTONE: u64 = 1;
const FIRST_HASH: u64 = 2;

/// A key's index hash: its raw hash kept clear of the two values a slot
/// uses for its empty and tombstone states. Every structure that stores or
/// compares index hashes takes this value, never the raw hash.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BucketHash(u64);

impl BucketHash {
    #[inline]
    pub(super) fn new(raw: u64) -> Self {
        Self(raw.max(FIRST_HASH))
    }

    #[inline]
    pub(super) fn get(self) -> u64 {
        self.0
    }
}

#[repr(C)]
pub(super) struct Bucket<P, R> {
    pub(super) hash: BucketHash,
    /// Never empty while the bucket is in the table, outside a commit.
    pub(super) pairs: Few<(P, R)>,
}

/// Invariant: `pairs` is initialised exactly when `hash >= FIRST_HASH`.
#[repr(C)]
pub(super) struct Slot<P, R> {
    hash: u64,
    pairs: MaybeUninit<Few<(P, R)>>,
}

impl<P, R> Slot<P, R> {
    #[inline]
    fn is_full(&self) -> bool {
        self.hash >= FIRST_HASH
    }

    /// # Safety
    /// The slot must be full.
    #[inline]
    unsafe fn full(&self) -> &Bucket<P, R> {
        // SAFETY: both types are `repr(C)` with a `u64`-sized hash followed
        // by the pairs, `BucketHash` and `MaybeUninit` are transparent, and
        // a full slot's hash is a `BucketHash` and its pairs initialised.
        unsafe { &*(self as *const Self).cast::<Bucket<P, R>>() }
    }

    /// The slot's bucket, or `None` for a free slot.
    #[inline]
    pub(super) fn bucket(&self) -> Option<&Bucket<P, R>> {
        // SAFETY: checked full.
        self.is_full().then(|| unsafe { self.full() })
    }

    /// Takes the bucket of a full slot, leaving it in state `free`.
    fn take(&mut self, free: u64) -> Option<Bucket<P, R>> {
        if !self.is_full() {
            return None;
        }
        let hash = BucketHash(mem::replace(&mut self.hash, free));
        // SAFETY: the slot was full and is now marked free, so this read
        // takes the only ownership of the pairs.
        let pairs = unsafe { self.pairs.assume_init_read() };
        Some(Bucket { hash, pairs })
    }
}

/// `capacity` slots, zero or a power of two. Dropping it frees the memory
/// but drops no bucket; [`Storage::drop_entries`] drops the live ones.
pub(super) struct Slots<P, R> {
    ptr: NonNull<Slot<P, R>>,
    capacity: usize,
    _owns: PhantomData<Bucket<P, R>>,
}

// SAFETY: the array owns its buckets the way a `Box<[Bucket<P, R>]>` would.
unsafe impl<P: Send, R: Send> Send for Slots<P, R> {}
// SAFETY: see `Send`.
unsafe impl<P: Sync, R: Sync> Sync for Slots<P, R> {}

impl<P, R> Slots<P, R> {
    pub(super) fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            capacity: 0,
            _owns: PhantomData,
        }
    }

    fn layout(capacity: usize) -> Layout {
        Layout::array::<Slot<P, R>>(capacity).expect("incr-core: key table capacity overflow")
    }

    #[inline]
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// The bytes an array of `capacity` slots allocates.
    pub(super) fn heap_bytes(capacity: usize) -> usize {
        capacity * mem::size_of::<Slot<P, R>>()
    }

    #[inline]
    fn as_slice(&self) -> &[Slot<P, R>] {
        // SAFETY: `capacity` slots start at `ptr` (or none, at a dangling
        // pointer), each a hash that the zeroed allocation initialised and
        // pairs that need no initialisation.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.capacity) }
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Slot<P, R>] {
        // SAFETY: as for `as_slice`.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.capacity) }
    }

    /// The slot at `i`, or `None` past the end.
    #[inline]
    pub(super) fn get(&self, i: usize) -> Option<&Slot<P, R>> {
        self.as_slice().get(i)
    }

    /// The pairs of the bucket at `i`; `None` past the end or for a free
    /// slot.
    #[inline]
    pub(super) fn pairs_mut(&mut self, i: usize) -> Option<&mut Few<(P, R)>> {
        let slot = self.as_mut_slice().get_mut(i)?;
        // SAFETY: a full slot's pairs are initialised.
        slot.is_full()
            .then(|| unsafe { slot.pairs.assume_init_mut() })
    }

    /// The slot of the bucket with `hash` that `matches` accepts. Runs
    /// `matches` and changes nothing.
    #[inline(always)]
    pub(super) fn find(
        &self,
        hash: BucketHash,
        mut matches: impl FnMut(&Bucket<P, R>) -> bool,
    ) -> Option<usize> {
        let slots = self.as_slice();
        if slots.is_empty() {
            return None;
        }
        let mask = slots.len() - 1;
        let mut i = hash.0 as usize & mask;
        loop {
            let slot = &slots[i];
            if slot.hash == hash.0 {
                // SAFETY: a `BucketHash` is `FIRST_HASH` or more, so a slot
                // storing one is full.
                if matches(unsafe { slot.full() }) {
                    return Some(i);
                }
            } else if slot.hash == EMPTY {
                return None;
            }
            i = (i + 1) & mask;
        }
    }

    /// The slot of the bucket with `hash` that `matches` accepts, in the
    /// old storage of a migration whose cursor is `cursor`. The probe skips
    /// the slots below the cursor, which hold no bucket and may read as
    /// empty: the run from a bucket's home to its slot has no empty slot,
    /// so none of the slots skipped could have ended the probe before it
    /// reached a bucket at or past the cursor.
    pub(super) fn find_from(
        &self,
        cursor: usize,
        hash: BucketHash,
        mut matches: impl FnMut(&Bucket<P, R>) -> bool,
    ) -> Option<usize> {
        let slots = self.as_slice();
        let home = hash.0 as usize & (slots.len() - 1);
        let wrapped = cursor..home.max(cursor);
        (home.max(cursor)..slots.len())
            .chain(wrapped)
            .take_while(|&k| slots[k].hash != EMPTY)
            .find(|&k| {
                // SAFETY: a `BucketHash` is `FIRST_HASH` or more, so a slot
                // storing one is full.
                slots[k].hash == hash.0 && matches(unsafe { slots[k].full() })
            })
    }

    /// The buckets from slot `from` on, each with its slot numbered from
    /// `base`.
    #[inline]
    pub(super) fn buckets(
        &self,
        from: usize,
        base: usize,
    ) -> impl Iterator<Item = (usize, &Bucket<P, R>)> {
        self.as_slice()[from..]
            .iter()
            .enumerate()
            .filter_map(move |(k, slot)| Some((base + from + k, slot.bucket()?)))
    }

    /// The first free slot on the probe run of `hash`. Requires an empty
    /// slot in the array.
    pub(super) fn free_slot(&self, hash: BucketHash) -> usize {
        let slots = self.as_slice();
        let mask = slots.len() - 1;
        let mut i = hash.0 as usize & mask;
        while slots[i].is_full() {
            i = (i + 1) & mask;
        }
        i
    }

    /// Whether the free slot `i` was never part of a probe run.
    pub(super) fn is_empty_slot(&self, i: usize) -> bool {
        self.as_slice()[i].hash == EMPTY
    }

    /// Stores `bucket` in the free slot `i`.
    pub(super) fn put(&mut self, i: usize, bucket: Bucket<P, R>) {
        let slot = &mut self.as_mut_slice()[i];
        assert!(!slot.is_full(), "incr-core: key table slot {i} is not free");
        slot.pairs.write(bucket.pairs);
        slot.hash = bucket.hash.0;
    }

    /// Takes the bucket at `i`, `None` for a free slot. The slot becomes
    /// empty when no probe run can pass through it, which the second field
    /// reports, and a tombstone otherwise.
    pub(super) fn remove(&mut self, i: usize) -> Option<(Bucket<P, R>, bool)> {
        let slots = self.as_mut_slice();
        let next = (i + 1) & (slots.len() - 1);
        // A bucket's probe run covers every slot from its home to its own
        // position, so a run can only pass through `i` if it continues to
        // `i + 1`. With `i + 1` empty, `i` can be emptied too.
        let emptied = slots[next].hash == EMPTY;
        let bucket = slots[i].take(if emptied { EMPTY } else { TOMBSTONE })?;
        Some((bucket, emptied))
    }
}

impl<P, R> Storage for Slots<P, R> {
    type Entry = Bucket<P, R>;

    fn zeroed(capacity: usize) -> Self {
        debug_assert!(capacity.is_power_of_two());
        Self {
            ptr: pages::alloc_zeroed(Self::layout(capacity)).cast(),
            capacity,
            _owns: PhantomData,
        }
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn take(&mut self, k: usize) -> Option<Bucket<P, R>> {
        self.as_mut_slice()[k].take(TOMBSTONE)
    }

    fn place(&mut self, bucket: Bucket<P, R>) -> (usize, bool) {
        let slot = self.free_slot(bucket.hash);
        let reused = !self.is_empty_slot(slot);
        self.put(slot, bucket);
        (slot, reused)
    }

    fn drop_entries(&mut self, from: usize) {
        if !mem::needs_drop::<Few<(P, R)>>() {
            return;
        }
        for slot in &mut self.as_mut_slice()[from..] {
            drop(slot.take(TOMBSTONE));
        }
    }

    fn release(&mut self, from: usize, to: usize) {
        let size = mem::size_of::<Slot<P, R>>();
        // SAFETY: the bytes are those of slots `from..to`, which the
        // caller no longer reads; given back, they read as empty slots.
        unsafe {
            pages::release(
                self.ptr.cast(),
                Self::layout(self.capacity),
                from * size,
                to * size,
            )
        }
    }
}

impl<P, R> Drop for Slots<P, R> {
    fn drop(&mut self) {
        if self.capacity != 0 {
            // SAFETY: allocated by `Storage::zeroed` with this layout.
            unsafe { pages::dealloc(self.ptr.cast(), Self::layout(self.capacity)) }
        }
    }
}
