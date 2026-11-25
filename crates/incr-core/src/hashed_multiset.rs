//! Open-addressing multiset keyed by caller-supplied hashes.
//!
//! The table stores each element's hash next to it, so `reserve` and the
//! migration it steps move entries by their stored hashes and never run
//! user code. Lookup is the only operation that calls user `Eq`; every
//! mutation takes an index that a lookup returned (for an absent value,
//! the first free slot on its probe run), so a caller can run all user
//! code first and then mutate without any user code running in between.
//! `T` carries no `Hash` bound at all, which makes "the table never
//! hashes" a compile-time fact.
//!
//! Removal leaves a tombstone (or an empty slot when no probe run can pass
//! through it) and never moves another entry, and only `reserve` moves
//! entries, a few per call while the table grows (see [`crate::grow`]), so
//! indices stay valid until the next `reserve`.

use std::alloc::{self, Layout};
use std::convert::Infallible;
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::ptr::NonNull;
use std::slice;
use std::time::{Duration, Instant};

use crate::capture::{self, Capture, CaptureStep};
use crate::grow::{Growing, Moves, Storage};
use crate::yield_hook::{NoTicks, Ticks};

const EMPTY: u8 = 0;
const TOMBSTONE: u8 = 1;
const FULL: u8 = 0x80;

/// Full slots store `FULL | top 7 hash bits`, so most probe mismatches
/// are rejected from the control byte without touching the entry.
fn tag(hash: u64) -> u8 {
    FULL | (hash >> 57) as u8
}

struct Entry<T> {
    hash: u64,
    count: usize,
    value: T,
}

/// `capacity` entries followed by `capacity` control bytes in a single
/// allocation, since every table needs both. Dropping a table frees the
/// memory but drops no entry; [`Storage::drop_entries`] drops the live
/// ones.
struct Table<T> {
    entries: NonNull<MaybeUninit<Entry<T>>>,
    ctrl: NonNull<u8>,
    capacity: usize,
    _owns: PhantomData<Entry<T>>,
}

// SAFETY: a table owns its entries the way a `Box<[Entry<T>]>` would.
unsafe impl<T: Send> Send for Table<T> {}
unsafe impl<T: Sync> Sync for Table<T> {}

impl<T> Table<T> {
    fn empty() -> Self {
        Self {
            entries: NonNull::dangling(),
            ctrl: NonNull::dangling(),
            capacity: 0,
            _owns: PhantomData,
        }
    }

    /// The allocation's layout and the control bytes' offset in it.
    fn layout(capacity: usize) -> (Layout, usize) {
        Layout::array::<Entry<T>>(capacity)
            .and_then(|entries| entries.extend(Layout::array::<u8>(capacity)?))
            .expect("incr-core: collection capacity overflow")
    }

    fn heap_bytes(capacity: usize) -> usize {
        match capacity {
            0 => 0,
            capacity => Self::layout(capacity).0.size(),
        }
    }

    fn ctrl(&self) -> &[u8] {
        // SAFETY: `capacity` initialised control bytes start at `ctrl`
        // (or none, at a dangling pointer).
        unsafe { slice::from_raw_parts(self.ctrl.as_ptr(), self.capacity) }
    }

    fn entries(&self) -> &[MaybeUninit<Entry<T>>] {
        // SAFETY: `capacity` entry slots start at `entries`, and a
        // `MaybeUninit` slot needs no initialisation.
        unsafe { slice::from_raw_parts(self.entries.as_ptr(), self.capacity) }
    }

    fn parts_mut(&mut self) -> (&mut [u8], &mut [MaybeUninit<Entry<T>>]) {
        // SAFETY: as for `ctrl` and `entries`; the two ranges are disjoint.
        unsafe {
            (
                slice::from_raw_parts_mut(self.ctrl.as_ptr(), self.capacity),
                slice::from_raw_parts_mut(self.entries.as_ptr(), self.capacity),
            )
        }
    }

    /// The entry in slot `k`, or `None` for a free slot.
    fn full(&self, k: usize) -> Option<&Entry<T>> {
        // SAFETY: a full slot's entry is initialised.
        (self.ctrl()[k] & FULL != 0).then(|| unsafe { self.entries()[k].assume_init_ref() })
    }

    /// Calls `visit` on the entry of each full slot from `from` on, with
    /// the slot numbered from `base`, until it returns an error.
    #[inline(always)]
    fn try_each<E>(
        &self,
        from: usize,
        base: usize,
        visit: &mut impl FnMut(usize, &T, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        let slots = self.ctrl()[from..].iter().zip(&self.entries()[from..]);
        for (k, (c, e)) in slots.enumerate() {
            if *c & FULL != 0 {
                // SAFETY: the slot is full, so its entry is initialised.
                let e = unsafe { e.assume_init_ref() };
                visit(base + from + k, &e.value, e.count)?;
            }
        }
        Ok(())
    }

    /// `Ok` with the slot of the entry equal to `value`, or `Err` with the
    /// first free slot on its probe run, the slot an insert of it takes.
    /// Requires a non-empty table.
    #[inline(always)]
    fn probe(&self, hash: u64, value: &T) -> Result<usize, usize>
    where
        T: PartialEq,
    {
        let (ctrl, entries) = (self.ctrl(), self.entries());
        let mask = ctrl.len() - 1;
        let tag = tag(hash);
        let mut i = hash as usize & mask;
        let mut free = usize::MAX;
        loop {
            let c = ctrl[i];
            if c == tag {
                // SAFETY: a tag byte has FULL set, so the entry is initialised.
                let e = unsafe { entries[i].assume_init_ref() };
                if e.hash == hash && e.value == *value {
                    return Ok(i);
                }
            } else if c & FULL == 0 {
                if c == EMPTY {
                    return Err(if free == usize::MAX { i } else { free });
                }
                if free == usize::MAX {
                    free = i;
                }
            }
            i = (i + 1) & mask;
        }
    }

    /// The slot of the entry equal to `value` in the old storage of a
    /// migration whose cursor is `cursor`. The probe skips the slots below
    /// the cursor, which hold no entry: the run from an entry's home to
    /// its slot has no empty slot, so none of the slots skipped could have
    /// ended the probe before it reached an entry at or past the cursor.
    fn probe_from(&self, cursor: usize, hash: u64, value: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        let (ctrl, entries) = (self.ctrl(), self.entries());
        let tag = tag(hash);
        let home = hash as usize & (ctrl.len() - 1);
        let wrapped = cursor..home.max(cursor);
        (home.max(cursor)..ctrl.len())
            .chain(wrapped)
            .take_while(|&k| ctrl[k] != EMPTY)
            .find(|&k| {
                // SAFETY: a tag byte has FULL set, so the entry is initialised.
                ctrl[k] == tag && {
                    let e = unsafe { entries[k].assume_init_ref() };
                    e.hash == hash && e.value == *value
                }
            })
    }
}

impl<T> Storage for Table<T> {
    type Entry = Entry<T>;

    fn zeroed(capacity: usize) -> Self {
        debug_assert!(capacity.is_power_of_two());
        let (layout, ctrl_offset) = Self::layout(capacity);
        // SAFETY: the size is not zero, because `capacity` is not and an
        // entry holds a `u64`.
        let base = unsafe { alloc::alloc_zeroed(layout) };
        let Some(base) = NonNull::new(base) else {
            alloc::handle_alloc_error(layout)
        };
        // SAFETY: the `capacity` control bytes at `ctrl_offset` lie inside
        // the allocation, and the entries at its start are aligned by the
        // layout. The zeroed allocation initialised every control byte to
        // EMPTY.
        let ctrl = unsafe { base.add(ctrl_offset) };
        Self {
            entries: base.cast(),
            ctrl,
            capacity,
            _owns: PhantomData,
        }
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn take(&mut self, k: usize) -> Option<Entry<T>> {
        let (ctrl, entries) = self.parts_mut();
        if ctrl[k] & FULL == 0 {
            return None;
        }
        ctrl[k] = TOMBSTONE;
        // SAFETY: the slot was full and is now marked free, so this read
        // takes the only ownership of the entry.
        Some(unsafe { entries[k].assume_init_read() })
    }

    fn place(&mut self, entry: Entry<T>) -> (usize, bool) {
        let (ctrl, entries) = self.parts_mut();
        let mask = ctrl.len() - 1;
        let mut j = entry.hash as usize & mask;
        while ctrl[j] & FULL != 0 {
            j = (j + 1) & mask;
        }
        let reused = ctrl[j] == TOMBSTONE;
        ctrl[j] = tag(entry.hash);
        entries[j].write(entry);
        (j, reused)
    }

    fn drop_entries(&mut self, from: usize) {
        if !mem::needs_drop::<T>() {
            return;
        }
        for k in from..self.capacity {
            drop(self.take(k));
        }
    }
}

impl<T> Drop for Table<T> {
    fn drop(&mut self) {
        if self.capacity != 0 {
            // SAFETY: allocated by `zeroed` with this layout.
            unsafe { alloc::dealloc(self.entries.as_ptr().cast(), Self::layout(self.capacity).0) }
        }
    }
}

pub(crate) struct HashedMultiset<T> {
    // Invariant: `ctrl[i] & FULL != 0` exactly when `entries[i]` is
    // initialised. The table's capacity is a power of two (or zero), and
    // at least one slot is EMPTY whenever it is not zero, so every probe
    // loop terminates.
    store: Growing<Table<T>>,
    len: usize,
    /// How many times `reserve` moved an entry or started or ended a
    /// migration, which a write whose user `Eq` panics after reserving
    /// can do without changing any element.
    rehashes: u64,
    capture: Option<Box<Capture<(T, usize)>>>,
}

/// Where one change lands, as found by [`HashedMultiset::locate`]. Valid
/// for the multiset that produced it until its next `reserve`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Located {
    remove: Option<usize>,
    add: Option<AddAt>,
}

/// Where one added occurrence lands, as found by
/// [`HashedMultiset::locate_add`]. Valid until the multiset next changes.
#[derive(Clone, Copy, Debug)]
pub(crate) enum AddAt {
    Existing(usize),
    /// No entry equals the value; `slot` is the first free slot on its
    /// probe run, where it is stored.
    Absent {
        hash: u64,
        slot: usize,
    },
}

impl Located {
    pub(crate) fn removes(&self) -> bool {
        self.remove.is_some()
    }
}

impl<T> HashedMultiset<T> {
    pub(crate) fn new() -> Self {
        Self {
            store: Growing::new(Table::empty()),
            len: 0,
            rehashes: 0,
            capture: None,
        }
    }

    /// The slots `reserve` has scanned for entries to move, over the
    /// multiset's life.
    #[cfg(test)]
    pub(crate) fn scanned(&self) -> usize {
        self.store.scanned()
    }

    /// Number of distinct elements.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The bytes of the table's allocation, and of the one it is growing
    /// out of.
    pub(crate) fn heap_bytes(&self) -> usize {
        Table::<T>::heap_bytes(self.store.table.capacity)
            + Table::<T>::heap_bytes(self.store.old_capacity())
    }

    /// Calls `visit` on each distinct element with its multiplicity, in
    /// table order.
    #[inline]
    pub(crate) fn for_each(&self, mut visit: impl FnMut(&T, usize)) {
        let Ok(()) = self.try_for_each_slot(|_, value, count| {
            visit(value, count);
            Ok::<(), Infallible>(())
        });
    }

    /// How many times the slots of entries changed; slots from
    /// [`Self::try_for_each_slot`] name the same entries while this and
    /// the elements stay unchanged.
    pub(crate) fn rehashes(&self) -> u64 {
        self.rehashes
    }

    /// [`Self::try_for_each_slot`] for a `visit` that cannot fail.
    #[cfg(test)]
    fn for_each_slot(&self, mut visit: impl FnMut(usize, &T, usize)) {
        let Ok(()) = self.try_for_each_slot(|slot, value, count| {
            visit(slot, value, count);
            Ok::<(), Infallible>(())
        });
    }

    /// Calls `visit` on each distinct element with its slot, which
    /// [`Self::stored`] and [`Self::count`] take until the multiset next
    /// changes or reserves, and its multiplicity, until it returns an
    /// error.
    #[inline]
    pub(crate) fn try_for_each_slot<E>(
        &self,
        mut visit: impl FnMut(usize, &T, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        self.store.table.try_each(0, 0, &mut visit)?;
        if self.store.migrating() {
            self.try_each_old(&mut visit)?;
        }
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn try_each_old<E>(
        &self,
        visit: &mut impl FnMut(usize, &T, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        match self.store.old() {
            Some(old) => old.storage.try_each(old.cursor, old.base, visit),
            None => Ok(()),
        }
    }

    /// Ensure `additional` more distinct elements can be inserted, and
    /// take a step of a running migration. Runs no user code, and
    /// invalidates every located index.
    #[inline]
    pub(crate) fn reserve(&mut self, additional: usize) {
        if self.store.must_grow(additional) {
            self.grow(additional, None, &NoTicks);
        }
    }

    /// [`Self::reserve`] that reports where the entries it moved went, or
    /// `None` when no slot changed, ticking `yielder` per entry moved.
    pub(crate) fn reserve_moving(
        &mut self,
        additional: usize,
        yielder: &impl Ticks,
    ) -> Option<Moves> {
        if !self.store.must_grow(additional) {
            return None;
        }
        let mut moves = Moves::default();
        self.grow(additional, Some(&mut moves), yielder)
            .then_some(moves)
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self, additional: usize, moves: Option<&mut Moves>, yielder: &impl Ticks) -> bool {
        let changed = self
            .store
            .grow(self.len, additional, &mut self.capture, moves, yielder);
        self.rehashes += u64::from(changed);
        changed
    }

    /// Installs a capture of every element with its count as of now.
    /// Panics if one is installed already.
    pub(crate) fn begin_capture(&mut self) {
        assert!(
            self.capture.is_none(),
            "incr-core: a capture is already installed"
        );
        self.capture = Some(Capture::new(
            self.store.table.capacity,
            self.store.old_capacity(),
        ));
    }

    /// Clones the unsettled entries from the capture's cursor on, at least
    /// one `unit` of slots and until `budget` since `start` has passed.
    /// Returns where the scan stopped and what it cloned, or `None` with
    /// no capture installed. Runs user `Clone`.
    pub(crate) fn capture_scan(
        &self,
        unit: usize,
        start: Instant,
        budget: Duration,
    ) -> Option<(usize, Vec<(T, usize)>)>
    where
        T: Clone,
    {
        let capture = self.capture.as_deref()?;
        let emptied = self.store.old().map_or(0, |old| old.cursor);
        let from = capture.cursor().max(emptied);
        let mut run = Vec::new();
        let end = capture::scan(from, capture.len(), unit, start, budget, |position| {
            let i = capture.slot(position);
            if let Some(e) = self.full(i) {
                if !capture.settled(i) {
                    run.push((e.value.clone(), e.count));
                }
            }
        });
        Some((end, run))
    }

    /// Adds a [`Self::capture_scan`] run that ended at `end`, made against
    /// the elements as they are now, and hands over what the capture took
    /// since the last step, removing it once it is done.
    pub(crate) fn capture_advance(
        &mut self,
        end: usize,
        run: Vec<(T, usize)>,
    ) -> CaptureStep<(T, usize)> {
        let Some(capture) = self.capture.as_deref_mut() else {
            panic!("incr-core: no capture is installed");
        };
        capture.advance(end, run);
        self.capture_drain()
    }

    /// Hands over what the capture took since the last step, removing it
    /// once it is done; an empty finished step without one.
    pub(crate) fn capture_drain(&mut self) -> CaptureStep<(T, usize)> {
        let step = match self.capture.as_deref_mut() {
            Some(capture) => capture.drain(),
            None => CaptureStep {
                rows: Vec::new(),
                done: true,
                discarded: Vec::new(),
            },
        };
        if step.done {
            self.capture = None;
        }
        step
    }

    /// Removes the capture, handing back what it took for the caller to
    /// drop.
    pub(crate) fn end_capture(&mut self) -> Vec<(T, usize)> {
        self.capture
            .take()
            .map(|capture| (*capture).into_rows())
            .unwrap_or_default()
    }

    /// Index of the entry equal to `value`, which must hash to `hash`.
    /// The only operation that runs user code (`T::eq`), and it changes
    /// nothing.
    pub(crate) fn find(&self, hash: u64, value: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        if self.store.table.capacity == 0 {
            return None;
        }
        self.probe(hash, value).ok()
    }

    /// `Ok` with the index of the entry equal to `value`, or `Err` with
    /// the first free slot on its probe run, the slot an insert of it
    /// takes. Requires a non-empty table.
    #[inline(always)]
    fn probe(&self, hash: u64, value: &T) -> Result<usize, usize>
    where
        T: PartialEq,
    {
        match self.store.table.probe(hash, value) {
            Err(free) if self.store.migrating() => self.probe_old(hash, value, free),
            found => found,
        }
    }

    #[cold]
    #[inline(never)]
    fn probe_old(&self, hash: u64, value: &T, free: usize) -> Result<usize, usize>
    where
        T: PartialEq,
    {
        self.store
            .old()
            .and_then(|old| Some(old.base + old.storage.probe_from(old.cursor, hash, value)?))
            .ok_or(free)
    }

    /// Requires a prior `reserve(1)`, so the table is not empty.
    #[inline(always)]
    fn locate_one(&self, hash: u64, value: &T) -> AddAt
    where
        T: PartialEq,
    {
        match self.probe(hash, value) {
            Ok(i) => AddAt::Existing(i),
            Err(slot) => AddAt::Absent { hash, slot },
        }
    }

    /// The entry in slot `i` of either storage, or `None` for a free slot.
    fn full(&self, i: usize) -> Option<&Entry<T>> {
        if i < self.store.table.capacity {
            return self.store.table.full(i);
        }
        let old = self.store.old()?;
        old.storage.full(i - old.base)
    }

    fn entry_mut(&mut self, i: usize) -> &mut Entry<T> {
        if i >= self.store.table.capacity {
            return self.old_entry_mut(i);
        }
        let (ctrl, entries) = self.store.table.parts_mut();
        assert!(
            ctrl[i] & FULL != 0,
            "incr-core: multiset index {i} is not a live entry"
        );
        // SAFETY: checked full just above.
        unsafe { entries[i].assume_init_mut() }
    }

    #[cold]
    #[inline(never)]
    fn old_entry_mut(&mut self, i: usize) -> &mut Entry<T> {
        let (old, k) = self.store.old_slot_mut(i);
        let (ctrl, entries) = old.parts_mut();
        assert!(
            ctrl[k] & FULL != 0,
            "incr-core: multiset index {i} is not a live entry"
        );
        // SAFETY: checked full just above.
        unsafe { entries[k].assume_init_mut() }
    }

    fn entry(&self, i: usize) -> &Entry<T> {
        let Some(&c) = self.store.table.ctrl().get(i) else {
            return self.old_entry(i);
        };
        assert!(
            c & FULL != 0,
            "incr-core: multiset index {i} is not a live entry"
        );
        // SAFETY: checked full just above.
        unsafe { self.store.table.entries()[i].assume_init_ref() }
    }

    #[cold]
    #[inline(never)]
    fn old_entry(&self, i: usize) -> &Entry<T> {
        match self.full(i) {
            Some(entry) => entry,
            None => panic!("incr-core: multiset index {i} is not a live entry"),
        }
    }

    /// The multiplicity of the live entry at `i`.
    pub(crate) fn count(&self, i: usize) -> usize {
        self.entry(i).count
    }

    /// The value stored at `i`, a live entry.
    pub(crate) fn stored(&self, i: usize) -> &T {
        &self.entry(i).value
    }

    /// The value stored for the element an add at `at` lands on, once
    /// the add has run: `value` itself if the element is new.
    pub(crate) fn stored_after_add<'a>(&'a self, at: AddAt, value: &'a T) -> &'a T {
        match at {
            AddAt::Existing(i) => self.stored(i),
            AddAt::Absent { .. } => value,
        }
    }

    /// The stored values a located change removes and adds an occurrence
    /// of, read before [`Self::apply`] runs: the removed side's as stored
    /// now, the added side's as stored after, which is `add` itself for a
    /// new element or for a count-1 entry that `apply` swaps.
    pub(crate) fn representatives<'a>(&'a self, at: Located, add: &'a T) -> [Option<&'a T>; 2] {
        let removed = at.remove.map(|i| self.stored(i));
        let added = at.add.map(|a| match a {
            AddAt::Existing(i) if at.remove == Some(i) && self.entry(i).count == 1 => add,
            a => self.stored_after_add(a, add),
        });
        [removed, added]
    }

    fn increment(&mut self, i: usize) {
        capture::check_settled(&self.capture, i);
        self.entry_mut(i).count += 1;
    }

    fn swap_value(&mut self, i: usize, new: T) -> T {
        capture::check_settled(&self.capture, i);
        mem::replace(&mut self.entry_mut(i).value, new)
    }

    #[inline]
    pub(crate) fn capture_installed(&self) -> bool {
        self.capture.is_some()
    }

    /// Settles the live entry at `i` for an installed capture, cloning its
    /// value and count into it unless it took them already. Runs user
    /// `Clone`; call it before the write that changes the entry.
    #[inline]
    pub(crate) fn settle(&mut self, i: usize)
    where
        T: Clone,
    {
        if self.capture.is_some() {
            self.settle_slow(i);
        }
    }

    #[cold]
    #[inline(never)]
    fn settle_slow(&mut self, i: usize)
    where
        T: Clone,
    {
        if self.capture.as_deref().is_none_or(|c| c.settled(i)) {
            return;
        }
        let e = self.entry(i);
        let content = (e.value.clone(), e.count);
        if let Some(capture) = self.capture.as_deref_mut() {
            capture.take(i, [content]);
        }
    }

    /// [`Self::find`] that settles the entry it finds, for a write that
    /// changes it next. It first takes a step of a running migration, so
    /// that writes which only remove still finish one.
    pub(crate) fn find_settling(&mut self, hash: u64, value: &T) -> Option<usize>
    where
        T: PartialEq + Clone,
    {
        self.reserve(0);
        let found = self.find(hash, value);
        if let Some(i) = found {
            self.settle(i);
        }
        found
    }

    /// Decrement the count at `i`, removing the entry and handing its
    /// value back (for the caller to drop) when the count reaches zero.
    /// Runs no user code.
    #[inline]
    pub(crate) fn decrement_or_remove(&mut self, i: usize) -> Option<T> {
        capture::check_settled(&self.capture, i);
        let entry = self.entry_mut(i);
        if entry.count > 1 {
            entry.count -= 1;
            return None;
        }
        if i >= self.store.table.capacity {
            return self.remove_old(i);
        }
        let (ctrl, entries) = self.store.table.parts_mut();
        let next = (i + 1) & (ctrl.len() - 1);
        // An entry's probe run covers every slot from its home to its own
        // position, so a run can only pass through `i` if it continues to
        // `i + 1`. With `i + 1` empty, `i` can be emptied too.
        ctrl[i] = if ctrl[next] == EMPTY {
            self.store.growth_left += 1;
            EMPTY
        } else {
            TOMBSTONE
        };
        self.len -= 1;
        // SAFETY: the slot was full and is now marked free, so this read
        // takes the only ownership of the entry.
        Some(unsafe { entries[i].assume_init_read() }.value)
    }

    /// Removes the entry in old slot `i`, whose room in the table the
    /// migration no longer needs. The slot always becomes a tombstone: the
    /// old storage takes no insert, and a lookup in it may rely on no slot
    /// at or past the migration's cursor turning empty.
    #[cold]
    #[inline(never)]
    fn remove_old(&mut self, i: usize) -> Option<T> {
        let (old, k) = self.store.old_slot_mut(i);
        let entry = old.take(k);
        self.store.growth_left += 1;
        self.len -= 1;
        entry.map(|entry| entry.value)
    }

    /// Insert `value`, which no entry equals, with count 1 at `i`, the
    /// free slot its lookup returned. Requires a prior `reserve(1)`.
    #[inline]
    fn insert_absent(&mut self, hash: u64, i: usize, value: T) {
        assert!(
            self.store.growth_left > 0,
            "incr-core: multiset insert without a reserve"
        );
        let (ctrl, entries) = self.store.table.parts_mut();
        assert!(
            ctrl[i] & FULL == 0,
            "incr-core: multiset slot {i} is not free"
        );
        if ctrl[i] == EMPTY {
            self.store.growth_left -= 1;
        }
        entries[i].write(Entry {
            hash,
            count: 1,
            value,
        });
        ctrl[i] = tag(hash);
        self.len += 1;
        capture::settle_free(&mut self.capture, i);
    }

    /// Insert `value` with count 1 at the first free slot of its probe
    /// run, comparing it with no entry, so the caller must already know
    /// that no entry equals it. Requires room from a prior `reserve`.
    /// Returns its index.
    pub(crate) fn insert_absent_probe(&mut self, hash: u64, value: T) -> usize {
        assert!(
            self.store.growth_left > 0,
            "incr-core: multiset insert without a reserve"
        );
        let ctrl = self.store.table.ctrl();
        let mask = ctrl.len() - 1;
        let mut i = hash as usize & mask;
        while ctrl[i] & FULL != 0 {
            i = (i + 1) & mask;
        }
        self.insert_absent(hash, i, value);
        i
    }

    /// Locate one change: removing one occurrence of `remove` and adding
    /// one occurrence of `add`, each given with its hash. Reserves room
    /// for the add, then runs the lookups (user `Eq`) and settles the
    /// entries found (user `Clone`). Changes no element, so a panic here
    /// leaves the multiset as it was.
    pub(crate) fn locate(&mut self, remove: Option<(u64, &T)>, add: Option<(u64, &T)>) -> Located
    where
        T: PartialEq + Clone,
    {
        self.reserve(usize::from(add.is_some()));
        let remove = remove.and_then(|(hash, value)| self.find(hash, value));
        let add = add.map(|(hash, value)| self.locate_one(hash, value));
        if let Some(i) = remove {
            self.settle(i);
        }
        if let Some(AddAt::Existing(i)) = add {
            self.settle(i);
        }
        Located { remove, add }
    }

    /// Locate one added occurrence of `value`: reserves room, then runs
    /// the lookup (user `Eq`) and settles the entry found (user `Clone`).
    /// Changes no element.
    #[inline]
    pub(crate) fn locate_add(&mut self, hash: u64, value: &T) -> AddAt
    where
        T: PartialEq + Clone,
    {
        self.reserve(1);
        let at = self.locate_one(hash, value);
        if let AddAt::Existing(i) = at {
            self.settle(i);
        }
        at
    }

    /// Add one occurrence at `at`. Runs no user code. Hands `value` back
    /// when an equal element was already stored, for the caller to drop
    /// when it is safe to run user `Drop`.
    #[inline]
    pub(crate) fn add(&mut self, at: AddAt, value: T) -> Option<T> {
        match at {
            AddAt::Existing(i) => {
                self.increment(i);
                Some(value)
            }
            AddAt::Absent { hash, slot } => {
                self.insert_absent(hash, slot, value);
                None
            }
        }
    }

    /// Apply a located change, with `add` the value to store if the added
    /// element is new. Runs no user code. Returns the values the change
    /// displaced, for the caller to drop when it is safe to run user
    /// `Drop`: the unused `add` when its element was already present, and
    /// the stored value when the last occurrence is removed.
    ///
    /// When remove and add name the same entry at count 1, the stored
    /// value is swapped for `add`, the same result as a removal followed
    /// by an insert; at a higher count the stored value stays.
    pub(crate) fn apply(&mut self, at: Located, add: Option<T>) -> [Option<T>; 2] {
        assert_eq!(
            at.add.is_some(),
            add.is_some(),
            "incr-core: multiset change applied with a mismatched value"
        );
        let mut displaced = [None, None];
        match (at.add, add) {
            (Some(AddAt::Existing(i)), Some(value)) if at.remove == Some(i) => {
                displaced[0] = Some(if self.count(i) == 1 {
                    self.swap_value(i, value)
                } else {
                    value
                });
                return displaced;
            }
            (Some(at), Some(value)) => displaced[0] = self.add(at, value),
            _ => {}
        }
        if let Some(i) = at.remove {
            displaced[1] = self.decrement_or_remove(i);
        }
        displaced
    }
}

impl<T> Default for HashedMultiset<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grow::{load_limit, with_step_slots};
    use proptest::prelude::*;
    use std::collections::HashMap;

    /// Equal by `key` only, so which copy the table keeps is observable
    /// through `id`. The boxed id lets Miri catch a double drop or a leak.
    #[derive(Clone, Debug)]
    struct V {
        key: u8,
        id: Box<u32>,
    }

    impl PartialEq for V {
        fn eq(&self, other: &Self) -> bool {
            self.key == other.key
        }
    }

    fn v(key: u8, id: u32) -> V {
        V {
            key,
            id: Box::new(id),
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Hashing {
        // Few homes and identical tags: long merged probe runs, and lookups
        // that reach `eq` on unequal values.
        Clustered,
        Spread,
    }

    fn hash(hashing: Hashing, key: u8) -> u64 {
        match hashing {
            Hashing::Clustered => u64::from(key % 3),
            Hashing::Spread => u64::from(key).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        }
    }

    #[derive(Clone, Debug)]
    enum Op {
        Insert(u8, u32),
        Remove(u8),
        Replace(Option<u8>, u8, u32),
        Reserve(usize),
    }

    const KEYS: u8 = 40;

    fn op() -> impl Strategy<Value = Op> {
        let key = 0_u8..KEYS;
        prop_oneof![
            3 => (key.clone(), any::<u32>()).prop_map(|(k, id)| Op::Insert(k, id)),
            2 => key.clone().prop_map(Op::Remove),
            3 => (proptest::option::of(key.clone()), key, any::<u32>())
                .prop_map(|(old, k, id)| Op::Replace(old, k, id)),
            1 => (0_usize..40).prop_map(Op::Reserve),
        ]
    }

    /// Model: key -> (count, id of the stored copy).
    type Model = HashMap<u8, (usize, u32)>;

    fn change(
        set: &mut HashedMultiset<V>,
        hashing: Hashing,
        remove: Option<u8>,
        add: Option<(u8, u32)>,
    ) -> (bool, Vec<u32>) {
        let remove = remove.map(|k| v(k, u32::MAX));
        let add = add.map(|(k, id)| v(k, id));
        let at = set.locate(
            remove.as_ref().map(|r| (hash(hashing, r.key), r)),
            add.as_ref().map(|a| (hash(hashing, a.key), a)),
        );
        let removed = at.removes();
        let mut displaced: Vec<u32> = set
            .apply(at, add)
            .into_iter()
            .flatten()
            .map(|d| *d.id)
            .collect();
        displaced.sort_unstable();
        (removed, displaced)
    }

    /// The single-sided paths the collection mutations take.
    fn add_one(set: &mut HashedMultiset<V>, hashing: Hashing, k: u8, id: u32) -> (bool, Vec<u32>) {
        let value = v(k, id);
        let at = set.locate_add(hash(hashing, k), &value);
        let displaced = set.add(at, value).map(|d| *d.id);
        (false, displaced.into_iter().collect())
    }

    fn remove_one(set: &mut HashedMultiset<V>, hashing: Hashing, k: u8) -> (bool, Vec<u32>) {
        let Some(i) = set.find(hash(hashing, k), &v(k, u32::MAX)) else {
            return (false, Vec::new());
        };
        let displaced = set.decrement_or_remove(i).map(|d| *d.id);
        (true, displaced.into_iter().collect())
    }

    fn model_change(
        model: &mut Model,
        remove: Option<u8>,
        add: Option<(u8, u32)>,
    ) -> (bool, Vec<u32>) {
        let removed = remove.is_some_and(|k| model.contains_key(&k));
        let mut displaced = Vec::new();
        if let (Some(old), Some((key, id))) = (remove.filter(|_| removed), add) {
            if old == key {
                let entry = model.get_mut(&key).unwrap();
                if entry.0 == 1 {
                    displaced.push(entry.1);
                    entry.1 = id;
                } else {
                    displaced.push(id);
                }
                return (true, displaced);
            }
        }
        if let Some((key, id)) = add {
            match model.get_mut(&key) {
                Some(entry) => {
                    entry.0 += 1;
                    displaced.push(id);
                }
                None => {
                    model.insert(key, (1, id));
                }
            }
        }
        if let Some(key) = remove.filter(|_| removed) {
            let entry = model.get_mut(&key).unwrap();
            entry.0 -= 1;
            if entry.0 == 0 {
                displaced.push(entry.1);
                model.remove(&key);
            }
        }
        displaced.sort_unstable();
        (removed, displaced)
    }

    fn assert_matches(set: &HashedMultiset<V>, hashing: Hashing, model: &Model) {
        assert_eq!(set.len(), model.len());
        let mut seen: Vec<(u8, usize, u32)> = Vec::new();
        set.for_each(|v, c| seen.push((v.key, c, *v.id)));
        seen.sort_unstable();
        let mut expected: Vec<(u8, usize, u32)> =
            model.iter().map(|(&k, &(c, id))| (k, c, id)).collect();
        expected.sort_unstable();
        assert_eq!(seen, expected);
        for key in 0..KEYS {
            let found = set.find(hash(hashing, key), &v(key, 0));
            assert_eq!(found.is_some(), model.contains_key(&key), "find({key})");
        }
        assert_accounting(set);
    }

    /// The length and the room left agree with the slots of both storages.
    fn assert_accounting(set: &HashedMultiset<V>) {
        let ctrl = set.store.table.ctrl();
        let full = ctrl.iter().filter(|c| **c & FULL != 0).count();
        let tombstones = ctrl.iter().filter(|c| **c == TOMBSTONE).count();
        let unmoved = set.store.old().map_or(0, |old| {
            let ctrl = &old.storage.ctrl()[old.cursor..];
            ctrl.iter().filter(|c| **c & FULL != 0).count()
        });
        assert_eq!(full + unmoved, set.len);
        assert_eq!(
            full + tombstones + unmoved + set.store.growth_left,
            load_limit(set.store.table.capacity)
        );
    }

    fn run(hashing: Hashing, ops: &[Op]) {
        let mut set = HashedMultiset::new();
        let mut model = Model::new();
        for op in ops {
            match *op {
                Op::Insert(k, id) => {
                    let got = add_one(&mut set, hashing, k, id);
                    assert_eq!(got, model_change(&mut model, None, Some((k, id))), "{op:?}");
                }
                Op::Remove(k) => {
                    let got = remove_one(&mut set, hashing, k);
                    assert_eq!(got, model_change(&mut model, Some(k), None), "{op:?}");
                }
                Op::Replace(old, k, id) => {
                    let got = change(&mut set, hashing, old, Some((k, id)));
                    assert_eq!(got, model_change(&mut model, old, Some((k, id))), "{op:?}");
                }
                Op::Reserve(n) => {
                    let before = set.store.table.capacity;
                    let mut held = Vec::new();
                    set.for_each_slot(|slot, value, _| held.push((slot, value.key)));
                    let moves = set.reserve_moving(n, &NoTicks);
                    assert!(set.store.growth_left >= n);
                    assert!(
                        set.store.table.capacity >= before,
                        "reserve shrank the table"
                    );
                    for (slot, key) in held {
                        let slot = moves.as_ref().map_or(slot, |moves| moves.remap(slot));
                        assert_eq!(set.stored(slot).key, key, "{op:?} moved key {key}");
                    }
                }
            }
            assert_matches(&set, hashing, &model);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: if cfg!(miri) { 6 } else { 512 },
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        #[test]
        fn matches_a_hashmap_model(
            clustered in any::<bool>(),
            step in prop_oneof![Just(1_usize), Just(3), Just(256)],
            ops in proptest::collection::vec(op(), 0..if cfg!(miri) { 40 } else { 200 }),
        ) {
            let hashing = if clustered { Hashing::Clustered } else { Hashing::Spread };
            with_step_slots(step, || run(hashing, &ops));
        }
    }

    #[test]
    fn replace_with_an_equal_value_at_count_one_swaps_the_stored_copy() {
        for hashing in [Hashing::Clustered, Hashing::Spread] {
            run(
                hashing,
                &[
                    Op::Insert(5, 1),
                    Op::Replace(Some(5), 5, 2),
                    Op::Replace(Some(5), 5, 3),
                ],
            );
        }
    }

    #[test]
    fn replace_with_an_equal_value_at_count_two_keeps_the_stored_copy() {
        for hashing in [Hashing::Clustered, Hashing::Spread] {
            run(
                hashing,
                &[
                    Op::Insert(5, 1),
                    Op::Insert(5, 2),
                    Op::Replace(Some(5), 5, 3),
                    Op::Remove(5),
                ],
            );
        }
    }

    #[test]
    fn replace_where_old_sits_earlier_in_new_probe_run() {
        // Keys 0, 3 and 6 share a home under clustered hashing. 0 takes the
        // home slot and 3 the next, so replacing 0 by 6 puts 6 after both,
        // then tombstones the home slot that 6 and 3 are found through.
        let ops = [
            Op::Insert(0, 1),
            Op::Insert(3, 2),
            Op::Replace(Some(0), 6, 3),
            Op::Replace(Some(3), 0, 4),
            Op::Replace(Some(6), 3, 5),
            Op::Insert(9, 6),
        ];
        run(Hashing::Clustered, &ops);
    }

    #[test]
    fn a_located_index_stays_valid_across_the_add_it_was_located_with() {
        let mut set = HashedMultiset::new();
        let hashing = Hashing::Clustered;
        for (k, id) in [(0, 0), (3, 1)] {
            change(&mut set, hashing, None, Some((k, id)));
        }
        let home = hash(hashing, 0) as usize & (set.store.table.capacity - 1);
        assert_eq!(set.find(hash(hashing, 0), &v(0, 0)), Some(home));
        let (removed, displaced) = change(&mut set, hashing, Some(0), Some((6, 2)));
        assert!(removed);
        assert_eq!(displaced, vec![0]);
        assert_eq!(set.store.table.ctrl()[home], TOMBSTONE);
        assert!(set.find(hash(hashing, 6), &v(6, 0)).is_some());
        assert!(set.find(hash(hashing, 3), &v(3, 0)).is_some());
    }

    #[test]
    fn a_write_that_skips_the_settling_lookup_panics_under_a_capture() {
        let mut set = HashedMultiset::new();
        for _ in 0..2 {
            let at = set.locate_add(7, &v(1, 1));
            let _ = set.add(at, v(1, 1));
        }
        set.begin_capture();
        let i = set.find(7, &v(1, 0)).expect("held");
        let text = |r: std::thread::Result<()>| {
            let err = r.expect_err("the write panics");
            err.downcast_ref::<String>().cloned().unwrap_or_default()
        };
        let increment = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = set.add(AddAt::Existing(i), v(1, 2));
        }));
        assert!(text(increment).contains("before the capture took it"));
        let decrement = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = set.decrement_or_remove(i);
        }));
        assert!(text(decrement).contains("before the capture took it"));
        assert_eq!(set.count(i), 2);
    }

    /// Adds an occurrence of key `k` under the hash `h`.
    fn add_at(set: &mut HashedMultiset<V>, h: u64, k: u8) {
        let value = v(k, u32::from(k));
        let at = set.locate_add(h, &value);
        drop(set.add(at, value));
    }

    /// Removes an occurrence of key `k`, held under the hash `h`.
    fn remove_at(set: &mut HashedMultiset<V>, h: u64, k: u8) {
        let i = set.find_settling(h, &v(k, 0)).expect("the key is held");
        drop(set.decrement_or_remove(i));
    }

    /// Twelve keys that fill a 16-slot table to its limit, key `k` in slot
    /// `k`, so that the next insert starts a migration into 32 slots.
    fn full_sixteen() -> HashedMultiset<V> {
        let mut set = HashedMultiset::new();
        for k in 0..12 {
            add_at(&mut set, u64::from(k), k);
        }
        assert_eq!(set.store.table.capacity, 16);
        assert_eq!(set.store.growth_left, 0);
        set
    }

    #[test]
    fn a_migration_spans_writes_and_keeps_every_entry_reachable() {
        with_step_slots(1, || {
            let mut set = full_sixteen();
            let one_table = set.heap_bytes();
            add_at(&mut set, 12, 12);
            let old = set.store.old().expect("the insert starts a migration");
            assert_eq!(
                (old.cursor, old.base),
                (5, 32),
                "the start scans 4 * 1 + 1 slots"
            );
            assert!(set.heap_bytes() > 2 * one_table);
            for k in 0..13 {
                let i = set.find(u64::from(k), &v(k, 0)).expect("held");
                assert_eq!(i >= 32, (5..12).contains(&k), "key {k} at slot {i}");
                assert_eq!((set.stored(i).key, set.count(i)), (k, 1));
            }
            assert_eq!(set.find(13, &v(13, 0)), None);
            assert_eq!(set.find(3, &v(35, 0)), None);

            remove_at(&mut set, 10, 10);
            assert_eq!(set.store.old().map(|old| old.cursor), Some(6));
            assert_eq!(
                set.find(10, &v(10, 0)),
                None,
                "removed from the old storage"
            );
            assert_eq!(set.len(), 12);
            assert_accounting(&set);
            add_at(&mut set, 11, 11);
            assert_eq!(set.store.old().map(|old| old.cursor), Some(11));
            let i = set.find(11, &v(11, 0)).expect("held");
            assert_eq!(
                (i, set.count(i)),
                (32 + 11, 2),
                "counted in the old storage"
            );
            assert_accounting(&set);

            let mut steps = 0;
            while set.store.migrating() {
                set.reserve(0);
                steps += 1;
                assert_accounting(&set);
            }
            assert_eq!(steps, 5, "a reserve of nothing scans one slot");
            for k in (0..10).chain([11, 12]) {
                let i = set.find(u64::from(k), &v(k, 0)).expect("held");
                assert!(i < 32, "key {k} moved into the table");
                assert_eq!(set.count(i), if k == 11 { 2 } else { 1 });
            }
            assert_eq!(set.heap_bytes(), 2 * one_table);
        });
    }

    #[test]
    fn a_moved_entry_does_not_take_a_stale_settled_bit_from_its_new_slot() {
        with_step_slots(1, || {
            let mut set = full_sixteen();
            set.begin_capture();
            add_at(&mut set, 12, 12);
            add_at(&mut set, 11, 50);
            assert_eq!(
                set.find(11, &v(50, 0)),
                Some(11),
                "in the table, while key 11 is still in the old storage"
            );
            remove_at(&mut set, 11, 50);
            assert_eq!(set.find(11, &v(11, 0)), Some(32 + 11));
            set.reserve(0);
            assert_eq!(
                set.find(11, &v(11, 0)),
                Some(11),
                "moved into the slot a settled entry left"
            );
            let mut keys = Vec::new();
            loop {
                let (end, run) = set
                    .capture_scan(4, Instant::now(), Duration::ZERO)
                    .expect("a capture is installed");
                let step = set.capture_advance(end, run);
                keys.extend(step.rows.iter().map(|(value, count)| (value.key, *count)));
                if step.done {
                    break;
                }
            }
            keys.sort_unstable();
            let begun: Vec<(u8, usize)> = (0..12).map(|k| (k, 1)).collect();
            assert_eq!(keys, begun);
        });
    }

    #[test]
    fn growth_rehash_keeps_every_entry() {
        let mut model = Model::new();
        let mut set = HashedMultiset::new();
        for k in 0..24 {
            let got = change(&mut set, Hashing::Clustered, None, Some((k, u32::from(k))));
            assert_eq!(got, model_change(&mut model, None, Some((k, u32::from(k)))));
        }
        assert_matches(&set, Hashing::Clustered, &model);
        assert!(set.store.table.capacity >= 32);
    }
}
