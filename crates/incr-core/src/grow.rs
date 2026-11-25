//! Growth by migration, shared by `KeyTable` and `HashedMultiset`.
//!
//! A table that needs more room, or its tombstones purged, gets a second
//! storage: `table`, where every insert goes from then on, next to `old`,
//! the storage it had. Every `reserve` made while the two exist, the one
//! that starts the migration included, moves the entries of the next few
//! old slots into `table` by their stored hashes, so no single call moves
//! a whole table. With `a` the keys a call makes room for, it scans at
//! most `8 * a + 256` old slots: `4 * a + 256` by pace, and under `4 * a`
//! more when it has to finish one migration before starting the next.
//!
//! The old storage's memory goes back as the migration passes it, where
//! [`crate::pages`] can release it, so ending a migration frees little.
//!
//! Slots `0..table.capacity()` name `table` and the slots from there on
//! name `old`, so an index into `table` costs what it does with no
//! migration, and a start renumbers every held slot. [`Moves`] reports what
//! one call did to the slots held before it.

#[cfg(test)]
use std::cell::Cell;
use std::mem;

use crate::capture::Capture;
use crate::yield_hook::Ticks;

const STEP_SLOTS: usize = 256;
const STEP_SLOTS_PER_KEY: usize = 4;

#[cfg(test)]
thread_local! {
    static TEST_STEP_SLOTS: Cell<usize> = const { Cell::new(STEP_SLOTS) };
}

/// Runs `f` with every migration step on this thread scanning `slots` old
/// slots plus four per key, so that tables of a few slots take several
/// writes to migrate.
#[cfg(test)]
pub(crate) fn with_step_slots<T>(slots: usize, f: impl FnOnce() -> T) -> T {
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_STEP_SLOTS.set(self.0);
        }
    }
    let _restore = Restore(TEST_STEP_SLOTS.replace(slots));
    f()
}

fn step_slots() -> usize {
    #[cfg(test)]
    {
        TEST_STEP_SLOTS.get()
    }
    #[cfg(not(test))]
    {
        STEP_SLOTS
    }
}
const MIN_CAPACITY: usize = 8;
const MIGRATING: usize = 1 << (usize::BITS - 1);

pub(crate) fn load_limit(capacity: usize) -> usize {
    capacity / 4 * 3
}

/// One storage of an open-addressing table that keeps each entry's hash.
/// None of its methods runs user code.
pub(crate) trait Storage {
    type Entry;

    /// `capacity` never-used slots, `capacity` a power of two.
    fn zeroed(capacity: usize) -> Self;

    fn capacity(&self) -> usize;

    /// Takes the entry in slot `k`, leaving a tombstone; `None` for a free
    /// slot.
    fn take(&mut self, k: usize) -> Option<Self::Entry>;

    /// Moves the entry of each full slot `k` in `from..to` into the first
    /// free slot of its probe run in `into`, calling `moved(k, slot)`.
    /// Leaves every slot of `from..to` free. Returns the entries moved and
    /// how many of them landed on a tombstone rather than a never-used
    /// slot. Requires a never-used slot in `into` after each move.
    fn move_range(
        &mut self,
        from: usize,
        to: usize,
        into: &mut Self,
        moved: impl FnMut(usize, usize),
    ) -> (usize, usize);

    /// Drops every entry from slot `from` on.
    fn drop_entries(&mut self, from: usize);

    /// Gives back what it can of the memory of slots `from..to`. No slot
    /// below `to` holds an entry or is read again, and the calls on one
    /// storage cover consecutive ranges from slot 0.
    fn release(&mut self, from: usize, to: usize);
}

/// What one `reserve` did to the slots of the entries held before it: the
/// moves of the migration that was running, then the renumbering of a
/// migration it started, then that migration's first moves.
#[derive(Debug, Default)]
pub(crate) struct Moves {
    stepped: Vec<(usize, usize)>,
    shift: usize,
    restepped: Vec<(usize, usize)>,
}

impl Moves {
    /// The slot, after the `reserve`, of the entry that was in `slot`
    /// before it.
    pub(crate) fn remap(&self, slot: usize) -> usize {
        moved(&self.restepped, moved(&self.stepped, slot) + self.shift)
    }
}

/// `steps` is sorted by its first field, since a migration moves entries
/// in slot order.
fn moved(steps: &[(usize, usize)], slot: usize) -> usize {
    match steps.binary_search_by_key(&slot, |&(from, _)| from) {
        Ok(at) => steps[at].1,
        Err(_) => slot,
    }
}

struct Migration<S> {
    old: S,
    /// No old slot below it holds an entry.
    cursor: usize,
}

impl<S: Storage> Migration<S> {
    /// Moves the entries of the next `budget` old slots into `table`.
    fn step<O>(
        &mut self,
        table: &mut S,
        budget: usize,
        capture: &mut Option<Box<Capture<O>>>,
        moves: Option<&mut Vec<(usize, usize)>>,
        yielder: &impl Ticks,
    ) -> Stepped {
        let base = table.capacity();
        let from = self.cursor;
        let end = from + budget.min(self.old.capacity() - from);
        let (moved, reused) = match (capture.as_deref_mut(), moves) {
            (None, None) => self.old.move_range(from, end, table, |_, _| yielder.tick()),
            (mut capture, mut moves) => self.old.move_range(from, end, table, |k, to| {
                yielder.tick();
                if let Some(capture) = capture.as_deref_mut() {
                    capture.moved(base + k, to);
                }
                if let Some(moves) = moves.as_deref_mut() {
                    moves.push((base + k, to));
                }
            }),
        };
        let stepped = Stepped {
            #[cfg(test)]
            scanned: end - from,
            moved,
            reused,
        };
        self.cursor = end;
        self.old.release(from, end);
        stepped
    }

    fn done(&self) -> bool {
        self.cursor == self.old.capacity()
    }
}

/// What one migration step did: the old slots it scanned, the entries it
/// moved, and how many of those landed on a tombstone, which uses up none
/// of the table's room.
struct Stepped {
    #[cfg(test)]
    scanned: usize,
    moved: usize,
    reused: usize,
}

/// The old storage of a running migration.
pub(crate) struct Old<'a, S> {
    pub(crate) storage: &'a S,
    /// No slot of `storage` below it holds an entry.
    pub(crate) cursor: usize,
    /// The table slot that names slot 0 of `storage`.
    pub(crate) base: usize,
}

pub(crate) struct Growing<S: Storage> {
    /// The storage every insert goes to.
    pub(crate) table: S,
    /// How many never-used slots of `table` may still fill before `grow`
    /// has to run: its load limit, less its full and tombstone slots and
    /// the entries still in the old storage. [`MIGRATING`] is set in it
    /// exactly while `migration` is `Some`, so that one signed compare
    /// answers [`Self::must_grow`].
    growth_left: usize,
    migration: Option<Box<Migration<S>>>,
    #[cfg(test)]
    scanned: usize,
}

impl<S: Storage> Growing<S> {
    pub(crate) fn new(table: S) -> Self {
        Self {
            table,
            growth_left: 0,
            migration: None,
            #[cfg(test)]
            scanned: 0,
        }
    }

    /// Whether a `reserve` of `additional` has to call [`Self::grow`].
    /// `additional` counts entries that fit in memory, so it is below
    /// `isize::MAX`.
    #[inline]
    pub(crate) fn must_grow(&self, additional: usize) -> bool {
        debug_assert!(additional <= isize::MAX as usize);
        additional as isize > self.growth_left as isize
    }

    #[inline]
    pub(crate) fn migrating(&self) -> bool {
        (self.growth_left as isize) < 0
    }

    /// How many never-used slots of `table` may still fill before `grow`
    /// has to run.
    #[inline]
    pub(crate) fn room(&self) -> usize {
        self.growth_left & !MIGRATING
    }

    /// Records an insert into a never-used slot of `table`. Requires room.
    #[inline]
    pub(crate) fn use_room(&mut self) {
        debug_assert!(self.room() > 0);
        self.growth_left -= 1;
    }

    /// Records `n` slots of `table` or of the old storage that no longer
    /// count against the room.
    #[inline]
    pub(crate) fn give_room(&mut self, n: usize) {
        self.growth_left += n;
    }

    fn set_migration(&mut self, migration: Option<Box<Migration<S>>>) {
        self.growth_left = match migration {
            Some(_) => self.growth_left | MIGRATING,
            None => self.room(),
        };
        self.migration = migration;
    }

    pub(crate) fn old(&self) -> Option<Old<'_, S>> {
        self.migration.as_deref().map(|migration| Old {
            storage: &migration.old,
            cursor: migration.cursor,
            base: self.table.capacity(),
        })
    }

    /// The old storage and the slot of it that table slot `i` names.
    /// Panics unless `i` names an old slot.
    pub(crate) fn old_slot_mut(&mut self, i: usize) -> (&mut S, usize) {
        let base = self.table.capacity();
        match self.migration.as_deref_mut() {
            Some(migration) if i >= base => (&mut migration.old, i - base),
            _ => panic!("incr-core: table slot {i} is out of range"),
        }
    }

    pub(crate) fn old_capacity(&self) -> usize {
        self.migration
            .as_deref()
            .map_or(0, |migration| migration.old.capacity())
    }

    /// The old slots `grow` has scanned for entries to move, over the
    /// table's life.
    #[cfg(test)]
    pub(crate) fn scanned(&self) -> usize {
        self.scanned
    }

    fn count(&mut self, stepped: &Stepped) {
        self.give_room(stepped.reused);
        #[cfg(test)]
        {
            self.scanned += stepped.scanned;
        }
    }

    /// Makes room for `additional` more entries in a table of `len`, and
    /// takes a step of the running migration. Runs no user code. Returns
    /// whether a slot or a scan position changed its meaning: an entry
    /// moved, or a migration started or ended.
    #[cold]
    #[inline(never)]
    pub(crate) fn grow<O>(
        &mut self,
        len: usize,
        additional: usize,
        capture: &mut Option<Box<Capture<O>>>,
        mut moves: Option<&mut Moves>,
        yielder: &impl Ticks,
    ) -> bool {
        let pace = STEP_SLOTS_PER_KEY
            .saturating_mul(additional)
            .saturating_add(step_slots());
        let mut changed = false;
        let room = self.room();
        if let Some(migration) = self.migration.as_deref_mut() {
            let budget = if additional > room { usize::MAX } else { pace };
            let stepped = moves.as_deref_mut().map(|moves| &mut moves.stepped);
            let stepped = migration.step(&mut self.table, budget, capture, stepped, yielder);
            changed = stepped.moved > 0;
            let done = migration.done();
            self.count(&stepped);
            if done {
                self.set_migration(None);
                finish(capture);
                changed = true;
            }
        }
        if additional > self.room() {
            let capacity = next_capacity(self.table.capacity(), len, additional);
            let old = mem::replace(&mut self.table, S::zeroed(capacity));
            self.growth_left = load_limit(capacity) - len;
            if let Some(capture) = capture.as_deref_mut() {
                capture.migration_started(capacity);
            }
            let restepped = moves.map(|moves| {
                moves.shift = capacity;
                &mut moves.restepped
            });
            if len == 0 {
                finish(capture);
            } else {
                let mut migration = Migration { old, cursor: 0 };
                let stepped = migration.step(&mut self.table, pace, capture, restepped, yielder);
                self.count(&stepped);
                if migration.done() {
                    finish(capture);
                } else {
                    self.set_migration(Some(Box::new(migration)));
                }
            }
            changed = true;
        }
        changed
    }
}

fn finish<O>(capture: &mut Option<Box<Capture<O>>>) {
    if let Some(capture) = capture.as_deref_mut() {
        capture.migration_finished();
    }
}

/// The capacity of the storage that replaces one of `capacity` slots
/// holding `len` entries when `additional` more must fit: the same, to
/// purge tombstones, when that frees at least half the load limit, which
/// keeps growth amortised O(1), and otherwise at least twice it.
fn next_capacity(capacity: usize, len: usize, additional: usize) -> usize {
    let overflow = "incr-core: collection capacity overflow";
    let need = len.checked_add(additional).expect(overflow);
    let limit = load_limit(capacity);
    if need <= limit / 2 {
        return capacity;
    }
    let need = need.max(limit + 1);
    let mut capacity = MIN_CAPACITY;
    while load_limit(capacity) < need {
        capacity = capacity.checked_mul(2).expect(overflow);
    }
    capacity
}

impl<S: Storage> Drop for Growing<S> {
    fn drop(&mut self) {
        self.table.drop_entries(0);
        if let Some(migration) = self.migration.as_deref_mut() {
            migration.old.drop_entries(migration.cursor);
        }
    }
}
