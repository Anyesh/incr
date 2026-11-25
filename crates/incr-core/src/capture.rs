//! The state a table keeps while a capture of its contents as of one
//! instant is installed: see [`crate::KeyedCollection::begin_capture`] and
//! [`crate::IncrCollection::begin_capture`].
//!
//! A slot is settled when its scan position is below the cursor or its bit
//! is set. The capture holds the content, as of the instant it was
//! installed, of every settled slot that was occupied then. Every write
//! that changes an occupied slot first settles it, cloning its content into
//! the capture; filling a free slot settles it with no copy; the scan
//! copies the unsettled occupied slots from the cursor on.
//!
//! The scan visits the old storage of a migration (see [`crate::grow`])
//! before the table's, so an entry that a migration step moves is never
//! behind the cursor unless it was settled: the step gives its new slot the
//! settled state of its old one.

use std::time::{Duration, Instant};

/// Table slots a capture scan copies between two reads of the clock.
pub(crate) const CAPTURE_UNIT: usize = 4096;

/// What one capture step hands over.
#[derive(Debug)]
pub struct CaptureStep<T> {
    /// Content the capture took since the previous step, by the scan and
    /// by the writes that settled a slot first, in no particular order.
    pub rows: Vec<T>,
    /// Whether the capture is finished: every slot was taken and the
    /// table no longer keeps capture state.
    pub done: bool,
    /// Content a scan cloned that is not part of the capture, because a
    /// write changed the table between the scan and the step. Handed over
    /// so that its values drop where the caller drops them.
    pub discarded: Vec<T>,
}

pub(crate) struct Capture<O> {
    /// One bit per slot; every bit from `table + old` on is zero.
    bits: Vec<u64>,
    /// A scan position: see [`Self::position`].
    cursor: usize,
    /// The capacity of the table's storage, slots `0..table`.
    table: usize,
    /// The capacity of a migration's old storage, slots `table..table +
    /// old`; zero with no migration.
    old: usize,
    out: Vec<O>,
}

impl<O> Capture<O> {
    /// A capture of a table whose storage has `table` slots and whose
    /// migration, if one runs, has `old` slots left to empty.
    pub(crate) fn new(table: usize, old: usize) -> Box<Self> {
        Box::new(Self {
            bits: vec![0; (table + old).div_ceil(64)],
            cursor: 0,
            table,
            old,
            out: Vec::new(),
        })
    }

    /// Where the scan visits slot `i`: the old storage's slots first, then
    /// the table's.
    #[inline]
    fn position(&self, i: usize) -> usize {
        if i >= self.table {
            i - self.table
        } else {
            i + self.old
        }
    }

    /// The slot the scan visits at `position`.
    #[inline]
    pub(crate) fn slot(&self, position: usize) -> usize {
        if position < self.old {
            position + self.table
        } else {
            position - self.old
        }
    }

    /// The number of scan positions.
    pub(crate) fn len(&self) -> usize {
        self.table + self.old
    }

    #[inline]
    fn bit(&self, i: usize) -> bool {
        self.bits[i / 64] >> (i % 64) & 1 != 0
    }

    #[inline]
    pub(crate) fn settled(&self, i: usize) -> bool {
        self.position(i) < self.cursor || self.bit(i)
    }

    #[inline]
    pub(crate) fn settle(&mut self, i: usize) {
        self.bits[i / 64] |= 1 << (i % 64);
    }

    /// Settles slot `i` with `content`, its content as of the capture,
    /// unless it is settled already, in which case `content` comes back.
    pub(crate) fn take<I: IntoIterator<Item = O>>(&mut self, i: usize, content: I) -> Option<I> {
        if self.settled(i) {
            return Some(content);
        }
        self.settle(i);
        self.out.extend(content);
        None
    }

    /// A migration into a storage of `capacity` slots started, which
    /// renumbers slot `i` to `capacity + i`. The cursor stays: the scan
    /// position of every slot it has passed is unchanged.
    pub(crate) fn migration_started(&mut self, capacity: usize) {
        debug_assert_eq!(self.old, 0);
        let old = self.table;
        let mut bits = vec![0u64; (capacity + old).div_ceil(64)];
        if capacity.is_multiple_of(64) {
            bits[capacity / 64..][..self.bits.len()].copy_from_slice(&self.bits);
        } else {
            for i in (0..old).filter(|&i| self.bit(i)) {
                bits[(capacity + i) / 64] |= 1 << ((capacity + i) % 64);
            }
        }
        self.bits = bits;
        self.table = capacity;
        self.old = old;
    }

    /// A migration step moved the entry in old slot `from` to table slot
    /// `to`, which takes its settled state. The bit is cleared for an
    /// unsettled entry because `to` may keep one from an entry inserted
    /// there and removed since; the scan has yet to reach `to`, so it
    /// still copies the entry.
    pub(crate) fn moved(&mut self, from: usize, to: usize) {
        if self.settled(from) {
            self.settle(to);
        } else {
            self.bits[to / 64] &= !(1 << (to % 64));
        }
    }

    /// The migration ended and its old slots are gone. A scan that had not
    /// left them restarts at the table's first slot, where every entry
    /// carries its settled state in its bit.
    pub(crate) fn migration_finished(&mut self) {
        self.bits.truncate(self.table.div_ceil(64));
        if !self.table.is_multiple_of(64) {
            if let Some(last) = self.bits.last_mut() {
                *last &= (1 << (self.table % 64)) - 1;
            }
        }
        self.cursor = self.cursor.saturating_sub(self.old);
        self.old = 0;
    }

    pub(crate) fn cursor(&self) -> usize {
        self.cursor
    }

    /// Adds a scan's `run`, taken from the cursor up to `end`, and moves
    /// the cursor there.
    pub(crate) fn advance(&mut self, end: usize, run: Vec<O>) {
        debug_assert!(end >= self.cursor);
        if self.out.is_empty() {
            self.out = run;
        } else {
            self.out.extend(run);
        }
        self.cursor = end;
    }

    /// Hands over what was taken since the last call; the capture is
    /// finished once the cursor has passed every scan position.
    pub(crate) fn drain(&mut self) -> CaptureStep<O> {
        CaptureStep {
            rows: std::mem::take(&mut self.out),
            done: self.cursor >= self.len(),
            discarded: Vec::new(),
        }
    }

    pub(crate) fn into_rows(self) -> Vec<O> {
        self.out
    }
}

/// Panics unless slot `i` is settled, for a function about to change the
/// occupied slot `i` while `capture` may be installed. All but the
/// `is_some` test is out of line, so the writers it sits in inline as they
/// did without it.
#[inline(always)]
pub(crate) fn check_settled<O>(capture: &Option<Box<Capture<O>>>, i: usize) {
    if capture.is_some() {
        check_settled_slow(capture, i);
    }
}

#[cold]
#[inline(never)]
fn check_settled_slow<O>(capture: &Option<Box<Capture<O>>>, i: usize) {
    if capture.as_deref().is_some_and(|c| !c.settled(i)) {
        panic!(
            "incr-core: slot {i} changed while a capture is installed before the capture \
             took it; the write path skipped its settling lookup"
        );
    }
}

/// Settles slot `i`, a free slot about to be filled, which held nothing
/// when `capture` was installed. Out of line as [`check_settled`].
#[inline(always)]
pub(crate) fn settle_free<O>(capture: &mut Option<Box<Capture<O>>>, i: usize) {
    if capture.is_some() {
        settle_free_slow(capture, i);
    }
}

#[cold]
#[inline(never)]
fn settle_free_slow<O>(capture: &mut Option<Box<Capture<O>>>, i: usize) {
    if let Some(capture) = capture.as_deref_mut() {
        capture.settle(i);
    }
}

/// Scans the positions from `from` to `capacity`, calling `visit` on each
/// until the end or until `budget` since `start` has passed at a unit
/// boundary, and returns where it stopped. At least one unit is scanned.
pub(crate) fn scan(
    from: usize,
    capacity: usize,
    unit: usize,
    start: Instant,
    budget: Duration,
    mut visit: impl FnMut(usize),
) -> usize {
    let mut i = from;
    while i < capacity {
        let stop = capacity.min(i.saturating_add(unit));
        (i..stop).for_each(&mut visit);
        i = stop;
        if start.elapsed() >= budget {
            break;
        }
    }
    i
}

#[cfg(all(test, not(loom)))]
#[path = "capture_tests.rs"]
pub(crate) mod tests;
