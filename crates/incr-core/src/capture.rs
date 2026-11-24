//! The state a table keeps while a capture of its contents as of one
//! instant is installed: see [`crate::KeyedCollection::begin_capture`] and
//! [`crate::IncrCollection::begin_capture`].
//!
//! A slot is settled when it is below the cursor or its bit is set. The
//! capture holds the content, as of the instant it was installed, of every
//! settled slot that was occupied then. Every write that changes an
//! occupied slot first settles it, cloning its content into the capture;
//! filling a free slot settles it with no copy; a rehash carries each
//! entry's settled state to its new slot and resets the cursor; the scan
//! copies the unsettled occupied slots from the cursor on.

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
    bits: Vec<u64>,
    cursor: usize,
    out: Vec<O>,
}

impl<O> Capture<O> {
    pub(crate) fn new(capacity: usize) -> Box<Self> {
        Box::new(Self {
            bits: vec![0; capacity.div_ceil(64)],
            cursor: 0,
            out: Vec::new(),
        })
    }

    #[inline]
    pub(crate) fn settled(&self, i: usize) -> bool {
        i < self.cursor || self.bits[i / 64] >> (i % 64) & 1 != 0
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

    /// After a rehash into `capacity` slots that moved old slot `i` to
    /// `moved[i]` (`usize::MAX` for a free one): carries each moved entry's
    /// settled state and restarts the scan.
    pub(crate) fn rehashed(&mut self, capacity: usize, moved: &[usize]) {
        let mut bits = vec![0u64; capacity.div_ceil(64)];
        for (old, &new) in moved.iter().enumerate() {
            if new != usize::MAX && self.settled(old) {
                bits[new / 64] |= 1 << (new % 64);
            }
        }
        self.bits = bits;
        self.cursor = 0;
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
    /// finished once the cursor reaches `capacity`.
    pub(crate) fn drain(&mut self, capacity: usize) -> CaptureStep<O> {
        CaptureStep {
            rows: std::mem::take(&mut self.out),
            done: self.cursor >= capacity,
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

/// Scans `capacity` slots from `from`, calling `visit` on each until the
/// end or until `budget` since `start` has passed at a unit boundary, and
/// returns where it stopped. At least one unit is scanned.
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
