//! An outermost engine entry's attempt, as other threads act on it under
//! `Shared`.
//!
//! One word per thread, kept across the entry's retries, which a waiter
//! reaches through the reservation it meets. Its owner raises
//! [`IN_FLIGHT`] while a doom's unwind is out of the engine's hands: the
//! unwinding frames, or a closure that caught it, may run user code that
//! joins a thread reading a node the doom reserved, which the owner can't
//! compute again before that code returns. A waiter on such an owner sets
//! [`WOUNDED`], and the reservations of a wounded attempt no longer hold
//! anyone: the owner turns its doom's unwind into a wound where the engine
//! catches it, and its entry retries with none of what the attempt kept.
//! Every transition is a CAS on the word, and the bits above [`FLAGS`]
//! count the attempts, so a waiter holding an old attempt's word acts on
//! no later one. A take-over takes no ticket: any waiter may restart an
//! older attempt, once per in-flight discard it meets. Bits 1 and 3 are
//! kept for a waiter's pin of a landed value, which a take-over still
//! wounds (the retry publishes pinned landings as taken), and for the
//! owner's mark of an episode others can see.

#[cfg(not(loom))]
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

#[cfg(loom)]
use loom::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

/// A doom's unwind is in flight: set and cleared by the owner only.
pub(crate) const IN_FLIGHT: u32 = 1;
/// A waiter acted for the attempt: it retries from its entry.
pub(crate) const WOUNDED: u32 = 4;
const FLAGS: u32 = 15;
const EPOCH: u32 = FLAGS + 1;

/// A thread's attempt word, and the attempt it was read for.
#[derive(Clone)]
pub(crate) struct Attempt {
    word: Arc<AtomicU32>,
    epoch: u32,
}

#[cfg(not(loom))]
thread_local! {
    static MINE: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));
}

#[cfg(loom)]
loom::thread_local! {
    static MINE: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));
}

/// This thread's current attempt.
pub(crate) fn mine() -> Attempt {
    MINE.with(|w| Attempt {
        word: w.clone(),
        epoch: w.load(Ordering::Relaxed) & !FLAGS,
    })
}

/// Starts a new attempt, at the outermost entry and at each of its
/// retries: every flag clears, and the epoch moves on.
pub(crate) fn begin() {
    MINE.with(|w| {
        let epoch = (w.load(Ordering::Relaxed) & !FLAGS).wrapping_add(EPOCH);
        w.store(epoch, Ordering::Release);
    });
}

/// The owner raises a doom's unwind.
pub(crate) fn enter_flight() {
    MINE.with(|w| w.fetch_or(IN_FLIGHT, Ordering::AcqRel));
}

/// The engine has caught this thread's doom unwind: whether a waiter
/// wounded the attempt while it was in flight.
pub(crate) fn leave_flight() -> bool {
    MINE.with(|w| w.fetch_and(!IN_FLIGHT, Ordering::AcqRel) & WOUNDED != 0)
}

impl Attempt {
    fn current(&self, word: u32) -> bool {
        word & !FLAGS == self.epoch
    }

    /// Whether this attempt was wounded; false once it has ended.
    pub(crate) fn wounded(&self) -> bool {
        let w = self.word.load(Ordering::Acquire);
        self.current(w) && w & WOUNDED != 0
    }

    /// A waiter on this attempt: wounds it if its doom's unwind is in
    /// flight and nothing pins it. True if it is wounded now.
    pub(crate) fn wound_in_flight(&self) -> bool {
        let mut w = self.word.load(Ordering::Acquire);
        loop {
            if !self.current(w) {
                return false;
            }
            if w & WOUNDED != 0 {
                return true;
            }
            if w & IN_FLIGHT == 0 {
                return false;
            }
            match self.word.compare_exchange_weak(
                w,
                w | WOUNDED,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    #[cfg(all(test, not(loom)))]
                    TAKE_OVERS.fetch_add(1, Ordering::Relaxed);
                    return true;
                }
                Err(now) => w = now,
            }
        }
    }
}

/// Whether this thread's attempt has a discard in flight, for tests.
#[cfg(all(test, not(loom)))]
pub(crate) fn in_flight() -> bool {
    MINE.with(|w| w.load(Ordering::Relaxed) & IN_FLIGHT != 0)
}

/// Attempts a waiter wounded, for tests and the scans' counts.
#[cfg(all(test, not(loom)))]
pub(crate) static TAKE_OVERS: AtomicU32 = AtomicU32::new(0);
