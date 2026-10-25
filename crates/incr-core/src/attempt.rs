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
//! older attempt, once per in-flight discard it meets, and lists the claim
//! it met before its wound: the owner's read never dooms or reserves that
//! claim again, so a read restarts for take-overs at most once per claim it
//! can reserve, however often it is doomed. A waiter that took
//! a value the attempt landed sets [`PINNED`]; a take-over still wounds a
//! pinned attempt, whose retry publishes its pinned landings as taken
//! rather than retract a value another thread has returned. Bit 3 is kept
//! for the owner's mark of an episode others can see.

#[cfg(not(loom))]
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex, PoisonError,
};

#[cfg(loom)]
use loom::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex,
};
#[cfg(loom)]
use std::sync::PoisonError;

use crate::waits::Claim;

/// A doom's unwind is in flight: set and cleared by the owner only.
pub(crate) const IN_FLIGHT: u32 = 1;
/// A waiter holds a value the attempt landed, to the attempt's end.
pub(crate) const PINNED: u32 = 2;
/// A waiter acted for the attempt: it retries from its entry.
pub(crate) const WOUNDED: u32 = 4;
const FLAGS: u32 = 15;
const EPOCH: u32 = FLAGS + 1;

/// Checks of one open flight after which a waiter takes it for an owner
/// held in user code. Waiting and taking over are both serializable, so
/// this and `STUCK_FOR` choose only the order: an owner still in the
/// engine's own unwind of a discard closes its flight sooner and keeps
/// its place, rather than restarting behind the waiter.
#[cfg(not(loom))]
const STUCK_AFTER: u32 = crate::waits::REGISTER_AFTER;
#[cfg(loom)]
const STUCK_AFTER: u32 = 2;

/// How long one flight must also stay open: an owner descheduled in its
/// unwind looks the same as one held in user code by checks alone. A
/// take-over of it is serializable too, but puts the slow owner's read
/// after the waiter's rather than in the order its start gave.
#[cfg(not(loom))]
const STUCK_FOR: std::time::Duration = std::time::Duration::from_millis(10);

/// The word, a count of the owner's flights, and the claims waiters met
/// as they took the flight over, with the epoch of each.
struct Word {
    state: AtomicU32,
    flights: AtomicU32,
    met: Mutex<Vec<(u32, Claim)>>,
}

/// A thread's attempt word, and the attempt it was read for.
#[derive(Clone)]
pub(crate) struct Attempt {
    word: Arc<Word>,
    epoch: u32,
}

fn new_word() -> Arc<Word> {
    Arc::new(Word {
        state: AtomicU32::new(0),
        flights: AtomicU32::new(0),
        met: Mutex::new(Vec::new()),
    })
}

#[cfg(not(loom))]
thread_local! {
    static MINE: Arc<Word> = new_word();
}

#[cfg(loom)]
loom::thread_local! {
    static MINE: Arc<Word> = new_word();
}

/// This thread's current attempt.
pub(crate) fn mine() -> Attempt {
    MINE.with(|w| Attempt {
        word: w.clone(),
        epoch: w.state.load(Ordering::Relaxed) & !FLAGS,
    })
}

/// Starts a new attempt, at the outermost entry and at each of its
/// retries: every flag clears, and the epoch moves on.
pub(crate) fn begin() {
    MINE.with(|w| {
        let epoch = (w.state.load(Ordering::Relaxed) & !FLAGS).wrapping_add(EPOCH);
        w.state.store(epoch, Ordering::Release);
    });
}

/// The owner raises a discard's unwind.
pub(crate) fn enter_flight() {
    MINE.with(|w| {
        w.flights.fetch_add(1, Ordering::Relaxed);
        w.state.fetch_or(IN_FLIGHT, Ordering::AcqRel);
    });
}

/// The engine has caught this thread's discard: whether a waiter wounded
/// the attempt while it was in flight.
pub(crate) fn leave_flight() -> bool {
    MINE.with(|w| w.state.fetch_and(!IN_FLIGHT, Ordering::AcqRel) & WOUNDED != 0)
}

/// The claims waiters met as they took over this thread's current
/// attempt, read by its owner once it has seen the wound: each waiter
/// lists its claim before the CAS that wounds.
pub(crate) fn take_met() -> Vec<Claim> {
    MINE.with(|w| {
        let epoch = w.state.load(Ordering::Relaxed) & !FLAGS;
        let mut met = w.met.lock().unwrap_or_else(PoisonError::into_inner);
        met.drain(..)
            .filter_map(|(e, claim)| (e == epoch).then_some(claim))
            .collect()
    })
}

/// A waiter's view of the flight it has seen open, and for how many of
/// its checks.
#[derive(Default)]
pub(crate) struct Patience {
    seen: Option<(usize, u32, u32)>,
    checks: u32,
    #[cfg(not(loom))]
    since: Option<std::time::Instant>,
}

impl Attempt {
    fn current(&self, word: u32) -> bool {
        word & !FLAGS == self.epoch
    }

    /// Whether this attempt was wounded; false once it has ended.
    pub(crate) fn wounded(&self) -> bool {
        let w = self.word.state.load(Ordering::Acquire);
        self.current(w) && w & WOUNDED != 0
    }

    /// A waiter on this attempt, which met it at `claim`: wounds it once
    /// one flight has stayed open through [`STUCK_AFTER`] of the waiter's
    /// checks and for `STUCK_FOR`. True if it is wounded now.
    pub(crate) fn wound_if_stuck(&self, claim: Claim, patience: &mut Patience) -> bool {
        let w = self.word.state.load(Ordering::Acquire);
        if !self.current(w) {
            return false;
        }
        if w & WOUNDED != 0 {
            return true;
        }
        if w & IN_FLIGHT == 0 {
            patience.seen = None;
            return false;
        }
        let flight = (
            Arc::as_ptr(&self.word) as usize,
            self.epoch,
            self.word.flights.load(Ordering::Relaxed),
        );
        if patience.seen != Some(flight) {
            *patience = Patience {
                seen: Some(flight),
                checks: 0,
                #[cfg(not(loom))]
                since: Some(std::time::Instant::now()),
            };
            return false;
        }
        patience.checks += 1;
        #[cfg(not(loom))]
        if patience.since.is_some_and(|t| t.elapsed() < STUCK_FOR) {
            return false;
        }
        patience.checks >= STUCK_AFTER && self.wound_in_flight(claim)
    }

    /// A waiter on this attempt, which met it at `claim`: wounds it if its
    /// discard is in flight, pinned or not, since its owner may be in user
    /// code that waits on the waiter. True if it is wounded now.
    pub(crate) fn wound_in_flight(&self, claim: Claim) -> bool {
        let mut w = self.word.state.load(Ordering::Acquire);
        let mut listed = false;
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
            if !listed {
                let mut met = self.word.met.lock().unwrap_or_else(PoisonError::into_inner);
                // Rechecked under the lock: a waiter of a later attempt that
                // listed first is seen here, and its entry is kept.
                if !self.current(self.word.state.load(Ordering::Acquire)) {
                    return false;
                }
                met.retain(|&(e, _)| e == self.epoch);
                met.push((self.epoch, claim));
                listed = true;
            }
            match self.word.state.compare_exchange_weak(
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

impl Attempt {
    /// A waiter takes a value this attempt landed: pins it unless it is
    /// wounded already, whose landings its retry may retract. True if it
    /// is pinned now.
    #[cfg_attr(
        any(not(test), loom),
        expect(
            dead_code,
            reason = "the landing hand-over of T8-44e3 is its first caller"
        )
    )]
    pub(crate) fn pin(&self) -> bool {
        let mut w = self.word.state.load(Ordering::Acquire);
        loop {
            if !self.current(w) || w & WOUNDED != 0 {
                return false;
            }
            if w & PINNED != 0 {
                return true;
            }
            match self.word.state.compare_exchange_weak(
                w,
                w | PINNED,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(now) => w = now,
            }
        }
    }

    /// The flags of this attempt's word; none once it has ended.
    #[cfg(all(test, not(loom)))]
    fn flags(&self) -> u32 {
        let w = self.word.state.load(Ordering::Acquire);
        if self.current(w) {
            w & FLAGS
        } else {
            0
        }
    }
}

/// Whether this thread's attempt has a discard in flight, for tests.
#[cfg(all(test, not(loom)))]
pub(crate) fn in_flight() -> bool {
    MINE.with(|w| w.state.load(Ordering::Relaxed) & IN_FLIGHT != 0)
}

/// Attempts a waiter wounded, for tests and the scans' counts.
#[cfg(all(test, not(loom)))]
pub(crate) static TAKE_OVERS: AtomicU32 = AtomicU32::new(0);

#[cfg(all(test, not(loom)))]
#[path = "attempt_tests.rs"]
mod tests;
