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
//! rather than retract a value another thread has returned. While the
//! attempt holds landings (`crate::hold`), which other threads see as
//! `Computing`, the owner raises [`HOLDING`]: it may sit in user code that
//! waits on a reader of one, and a waiter that wounds it takes over the
//! landing it waits on (`Runtime::hand_over`).

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
/// The attempt holds landings: set and cleared by the owner only.
pub(crate) const HOLDING: u32 = 8;
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

/// How long one flight must also stay open with its owner outside the
/// engine: an owner descheduled in its unwind looks the same as one held
/// in user code by checks alone. A
/// take-over of it is serializable too, but puts the slow owner's read
/// after the waiter's rather than in the order its start gave.
#[cfg(not(loom))]
const STUCK_FOR: std::time::Duration = std::time::Duration::from_millis(10);

/// The word, a count of the owner's flights, and the claims waiters met
/// as they took the flight over, with the epoch of each.
struct Word {
    state: AtomicU32,
    flights: AtomicU32,
    /// Nonzero while the owner runs user code or waits, the only places
    /// it cannot move on its own: written by the owner alone, so a plain
    /// load and store, and a std atomic under loom too, where waiters
    /// ignore it.
    outside: std::sync::atomic::AtomicU32,
    met: Mutex<Vec<(u32, Claim)>>,
    /// The owner is stopped in a test's park hook, which stands for an
    /// engine step however long the test holds it: no holder is stuck there.
    #[cfg(all(test, not(loom)))]
    parked: AtomicU32,
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
        outside: std::sync::atomic::AtomicU32::new(0),
        met: Mutex::new(Vec::new()),
        #[cfg(all(test, not(loom)))]
        parked: AtomicU32::new(0),
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

/// This thread's attempt word, which outlives its attempts.
pub(crate) fn word_id() -> usize {
    MINE.with(|w| Arc::as_ptr(w) as usize)
}

/// While it lives, this thread is outside the engine (`true`: in user code
/// or a wait) or back in it (`false`), as a take-over judges its owner;
/// the drop puts the enclosing state back.
pub(crate) struct Outside(u32);

impl Outside {
    #[inline(always)]
    pub(crate) fn enter(outside: bool) -> Self {
        MINE.with(|w| {
            let was = w.outside.load(std::sync::atomic::Ordering::Relaxed);
            w.outside
                .store(u32::from(outside), std::sync::atomic::Ordering::Relaxed);
            Self(was)
        })
    }
}

impl Drop for Outside {
    #[inline(always)]
    fn drop(&mut self) {
        MINE.with(|w| {
            w.outside
                .store(self.0, std::sync::atomic::Ordering::Relaxed)
        });
    }
}

#[cfg(all(test, not(loom)))]
pub(crate) fn outside_now() -> bool {
    MINE.with(|w| w.outside.load(std::sync::atomic::Ordering::Relaxed) != 0)
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

/// The engine raises a discard it caught again, out of its hands: a
/// flight of its own, unless one is open.
pub(crate) fn reopen_flight() {
    MINE.with(|w| {
        if w.state.load(Ordering::Relaxed) & IN_FLIGHT == 0 {
            w.flights.fetch_add(1, Ordering::Relaxed);
            w.state.fetch_or(IN_FLIGHT, Ordering::AcqRel);
        }
    });
}

/// The owner opens a held landing: its attempt, now holding. Its flights
/// count moves on with this and every [`hold_progress`], so a waiter's
/// patience runs out only on a holder that makes no engine progress.
pub(crate) fn enter_hold() -> Attempt {
    MINE.with(|w| {
        w.flights.fetch_add(1, Ordering::Relaxed);
        let was = w.state.fetch_or(HOLDING, Ordering::AcqRel);
        Attempt {
            word: w.clone(),
            epoch: was & !FLAGS,
        }
    })
}

/// The holding owner made engine progress: a waiter's patience restarts.
pub(crate) fn hold_progress() {
    MINE.with(|w| {
        w.flights.fetch_add(1, Ordering::Relaxed);
    });
}

/// The owner's held landings end: whether a waiter wounded the attempt
/// while it held them.
pub(crate) fn leave_hold() -> bool {
    MINE.with(|w| w.state.fetch_and(!HOLDING, Ordering::AcqRel) & WOUNDED != 0)
}

/// Whether a waiter wounded this thread's current attempt.
pub(crate) fn wounded() -> bool {
    MINE.with(|w| w.state.load(Ordering::Acquire) & WOUNDED != 0)
}

/// Whether a waiter wounded this thread's attempt while it holds landings.
pub(crate) fn hold_wounded() -> bool {
    MINE.with(|w| w.state.load(Ordering::Acquire) & (HOLDING | WOUNDED) == HOLDING | WOUNDED)
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
    /// The time the flight has stayed open with its owner outside the
    /// engine, and the last check that saw it outside.
    #[cfg(not(loom))]
    outside_for: std::time::Duration,
    #[cfg(not(loom))]
    last_outside: Option<std::time::Instant>,
}

/// A waiter's patience that has already seen this thread's open flight
/// through all but one of [`STUCK_AFTER`] checks, for a model's waiter to
/// reach the wound within a small preemption bound.
#[cfg(all(test, loom))]
pub(crate) fn patience_one_check_short() -> Patience {
    MINE.with(|w| Patience {
        seen: Some((
            Arc::as_ptr(w) as usize,
            w.state.load(Ordering::Relaxed) & !FLAGS,
            w.flights.load(Ordering::Relaxed),
        )),
        checks: STUCK_AFTER - 1,
    })
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
        #[cfg(all(test, not(loom)))]
        let parked_holder = w & IN_FLIGHT == 0 && self.word.parked.load(Ordering::Acquire) != 0;
        #[cfg(any(not(test), loom))]
        let parked_holder = false;
        if w & (IN_FLIGHT | HOLDING) == 0 || parked_holder {
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
                ..Patience::default()
            };
            return false;
        }
        // An owner in the engine moves on its own, however long one of its
        // steps takes: only checks that find it in user code or a wait
        // count, and only the time between two such checks.
        #[cfg(not(loom))]
        {
            if self.word.outside.load(std::sync::atomic::Ordering::Relaxed) == 0 {
                patience.last_outside = None;
                return false;
            }
            let now = std::time::Instant::now();
            if let Some(last) = patience.last_outside.replace(now) {
                patience.outside_for += now - last;
            }
        }
        patience.checks += 1;
        #[cfg(not(loom))]
        if patience.outside_for < STUCK_FOR {
            return false;
        }
        patience.checks >= STUCK_AFTER && self.take_over(claim, Some(flight.2))
    }

    /// A waiter on this attempt, which met it at `claim`: wounds it if its
    /// discard is in flight or it holds landings, pinned or not, since its
    /// owner may be in user code that waits on the waiter, unless the
    /// owner waits in the registry older than the waiter
    /// ([`crate::waits::take_over_in_order`]). True if it is wounded now.
    #[cfg(test)]
    pub(crate) fn wound_in_flight(&self, claim: Claim) -> bool {
        self.take_over(claim, None)
    }

    /// [`Self::wound_in_flight`], refused if the owner's flights count has
    /// moved from `seen` when the registry lock is taken.
    fn take_over(&self, claim: Claim, seen: Option<u32>) -> bool {
        let w = self.word.state.load(Ordering::Acquire);
        if !self.current(w) || w & (IN_FLIGHT | HOLDING) == 0 {
            return false;
        }
        if w & WOUNDED != 0 {
            return true;
        }
        crate::waits::take_over_in_order(
            Arc::as_ptr(&self.word) as usize,
            crate::cycle_state::ticket_now(),
            || {
                seen.is_none_or(|f| self.word.flights.load(Ordering::Relaxed) == f)
                    && self.wound_now(claim)
            },
        )
    }

    fn wound_now(&self, claim: Claim) -> bool {
        let mut w = self.word.state.load(Ordering::Acquire);
        let mut listed = false;
        loop {
            if !self.current(w) {
                return false;
            }
            if w & WOUNDED != 0 {
                return true;
            }
            if w & (IN_FLIGHT | HOLDING) == 0 {
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
                    #[cfg(test)]
                    TAKE_OVERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

/// Whether this thread's attempt has a discard in flight, for tests and
/// debug checks.
pub(crate) fn in_flight() -> bool {
    MINE.with(|w| w.state.load(Ordering::Relaxed) & IN_FLIGHT != 0)
}

/// While it lives, this thread is stopped in a test's park hook.
#[cfg(all(test, not(loom)))]
pub(crate) struct Parked;

#[cfg(all(test, not(loom)))]
impl Parked {
    pub(crate) fn new() -> Self {
        MINE.with(|w| w.parked.fetch_add(1, Ordering::AcqRel));
        Self
    }
}

#[cfg(all(test, not(loom)))]
impl Drop for Parked {
    fn drop(&mut self) {
        MINE.with(|w| w.parked.fetch_sub(1, Ordering::AcqRel));
    }
}

/// This thread's attempt epoch, which moves on only at the outermost
/// entry and its retries: a model tells a wound's retry from a replay
/// retry by it.
#[cfg(all(test, loom))]
pub(crate) fn epoch() -> u32 {
    MINE.with(|w| w.state.load(Ordering::Relaxed) & !FLAGS)
}

/// Attempts a waiter wounded, for tests and the scans' counts; a plain
/// atomic under loom too, which a model reads after its threads join.
#[cfg(test)]
pub(crate) static TAKE_OVERS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[cfg(all(test, not(loom)))]
#[path = "attempt_tests.rs"]
mod tests;
