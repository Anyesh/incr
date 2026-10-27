//! Wound-wait for claim waits under `Shared`.
//!
//! A thread that waits on another thread's claim for long registers in
//! one process-wide registry: what it waits on, the claims it holds on
//! every runtime, and its ticket, taken once per outermost engine entry
//! and kept across that entry's retries, so the oldest entry has the
//! lowest ticket. The thread whose registration closes a waits-for cycle
//! marks every waiter on it wounded except the oldest. A wounded waiter
//! deregisters and unwinds to its outermost engine entry, releasing every
//! claim it holds on the way out, and the entry retries. The oldest
//! thread then holds the whole cycle on its own stack and resolves it by
//! the single-thread rules, so the result does not depend on timing.
//!
//! The registry lock is never held with a runtime's `inner` lock, and no
//! user code runs under it.

use crate::handle::RuntimeId;

#[cfg(not(loom))]
use std::sync::{
    atomic::{AtomicU64, AtomicU8, Ordering},
    Arc, Mutex,
};

#[cfg(loom)]
use loom::sync::{
    atomic::{AtomicU8, Ordering},
    Arc, Mutex,
};

/// Spins a claim wait makes before it registers.
#[cfg(not(loom))]
pub(crate) const REGISTER_AFTER: u32 = 64;
#[cfg(loom)]
pub(crate) const REGISTER_AFTER: u32 = 1;

const WAITING: u8 = 0;
const WOUNDED: u8 = 1;
/// The registration ended.
const DONE: u8 = 2;

/// A claim: a runtime and a node slot.
pub(crate) type Claim = (RuntimeId, u32);

struct Waiter {
    ticket: u64,
    /// The waiting thread's attempt word, as [`crate::attempt::word_id`].
    owner: usize,
    waiting_on: Claim,
    holds: Vec<Claim>,
    state: Arc<AtomicU8>,
    /// For a wounded waiter, the oldest waiter on the cycle it was on and
    /// the claim that waiter waits on.
    winner: Option<(Arc<AtomicU8>, Claim)>,
}

struct Registry {
    waiters: Vec<Waiter>,
    next_ticket: u64,
}

#[cfg(not(loom))]
static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    waiters: Vec::new(),
    next_ticket: 1,
});

#[cfg(loom)]
loom::lazy_static! {
    static ref REGISTRY: Mutex<Registry> = Mutex::new(Registry {
        waiters: Vec::new(),
        next_ticket: 1,
    });
}

#[cfg(not(loom))]
thread_local! {
    static WINNER: std::cell::RefCell<Option<(Arc<AtomicU8>, Claim)>> = const { std::cell::RefCell::new(None) };
}

#[cfg(loom)]
loom::thread_local! {
    static WINNER: std::cell::RefCell<Option<(Arc<AtomicU8>, Claim)>> = std::cell::RefCell::new(None);
}

/// Wounds dealt, for tests.
#[cfg(not(loom))]
pub(crate) static WOUNDS: AtomicU64 = AtomicU64::new(0);

/// Head-start checks a runtime made of a claim of its own, for tests.
#[cfg(all(test, not(loom)))]
pub(crate) static OWN_CLAIM_CHECKS: AtomicU64 = AtomicU64::new(0);

/// Threads registered now, for tests.
#[cfg(test)]
pub(crate) fn registered() -> usize {
    registry().waiters.len()
}

/// Whether a registered thread is waiting on `claim`, for tests.
#[cfg(all(test, not(loom)))]
pub(crate) fn waited_on(claim: Claim) -> bool {
    registry().waiters.iter().any(|w| w.waiting_on == claim)
}

fn registry() -> impl std::ops::DerefMut<Target = Registry> {
    #[cfg(not(loom))]
    {
        REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
    }
    #[cfg(loom)]
    {
        REGISTRY.lock().unwrap()
    }
}

/// This thread's place in the registry while it waits; dropping it
/// deregisters.
pub(crate) struct Registration {
    state: Arc<AtomicU8>,
}

impl Registration {
    /// Registers a thread holding `holds` as waiting on `waiting_on`, and
    /// wounds every younger waiter on a waits-for cycle it closes.
    pub(crate) fn new(
        ticket: impl FnOnce(&mut dyn FnMut() -> u64) -> u64,
        waiting_on: Claim,
        holds: Vec<Claim>,
    ) -> Self {
        crate::lock_order::assert_may_wait_on_claim();
        let state = Arc::new(AtomicU8::new(WAITING));
        let mut reg = registry();
        let reg = &mut *reg;
        let mut next = || {
            let t = reg.next_ticket;
            reg.next_ticket += 1;
            t
        };
        let ticket = ticket(&mut next);
        reg.waiters.push(Waiter {
            ticket,
            owner: crate::attempt::word_id(),
            waiting_on,
            holds,
            state: state.clone(),
            winner: None,
        });
        let me = reg.waiters.len() - 1;
        let mut chain = vec![me];
        let mut on = waiting_on;
        while let Some(owner) = reg.waiters.iter().position(|w| w.holds.contains(&on)) {
            if owner == me {
                let oldest = chain
                    .iter()
                    .copied()
                    .min_by_key(|&i| reg.waiters[i].ticket)
                    .expect("a cycle has a waiter");
                let winner = (
                    reg.waiters[oldest].state.clone(),
                    reg.waiters[oldest].waiting_on,
                );
                for &i in &chain {
                    if i != oldest {
                        reg.waiters[i].winner = Some(winner.clone());
                        reg.waiters[i].state.store(WOUNDED, Ordering::Release);
                        #[cfg(not(loom))]
                        WOUNDS.fetch_add(1, Ordering::Relaxed);
                    }
                }
                break;
            }
            if chain.contains(&owner) {
                // A cycle that does not pass through this waiter: the
                // waiter that closed it has resolved it already.
                break;
            }
            chain.push(owner);
            on = reg.waiters[owner].waiting_on;
        }
        Self { state }
    }

    /// Whether a registration that closed a cycle wounded this waiter.
    pub(crate) fn wounded(&self) -> bool {
        self.state.load(Ordering::Acquire) == WOUNDED
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        let mut reg = registry();
        if let Some(i) = reg
            .waiters
            .iter()
            .position(|w| Arc::ptr_eq(&w.state, &self.state))
        {
            if let Some(winner) = reg.waiters.swap_remove(i).winner {
                WINNER.with(|w| *w.borrow_mut() = Some(winner));
            }
        }
        self.state.store(DONE, Ordering::Release);
    }
}

/// Runs `wound`, a waiter's take-over of the attempt whose word is
/// `owner`, under the registry lock, unless that owner waits here with an
/// older ticket than the waiter's `ticket` (0 for one not taken yet, which
/// is younger than every taken one): the wound order settles that owner's
/// waits, and a take-over is only for an owner held where the registry
/// cannot see it. False if refused.
pub(crate) fn take_over_in_order(owner: usize, ticket: u64, wound: impl FnOnce() -> bool) -> bool {
    let reg = registry();
    let mine = if ticket == 0 { u64::MAX } else { ticket };
    let older_waits = reg
        .waiters
        .iter()
        .any(|w| w.owner == owner && w.ticket < mine && w.state.load(Ordering::Acquire) == WAITING);
    if older_waits {
        return false;
    }
    wound()
}

/// Before a wounded entry retries: waits until the oldest waiter on the
/// cycle it was wounded on has stopped waiting, and then until `taken`
/// says the claim it waited on is no longer free, for at most as many
/// backoffs again. Its wait ends once this thread's released claims are
/// free, and the head start lets it take them before a retry takes them
/// straight back and is wounded again over the same cycle; a fixed count
/// of backoffs gave up a time slice each on a loaded box. The bound
/// covers a winner that takes another way, and waiting for its whole
/// entry could deadlock on a lock the caller holds across this one. The
/// wounded thread holds nothing while it waits.
pub(crate) fn await_winner(mut backoff: impl FnMut(u32), taken: impl Fn(Claim) -> bool) {
    let Some((winner, claim)) = WINNER.with(|w| w.borrow_mut().take()) else {
        return;
    };
    let mut spins = 0;
    while winner.load(Ordering::Acquire) != DONE {
        spins += 1;
        backoff(spins);
    }
    for spins in REGISTER_AFTER..2 * REGISTER_AFTER {
        if taken(claim) {
            return;
        }
        backoff(spins);
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::handle::RuntimeId;

    /// A wounded thread's head start ends once the claim its winner waited
    /// on is taken, not after a fixed number of backoffs, each of which can
    /// give up a time slice on a loaded box.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn head_start_ends_once_the_winners_claim_is_taken() {
        crate::child_test::run_in_child(
            "waits::tests::head_start_ends_once_the_winners_claim_is_taken",
            || {
                let rt = RuntimeId::for_test(u64::MAX - 7);
                let (c1, c2) = ((rt, 1), (rt, 2));
                let older = Registration::new(|_| 1, c2, vec![c1]);
                let younger = Registration::new(|_| 2, c1, vec![c2]);
                assert!(younger.wounded() && !older.wounded());
                drop(younger);
                drop(older);
                let mut backoffs = 0;
                await_winner(
                    |_| backoffs += 1,
                    |claim| {
                        assert_eq!(claim, c2, "the claim the winner waited on");
                        true
                    },
                );
                assert_eq!(backoffs, 0);
            },
        )
        .assert_clean();
    }
}
