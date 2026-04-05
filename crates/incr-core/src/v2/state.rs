//! Node state machine.
//!
//! A node's lifecycle is governed by a small atomic state machine. The states
//! and their transitions are specified in section 7 of the concurrent core
//! rewrite spec. This file implements the state enum, the atomic state cell,
//! and helpers for the transition patterns used by the runtime.
//!
//! ## States
//!
//! - [`NodeState::New`] — node exists but has never been computed.
//! - [`NodeState::Dirty`] — needs recomputation because a dependency changed.
//! - [`NodeState::Computing`] — a thread is actively running the compute function.
//! - [`NodeState::Clean`] — value is current and readable.
//! - [`NodeState::Failed`] — last compute returned an error or panicked.
//!
//! ## Transitions and ordering
//!
//! Transitions into `Computing` happen only via CAS, guaranteeing at most one
//! thread computes a given node at a time. Transitions out of `Computing`
//! (to `Clean` or `Failed`) use `Release` ordering to publish the compute's
//! writes (value, deps, timestamps) to readers who Acquire-load the state.
//! Transitions from `Clean` to `Dirty` (by the writer's dirty walk) also use
//! `Release` ordering so that readers observing `Dirty` see the revision bump
//! that caused the transition.

use std::sync::atomic::{AtomicU8, Ordering};

/// The lifecycle state of a node.
///
/// Stored as a `u8` so it fits in a single byte and can be represented
/// compactly inside an atomic cell. The numeric values are load-bearing
/// for the `AtomicNodeState` compare-and-swap helpers; do not reorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub(crate) enum NodeState {
    /// Created but never computed. First reader will CAS to Computing.
    New = 0,
    /// A dependency has changed; the value is stale. Next reader recomputes.
    Dirty = 1,
    /// A thread is currently running this node's compute function. Other
    /// readers must wait for it to transition to Clean or Failed.
    Computing = 2,
    /// The value matches the current dependencies and is safe to read.
    Clean = 3,
    /// The last compute panicked or returned an error. The node has a
    /// failure payload stored separately. Readers of a Failed node see
    /// the error. Failed transitions to Dirty if a dependency changes.
    Failed = 4,
}

impl NodeState {
    /// Decode a raw `u8` into a `NodeState`. Panics on unknown values to
    /// catch memory corruption early.
    #[inline]
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::New,
            1 => Self::Dirty,
            2 => Self::Computing,
            3 => Self::Clean,
            4 => Self::Failed,
            _ => panic!("invalid NodeState value: {}", v),
        }
    }
}

/// Atomic cell holding a [`NodeState`].
///
/// This is the single source of truth for a node's lifecycle. All transitions
/// happen through the methods on this type, which encode the correct memory
/// ordering for each case. Direct access to the underlying `AtomicU8` is
/// intentionally not exposed; the transition patterns are the API.
#[derive(Debug)]
pub(crate) struct AtomicNodeState {
    cell: AtomicU8,
}

impl AtomicNodeState {
    /// Create a new state cell initialized to `state`.
    pub(crate) fn new(state: NodeState) -> Self {
        Self {
            cell: AtomicU8::new(state as u8),
        }
    }

    /// Load the current state with `Acquire` ordering.
    ///
    /// This is the correct load for readers on the hot path: if the returned
    /// state is `Clean`, the Acquire synchronizes with the Release store
    /// that transitioned the node to Clean, so subsequent Relaxed reads of
    /// the node's value, deps, and timestamps are guaranteed to observe
    /// the writes that happened before that transition.
    #[inline]
    pub(crate) fn load_acquire(&self) -> NodeState {
        NodeState::from_u8(self.cell.load(Ordering::Acquire))
    }

    /// Load the current state with `Relaxed` ordering.
    ///
    /// Use this only when no synchronization with other fields is required,
    /// for example for debug assertions, diagnostics, or when the caller
    /// has already established happens-before via another Acquire load.
    /// Do not use it on the hot path before reading a node's value.
    #[inline]
    pub(crate) fn load_relaxed(&self) -> NodeState {
        NodeState::from_u8(self.cell.load(Ordering::Relaxed))
    }

    /// Store a new state with `Release` ordering.
    ///
    /// Use this when transitioning out of `Computing` (to `Clean` or `Failed`)
    /// after writing the node's value, deps, and timestamps. The Release
    /// publishes those Relaxed writes to readers who Acquire-load the state.
    ///
    /// Also used for transitioning from `Clean` to `Dirty` during the
    /// writer's dirty propagation walk, so readers observing `Dirty` see
    /// the revision bump.
    #[inline]
    pub(crate) fn store_release(&self, state: NodeState) {
        self.cell.store(state as u8, Ordering::Release);
    }

    /// Attempt to transition from `expected` to `new` via compare-and-swap.
    ///
    /// Returns `Ok(())` if the transition succeeded (this thread now owns
    /// whatever invariant `new` represents), or `Err(observed)` with the
    /// state we actually observed if the CAS failed.
    ///
    /// Success uses `AcqRel` (Acquire to synchronize with the prior state's
    /// Release, Release to publish this transition). Failure uses `Acquire`
    /// so the caller sees the current state coherently with other fields.
    #[inline]
    pub(crate) fn try_transition(
        &self,
        expected: NodeState,
        new: NodeState,
    ) -> Result<(), NodeState> {
        match self.cell.compare_exchange(
            expected as u8,
            new as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(()),
            Err(observed) => Err(NodeState::from_u8(observed)),
        }
    }

    /// Attempt to transition to `Computing` from any state that permits it.
    ///
    /// A reader encountering a node that needs recomputation uses this to
    /// claim the right to run the compute function. Valid source states
    /// are `New` (first computation) and `Dirty` (recomputation after a
    /// dependency changed). `Failed` is NOT a valid source because a Failed
    /// node stays Failed until the writer's dirty walk transitions it to
    /// Dirty first.
    ///
    /// Returns `Ok(())` if this thread now owns compute, or `Err(observed)`
    /// if the state was something else (Clean, Computing, or Failed).
    #[inline]
    pub(crate) fn try_claim_compute(&self) -> Result<(), NodeState> {
        // Try the two valid source states in order of expected likelihood.
        // Dirty is more common than New in steady state.
        if self
            .try_transition(NodeState::Dirty, NodeState::Computing)
            .is_ok()
        {
            return Ok(());
        }
        self.try_transition(NodeState::New, NodeState::Computing)
    }
}

// The type is `Send + Sync` automatically because `AtomicU8` is.

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn new_state_starts_at_new() {
        let state = AtomicNodeState::new(NodeState::New);
        assert_eq!(state.load_acquire(), NodeState::New);
    }

    #[test]
    fn store_release_updates_state() {
        let state = AtomicNodeState::new(NodeState::New);
        state.store_release(NodeState::Clean);
        assert_eq!(state.load_acquire(), NodeState::Clean);
    }

    #[test]
    fn try_transition_from_expected_succeeds() {
        let state = AtomicNodeState::new(NodeState::Dirty);
        let result = state.try_transition(NodeState::Dirty, NodeState::Computing);
        assert!(result.is_ok());
        assert_eq!(state.load_acquire(), NodeState::Computing);
    }

    #[test]
    fn try_transition_from_unexpected_fails() {
        let state = AtomicNodeState::new(NodeState::Clean);
        let result = state.try_transition(NodeState::Dirty, NodeState::Computing);
        assert_eq!(result, Err(NodeState::Clean));
        // State must be unchanged
        assert_eq!(state.load_acquire(), NodeState::Clean);
    }

    #[test]
    fn try_claim_compute_from_new() {
        let state = AtomicNodeState::new(NodeState::New);
        assert!(state.try_claim_compute().is_ok());
        assert_eq!(state.load_acquire(), NodeState::Computing);
    }

    #[test]
    fn try_claim_compute_from_dirty() {
        let state = AtomicNodeState::new(NodeState::Dirty);
        assert!(state.try_claim_compute().is_ok());
        assert_eq!(state.load_acquire(), NodeState::Computing);
    }

    #[test]
    fn try_claim_compute_from_clean_fails() {
        let state = AtomicNodeState::new(NodeState::Clean);
        let result = state.try_claim_compute();
        // try_claim_compute tries Dirty first, then New. Both fail from Clean.
        // The returned Err carries the observed state from the last attempt
        // (the New→Computing CAS), which is Clean because that is what the
        // CAS actually saw.
        assert_eq!(result, Err(NodeState::Clean));
        // State must be unchanged.
        assert_eq!(state.load_acquire(), NodeState::Clean);
    }

    #[test]
    fn try_claim_compute_from_computing_fails() {
        let state = AtomicNodeState::new(NodeState::Computing);
        let result = state.try_claim_compute();
        assert!(result.is_err());
        assert_eq!(state.load_acquire(), NodeState::Computing);
    }

    #[test]
    fn try_claim_compute_from_failed_fails() {
        let state = AtomicNodeState::new(NodeState::Failed);
        let result = state.try_claim_compute();
        assert!(result.is_err());
        assert_eq!(state.load_acquire(), NodeState::Failed);
    }

    #[test]
    fn concurrent_compute_claim_exactly_one_winner() {
        // This is the critical concurrency invariant: when many threads race
        // to claim compute on the same dirty node, exactly one succeeds.
        const THREADS: usize = 16;
        const ROUNDS: usize = 1000;

        for _ in 0..ROUNDS {
            let state = Arc::new(AtomicNodeState::new(NodeState::Dirty));
            let winners = Arc::new(std::sync::atomic::AtomicUsize::new(0));

            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    let state = state.clone();
                    let winners = winners.clone();
                    thread::spawn(move || {
                        if state.try_claim_compute().is_ok() {
                            winners.fetch_add(1, Ordering::Relaxed);
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }

            assert_eq!(
                winners.load(Ordering::Relaxed),
                1,
                "expected exactly one thread to claim compute, got {}",
                winners.load(Ordering::Relaxed)
            );
            assert_eq!(state.load_acquire(), NodeState::Computing);
        }
    }

    #[test]
    fn release_acquire_synchronizes_with_sibling_data() {
        // This test verifies the core ordering invariant the runtime depends on:
        // when a writer transitions state to Clean with Release, a reader that
        // observes Clean via Acquire load also sees the sibling data the writer
        // wrote with Relaxed stores before the transition.
        use std::sync::atomic::AtomicU64;

        const ROUNDS: usize = 10_000;

        for round in 0..ROUNDS {
            let state = Arc::new(AtomicNodeState::new(NodeState::New));
            let value = Arc::new(AtomicU64::new(0));

            let writer_state = state.clone();
            let writer_value = value.clone();
            let writer = thread::spawn(move || {
                // Simulate compute: write value Relaxed, then state Release.
                writer_value.store(round as u64 + 1, Ordering::Relaxed);
                writer_state.store_release(NodeState::Clean);
            });

            // Reader spins until it sees Clean, then checks value.
            loop {
                if state.load_acquire() == NodeState::Clean {
                    let seen = value.load(Ordering::Relaxed);
                    assert_eq!(
                        seen,
                        round as u64 + 1,
                        "reader observed Clean state but stale value (round {})",
                        round
                    );
                    break;
                }
                std::hint::spin_loop();
            }

            writer.join().unwrap();
        }
    }
}
