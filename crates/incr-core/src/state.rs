//! Node state encoding and lifecycle.
//!
//! The state cell itself is provided by the active [`Cells`] strategy
//! (`Cells::State`); this module only fixes the encoding and provides the
//! transition helpers that operate on the cell through the strategy.
//!
//! States:
//! - [`NodeState::New`]: created but never computed. First reader CASes to
//!   `Computing`.
//! - [`NodeState::Dirty`]: a dependency changed; the value is stale.
//! - [`NodeState::Computing`]: a thread is currently running compute. Only
//!   one thread holds this state at a time (enforced by CAS on `Shared`,
//!   by the single-threaded execution model on `Local`).
//! - [`NodeState::Clean`]: value is current and readable.
//! - [`NodeState::Failed`]: last compute panicked. Transitions to `Dirty`
//!   when a dependency changes.
//!
//! Transitions into `Computing` happen via [`Cells::state_try_transition`]
//! (CAS on `Shared`, conditional check on `Local`). Transitions out of
//! `Computing` (to `Clean` or `Failed`) use Release ordering on `Shared`
//! to publish the writes to value / deps / timestamps that happened
//! during compute.
//!
//! The transition helpers below take `&<C as Cells>::State` and the
//! associated `Cells` impl as a generic parameter, so all calls inline
//! through the strategy's `#[inline(always)]` methods.

use crate::cells::Cells;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NodeState {
    New = 0,
    Dirty = 1,
    Computing = 2,
    Clean = 3,
    Failed = 4,
}

impl NodeState {
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::New,
            1 => Self::Dirty,
            2 => Self::Computing,
            3 => Self::Clean,
            4 => Self::Failed,
            other => panic!("invalid NodeState value: {}", other),
        }
    }

    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Load the current state with Acquire ordering through the strategy's
/// state cell. Use on the reader hot path to synchronize with the
/// Release store that transitioned the node to its current state.
#[inline(always)]
pub fn load<C: Cells>(cell: &C::State) -> NodeState {
    NodeState::from_u8(C::state_load_acquire(cell))
}

/// Store a new state with Release ordering through the strategy's state
/// cell. Use when transitioning to `Clean` or `Failed` after writing the
/// node's value, deps, and timestamps.
#[inline(always)]
pub fn store<C: Cells>(cell: &C::State, new: NodeState) {
    C::state_store_release(cell, new.as_u8());
}

/// Attempt to transition from `expected` to `new`. On `Shared` this is a
/// CAS with AcqRel success ordering; on `Local` it is a conditional set
/// (semantically equivalent under single-threaded execution).
#[inline(always)]
pub fn try_transition<C: Cells>(
    cell: &C::State,
    expected: NodeState,
    new: NodeState,
) -> Result<(), NodeState> {
    C::state_try_transition(cell, expected.as_u8(), new.as_u8()).map_err(NodeState::from_u8)
}

/// Claim the right to compute this node by transitioning to `Computing`
/// from one of the valid source states (`New` or `Dirty`). `Failed` is
/// not a valid source: a `Failed` node stays `Failed` until the writer's
/// dirty walk transitions it to `Dirty` first.
///
/// Returns `Ok(())` if this caller now owns compute, or `Err(observed)`
/// with the state we actually saw. Under `Shared`, exactly one of many
/// racing threads succeeds.
#[inline(always)]
pub fn try_claim_compute<C: Cells>(cell: &C::State) -> Result<(), NodeState> {
    // Try Dirty first (more common in steady state), then New.
    if try_transition::<C>(cell, NodeState::Dirty, NodeState::Computing).is_ok() {
        return Ok(());
    }
    try_transition::<C>(cell, NodeState::New, NodeState::Computing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    #[test]
    fn local_claim_from_dirty() {
        let s = Local::new_state(NodeState::Dirty.as_u8());
        assert!(try_claim_compute::<Local>(&s).is_ok());
        assert_eq!(load::<Local>(&s), NodeState::Computing);
    }

    #[test]
    fn local_claim_from_new() {
        let s = Local::new_state(NodeState::New.as_u8());
        assert!(try_claim_compute::<Local>(&s).is_ok());
        assert_eq!(load::<Local>(&s), NodeState::Computing);
    }

    #[test]
    fn local_claim_from_clean_fails() {
        let s = Local::new_state(NodeState::Clean.as_u8());
        assert!(try_claim_compute::<Local>(&s).is_err());
        assert_eq!(load::<Local>(&s), NodeState::Clean);
    }

    #[test]
    fn shared_claim_from_dirty() {
        let s = Shared::new_state(NodeState::Dirty.as_u8());
        assert!(try_claim_compute::<Shared>(&s).is_ok());
        assert_eq!(load::<Shared>(&s), NodeState::Computing);
    }

    #[test]
    fn shared_claim_from_clean_fails() {
        let s = Shared::new_state(NodeState::Clean.as_u8());
        assert!(try_claim_compute::<Shared>(&s).is_err());
        assert_eq!(load::<Shared>(&s), NodeState::Clean);
    }

    #[test]
    fn shared_concurrent_claim_one_winner() {
        use std::sync::atomic::{AtomicUsize, Ordering as O};
        use std::sync::Arc;
        use std::thread;

        const THREADS: usize = 16;
        const ROUNDS: usize = 200;

        for _ in 0..ROUNDS {
            let s = Arc::new(Shared::new_state(NodeState::Dirty.as_u8()));
            let winners = Arc::new(AtomicUsize::new(0));

            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    let s = Arc::clone(&s);
                    let w = Arc::clone(&winners);
                    thread::spawn(move || {
                        if try_claim_compute::<Shared>(&s).is_ok() {
                            w.fetch_add(1, O::Relaxed);
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }

            assert_eq!(
                winners.load(O::Relaxed),
                1,
                "expected exactly one thread to claim compute"
            );
        }
    }
}
