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
//! - [`NodeState::ComputingDirty`]: a dependency changed while a thread
//!   was mid-compute. The dirty walk CASes `Computing -> ComputingDirty`
//!   so the in-flight compute cannot swallow the change: the computing
//!   thread's finishing CAS (`Computing -> Clean`) fails and it stores
//!   `Dirty` instead, forcing the next reader to recompute. Not a valid
//!   claim source (the original computer still owns the node).
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
#[non_exhaustive]
pub enum NodeState {
    New = 0,
    Dirty = 1,
    Computing = 2,
    Clean = 3,
    Failed = 4,
    ComputingDirty = 5,
    /// Deleted via `delete_node`; the slot is on the free list awaiting
    /// recycling. Readers can only reach Dead through internal walks
    /// racing a delete (stale user handles are rejected by the
    /// generation check before any state load).
    Dead = 6,
    /// A member of a dependency cycle with no strategy: it has no value,
    /// and a read gets the cycle. Settled like Clean, so the bit pattern
    /// `(s | 4) == 7` holds for exactly these two.
    Cyclic = 7,
    /// Claimed by a delete of an owned set (`claim_set`) that may still
    /// roll the claim back. It exists only inside the claimant's
    /// `inner.write()` hold, so a path that finds it holds no lock and
    /// spins until the state moves to `Dead` or back. 8, not 3 or 7, so
    /// `is_settled` stays false for it.
    Deleting = 8,
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
            5 => Self::ComputingDirty,
            6 => Self::Dead,
            7 => Self::Cyclic,
            8 => Self::Deleting,
            other => panic!("invalid NodeState value: {}", other),
        }
    }

    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Clean or Cyclic: a landed state a dependent can verify against.
    #[inline(always)]
    pub fn is_settled(self) -> bool {
        (self as u8 | 4) == 7
    }
}

/// `Clean`'s byte in a state cell, which `get`'s fast path compares the
/// raw load with.
pub(crate) const CLEAN_BYTE: u8 = NodeState::Clean as u8;

const _: () = assert!(
    std::mem::size_of::<NodeState>() == 1 && CLEAN_BYTE == 3,
    "the Clean fast path compares a state cell's raw byte with Clean's"
);

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
    match C::state_try_transition(cell, expected.as_u8(), new.as_u8()) {
        Ok(()) => Ok(()),
        Err(raw) if C::SINGLE_THREADED => Err(NodeState::from_u8(raw)),
        Err(raw) => claim_own_held::<C>(cell, expected, new, raw),
    }
}

/// A transition of a node this thread's attempt holds, from the state the
/// view shows it in, while the node's word stays `Computing`: the owner
/// claims its own landing back to rerun or reverify it, or its set's walk
/// moves the landing as it would a published one.
#[cold]
#[inline(never)]
fn claim_own_held<C: Cells>(
    cell: &C::State,
    expected: NodeState,
    new: NodeState,
    raw: u8,
) -> Result<(), NodeState> {
    let seen = NodeState::from_u8(raw);
    if matches!(seen, NodeState::Computing | NodeState::ComputingDirty) {
        let key = cell_key::<C>(cell);
        let own = match new {
            NodeState::Computing => crate::hold::reclaim(key, expected, seen),
            _ => crate::hold::flip(key, expected, new, seen),
        };
        if own {
            // The claim's run reads after the set that flagged the word.
            if new == NodeState::Computing && seen == NodeState::ComputingDirty {
                let cleared = C::state_try_transition(cell, raw, NodeState::Computing.as_u8());
                debug_assert!(
                    cleared.is_ok(),
                    "incr-core: only the owner moves a held word"
                );
            }
            return Ok(());
        }
    }
    Err(seen)
}

/// The address that keys a node's held landing in `crate::hold`.
#[inline(always)]
pub(crate) fn cell_key<C: Cells>(cell: &C::State) -> usize {
    std::ptr::from_ref(cell).cast::<u8>() as usize
}

/// Claim the right to compute this node by transitioning to `Computing`
/// from one of the valid source states (`New` or `Dirty`). `Failed` is
/// not a valid source: a `Failed` node stays `Failed` until the writer's
/// dirty walk transitions it to `Dirty` first. `ComputingDirty` is not a
/// valid source either: the original computer still owns the node.
///
/// Returns `Ok(source)` with the state we claimed from (so the caller
/// knows whether this is a first compute or a recompute without a racy
/// separate load), or `Err(observed)` with the state we actually saw.
/// Under `Shared`, exactly one of many racing threads succeeds.
#[inline(always)]
pub fn try_claim_compute<C: Cells>(cell: &C::State) -> Result<NodeState, NodeState> {
    // Try Dirty first (more common in steady state), then New.
    if try_transition::<C>(cell, NodeState::Dirty, NodeState::Computing).is_ok() {
        return Ok(NodeState::Dirty);
    }
    try_transition::<C>(cell, NodeState::New, NodeState::Computing).map(|_| NodeState::New)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    #[test]
    fn local_claim_from_dirty() {
        let s = Local::new_state(NodeState::Dirty.as_u8());
        assert_eq!(try_claim_compute::<Local>(&s), Ok(NodeState::Dirty));
        assert_eq!(load::<Local>(&s), NodeState::Computing);
    }

    #[test]
    fn local_claim_from_new() {
        let s = Local::new_state(NodeState::New.as_u8());
        assert_eq!(try_claim_compute::<Local>(&s), Ok(NodeState::New));
        assert_eq!(load::<Local>(&s), NodeState::Computing);
    }

    #[test]
    fn computing_dirty_is_not_claimable() {
        let s = Shared::new_state(NodeState::ComputingDirty.as_u8());
        assert_eq!(
            try_claim_compute::<Shared>(&s),
            Err(NodeState::ComputingDirty)
        );
        assert_eq!(load::<Shared>(&s), NodeState::ComputingDirty);
    }

    #[test]
    fn computing_dirty_roundtrips_through_u8() {
        assert_eq!(NodeState::from_u8(5), NodeState::ComputingDirty);
        assert_eq!(NodeState::ComputingDirty.as_u8(), 5);
    }

    #[test]
    fn a_clean_cell_holds_the_fast_paths_byte() {
        let s = Local::new_state(NodeState::Clean.as_u8());
        assert_eq!(Local::state_load_acquire(&s), CLEAN_BYTE);
        assert_eq!(NodeState::from_u8(CLEAN_BYTE), NodeState::Clean);
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
