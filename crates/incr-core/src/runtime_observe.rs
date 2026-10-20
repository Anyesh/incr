//! `observer_read`, the read an observer's run makes: the read path of
//! `try_get` with every stale-handle panic turned into `ObsErr::Gone`,
//! because a Shared delete can land between `stabilize` taking its
//! observers and an observer's run.

use std::cell::Cell;

use super::*;

/// Why an observer's read has no value to fire with.
#[derive(Debug)]
pub(crate) enum ObsErr {
    /// The node was deleted, or its slot recycled, since the observer
    /// was taken; the observer went with it.
    Gone,
    /// The node has no value because of a cycle; the stamp of its last
    /// landing.
    Cycle(u64),
}

#[cfg(not(loom))]
thread_local! {
    static OBSERVED: Cell<Option<(RuntimeId, u32)>> = const { Cell::new(None) };
}

#[cfg(loom)]
loom::thread_local! {
    static OBSERVED: Cell<Option<(RuntimeId, u32)>> = Cell::new(None);
}

/// Whether `slot` of `rt` is the node an `observer_read` on this thread
/// is bringing up to date, for the cold Dead arm of `compute_attempts`.
pub(super) fn is_observed(rt: RuntimeId, slot: u32) -> bool {
    OBSERVED.with(|o| o.get() == Some((rt, slot)))
}

/// Whether no `observer_read` on this thread has left its node recorded.
#[cfg(all(test, not(loom)))]
pub(crate) fn observed_is_clear() -> bool {
    OBSERVED.with(|o| o.get().is_none())
}

/// Records the observed node for its lifetime and restores the one
/// before it on drop, so an unwind out of the walk leaves no record.
struct Observing(Option<(RuntimeId, u32)>);

impl Observing {
    fn record(rt: RuntimeId, slot: u32) -> Self {
        Self(OBSERVED.with(|o| o.replace(Some((rt, slot)))))
    }
}

impl Drop for Observing {
    fn drop(&mut self) {
        OBSERVED.with(|o| o.set(self.0));
    }
}

impl<C: Cells> Runtime<C> {
    /// Bring the node up to date and read it with its change stamp, as
    /// `try_get` does, except that a node deleted or recycled under the
    /// handle reads `Gone` instead of panicking. The stamp is loaded
    /// before the value, so a change landing between them fires again
    /// rather than not at all.
    ///
    /// Inside a compute the value is read by `try_get`, so that the read
    /// records its dependency and follows the frame's doom rules.
    pub(super) fn observer_read<T: Value>(&self, handle: Incr<T>) -> Result<(T, u64), ObsErr> {
        let id = NodeId(handle.slot());
        let gen = handle.generation();
        let node = self.nodes.get(id.0);
        let observed = {
            let _inner = self.inner.read();
            if node.generation() != gen {
                return Err(ObsErr::Gone);
            }
            state::load::<C>(node.state_cell())
        };
        self.park(ParkPoint::ObserverAfterEntryCheck(id));
        let in_compute = self.dep_stack.current_frame_active(self.runtime_id);
        if observed != NodeState::Clean {
            self.observer_walk(handle, observed, in_compute)?;
        }
        self.park(ParkPoint::ObserverAfterEnsureClean(id));
        if node.generation() != gen {
            return Err(ObsErr::Gone);
        }
        let stamp = self.observed_stamp(node);
        if in_compute {
            return self
                .try_get(handle)
                .map(|value| (value, stamp))
                .map_err(|_| ObsErr::Cycle(stamp));
        }
        self.park(ParkPoint::ObserverBeforePreCheck(id));
        let type_tag = node.type_tag();
        let arena_slot = node.arena_slot();
        C::fence_acquire();
        if node.generation() != gen {
            return Err(ObsErr::Gone);
        }
        let arena = self.arenas.tagged_arena::<T>(type_tag);
        let value = value_code::run(|| arena.and_then(|arena| arena.try_read(arena_slot)));
        C::fence_acquire();
        if node.generation() != gen {
            return Err(ObsErr::Gone);
        }
        match value {
            Some(value) => Ok((value, stamp)),
            None => {
                self.read_missing(handle, node);
                Err(ObsErr::Cycle(stamp))
            }
        }
    }

    /// `observer_read`'s walk of a node it found not Clean, under the
    /// record that lets the Dead arm return `Gone` for this node.
    #[cold]
    #[inline(never)]
    fn observer_walk<T: Value>(
        &self,
        handle: Incr<T>,
        observed: NodeState,
        in_compute: bool,
    ) -> Result<(), ObsErr> {
        let id = NodeId(handle.slot());
        let node = self.nodes.get(id.0);
        if observed == NodeState::Cyclic || is_computing(observed) {
            match self.cycle_read(handle, observed, ReadKind::TryGet, None) {
                Some(Err(_)) => return Err(ObsErr::Cycle(self.observed_stamp(node))),
                // A node on this thread's stack: `try_get` reads it by the
                // same rule.
                Some(Ok(_)) if in_compute => return Ok(()),
                Some(Ok(_)) => unreachable!("incr-core: a held node outside a compute"),
                None => {}
            }
        }
        let _observing = Observing::record(self.runtime_id, id.0);
        let computed = self
            .entered(ReadKind::TryGet, || {
                self.ensure_clean(id, handle.generation())
            })
            .map_err(|_| ObsErr::Cycle(self.observed_stamp(node)))?;
        match computed {
            Computed::Done | Computed::Unsettled => Ok(()),
            Computed::Gone => Err(ObsErr::Gone),
            // The frame's read rules handle these under `try_get`.
            Computed::Deferred | Computed::Doomed | Computed::Discarded if in_compute => Ok(()),
            _ => unreachable!(
                "incr-core: an observer's walk outside a compute returned {computed:?}"
            ),
        }
    }

    /// The node's change stamp, with a pending `set`'s marker clamped to
    /// the current revision so that a later landing still exceeds it.
    fn observed_stamp(&self, node: &NodeData<C>) -> u64 {
        node.changed_at().min(self.current_revision())
    }
}
