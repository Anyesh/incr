//! Reads by id that never run a compute: one node's snapshot entry, and
//! the trace a binding kept from the last `get_traced` that recomputed a
//! root.

use super::*;
use crate::kept_trace::{KeptTrace, KeptTraceMemory, StepValue};
use crate::trace::{
    LastTrace, NodeInfo, NodeTrace, PropagationTrace, Retain, StampPair, TraceCapture,
};

impl<C: Cells> Runtime<C> {
    /// [`Self::get_traced`], with what the trace saw of its root for
    /// [`Self::retain_trace`].
    pub fn get_traced_retaining<T: Value>(
        &self,
        handle: Incr<T>,
    ) -> (T, PropagationTrace, TraceCapture) {
        self.get_traced_capturing(handle)
    }

    /// `trace` in the compact form [`Self::retain_trace`] keeps, for the
    /// root of the `get_traced_retaining` call that returned `capture`:
    /// each step's slot generation as of now, and its value as `value`
    /// gives it. Takes no lock while `value` runs.
    pub fn kept_trace<V: Send + Sync + 'static>(
        &self,
        capture: &TraceCapture,
        trace: &PropagationTrace,
        value: impl FnMut(&NodeTrace) -> StepValue<V>,
    ) -> KeptTrace {
        let generation_of = |slot: u32| {
            if slot < self.nodes.len() {
                self.nodes.get(slot).generation()
            } else {
                0
            }
        };
        KeptTrace::build(capture.generation, trace, generation_of, value)
    }

    /// Keeps `kept` for the root of the `get_traced_retaining` call that
    /// returned `capture`, when that call's last entry for the root is
    /// `Recomputed` and the root's stamps have not moved since the entry.
    /// When the call's only root entry is `VerifiedClean` and the kept
    /// trace described the root as the call found it, the kept trace is
    /// moved to the verified stamps and `kept` is dropped. Anything else
    /// leaves the kept trace as it is. A trace this replaces, or `kept`
    /// when it is not kept, drops after the runtime lock is released.
    ///
    /// # Panics
    ///
    /// If `capture` came from another runtime, or `kept` was not built by
    /// [`Self::kept_trace`] from `capture` or has a clone, whose stamps it
    /// would share.
    pub fn retain_trace(&self, capture: TraceCapture, kept: KeptTrace) {
        assert!(
            capture.runtime == self.runtime_id,
            "incr-core: retain_trace given a capture from another runtime"
        );
        assert!(
            kept.generation() == capture.generation && kept.is_unique(),
            "incr-core: retain_trace given a trace kept for another capture, or shared"
        );
        let slot = capture.slot;
        let replaced = match capture.decide() {
            Retain::Nothing => Some(kept),
            Retain::Keep(stamps) => {
                let mut inner = self.inner.write();
                if self.still_at(&capture, stamps) {
                    kept.set_stamps(stamps);
                    inner.last_traces.insert(slot, kept)
                } else {
                    Some(kept)
                }
            }
            Retain::Restamp { from, to } => {
                let inner = self.inner.write();
                if self.still_at(&capture, to) {
                    if let Some(old) = inner
                        .last_traces
                        .get(slot)
                        .filter(|old| old.generation() == capture.generation)
                        .filter(|old| old.stamps() == from)
                    {
                        old.set_stamps(to);
                    }
                }
                Some(kept)
            }
        };
        drop(replaced);
    }

    /// The state of the live node at `id` and `generation`, and the trace
    /// kept for it by [`Self::retain_trace`] if the node is Clean at the
    /// stamps the trace was kept with; `None` when no live node has that
    /// slot and generation. Runs no compute and records no dependency.
    pub fn last_trace(&self, id: NodeId, generation: u32) -> Option<LastTrace> {
        self.erased_handle_at(id, generation)?;
        let node = self.nodes.get(id.0);
        let inner = self.inner.read();
        let state = state::load::<C>(node.state_cell());
        if state == NodeState::Dead || node.generation() != generation {
            return None;
        }
        let trace = inner
            .last_traces
            .get(id.0)
            .filter(|kept| kept.generation() == generation && state == NodeState::Clean)
            .filter(|kept| kept.stamps().is_landed() && self.stamp_pair(id.0) == kept.stamps())
            .cloned();
        Some(LastTrace { state, trace })
    }

    /// The count, bytes and references of the kept traces, from running
    /// totals: the traces' allocations and the table holding them.
    pub fn kept_trace_memory(&self) -> KeptTraceMemory {
        self.inner.read().last_traces.memory()
    }

    /// The [`Self::graph_snapshot`] entry of the live node at `id` and
    /// `generation`, or `None` when no live node has both. Records no
    /// dependency.
    pub fn node_info(&self, id: NodeId, generation: u32) -> Option<NodeInfo> {
        self.erased_handle_at(id, generation)?;
        let inner = self.inner.read();
        self.live_node_info(&inner, id.0)
            .filter(|info| info.generation == generation)
    }

    /// The trace root a `get_traced` of `handle` records entries for:
    /// none for a handle of another runtime, or a pending one, whose slot
    /// may be a live node's here.
    pub(crate) fn trace_root_for<T: Value>(
        &self,
        handle: Incr<T>,
    ) -> Option<(u32, crate::trace::RootEntries)> {
        (handle.runtime_id() == self.runtime_id).then(|| (handle.slot(), Default::default()))
    }

    /// `handle`'s stamps, or zeros, which no kept trace matches, for a
    /// handle of another runtime or past this one's slots, which `get`
    /// refuses or reads as a placeholder.
    pub(crate) fn handle_stamps<T: Value>(&self, handle: Incr<T>) -> StampPair {
        if handle.runtime_id() == self.runtime_id && handle.slot() < self.nodes.len() {
            self.stamp_pair(handle.slot())
        } else {
            StampPair {
                verified_at: 0,
                changed_at: 0,
            }
        }
    }

    pub(crate) fn stamp_pair(&self, slot: u32) -> StampPair {
        let node = self.nodes.get(slot);
        StampPair {
            verified_at: node.verified_at(),
            changed_at: node.changed_at(),
        }
    }

    /// Whether the captured root is still the node the capture saw, at
    /// `stamps`. Called under the inner write lock, which a delete holds
    /// while it bumps the generation.
    fn still_at(&self, capture: &TraceCapture, stamps: StampPair) -> bool {
        capture.slot < self.nodes.len()
            && self.nodes.get(capture.slot).generation() == capture.generation
            && self.stamp_pair(capture.slot) == stamps
    }
}
