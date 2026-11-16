//! Reads by id that never run a compute: one node's snapshot entry, and
//! the payload a binding kept from the last `get_traced` that recomputed
//! a root.

use std::sync::Arc;

use super::*;
use crate::trace::{
    LastTrace, NodeInfo, PropagationTrace, Retain, RetainedTrace, StampPair, TraceCapture,
    TracePayload,
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

    /// Keeps `payload` for the root of the `get_traced_retaining` call
    /// that returned `capture`, when that call's last entry for the root
    /// is `Recomputed` and the root's stamps have not moved since the
    /// entry. When the call's only root entry is `VerifiedClean` and the
    /// kept payload described the root as the call found it, the kept
    /// payload is moved to the verified stamps and `payload` is dropped.
    /// Anything else leaves the kept payload as it is. A payload this
    /// replaces, or `payload` when it is not kept, drops after the
    /// runtime lock is released.
    pub fn retain_trace(&self, capture: TraceCapture, payload: TracePayload) {
        let slot = capture.slot;
        let replaced = match capture.decide() {
            Retain::Nothing => None,
            Retain::Keep(stamps) => {
                let mut inner = self.inner.write();
                if self.still_at(&capture, stamps) {
                    let kept = RetainedTrace {
                        generation: capture.generation,
                        stamps,
                        payload,
                    };
                    inner.last_traces.insert(slot, kept).map(|old| old.payload)
                } else {
                    Some(payload)
                }
            }
            Retain::Restamp { from, to } => {
                let mut inner = self.inner.write();
                if self.still_at(&capture, to) {
                    if let Some(kept) = inner
                        .last_traces
                        .get_mut(&slot)
                        .filter(|kept| kept.generation == capture.generation)
                        .filter(|kept| kept.stamps == from)
                    {
                        kept.stamps = to;
                    }
                }
                Some(payload)
            }
        };
        drop(replaced);
    }

    /// The state of the live node at `id` and `generation`, and the payload
    /// kept for it by [`Self::retain_trace`] if the node is Clean at the
    /// stamps the payload was kept with; `None` when no live node has that
    /// slot and generation. Runs no compute and records no dependency.
    pub fn last_trace(&self, id: NodeId, generation: u32) -> Option<LastTrace> {
        self.erased_handle_at(id, generation)?;
        let node = self.nodes.get(id.0);
        let inner = self.inner.read();
        let state = state::load::<C>(node.state_cell());
        if state == NodeState::Dead || node.generation() != generation {
            return None;
        }
        let payload = inner
            .last_traces
            .get(&id.0)
            .filter(|kept| kept.generation == generation && state == NodeState::Clean)
            .filter(|kept| kept.stamps.is_landed() && self.stamp_pair(id.0) == kept.stamps)
            .map(|kept| Arc::clone(&kept.payload));
        Some(LastTrace { state, payload })
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
        self.nodes.get(capture.slot).generation() == capture.generation
            && self.stamp_pair(capture.slot) == stamps
    }
}
