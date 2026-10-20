//! Dirty-driven watching: `watch`, `unwatch`, `drain_dirtied_watched`,
//! and the queue that the dirty walk and the landings off the Clean path
//! fill.
//!
//! Every push and removal happens inside an `inner` section that read
//! `WATCHED`, and the delete that bumps a generation removes the node's
//! entry under its write hold, so whenever `inner` is free every queued
//! node is watched and live. Lock order: `inner` before the queue, and
//! the queue lock is a leaf.

use std::collections::VecDeque;

use super::*;

/// The watched nodes moved off Clean since the last drain, each at most
/// once, in the order they were found, with the `QUEUED` bitset beside
/// them. Entries carry the generation read when they were pushed.
pub(crate) struct WatchQueue {
    entries: VecDeque<(u32, u32)>,
    queued: Vec<u64>,
}

impl WatchQueue {
    pub(crate) fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            queued: Vec::new(),
        }
    }

    fn is_queued(&self, slot: u32) -> bool {
        self.queued
            .get(slot as usize / 64)
            .is_some_and(|w| w & (1 << (slot % 64)) != 0)
    }

    pub(super) fn push(&mut self, slot: u32, generation: u32) {
        if self.mark(slot) {
            self.entries.push_back((slot, generation));
        }
    }

    fn push_front(&mut self, slot: u32, generation: u32) {
        if self.mark(slot) {
            self.entries.push_front((slot, generation));
        }
    }

    /// Sets `slot`'s `QUEUED` bit; false if it was set already.
    fn mark(&mut self, slot: u32) -> bool {
        if self.is_queued(slot) {
            return false;
        }
        let word = slot as usize / 64;
        if word >= self.queued.len() {
            self.queued.resize(word + 1, 0);
        }
        self.queued[word] |= 1 << (slot % 64);
        true
    }

    fn remove(&mut self, slot: u32) {
        if !self.is_queued(slot) {
            return;
        }
        self.queued[slot as usize / 64] &= !(1 << (slot % 64));
        let at = self
            .entries
            .iter()
            .position(|&(s, _)| s == slot)
            .expect("incr-core: QUEUED is set on a slot with no entry");
        self.entries.remove(at);
    }

    pub(super) fn take(&mut self) -> VecDeque<(u32, u32)> {
        for &(slot, _) in &self.entries {
            self.queued[slot as usize / 64] &= !(1 << (slot % 64));
        }
        std::mem::take(&mut self.entries)
    }
}

impl<C: Cells> Runtime<C> {
    /// Report the node through [`Self::drain_dirtied_watched`] whenever
    /// a change moves it off Clean. A node that is not Clean (or Cyclic)
    /// when watched is reported at once. Watching a watched node does
    /// nothing. Allowed inside a compute.
    pub fn watch<T: Value>(&self, handle: Incr<T>) {
        value_code::check();
        self.check_handle(handle);
        let slot = handle.slot();
        let inner = self.inner.write();
        if self.nodes.get(slot).generation() != handle.generation() {
            drop(inner);
            self.stale_handle(handle);
        }
        self.watch_held(slot, handle.generation());
        drop(inner);
    }

    /// [`Self::watch`] under its caller's `inner.write()`, for a node
    /// whose generation it compared there.
    pub(super) fn watch_held(&self, slot: u32, generation: u32) {
        let (node, flags) = self.nodes.get_with_flags(slot);
        let bits = C::u8_load_relaxed(flags);
        if bits & WATCHED == 0 {
            C::u8_store_relaxed(flags, bits | WATCHED);
            // A landing whose read section came first has its state
            // stored by now, and Clean and Cyclic landings never report.
            if !matches!(node.state(), NodeState::Clean | NodeState::Cyclic) {
                self.watch_queue.write().push(slot, generation);
            }
        }
    }

    /// Stop reporting the node, and drop its pending report. Unwatching
    /// a node that is not watched does nothing. Allowed inside a compute.
    pub fn unwatch<T: Value>(&self, handle: Incr<T>) {
        value_code::check();
        self.check_handle(handle);
        let slot = handle.slot();
        let flags = self.nodes.get_with_flags(slot).1;
        let inner = self.inner.write();
        if self.nodes.get(slot).generation() != handle.generation() {
            drop(inner);
            self.stale_handle(handle);
        }
        let bits = C::u8_load_relaxed(flags);
        if bits & WATCHED != 0 {
            C::u8_store_relaxed(flags, bits & !WATCHED);
            self.park(ParkPoint::UnwatchBeforeDequeue(NodeId(slot)));
            // Under the same hold, so the slot cannot have been recycled
            // and watched again since the bit was cleared.
            self.watch_queue.write().remove(slot);
        }
        drop(inner);
    }

    /// The watched nodes a change moved off Clean since the last drain,
    /// each once, in the order the dirty walks found them. A node left
    /// Dirty by an earlier drain and not read since is not reported
    /// again, because the next walk stops at it.
    pub fn drain_dirtied_watched(&self) -> Vec<ErasedIncr> {
        value_code::check();
        let taken = self.watch_queue.write().take();
        taken
            .into_iter()
            .map(|(slot, generation)| ErasedIncr::new(slot, generation, self.runtime_id))
            .collect()
    }

    /// The queued slots, in order.
    #[cfg(all(test, not(loom)))]
    pub(crate) fn queued_slots(&self) -> Vec<u32> {
        let queue = self.watch_queue.read();
        queue.entries.iter().map(|&(slot, _)| slot).collect()
    }

    /// Queue `id`, which the caller found watched under its `inner` hold.
    #[cold]
    #[inline(never)]
    pub(super) fn enqueue_watched(&self, id: NodeId, node: &NodeData<C>) {
        self.watch_queue.write().push(id.0, node.generation());
    }

    /// After a landing whose target is neither Clean nor Cyclic: queue
    /// the node if it is watched. The load sits in a read section so that
    /// a concurrent `watch` either precedes it or sees the landed state.
    #[cold]
    #[inline(never)]
    pub(super) fn report_landing(&self, id: NodeId) {
        let inner = self.inner.read();
        let (node, flags) = self.nodes.get_with_flags(id.0);
        if C::u8_load_relaxed(flags) & WATCHED != 0 {
            self.enqueue_watched(id, node);
        }
        drop(inner);
    }

    /// Put `entries` back at the front of the queue, in their order,
    /// skipping any that is queued already, no longer watched, dead or
    /// recycled. The checks and pushes share one `inner` section, so no
    /// `unwatch` or delete falls between a check and its push.
    pub(super) fn requeue_front(&self, entries: impl DoubleEndedIterator<Item = (u32, u32)>) {
        let inner = self.inner.read();
        let mut queue = self.watch_queue.write();
        for (slot, generation) in entries.rev() {
            let (node, flags) = self.nodes.get_with_flags(slot);
            if node.generation() == generation
                && C::u8_load_relaxed(flags) & WATCHED != 0
                && node.state() != NodeState::Dead
            {
                queue.push_front(slot, generation);
            }
        }
        drop(queue);
        drop(inner);
    }

    /// The delete's part, under its write hold: a watched node's entry
    /// goes with it. The caller clears the flags byte.
    #[inline(always)]
    pub(super) fn unwatch_deleted(&self, slot: u32, bits: u8) {
        if bits & WATCHED != 0 {
            self.watch_queue.write().remove(slot);
        }
    }
}
