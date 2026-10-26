//! Observers: the registry, `stabilize` over the watch queue, and
//! `observer_read`, the read an observer's run makes. That read is the
//! read path of `try_get` with every stale-handle panic turned into
//! `ObsErr::Gone`, because a Shared delete can land between `stabilize`
//! taking a node's observers and their run, and a Failed observed node's
//! panic into `ObsErr::Failed`, so it does not stop `stabilize`.

use std::any::Any;
use std::cell::Cell;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use super::*;

/// Type-erased observer runner: takes the runtime and the observer's
/// last-seen stamp, returns the new stamp if it advanced.
type ObserverRunFn<C> = Arc<dyn Fn(&Runtime<C>, u64) -> Option<u64> + Send + Sync>;

/// One registered observer: its slot, the change stamp it last fired
/// (or skipped a cycle) for, and its runner.
pub(crate) struct ObserverEntry<C: Cells> {
    slot: u32,
    last_seen: u64,
    run: ObserverRunFn<C>,
}

/// An observer taken for a run: its id, last-seen stamp and runner.
type ObserverRun<C> = (u64, u64, ObserverRunFn<C>);

/// The observer registry: entries by id, and each observed slot's ids in
/// registration order.
pub(crate) struct Observers<C: Cells> {
    by_id: HashMap<u64, ObserverEntry<C>>,
    by_slot: HashMap<u32, Vec<u64>>,
    next_id: u64,
}

impl<C: Cells> Observers<C> {
    pub(crate) fn new() -> Self {
        Self {
            by_id: HashMap::new(),
            by_slot: HashMap::new(),
            next_id: 1,
        }
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    fn insert(&mut self, slot: u32, last_seen: u64, run: ObserverRunFn<C>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.by_id.insert(
            id,
            ObserverEntry {
                slot,
                last_seen,
                run,
            },
        );
        self.by_slot.entry(slot).or_default().push(id);
        id
    }

    fn remove(&mut self, id: u64) -> Option<ObserverEntry<C>> {
        let entry = self.by_id.remove(&id)?;
        if let Some(ids) = self.by_slot.get_mut(&entry.slot) {
            ids.retain(|&other| other != id);
            if ids.is_empty() {
                self.by_slot.remove(&entry.slot);
            }
        }
        Some(entry)
    }

    /// A deleted node's observers.
    pub(super) fn take_slot(&mut self, slot: u32) -> Vec<ObserverEntry<C>> {
        if self.by_id.is_empty() {
            return Vec::new();
        }
        let ids = self.by_slot.remove(&slot).unwrap_or_default();
        ids.iter().filter_map(|id| self.by_id.remove(id)).collect()
    }

    fn runs_of(&self, slot: u32) -> Vec<ObserverRun<C>> {
        self.by_slot.get(&slot).map_or_else(Vec::new, |ids| {
            ids.iter()
                .map(|id| {
                    let e = &self.by_id[id];
                    (*id, e.last_seen, Arc::clone(&e.run))
                })
                .collect()
        })
    }

    fn advance(&mut self, id: u64, stamp: u64) {
        if let Some(e) = self.by_id.get_mut(&id) {
            e.last_seen = e.last_seen.max(stamp);
        }
    }
}

/// The entries one `stabilize` drained, each with its node's observers,
/// and how many it has finished. Dropped before the end, by a wound or a
/// panic, it puts the rest back, the one in progress included.
struct Pending<'a, C: Cells> {
    rt: &'a Runtime<C>,
    entries: Vec<(u32, u32, Vec<ObserverRun<C>>)>,
    next: usize,
}

impl<C: Cells> Drop for Pending<'_, C> {
    fn drop(&mut self) {
        if self.next < self.entries.len() {
            self.rt.requeue_front(
                self.entries[self.next..]
                    .iter()
                    .map(|&(slot, generation, _)| (slot, generation)),
            );
        }
    }
}

/// Why an observer's read has no value to fire with.
#[derive(Debug)]
pub(crate) enum ObsErr {
    /// The node was deleted, or its slot recycled, since the observer
    /// was taken; the observer went with it.
    Gone,
    /// The node has no value because of a cycle; the stamp of its last
    /// landing.
    Cycle(u64),
    /// The node's last compute panicked; the stamp of its last landing.
    Failed(u64),
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
    /// Register a callback fired by [`Self::stabilize`] whenever the
    /// node's value has changed since the last firing. Observation does
    /// not change the engine's pull model: nothing runs until someone
    /// calls `stabilize` (or `get`); the observer registry is a batch
    /// poll with last-seen bookkeeping, the same shape as Jane Street
    /// Incremental's observers.
    ///
    /// Observing a node watches it ([`Self::watch`]): `stabilize` runs
    /// the observers of the nodes a change moved off Clean, and after an
    /// [`Self::unwatch`] the node's observers stay quiet until it is
    /// watched again.
    ///
    /// A node that has never computed fires on the first stabilize, and a
    /// node that is Failed or Cyclic does not fire until it has a value. A
    /// stabilize racing a concurrent `set` may fire twice for one
    /// change; it misses one only when a [`Self::drain_dirtied_watched`]
    /// took the report first.
    pub fn observe<T: Value, F>(&self, handle: Incr<T>, callback: F) -> ObserverId
    where
        F: Fn(&T) + Send + Sync + 'static,
    {
        value_code::check();
        self.check_handle(handle);
        let run = Arc::new(
            move |rt: &Runtime<C>, last_seen: u64| match rt.observer_read(handle) {
                Ok((value, stamp)) if stamp > last_seen => {
                    callback(&value);
                    Some(stamp)
                }
                Err(ObsErr::Cycle(stamp) | ObsErr::Failed(stamp)) if stamp > last_seen => {
                    Some(stamp)
                }
                Ok(_) | Err(ObsErr::Cycle(_) | ObsErr::Failed(_) | ObsErr::Gone) => None,
            },
        );
        let slot = handle.slot();
        let node = self.nodes.get(slot);
        let mut inner = self.inner.write();
        if node.generation() != handle.generation() {
            drop(inner);
            self.stale_handle(handle);
        }
        self.watch_held(slot, handle.generation());
        // Queued even if it was watched already: a drain may have taken
        // its entry, and the walk stops at a node that is not Clean.
        if !matches!(node.state(), NodeState::Clean | NodeState::Cyclic) {
            self.watch_queue.write().push(slot, handle.generation());
        }
        // The baseline is loaded after that, so a landing it misses was
        // off Clean then and is queued. A node with a clean value fires
        // only on later changes; a never-computed one starts at 0 so the
        // first stabilize computes and fires.
        let last_seen = match state::load::<C>(node.state_cell()) {
            NodeState::New => 0,
            _ => self.observed_stamp(node),
        };
        let id = inner.observers.insert(slot, last_seen, run);
        drop(inner);
        ObserverId(id)
    }

    /// Remove an observer. Safe to call with an already-removed id.
    pub fn unobserve(&self, id: ObserverId) {
        value_code::check();
        // Bound to a local so the entry, whose callback is user code,
        // drops after the write guard does.
        let removed = self.inner.write().observers.remove(id.0);
        drop(removed);
    }

    /// Run the observers of the watched nodes reported since the last
    /// drain, in the order they were reported: each node is brought up
    /// to date and its callbacks fire if its value changed since they
    /// last fired. The reports are the queue [`Self::drain_dirtied_watched`]
    /// takes, so a drain between two stabilizes takes them from the
    /// second. A node whose compute panics, leaving it Failed, fires none
    /// of its callbacks and does not stop the run. If a callback panics,
    /// the panic leaves `stabilize` and the nodes not finished, the one in
    /// progress included, are reported again at the front of the queue, so
    /// a callback that panics every time blocks the observers queued after
    /// it.
    ///
    /// Callbacks run with no engine lock held; they may read from the
    /// runtime but must not call `stabilize` reentrantly from another
    /// thread if duplicate notifications matter.
    pub fn stabilize(&self) {
        value_code::check();
        self.enter(|| self.stabilize_entered());
    }

    fn stabilize_entered(&self) {
        let mut pending = Pending {
            rt: self,
            entries: self.drain_observed(),
            next: 0,
        };
        self.park(ParkPoint::StabilizeAfterDrain);
        while pending.next < pending.entries.len() {
            let runs = std::mem::take(&mut pending.entries[pending.next].2);
            for (id, last_seen, run) in runs {
                if let Some(stamp) = run(self, last_seen) {
                    self.inner.write().observers.advance(id, stamp);
                }
            }
            pending.next += 1;
        }
    }

    /// The queue's entries, each with its node's observers, taken in one
    /// `inner` section, in which every queued node is live and watched.
    fn drain_observed(&self) -> Vec<(u32, u32, Vec<ObserverRun<C>>)> {
        let inner = self.inner.read();
        let taken = self.watch_queue.write().take();
        let entries = taken
            .into_iter()
            .map(|(slot, generation)| (slot, generation, inner.observers.runs_of(slot)))
            .collect();
        drop(inner);
        entries
    }

    /// Bring the node up to date and read it with its change stamp, as
    /// `try_get` does, except that a node deleted or recycled under the
    /// handle reads `Gone`, and a Failed node `Failed`, instead of
    /// panicking. Each stamp is loaded
    /// before the state or value it goes with, so a change landing
    /// between them fires again rather than not at all.
    ///
    /// Inside a compute the read records its dependency, and a read the
    /// frame's rules settle (a node held by this thread, or a doomed or
    /// discarded run) does not fire.
    pub(super) fn observer_read<T: Value>(&self, handle: Incr<T>) -> Result<(T, u64), ObsErr> {
        let id = NodeId(handle.slot());
        let gen = handle.generation();
        let node = self.nodes.get(id.0);
        let (observed, entry_stamp) = {
            let _inner = self.inner.read();
            if node.generation() != gen {
                return Err(ObsErr::Gone);
            }
            let stamp = self.observed_stamp(node);
            C::fence_acquire();
            (state::load::<C>(node.state_cell()), stamp)
        };
        // No kept read carries an observer's read to a retry of another
        // runtime's frame.
        if self
            .dep_stack
            .record_dep_or_foreign(self.runtime_id, id, gen)
        {
            self.dep_stack.hold_foreign_reader();
        }
        self.park(ParkPoint::ObserverAfterEntryCheck(id));
        if observed != NodeState::Clean {
            self.observer_walk(handle, observed, entry_stamp)?;
        }
        self.park(ParkPoint::ObserverAfterEnsureClean(id));
        if node.generation() != gen {
            return Err(ObsErr::Gone);
        }
        let stamp = self.observed_stamp(node);
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
            None => self.observer_read_missing(handle, node, stamp),
        }
    }

    /// `observer_read`'s walk of a node it found not Clean, under the
    /// record that lets the Dead arm return `Gone` for this node.
    /// `stamp` was loaded before `observed`.
    #[cold]
    #[inline(never)]
    fn observer_walk<T: Value>(
        &self,
        handle: Incr<T>,
        observed: NodeState,
        stamp: u64,
    ) -> Result<(), ObsErr> {
        let _kept = cycle_state::KeptUnwind::if_unwinding();
        cycle_state::raise_if_lazy();
        let id = NodeId(handle.slot());
        if observed == NodeState::Cyclic || is_computing(observed) {
            // `Ok` is a node on this thread's stack or held map, in a
            // compute: its value is not settled, so nothing fires.
            if self
                .cycle_read(handle, observed, ReadKind::TryGet, None)
                .is_some()
            {
                return Err(ObsErr::Cycle(stamp));
            }
        }
        let _observing = Observing::record(self.runtime_id, id.0);
        let walked = catch_unwind(AssertUnwindSafe(|| {
            self.entered(ReadKind::TryGet, || {
                self.ensure_clean(id, handle.generation())
            })
        }));
        let computed = match walked {
            Ok(computed) => computed.map_err(|_| ObsErr::Cycle(stamp))?,
            Err(payload) => return Err(self.observed_failed(handle, payload)),
        };
        match computed {
            Computed::Done | Computed::Unsettled => Ok(()),
            Computed::Gone => Err(ObsErr::Gone),
            Computed::Deferred => Err(ObsErr::Cycle(stamp)),
            // The reading run is thrown away and its retry reads again.
            Computed::Doomed | Computed::Discarded => Err(ObsErr::Gone),
            Computed::Skipped(_) | Computed::Replan | Computed::Revisit => {
                unreachable!("incr-core: a walk's root is never skipped or replanned")
            }
        }
    }

    /// An unwind out of `observer_walk`: a user panic that left the
    /// observed node Failed is that node's error, so `stabilize` goes on to
    /// the next observer; a wound, an engine unwind or any other panic
    /// unwinds on.
    #[cold]
    #[inline(never)]
    fn observed_failed<T: Value>(&self, handle: Incr<T>, payload: Box<dyn Any + Send>) -> ObsErr {
        let node = self.nodes.get(handle.slot());
        let stamp = self.observed_stamp(node);
        C::fence_acquire();
        let failed = !cycle_state::wounded()
            && cycle_state::unwind_kind().is_none()
            && node.state() == NodeState::Failed
            && node.generation() == handle.generation();
        if !failed {
            resume_unwind(payload);
        }
        ObsErr::Failed(stamp)
    }

    /// `observer_read` found no value under an unmoved generation: the
    /// node is without one because of a cycle, or a delete has claimed
    /// it since the walk.
    #[cold]
    #[inline(never)]
    fn observer_read_missing<T: Value>(
        &self,
        handle: Incr<T>,
        node: &NodeData<C>,
        stamp: u64,
    ) -> Result<(T, u64), ObsErr> {
        let state = node.state();
        if node.generation() != handle.generation() {
            return Err(ObsErr::Gone);
        }
        match state {
            NodeState::Cyclic | NodeState::New => Err(ObsErr::Cycle(stamp)),
            NodeState::Dead | NodeState::Deleting => Err(ObsErr::Gone),
            NodeState::Failed => Err(ObsErr::Failed(stamp)),
            _ => self.read_failed(handle),
        }
    }

    /// The node's change stamp, with a pending `set`'s marker clamped to
    /// the current revision so that a later landing still exceeds it.
    fn observed_stamp(&self, node: &NodeData<C>) -> u64 {
        node.changed_at().min(self.current_revision())
    }
}
