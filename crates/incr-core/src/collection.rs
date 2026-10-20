//! `IncrCollection<T, C>`: incremental collection with delta-log propagation.
//!
//! Each collection holds a delta log of inserts and deletes plus an
//! `Incr<u64>` version node. Operators (filter, map, count, aggregate,
//! reduce, join, group_by) are query closures that scan new deltas since
//! their last evaluation cursor and update their own state incrementally.
//!
//! ## Consumer cursors and compaction
//!
//! Every operator registers a cursor with its upstream log at creation
//! time and bootstraps its state from a snapshot of the live elements,
//! so the log never needs history older than the slowest registered
//! cursor. `maybe_compact` periodically drops the delta prefix all
//! consumers have passed (all deltas, if there are no consumers), which
//! bounds log memory by consumer lag instead of by collection lifetime.
//!
//! ## Stage-then-apply
//!
//! User closures (predicates, mappers, key extractors, `lift`) run with
//! NO lock held: each operator clones the pending delta slice out of the
//! upstream log, evaluates user code into staged effects, and only then
//! takes its own locks to apply them and advance the cursor. A panicking
//! closure therefore applies nothing, the cursor stays put, and the
//! compute panic boundary retries the whole batch cleanly. The exception
//! is `aggregate`'s `combine`, which runs under the aggregate's state
//! lock; only that aggregate's compute ever takes it.
//!
//! The apply phase still runs user `Hash`, `Eq`, `Clone` and `Drop`, and
//! `combine`. A panic there leaves the batch partly applied, so the
//! operator marks itself for a rebuild (`NeedsRebuild`). Its next run
//! reads each upstream's snapshot and end under one guard, rebuilds its
//! state into fresh structures, converges each output log to the
//! recomputed contents with ordinary deltas, and moves the cursor to the
//! snapshot's end.
//!
//! Source mutations (`insert`, `delete`, `replace`) hash and clone with
//! no lock held, run their lookups (user `Eq`) before changing anything,
//! and drop what they displaced only after the version node is set, so a
//! panic in any user impl leaves the collection either untouched or
//! fully changed and published.
//!
//! The exactly-once cursor discipline is safe because the runtime's
//! claim protocol guarantees a single thread executes a given operator
//! closure at a time.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, RandomState};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as MemOrdering};
use std::sync::{Arc, OnceLock};

use crate::cells::Cells;
use crate::handle::{ErasedIncr, Incr};
use crate::hashed_multiset::HashedMultiset;
use crate::lock_order::{self, LockKind, Ordered};
use crate::locks::{Lock, OrderedLock};
use crate::persist_hooks::{RepSide, RepSink};
use crate::runtime::{ParkPoint, Runtime};
use crate::value::Value;
use crate::value_code;

#[cfg(test)]
pub(crate) fn park<C: Cells>(rt: &Runtime<C>, point: ParkPoint) {
    if let Some(hook) = rt.park_hook.get() {
        hook(point);
    }
}

#[cfg(not(test))]
#[inline(always)]
pub(crate) fn park<C: Cells>(_rt: &Runtime<C>, _point: ParkPoint) {}

/// Strategy-parameterized lock for operator state. Under `Local` this is
/// a `RefCell` (no atomics, which is the point: collection ops on the
/// single-threaded runtime pay no synchronization); under `Shared` it is
/// the poison-ignoring `RwLock`.
pub(crate) struct OpLock<T: 'static, C: Cells>(OrderedLock<T, C>);

impl<T: 'static, C: Cells> OpLock<T, C> {
    pub(crate) fn new(kind: LockKind, v: T) -> Self {
        Self(OrderedLock::new(kind, v))
    }

    pub(crate) fn read(&self) -> Ordered<<C::Lock<T> as Lock<T>>::ReadGuard<'_>> {
        self.0.read()
    }

    pub(crate) fn write(&self) -> Ordered<<C::Lock<T> as Lock<T>>::WriteGuard<'_>> {
        self.0.write()
    }
}

// SAFETY: required so operator closures (ComputeFn: Send + Sync) can
// capture Arc<OpLock<..>> under both strategies. The impls are
// unconditional (no T: Send bound) because group_by stores
// IncrCollection values whose !Send-ness under Local is itself only the
// artificial confinement marker. The claim is sound per strategy:
// - Shared: every in-crate instantiation wraps types that are genuinely
//   Send + Sync under Shared (logs, indexes, mirrors of T: Value).
// - Local: the marker is never exercised across threads. The closures
//   live inside Runtime<Local> (!Send + !Sync), and user-facing handles
//   carry a PhantomData<C::Ptr<()>> confinement marker making them
//   !Send + !Sync under Local, so no OpLock<_, Local> is ever reachable
//   from a second thread.
// OpLock is pub(crate); auditing instantiations is the enforcement.
unsafe impl<T: 'static, C: Cells> Send for OpLock<T, C> {}
unsafe impl<T: 'static, C: Cells> Sync for OpLock<T, C> {}

/// One delta event in a collection log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Delta<T> {
    Insert(T),
    Delete(T),
}

/// Compaction batch: how many deltas accumulate before a compaction
/// sweep looks at consumer cursors. Sweeps are O(consumers) plus the
/// drain memmove, so they are amortized across the batch.
const COMPACT_EVERY: usize = 1024;

/// Delta log + multiset bookkeeping for a single collection.
///
/// `deltas[0]` corresponds to absolute index `base`; consumer cursors
/// are absolute and monotonic. `elements` is the live multiset used to
/// validate deletes and to bootstrap new consumers.
pub struct CollectionLog<T: Hash + Eq + Clone> {
    pub(crate) deltas: Vec<Delta<T>>,
    /// An empty buffer the next compaction moves the kept deltas into.
    pub(crate) spare: Vec<Delta<T>>,
    pub(crate) base: u64,
    pub(crate) cursors: Vec<Arc<AtomicU64>>,
    pub(crate) elements: HashedMultiset<T>,
    pub(crate) hasher: RandomState,
    pub(crate) version: u64,
}

impl<T: Hash + Eq + Clone> CollectionLog<T> {
    pub fn new() -> Self {
        Self {
            deltas: Vec::new(),
            spare: Vec::new(),
            base: 0,
            cursors: Vec::new(),
            elements: HashedMultiset::new(),
            hasher: RandomState::new(),
            version: 0,
        }
    }

    /// Absolute index one past the newest delta.
    pub(crate) fn end(&self) -> u64 {
        self.base + self.deltas.len() as u64
    }

    /// Pending deltas for a consumer positioned at absolute `from`.
    /// `from` can never be below `base`: compaction never passes the
    /// minimum registered cursor.
    pub(crate) fn pending_from(&self, from: u64) -> &[Delta<T>] {
        &self.deltas[(from - self.base) as usize..]
    }

    /// A copy of the pending deltas (user `Clone`), for a consumer to
    /// stage outside this guard.
    pub(crate) fn pending_cloned(&self, from: u64) -> Vec<Delta<T>> {
        value_code::run(|| self.pending_from(from).to_vec())
    }

    /// Register a new consumer: returns its cursor (positioned at the
    /// current end) and a bootstrap snapshot of the live multiset that
    /// stands in for the compacted history.
    pub(crate) fn register_consumer(&mut self) -> (Arc<AtomicU64>, Vec<T>) {
        let cursor = Arc::new(AtomicU64::new(self.end()));
        self.cursors.push(Arc::clone(&cursor));
        (cursor, self.elements_vec())
    }

    /// Cut off the delta prefix every registered consumer has passed and
    /// return it, for the caller to drop when running user `Drop` is
    /// safe. Consumers whose cursor Arc was dropped (operator gone) are
    /// pruned. With no consumers the whole log is cut eagerly.
    #[inline(always)]
    fn maybe_compact(&mut self) -> Vec<Delta<T>> {
        if self.deltas.len() < COMPACT_EVERY {
            return Vec::new();
        }
        self.compact()
    }

    #[cold]
    #[inline(never)]
    fn compact(&mut self) -> Vec<Delta<T>> {
        self.cursors.retain(|c| Arc::strong_count(c) > 1);
        let min_cursor = self
            .cursors
            .iter()
            .map(|c| c.load(MemOrdering::Acquire))
            .min()
            .unwrap_or_else(|| self.end());
        if min_cursor <= self.base {
            return Vec::new();
        }
        let cut = (min_cursor - self.base) as usize;
        let mut kept = std::mem::take(&mut self.spare);
        kept.reserve(self.deltas.capacity());
        kept.extend(self.deltas.drain(cut..));
        let compacted = std::mem::replace(&mut self.deltas, kept);
        self.base = min_cursor;
        compacted
    }

    /// Takes back a compaction's buffer once the caller has dropped its
    /// deltas. Reusing it spares each compaction a large allocation, which
    /// would make glibc consolidate its fastbins and slow later small ones.
    fn recycle(&mut self, buffer: Vec<Delta<T>>) {
        debug_assert!(buffer.is_empty());
        if buffer.capacity() > self.spare.capacity() {
            self.spare = buffer;
        }
    }

    /// Phases 2 and 3 of an insert, under the log guard. The lookup (user
    /// `Eq`) runs before anything changes; from the first change on no
    /// user code runs, and what the insert displaced comes back for the
    /// caller to drop.
    pub(crate) fn insert_staged<R: RepSink<T>>(
        &mut self,
        staged: StagedInsert<T>,
        on_rep: &mut R,
    ) -> Committed<T> {
        let at = value_code::run(|| {
            let at = self.elements.locate_add(staged.hash, &staged.stored);
            if R::ACTIVE {
                let rep = self.elements.stored_after_add(at, &staged.stored);
                on_rep.on_rep(RepSide::Added, rep);
            }
            at
        });
        let unused = self.elements.add(at, staged.stored);
        self.deltas.push(Delta::Insert(staged.delta));
        self.version += 1;
        Committed {
            removed: false,
            version: Some(self.version),
            displaced: Displaced {
                values: [unused],
                compacted: self.maybe_compact(),
            },
        }
    }

    /// Phases 2 and 3 of a delete, as [`Self::insert_staged`]. A value
    /// that is not present records nothing and comes back displaced.
    pub(crate) fn delete_staged<R: RepSink<T>>(
        &mut self,
        staged: StagedDelete<T>,
        on_rep: &mut R,
    ) -> Committed<T> {
        let found = value_code::run(|| {
            let found = self.elements.find(staged.hash, &staged.value);
            if let (true, Some(i)) = (R::ACTIVE, found) {
                on_rep.on_rep(RepSide::Removed, self.elements.stored(i));
            }
            found
        });
        let (version, value) = match found {
            Some(i) => {
                let stored = self.elements.decrement_or_remove(i);
                self.deltas.push(Delta::Delete(staged.value));
                self.version += 1;
                (Some(self.version), stored)
            }
            None => (None, Some(staged.value)),
        };
        Committed {
            removed: version.is_some(),
            version,
            displaced: Displaced {
                values: [value],
                compacted: self.maybe_compact(),
            },
        }
    }

    /// Phases 2 and 3 of a replace: both lookups run before the first
    /// change, the add is applied before the remove, and both deltas are
    /// pushed under this one guard.
    fn replace_staged<R: RepSink<T>>(
        &mut self,
        staged: StagedReplace<T>,
        on_rep: &mut R,
    ) -> Committed<T, 2> {
        let at = value_code::run(|| {
            let at = self.elements.locate(
                staged.remove.as_ref().map(|s| (s.hash, &s.value)),
                Some((staged.add.hash, &staged.add.stored)),
            );
            if R::ACTIVE {
                let [removed, added] = self.elements.representatives(at, &staged.add.stored);
                if let Some(rep) = removed {
                    on_rep.on_rep(RepSide::Removed, rep);
                }
                if let Some(rep) = added {
                    on_rep.on_rep(RepSide::Added, rep);
                }
            }
            at
        });
        let removed = at.removes();
        let [first, second] = self.elements.apply(at, Some(staged.add.stored));
        let mut unused = None;
        match staged.remove {
            Some(remove) if removed => {
                self.deltas.push(Delta::Delete(remove.value));
                self.version += 1;
            }
            other => unused = other.map(|s| s.value),
        }
        self.deltas.push(Delta::Insert(staged.add.delta));
        self.version += 1;
        Committed {
            removed,
            version: Some(self.version),
            displaced: Displaced {
                // A found remove consumes its delta copy, so at most one of
                // `second` and `unused` is set.
                values: [first, second.or(unused)],
                compacted: self.maybe_compact(),
            },
        }
    }

    /// One change of an operator's batch: applies `staged` and keeps what
    /// it displaced in `keep`, for the caller to drop once it releases its
    /// guards. Returns the new version, if a delta was recorded.
    #[inline]
    pub(crate) fn apply_kept(
        &mut self,
        staged: Staged<T>,
        keep: &mut Vec<Displaced<T>>,
    ) -> Option<u64> {
        let committed = match staged {
            Staged::Insert(s) => self.insert_staged(s, &mut ()),
            Staged::Delete(s) => self.delete_staged(s, &mut ()),
        };
        committed.keep(keep)
    }

    pub(crate) fn insert_kept(&mut self, value: T, keep: &mut Vec<Displaced<T>>) -> Option<u64> {
        let staged = StagedInsert::new(&self.hasher, value);
        self.insert_staged(staged, &mut ()).keep(keep)
    }

    pub(crate) fn delete_kept(&mut self, value: T, keep: &mut Vec<Displaced<T>>) -> Option<u64> {
        let staged = StagedDelete::owned(&self.hasher, value);
        self.delete_staged(staged, &mut ()).keep(keep)
    }

    /// The live elements and the log's end, under one guard, so the deltas
    /// from the end on are exactly the changes the snapshot lacks.
    pub(crate) fn snapshot(&self) -> (Vec<T>, u64) {
        (self.elements_vec(), self.end())
    }

    /// Pushes the deltas that turn the live contents into `target`, one
    /// change at a time. Each change is atomic under user panics, so a
    /// panic part-way leaves a log that a later converge finishes from.
    /// Returns the new version, if anything changed.
    pub(crate) fn converge(&mut self, target: Vec<T>, keep: &mut Vec<Displaced<T>>) -> Option<u64> {
        let mut wanted: HashMap<T, usize> = HashMap::new();
        let mut surplus = Vec::new();
        value_code::run(|| {
            for value in target {
                *wanted.entry(value).or_default() += 1;
            }
            for (value, have) in self.elements.iter() {
                let kept = wanted.get_mut(value).map_or(0, |want| {
                    let kept = have.min(*want);
                    *want -= kept;
                    kept
                });
                surplus.extend(std::iter::repeat_n(value, have - kept).cloned());
            }
        });
        let mut version = None;
        for value in surplus {
            version = self.delete_kept(value, keep).or(version);
        }
        for (value, missing) in wanted {
            for _ in 0..missing {
                version = self.insert_kept(cloned(&value), keep).or(version);
            }
        }
        version
    }

    /// Insert `value`. Always accepted; multiset count for the element
    /// is incremented. Returns the new version.
    pub fn insert(&mut self, value: T) -> u64 {
        let staged = StagedInsert::new(&self.hasher, value);
        let committed = self.insert_staged(staged, &mut ());
        committed.version.expect("an insert always records a delta")
    }

    /// Delete one occurrence of `value`. Returns `Some(new_version)` if
    /// the element was present and a delete was recorded; `None` if the
    /// element was not in the collection (no delta recorded).
    pub fn delete(&mut self, value: &T) -> Option<u64> {
        let staged = StagedDelete::cloned(&self.hasher, value);
        self.delete_staged(staged, &mut ()).version
    }

    /// Snapshot of all live elements, with multiset duplicates expanded.
    pub fn elements_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len());
        value_code::run(|| {
            for (val, count) in self.elements.iter() {
                for _ in 0..count {
                    out.push(val.clone());
                }
            }
        });
        out
    }

    /// Number of live elements, with multiset duplicates counted.
    pub(crate) fn len(&self) -> usize {
        self.elements.iter().map(|(_, count)| count).sum()
    }
}

/// Phase 1 of an insert, computed with no lock held: the hash (user
/// `Hash`) and the copy the delta needs (user `Clone`).
pub(crate) struct StagedInsert<T> {
    hash: u64,
    delta: T,
    stored: T,
}

/// Phase 1 of a delete: the hash, and the value that becomes the delta.
pub(crate) struct StagedDelete<T> {
    hash: u64,
    value: T,
}

/// One staged change of an operator's batch.
pub(crate) enum Staged<T> {
    Insert(StagedInsert<T>),
    Delete(StagedDelete<T>),
}

struct StagedReplace<T> {
    remove: Option<StagedDelete<T>>,
    add: StagedInsert<T>,
}

impl<T: Hash + Clone> StagedInsert<T> {
    pub(crate) fn new(hasher: &RandomState, value: T) -> Self {
        value_code::run(|| Self {
            hash: hasher.hash_one(&value),
            delta: value.clone(),
            stored: value,
        })
    }
}

impl<T: Hash + Clone> StagedDelete<T> {
    fn cloned(hasher: &RandomState, value: &T) -> Self {
        value_code::run(|| Self {
            hash: hasher.hash_one(value),
            value: value.clone(),
        })
    }

    /// Like [`StagedDelete::cloned`], with the value owned, so it becomes
    /// the delta without a copy.
    pub(crate) fn owned(hasher: &RandomState, value: T) -> Self {
        Self {
            hash: value_code::run(|| hasher.hash_one(&value)),
            value,
        }
    }
}

impl<T: Hash + Clone> Staged<T> {
    pub(crate) fn delta(hasher: &RandomState, delta: Delta<T>) -> Self {
        match delta {
            Delta::Insert(value) => Self::Insert(StagedInsert::new(hasher, value)),
            Delta::Delete(value) => Self::Delete(StagedDelete::owned(hasher, value)),
        }
    }
}

pub(crate) struct Committed<T, const N: usize = 1> {
    /// Whether the remove side found its element.
    pub(crate) removed: bool,
    /// The new log version, if any delta was recorded.
    pub(crate) version: Option<u64>,
    pub(crate) displaced: Displaced<T, N>,
}

impl<T> Committed<T> {
    fn keep(self, keep: &mut Vec<Displaced<T>>) -> Option<u64> {
        if !self.displaced.is_empty() {
            keep.push(self.displaced);
        }
        self.version
    }
}

/// Elements and compacted deltas a mutation displaced. Dropping them
/// runs user `Drop`, so a caller drops them only after releasing the log
/// guard and setting the version node. A single insert or delete
/// displaces at most one element, a replace at most two.
#[must_use]
#[doc(hidden)]
pub struct Displaced<T, const N: usize = 1> {
    values: [Option<T>; N],
    compacted: Vec<Delta<T>>,
}

impl<T, const N: usize> Displaced<T, N> {
    fn is_empty(&self) -> bool {
        self.values.iter().all(Option::is_none) && self.compacted.is_empty()
    }

    /// Moves everything displaced into `keep`, one element per entry, for
    /// a caller that drops it later.
    pub(crate) fn keep_into(self, keep: &mut Vec<Displaced<T>>) {
        let mut compacted = Some(self.compacted);
        for value in self.values {
            let one = Displaced {
                values: [value],
                compacted: compacted.take().unwrap_or_default(),
            };
            if !one.is_empty() {
                keep.push(one);
            }
        }
    }

    /// Drops everything displaced, at a point the type's rule allows, and
    /// gives a compaction's emptied buffer back to `log` for its next
    /// compaction.
    #[inline(always)]
    fn release<C: Cells>(self, log: &OpLock<CollectionLog<T>, C>)
    where
        T: Hash + Eq + Clone,
    {
        if self.compacted.capacity() == 0 {
            drop(self);
        } else {
            release_each(log, [self]);
        }
    }

    fn into_buffer(self) -> Vec<Delta<T>> {
        let Self {
            values,
            mut compacted,
        } = self;
        drop(values);
        compacted.clear();
        compacted
    }
}

/// [`Displaced::release`] for an operator's batch of changes to `log`.
#[inline(always)]
pub(crate) fn release_kept<T: Hash + Eq + Clone, C: Cells>(
    log: &OpLock<CollectionLog<T>, C>,
    keep: Vec<Displaced<T>>,
) {
    if !keep.is_empty() {
        release_each(log, keep);
    }
}

/// Keeps the largest buffer when one batch compacted more than once.
#[inline(never)]
fn release_each<T: Hash + Eq + Clone, C: Cells, const N: usize>(
    log: &OpLock<CollectionLog<T>, C>,
    displaced: impl IntoIterator<Item = Displaced<T, N>>,
) {
    let mut buffer = Vec::new();
    for d in displaced {
        let emptied = d.into_buffer();
        if emptied.capacity() > buffer.capacity() {
            buffer = emptied;
        }
    }
    if buffer.capacity() != 0 {
        log.write().recycle(buffer);
    }
}

/// Set when a user panic stopped an operator part-way through a batch,
/// leaving its state and output logs partly updated. The operator's next
/// run rebuilds from upstream snapshots instead of applying deltas, and
/// clears the flag only once the rebuild has fully converged.
///
/// Relaxed is enough because only the operator's own compute touches the
/// flag, and the claim protocol orders one compute before the next.
pub(crate) struct NeedsRebuild(AtomicBool);

impl NeedsRebuild {
    pub(crate) fn new() -> Self {
        Self(AtomicBool::new(false))
    }

    #[inline]
    pub(crate) fn is_set(&self) -> bool {
        self.0.load(MemOrdering::Relaxed)
    }

    #[inline]
    pub(crate) fn clear(&self) {
        self.0.store(false, MemOrdering::Relaxed);
    }

    /// Whether this run rebuilds: the flag is set, or the run is a known
    /// episode member's, which rebuilds against the current provisional
    /// values whatever the flag says (bug k's rebuild is idempotent).
    #[inline]
    pub(crate) fn due<C: Cells>(&self, rt: &Runtime<C>) -> bool {
        self.is_set() || rt.running_as_member()
    }

    /// Sets the flag when the run ends, on every exit, if the run read a
    /// provisional value or ran as an episode member: its state then rests
    /// on values the episode may still change or discard, so only an
    /// ordinary run outside any episode leaves the flag clear.
    #[inline]
    pub(crate) fn provisional<'a, C: Cells>(&'a self, rt: &'a Runtime<C>) -> Provisional<'a, C> {
        Provisional { flag: self, rt }
    }

    /// Runs the part of a batch that changes state, setting the flag if a
    /// user panic unwinds out of it. The panic keeps unwinding.
    #[inline(always)]
    pub(crate) fn guard<R>(&self, batch: impl FnOnce() -> R) -> R {
        let armed = Armed(self);
        let r = batch();
        std::mem::forget(armed);
        r
    }
}

pub(crate) struct Provisional<'a, C: Cells> {
    flag: &'a NeedsRebuild,
    rt: &'a Runtime<C>,
}

impl<C: Cells> Drop for Provisional<'_, C> {
    fn drop(&mut self) {
        if self.rt.running_as_member() || self.rt.read_provisional() {
            self.flag.0.store(true, MemOrdering::Relaxed);
        }
    }
}

/// Sets its flag when dropped, which happens only when a panic unwinds
/// through [`NeedsRebuild::guard`]: a completed batch forgets it. Cheaper
/// than `catch_unwind`, which moves the batch behind a call and its
/// result through memory.
struct Armed<'a>(&'a NeedsRebuild);

impl Drop for Armed<'_> {
    fn drop(&mut self) {
        self.0 .0.store(true, MemOrdering::Relaxed);
    }
}

/// Reads an upstream for a rebuild under one guard and releases it before
/// the rebuild runs any user function.
pub(crate) fn rebuild_snapshot<S: 'static, R, C: Cells>(
    rt: &Runtime<C>,
    upstream: &OpLock<S, C>,
    read: impl FnOnce(&S) -> R,
) -> R {
    let snapshot = read(&upstream.read());
    park(rt, ParkPoint::RebuildAfterSnapshot);
    snapshot
}

impl<T: Hash + Eq + Clone> Default for CollectionLog<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Public collection handle. Cheap to clone (Arc + Copy handle).
///
/// The `_confine` marker is `PhantomData<C::Ptr<()>>`: under `Local`
/// that is a `Cell<*mut ()>`, making the handle `!Send + !Sync` so it
/// can never carry the RefCell-backed operator locks across threads;
/// under `Shared` it is an `AtomicPtr<()>` and the handle stays
/// `Send + Sync`.
pub struct IncrCollection<T: Value + Hash + Eq, C: Cells> {
    pub(crate) log: Arc<OpLock<CollectionLog<T>, C>>,
    /// A copy of the log's hasher, so mutations hash with no lock held.
    hasher: RandomState,
    pub(crate) version_node: Incr<u64>,
    /// True for operator outputs (filter/map/join/...) and group_by
    /// sub-collections. Direct mutation of those would set a query node
    /// or bypass the routing operator, corrupting the graph.
    pub(crate) derived: bool,
    pub(crate) _confine: std::marker::PhantomData<C::Ptr<()>>,
}

impl<T: Value + Hash + Eq, C: Cells> Clone for IncrCollection<T, C> {
    fn clone(&self) -> Self {
        Self {
            log: Arc::clone(&self.log),
            hasher: self.hasher.clone(),
            version_node: self.version_node,
            derived: self.derived,
            _confine: std::marker::PhantomData,
        }
    }
}

impl<T: Value + Hash + Eq, C: Cells> IncrCollection<T, C> {
    pub(crate) fn new(rt: &Runtime<C>) -> Self {
        let log = CollectionLog::new();
        Self {
            hasher: log.hasher.clone(),
            log: Arc::new(OpLock::new(LockKind::CollectionLog, log)),
            version_node: rt.create_input(0_u64),
            derived: false,
            _confine: std::marker::PhantomData,
        }
    }

    pub(crate) fn derived_with(
        log: Arc<OpLock<CollectionLog<T>, C>>,
        version_node: Incr<u64>,
    ) -> Self {
        let hasher = log.read().hasher.clone();
        Self {
            log,
            hasher,
            version_node,
            derived: true,
            _confine: std::marker::PhantomData,
        }
    }

    /// Internal: create a sub-collection from inside a compute closure
    /// (used by `group_by`). Marked derived: only the routing operator
    /// may mutate it.
    pub(crate) fn new_in_compute(rt: &Runtime<C>) -> Self {
        let log = CollectionLog::new();
        Self {
            hasher: log.hasher.clone(),
            log: Arc::new(OpLock::new(LockKind::CollectionLog, log)),
            version_node: rt.create_input_unchecked(0_u64),
            derived: true,
            _confine: std::marker::PhantomData,
        }
    }

    /// Public accessor for the collection's version node. Useful when a
    /// user query wants to depend on the collection without going through
    /// an operator.
    pub fn version_node(&self) -> Incr<u64> {
        self.version_node
    }

    #[inline]
    fn check_mutable(&self) {
        assert!(
            !self.derived,
            "incr-core: insert/delete on a derived collection; derived collections are \
             maintained by their operator, mutate the source collection instead",
        );
    }

    /// Insert a value. Bumps the underlying log version and notifies
    /// downstream queries by setting `version_node`.
    ///
    /// Panics if this collection is the output of an operator.
    pub fn insert(&self, rt: &Runtime<C>, value: T) {
        self.insert_with(rt, value, &mut (), |d| d.release(&self.log));
    }

    /// [`Self::insert`] reporting its representative to `on_rep` and
    /// handing what it displaced to `dispose`.
    #[inline(always)]
    pub(crate) fn insert_with<R: RepSink<T>>(
        &self,
        rt: &Runtime<C>,
        value: T,
        on_rep: &mut R,
        dispose: impl FnOnce(Displaced<T>),
    ) {
        self.check_mutable();
        value_code::check();
        let staged = StagedInsert::new(&self.hasher, value);
        self.publish(rt, |log| log.insert_staged(staged, on_rep), dispose);
    }

    /// Delete one occurrence. No-op (no log delta, no version bump) if
    /// the value was not present. Returns whether a delete was recorded.
    ///
    /// Panics if this collection is the output of an operator.
    pub fn delete(&self, rt: &Runtime<C>, value: &T) -> bool {
        self.delete_with(rt, value, &mut (), |d| d.release(&self.log))
    }

    /// [`Self::delete`] as [`Self::insert_with`].
    #[inline(always)]
    pub(crate) fn delete_with<R: RepSink<T>>(
        &self,
        rt: &Runtime<C>,
        value: &T,
        on_rep: &mut R,
        dispose: impl FnOnce(Displaced<T>),
    ) -> bool {
        self.check_mutable();
        value_code::check();
        let staged = StagedDelete::cloned(&self.hasher, value);
        self.publish(rt, |log| log.delete_staged(staged, on_rep), dispose)
    }

    /// Replace `old` (if present) with `new` as a single atomic step:
    /// both deltas are pushed under one log guard and only one `rt.set`
    /// is issued, so a reader can never observe `old` as transiently
    /// absent between the delete and the insert the way two separate
    /// `delete` + `insert` calls would expose. Returns whether the
    /// delete side recorded a delta (same contract as `delete`).
    ///
    /// Panics if this collection is the output of an operator.
    pub fn replace(&self, rt: &Runtime<C>, old: Option<&T>, new: T) -> bool {
        self.replace_with(rt, old, new, &mut (), |d| d.release(&self.log))
    }

    /// [`Self::replace`] as [`Self::insert_with`].
    #[inline(always)]
    pub(crate) fn replace_with<R: RepSink<T>>(
        &self,
        rt: &Runtime<C>,
        old: Option<&T>,
        new: T,
        on_rep: &mut R,
        dispose: impl FnOnce(Displaced<T, 2>),
    ) -> bool {
        self.check_mutable();
        value_code::check();
        let staged = StagedReplace {
            remove: old.map(|old| StagedDelete::cloned(&self.hasher, old)),
            add: StagedInsert::new(&self.hasher, new),
        };
        self.publish(rt, |log| log.replace_staged(staged, on_rep), dispose)
    }

    /// Runs a staged mutation's lookups and changes under the log guard,
    /// then sets the version node, and only then hands what the change
    /// displaced to `dispose`. User `Hash` and `Clone` ran in staging, before the guard,
    /// and user `Eq` runs before the first change, so a panic in either
    /// leaves the collection as it was. From the first change to the
    /// version set no user code runs. Returns whether a remove side found
    /// its element.
    #[inline(always)]
    fn publish<const N: usize>(
        &self,
        rt: &Runtime<C>,
        change: impl FnOnce(&mut CollectionLog<T>) -> Committed<T, N>,
        dispose: impl FnOnce(Displaced<T, N>),
    ) -> bool {
        let committed = change(&mut self.log.write());
        if let Some(version) = committed.version {
            rt.set(self.version_node, version);
        }
        dispose(committed.displaced);
        committed.removed
    }

    /// Number of live elements (with multiset duplicates counted).
    pub fn snapshot_len(&self) -> usize {
        value_code::check();
        self.log.read().len()
    }

    /// Current elements, with multiset duplicates expanded. Forces
    /// `version_node` first, so an operator output (a `filter`/`map`
    /// result, say) is brought up to date with its upstream before the
    /// read, the same way `count`/`reduce` force it via their own query
    /// closures.
    ///
    /// This does *not* extend to a `group_by` sub-collection: its
    /// `version_node` is an input the operator sets directly, not a
    /// query `rt.get` can force, so `values()` on a group only reflects
    /// whatever the group last observed. Force `GroupedCollection`'s own
    /// `version_node()` first if a group's contents need to be current.
    pub fn values(&self, rt: &Runtime<C>) -> Vec<T> {
        value_code::check();
        rt.get(self.version_node);
        self.log.read().elements_vec()
    }

    /// A derived view of this collection: same underlying log, but
    /// `insert`/`delete`/`replace` panic on it the way an operator
    /// output's do. Lets a wrapper type hand out read/operator access
    /// to its source collection without letting the caller bypass an
    /// index the wrapper maintains alongside it.
    pub fn as_view(&self) -> Self {
        Self {
            log: Arc::clone(&self.log),
            hasher: self.hasher.clone(),
            version_node: self.version_node,
            derived: true,
            _confine: std::marker::PhantomData,
        }
    }
}

impl<C: Cells> Runtime<C> {
    /// Create a fresh empty collection in this runtime.
    pub fn create_collection<T: Value + Hash + Eq>(&self) -> IncrCollection<T, C> {
        IncrCollection::new(self)
    }
}

impl<T, C> IncrCollection<T, C>
where
    T: Value + Hash + Eq,
    C: Cells,
{
    /// Filter: keep elements for which `pred(&t)` is true. Returns a new
    /// derived collection containing the filtered subset, propagated
    /// incrementally (O(new deltas) per observation).
    pub fn filter<F>(&self, rt: &Runtime<C>, pred: F) -> IncrCollection<T, C>
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        let pred = lock_order::unlocked("pred", pred);
        self.per_element(rt, move |v| pred(&v).then_some(v))
    }

    /// Map: transform every element via `f`. Returns a new derived
    /// collection.
    pub fn map<U, F>(&self, rt: &Runtime<C>, f: F) -> IncrCollection<U, C>
    where
        U: Value + Hash + Eq,
        F: Fn(&T) -> U + Send + Sync + 'static,
    {
        let f = lock_order::unlocked("f", f);
        self.per_element(rt, move |v| Some(f(&v)))
    }

    /// An operator whose output holds `image(v)` for every element `v`
    /// with an image, so each upstream delta maps to at most one output
    /// delta. `image` runs with no lock held, and so do the output's
    /// hashes and copies.
    fn per_element<U, I>(&self, rt: &Runtime<C>, image: I) -> IncrCollection<U, C>
    where
        U: Value + Hash + Eq,
        I: Fn(T) -> Option<U> + Send + Sync + 'static,
    {
        value_code::check();
        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        let output = CollectionLog::new();
        let out_hasher = output.hasher.clone();
        let output_log: Arc<OpLock<CollectionLog<U>, C>> =
            Arc::new(OpLock::new(LockKind::CollectionLog, output));

        // Bootstrap: seed the output from the live snapshot; the cursor
        // starts past everything the snapshot covered.
        let (cursor, bootstrap) = upstream_log.write().register_consumer();
        {
            let seeded: Vec<U> = bootstrap.into_iter().filter_map(&image).collect();
            let mut out = output_log.write();
            for v in seeded {
                out.insert(v);
            }
        }

        let rebuild = NeedsRebuild::new();
        let output_log_for_query = Arc::clone(&output_log);
        let version_node = rt.create_operator_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);
            let mut keep = Vec::new();

            let _provisional = rebuild.provisional(rt);
            if rebuild.due(rt) {
                let (snapshot, end) = rebuild_snapshot(rt, &upstream_log, CollectionLog::snapshot);
                let target: Vec<U> = snapshot.into_iter().filter_map(&image).collect();
                let mut out = output_log_for_query.write();
                out.converge(target, &mut keep);
                cursor.store(end, MemOrdering::Release);
                rebuild.clear();
                return out.version;
            }

            let (from, pending) = {
                let up = upstream_log.read();
                let from = cursor.load(MemOrdering::Acquire);
                if from >= up.end() {
                    drop(up);
                    return output_log_for_query.read().version;
                }
                (from, up.pending_cloned(from))
            };
            let to = from + pending.len() as u64;

            let staged: Vec<Staged<U>> = pending
                .into_iter()
                .filter_map(|d| match d {
                    Delta::Insert(v) => image(v).map(Delta::Insert),
                    Delta::Delete(v) => image(v).map(Delta::Delete),
                })
                .map(|d| Staged::delta(&out_hasher, d))
                .collect();

            let mut out = output_log_for_query.write();
            rebuild.guard(|| {
                for change in staged {
                    out.apply_kept(change, &mut keep);
                }
            });
            cursor.store(to, MemOrdering::Release);
            let version = out.version;
            drop(out);
            release_kept(&output_log_for_query, keep);
            version
        });

        IncrCollection::derived_with(output_log, version_node)
    }
    /// Count: number of live elements as an `Incr<u64>`. Maintains a
    /// running tally incrementally from upstream deltas; O(new deltas)
    /// per get rather than O(N) sum over the multiset.
    pub fn count(&self, rt: &Runtime<C>) -> Incr<u64> {
        use std::sync::atomic::AtomicI64;

        value_code::check();
        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        let (cursor, bootstrap) = upstream_log.write().register_consumer();
        // Signed running count so a stray delete-of-absent cannot
        // underflow; clamped to zero on read.
        let running = Arc::new(AtomicI64::new(bootstrap.len() as i64));

        let running_for_query = Arc::clone(&running);
        rt.create_operator_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);
            let up = upstream_log.read();
            let from = cursor.load(MemOrdering::Acquire);
            if from < up.end() {
                let mut delta = 0_i64;
                for d in up.pending_from(from) {
                    match d {
                        Delta::Insert(_) => delta += 1,
                        Delta::Delete(_) => delta -= 1,
                    }
                }
                running_for_query.fetch_add(delta, MemOrdering::Relaxed);
                cursor.store(up.end(), MemOrdering::Release);
            }
            running_for_query.load(MemOrdering::Relaxed).max(0) as u64
        })
    }

    /// Reduce: fold all live elements through `fold_fn`. The fold runs
    /// over a snapshot of the collection on every change: O(N) per
    /// change by construction, because an arbitrary fold has no inverse
    /// or associativity to exploit. For folds expressible as a monoid
    /// (sum, min, max, count, custom semigroups with identity), use
    /// [`Self::aggregate`], which is O(log N) per change.
    pub fn reduce<U, F>(&self, rt: &Runtime<C>, fold_fn: F) -> Incr<U>
    where
        U: Value,
        F: Fn(&[T]) -> U + Send + Sync + 'static,
    {
        value_code::check();
        let log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        rt.create_operator_query(move |rt| -> U {
            let _uv = rt.get(upstream_version);
            // Snapshot under the read guard, fold outside it: the user
            // fold must not run while the log is locked.
            let elements = log.read().elements_vec();
            fold_fn(&elements)
        })
    }

    /// Aggregate: incrementally maintained monoid fold. `lift` maps each
    /// element into the monoid, `combine` is the associative operation,
    /// `identity` its unit. Each insert or delete costs O(log N) combine
    /// calls against a balanced aggregation tree, instead of re-folding
    /// the whole collection.
    ///
    /// `combine` must be associative with `identity` as unit, or the
    /// result is unspecified. If `combine` or `lift` panics, the
    /// aggregate rebuilds itself from a snapshot on the next observation
    /// after recovery.
    pub fn aggregate<U, L, Cmb>(
        &self,
        rt: &Runtime<C>,
        identity: U,
        lift: L,
        combine: Cmb,
    ) -> Incr<U>
    where
        U: Value,
        L: Fn(&T) -> U + Send + Sync + 'static,
        Cmb: Fn(&U, &U) -> U + Send + Sync + 'static,
    {
        value_code::check();
        let lift = lock_order::unlocked("lift", lift);
        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        let (cursor, bootstrap) = upstream_log.write().register_consumer();
        let state = AggState::build(cloned(&identity), paired_with(&lift, bootstrap), &combine);
        let state: Arc<OpLock<AggState<T, U>, C>> =
            Arc::new(OpLock::new(LockKind::AggregateState, state));

        let rebuild = NeedsRebuild::new();
        let state_for_query = Arc::clone(&state);
        rt.create_operator_query(move |rt| -> U {
            let _uv = rt.get(upstream_version);

            let _provisional = rebuild.provisional(rt);
            if rebuild.due(rt) {
                let (snapshot, end) = rebuild_snapshot(rt, &upstream_log, CollectionLog::snapshot);
                let fresh =
                    AggState::build(cloned(&identity), paired_with(&lift, snapshot), &combine);
                let root = fresh.cloned_root();
                let stale = std::mem::replace(&mut *state_for_query.write(), fresh);
                cursor.store(end, MemOrdering::Release);
                rebuild.clear();
                drop(stale);
                return root;
            }

            let (from, pending) = {
                let up = upstream_log.read();
                let from = cursor.load(MemOrdering::Acquire);
                if from >= up.end() {
                    drop(up);
                    return state_for_query.read().cloned_root();
                }
                (from, up.pending_cloned(from))
            };
            let to = from + pending.len() as u64;

            // `lift` runs with no lock held. `combine` runs under the state
            // lock, which no compute but this one ever takes.
            let staged: Vec<AggChange<T, U>> = pending
                .into_iter()
                .map(|d| match d {
                    Delta::Insert(v) => {
                        let lifted = lift(&v);
                        AggChange::Insert(v, lifted)
                    }
                    Delta::Delete(v) => AggChange::Delete(v),
                })
                .collect();

            let mut st = state_for_query.write();
            rebuild.guard(|| {
                for change in staged {
                    match change {
                        AggChange::Insert(v, lifted) => st.insert(v, lifted, &combine),
                        AggChange::Delete(v) => st.delete(&v, &combine),
                    }
                }
            });
            cursor.store(to, MemOrdering::Release);
            st.cloned_root()
        })
    }

    /// Join with another collection on a shared key. Emits the
    /// cross-product of matching elements as `(T, U)` pairs, maintained
    /// incrementally from both sides' deltas.
    pub fn join<U, K, FL, FR>(
        &self,
        rt: &Runtime<C>,
        right: &IncrCollection<U, C>,
        left_key: FL,
        right_key: FR,
    ) -> IncrCollection<(T, U), C>
    where
        U: Value + Hash + Eq,
        K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
        FL: Fn(&T) -> K + Send + Sync + 'static,
        FR: Fn(&U) -> K + Send + Sync + 'static,
    {
        value_code::check();
        let left_key = lock_order::unlocked("left_key", left_key);
        let right_key = lock_order::unlocked("right_key", right_key);
        let left_log = Arc::clone(&self.log);
        let right_log = Arc::clone(&right.log);
        let left_version = self.version_node;
        let right_version = right.version_node;

        let (left_cursor, left_boot) = left_log.write().register_consumer();
        let (right_cursor, right_boot) = right_log.write().register_consumer();

        // Key extraction (user code) happens before the output lock.
        let (st, pairs) = JoinState::build(
            paired_with(&left_key, left_boot),
            paired_with(&right_key, right_boot),
        );
        let output_log: Arc<OpLock<CollectionLog<(T, U)>, C>> =
            Arc::new(OpLock::new(LockKind::CollectionLog, CollectionLog::new()));
        {
            let mut out = output_log.write();
            for pair in pairs {
                out.insert(pair);
            }
        }
        let state: Arc<OpLock<JoinState<T, U, K>, C>> =
            Arc::new(OpLock::new(LockKind::OperatorState, st));

        let rebuild = NeedsRebuild::new();
        let state_for_query = Arc::clone(&state);
        let output_log_for_query = Arc::clone(&output_log);
        let version_node = rt.create_operator_query(move |rt| -> u64 {
            let _lv = rt.get(left_version);
            let _rv = rt.get(right_version);
            let mut keep = Vec::new();

            let _provisional = rebuild.provisional(rt);
            if rebuild.due(rt) {
                let (l_snapshot, l_end) = rebuild_snapshot(rt, &left_log, CollectionLog::snapshot);
                let (r_snapshot, r_end) = rebuild_snapshot(rt, &right_log, CollectionLog::snapshot);
                let (fresh, pairs) = JoinState::build(
                    paired_with(&left_key, l_snapshot),
                    paired_with(&right_key, r_snapshot),
                );
                let mut st = state_for_query.write();
                let mut out = output_log_for_query.write();
                out.converge(pairs, &mut keep);
                let stale = std::mem::replace(&mut *st, fresh);
                left_cursor.store(l_end, MemOrdering::Release);
                right_cursor.store(r_end, MemOrdering::Release);
                rebuild.clear();
                let version = out.version;
                drop(out);
                drop(st);
                drop(stale);
                return version;
            }

            let (l_from, l_pending) = {
                let l = left_log.read();
                let from = left_cursor.load(MemOrdering::Acquire);
                let p = if from >= l.end() {
                    Vec::new()
                } else {
                    l.pending_cloned(from)
                };
                (from, p)
            };
            let (r_from, r_pending) = {
                let r = right_log.read();
                let from = right_cursor.load(MemOrdering::Acquire);
                let p = if from >= r.end() {
                    Vec::new()
                } else {
                    r.pending_cloned(from)
                };
                (from, p)
            };

            if l_pending.is_empty() && r_pending.is_empty() {
                return output_log_for_query.read().version;
            }
            let l_to = l_from + l_pending.len() as u64;
            let r_to = r_from + r_pending.len() as u64;

            // Stage: key extraction (user code) with no lock held. Deletes
            // use the key their insert was filed under instead.
            let l_keyed = keyed_inserts(&left_key, l_pending);
            let r_keyed = keyed_inserts(&right_key, r_pending);

            let mut st = state_for_query.write();
            let mut out = output_log_for_query.write();
            rebuild.guard(|| {
                let JoinState { left, right } = &mut *st;
                for change in l_keyed {
                    match change {
                        Keyed::Insert(v, k) => left.insert(v, k, right, |l, r| {
                            out.insert_kept((l.clone(), r.clone()), &mut keep);
                        }),
                        Keyed::Delete(v) => left.delete(&v, right, |l, r| {
                            out.delete_kept((l.clone(), r.clone()), &mut keep);
                        }),
                    }
                }
                for change in r_keyed {
                    match change {
                        Keyed::Insert(u, k) => right.insert(u, k, left, |r, l| {
                            out.insert_kept((l.clone(), r.clone()), &mut keep);
                        }),
                        Keyed::Delete(u) => right.delete(&u, left, |r, l| {
                            out.delete_kept((l.clone(), r.clone()), &mut keep);
                        }),
                    }
                }
            });
            left_cursor.store(l_to, MemOrdering::Release);
            right_cursor.store(r_to, MemOrdering::Release);
            let version = out.version;
            drop(out);
            drop(st);
            release_kept(&output_log_for_query, keep);
            version
        });

        IncrCollection::derived_with(output_log, version_node)
    }

    /// Group by an extracted key. Returns a `GroupedCollection<K, T, C>`
    /// holding one derived [`IncrCollection<T, C>`] per encountered key,
    /// each populated incrementally as upstream deltas arrive.
    pub fn group_by<K, F>(&self, rt: &Runtime<C>, key_fn: F) -> GroupedCollection<K, T, C>
    where
        K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
        F: Fn(&T) -> K + Send + Sync + 'static,
    {
        value_code::check();
        let key_fn = lock_order::unlocked("key_fn", key_fn);
        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        let (cursor, bootstrap) = upstream_log.write().register_consumer();

        let groups: GroupMap<K, T, C> =
            Arc::new(OpLock::new(LockKind::OperatorState, HashMap::new()));
        let key_cache: Arc<OpLock<KeyCache<T, K>, C>> =
            Arc::new(OpLock::new(LockKind::KeyCache, KeyCache::new()));
        let output_version_counter = Arc::new(AtomicU64::new(0));

        {
            // Bootstrap groups for pre-existing elements. Key extraction
            // (user code) happens before the group locks are taken.
            let keyed: Vec<(&T, K)> = bootstrap.iter().map(|v| (v, key_fn(v))).collect();
            let mut grps = groups.write();
            let mut kc = key_cache.write();
            for (v, k) in keyed {
                let k = kc.insert(cloned(v), k);
                let group = value_code::run(|| grps.entry(k))
                    .or_insert_with(|| IncrCollection::<T, C>::new_in_compute(rt));
                let new_ver = group.log.write().insert(cloned(v));
                rt.set(group.version_node, new_ver);
            }
        }

        let groups_for_query = Arc::clone(&groups);
        let key_cache_for_query = Arc::clone(&key_cache);
        let output_version_counter_for_query = Arc::clone(&output_version_counter);
        // The routing closure adopts the groups it creates, but it has to
        // exist before the router it belongs to. Filled right after
        // create_operator_query, and a query never runs before its first read.
        let router: Arc<OnceLock<ErasedIncr>> = Arc::new(OnceLock::new());
        let router_for_query = Arc::clone(&router);

        let rebuild = NeedsRebuild::new();
        let version_node = rt.create_operator_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);
            // A router's groups are inputs, which never hold a provisional
            // value, so it refuses one with the cycle before routing.
            rt.refuse_provisional();
            let new_group = || {
                let group = IncrCollection::<T, C>::new_in_compute(rt);
                let router = *router_for_query
                    .get()
                    .expect("group_by router ran before its handle was recorded");
                rt.adopt(router, group.version_node.erase());
                group
            };
            let mut keep = Vec::new();

            let _provisional = rebuild.provisional(rt);
            if rebuild.due(rt) {
                let (snapshot, end) = rebuild_snapshot(rt, &upstream_log, CollectionLog::snapshot);
                let mut fresh = KeyCache::new();
                let mut targets: HashMap<K, Vec<T>> = HashMap::new();
                for (v, k) in paired_with(&key_fn, snapshot) {
                    let k = fresh.insert(cloned(&v), k);
                    value_code::run(|| targets.entry(k).or_default()).push(v);
                }
                rt.refuse_provisional();

                let mut grps = groups_for_query.write();
                let mut kc = key_cache_for_query.write();
                value_code::run(|| {
                    for k in grps.keys() {
                        if !targets.contains_key(k) {
                            targets.insert(k.clone(), Vec::new());
                        }
                    }
                });
                for (k, target) in targets {
                    let group = value_code::run(|| grps.entry(k)).or_insert_with(new_group);
                    let version = group.log.write().converge(target, &mut keep);
                    if let Some(version) = version {
                        rt.set(group.version_node, version);
                    }
                }
                let stale = std::mem::replace(&mut *kc, fresh);
                cursor.store(end, MemOrdering::Release);
                rebuild.clear();
                drop(kc);
                drop(grps);
                drop(stale);
                return output_version_counter_for_query.fetch_add(1, MemOrdering::Relaxed) + 1;
            }

            let (from, pending) = {
                let up = upstream_log.read();
                let from = cursor.load(MemOrdering::Acquire);
                if from >= up.end() {
                    drop(up);
                    return output_version_counter_for_query.load(MemOrdering::Relaxed);
                }
                (from, up.pending_cloned(from))
            };
            let to = from + pending.len() as u64;

            // Stage: key extraction outside the locks. Deletes look up
            // the cached key during apply instead.
            let keyed = keyed_inserts(&key_fn, pending);
            rt.refuse_provisional();

            let mut grps = groups_for_query.write();
            let mut kc = key_cache_for_query.write();
            rebuild.guard(|| {
                // Bucket by destination group first, mutating key_cache in
                // log order as we go (a Delete's group comes only from the
                // cache, so a same-batch Insert-then-Delete of the same
                // value must see the Insert's cache write before the
                // Delete looks it up). Then apply each touched group's
                // deltas under one log guard and issue one `rt.set`, so a
                // same-group replace (Delete old + Insert new landing in
                // the same group) can't expose the group's count as
                // transiently short by one the way per-delta `rt.set` did.
                let mut by_group: HashMap<K, Vec<Delta<T>>> = HashMap::new();
                for change in keyed {
                    match change {
                        Keyed::Insert(v, k) => {
                            let k = kc.insert(cloned(&v), k);
                            value_code::run(|| by_group.entry(k).or_default())
                                .push(Delta::Insert(v));
                        }
                        Keyed::Delete(v) => {
                            if let Some(k) = kc.remove(&v) {
                                value_code::run(|| by_group.entry(k).or_default())
                                    .push(Delta::Delete(v));
                            }
                        }
                    }
                }

                for (k, deltas) in by_group {
                    let group = value_code::run(|| grps.entry(k)).or_insert_with(new_group);
                    let mut log = group.log.write();
                    let mut last_version = None;
                    for d in deltas {
                        let version = match d {
                            Delta::Insert(v) => log.insert_kept(v, &mut keep),
                            Delta::Delete(v) => log.delete_kept(v, &mut keep),
                        };
                        last_version = version.or(last_version);
                    }
                    drop(log);
                    if let Some(ver) = last_version {
                        rt.set(group.version_node, ver);
                    }
                }
            });
            cursor.store(to, MemOrdering::Release);
            drop(kc);
            drop(grps);
            output_version_counter_for_query.fetch_add(1, MemOrdering::Relaxed) + 1
        });
        router
            .set(version_node.erase())
            .expect("group_by router recorded twice");
        for group in groups.read().values() {
            rt.adopt(version_node.erase(), group.version_node.erase());
        }

        GroupedCollection {
            groups,
            version_node,
            #[cfg(test)]
            key_cache,
            _confine: std::marker::PhantomData,
        }
    }
}

/// One side of a join: its elements by key, and the key each copy was
/// filed under.
struct JoinSide<A, K> {
    index: HashMap<K, Vec<A>>,
    keys: KeyCache<A, K>,
}

impl<A: Hash + Eq + Clone, K: Hash + Eq + Clone> JoinSide<A, K> {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
            keys: KeyCache::new(),
        }
    }

    /// Files `value` under `key` and passes it with every element of
    /// `other` filed under the same key to `matched`.
    fn insert<B>(
        &mut self,
        value: A,
        key: K,
        other: &JoinSide<B, K>,
        mut matched: impl FnMut(&A, &B),
    ) {
        let _guard = value_code::enter();
        let key = self.keys.insert(value.clone(), key);
        if let Some(matches) = other.index.get(&key) {
            for m in matches {
                matched(&value, m);
            }
        }
        self.index.entry(key).or_default().push(value);
    }

    /// Removes one copy of `value`, if present, and passes it with every
    /// element of `other` filed under its key to `matched`.
    fn delete<B>(&mut self, value: &A, other: &JoinSide<B, K>, mut matched: impl FnMut(&A, &B)) {
        let guard = value_code::enter();
        let Some(key) = self.keys.remove(value) else {
            return;
        };
        let mut removed = None;
        if let Some(bucket) = self.index.get_mut(&key) {
            if let Some(pos) = bucket.iter().position(|x| x == value) {
                removed = Some(bucket.remove(pos));
            }
            if bucket.is_empty() {
                self.index.remove(&key);
            }
        }
        if let Some(matches) = other.index.get(&key) {
            for m in matches {
                matched(value, m);
            }
        }
        drop(guard);
        drop(removed);
    }
}

struct JoinState<T, U, K> {
    left: JoinSide<T, K>,
    right: JoinSide<U, K>,
}

impl<T, U, K> JoinState<T, U, K>
where
    T: Hash + Eq + Clone,
    U: Hash + Eq + Clone,
    K: Hash + Eq + Clone,
{
    /// The state over two keyed sides, and the pairs it joins.
    fn build(left: Vec<(T, K)>, right: Vec<(U, K)>) -> (Self, Vec<(T, U)>) {
        let mut st = Self {
            left: JoinSide::new(),
            right: JoinSide::new(),
        };
        let mut pairs = Vec::new();
        for (l, k) in left {
            st.left.insert(l, k, &st.right, |_, _| {});
        }
        for (r, k) in right {
            st.right
                .insert(r, k, &st.left, |r, l| pairs.push((l.clone(), r.clone())));
        }
        (st, pairs)
    }
}

/// A delta staged with its key: inserts carry the key the user function
/// gave, deletes take the one their insert was filed under.
enum Keyed<A, K> {
    Insert(A, K),
    Delete(A),
}

fn keyed_inserts<A, K>(key: &impl Fn(&A) -> K, deltas: Vec<Delta<A>>) -> Vec<Keyed<A, K>> {
    deltas
        .into_iter()
        .map(|d| match d {
            Delta::Insert(v) => {
                let k = key(&v);
                Keyed::Insert(v, k)
            }
            Delta::Delete(v) => Keyed::Delete(v),
        })
        .collect()
}

/// The key each live element was filed under, counted per copy, so every
/// copy's delete reaches the bucket its insert used even when the key
/// function is expensive or non-deterministic across calls.
struct KeyCache<T, K>(HashMap<T, (K, usize)>);

impl<T: Hash + Eq, K: Clone> KeyCache<T, K> {
    fn new() -> Self {
        Self(HashMap::new())
    }

    /// Records one copy of `value` and returns the key to file it under:
    /// `key` for its first copy, and the first copy's key after that.
    fn insert(&mut self, value: T, key: K) -> K {
        value_code::run(|| match self.0.entry(value) {
            Entry::Occupied(mut e) => {
                e.get_mut().1 += 1;
                e.get().0.clone()
            }
            Entry::Vacant(e) => e.insert((key, 1)).0.clone(),
        })
    }

    fn remove(&mut self, value: &T) -> Option<K> {
        value_code::run(|| {
            let (key, copies) = self.0.get_mut(value)?;
            if *copies > 1 {
                *copies -= 1;
                return Some(key.clone());
            }
            self.0.remove(value).map(|(key, _)| key)
        })
    }
}

/// Balanced aggregation tree over a dynamic multiset: leaves hold lifted
/// values (or the identity for free slots), internal nodes hold the
/// combine of their children. Insert/delete touch one leaf and bubble
/// O(log capacity) combines to the root.
struct AggTree<U> {
    nodes: Vec<U>,
    cap: usize,
    identity: U,
}

impl<U: Clone> AggTree<U> {
    fn new(identity: U) -> Self {
        Self {
            nodes: value_code::run(|| vec![identity.clone(); 2]),
            cap: 1,
            identity,
        }
    }

    fn cloned_root(&self) -> U {
        cloned(&self.nodes[1])
    }

    fn set_leaf(&mut self, slot: usize, value: U, combine: &impl Fn(&U, &U) -> U) {
        let mut i = self.cap + slot;
        self.nodes[i] = value;
        while i > 1 {
            i /= 2;
            self.nodes[i] = combine(&self.nodes[2 * i], &self.nodes[2 * i + 1]);
        }
    }

    fn grow(&mut self, combine: &impl Fn(&U, &U) -> U) {
        let new_cap = self.cap * 2;
        let mut nodes = value_code::run(|| {
            let mut nodes = vec![self.identity.clone(); 2 * new_cap];
            nodes[new_cap..new_cap + self.cap]
                .clone_from_slice(&self.nodes[self.cap..2 * self.cap]);
            nodes
        });
        for i in (1..new_cap).rev() {
            nodes[i] = combine(&nodes[2 * i], &nodes[2 * i + 1]);
        }
        self.nodes = nodes;
        self.cap = new_cap;
    }
}

struct AggState<T: Hash + Eq + Clone, U> {
    tree: AggTree<U>,
    slots: HashMap<T, Vec<usize>>,
    free: Vec<usize>,
    high_water: usize,
}

enum AggChange<T, U> {
    Insert(T, U),
    Delete(T),
}

/// A user `Clone` made by the engine, inside the value-code counter.
pub(crate) fn cloned<T: Clone>(value: &T) -> T {
    value_code::run(|| value.clone())
}

/// Pairs each value with `f` of it, run with no lock held.
pub(crate) fn paired_with<T, U>(f: &impl Fn(&T) -> U, values: Vec<T>) -> Vec<(T, U)> {
    values
        .into_iter()
        .map(|v| {
            let image = f(&v);
            (v, image)
        })
        .collect()
}

impl<T: Hash + Eq + Clone, U: Clone> AggState<T, U> {
    fn new(identity: U) -> Self {
        Self {
            tree: AggTree::new(identity),
            slots: HashMap::new(),
            free: Vec::new(),
            high_water: 0,
        }
    }

    fn build(identity: U, lifted: Vec<(T, U)>, combine: &impl Fn(&U, &U) -> U) -> Self {
        let mut state = Self::new(identity);
        for (value, lifted) in lifted {
            state.insert(value, lifted, combine);
        }
        state
    }

    fn cloned_root(&self) -> U {
        self.tree.cloned_root()
    }

    fn insert(&mut self, value: T, lifted: U, combine: &impl Fn(&U, &U) -> U) {
        let slot = self.free.pop().unwrap_or_else(|| {
            if self.high_water == self.tree.cap {
                self.tree.grow(combine);
            }
            let s = self.high_water;
            self.high_water += 1;
            s
        });
        value_code::run(|| self.slots.entry(value).or_default()).push(slot);
        self.tree.set_leaf(slot, lifted, combine);
    }

    fn delete(&mut self, value: &T, combine: &impl Fn(&U, &U) -> U) {
        let guard = value_code::enter();
        let Some(bucket) = self.slots.get_mut(value) else {
            return;
        };
        let Some(slot) = bucket.pop() else {
            return;
        };
        let emptied = bucket.is_empty().then(|| self.slots.remove_entry(value));
        let identity = self.tree.identity.clone();
        drop(guard);
        drop(emptied);
        self.tree.set_leaf(slot, identity, combine);
        self.free.push(slot);
    }
}

/// Collection partitioned by key. Each key maps to a derived
/// [`IncrCollection<T, C>`] containing only that key's elements.
///
/// `version_node` bumps whenever any group changes. To depend on a
/// specific group, use `get_group(&k)` and depend on that
/// sub-collection's version_node directly.
pub struct GroupedCollection<K, T, C>
where
    K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
    T: Value + Hash + Eq,
    C: Cells,
{
    pub(crate) groups: GroupMap<K, T, C>,
    pub(crate) version_node: Incr<u64>,
    #[cfg(test)]
    key_cache: Arc<OpLock<KeyCache<T, K>, C>>,
    pub(crate) _confine: std::marker::PhantomData<C::Ptr<()>>,
}

/// Shared map from group key to that key's derived sub-collection,
/// shared between the group_by operator closure and the handle.
type GroupMap<K, T, C> = Arc<OpLock<HashMap<K, IncrCollection<T, C>>, C>>;

impl<K, T, C> Clone for GroupedCollection<K, T, C>
where
    K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
    T: Value + Hash + Eq,
    C: Cells,
{
    fn clone(&self) -> Self {
        Self {
            groups: Arc::clone(&self.groups),
            version_node: self.version_node,
            #[cfg(test)]
            key_cache: Arc::clone(&self.key_cache),
            _confine: std::marker::PhantomData,
        }
    }
}

impl<K, T, C> GroupedCollection<K, T, C>
where
    K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
    T: Value + Hash + Eq,
    C: Cells,
{
    pub fn version_node(&self) -> Incr<u64> {
        self.version_node
    }

    pub fn keys(&self) -> Vec<K> {
        value_code::check();
        let groups = self.groups.read();
        value_code::run(|| groups.keys().cloned().collect())
    }

    pub fn get_group(&self, key: &K) -> Option<IncrCollection<T, C>> {
        value_code::check();
        let groups = self.groups.read();
        value_code::run(|| groups.get(key)).cloned()
    }

    pub fn group_count(&self) -> usize {
        value_code::check();
        self.groups.read().len()
    }

    /// Everything a routing run can change: each group's elements, log
    /// end, log version and version node value, and the key cache.
    #[cfg(test)]
    pub(crate) fn routed_state(&self, rt: &Runtime<C>) -> RoutedState<K, T> {
        let groups = self.groups.read();
        let groups = groups
            .iter()
            .map(|(k, g)| {
                let log = g.log.read();
                let (elements, end, version) = (log.elements_vec(), log.end(), log.version);
                drop(log);
                (k.clone(), (elements, end, version, rt.get(g.version_node)))
            })
            .collect();
        let key_cache = self.key_cache.read().0.clone();
        RoutedState { groups, key_cache }
    }
}

/// A group's elements, log end, log version and version node value.
#[cfg(test)]
type GroupState<T> = (Vec<T>, u64, u64, u64);

#[cfg(test)]
#[derive(Debug, PartialEq)]
pub(crate) struct RoutedState<K: Hash + Eq, T: Hash + Eq> {
    groups: HashMap<K, GroupState<T>>,
    key_cache: HashMap<T, (K, usize)>,
}

#[cfg(test)]
#[path = "collection_rebuild_tests.rs"]
mod rebuild_tests;

#[cfg(test)]
#[path = "collection_cycle_tests.rs"]
mod cycle_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};
    use crate::runtime::DeleteError;

    #[test]
    fn local_collection_basic_insert() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        c.insert(&rt, 10);
        c.insert(&rt, 20);
        c.insert(&rt, 30);
        assert_eq!(c.snapshot_len(), 3);
    }

    #[test]
    fn shared_collection_basic_insert() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        c.insert(&rt, 10);
        c.insert(&rt, 20);
        c.insert(&rt, 30);
        assert_eq!(c.snapshot_len(), 3);
    }

    #[test]
    fn shared_replace_basic() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        c.insert(&rt, 1);
        let n = c.count(&rt);
        assert_eq!(rt.get(n), 1);

        let deleted = c.replace(&rt, Some(&1), 2);
        assert!(deleted, "replace should report the old value was found");
        assert_eq!(rt.get(n), 1);
        assert_eq!(c.snapshot_len(), 1);

        let deleted = c.replace(&rt, Some(&99), 3);
        assert!(!deleted, "replace should report a missing old value");
        assert_eq!(rt.get(n), 2);
    }

    #[test]
    fn shared_values_reads_current_elements_including_through_a_derived_view() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        c.insert(&rt, 1);
        c.insert(&rt, 2);
        c.insert(&rt, 3);

        let evens = c.filter(&rt, |v| v % 2 == 0);
        let mut got = evens.values(&rt);
        got.sort_unstable();
        assert_eq!(got, vec![2]);

        c.insert(&rt, 4);
        let mut got = evens.values(&rt);
        got.sort_unstable();
        assert_eq!(got, vec![2, 4]);
    }

    #[test]
    fn shared_as_view_reads_through() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        c.insert(&rt, 1);
        c.insert(&rt, 2);
        let view = c.as_view();
        assert_eq!(view.snapshot_len(), 2);
        let n = view.count(&rt);
        assert_eq!(rt.get(n), 2);
        c.insert(&rt, 3);
        assert_eq!(rt.get(n), 3);
    }

    #[test]
    #[should_panic(expected = "derived collection")]
    fn shared_as_view_panics_on_insert() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let view = c.as_view();
        view.insert(&rt, 1);
    }

    #[test]
    fn shared_replace_never_transiently_drops_count() {
        let rt: Arc<Runtime<Shared>> = Arc::new(Runtime::new());
        let c = rt.create_collection::<i64>();
        c.insert(&rt, 0);
        let n = c.count(&rt);
        assert_eq!(rt.get(n), 1);

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = {
            let rt = Arc::clone(&rt);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    assert_eq!(rt.get(n), 1, "replace exposed a transient count");
                }
            })
        };

        let iters = if cfg!(miri) { 20 } else { 2000 };
        let mut current = 0_i64;
        for _ in 0..iters {
            let next = current + 1;
            c.replace(&rt, Some(&current), next);
            current = next;
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        reader.join().unwrap();
        assert_eq!(rt.get(n), 1);
    }

    #[test]
    fn shared_replace_group_by_never_transiently_drops_group_count() {
        let rt: Arc<Runtime<Shared>> = Arc::new(Runtime::new());
        let c = rt.create_collection::<i64>();
        c.insert(&rt, 0);
        let groups = c.group_by(&rt, |x| x % 2);
        let _ = rt.get(groups.version_node);
        let evens = groups.get_group(&0).expect("group 0 missing");
        let n = evens.count(&rt);
        assert_eq!(rt.get(n), 1);

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = {
            let rt = Arc::clone(&rt);
            let stop = Arc::clone(&stop);
            let groups_version = groups.version_node;
            std::thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = rt.get(groups_version);
                    assert_eq!(
                        rt.get(n),
                        1,
                        "group_by replace exposed a transient group count"
                    );
                }
            })
        };

        // Same-key replace whose old and new value both hash to group 0
        // (even), so every replace is a same-group operation, the case
        // the batched-apply fix targets.
        let iters = if cfg!(miri) { 20 } else { 2000 };
        let mut current = 0_i64;
        for _ in 0..iters {
            let next = current + 2;
            c.replace(&rt, Some(&current), next);
            current = next;
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        reader.join().unwrap();
        let _ = rt.get(groups.version_node);
        assert_eq!(rt.get(n), 1);
    }

    #[test]
    fn local_count_basic() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let n = c.count(&rt);
        assert_eq!(rt.get(n), 0);
        c.insert(&rt, 5);
        c.insert(&rt, 7);
        assert_eq!(rt.get(n), 2);
        c.delete(&rt, &5);
        assert_eq!(rt.get(n), 1);
    }

    #[test]
    fn shared_count_basic() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let n = c.count(&rt);
        assert_eq!(rt.get(n), 0);
        c.insert(&rt, 5);
        c.insert(&rt, 7);
        assert_eq!(rt.get(n), 2);
        c.delete(&rt, &5);
        assert_eq!(rt.get(n), 1);
    }

    #[test]
    fn local_filter_count() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let evens = c.filter(&rt, |x| x % 2 == 0);
        let n_evens = evens.count(&rt);
        for i in 1..=10 {
            c.insert(&rt, i);
        }
        assert_eq!(rt.get(n_evens), 5); // 2, 4, 6, 8, 10
    }

    #[test]
    fn shared_filter_count() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let evens = c.filter(&rt, |x| x % 2 == 0);
        let n_evens = evens.count(&rt);
        for i in 1..=10 {
            c.insert(&rt, i);
        }
        assert_eq!(rt.get(n_evens), 5);
    }

    #[test]
    fn operator_attached_to_populated_collection_bootstraps() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        for i in 1..=10 {
            c.insert(&rt, i);
        }
        // Attach AFTER data exists: the operator must see the snapshot.
        let evens = c.filter(&rt, |x| x % 2 == 0);
        let n = evens.count(&rt);
        assert_eq!(rt.get(n), 5);
        c.insert(&rt, 12);
        assert_eq!(rt.get(n), 6);
    }

    #[test]
    fn local_map_then_reduce_sum() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let doubled = c.map(&rt, |x| x * 2);
        let total = doubled.reduce(&rt, |xs| xs.iter().sum::<i64>());
        for i in 1..=5 {
            c.insert(&rt, i);
        }
        // doubled = [2, 4, 6, 8, 10] → sum 30
        assert_eq!(rt.get(total), 30);
    }

    #[test]
    fn shared_filter_map_reduce_pipeline() {
        let rt: Runtime<Shared> = Runtime::new();
        let scores = rt.create_collection::<i64>();
        let passing = scores.filter(&rt, |s| *s >= 50);
        let curved = passing.map(&rt, |s| s + 10);
        let total = curved.reduce(&rt, |xs| xs.iter().sum::<i64>());
        scores.insert(&rt, 80);
        scores.insert(&rt, 95);
        scores.insert(&rt, 60);
        scores.insert(&rt, 42);
        // passing = [80, 95, 60] → curved = [90, 105, 70] → sum 265
        assert_eq!(rt.get(total), 265);
    }

    #[test]
    fn aggregate_sum_tracks_inserts_and_deletes() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let total = c.aggregate(&rt, 0_i64, |x| *x, |a, b| a + b);
        assert_eq!(rt.get(total), 0);
        for i in 1..=100 {
            c.insert(&rt, i);
        }
        assert_eq!(rt.get(total), 5050);
        c.delete(&rt, &100);
        c.delete(&rt, &1);
        assert_eq!(rt.get(total), 4949);
        c.insert(&rt, 1000);
        assert_eq!(rt.get(total), 5949);
    }

    #[test]
    fn aggregate_max_via_monoid() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let max = c.aggregate(&rt, i64::MIN, |x| *x, |a, b| *a.max(b));
        c.insert(&rt, 3);
        c.insert(&rt, 9);
        c.insert(&rt, 5);
        assert_eq!(rt.get(max), 9);
        // Non-invertible op: deleting the max must still produce the
        // correct new max (the tree recombines, nothing is "subtracted").
        c.delete(&rt, &9);
        assert_eq!(rt.get(max), 5);
    }

    #[test]
    fn aggregate_bootstraps_from_existing_elements() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        c.insert(&rt, 10);
        c.insert(&rt, 20);
        let total = c.aggregate(&rt, 0_i64, |x| *x, |a, b| a + b);
        assert_eq!(rt.get(total), 30);
    }

    #[test]
    #[should_panic(expected = "derived collection")]
    fn insert_on_derived_collection_panics() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let evens = c.filter(&rt, |x| x % 2 == 0);
        evens.insert(&rt, 2);
    }

    #[test]
    #[should_panic(expected = "derived collection")]
    fn insert_on_group_subcollection_panics() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let groups = c.group_by(&rt, |x| x % 2);
        c.insert(&rt, 2);
        let _ = rt.get(groups.version_node);
        let g = groups.get_group(&0).unwrap();
        g.insert(&rt, 4);
    }

    #[test]
    fn log_compacts_behind_consumers() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let n = c.count(&rt);
        for i in 0..(COMPACT_EVERY as i64 * 3) {
            c.insert(&rt, i);
            if i % 64 == 0 {
                // Keep the consumer caught up so compaction can run.
                let _ = rt.get(n);
            }
        }
        let _ = rt.get(n);
        let log = c.log.read();
        assert!(
            log.deltas.len() < COMPACT_EVERY * 2,
            "log retained {} deltas; compaction is not keeping up",
            log.deltas.len(),
        );
        assert_eq!(rt.get(n), COMPACT_EVERY as u64 * 3);
    }

    #[test]
    fn log_with_no_consumers_stays_bounded() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        for i in 0..(COMPACT_EVERY as i64 * 4) {
            c.insert(&rt, i);
        }
        let log = c.log.read();
        assert!(
            log.deltas.len() <= COMPACT_EVERY,
            "consumerless log retained {} deltas",
            log.deltas.len(),
        );
    }

    /// Steps until `log` compacts, three times, and checks that the buffer
    /// its deltas filled comes back as the spare the next compaction
    /// fills, so a compaction in steady state allocates nothing.
    fn assert_compaction_recycles<T: Hash + Eq + Clone>(
        log: &OpLock<CollectionLog<T>, Shared>,
        mut step: impl FnMut(),
    ) {
        for _ in 0..3 {
            let base = log.read().base;
            let mut filled = log.read().deltas.as_ptr();
            while log.read().base == base {
                filled = log.read().deltas.as_ptr();
                step();
            }
            let log = log.read();
            assert!(
                log.spare.capacity() >= COMPACT_EVERY,
                "no spare after a compaction"
            );
            assert_eq!(
                log.spare.as_ptr(),
                filled,
                "the filled buffer is not the spare"
            );
            assert!(log.spare.is_empty());
        }
    }

    #[test]
    fn source_compaction_recycles_its_buffer() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let mut next = 0;
        assert_compaction_recycles(&c.log, || {
            c.insert(&rt, next);
            next += 1;
        });
    }

    #[test]
    fn operator_compaction_recycles_its_buffer() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let doubled = c.map(&rt, |x| x * 2);
        let count = doubled.count(&rt);
        let mut next = 0;
        assert_compaction_recycles(&doubled.log, || {
            c.insert(&rt, next);
            next += 1;
            let _ = rt.get(count);
        });
    }

    /// A panicking predicate must not poison the pipeline or corrupt the
    /// cursor: after recovery the operator replays the same batch
    /// exactly once (the staged-but-never-applied first attempt must not
    /// duplicate effects).
    #[test]
    fn filter_predicate_panic_recovers_exactly_once() {
        use std::sync::atomic::AtomicBool;

        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let pill = Arc::new(AtomicBool::new(true));
        let pill_for_pred = Arc::clone(&pill);
        let filtered = c.filter(&rt, move |x| {
            assert!(
                !(pill_for_pred.load(MemOrdering::Relaxed) && *x == 13),
                "intentional test panic"
            );
            *x % 2 == 0
        });
        let n = filtered.count(&rt);
        c.insert(&rt, 2);
        assert_eq!(rt.get(n), 1);

        c.insert(&rt, 13);
        c.insert(&rt, 4);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(n))).is_err());

        // Disarm the pill. Failed is sticky until a dependency changes,
        // so push one more element; the whole pending batch then replays
        // exactly once: 13 is odd and filtered out, 4 and 6 land once.
        pill.store(false, MemOrdering::Relaxed);
        c.insert(&rt, 6);
        assert_eq!(rt.get(n), 3); // 2, 4, 6
    }

    #[test]
    fn local_incremental_insert_only_changes_count() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let n = c.count(&rt);
        for i in 0..100 {
            c.insert(&rt, i);
        }
        assert_eq!(rt.get(n), 100);
        c.insert(&rt, 999);
        assert_eq!(rt.get(n), 101);
    }

    #[test]
    fn local_group_by_partitions() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let groups = c.group_by(&rt, |x| x % 3);
        c.insert(&rt, 1);
        c.insert(&rt, 2);
        c.insert(&rt, 3);
        c.insert(&rt, 4);
        c.insert(&rt, 5);
        c.insert(&rt, 6);
        let _ = rt.get(groups.version_node);
        assert_eq!(groups.group_count(), 3);
        let mut ks = groups.keys();
        ks.sort();
        assert_eq!(ks, vec![0, 1, 2]);
        let g0 = groups.get_group(&0).expect("group 0 missing");
        let g1 = groups.get_group(&1).expect("group 1 missing");
        let g2 = groups.get_group(&2).expect("group 2 missing");
        assert_eq!(g0.snapshot_len(), 2); // 3, 6
        assert_eq!(g1.snapshot_len(), 2); // 1, 4
        assert_eq!(g2.snapshot_len(), 2); // 2, 5
    }

    #[test]
    fn shared_group_by_per_group_count() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let groups = c.group_by(&rt, |x| if *x >= 0 { "pos" } else { "neg" });
        c.insert(&rt, 1);
        c.insert(&rt, -1);
        c.insert(&rt, 2);
        c.insert(&rt, -2);
        c.insert(&rt, 3);
        let _ = rt.get(groups.version_node);
        let pos = groups.get_group(&"pos").expect("pos group missing");
        let neg = groups.get_group(&"neg").expect("neg group missing");
        let pos_count = pos.count(&rt);
        let neg_count = neg.count(&rt);
        assert_eq!(rt.get(pos_count), 3);
        assert_eq!(rt.get(neg_count), 2);
    }

    #[test]
    fn local_group_by_delete_removes_from_group() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let groups = c.group_by(&rt, |x| x % 2);
        c.insert(&rt, 2);
        c.insert(&rt, 4);
        c.insert(&rt, 6);
        let _ = rt.get(groups.version_node);
        let evens = groups.get_group(&0).expect("group 0 missing");
        assert_eq!(evens.snapshot_len(), 3);
        c.delete(&rt, &4);
        let _ = rt.get(groups.version_node);
        assert_eq!(evens.snapshot_len(), 2);
    }

    #[test]
    fn local_join_simple() {
        let rt: Runtime<Local> = Runtime::new();
        let users = rt.create_collection::<(i64, String)>(); // (id, name)
        let orders = rt.create_collection::<(i64, i64)>(); // (user_id, amount)
        let joined = users.join(&rt, &orders, |u| u.0, |o| o.0);
        users.insert(&rt, (1, "alice".to_string()));
        users.insert(&rt, (2, "bob".to_string()));
        orders.insert(&rt, (1, 100));
        orders.insert(&rt, (1, 200));
        orders.insert(&rt, (3, 50)); // no matching user
        let n = joined.count(&rt);
        // (alice, 100), (alice, 200) — 2 pairs
        assert_eq!(rt.get(n), 2);
    }

    #[test]
    fn shared_join_symmetric_order() {
        let rt: Runtime<Shared> = Runtime::new();
        let a = rt.create_collection::<(i32, &'static str)>();
        let b = rt.create_collection::<(i32, i32)>();
        let j = a.join(&rt, &b, |x| x.0, |y| y.0);
        // Insert b first, then a; pairs should still emit.
        b.insert(&rt, (1, 100));
        a.insert(&rt, (1, "x"));
        a.insert(&rt, (1, "y"));
        b.insert(&rt, (1, 200));
        let n = j.count(&rt);
        // pairs: (x,100), (y,100), (x,200), (y,200) — 4
        assert_eq!(rt.get(n), 4);
    }

    #[test]
    fn join_bootstraps_from_populated_sides() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_collection::<(i32, i32)>();
        let b = rt.create_collection::<(i32, i32)>();
        a.insert(&rt, (1, 10));
        b.insert(&rt, (1, 100));
        let j = a.join(&rt, &b, |x| x.0, |y| y.0);
        let n = j.count(&rt);
        assert_eq!(rt.get(n), 1);
        b.insert(&rt, (1, 200));
        assert_eq!(rt.get(n), 2);
    }

    #[test]
    fn local_join_delete_removes_pairs() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_collection::<(i32, i32)>();
        let b = rt.create_collection::<(i32, i32)>();
        let j = a.join(&rt, &b, |x| x.0, |y| y.0);
        a.insert(&rt, (1, 10));
        b.insert(&rt, (1, 100));
        b.insert(&rt, (1, 200));
        let n = j.count(&rt);
        assert_eq!(rt.get(n), 2);
        b.delete(&rt, &(1, 100));
        assert_eq!(rt.get(n), 1);
    }

    #[test]
    fn group_by_deletes_every_copy_of_a_duplicate() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let groups = c.group_by(&rt, |x| x % 2);
        c.insert(&rt, 4);
        c.insert(&rt, 4);
        let _ = rt.get(groups.version_node);
        c.delete(&rt, &4);
        c.delete(&rt, &4);
        let _ = rt.get(groups.version_node);
        let evens = groups.get_group(&0).expect("group 0 missing");
        assert_eq!(evens.snapshot_len(), 0);
    }

    #[test]
    fn join_deletes_every_copy_of_a_duplicate() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_collection::<(i32, i32)>();
        let b = rt.create_collection::<(i32, i32)>();
        let j = a.join(&rt, &b, |x| x.0, |y| y.0);
        let n = j.count(&rt);
        a.insert(&rt, (1, 7));
        b.insert(&rt, (1, 100));
        for side in [&a, &b] {
            side.insert(&rt, (1, 10));
            side.insert(&rt, (1, 10));
            assert_eq!(rt.get(n), 3);
            side.delete(&rt, &(1, 10));
            side.delete(&rt, &(1, 10));
            assert_eq!(rt.get(n), 1);
        }
    }

    /// Source with keys 0 and 1 present before `group_by` (bootstrap
    /// groups), then key 2 inserted after the router ran once (a group
    /// created by the routing closure). Returns the grouping and every
    /// group's version node.
    fn grouped_with_late_group<C: Cells>(
        rt: &Runtime<C>,
    ) -> (GroupedCollection<i64, i64, C>, Vec<Incr<u64>>) {
        let c = rt.create_collection::<i64>();
        c.insert(rt, 3);
        c.insert(rt, 4);
        let groups = c.group_by(rt, |x| x % 3);
        let _ = rt.get(groups.version_node());
        c.insert(rt, 5);
        let _ = rt.get(groups.version_node());
        assert_eq!(groups.group_count(), 3);
        let versions = (0..3)
            .map(|k| groups.get_group(&k).unwrap().version_node())
            .collect();
        (groups, versions)
    }

    fn deleting_router_releases_its_groups<C: Cells>(rt: &Runtime<C>) {
        let (groups, versions) = grouped_with_late_group(rt);
        let before = rt.node_count();
        rt.delete_node(groups.version_node());
        assert!(!rt.is_alive(groups.version_node()));
        assert!(versions.iter().all(|&v| !rt.is_alive(v)));
        assert_eq!(rt.node_count(), before - 4);
    }

    #[test]
    fn local_deleting_router_releases_its_groups() {
        deleting_router_releases_its_groups(&Runtime::<Local>::new());
    }

    #[test]
    fn shared_deleting_router_releases_its_groups() {
        deleting_router_releases_its_groups(&Runtime::<Shared>::new());
    }

    fn deleting_router_with_group_consumer_panics<C: Cells>(rt: &Runtime<C>) {
        let (groups, versions) = grouped_with_late_group(rt);
        let late = groups.get_group(&2).unwrap();
        let n = late.count(rt);
        assert_eq!(rt.get(n), 1);

        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rt.delete_node(groups.version_node())
        }))
        .unwrap_err();
        let msg = err.downcast_ref::<String>().unwrap();
        assert!(msg.contains("still has dependents"), "got: {msg}");
        assert!(rt.is_alive(groups.version_node()));
        assert!(
            versions.iter().all(|&v| rt.is_alive(v)),
            "a failed strict delete released some groups"
        );

        rt.delete_node(n);
        rt.delete_node(groups.version_node());
        assert!(versions.iter().all(|&v| !rt.is_alive(v)));
    }

    #[test]
    fn local_deleting_router_with_group_consumer_panics() {
        deleting_router_with_group_consumer_panics(&Runtime::<Local>::new());
    }

    #[test]
    fn shared_deleting_router_with_group_consumer_panics() {
        deleting_router_with_group_consumer_panics(&Runtime::<Shared>::new());
    }

    fn try_delete_of_a_router_is_all_or_nothing<C: Cells>(rt: &Runtime<C>) {
        let (groups, versions) = grouped_with_late_group(rt);
        let n = groups.get_group(&2).unwrap().count(rt);
        assert_eq!(rt.get(n), 1);

        assert_eq!(
            rt.try_delete_node(groups.version_node()),
            Err(DeleteError::HasDependents)
        );
        assert!(rt.is_alive(groups.version_node()));
        assert!(versions.iter().all(|&v| rt.is_alive(v)));

        rt.delete_node(n);
        let before = rt.node_count();
        assert_eq!(rt.try_delete_node(groups.version_node()), Ok(()));
        assert!(versions.iter().all(|&v| !rt.is_alive(v)));
        assert_eq!(rt.node_count(), before - 4);
        rt.assert_invariant_i();
    }

    #[test]
    fn local_try_delete_of_a_router_is_all_or_nothing() {
        try_delete_of_a_router_is_all_or_nothing(&Runtime::<Local>::new());
    }

    #[test]
    fn shared_try_delete_of_a_router_is_all_or_nothing() {
        try_delete_of_a_router_is_all_or_nothing(&Runtime::<Shared>::new());
    }

    #[test]
    fn deleting_an_owned_group_directly_unhooks_it_from_the_router() {
        let rt: Runtime<Local> = Runtime::new();
        let (groups, versions) = grouped_with_late_group(&rt);
        rt.delete_node(versions[1]);
        let recycled = rt.create_input(0_u64);
        assert_eq!(recycled.slot(), versions[1].slot());

        rt.delete_node(groups.version_node());
        assert!(
            rt.is_alive(recycled),
            "the recycled slot was taken as owned"
        );
        assert!(!rt.is_alive(versions[0]) && !rt.is_alive(versions[2]));
    }

    #[test]
    fn collection_handles_confinement_matches_strategy() {
        fn assert_send_sync<X: Send + Sync>() {}
        assert_send_sync::<IncrCollection<i64, Shared>>();
        // The Local variant must NOT be Send or Sync; this is enforced
        // at compile time by PhantomData<C::Ptr<()>>. (A negative-impl
        // assertion is not expressible in stable Rust; the unsendable
        // marker is exercised by the doc-comment contract and by the
        // Python bindings, which wrap Local collections as unsendable.)
    }

    pub(super) mod user_code_panics {
        use super::*;
        use std::cell::Cell;
        use std::hash::Hasher;
        use std::panic::{catch_unwind, AssertUnwindSafe};

        #[derive(Clone, Copy, PartialEq, Debug)]
        pub(in crate::collection) enum Call {
            Hash,
            Eq,
            Clone,
            Drop,
            Closure,
        }

        const CALLS: [Call; 4] = [Call::Hash, Call::Eq, Call::Clone, Call::Drop];

        thread_local! {
            static ARMED: Cell<Option<(Call, usize)>> = const { Cell::new(None) };
        }

        pub(in crate::collection) fn tick(call: Call) {
            ARMED.with(|armed| {
                if let Some((armed_call, n)) = armed.get() {
                    if armed_call == call {
                        // Fires once, so a second armed call during the
                        // unwind cannot turn the panic into an abort.
                        if n == 1 {
                            armed.set(None);
                            panic!("chosen {call:?} call");
                        }
                        armed.set(Some((armed_call, n - 1)));
                    }
                }
            });
        }

        pub(in crate::collection) fn arm(call: Call, nth: usize) {
            ARMED.with(|armed| armed.set(Some((call, nth))));
        }

        pub(in crate::collection) fn disarm() {
            ARMED.with(|armed| armed.set(None));
        }

        #[derive(Debug)]
        pub(in crate::collection) struct Bomb(pub(in crate::collection) i64);

        impl Clone for Bomb {
            fn clone(&self) -> Self {
                tick(Call::Clone);
                Bomb(self.0)
            }
        }

        impl PartialEq for Bomb {
            fn eq(&self, other: &Self) -> bool {
                tick(Call::Eq);
                self.0 == other.0
            }
        }

        impl Eq for Bomb {}

        impl Hash for Bomb {
            fn hash<H: Hasher>(&self, state: &mut H) {
                tick(Call::Hash);
                self.0.hash(state);
            }
        }

        impl Drop for Bomb {
            fn drop(&mut self) {
                tick(Call::Drop);
            }
        }

        #[derive(Clone, Copy, Debug)]
        enum Mutation {
            Insert(i64),
            Delete(i64),
            Replace(Option<i64>, i64),
        }

        fn model(mut elements: Vec<i64>, mutation: Mutation) -> Vec<i64> {
            let mut remove = |x: i64| {
                if let Some(i) = elements.iter().position(|&e| e == x) {
                    elements.remove(i);
                }
            };
            match mutation {
                Mutation::Insert(x) => elements.push(x),
                Mutation::Delete(x) => remove(x),
                Mutation::Replace(old, new) => {
                    if let Some(old) = old {
                        remove(old);
                    }
                    elements.push(new);
                }
            }
            elements.sort_unstable();
            elements
        }

        pub(in crate::collection) fn sorted(values: Vec<Bomb>) -> Vec<i64> {
            let mut out: Vec<i64> = values.iter().map(|b| b.0).collect();
            out.sort_unstable();
            out
        }

        fn mutate<C: Cells>(rt: &Runtime<C>, c: &IncrCollection<Bomb, C>, mutation: Mutation) {
            match mutation {
                Mutation::Insert(x) => c.insert(rt, Bomb(x)),
                Mutation::Delete(x) => {
                    c.delete(rt, &Bomb(x));
                }
                Mutation::Replace(old, new) => {
                    c.replace(rt, old.map(Bomb).as_ref(), Bomb(new));
                }
            }
        }

        pub(in crate::collection) fn populated<C: Cells>(
            rt: &Runtime<C>,
            initial: &[i64],
        ) -> IncrCollection<Bomb, C> {
            let c = rt.create_collection::<Bomb>();
            for &x in initial {
                c.insert(rt, Bomb(x));
            }
            c
        }

        /// Asserts the collection holds `expected` and that a `count` and a
        /// `filter` built before the mutation agree with it, before and
        /// after one more insert.
        fn assert_agree<C: Cells>(
            rt: &Runtime<C>,
            c: &IncrCollection<Bomb, C>,
            count: Incr<u64>,
            all: &IncrCollection<Bomb, C>,
            expected: &[i64],
            context: &str,
        ) {
            let state = sorted(c.values(rt));
            assert_eq!(state, expected, "collection after {context}");
            assert_eq!(
                c.snapshot_len(),
                state.len(),
                "snapshot_len after {context}"
            );
            assert_eq!(rt.get(count), state.len() as u64, "count after {context}");
            assert_eq!(
                sorted(all.values(rt)),
                state,
                "filter output after {context}"
            );

            c.insert(rt, Bomb(-1));
            let len = state.len() as u64 + 1;
            assert_eq!(rt.get(count), len, "count after a later insert, {context}");
            assert_eq!(
                all.values(rt).len() as u64,
                len,
                "filter after a later insert, {context}"
            );
        }

        pub(in crate::collection) fn failure(run: impl FnOnce()) -> Option<String> {
            catch_unwind(AssertUnwindSafe(run)).err().map(|payload| {
                payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default()
            })
        }

        fn assert_no_failures(failures: Vec<String>, runs: usize) {
            assert!(
                failures.is_empty(),
                "{} of {runs} runs left the collection and its readers disagreeing:\n{}",
                failures.len(),
                failures.join("\n")
            );
        }

        /// Runs `mutation` over `initial` with the `nth` call of `call`
        /// panicking. A panic in `Hash`, `Eq` or `Clone` must leave the
        /// collection untouched, and user `Drop` must run only once the
        /// change is complete. Either way its readers must agree with it.
        fn check<C: Cells>(initial: &[i64], mutation: Mutation, call: Call, nth: usize) {
            let rt: Runtime<C> = Runtime::new();
            let c = populated(&rt, initial);
            let count = c.count(&rt);
            let all = c.filter(&rt, |_| true);
            let _ = rt.get(count);
            let _ = all.values(&rt);

            let mut before = initial.to_vec();
            before.sort_unstable();
            let after = model(before.clone(), mutation);

            arm(call, nth);
            let panicked = failure(|| mutate(&rt, &c, mutation)).is_some();
            disarm();

            let expected = if panicked && call != Call::Drop {
                &before
            } else {
                &after
            };
            let shown = if initial.len() > 8 {
                format!("{} elements", initial.len())
            } else {
                format!("{initial:?}")
            };
            let context =
                format!("{mutation:?} over {shown}, {call:?} call {nth} (panicked: {panicked})");
            assert_agree(&rt, &c, count, &all, expected, &context);
        }

        fn mutations_are_atomic_under_user_panics<C: Cells>() {
            use Mutation::*;
            let cases: &[(&[i64], Mutation)] = &[
                (&[1, 2], Insert(3)),
                (&[1, 2], Insert(2)),
                (&[1, 2], Delete(2)),
                (&[1, 2, 2], Delete(2)),
                (&[1, 2], Delete(9)),
                (&[1, 2], Replace(Some(2), 3)),
                (&[1, 2], Replace(Some(1), 2)),
                (&[1, 2], Replace(Some(2), 2)),
                (&[2, 2], Replace(Some(2), 2)),
                (&[1], Replace(Some(9), 3)),
                (&[1], Replace(None, 1)),
                (&[1], Replace(None, 4)),
            ];
            let mut failures = Vec::new();
            let combos = cases
                .iter()
                .flat_map(|&case| CALLS.into_iter().map(move |call| (case, call)))
                .flat_map(|(case, call)| (1..=6).map(move |nth| (case, call, nth)));
            // Miri runs about a second per combination, so it samples every
            // 13th; the stride is prime to the inner loop sizes, so the
            // sample still spans every case and every call.
            let stride = if cfg!(miri) { 13 } else { 1 };
            let mut runs = 0;
            for ((initial, mutation), call, nth) in combos.step_by(stride) {
                runs += 1;
                failures.extend(failure(|| check::<C>(initial, mutation, call, nth)));
            }
            assert_no_failures(failures, runs);
        }

        #[test]
        fn local_mutations_are_atomic_under_user_panics() {
            mutations_are_atomic_under_user_panics::<Local>();
        }

        #[test]
        fn shared_mutations_are_atomic_under_user_panics() {
            mutations_are_atomic_under_user_panics::<Shared>();
        }

        /// The mutation that brings the log to `COMPACT_EVERY` deltas drops
        /// the compacted prefix, running user `Drop` on each delta.
        fn compaction_drops_after_the_version_set<C: Cells>() {
            let initial: Vec<i64> = (0..COMPACT_EVERY as i64 - 1).collect();
            let mutations = [
                Mutation::Insert(5000),
                Mutation::Delete(5),
                Mutation::Replace(Some(5), 5001),
            ];
            let mut failures = Vec::new();
            for mutation in mutations {
                for nth in [1, 2] {
                    failures.extend(failure(|| check::<C>(&initial, mutation, Call::Drop, nth)));
                }
            }
            assert_no_failures(failures, mutations.len() * 2);
        }

        #[test]
        #[cfg_attr(miri, ignore)]
        fn local_compaction_drops_after_the_version_set() {
            compaction_drops_after_the_version_set::<Local>();
        }

        #[test]
        #[cfg_attr(miri, ignore)]
        fn shared_compaction_drops_after_the_version_set() {
            compaction_drops_after_the_version_set::<Shared>();
        }
    }
}
