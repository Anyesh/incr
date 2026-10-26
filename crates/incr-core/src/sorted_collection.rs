//! `SortedCollection<T, K, C>`: a collection viewed in key-sorted order.
//!
//! Produced by `IncrCollection::sort_by_key`. The sorted view is what
//! enables positional operators (`pairwise`, `window`). Internally the
//! sorted state is a `Vec<(K, T)>` maintained incrementally: each
//! upstream Insert is binary-searched into position, each Delete is
//! binary-searched and removed. Keys are computed once per delta (with
//! no lock held) and cached alongside the elements, so re-sorting never
//! re-invokes the user key function on existing elements.
//!
//! Positional deltas (`SortDelta`) are the channel downstream operators
//! consume: `pairwise` and `window` mirror the sorted order and touch
//! only the O(1) pairs / O(window) windows adjacent to each delta's
//! position, so one upstream row produces a constant number of output
//! deltas regardless of collection size. The mirror update itself is a
//! `Vec` insert/remove (a memmove, no clones, no hashing); a B-tree
//! mirror could make that O(log N) if profiles ever demand it.
//!
//! The same consumer-cursor + compaction + stage-then-apply discipline,
//! and the same rebuild after a panicked batch, as `collection.rs`
//! applies; see that module's docs.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering as MemOrdering};
use std::sync::Arc;

use crate::cells::Cells;
use crate::collection::{
    cloned, paired_with, rebuild_snapshot, release_kept, CollectionLog, Delta, Displaced,
    IncrCollection, NeedsRebuild, OpLock,
};
use crate::handle::Incr;
use crate::lock_order::{self, LockKind};
use crate::runtime::Runtime;
use crate::value::Value;
use crate::value_code;

/// Positional delta on a sorted view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SortDelta<T> {
    /// `value` was inserted at sorted index `pos`.
    Insert { pos: usize, value: T },
    /// `value` was removed from sorted index `pos`.
    Remove { pos: usize, value: T },
}

const COMPACT_EVERY: usize = 1024;

/// Sorted-view state shared between the sort operator and its downstream
/// consumers. The `(key, element)` vec is the source of truth for the
/// current order; the delta log is the channel positional consumers
/// scan, with the same base/cursor compaction scheme as CollectionLog.
pub(crate) struct SortedState<T, K> {
    pub(crate) sorted: Vec<(K, T)>,
    pub(crate) deltas: Vec<SortDelta<T>>,
    /// An empty buffer the next compaction moves the kept deltas into.
    pub(crate) spare: Vec<SortDelta<T>>,
    pub(crate) base: u64,
    pub(crate) cursors: Vec<Arc<AtomicU64>>,
    pub(crate) version: u64,
}

impl<T: Clone, K> SortedState<T, K> {
    pub(crate) fn new() -> Self {
        Self {
            sorted: Vec::new(),
            deltas: Vec::new(),
            spare: Vec::new(),
            base: 0,
            cursors: Vec::new(),
            version: 0,
        }
    }

    fn end(&self) -> u64 {
        self.base + self.deltas.len() as u64
    }

    fn pending_from(&self, from: u64) -> &[SortDelta<T>] {
        &self.deltas[(from - self.base) as usize..]
    }

    /// A copy of the pending deltas (user `Clone`), for a consumer to
    /// stage outside this guard.
    fn pending_cloned(&self, from: u64) -> Vec<SortDelta<T>> {
        value_code::run(|| self.pending_from(from).to_vec())
    }

    /// The current order and the log's end, under one guard.
    fn snapshot(&self) -> (Vec<T>, u64) {
        let order = value_code::run(|| self.sorted.iter().map(|(_, t)| t.clone()).collect());
        (order, self.end())
    }

    /// Register a positional consumer: cursor at the current end plus a
    /// bootstrap snapshot of the current order.
    fn register_consumer(&mut self) -> (Arc<AtomicU64>, Vec<T>) {
        let (snapshot, end) = self.snapshot();
        let cursor = Arc::new(AtomicU64::new(end));
        self.cursors.push(Arc::clone(&cursor));
        (cursor, snapshot)
    }

    fn push_delta(&mut self, d: SortDelta<T>) {
        self.deltas.push(d);
        self.version += 1;
        if self.deltas.len() >= COMPACT_EVERY {
            self.cursors.retain(|c| Arc::strong_count(c) > 1);
            let min_cursor = self
                .cursors
                .iter()
                .map(|c| c.load(MemOrdering::Acquire))
                .min()
                .unwrap_or_else(|| self.end());
            if min_cursor > self.base {
                let cut = (min_cursor - self.base) as usize;
                // The kept deltas move into the spare, not a fresh buffer,
                // because a large fresh request makes glibc consolidate its
                // fastbins and slows later small allocations.
                let mut kept = std::mem::take(&mut self.spare);
                kept.reserve(self.deltas.capacity());
                kept.extend(self.deltas.drain(cut..));
                let mut compacted = std::mem::replace(&mut self.deltas, kept);
                self.base = min_cursor;
                // Cleared only now, so that a panicking user `Drop` cannot
                // leave `base` out of step with the deltas; the buffer is
                // then lost and the next compaction allocates afresh.
                compacted.clear();
                self.spare = compacted;
            }
        }
    }

    fn insert(&mut self, value: T, key: K)
    where
        K: Ord,
    {
        let (pos, delta) = value_code::run(|| {
            let pos = self.sorted.partition_point(|(k2, _)| k2 <= &key);
            (pos, value.clone())
        });
        self.sorted.insert(pos, (key, value));
        self.push_delta(SortDelta::Insert { pos, value: delta });
    }

    fn remove(&mut self, pos: usize) {
        let (_, value) = self.sorted.remove(pos);
        self.push_delta(SortDelta::Remove { pos, value });
    }

    /// Pushes the positional deltas that turn the view into `target`. An
    /// entry stays put when the target holds the same element under an
    /// equal key. Every delta is pushed with the view already matching
    /// it, so a panic part-way leaves a view a later converge finishes
    /// from.
    fn converge(&mut self, target: Vec<(T, K)>)
    where
        T: Hash + Eq,
        K: Ord,
    {
        let mut wanted: HashMap<T, Vec<K>> = HashMap::new();
        let mut surplus = Vec::new();
        value_code::run(|| {
            for (value, key) in target {
                wanted.entry(value).or_default().push(key);
            }
            for (pos, (key, value)) in self.sorted.iter().enumerate() {
                let kept = wanted.get_mut(value).and_then(|keys| {
                    let at = keys.iter().position(|k| k == key)?;
                    Some(keys.swap_remove(at))
                });
                if kept.is_none() {
                    surplus.push(pos);
                }
            }
        });
        for pos in surplus.into_iter().rev() {
            self.remove(pos);
        }
        for (value, keys) in wanted {
            for key in keys {
                self.insert(cloned(&value), key);
            }
        }
    }
}

fn pair<T: Clone>(a: &T, b: &T) -> (T, T) {
    value_code::run(|| (a.clone(), b.clone()))
}

fn window_of<T: Clone>(window: &[T]) -> Vec<T> {
    value_code::run(|| window.to_vec())
}

/// Sorted view of an upstream collection.
pub struct SortedCollection<T, K, C>
where
    T: Value + Hash + Eq,
    K: Ord + Clone + Send + Sync + 'static,
    C: Cells,
{
    pub(crate) state: Arc<OpLock<SortedState<T, K>, C>>,
    pub(crate) version_node: Incr<u64>,
    pub(crate) _confine: std::marker::PhantomData<C::Ptr<()>>,
}

impl<T, K, C> Clone for SortedCollection<T, K, C>
where
    T: Value + Hash + Eq,
    K: Ord + Clone + Send + Sync + 'static,
    C: Cells,
{
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            version_node: self.version_node,
            _confine: std::marker::PhantomData,
        }
    }
}

impl<T, K, C> SortedCollection<T, K, C>
where
    T: Value + Hash + Eq,
    K: Ord + Clone + Send + Sync + 'static,
    C: Cells,
{
    pub fn version_node(&self) -> Incr<u64> {
        self.version_node
    }

    /// Snapshot of the current sorted view. A clone of the entire vec;
    /// do not call in inner loops.
    pub fn snapshot(&self) -> Vec<T> {
        value_code::check();
        crate::dep_stack::unrecorded_lookup::<C>();
        self.state.read().snapshot().0
    }

    pub fn snapshot_len(&self) -> usize {
        value_code::check();
        crate::dep_stack::unrecorded_lookup::<C>();
        self.state.read().sorted.len()
    }
}

impl<T, C> IncrCollection<T, C>
where
    T: Value + Hash + Eq,
    C: Cells,
{
    /// Sort by an extracted key. Returns a [`SortedCollection`] whose
    /// elements are kept in key order. Insertions binary-search into the
    /// right position; deletions binary-search and remove. Stable: an
    /// element with an existing key lands after the existing run.
    pub fn sort_by_key<K, F>(&self, rt: &Runtime<C>, key_fn: F) -> SortedCollection<T, K, C>
    where
        K: Ord + Clone + Send + Sync + 'static,
        F: Fn(&T) -> K + Send + Sync + 'static,
    {
        value_code::check();
        let key_fn = lock_order::unlocked("key_fn", key_fn);
        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        let (cursor, bootstrap) = upstream_log.write().register_consumer();
        let mut st = SortedState::<T, K>::new();
        for (v, key) in paired_with(&key_fn, bootstrap) {
            st.insert(v, key);
        }
        let state: Arc<OpLock<SortedState<T, K>, C>> =
            Arc::new(OpLock::new(LockKind::CollectionLog, st));

        let rebuild = NeedsRebuild::new();
        let state_for_query = Arc::clone(&state);
        let version_node = rt.create_operator_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);

            let _provisional = rebuild.provisional(rt);
            if rebuild.due(rt) {
                let (snapshot, end) = rebuild_snapshot(rt, &upstream_log, CollectionLog::snapshot);
                let target = paired_with(&key_fn, snapshot);
                let mut st = state_for_query.write();
                st.converge(target);
                cursor.store(end, MemOrdering::Release);
                rebuild.clear();
                return st.version;
            }

            let (from, pending) = {
                let up = upstream_log.read();
                let from = cursor.load(MemOrdering::Acquire);
                if from >= up.end() {
                    drop(up);
                    return state_for_query.read().version;
                }
                (from, up.pending_cloned(from))
            };
            let to = from + pending.len() as u64;

            // Stage: the user key function runs with no lock held; the
            // apply phase below only compares cached keys via K::Ord.
            let keyed: Vec<(Delta<T>, K)> = pending
                .into_iter()
                .map(|d| {
                    let k = match &d {
                        Delta::Insert(v) | Delta::Delete(v) => key_fn(v),
                    };
                    (d, k)
                })
                .collect();

            let mut st = state_for_query.write();
            rebuild.guard(|| {
                for (d, key) in keyed {
                    match d {
                        Delta::Insert(v) => st.insert(v, key),
                        Delta::Delete(v) => {
                            let found = value_code::run(|| {
                                let range_start = st.sorted.partition_point(|(k2, _)| k2 < &key);
                                let range_end = st.sorted.partition_point(|(k2, _)| k2 <= &key);
                                (range_start..range_end).find(|&i| st.sorted[i].1 == v)
                            });
                            if let Some(pos) = found {
                                st.remove(pos);
                            }
                        }
                    }
                }
            });
            cursor.store(to, MemOrdering::Release);
            st.version
        });

        SortedCollection {
            state,
            version_node,
            _confine: std::marker::PhantomData,
        }
    }
}

impl<T, K, C> SortedCollection<T, K, C>
where
    T: Value + Hash + Eq,
    K: Ord + Clone + Send + Sync + 'static,
    C: Cells,
{
    /// Pairwise: emit `(prev, next)` for every consecutive pair in the
    /// sorted view, as a derived [`IncrCollection`] of pairs.
    ///
    /// Incremental: each positional delta touches at most three pairs
    /// (the seam it breaks plus the two it creates), so one upstream row
    /// produces O(1) output deltas regardless of collection size.
    pub fn pairwise(&self, rt: &Runtime<C>) -> IncrCollection<(T, T), C> {
        let pairs = |mirror: &[T]| -> Vec<(T, T)> {
            mirror.windows(2).map(|w| pair(&w[0], &w[1])).collect()
        };
        self.positional(rt, pairs, |mirror, out, d, keep| match d {
            SortDelta::Insert { pos, value } => {
                let pos = *pos;
                let len = mirror.len();
                if pos > 0 && pos < len {
                    out.delete_kept(pair(&mirror[pos - 1], &mirror[pos]), keep);
                }
                if pos > 0 {
                    out.insert_kept(pair(&mirror[pos - 1], value), keep);
                }
                if pos < len {
                    out.insert_kept(pair(value, &mirror[pos]), keep);
                }
                mirror.insert(pos, cloned(value));
            }
            SortDelta::Remove { pos, .. } => {
                let pos = *pos;
                let v = cloned(&mirror[pos]);
                if pos > 0 {
                    out.delete_kept(pair(&mirror[pos - 1], &v), keep);
                }
                if pos + 1 < mirror.len() {
                    out.delete_kept(pair(&v, &mirror[pos + 1]), keep);
                }
                if pos > 0 && pos + 1 < mirror.len() {
                    out.insert_kept(pair(&mirror[pos - 1], &mirror[pos + 1]), keep);
                }
                mirror.remove(pos);
            }
        })
    }

    /// Window: emit sliding windows of `size` over the sorted view, as a
    /// derived collection of `Vec<T>`.
    ///
    /// Incremental: a positional delta replaces only the windows that
    /// overlap its position, O(size) windows of O(size) elements each.
    pub fn window(&self, rt: &Runtime<C>, size: usize) -> IncrCollection<Vec<T>, C> {
        assert!(size > 0, "window size must be positive");
        let windows =
            move |mirror: &[T]| -> Vec<Vec<T>> { mirror.windows(size).map(window_of).collect() };
        self.positional(rt, windows, move |mirror, out, d, keep| match d {
            SortDelta::Insert { pos, value } => {
                let pos = *pos;
                let old_len = mirror.len();
                let lo = pos.saturating_sub(size - 1);
                // Windows that spanned the insertion seam are
                // replaced; windows containing the new element
                // are added. Both bands are at most `size` wide.
                if pos >= 1 && old_len >= size {
                    for s in lo..=(pos - 1).min(old_len - size) {
                        out.delete_kept(window_of(&mirror[s..s + size]), keep);
                    }
                }
                mirror.insert(pos, cloned(value));
                let new_len = mirror.len();
                if new_len >= size {
                    for s in lo..=pos.min(new_len - size) {
                        out.insert_kept(window_of(&mirror[s..s + size]), keep);
                    }
                }
            }
            SortDelta::Remove { pos, .. } => {
                let pos = *pos;
                let old_len = mirror.len();
                let lo = pos.saturating_sub(size - 1);
                if old_len >= size {
                    for s in lo..=pos.min(old_len - size) {
                        out.delete_kept(window_of(&mirror[s..s + size]), keep);
                    }
                }
                mirror.remove(pos);
                let new_len = mirror.len();
                if pos >= 1 && new_len >= size {
                    for s in lo..=(pos - 1).min(new_len - size) {
                        out.insert_kept(window_of(&mirror[s..s + size]), keep);
                    }
                }
            }
        })
    }

    /// A consumer of the view's positional deltas that mirrors the order
    /// and derives an output from it: `outputs` lists every output of a
    /// mirror, for the bootstrap and a rebuild, and `apply` updates the
    /// mirror and the output for one delta.
    fn positional<O, A>(
        &self,
        rt: &Runtime<C>,
        outputs: impl Fn(&[T]) -> Vec<O> + Send + Sync + 'static,
        apply: A,
    ) -> IncrCollection<O, C>
    where
        O: Value + Hash + Eq,
        A: Fn(&mut Vec<T>, &mut CollectionLog<O>, &SortDelta<T>, &mut Vec<Displaced<O>>)
            + Send
            + Sync
            + 'static,
    {
        value_code::check();
        let state = Arc::clone(&self.state);
        let upstream_version = self.version_node;

        let (cursor, bootstrap) = state.write().register_consumer();
        let output_log: Arc<OpLock<CollectionLog<O>, C>> =
            Arc::new(OpLock::new(LockKind::CollectionLog, CollectionLog::new()));
        {
            let seeded = value_code::run(|| outputs(&bootstrap));
            let mut out = output_log.write();
            for o in seeded {
                out.insert(o);
            }
        }
        let mirror: Arc<OpLock<Vec<T>, C>> =
            Arc::new(OpLock::new(LockKind::OperatorState, bootstrap));

        let rebuild = NeedsRebuild::new();
        let output_log_for_query = Arc::clone(&output_log);
        let version_node = rt.create_operator_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);
            let mut keep = Vec::new();

            let _provisional = rebuild.provisional(rt);
            if rebuild.due(rt) {
                let (snapshot, end) = rebuild_snapshot(rt, &state, SortedState::snapshot);
                let target = value_code::run(|| outputs(&snapshot));
                let mut mirror = mirror.write();
                let mut out = output_log_for_query.write();
                out.converge(target, &mut keep);
                let stale = std::mem::replace(&mut *mirror, snapshot);
                cursor.store(end, MemOrdering::Release);
                rebuild.clear();
                let version = out.version;
                drop(out);
                drop(mirror);
                drop(stale);
                return version;
            }

            let (from, pending) = {
                let st = state.read();
                let from = cursor.load(MemOrdering::Acquire);
                if from >= st.end() {
                    drop(st);
                    return output_log_for_query.read().version;
                }
                (from, st.pending_cloned(from))
            };

            let mut mirror = mirror.write();
            let mut out = output_log_for_query.write();
            rebuild.guard(|| {
                for d in &pending {
                    apply(&mut *mirror, &mut *out, d, &mut keep);
                }
            });
            cursor.store(from + pending.len() as u64, MemOrdering::Release);
            let version = out.version;
            drop(out);
            drop(mirror);
            drop(pending);
            release_kept(&output_log_for_query, keep);
            version
        });

        IncrCollection::derived_with(output_log, version_node)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    fn sorted_elements<T: Value + Hash + Eq + Ord, C: Cells>(c: &IncrCollection<T, C>) -> Vec<T> {
        let mut v = c.log.read().elements_vec();
        v.sort();
        v
    }

    #[test]
    fn local_sort_by_key_basic() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let sorted = c.sort_by_key(&rt, |x| *x);
        c.insert(&rt, 3);
        c.insert(&rt, 1);
        c.insert(&rt, 4);
        c.insert(&rt, 1);
        c.insert(&rt, 5);
        let _ = rt.get(sorted.version_node);
        assert_eq!(sorted.snapshot(), vec![1, 1, 3, 4, 5]);
    }

    #[test]
    fn shared_sort_by_key_basic() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let sorted = c.sort_by_key(&rt, |x| *x);
        c.insert(&rt, 3);
        c.insert(&rt, 1);
        c.insert(&rt, 4);
        let _ = rt.get(sorted.version_node);
        assert_eq!(sorted.snapshot(), vec![1, 3, 4]);
    }

    #[test]
    fn local_sort_delete_removes_correct_element() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let sorted = c.sort_by_key(&rt, |x| *x);
        c.insert(&rt, 3);
        c.insert(&rt, 1);
        c.insert(&rt, 5);
        c.delete(&rt, &3);
        let _ = rt.get(sorted.version_node);
        assert_eq!(sorted.snapshot(), vec![1, 5]);
    }

    #[test]
    fn sort_bootstraps_from_populated_collection() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        c.insert(&rt, 9);
        c.insert(&rt, 2);
        let sorted = c.sort_by_key(&rt, |x| *x);
        let _ = rt.get(sorted.version_node);
        assert_eq!(sorted.snapshot(), vec![2, 9]);
        c.insert(&rt, 5);
        let _ = rt.get(sorted.version_node);
        assert_eq!(sorted.snapshot(), vec![2, 5, 9]);
    }

    #[test]
    fn shared_pairwise_consecutive() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let sorted = c.sort_by_key(&rt, |x| *x);
        let pairs = sorted.pairwise(&rt);
        c.insert(&rt, 10);
        c.insert(&rt, 20);
        c.insert(&rt, 30);
        let n = pairs.count(&rt);
        // (10,20) and (20,30) → 2 pairs
        assert_eq!(rt.get(n), 2);
    }

    #[test]
    fn pairwise_middle_insert_replaces_one_pair() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let sorted = c.sort_by_key(&rt, |x| *x);
        let pairs = sorted.pairwise(&rt);
        let n = pairs.count(&rt);
        c.insert(&rt, 10);
        c.insert(&rt, 30);
        assert_eq!(rt.get(n), 1);
        assert_eq!(sorted_elements(&pairs), vec![(10, 30)]);

        // Insert into the middle: pair (10,30) must be replaced by
        // (10,20) and (20,30).
        c.insert(&rt, 20);
        assert_eq!(rt.get(n), 2);
        assert_eq!(sorted_elements(&pairs), vec![(10, 20), (20, 30)]);

        // Delete the middle: back to the bridging pair.
        c.delete(&rt, &20);
        assert_eq!(rt.get(n), 1);
        assert_eq!(sorted_elements(&pairs), vec![(10, 30)]);
    }

    #[test]
    fn pairwise_endpoint_deletes() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let sorted = c.sort_by_key(&rt, |x| *x);
        let pairs = sorted.pairwise(&rt);
        let n = pairs.count(&rt);
        for v in [1, 2, 3, 4] {
            c.insert(&rt, v);
        }
        assert_eq!(rt.get(n), 3);
        c.delete(&rt, &1);
        let _ = rt.get(n);
        assert_eq!(sorted_elements(&pairs), vec![(2, 3), (3, 4)]);
        c.delete(&rt, &4);
        let _ = rt.get(n);
        assert_eq!(sorted_elements(&pairs), vec![(2, 3)]);
        c.delete(&rt, &2);
        assert_eq!(rt.get(n), 0);
        c.delete(&rt, &3);
        assert_eq!(rt.get(n), 0);
    }

    #[test]
    fn local_window_size_3() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let sorted = c.sort_by_key(&rt, |x| *x);
        let windows = sorted.window(&rt, 3);
        for i in 1..=5 {
            c.insert(&rt, i);
        }
        let n = windows.count(&rt);
        // [1,2,3,4,5] → windows [1,2,3], [2,3,4], [3,4,5]
        assert_eq!(rt.get(n), 3);
    }

    #[test]
    fn window_middle_insert_and_delete_track_exactly() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let sorted = c.sort_by_key(&rt, |x| *x);
        let windows = sorted.window(&rt, 2);
        let n = windows.count(&rt);
        c.insert(&rt, 10);
        c.insert(&rt, 40);
        assert_eq!(rt.get(n), 1);
        assert_eq!(sorted_elements(&windows), vec![vec![10, 40]]);

        c.insert(&rt, 20);
        let _ = rt.get(n);
        assert_eq!(sorted_elements(&windows), vec![vec![10, 20], vec![20, 40]]);
        c.insert(&rt, 30);
        let _ = rt.get(n);
        assert_eq!(
            sorted_elements(&windows),
            vec![vec![10, 20], vec![20, 30], vec![30, 40]]
        );
        c.delete(&rt, &20);
        let _ = rt.get(n);
        assert_eq!(sorted_elements(&windows), vec![vec![10, 30], vec![30, 40]]);
        c.delete(&rt, &10);
        let _ = rt.get(n);
        assert_eq!(sorted_elements(&windows), vec![vec![30, 40]]);
    }

    #[test]
    fn window_matches_batch_rebuild_under_churn() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let sorted = c.sort_by_key(&rt, |x| *x);
        let windows = sorted.window(&rt, 3);
        let n = windows.count(&rt);

        let mut live: Vec<i64> = Vec::new();
        let ops: Vec<(bool, i64)> = vec![
            (true, 5),
            (true, 1),
            (true, 9),
            (true, 3),
            (false, 5),
            (true, 7),
            (true, 2),
            (false, 1),
            (true, 8),
            (false, 9),
            (true, 4),
            (true, 6),
            (false, 3),
        ];
        for (is_insert, v) in ops {
            if is_insert {
                c.insert(&rt, v);
                live.push(v);
            } else {
                c.delete(&rt, &v);
                live.retain_first(&v);
            }
            let _ = rt.get(n);
            live.sort();
            let expected: Vec<Vec<i64>> = if live.len() >= 3 {
                live.windows(3).map(|w| w.to_vec()).collect()
            } else {
                Vec::new()
            };
            let mut expected_sorted = expected;
            expected_sorted.sort();
            assert_eq!(sorted_elements(&windows), expected_sorted);
        }
    }

    /// Steps until the sorted log compacts, three times, and checks that
    /// the buffer its deltas filled comes back as the spare the next
    /// compaction fills, so a compaction in steady state allocates nothing.
    #[test]
    fn sort_compaction_recycles_its_buffer() {
        let rt: Runtime<Shared> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let sorted = c.sort_by_key(&rt, |x| *x);
        let count = sorted.pairwise(&rt).count(&rt);
        let mut next = 0;
        for _ in 0..3 {
            let base = sorted.state.read().base;
            let mut filled = sorted.state.read().deltas.as_ptr();
            while sorted.state.read().base == base {
                filled = sorted.state.read().deltas.as_ptr();
                c.insert(&rt, next);
                next += 1;
                let _ = rt.get(count);
            }
            let st = sorted.state.read();
            assert!(
                st.spare.capacity() >= COMPACT_EVERY,
                "no spare after a compaction"
            );
            assert_eq!(
                st.spare.as_ptr(),
                filled,
                "the filled buffer is not the spare"
            );
            assert!(st.spare.is_empty());
        }
    }

    thread_local! {
        static FUSE_ARMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Panics on the first drop of a `Fuse(0)` while armed.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Fuse(i64);

    impl Drop for Fuse {
        fn drop(&mut self) {
            if self.0 == 0 && FUSE_ARMED.with(|armed| armed.replace(false)) {
                panic!("chosen Drop call");
            }
        }
    }

    /// The upstream insert that brings the sorted log to `COMPACT_EVERY`
    /// deltas has the compaction drop `Fuse(0)`'s delta and panic; the
    /// view, its log and a positional consumer must still agree after
    /// recovery.
    fn sort_compaction_survives_a_panicking_drop<C: Cells>() {
        let rt: Runtime<C> = Runtime::new();
        let c = rt.create_collection::<Fuse>();
        let n = COMPACT_EVERY as i64;
        for i in 0..n - 1 {
            c.insert(&rt, Fuse(i));
        }
        let sorted = c.sort_by_key(&rt, |f| f.0);
        let pairs = sorted.pairwise(&rt);
        let _ = rt.get(pairs.version_node);
        assert_eq!(sorted.state.read().base, 0);

        c.insert(&rt, Fuse(n - 1));
        FUSE_ARMED.with(|armed| armed.set(true));
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rt.get(sorted.version_node);
        }))
        .is_err();
        let armed = FUSE_ARMED.with(|armed| armed.replace(false));
        assert!(panicked && !armed, "the compaction did not drop Fuse(0)");

        c.insert(&rt, Fuse(n));
        let _ = rt.get(pairs.version_node);
        let st = sorted.state.read();
        assert!(st.base > 0, "the sorted log never compacted");
        assert_eq!(st.base + st.deltas.len() as u64, st.version);
        drop(st);
        let order: Vec<i64> = sorted.snapshot().iter().map(|f| f.0).collect();
        assert_eq!(order, (0..=n).collect::<Vec<_>>());
        let mut got: Vec<(i64, i64)> = pairs
            .log
            .read()
            .elements_vec()
            .iter()
            .map(|(a, b)| (a.0, b.0))
            .collect();
        got.sort();
        assert_eq!(got, (0..n).map(|i| (i, i + 1)).collect::<Vec<_>>());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn local_sort_compaction_survives_a_panicking_drop() {
        sort_compaction_survives_a_panicking_drop::<Local>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shared_sort_compaction_survives_a_panicking_drop() {
        sort_compaction_survives_a_panicking_drop::<Shared>();
    }

    trait RetainFirst<T> {
        fn retain_first(&mut self, v: &T);
    }
    impl<T: PartialEq> RetainFirst<T> for Vec<T> {
        fn retain_first(&mut self, v: &T) {
            if let Some(pos) = self.iter().position(|x| x == v) {
                self.remove(pos);
            }
        }
    }
}
