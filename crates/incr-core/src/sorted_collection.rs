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
//! The same consumer-cursor + compaction + stage-then-apply discipline
//! as `collection.rs` applies; see that module's docs.

use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering as MemOrdering};
use std::sync::Arc;

use crate::cells::Cells;
use crate::collection::{CollectionLog, Delta, IncrCollection, OpLock};
use crate::handle::Incr;
use crate::runtime::Runtime;
use crate::value::Value;

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
    pub(crate) base: u64,
    pub(crate) cursors: Vec<Arc<AtomicU64>>,
    pub(crate) version: u64,
}

impl<T: Clone, K> SortedState<T, K> {
    pub(crate) fn new() -> Self {
        Self {
            sorted: Vec::new(),
            deltas: Vec::new(),
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

    /// Register a positional consumer: cursor at the current end plus a
    /// bootstrap snapshot of the current order.
    fn register_consumer(&mut self) -> (Arc<AtomicU64>, Vec<T>) {
        let cursor = Arc::new(AtomicU64::new(self.end()));
        self.cursors.push(Arc::clone(&cursor));
        let snapshot = self.sorted.iter().map(|(_, t)| t.clone()).collect();
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
                self.deltas.drain(..(min_cursor - self.base) as usize);
                self.base = min_cursor;
            }
        }
    }
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
        self.state
            .read()
            .sorted
            .iter()
            .map(|(_, t)| t.clone())
            .collect()
    }

    pub fn snapshot_len(&self) -> usize {
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
        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        let (cursor, bootstrap) = upstream_log.write().register_consumer();
        let mut st = SortedState::<T, K>::new();
        for v in bootstrap {
            let key = key_fn(&v);
            let pos = st.sorted.partition_point(|(k2, _)| k2 <= &key);
            st.sorted.insert(pos, (key, v.clone()));
            st.push_delta(SortDelta::Insert { pos, value: v });
        }
        let state: Arc<OpLock<SortedState<T, K>, C>> = Arc::new(OpLock::new(st));

        let state_for_query = Arc::clone(&state);
        let version_node = rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);

            let (from, pending) = {
                let up = upstream_log.read();
                let from = cursor.load(MemOrdering::Acquire);
                if from >= up.end() {
                    drop(up);
                    return state_for_query.read().version;
                }
                (from, up.pending_from(from).to_vec())
            };

            // Stage: the user key function runs with no lock held; the
            // apply phase below only compares cached keys via K::Ord.
            let keyed: Vec<(Delta<T>, K)> = pending
                .iter()
                .map(|d| {
                    let k = match d {
                        Delta::Insert(v) | Delta::Delete(v) => key_fn(v),
                    };
                    (d.clone(), k)
                })
                .collect();

            let mut st = state_for_query.write();
            for (d, key) in keyed {
                match d {
                    Delta::Insert(v) => {
                        let pos = st.sorted.partition_point(|(k2, _)| k2 <= &key);
                        st.sorted.insert(pos, (key, v.clone()));
                        st.push_delta(SortDelta::Insert { pos, value: v });
                    }
                    Delta::Delete(v) => {
                        let range_start = st.sorted.partition_point(|(k2, _)| k2 < &key);
                        let range_end = st.sorted.partition_point(|(k2, _)| k2 <= &key);
                        let found = (range_start..range_end).find(|&i| st.sorted[i].1 == v);
                        if let Some(pos) = found {
                            let (_, removed) = st.sorted.remove(pos);
                            st.push_delta(SortDelta::Remove {
                                pos,
                                value: removed,
                            });
                        }
                    }
                }
            }
            cursor.store(from + pending.len() as u64, MemOrdering::Release);
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
        let state = Arc::clone(&self.state);
        let upstream_version = self.version_node;

        let (cursor, bootstrap) = state.write().register_consumer();
        let output_log: Arc<OpLock<CollectionLog<(T, T)>, C>> =
            Arc::new(OpLock::new(CollectionLog::new()));
        {
            let mut out = output_log.write();
            for w in bootstrap.windows(2) {
                out.insert((w[0].clone(), w[1].clone()));
            }
        }
        let mirror: Arc<OpLock<Vec<T>, C>> = Arc::new(OpLock::new(bootstrap));

        let output_log_for_query = Arc::clone(&output_log);
        let mirror_for_query = Arc::clone(&mirror);
        let version_node = rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);

            let (from, pending) = {
                let st = state.read();
                let from = cursor.load(MemOrdering::Acquire);
                if from >= st.end() {
                    drop(st);
                    return output_log_for_query.read().version;
                }
                (from, st.pending_from(from).to_vec())
            };

            let mut mirror = mirror_for_query.write();
            let mut out = output_log_for_query.write();
            for d in &pending {
                match d {
                    SortDelta::Insert { pos, value } => {
                        let pos = *pos;
                        let len = mirror.len();
                        if pos > 0 && pos < len {
                            out.delete(&(mirror[pos - 1].clone(), mirror[pos].clone()));
                        }
                        if pos > 0 {
                            out.insert((mirror[pos - 1].clone(), value.clone()));
                        }
                        if pos < len {
                            out.insert((value.clone(), mirror[pos].clone()));
                        }
                        mirror.insert(pos, value.clone());
                    }
                    SortDelta::Remove { pos, .. } => {
                        let pos = *pos;
                        let v = mirror[pos].clone();
                        if pos > 0 {
                            out.delete(&(mirror[pos - 1].clone(), v.clone()));
                        }
                        if pos + 1 < mirror.len() {
                            out.delete(&(v.clone(), mirror[pos + 1].clone()));
                        }
                        if pos > 0 && pos + 1 < mirror.len() {
                            out.insert((mirror[pos - 1].clone(), mirror[pos + 1].clone()));
                        }
                        mirror.remove(pos);
                    }
                }
            }
            cursor.store(from + pending.len() as u64, MemOrdering::Release);
            out.version
        });

        IncrCollection::derived_with(output_log, version_node)
    }

    /// Window: emit sliding windows of `size` over the sorted view, as a
    /// derived collection of `Vec<T>`.
    ///
    /// Incremental: a positional delta replaces only the windows that
    /// overlap its position, O(size) windows of O(size) elements each.
    pub fn window(&self, rt: &Runtime<C>, size: usize) -> IncrCollection<Vec<T>, C> {
        assert!(size > 0, "window size must be positive");
        let state = Arc::clone(&self.state);
        let upstream_version = self.version_node;

        let (cursor, bootstrap) = state.write().register_consumer();
        let output_log: Arc<OpLock<CollectionLog<Vec<T>>, C>> =
            Arc::new(OpLock::new(CollectionLog::new()));
        {
            let mut out = output_log.write();
            if bootstrap.len() >= size {
                for w in bootstrap.windows(size) {
                    out.insert(w.to_vec());
                }
            }
        }
        let mirror: Arc<OpLock<Vec<T>, C>> = Arc::new(OpLock::new(bootstrap));

        let output_log_for_query = Arc::clone(&output_log);
        let mirror_for_query = Arc::clone(&mirror);
        let version_node = rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);

            let (from, pending) = {
                let st = state.read();
                let from = cursor.load(MemOrdering::Acquire);
                if from >= st.end() {
                    drop(st);
                    return output_log_for_query.read().version;
                }
                (from, st.pending_from(from).to_vec())
            };

            let mut mirror = mirror_for_query.write();
            let mut out = output_log_for_query.write();
            for d in &pending {
                match d {
                    SortDelta::Insert { pos, value } => {
                        let pos = *pos;
                        let old_len = mirror.len();
                        let lo = pos.saturating_sub(size - 1);
                        // Windows that spanned the insertion seam are
                        // replaced; windows containing the new element
                        // are added. Both bands are at most `size` wide.
                        if pos >= 1 && old_len >= size {
                            for s in lo..=(pos - 1).min(old_len - size) {
                                out.delete(&mirror[s..s + size].to_vec());
                            }
                        }
                        mirror.insert(pos, value.clone());
                        let new_len = mirror.len();
                        if new_len >= size {
                            for s in lo..=pos.min(new_len - size) {
                                out.insert(mirror[s..s + size].to_vec());
                            }
                        }
                    }
                    SortDelta::Remove { pos, .. } => {
                        let pos = *pos;
                        let old_len = mirror.len();
                        let lo = pos.saturating_sub(size - 1);
                        if old_len >= size {
                            for s in lo..=pos.min(old_len - size) {
                                out.delete(&mirror[s..s + size].to_vec());
                            }
                        }
                        mirror.remove(pos);
                        let new_len = mirror.len();
                        if pos >= 1 && new_len >= size {
                            for s in lo..=(pos - 1).min(new_len - size) {
                                out.insert(mirror[s..s + size].to_vec());
                            }
                        }
                    }
                }
            }
            cursor.store(from + pending.len() as u64, MemOrdering::Release);
            out.version
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
