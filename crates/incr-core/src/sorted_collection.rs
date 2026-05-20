//! `SortedCollection<T, K, C>`: a collection viewed in key-sorted order.
//!
//! Produced by `IncrCollection::sort_by_key`. The sorted view is what
//! enables positional operators like `pairwise` and `window` (which need
//! a stable order). Internally the sorted state is a `Vec<T>` maintained
//! incrementally: each upstream Insert is binary-searched into the right
//! position; each upstream Delete is binary-searched and removed.
//!
//! Storage:
//! - `sorted: Vec<T>` of elements in key order.
//! - `version_node: Incr<u64>` query that processes upstream deltas and
//!   returns the current version.
//! - `key_fn`: closure that extracts the sort key from each element.
//!
//! Positional deltas (`SortDelta`) are not yet exposed externally. The
//! production crate emits them so downstream operators can react to
//! exactly the insert/remove positions; we ship the snapshot-vec
//! semantics first and add positional deltas when we port `pairwise`
//! and `window` past the first cut.

use std::cmp::Ordering;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use crate::cells::Cells;
use crate::collection::{CollectionLog, Delta, IncrCollection};
use crate::handle::Incr;
use crate::runtime::Runtime;
use crate::value::Value;

/// Positional delta on a sorted view. Used internally by pairwise/window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SortDelta<T> {
    /// `value` was inserted at sorted index `pos`.
    Insert { pos: usize, value: T },
    /// `value` was removed from sorted index `pos`.
    Remove { pos: usize, value: T },
}

/// Sorted-view state shared between the sort operator and its downstream
/// consumers. The Vec is the source of truth for the current sorted order;
/// the delta log is the channel that downstream operators consume.
pub(crate) struct SortedState<T, K> {
    pub(crate) sorted: Vec<T>,
    pub(crate) deltas: Vec<SortDelta<T>>,
    pub(crate) version: u64,
    pub(crate) _phantom: std::marker::PhantomData<fn() -> K>,
}

impl<T, K> SortedState<T, K> {
    pub(crate) fn new() -> Self {
        Self {
            sorted: Vec::new(),
            deltas: Vec::new(),
            version: 0,
            _phantom: std::marker::PhantomData,
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
    pub(crate) state: Arc<RwLock<SortedState<T, K>>>,
    pub(crate) version_node: Incr<u64>,
    pub(crate) _phantom: std::marker::PhantomData<fn() -> C>,
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
            _phantom: std::marker::PhantomData,
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

    /// Snapshot of the current sorted view. Acquires the read lock; cheap
    /// in absolute terms but a clone of the entire vec, so do not call in
    /// inner loops.
    pub fn snapshot(&self) -> Vec<T> {
        self.state
            .read()
            .expect("sorted state poisoned")
            .sorted
            .clone()
    }

    pub fn snapshot_len(&self) -> usize {
        self.state
            .read()
            .expect("sorted state poisoned")
            .sorted
            .len()
    }
}

impl<T, C> IncrCollection<T, C>
where
    T: Value + Hash + Eq,
    C: Cells,
{
    /// Sort by an extracted key. Returns a [`SortedCollection`] whose
    /// elements are kept in key order. Insertions binary-search into the
    /// right position; deletions binary-search and remove.
    ///
    /// The sort is stable across re-runs: an element with the same key
    /// as an existing one is placed after the existing one.
    pub fn sort_by_key<K, F>(&self, rt: &Runtime<C>, key_fn: F) -> SortedCollection<T, K, C>
    where
        K: Ord + Clone + Send + Sync + 'static,
        F: Fn(&T) -> K + Send + Sync + 'static,
    {
        use std::sync::atomic::{AtomicUsize, Ordering as MemOrdering};

        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;
        let last_idx = Arc::new(AtomicUsize::new(0));

        let state: Arc<RwLock<SortedState<T, K>>> = Arc::new(RwLock::new(SortedState::new()));
        let state_for_query = Arc::clone(&state);

        let version_node = rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);

            let upstream = upstream_log.read().expect("collection log poisoned");
            let start = last_idx.load(MemOrdering::Relaxed);
            if start >= upstream.deltas.len() {
                return state_for_query
                    .read()
                    .expect("sorted state poisoned")
                    .version;
            }

            let mut st = state_for_query.write().expect("sorted state poisoned");
            for delta in &upstream.deltas[start..] {
                match delta {
                    Delta::Insert(v) => {
                        let key = key_fn(v);
                        // Find insertion point: after the last existing element
                        // with key <= our key (stable order).
                        let pos = st.sorted.partition_point(|other| key_fn(other) <= key);
                        st.sorted.insert(pos, v.clone());
                        st.deltas.push(SortDelta::Insert {
                            pos,
                            value: v.clone(),
                        });
                        st.version = st
                            .version
                            .checked_add(1)
                            .expect("SortedState version overflow");
                    }
                    Delta::Delete(v) => {
                        let key = key_fn(v);
                        // Find a matching element by key, then equality.
                        // Linear scan within the key's range; stable order
                        // means we remove the first match.
                        let range_start = st.sorted.partition_point(|other| key_fn(other) < key);
                        let range_end = st.sorted.partition_point(|other| key_fn(other) <= key);
                        let mut found = None;
                        for i in range_start..range_end {
                            if &st.sorted[i] == v {
                                found = Some(i);
                                break;
                            }
                        }
                        if let Some(pos) = found {
                            let removed = st.sorted.remove(pos);
                            st.deltas.push(SortDelta::Remove {
                                pos,
                                value: removed,
                            });
                            st.version = st
                                .version
                                .checked_add(1)
                                .expect("SortedState version overflow");
                        }
                    }
                }
            }
            last_idx.store(upstream.deltas.len(), MemOrdering::Relaxed);
            st.version
        });

        SortedCollection {
            state,
            version_node,
            _phantom: std::marker::PhantomData,
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
    /// sorted view. The output is a regular [`IncrCollection`] of pairs.
    ///
    /// First-cut implementation: re-derive all pairs from the snapshot on
    /// every change. Truly incremental positional propagation (only the
    /// affected neighbors change) lands when the `SortDelta` channel is
    /// wired in the next slice. Tests confirm correctness; the perf gap
    /// vs production is bounded and we close it before 0.2 ships.
    pub fn pairwise(&self, rt: &Runtime<C>) -> IncrCollection<(T, T), C> {
        let state = Arc::clone(&self.state);
        let upstream_version = self.version_node;

        let output_log: Arc<RwLock<CollectionLog<(T, T)>>> =
            Arc::new(RwLock::new(CollectionLog::new()));
        let output_log_for_query = Arc::clone(&output_log);

        let version_node = rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);

            // Re-derive pairs from the current snapshot.
            let snapshot = state.read().expect("sorted state poisoned").sorted.clone();
            let new_pairs: Vec<(T, T)> = if snapshot.len() < 2 {
                Vec::new()
            } else {
                snapshot
                    .windows(2)
                    .map(|w| (w[0].clone(), w[1].clone()))
                    .collect()
            };

            // Rebuild the output log to match. This is the snapshot
            // semantics; the next slice replaces this with positional
            // updates driven by SortDelta.
            let mut out = output_log_for_query
                .write()
                .expect("collection log poisoned");
            // Drop all old elements; rebuild from new_pairs.
            let to_remove: Vec<(T, T)> = out
                .elements
                .iter()
                .flat_map(|(p, &n)| std::iter::repeat_n(p.clone(), n))
                .collect();
            for p in to_remove {
                out.delete(&p);
            }
            for p in new_pairs {
                out.insert(p);
            }
            out.version
        });

        IncrCollection {
            log: output_log,
            version_node,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Window: emit sliding windows of `size` from the sorted view.
    /// Output is a collection of `Vec<T>` snapshots, one per window
    /// position. Like pairwise, first-cut re-derives from the snapshot.
    pub fn window(&self, rt: &Runtime<C>, size: usize) -> IncrCollection<Vec<T>, C> {
        assert!(size > 0, "window size must be positive");
        let state = Arc::clone(&self.state);
        let upstream_version = self.version_node;

        let output_log: Arc<RwLock<CollectionLog<Vec<T>>>> =
            Arc::new(RwLock::new(CollectionLog::new()));
        let output_log_for_query = Arc::clone(&output_log);

        let version_node = rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);

            let snapshot = state.read().expect("sorted state poisoned").sorted.clone();
            let new_windows: Vec<Vec<T>> = if snapshot.len() < size {
                Vec::new()
            } else {
                snapshot.windows(size).map(|w| w.to_vec()).collect()
            };

            let mut out = output_log_for_query
                .write()
                .expect("collection log poisoned");
            let to_remove: Vec<Vec<T>> = out
                .elements
                .iter()
                .flat_map(|(p, &n)| std::iter::repeat_n(p.clone(), n))
                .collect();
            for w in to_remove {
                out.delete(&w);
            }
            for w in new_windows {
                out.insert(w);
            }
            out.version
        });

        IncrCollection {
            log: output_log,
            version_node,
            _phantom: std::marker::PhantomData,
        }
    }
}

// Suppress unused warning until SortDelta consumers ship.
#[allow(dead_code)]
fn _sort_delta_keep_used() -> Ordering {
    Ordering::Equal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

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
        // Force the sort query to run by reading version_node.
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
    fn local_window_size_3() {
        let rt: Runtime<Local> = Runtime::new();
        let c = rt.create_collection::<i64>();
        let sorted = c.sort_by_key(&rt, |x| *x);
        let windows = sorted.window(&rt, 3);
        for i in 1..=5 {
            c.insert(&rt, i);
        }
        let n = windows.count(&rt);
        // Snapshot [1,2,3,4,5] → windows [1,2,3], [2,3,4], [3,4,5] = 3
        assert_eq!(rt.get(n), 3);
    }
}
