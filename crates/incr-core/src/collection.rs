//! `IncrCollection<T, C>`: incremental collection with delta-log propagation.
//!
//! Each collection holds an append-only log of inserts and deletes plus an
//! `Incr<u64>` version node. Operators (filter, map, count, reduce) are
//! query closures that scan new deltas since their last evaluation index
//! and update their own state incrementally.
//!
//! Storage layout per collection:
//! - `log`: `Arc<C::Lock<CollectionLog<T>>>`, shared across operator
//!   closures that read from this collection.
//! - `version_node`: `Incr<u64>` input node. Bumped on every successful
//!   insert/delete; downstream queries depend on it through `rt.get`.
//!
//! Operator pattern:
//! 1. Capture clones of `upstream_log`, `upstream_version_node`, and a
//!    fresh `last_idx: AtomicUsize` (read-from-upstream cursor).
//! 2. Inside the query, call `rt.get(upstream_version_node)` so the
//!    runtime tracks the version dep.
//! 3. Read the log, scan `deltas[last_idx..]`, process each, advance the
//!    cursor.
//! 4. For filter/map, also push into the operator's own collection log
//!    and bump the output version. For count/reduce, return the
//!    aggregated value directly.
//!
//! This first slice covers filter, map, count, and reduce. sort_by_key,
//! pairwise, group_by, join, and window land in the next slice (they
//! need additional sorted-collection machinery).

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use crate::cells::Cells;
use crate::handle::Incr;
use crate::runtime::Runtime;
use crate::value::Value;

/// One delta event in a collection log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Delta<T> {
    Insert(T),
    Delete(T),
}

/// Append-only delta log + multiset bookkeeping for a single collection.
///
/// `deltas` is the source of truth that operators scan. `elements` is the
/// multiset that lets us validate deletes (no-op if element not present)
/// and supports the `elements_vec()` convenience. `version` is the
/// monotonic counter bumped on every accepted insert/delete; it's the
/// value the `version_node` carries to downstream queries.
pub struct CollectionLog<T: Hash + Eq + Clone> {
    pub(crate) deltas: Vec<Delta<T>>,
    pub(crate) elements: HashMap<T, usize>,
    pub(crate) version: u64,
}

impl<T: Hash + Eq + Clone> CollectionLog<T> {
    pub fn new() -> Self {
        Self {
            deltas: Vec::new(),
            elements: HashMap::new(),
            version: 0,
        }
    }

    /// Insert `value`. Always accepted; multiset count for the element
    /// is incremented. Returns the new version.
    pub fn insert(&mut self, value: T) -> u64 {
        *self.elements.entry(value.clone()).or_insert(0) += 1;
        self.deltas.push(Delta::Insert(value));
        self.version = self
            .version
            .checked_add(1)
            .expect("CollectionLog version overflow");
        self.version
    }

    /// Delete one occurrence of `value`. Returns `Some(new_version)` if
    /// the element was present and a delete was recorded; `None` if the
    /// element was not in the collection (no delta recorded).
    pub fn delete(&mut self, value: &T) -> Option<u64> {
        let count = self.elements.get_mut(value)?;
        *count -= 1;
        if *count == 0 {
            self.elements.remove(value);
        }
        self.deltas.push(Delta::Delete(value.clone()));
        self.version = self
            .version
            .checked_add(1)
            .expect("CollectionLog version overflow");
        Some(self.version)
    }

    /// Snapshot of all live elements, with multiset duplicates expanded.
    pub fn elements_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.elements.values().sum());
        for (val, &count) in &self.elements {
            for _ in 0..count {
                out.push(val.clone());
            }
        }
        out
    }
}

impl<T: Hash + Eq + Clone> Default for CollectionLog<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Public collection handle. Cheap to clone (Arc + Copy handle).
///
/// The log uses `std::sync::RwLock` rather than `C::Lock` so the same
/// type works under both strategies. Under `Local`, this costs one
/// uncontended RwLock acquire per collection op (~5 ns); the alternative
/// would be to thread an `unsafe impl Sync` through `LocalLock` to make
/// it shareable inside Send+Sync compute closures, which would be a
/// footgun for unrelated uses of `LocalLock`. Uniformity wins; the
/// 5 ns per insert/delete is invisible against the rest of the runtime.
pub struct IncrCollection<T: Value + Hash + Eq, C: Cells> {
    pub(crate) log: Arc<RwLock<CollectionLog<T>>>,
    pub(crate) version_node: Incr<u64>,
    pub(crate) _phantom: std::marker::PhantomData<fn() -> C>,
}

impl<T: Value + Hash + Eq, C: Cells> Clone for IncrCollection<T, C> {
    fn clone(&self) -> Self {
        Self {
            log: Arc::clone(&self.log),
            version_node: self.version_node,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Value + Hash + Eq, C: Cells> IncrCollection<T, C> {
    pub(crate) fn new(rt: &Runtime<C>) -> Self {
        Self {
            log: Arc::new(RwLock::new(CollectionLog::new())),
            version_node: rt.create_input(0_u64),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Public accessor for the collection's version node. Useful when a
    /// user query wants to depend on the collection without going through
    /// an operator.
    pub fn version_node(&self) -> Incr<u64> {
        self.version_node
    }

    /// Insert a value. Bumps the underlying log version and notifies
    /// downstream queries by setting `version_node`.
    pub fn insert(&self, rt: &Runtime<C>, value: T) {
        let new_version = self
            .log
            .write()
            .expect("collection log poisoned")
            .insert(value);
        rt.set(self.version_node, new_version);
    }

    /// Delete one occurrence. No-op (no log delta, no version bump) if
    /// the value was not present. Returns whether a delete was recorded.
    pub fn delete(&self, rt: &Runtime<C>, value: &T) -> bool {
        let new_version = self
            .log
            .write()
            .expect("collection log poisoned")
            .delete(value);
        match new_version {
            Some(v) => {
                rt.set(self.version_node, v);
                true
            }
            None => false,
        }
    }

    /// Number of live elements (with multiset duplicates counted).
    pub fn snapshot_len(&self) -> usize {
        self.log
            .read()
            .expect("collection log poisoned")
            .elements
            .values()
            .sum()
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
    /// collection containing the filtered subset, propagated incrementally.
    ///
    /// The returned collection's `version_node` is a query node that, when
    /// observed, scans new upstream deltas, applies the predicate, and
    /// updates the output log. Calling `insert` or `delete` on a derived
    /// collection is not supported (it would set a query node directly,
    /// bypassing the operator and corrupting the state machine); this
    /// constraint is documented and will be enforced by a runtime check
    /// in the API-cleanup slice.
    pub fn filter<F>(&self, rt: &Runtime<C>, pred: F) -> IncrCollection<T, C>
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;
        let last_idx = Arc::new(AtomicUsize::new(0));

        let output_log: Arc<RwLock<CollectionLog<T>>> = Arc::new(RwLock::new(CollectionLog::new()));
        let output_log_for_query = Arc::clone(&output_log);

        let version_node = rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);

            let upstream = upstream_log.read().expect("collection log poisoned");
            let start = last_idx.load(Ordering::Relaxed);
            if start >= upstream.deltas.len() {
                return output_log_for_query
                    .read()
                    .expect("collection log poisoned")
                    .version;
            }

            let mut out = output_log_for_query
                .write()
                .expect("collection log poisoned");
            for delta in &upstream.deltas[start..] {
                match delta {
                    Delta::Insert(v) => {
                        if pred(v) {
                            out.insert(v.clone());
                        }
                    }
                    Delta::Delete(v) => {
                        if pred(v) {
                            out.delete(v);
                        }
                    }
                }
            }
            last_idx.store(upstream.deltas.len(), Ordering::Relaxed);
            out.version
        });

        IncrCollection {
            log: output_log,
            version_node,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Map: transform every element via `f`. Returns a new collection.
    ///
    /// The output collection's `version_node` is a query node; same
    /// derived-collection constraints as `filter`.
    pub fn map<U, F>(&self, rt: &Runtime<C>, f: F) -> IncrCollection<U, C>
    where
        U: Value + Hash + Eq,
        F: Fn(&T) -> U + Send + Sync + 'static,
    {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;
        let last_idx = Arc::new(AtomicUsize::new(0));

        let output_log: Arc<RwLock<CollectionLog<U>>> = Arc::new(RwLock::new(CollectionLog::new()));
        let output_log_for_query = Arc::clone(&output_log);

        let version_node = rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);

            let upstream = upstream_log.read().expect("collection log poisoned");
            let start = last_idx.load(Ordering::Relaxed);
            if start >= upstream.deltas.len() {
                return output_log_for_query
                    .read()
                    .expect("collection log poisoned")
                    .version;
            }

            let mut out = output_log_for_query
                .write()
                .expect("collection log poisoned");
            for delta in &upstream.deltas[start..] {
                match delta {
                    Delta::Insert(v) => {
                        let mapped = f(v);
                        out.insert(mapped);
                    }
                    Delta::Delete(v) => {
                        let mapped = f(v);
                        out.delete(&mapped);
                    }
                }
            }
            last_idx.store(upstream.deltas.len(), Ordering::Relaxed);
            out.version
        });

        IncrCollection {
            log: output_log,
            version_node,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Count: number of live elements as an `Incr<u64>`.
    pub fn count(&self, rt: &Runtime<C>) -> Incr<u64> {
        let log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);
            log.read()
                .expect("collection log poisoned")
                .elements
                .values()
                .sum::<usize>() as u64
        })
    }

    /// Reduce: fold all live elements through `fold_fn`. The fold runs
    /// over a snapshot of the collection on every change. This is the
    /// production semantics (reduce isn't truly incremental); a future
    /// incremental-reduce variant could maintain running aggregates.
    pub fn reduce<U, F>(&self, rt: &Runtime<C>, fold_fn: F) -> Incr<U>
    where
        U: Value,
        F: Fn(&[T]) -> U + Send + Sync + 'static,
    {
        let log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        rt.create_query(move |rt| -> U {
            let _uv = rt.get(upstream_version);
            let elements = log.read().expect("collection log poisoned").elements_vec();
            fold_fn(&elements)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

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
        // Note: the production test uses 255 because it sums 90 + 105 + 60 (no map),
        // but we do 90 + 105 + 70 = 265 because curve adds 10 to each passing.
        assert_eq!(rt.get(total), 265);
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
}
