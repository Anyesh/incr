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

    /// Internal: create a collection from inside a compute closure (used
    /// by `group_by` for lazy sub-collection creation). Skips the
    /// dep-stack-empty check; the caller is responsible for ensuring
    /// the new version_node is not implicitly a dep of the current
    /// compute.
    pub(crate) fn new_in_compute(rt: &Runtime<C>) -> Self {
        Self {
            log: Arc::new(RwLock::new(CollectionLog::new())),
            version_node: rt.create_input_unchecked(0_u64),
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

    /// Count: number of live elements as an `Incr<u64>`. Maintains a
    /// running tally incrementally from upstream deltas; O(new deltas)
    /// per get rather than O(N) sum over the multiset.
    pub fn count(&self, rt: &Runtime<C>) -> Incr<u64> {
        use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering as MemOrdering};

        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;
        let last_idx = Arc::new(AtomicUsize::new(0));
        // Use signed running count so a stray Delete-of-absent that
        // somehow leaks through doesn't underflow. Cast to u64 on read.
        let running = Arc::new(AtomicI64::new(0));
        let running_for_query = Arc::clone(&running);

        rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);
            let log = upstream_log.read().expect("collection log poisoned");
            let start = last_idx.load(MemOrdering::Relaxed);
            if start < log.deltas.len() {
                let mut delta = 0_i64;
                for d in &log.deltas[start..] {
                    match d {
                        Delta::Insert(_) => delta += 1,
                        Delta::Delete(_) => delta -= 1,
                    }
                }
                running_for_query.fetch_add(delta, MemOrdering::Relaxed);
                last_idx.store(log.deltas.len(), MemOrdering::Relaxed);
            }
            running_for_query.load(MemOrdering::Relaxed).max(0) as u64
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

    /// Join with another collection on a shared key. Emits the
    /// cross-product of matching elements as `(T, U)` pairs. Pairs are
    /// added and removed incrementally as upstream deltas arrive on
    /// either side.
    ///
    /// Both sides maintain a `HashMap<K, Vec<...>>` index keyed by the
    /// extracted key, plus a per-element key cache so deletes route to
    /// the correct bucket. When a new element arrives on one side, we
    /// look up the matching bucket on the other side and emit pairs.
    /// When an element is deleted, we walk the same bucket and emit
    /// corresponding pair removals.
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
        use std::sync::atomic::{AtomicUsize, Ordering as MemOrdering};

        let left_log = Arc::clone(&self.log);
        let right_log = Arc::clone(&right.log);
        let left_version = self.version_node;
        let right_version = right.version_node;
        let left_last = Arc::new(AtomicUsize::new(0));
        let right_last = Arc::new(AtomicUsize::new(0));

        let left_index: Arc<RwLock<HashMap<K, Vec<T>>>> = Arc::new(RwLock::new(HashMap::new()));
        let right_index: Arc<RwLock<HashMap<K, Vec<U>>>> = Arc::new(RwLock::new(HashMap::new()));
        let left_key_cache: Arc<RwLock<HashMap<T, K>>> = Arc::new(RwLock::new(HashMap::new()));
        let right_key_cache: Arc<RwLock<HashMap<U, K>>> = Arc::new(RwLock::new(HashMap::new()));

        let li_for_query = Arc::clone(&left_index);
        let ri_for_query = Arc::clone(&right_index);
        let lkc_for_query = Arc::clone(&left_key_cache);
        let rkc_for_query = Arc::clone(&right_key_cache);

        let output_log: Arc<RwLock<CollectionLog<(T, U)>>> =
            Arc::new(RwLock::new(CollectionLog::new()));
        let output_log_for_query = Arc::clone(&output_log);

        let version_node = rt.create_query(move |rt| -> u64 {
            let _lv = rt.get(left_version);
            let _rv = rt.get(right_version);

            let left = left_log.read().expect("collection log poisoned");
            let right = right_log.read().expect("collection log poisoned");
            let l_start = left_last.load(MemOrdering::Relaxed);
            let r_start = right_last.load(MemOrdering::Relaxed);

            if l_start >= left.deltas.len() && r_start >= right.deltas.len() {
                return output_log_for_query
                    .read()
                    .expect("collection log poisoned")
                    .version;
            }

            let mut li = li_for_query.write().expect("join index poisoned");
            let mut ri = ri_for_query.write().expect("join index poisoned");
            let mut lkc = lkc_for_query.write().expect("key cache poisoned");
            let mut rkc = rkc_for_query.write().expect("key cache poisoned");
            let mut out = output_log_for_query
                .write()
                .expect("collection log poisoned");

            // Process left-side deltas: update left index + key cache,
            // then emit pairs with all matching right-side elements.
            for delta in &left.deltas[l_start..] {
                match delta {
                    Delta::Insert(v) => {
                        let k = left_key(v);
                        lkc.insert(v.clone(), k.clone());
                        li.entry(k.clone()).or_default().push(v.clone());
                        if let Some(matches) = ri.get(&k) {
                            for r in matches {
                                out.insert((v.clone(), r.clone()));
                            }
                        }
                    }
                    Delta::Delete(v) => {
                        if let Some(k) = lkc.remove(v) {
                            if let Some(bucket) = li.get_mut(&k) {
                                if let Some(pos) = bucket.iter().position(|x| x == v) {
                                    bucket.remove(pos);
                                }
                                if bucket.is_empty() {
                                    li.remove(&k);
                                }
                            }
                            if let Some(matches) = ri.get(&k) {
                                for r in matches {
                                    out.delete(&(v.clone(), r.clone()));
                                }
                            }
                        }
                    }
                }
            }
            left_last.store(left.deltas.len(), MemOrdering::Relaxed);

            // Right side, symmetric.
            for delta in &right.deltas[r_start..] {
                match delta {
                    Delta::Insert(u) => {
                        let k = right_key(u);
                        rkc.insert(u.clone(), k.clone());
                        ri.entry(k.clone()).or_default().push(u.clone());
                        if let Some(matches) = li.get(&k) {
                            for l in matches {
                                out.insert((l.clone(), u.clone()));
                            }
                        }
                    }
                    Delta::Delete(u) => {
                        if let Some(k) = rkc.remove(u) {
                            if let Some(bucket) = ri.get_mut(&k) {
                                if let Some(pos) = bucket.iter().position(|x| x == u) {
                                    bucket.remove(pos);
                                }
                                if bucket.is_empty() {
                                    ri.remove(&k);
                                }
                            }
                            if let Some(matches) = li.get(&k) {
                                for l in matches {
                                    out.delete(&(l.clone(), u.clone()));
                                }
                            }
                        }
                    }
                }
            }
            right_last.store(right.deltas.len(), MemOrdering::Relaxed);

            out.version
        });

        IncrCollection {
            log: output_log,
            version_node,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Group by an extracted key. Returns a `GroupedCollection<K, T, C>`
    /// holding one [`IncrCollection<T, C>`] per encountered key. Each
    /// sub-collection is populated incrementally as upstream deltas
    /// arrive: an Insert routes to the group keyed by `key_fn(&value)`,
    /// a Delete removes from the same group.
    ///
    /// Sub-collections are created lazily the first time a key is seen
    /// (via `create_input_unchecked` since the operator runs inside a
    /// compute closure). Their version_nodes are inputs, so users can
    /// continue to compose operators on per-group collections.
    pub fn group_by<K, F>(&self, rt: &Runtime<C>, key_fn: F) -> GroupedCollection<K, T, C>
    where
        K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
        F: Fn(&T) -> K + Send + Sync + 'static,
    {
        use std::sync::atomic::{AtomicUsize, Ordering as MemOrdering};

        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;
        let last_idx = Arc::new(AtomicUsize::new(0));

        let groups: Arc<RwLock<HashMap<K, IncrCollection<T, C>>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let groups_for_query = Arc::clone(&groups);

        // Maps elements to the key they were inserted under, so a Delete
        // for the same value reaches the right group even if the key
        // function is expensive or non-deterministic across calls.
        let key_cache: Arc<RwLock<HashMap<T, K>>> = Arc::new(RwLock::new(HashMap::new()));
        let key_cache_for_query = Arc::clone(&key_cache);

        let output_version_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let output_version_counter_for_query = Arc::clone(&output_version_counter);

        let version_node = rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);

            let upstream = upstream_log.read().expect("collection log poisoned");
            let start = last_idx.load(MemOrdering::Relaxed);
            if start >= upstream.deltas.len() {
                return output_version_counter_for_query.load(MemOrdering::Relaxed);
            }

            let mut grps = groups_for_query.write().expect("grouped state poisoned");
            let mut kc = key_cache_for_query.write().expect("key cache poisoned");

            for delta in &upstream.deltas[start..] {
                match delta {
                    Delta::Insert(v) => {
                        let k = key_fn(v);
                        kc.insert(v.clone(), k.clone());
                        let group = grps
                            .entry(k)
                            .or_insert_with(|| IncrCollection::<T, C>::new_in_compute(rt));
                        let new_ver = group
                            .log
                            .write()
                            .expect("collection log poisoned")
                            .insert(v.clone());
                        rt.set(group.version_node, new_ver);
                    }
                    Delta::Delete(v) => {
                        if let Some(k) = kc.remove(v) {
                            if let Some(group) = grps.get(&k) {
                                let new_ver = group
                                    .log
                                    .write()
                                    .expect("collection log poisoned")
                                    .delete(v);
                                if let Some(ver) = new_ver {
                                    rt.set(group.version_node, ver);
                                }
                            }
                        }
                    }
                }
            }
            last_idx.store(upstream.deltas.len(), MemOrdering::Relaxed);
            output_version_counter_for_query.fetch_add(1, MemOrdering::Relaxed) + 1
        });

        GroupedCollection {
            groups,
            version_node,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Collection partitioned by key. Each key maps to an [`IncrCollection<T, C>`]
/// containing only the elements that belong to that key.
///
/// `version_node` bumps whenever any group changes; downstream queries
/// can depend on it to be notified of any group-level change. To depend
/// on a specific group, use `get_group(&k)` and then depend on that
/// sub-collection's version_node directly.
pub struct GroupedCollection<K, T, C>
where
    K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
    T: Value + Hash + Eq,
    C: Cells,
{
    pub(crate) groups: Arc<RwLock<HashMap<K, IncrCollection<T, C>>>>,
    pub(crate) version_node: Incr<u64>,
    pub(crate) _phantom: std::marker::PhantomData<fn() -> C>,
}

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
            _phantom: std::marker::PhantomData,
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
        self.groups
            .read()
            .expect("grouped state poisoned")
            .keys()
            .cloned()
            .collect()
    }

    pub fn get_group(&self, key: &K) -> Option<IncrCollection<T, C>> {
        self.groups
            .read()
            .expect("grouped state poisoned")
            .get(key)
            .cloned()
    }

    pub fn group_count(&self) -> usize {
        self.groups.read().expect("grouped state poisoned").len()
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
}
