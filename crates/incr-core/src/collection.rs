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
//! User closures (predicates, mappers, key extractors, lift/combine)
//! run with NO lock held: each operator clones the pending delta slice
//! out of the upstream log, evaluates user code into staged effects, and
//! only then takes its output lock to apply clones and hash operations
//! plus the cursor advance. A panicking user closure therefore applies
//! nothing, the cursor stays put, and the compute panic boundary retries
//! the whole batch cleanly. Panics inside user `Hash`/`Eq` impls during
//! the apply phase are outside this guarantee and are documented as
//! unsupported on `Value`.
//!
//! The exactly-once cursor discipline is safe because the runtime's
//! claim protocol guarantees a single thread executes a given operator
//! closure at a time.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering as MemOrdering};
use std::sync::Arc;

use crate::cells::Cells;
use crate::handle::Incr;
use crate::locks::Lock;
use crate::runtime::Runtime;
use crate::value::Value;

/// Strategy-parameterized lock for operator state. Under `Local` this is
/// a `RefCell` (no atomics, which is the point: collection ops on the
/// single-threaded runtime pay no synchronization); under `Shared` it is
/// the poison-ignoring `RwLock`.
pub(crate) struct OpLock<T: 'static, C: Cells>(C::Lock<T>);

impl<T: 'static, C: Cells> OpLock<T, C> {
    pub(crate) fn new(v: T) -> Self {
        Self(<C::Lock<T> as Lock<T>>::new(v))
    }

    pub(crate) fn read(&self) -> <C::Lock<T> as Lock<T>>::ReadGuard<'_> {
        self.0.read()
    }

    pub(crate) fn write(&self) -> <C::Lock<T> as Lock<T>>::WriteGuard<'_> {
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
    pub(crate) base: u64,
    pub(crate) cursors: Vec<Arc<AtomicU64>>,
    pub(crate) elements: HashMap<T, usize>,
    pub(crate) version: u64,
}

impl<T: Hash + Eq + Clone> CollectionLog<T> {
    pub fn new() -> Self {
        Self {
            deltas: Vec::new(),
            base: 0,
            cursors: Vec::new(),
            elements: HashMap::new(),
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

    /// Register a new consumer: returns its cursor (positioned at the
    /// current end) and a bootstrap snapshot of the live multiset that
    /// stands in for the compacted history.
    pub(crate) fn register_consumer(&mut self) -> (Arc<AtomicU64>, Vec<T>) {
        let cursor = Arc::new(AtomicU64::new(self.end()));
        self.cursors.push(Arc::clone(&cursor));
        (cursor, self.elements_vec())
    }

    /// Drop the delta prefix every registered consumer has passed.
    /// Consumers whose cursor Arc was dropped (operator gone) are
    /// pruned. With no consumers the whole log is dropped eagerly.
    fn maybe_compact(&mut self) {
        if self.deltas.len() < COMPACT_EVERY {
            return;
        }
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

    /// Insert `value`. Always accepted; multiset count for the element
    /// is incremented. Returns the new version.
    pub fn insert(&mut self, value: T) -> u64 {
        *self.elements.entry(value.clone()).or_insert(0) += 1;
        self.deltas.push(Delta::Insert(value));
        self.version += 1;
        self.maybe_compact();
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
        self.version += 1;
        self.maybe_compact();
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
/// The `_confine` marker is `PhantomData<C::Ptr<()>>`: under `Local`
/// that is a `Cell<*mut ()>`, making the handle `!Send + !Sync` so it
/// can never carry the RefCell-backed operator locks across threads;
/// under `Shared` it is an `AtomicPtr<()>` and the handle stays
/// `Send + Sync`.
pub struct IncrCollection<T: Value + Hash + Eq, C: Cells> {
    pub(crate) log: Arc<OpLock<CollectionLog<T>, C>>,
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
            version_node: self.version_node,
            derived: self.derived,
            _confine: std::marker::PhantomData,
        }
    }
}

impl<T: Value + Hash + Eq, C: Cells> IncrCollection<T, C> {
    pub(crate) fn new(rt: &Runtime<C>) -> Self {
        Self {
            log: Arc::new(OpLock::new(CollectionLog::new())),
            version_node: rt.create_input(0_u64),
            derived: false,
            _confine: std::marker::PhantomData,
        }
    }

    pub(crate) fn derived_with(
        log: Arc<OpLock<CollectionLog<T>, C>>,
        version_node: Incr<u64>,
    ) -> Self {
        Self {
            log,
            version_node,
            derived: true,
            _confine: std::marker::PhantomData,
        }
    }

    /// Internal: create a sub-collection from inside a compute closure
    /// (used by `group_by`). Marked derived: only the routing operator
    /// may mutate it.
    pub(crate) fn new_in_compute(rt: &Runtime<C>) -> Self {
        Self {
            log: Arc::new(OpLock::new(CollectionLog::new())),
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
        self.check_mutable();
        let new_version = self.log.write().insert(value);
        rt.set(self.version_node, new_version);
    }

    /// Delete one occurrence. No-op (no log delta, no version bump) if
    /// the value was not present. Returns whether a delete was recorded.
    ///
    /// Panics if this collection is the output of an operator.
    pub fn delete(&self, rt: &Runtime<C>, value: &T) -> bool {
        self.check_mutable();
        let new_version = self.log.write().delete(value);
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
        self.log.read().elements.values().sum()
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
        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        let output_log: Arc<OpLock<CollectionLog<T>, C>> =
            Arc::new(OpLock::new(CollectionLog::new()));

        // Bootstrap: seed the output from the live snapshot; the cursor
        // starts past everything the snapshot covered. The predicate
        // runs before the output lock is taken (stage-then-apply).
        let (cursor, bootstrap) = upstream_log.write().register_consumer();
        {
            let kept: Vec<&T> = bootstrap.iter().filter(|v| pred(v)).collect();
            let mut out = output_log.write();
            for v in kept {
                out.insert(v.clone());
            }
        }

        let output_log_for_query = Arc::clone(&output_log);
        let version_node = rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);

            let (from, pending) = {
                let up = upstream_log.read();
                let from = cursor.load(MemOrdering::Acquire);
                if from >= up.end() {
                    drop(up);
                    return output_log_for_query.read().version;
                }
                (from, up.pending_from(from).to_vec())
            };

            // Stage: user predicate runs with no lock held.
            let staged: Vec<Delta<T>> = pending
                .iter()
                .filter(|d| match d {
                    Delta::Insert(v) | Delta::Delete(v) => pred(v),
                })
                .cloned()
                .collect();

            let mut out = output_log_for_query.write();
            for d in staged {
                match d {
                    Delta::Insert(v) => {
                        out.insert(v);
                    }
                    Delta::Delete(v) => {
                        out.delete(&v);
                    }
                }
            }
            cursor.store(from + pending.len() as u64, MemOrdering::Release);
            out.version
        });

        IncrCollection::derived_with(output_log, version_node)
    }

    /// Map: transform every element via `f`. Returns a new derived
    /// collection.
    pub fn map<U, F>(&self, rt: &Runtime<C>, f: F) -> IncrCollection<U, C>
    where
        U: Value + Hash + Eq,
        F: Fn(&T) -> U + Send + Sync + 'static,
    {
        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        let output_log: Arc<OpLock<CollectionLog<U>, C>> =
            Arc::new(OpLock::new(CollectionLog::new()));

        let (cursor, bootstrap) = upstream_log.write().register_consumer();
        {
            let mapped: Vec<U> = bootstrap.iter().map(&f).collect();
            let mut out = output_log.write();
            for v in mapped {
                out.insert(v);
            }
        }

        let output_log_for_query = Arc::clone(&output_log);
        let version_node = rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);

            let (from, pending) = {
                let up = upstream_log.read();
                let from = cursor.load(MemOrdering::Acquire);
                if from >= up.end() {
                    drop(up);
                    return output_log_for_query.read().version;
                }
                (from, up.pending_from(from).to_vec())
            };

            let staged: Vec<Delta<U>> = pending
                .iter()
                .map(|d| match d {
                    Delta::Insert(v) => Delta::Insert(f(v)),
                    Delta::Delete(v) => Delta::Delete(f(v)),
                })
                .collect();

            let mut out = output_log_for_query.write();
            for d in staged {
                match d {
                    Delta::Insert(v) => {
                        out.insert(v);
                    }
                    Delta::Delete(v) => {
                        out.delete(&v);
                    }
                }
            }
            cursor.store(from + pending.len() as u64, MemOrdering::Release);
            out.version
        });

        IncrCollection::derived_with(output_log, version_node)
    }

    /// Count: number of live elements as an `Incr<u64>`. Maintains a
    /// running tally incrementally from upstream deltas; O(new deltas)
    /// per get rather than O(N) sum over the multiset.
    pub fn count(&self, rt: &Runtime<C>) -> Incr<u64> {
        use std::sync::atomic::AtomicI64;

        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        let (cursor, bootstrap) = upstream_log.write().register_consumer();
        // Signed running count so a stray delete-of-absent cannot
        // underflow; clamped to zero on read.
        let running = Arc::new(AtomicI64::new(bootstrap.len() as i64));

        let running_for_query = Arc::clone(&running);
        rt.create_query(move |rt| -> u64 {
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
        let log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        rt.create_query(move |rt| -> U {
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
        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        let (cursor, bootstrap) = upstream_log.write().register_consumer();
        let mut state = AggState::new(identity.clone());
        for v in &bootstrap {
            state.insert(v.clone(), &lift, &combine);
        }
        let state: Arc<OpLock<AggState<T, U>, C>> = Arc::new(OpLock::new(state));

        let state_for_query = Arc::clone(&state);
        rt.create_query(move |rt| -> U {
            let _uv = rt.get(upstream_version);

            let (from, pending) = {
                let up = upstream_log.read();
                let from = cursor.load(MemOrdering::Acquire);
                let pending = if from >= up.end() {
                    Vec::new()
                } else {
                    up.pending_from(from).to_vec()
                };
                (from, pending)
            };

            if !pending.is_empty() {
                let mut st = state_for_query.write();
                if st.needs_rebuild {
                    // A previous combine/lift panic left the tree
                    // unspecified; rebuild from the live snapshot and
                    // skip the pending deltas it already covers.
                    let snapshot = upstream_log.read().elements_vec();
                    let end = upstream_log.read().end();
                    st.rebuild(snapshot, &lift, &combine);
                    cursor.store(end, MemOrdering::Release);
                } else {
                    let applied = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        for d in &pending {
                            match d {
                                Delta::Insert(v) => st.insert(v.clone(), &lift, &combine),
                                Delta::Delete(v) => st.delete(v, &combine),
                            }
                        }
                    }));
                    match applied {
                        Ok(()) => cursor.store(from + pending.len() as u64, MemOrdering::Release),
                        Err(payload) => {
                            st.needs_rebuild = true;
                            std::panic::resume_unwind(payload);
                        }
                    }
                }
            }
            state_for_query.read().root().clone()
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
        struct JoinState<T2, U2, K2> {
            left_index: HashMap<K2, Vec<T2>>,
            right_index: HashMap<K2, Vec<U2>>,
            left_keys: HashMap<T2, K2>,
            right_keys: HashMap<U2, K2>,
        }

        let left_log = Arc::clone(&self.log);
        let right_log = Arc::clone(&right.log);
        let left_version = self.version_node;
        let right_version = right.version_node;

        let (left_cursor, left_boot) = left_log.write().register_consumer();
        let (right_cursor, right_boot) = right_log.write().register_consumer();

        let mut st = JoinState::<T, U, K> {
            left_index: HashMap::new(),
            right_index: HashMap::new(),
            left_keys: HashMap::new(),
            right_keys: HashMap::new(),
        };
        let output_log: Arc<OpLock<CollectionLog<(T, U)>, C>> =
            Arc::new(OpLock::new(CollectionLog::new()));
        {
            // Key extraction (user code) happens before the output lock.
            let left_keyed: Vec<(&T, K)> = left_boot.iter().map(|l| (l, left_key(l))).collect();
            let right_keyed: Vec<(&U, K)> = right_boot.iter().map(|r| (r, right_key(r))).collect();
            let mut out = output_log.write();
            for (l, k) in left_keyed {
                st.left_keys.insert(l.clone(), k.clone());
                st.left_index.entry(k).or_default().push(l.clone());
            }
            for (r, k) in right_keyed {
                st.right_keys.insert(r.clone(), k.clone());
                if let Some(ls) = st.left_index.get(&k) {
                    for l in ls {
                        out.insert((l.clone(), r.clone()));
                    }
                }
                st.right_index.entry(k).or_default().push(r.clone());
            }
        }
        let state: Arc<OpLock<JoinState<T, U, K>, C>> = Arc::new(OpLock::new(st));

        let state_for_query = Arc::clone(&state);
        let output_log_for_query = Arc::clone(&output_log);
        let version_node = rt.create_query(move |rt| -> u64 {
            let _lv = rt.get(left_version);
            let _rv = rt.get(right_version);

            let (l_from, l_pending) = {
                let l = left_log.read();
                let from = left_cursor.load(MemOrdering::Acquire);
                let p = if from >= l.end() {
                    Vec::new()
                } else {
                    l.pending_from(from).to_vec()
                };
                (from, p)
            };
            let (r_from, r_pending) = {
                let r = right_log.read();
                let from = right_cursor.load(MemOrdering::Acquire);
                let p = if from >= r.end() {
                    Vec::new()
                } else {
                    r.pending_from(from).to_vec()
                };
                (from, p)
            };

            if l_pending.is_empty() && r_pending.is_empty() {
                return output_log_for_query.read().version;
            }

            // Stage: key extraction (user code) with no lock held.
            let l_keyed: Vec<(Delta<T>, K)> = l_pending
                .iter()
                .map(|d| {
                    let k = match d {
                        Delta::Insert(v) | Delta::Delete(v) => left_key(v),
                    };
                    (d.clone(), k)
                })
                .collect();
            let r_keyed: Vec<(Delta<U>, K)> = r_pending
                .iter()
                .map(|d| {
                    let k = match d {
                        Delta::Insert(u) | Delta::Delete(u) => right_key(u),
                    };
                    (d.clone(), k)
                })
                .collect();

            let mut st = state_for_query.write();
            let mut out = output_log_for_query.write();

            for (d, k) in l_keyed {
                match d {
                    Delta::Insert(v) => {
                        st.left_keys.insert(v.clone(), k.clone());
                        if let Some(matches) = st.right_index.get(&k) {
                            for r in matches {
                                out.insert((v.clone(), r.clone()));
                            }
                        }
                        st.left_index.entry(k).or_default().push(v);
                    }
                    Delta::Delete(v) => {
                        if st.left_keys.remove(&v).is_some() {
                            if let Some(bucket) = st.left_index.get_mut(&k) {
                                if let Some(pos) = bucket.iter().position(|x| x == &v) {
                                    bucket.remove(pos);
                                }
                                if bucket.is_empty() {
                                    st.left_index.remove(&k);
                                }
                            }
                            if let Some(matches) = st.right_index.get(&k) {
                                for r in matches {
                                    out.delete(&(v.clone(), r.clone()));
                                }
                            }
                        }
                    }
                }
            }
            for (d, k) in r_keyed {
                match d {
                    Delta::Insert(u) => {
                        st.right_keys.insert(u.clone(), k.clone());
                        if let Some(matches) = st.left_index.get(&k) {
                            for l in matches {
                                out.insert((l.clone(), u.clone()));
                            }
                        }
                        st.right_index.entry(k).or_default().push(u);
                    }
                    Delta::Delete(u) => {
                        if st.right_keys.remove(&u).is_some() {
                            if let Some(bucket) = st.right_index.get_mut(&k) {
                                if let Some(pos) = bucket.iter().position(|x| x == &u) {
                                    bucket.remove(pos);
                                }
                                if bucket.is_empty() {
                                    st.right_index.remove(&k);
                                }
                            }
                            if let Some(matches) = st.left_index.get(&k) {
                                for l in matches {
                                    out.delete(&(l.clone(), u.clone()));
                                }
                            }
                        }
                    }
                }
            }
            left_cursor.store(l_from + l_pending.len() as u64, MemOrdering::Release);
            right_cursor.store(r_from + r_pending.len() as u64, MemOrdering::Release);

            out.version
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
        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        let (cursor, bootstrap) = upstream_log.write().register_consumer();

        let groups: Arc<OpLock<HashMap<K, IncrCollection<T, C>>, C>> =
            Arc::new(OpLock::new(HashMap::new()));
        // Maps elements to the key they were inserted under, so a Delete
        // for the same value reaches the right group even if the key
        // function is expensive or non-deterministic across calls.
        let key_cache: Arc<OpLock<HashMap<T, K>, C>> = Arc::new(OpLock::new(HashMap::new()));
        let output_version_counter = Arc::new(AtomicU64::new(0));

        {
            // Bootstrap groups for pre-existing elements. Key extraction
            // (user code) happens before the group locks are taken.
            let keyed: Vec<(&T, K)> = bootstrap.iter().map(|v| (v, key_fn(v))).collect();
            let mut grps = groups.write();
            let mut kc = key_cache.write();
            for (v, k) in keyed {
                kc.insert(v.clone(), k.clone());
                let group = grps
                    .entry(k)
                    .or_insert_with(|| IncrCollection::<T, C>::new_in_compute(rt));
                let new_ver = group.log.write().insert(v.clone());
                rt.set(group.version_node, new_ver);
            }
        }

        let groups_for_query = Arc::clone(&groups);
        let key_cache_for_query = Arc::clone(&key_cache);
        let output_version_counter_for_query = Arc::clone(&output_version_counter);

        let version_node = rt.create_query(move |rt| -> u64 {
            let _uv = rt.get(upstream_version);

            let (from, pending) = {
                let up = upstream_log.read();
                let from = cursor.load(MemOrdering::Acquire);
                if from >= up.end() {
                    drop(up);
                    return output_version_counter_for_query.load(MemOrdering::Relaxed);
                }
                (from, up.pending_from(from).to_vec())
            };

            // Stage: key extraction outside the locks. Deletes look up
            // the cached key during apply instead.
            let keyed: Vec<(Delta<T>, Option<K>)> = pending
                .iter()
                .map(|d| match d {
                    Delta::Insert(v) => (d.clone(), Some(key_fn(v))),
                    Delta::Delete(_) => (d.clone(), None),
                })
                .collect();

            let mut grps = groups_for_query.write();
            let mut kc = key_cache_for_query.write();

            for (d, k) in keyed {
                match d {
                    Delta::Insert(v) => {
                        let k = k.expect("insert delta staged without key");
                        kc.insert(v.clone(), k.clone());
                        let group = grps
                            .entry(k)
                            .or_insert_with(|| IncrCollection::<T, C>::new_in_compute(rt));
                        let new_ver = group.log.write().insert(v.clone());
                        rt.set(group.version_node, new_ver);
                    }
                    Delta::Delete(v) => {
                        if let Some(k) = kc.remove(&v) {
                            if let Some(group) = grps.get(&k) {
                                let new_ver = group.log.write().delete(&v);
                                if let Some(ver) = new_ver {
                                    rt.set(group.version_node, ver);
                                }
                            }
                        }
                    }
                }
            }
            cursor.store(from + pending.len() as u64, MemOrdering::Release);
            output_version_counter_for_query.fetch_add(1, MemOrdering::Relaxed) + 1
        });

        GroupedCollection {
            groups,
            version_node,
            _confine: std::marker::PhantomData,
        }
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
            nodes: vec![identity.clone(); 2],
            cap: 1,
            identity,
        }
    }

    fn root(&self) -> &U {
        &self.nodes[1]
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
        let mut nodes = vec![self.identity.clone(); 2 * new_cap];
        nodes[new_cap..new_cap + self.cap].clone_from_slice(&self.nodes[self.cap..2 * self.cap]);
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
    needs_rebuild: bool,
}

impl<T: Hash + Eq + Clone, U: Clone> AggState<T, U> {
    fn new(identity: U) -> Self {
        Self {
            tree: AggTree::new(identity),
            slots: HashMap::new(),
            free: Vec::new(),
            high_water: 0,
            needs_rebuild: false,
        }
    }

    fn root(&self) -> &U {
        self.tree.root()
    }

    fn insert(&mut self, value: T, lift: &impl Fn(&T) -> U, combine: &impl Fn(&U, &U) -> U) {
        let lifted = lift(&value);
        let slot = self.free.pop().unwrap_or_else(|| {
            if self.high_water == self.tree.cap {
                self.tree.grow(combine);
            }
            let s = self.high_water;
            self.high_water += 1;
            s
        });
        self.slots.entry(value).or_default().push(slot);
        self.tree.set_leaf(slot, lifted, combine);
    }

    fn delete(&mut self, value: &T, combine: &impl Fn(&U, &U) -> U) {
        let Some(bucket) = self.slots.get_mut(value) else {
            return;
        };
        let Some(slot) = bucket.pop() else {
            return;
        };
        if bucket.is_empty() {
            self.slots.remove(value);
        }
        self.tree
            .set_leaf(slot, self.tree.identity.clone(), combine);
        self.free.push(slot);
    }

    fn rebuild(
        &mut self,
        snapshot: Vec<T>,
        lift: &impl Fn(&T) -> U,
        combine: &impl Fn(&U, &U) -> U,
    ) {
        let identity = self.tree.identity.clone();
        *self = Self::new(identity);
        for v in snapshot {
            self.insert(v, lift, combine);
        }
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
    pub(crate) groups: Arc<OpLock<HashMap<K, IncrCollection<T, C>>, C>>,
    pub(crate) version_node: Incr<u64>,
    pub(crate) _confine: std::marker::PhantomData<C::Ptr<()>>,
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
        self.groups.read().keys().cloned().collect()
    }

    pub fn get_group(&self, key: &K) -> Option<IncrCollection<T, C>> {
        self.groups.read().get(key).cloned()
    }

    pub fn group_count(&self) -> usize {
        self.groups.read().len()
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
    fn collection_handles_confinement_matches_strategy() {
        fn assert_send_sync<X: Send + Sync>() {}
        assert_send_sync::<IncrCollection<i64, Shared>>();
        // The Local variant must NOT be Send or Sync; this is enforced
        // at compile time by PhantomData<C::Ptr<()>>. (A negative-impl
        // assertion is not expressible in stable Rust; the unsendable
        // marker is exercised by the doc-comment contract and by the
        // Python bindings, which wrap Local collections as unsendable.)
    }
}
