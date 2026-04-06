use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use super::handle::Incr;
use super::runtime::Runtime;

#[derive(Clone, Debug)]
pub enum Delta<T> {
    Insert(T),
    Delete(T),
}

#[derive(Clone, Debug)]
pub(crate) struct VersionedDelta<T> {
    pub version: u64,
    pub delta: Delta<T>,
}

pub(crate) struct CollectionLog<T: Clone + Hash + Eq> {
    pub elements: HashMap<T, usize>,
    pub deltas: Vec<VersionedDelta<T>>,
    pub version: u64,
    multiset: bool,
}

impl<T: Clone + Hash + Eq> CollectionLog<T> {
    pub fn new() -> Self {
        CollectionLog {
            elements: HashMap::new(),
            deltas: Vec::new(),
            version: 0,
            multiset: false,
        }
    }

    pub fn new_multiset() -> Self {
        CollectionLog {
            elements: HashMap::new(),
            deltas: Vec::new(),
            version: 0,
            multiset: true,
        }
    }

    pub fn insert(&mut self, value: T) -> bool {
        if self.multiset {
            let count = self.elements.entry(value.clone()).or_insert(0);
            *count += 1;
            self.version += 1;
            self.deltas.push(VersionedDelta {
                version: self.version,
                delta: Delta::Insert(value),
            });
            true
        } else {
            let count = self.elements.entry(value.clone()).or_insert(0);
            if *count == 0 {
                *count = 1;
                self.version += 1;
                self.deltas.push(VersionedDelta {
                    version: self.version,
                    delta: Delta::Insert(value),
                });
                true
            } else {
                false
            }
        }
    }

    pub fn delete(&mut self, value: &T) -> bool {
        if self.multiset {
            if let Some(count) = self.elements.get_mut(value) {
                *count -= 1;
                self.version += 1;
                self.deltas.push(VersionedDelta {
                    version: self.version,
                    delta: Delta::Delete(value.clone()),
                });
                if *count == 0 {
                    self.elements.remove(value);
                }
                true
            } else {
                false
            }
        } else if self.elements.remove(value).is_some() {
            self.version += 1;
            self.deltas.push(VersionedDelta {
                version: self.version,
                delta: Delta::Delete(value.clone()),
            });
            true
        } else {
            false
        }
    }

    pub fn distinct_elements(&self) -> HashSet<T> {
        self.elements.keys().cloned().collect()
    }

    pub fn elements_vec(&self) -> Vec<T> {
        self.elements
            .iter()
            .flat_map(|(v, &count)| std::iter::repeat(v.clone()).take(count))
            .collect()
    }
}

pub struct IncrCollection<T>
where
    T: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
{
    pub(crate) log: Arc<RwLock<CollectionLog<T>>>,
    pub(crate) version_node: Incr<u64>,
}

impl<T> Clone for IncrCollection<T>
where
    T: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        IncrCollection {
            log: self.log.clone(),
            version_node: self.version_node,
        }
    }
}

/// A raw pointer wrapper that is `Send + Sync`.
///
/// # Safety
/// The caller must ensure the pointer is only dereferenced during
/// stabilization, where the Runtime is alive and node execution is
/// single-threaded per-node (guaranteed by the Computing CAS).
struct SendSyncPtr<T>(*const T);
// Manual Copy/Clone impls to avoid the implicit `T: Copy` bound that #[derive] generates.
impl<T> Copy for SendSyncPtr<T> {}
impl<T> Clone for SendSyncPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
unsafe impl<T> Send for SendSyncPtr<T> {}
unsafe impl<T> Sync for SendSyncPtr<T> {}
impl<T> SendSyncPtr<T> {
    /// # Safety
    /// Caller must ensure the pointee is alive and no mutable alias exists.
    unsafe fn as_ref(&self) -> &T {
        &*self.0
    }
}

pub struct GroupedCollection<K, T>
where
    K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
    T: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
{
    pub(crate) groups: Arc<RwLock<HashMap<K, IncrCollection<T>>>>,
    pub(crate) version_node: Incr<u64>,
    rt_ptr: *const Runtime,
}

// SAFETY: All fields except rt_ptr are Send+Sync. rt_ptr is only
// dereferenced inside compute closures during stabilization (which
// is single-threaded per-node by the state machine's Computing CAS).
unsafe impl<K, T> Send for GroupedCollection<K, T>
where
    K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
    T: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
{
}
unsafe impl<K, T> Sync for GroupedCollection<K, T>
where
    K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
    T: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
{
}

impl<K, T> GroupedCollection<K, T>
where
    K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
    T: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
{
    pub fn keys(&self) -> Vec<K> {
        self.groups.read().unwrap().keys().cloned().collect()
    }

    pub fn get_group(&self, key: &K) -> Option<IncrCollection<T>> {
        self.groups.read().unwrap().get(key).cloned()
    }

    pub fn version_node(&self) -> Incr<u64> {
        self.version_node
    }
}

impl<T> IncrCollection<T>
where
    T: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
{
    pub fn insert(&self, rt: &Runtime, value: T) {
        let ver = {
            let mut log = self.log.write().unwrap();
            if log.insert(value) {
                Some(log.version)
            } else {
                None
            }
        };
        if let Some(v) = ver {
            rt.set(self.version_node, v);
        }
    }

    pub fn delete(&self, rt: &Runtime, value: &T) {
        let ver = {
            let mut log = self.log.write().unwrap();
            if log.delete(value) {
                Some(log.version)
            } else {
                None
            }
        };
        if let Some(v) = ver {
            rt.set(self.version_node, v);
        }
    }

    pub fn elements(&self) -> HashSet<T> {
        self.log.read().unwrap().distinct_elements()
    }

    pub fn version_node(&self) -> Incr<u64> {
        self.version_node
    }

    pub fn filter<F>(&self, rt: &Runtime, predicate: F) -> IncrCollection<T>
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        let upstream_log = self.log.clone();
        let output_log = Arc::new(RwLock::new(CollectionLog::new()));
        let output_log_ref = output_log.clone();
        let last_idx = Arc::new(AtomicUsize::new(0));
        let upstream_ver = self.version_node;

        let version_node = rt.create_query(move |rt| -> u64 {
            let _upstream_v = rt.get(upstream_ver);

            let upstream = upstream_log.read().unwrap();
            let start = last_idx.load(Ordering::Relaxed);
            if start >= upstream.deltas.len() {
                return output_log_ref.read().unwrap().version;
            }

            let mut output = output_log_ref.write().unwrap();

            for vd in &upstream.deltas[start..] {
                match &vd.delta {
                    Delta::Insert(x) => {
                        if predicate(x) {
                            output.insert(x.clone());
                        }
                    }
                    Delta::Delete(x) => {
                        if predicate(x) {
                            output.delete(x);
                        }
                    }
                }
            }

            last_idx.store(upstream.deltas.len(), Ordering::Relaxed);
            output.version
        });

        IncrCollection {
            log: output_log,
            version_node,
        }
    }

    pub fn map<U, F>(&self, rt: &Runtime, f: F) -> IncrCollection<U>
    where
        U: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
        F: Fn(&T) -> U + Send + Sync + 'static,
    {
        let upstream_log = self.log.clone();
        let output_log = Arc::new(RwLock::new(CollectionLog::new_multiset()));
        let output_log_ref = output_log.clone();
        let last_idx = Arc::new(AtomicUsize::new(0));
        let mapping: Arc<RwLock<HashMap<T, U>>> = Arc::new(RwLock::new(HashMap::new()));
        let mapping_ref = mapping.clone();
        let upstream_ver = self.version_node;

        let version_node = rt.create_query(move |rt| -> u64 {
            let _upstream_v = rt.get(upstream_ver);

            let upstream = upstream_log.read().unwrap();
            let start = last_idx.load(Ordering::Relaxed);
            if start >= upstream.deltas.len() {
                return output_log_ref.read().unwrap().version;
            }

            let mut output = output_log_ref.write().unwrap();
            let mut map_state = mapping_ref.write().unwrap();

            for vd in &upstream.deltas[start..] {
                match &vd.delta {
                    Delta::Insert(x) => {
                        let y = f(x);
                        map_state.insert(x.clone(), y.clone());
                        output.insert(y);
                    }
                    Delta::Delete(x) => {
                        if let Some(y) = map_state.remove(x) {
                            output.delete(&y);
                        }
                    }
                }
            }

            last_idx.store(upstream.deltas.len(), Ordering::Relaxed);
            output.version
        });

        IncrCollection {
            log: output_log,
            version_node,
        }
    }

    pub fn count(&self, rt: &Runtime) -> Incr<u64> {
        let upstream_log = self.log.clone();
        let upstream_ver = self.version_node;
        let current_count = Arc::new(AtomicUsize::new(0));
        let count_ref = current_count.clone();
        let last_idx = Arc::new(AtomicUsize::new(0));

        rt.create_query(move |rt| -> u64 {
            let _upstream_v = rt.get(upstream_ver);

            let upstream = upstream_log.read().unwrap();
            let start = last_idx.load(Ordering::Relaxed);
            if start >= upstream.deltas.len() {
                return count_ref.load(Ordering::Relaxed) as u64;
            }

            let mut count = count_ref.load(Ordering::Relaxed);

            for vd in &upstream.deltas[start..] {
                match &vd.delta {
                    Delta::Insert(_) => count += 1,
                    Delta::Delete(_) => count -= 1,
                }
            }

            last_idx.store(upstream.deltas.len(), Ordering::Relaxed);
            count_ref.store(count, Ordering::Relaxed);
            count as u64
        })
    }

    pub fn reduce<A, F>(&self, rt: &Runtime, fold_fn: F) -> Incr<A>
    where
        A: super::value::Value,
        F: Fn(&Vec<T>) -> A + Send + Sync + 'static,
    {
        let upstream_log = self.log.clone();
        let upstream_ver = self.version_node;
        let last_idx = Arc::new(AtomicUsize::new(0));

        rt.create_query(move |rt| -> A {
            let _upstream_v = rt.get(upstream_ver);

            let upstream = upstream_log.read().unwrap();
            let start = last_idx.load(Ordering::Relaxed);
            if start >= upstream.deltas.len() {
                let elems = upstream.elements_vec();
                return fold_fn(&elems);
            }

            last_idx.store(upstream.deltas.len(), Ordering::Relaxed);
            let elems = upstream.elements_vec();
            fold_fn(&elems)
        })
    }

    pub fn group_by<K, F>(&self, rt: &Runtime, key_fn: F) -> GroupedCollection<K, T>
    where
        K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
        F: Fn(&T) -> K + Send + Sync + 'static,
    {
        let upstream_log = self.log.clone();
        let upstream_ver = self.version_node;
        let last_idx = Arc::new(AtomicUsize::new(0));
        let groups: Arc<RwLock<HashMap<K, IncrCollection<T>>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let groups_ref = groups.clone();
        let key_cache: Arc<RwLock<HashMap<T, K>>> = Arc::new(RwLock::new(HashMap::new()));
        let key_cache_ref = key_cache.clone();
        let rt_ptr = SendSyncPtr(rt as *const Runtime);

        let version_counter = Arc::new(AtomicU64::new(0));
        let version_counter_ref = version_counter.clone();

        let version_node = rt.create_query(move |_rt| -> u64 {
            let rt = unsafe { rt_ptr.as_ref() };
            let _upstream_v = rt.get(upstream_ver);

            let upstream = upstream_log.read().unwrap();
            let start = last_idx.load(Ordering::Relaxed);
            if start >= upstream.deltas.len() {
                return version_counter_ref.load(Ordering::Relaxed);
            }

            let mut grps = groups_ref.write().unwrap();
            let mut kc = key_cache_ref.write().unwrap();

            for vd in &upstream.deltas[start..] {
                match &vd.delta {
                    Delta::Insert(x) => {
                        let k = key_fn(x);
                        kc.insert(x.clone(), k.clone());
                        let group = grps.entry(k).or_insert_with(|| rt.create_collection::<T>());
                        let ver = {
                            let mut log = group.log.write().unwrap();
                            log.insert(x.clone());
                            log.version
                        };
                        rt.set(group.version_node, ver);
                    }
                    Delta::Delete(x) => {
                        if let Some(k) = kc.remove(x) {
                            if let Some(group) = grps.get(&k) {
                                let ver = {
                                    let mut log = group.log.write().unwrap();
                                    log.delete(x);
                                    log.version
                                };
                                rt.set(group.version_node, ver);
                            }
                        }
                    }
                }
            }

            last_idx.store(upstream.deltas.len(), Ordering::Relaxed);
            version_counter_ref.fetch_add(1, Ordering::Relaxed) + 1
        });

        GroupedCollection {
            groups,
            version_node,
            rt_ptr: rt_ptr.0,
        }
    }

    pub fn join<U, K, FL, FR>(
        &self,
        rt: &Runtime,
        right: &IncrCollection<U>,
        left_key: FL,
        right_key: FR,
    ) -> IncrCollection<(T, U)>
    where
        U: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
        K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
        FL: Fn(&T) -> K + Send + Sync + 'static,
        FR: Fn(&U) -> K + Send + Sync + 'static,
    {
        let left_log = self.log.clone();
        let right_log = right.log.clone();
        let left_ver = self.version_node;
        let right_ver = right.version_node;
        let left_last = Arc::new(AtomicUsize::new(0));
        let right_last = Arc::new(AtomicUsize::new(0));

        let left_index: Arc<RwLock<HashMap<K, Vec<T>>>> = Arc::new(RwLock::new(HashMap::new()));
        let right_index: Arc<RwLock<HashMap<K, Vec<U>>>> = Arc::new(RwLock::new(HashMap::new()));
        let left_key_cache: Arc<RwLock<HashMap<T, K>>> = Arc::new(RwLock::new(HashMap::new()));
        let right_key_cache: Arc<RwLock<HashMap<U, K>>> = Arc::new(RwLock::new(HashMap::new()));

        let left_idx_ref = left_index.clone();
        let right_idx_ref = right_index.clone();
        let left_kc_ref = left_key_cache.clone();
        let right_kc_ref = right_key_cache.clone();

        let output_log = Arc::new(RwLock::new(CollectionLog::new_multiset()));
        let output_log_ref = output_log.clone();

        let version_node = rt.create_query(move |rt| -> u64 {
            let _lv = rt.get(left_ver);
            let _rv = rt.get(right_ver);

            let left_up = left_log.read().unwrap();
            let right_up = right_log.read().unwrap();
            let l_start = left_last.load(Ordering::Relaxed);
            let r_start = right_last.load(Ordering::Relaxed);

            if l_start >= left_up.deltas.len() && r_start >= right_up.deltas.len() {
                return output_log_ref.read().unwrap().version;
            }

            let mut li = left_idx_ref.write().unwrap();
            let mut ri = right_idx_ref.write().unwrap();
            let mut lkc = left_kc_ref.write().unwrap();
            let mut rkc = right_kc_ref.write().unwrap();
            let mut output = output_log_ref.write().unwrap();

            // Process left deltas
            for vd in &left_up.deltas[l_start..] {
                match &vd.delta {
                    Delta::Insert(x) => {
                        let k = left_key(x);
                        lkc.insert(x.clone(), k.clone());
                        li.entry(k.clone()).or_default().push(x.clone());
                        if let Some(rights) = ri.get(&k) {
                            for r in rights {
                                output.insert((x.clone(), r.clone()));
                            }
                        }
                    }
                    Delta::Delete(x) => {
                        if let Some(k) = lkc.remove(x) {
                            if let Some(lefts) = li.get_mut(&k) {
                                lefts.retain(|l| l != x);
                            }
                            if let Some(rights) = ri.get(&k) {
                                for r in rights {
                                    output.delete(&(x.clone(), r.clone()));
                                }
                            }
                        }
                    }
                }
            }

            // Process right deltas
            for vd in &right_up.deltas[r_start..] {
                match &vd.delta {
                    Delta::Insert(x) => {
                        let k = right_key(x);
                        rkc.insert(x.clone(), k.clone());
                        ri.entry(k.clone()).or_default().push(x.clone());
                        if let Some(lefts) = li.get(&k) {
                            for l in lefts {
                                output.insert((l.clone(), x.clone()));
                            }
                        }
                    }
                    Delta::Delete(x) => {
                        if let Some(k) = rkc.remove(x) {
                            if let Some(rights) = ri.get_mut(&k) {
                                rights.retain(|r| r != x);
                            }
                            if let Some(lefts) = li.get(&k) {
                                for l in lefts {
                                    output.delete(&(l.clone(), x.clone()));
                                }
                            }
                        }
                    }
                }
            }

            left_last.store(left_up.deltas.len(), Ordering::Relaxed);
            right_last.store(right_up.deltas.len(), Ordering::Relaxed);
            output.version
        });

        IncrCollection {
            log: output_log,
            version_node,
        }
    }
}

impl Runtime {
    pub fn create_collection<T>(&self) -> IncrCollection<T>
    where
        T: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
    {
        let log = Arc::new(RwLock::new(CollectionLog::new()));
        let version_node = self.create_input::<u64>(0);
        IncrCollection { log, version_node }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_insert() {
        let mut log = CollectionLog::new();
        assert!(log.insert(1_i64));
        assert_eq!(log.elements.len(), 1);
        assert_eq!(log.version, 1);
        assert_eq!(log.deltas.len(), 1);
    }

    #[test]
    fn log_insert_duplicate_is_noop() {
        let mut log = CollectionLog::new();
        assert!(log.insert(1_i64));
        assert!(!log.insert(1_i64));
        assert_eq!(log.version, 1);
    }

    #[test]
    fn log_delete() {
        let mut log = CollectionLog::new();
        log.insert(1_i64);
        assert!(log.delete(&1));
        assert_eq!(log.elements.len(), 0);
        assert_eq!(log.version, 2);
    }

    #[test]
    fn log_delete_missing_is_noop() {
        let mut log: CollectionLog<i64> = CollectionLog::new();
        assert!(!log.delete(&1));
    }

    #[test]
    fn log_multiset_allows_duplicates() {
        let mut log = CollectionLog::new_multiset();
        assert!(log.insert(1_i64));
        assert!(log.insert(1_i64));
        assert_eq!(*log.elements.get(&1).unwrap(), 2);
        assert_eq!(log.version, 2);
    }

    #[test]
    fn create_and_insert() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        col.insert(&rt, 1);
        col.insert(&rt, 2);
        col.insert(&rt, 3);
        assert_eq!(col.log.read().unwrap().elements.len(), 3);
    }

    #[test]
    fn insert_bumps_version_node() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        assert_eq!(rt.get(col.version_node), 0);
        col.insert(&rt, 1);
        assert_eq!(rt.get(col.version_node), 1);
        col.insert(&rt, 2);
        assert_eq!(rt.get(col.version_node), 2);
    }

    #[test]
    fn delete_bumps_version_node() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        col.insert(&rt, 1);
        col.insert(&rt, 2);
        col.delete(&rt, &1);
        assert_eq!(rt.get(col.version_node), 3);
    }

    #[test]
    fn duplicate_insert_no_version_bump() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        col.insert(&rt, 1);
        assert_eq!(rt.get(col.version_node), 1);
        col.insert(&rt, 1);
        assert_eq!(rt.get(col.version_node), 1);
    }

    #[test]
    fn elements_returns_distinct_set() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        col.insert(&rt, 1);
        col.insert(&rt, 2);
        col.insert(&rt, 3);
        let elems = col.elements();
        assert_eq!(elems.len(), 3);
        assert!(elems.contains(&1));
        assert!(elems.contains(&2));
        assert!(elems.contains(&3));
    }

    // ── filter tests ────────────────────────────────────────────────────────

    #[test]
    fn filter_basic() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let evens = col.filter(&rt, |x| x % 2 == 0);

        col.insert(&rt, 1);
        col.insert(&rt, 2);
        col.insert(&rt, 3);
        col.insert(&rt, 4);

        let _ = rt.get(evens.version_node);
        assert_eq!(evens.log.read().unwrap().elements.len(), 2);
    }

    #[test]
    fn filter_incremental_insert() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let evens = col.filter(&rt, |x| x % 2 == 0);

        col.insert(&rt, 2);
        let _ = rt.get(evens.version_node);
        assert_eq!(evens.log.read().unwrap().elements.len(), 1);

        col.insert(&rt, 4);
        let _ = rt.get(evens.version_node);
        assert_eq!(evens.log.read().unwrap().elements.len(), 2);

        col.insert(&rt, 3);
        let _ = rt.get(evens.version_node);
        assert_eq!(evens.log.read().unwrap().elements.len(), 2);
    }

    #[test]
    fn filter_incremental_delete() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let evens = col.filter(&rt, |x| x % 2 == 0);

        col.insert(&rt, 2);
        col.insert(&rt, 4);
        let _ = rt.get(evens.version_node);
        assert_eq!(evens.log.read().unwrap().elements.len(), 2);

        col.delete(&rt, &2);
        let _ = rt.get(evens.version_node);
        assert_eq!(evens.log.read().unwrap().elements.len(), 1);
    }

    #[test]
    fn filter_chained() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let positive = col.filter(&rt, |x| *x > 0);
        let small = positive.filter(&rt, |x| *x < 10);

        col.insert(&rt, -5);
        col.insert(&rt, 3);
        col.insert(&rt, 15);
        col.insert(&rt, 7);

        let _ = rt.get(small.version_node);
        assert_eq!(small.log.read().unwrap().elements.len(), 2);
    }

    // ── map tests ───────────────────────────────────────────────────────────

    #[test]
    fn map_basic() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let doubled = col.map(&rt, |x| x * 2);

        col.insert(&rt, 1);
        col.insert(&rt, 2);
        col.insert(&rt, 3);

        let _ = rt.get(doubled.version_node);
        let elements = doubled.log.read().unwrap().elements_vec();
        assert_eq!(elements.len(), 3);
        assert!(elements.contains(&2));
        assert!(elements.contains(&4));
        assert!(elements.contains(&6));
    }

    #[test]
    fn map_delete_propagates() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let doubled = col.map(&rt, |x| x * 2);

        col.insert(&rt, 1);
        col.insert(&rt, 2);
        let _ = rt.get(doubled.version_node);
        assert_eq!(doubled.log.read().unwrap().elements.len(), 2);

        col.delete(&rt, &1);
        let _ = rt.get(doubled.version_node);
        assert_eq!(doubled.log.read().unwrap().elements.len(), 1);
        assert!(doubled.log.read().unwrap().elements.contains_key(&4));
    }

    #[test]
    fn filter_then_map() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let evens = col.filter(&rt, |x| x % 2 == 0);
        let doubled = evens.map(&rt, |x| x * 2);

        col.insert(&rt, 1);
        col.insert(&rt, 2);
        col.insert(&rt, 3);
        col.insert(&rt, 4);

        let _ = rt.get(doubled.version_node);
        let elements = doubled.log.read().unwrap().elements_vec();
        assert_eq!(elements.len(), 2);
        assert!(elements.contains(&4));
        assert!(elements.contains(&8));
    }

    // ── count tests ─────────────────────────────────────────────────────────

    #[test]
    fn count_basic() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let count = col.count(&rt);

        assert_eq!(rt.get(count), 0);
        col.insert(&rt, 1);
        assert_eq!(rt.get(count), 1);
        col.insert(&rt, 2);
        assert_eq!(rt.get(count), 2);
        col.delete(&rt, &1);
        assert_eq!(rt.get(count), 1);
    }

    #[test]
    fn count_after_filter() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let evens = col.filter(&rt, |x| x % 2 == 0);
        let count = evens.count(&rt);

        col.insert(&rt, 1);
        col.insert(&rt, 2);
        col.insert(&rt, 3);
        col.insert(&rt, 4);
        assert_eq!(rt.get(count), 2);

        col.insert(&rt, 6);
        assert_eq!(rt.get(count), 3);
    }

    #[test]
    fn count_early_cutoff() {
        use std::sync::atomic::AtomicU32;

        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let evens = col.filter(&rt, |x| x % 2 == 0);
        let count = evens.count(&rt);

        let downstream_count = Arc::new(AtomicU32::new(0));
        let dc = downstream_count.clone();
        let label = rt.create_query(move |rt| {
            dc.fetch_add(1, Ordering::Relaxed);
            format!("{} evens", rt.get(count))
        });

        col.insert(&rt, 2);
        assert_eq!(rt.get(label), "1 evens");
        assert_eq!(downstream_count.load(Ordering::Relaxed), 1);

        col.insert(&rt, 3); // odd, count unchanged
        assert_eq!(rt.get(label), "1 evens");
        assert_eq!(downstream_count.load(Ordering::Relaxed), 1); // early cutoff
    }

    // ── reduce tests ────────────────────────────────────────────────────────

    #[test]
    fn reduce_sum() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sum = col.reduce(&rt, |elements| -> i64 { elements.iter().sum() });

        assert_eq!(rt.get(sum), 0);
        col.insert(&rt, 10);
        assert_eq!(rt.get(sum), 10);
        col.insert(&rt, 20);
        assert_eq!(rt.get(sum), 30);
        col.delete(&rt, &10);
        assert_eq!(rt.get(sum), 20);
    }

    #[test]
    fn reduce_max() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let max = col.reduce(&rt, |elements| -> Option<i64> {
            elements.iter().copied().max()
        });

        assert_eq!(rt.get(max), None);
        col.insert(&rt, 5);
        assert_eq!(rt.get(max), Some(5));
        col.insert(&rt, 3);
        assert_eq!(rt.get(max), Some(5));
        col.insert(&rt, 8);
        assert_eq!(rt.get(max), Some(8));
        col.delete(&rt, &8);
        assert_eq!(rt.get(max), Some(5));
    }

    #[test]
    fn reduce_after_filter() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let evens = col.filter(&rt, |x| x % 2 == 0);
        let sum = evens.reduce(&rt, |elements| -> i64 { elements.iter().sum() });

        col.insert(&rt, 1);
        col.insert(&rt, 2);
        col.insert(&rt, 3);
        col.insert(&rt, 4);
        assert_eq!(rt.get(sum), 6);
    }

    // ── group_by tests ──────────────────────────────────────────────────────

    #[test]
    fn group_by_basic() {
        let rt = Runtime::new();
        let col = rt.create_collection::<(String, i64)>();
        let grouped = col.group_by(&rt, |x: &(String, i64)| x.0.clone());

        col.insert(&rt, ("a".to_string(), 1));
        col.insert(&rt, ("b".to_string(), 2));
        col.insert(&rt, ("a".to_string(), 3));

        let _ = rt.get(grouped.version_node);
        let groups = grouped.groups.read().unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups.get("a").unwrap().elements().len(), 2);
        assert_eq!(groups.get("b").unwrap().elements().len(), 1);
    }

    #[test]
    fn group_by_delete() {
        let rt = Runtime::new();
        let col = rt.create_collection::<(String, i64)>();
        let grouped = col.group_by(&rt, |x: &(String, i64)| x.0.clone());

        col.insert(&rt, ("a".to_string(), 1));
        col.insert(&rt, ("a".to_string(), 2));
        let _ = rt.get(grouped.version_node);

        col.delete(&rt, &("a".to_string(), 1));
        let _ = rt.get(grouped.version_node);
        let groups = grouped.groups.read().unwrap();
        assert_eq!(groups.get("a").unwrap().elements().len(), 1);
    }

    // ── join tests ──────────────────────────────────────────────────────────

    #[test]
    fn join_basic() {
        let rt = Runtime::new();
        let left = rt.create_collection::<(String, i64)>();
        let right = rt.create_collection::<(String, String)>();

        let joined = left.join(
            &rt,
            &right,
            |l: &(String, i64)| l.0.clone(),
            |r: &(String, String)| r.0.clone(),
        );

        left.insert(&rt, ("a".to_string(), 1));
        left.insert(&rt, ("b".to_string(), 2));
        right.insert(&rt, ("a".to_string(), "x".to_string()));
        right.insert(&rt, ("c".to_string(), "y".to_string()));

        let _ = rt.get(joined.version_node);
        let elems = joined.elements();
        assert_eq!(elems.len(), 1);
        assert!(elems.contains(&(("a".to_string(), 1), ("a".to_string(), "x".to_string()))));
    }

    #[test]
    fn join_multiple_matches() {
        let rt = Runtime::new();
        let left = rt.create_collection::<(i64, i64)>();
        let right = rt.create_collection::<(i64, i64)>();

        let joined = left.join(&rt, &right, |l: &(i64, i64)| l.0, |r: &(i64, i64)| r.0);

        left.insert(&rt, (1, 10));
        left.insert(&rt, (1, 20));
        right.insert(&rt, (1, 100));

        let _ = rt.get(joined.version_node);
        let elems = joined.elements();
        assert_eq!(elems.len(), 2);
    }

    #[test]
    fn join_delete_propagates() {
        let rt = Runtime::new();
        let left = rt.create_collection::<(i64, i64)>();
        let right = rt.create_collection::<(i64, i64)>();

        let joined = left.join(&rt, &right, |l: &(i64, i64)| l.0, |r: &(i64, i64)| r.0);

        left.insert(&rt, (1, 10));
        right.insert(&rt, (1, 100));
        let _ = rt.get(joined.version_node);
        assert_eq!(joined.elements().len(), 1);

        left.delete(&rt, &(1, 10));
        let _ = rt.get(joined.version_node);
        assert_eq!(joined.elements().len(), 0);
    }
}
