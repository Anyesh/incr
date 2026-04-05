use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};
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
}
