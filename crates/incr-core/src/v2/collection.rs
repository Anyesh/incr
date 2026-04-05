use std::collections::{HashMap, HashSet};
use std::hash::Hash;
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
}
