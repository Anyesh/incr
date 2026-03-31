use std::any::Any;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::Hash;
use std::rc::Rc;

use crate::runtime::Runtime;
use crate::types::Incr;

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
    pub elements: HashSet<T>,
    pub deltas: Vec<VersionedDelta<T>>,
    pub version: u64,
}

impl<T: Clone + Hash + Eq> CollectionLog<T> {
    pub fn new() -> Self {
        CollectionLog {
            elements: HashSet::new(),
            deltas: Vec::new(),
            version: 0,
        }
    }

    pub fn insert(&mut self, value: T) -> bool {
        if self.elements.insert(value.clone()) {
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

    pub fn delete(&mut self, value: &T) -> bool {
        if self.elements.remove(value) {
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
}

pub struct IncrCollection<T: Any + Clone + Hash + Eq + 'static> {
    pub(crate) log: Rc<RefCell<CollectionLog<T>>>,
    pub(crate) version_node: Incr<u64>,
}

impl<T: Any + Clone + Hash + Eq + 'static> IncrCollection<T> {
    pub fn insert(&self, rt: &Runtime, value: T) {
        let changed = self.log.borrow_mut().insert(value);
        if changed {
            let ver = self.log.borrow().version;
            rt.set(self.version_node, ver);
        }
    }

    pub fn delete(&self, rt: &Runtime, value: &T) {
        let changed = self.log.borrow_mut().delete(value);
        if changed {
            let ver = self.log.borrow().version;
            rt.set(self.version_node, ver);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Runtime;

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
        assert_eq!(log.elements.len(), 1);
        assert_eq!(log.version, 1);
    }

    #[test]
    fn log_delete() {
        let mut log = CollectionLog::new();
        log.insert(1_i64);
        assert!(log.delete(&1));
        assert_eq!(log.elements.len(), 0);
        assert_eq!(log.version, 2);
        assert_eq!(log.deltas.len(), 2);
    }

    #[test]
    fn log_delete_missing_is_noop() {
        let mut log: CollectionLog<i64> = CollectionLog::new();
        assert!(!log.delete(&1));
        assert_eq!(log.version, 0);
    }

    #[test]
    fn log_deltas_are_versioned() {
        let mut log = CollectionLog::new();
        log.insert(10_i64);
        log.insert(20);
        log.delete(&10);

        assert_eq!(log.deltas.len(), 3);
        assert_eq!(log.deltas[0].version, 1);
        assert_eq!(log.deltas[1].version, 2);
        assert_eq!(log.deltas[2].version, 3);
        assert!(matches!(log.deltas[0].delta, Delta::Insert(10)));
        assert!(matches!(log.deltas[1].delta, Delta::Insert(20)));
        assert!(matches!(log.deltas[2].delta, Delta::Delete(10)));
    }

    #[test]
    fn create_and_insert() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        col.insert(&rt, 1);
        col.insert(&rt, 2);
        col.insert(&rt, 3);
        assert_eq!(col.log.borrow().elements.len(), 3);
    }

    #[test]
    fn insert_bumps_graph_version() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        assert_eq!(rt.get(col.version_node), 0);
        col.insert(&rt, 1);
        assert_eq!(rt.get(col.version_node), 1);
        col.insert(&rt, 2);
        assert_eq!(rt.get(col.version_node), 2);
    }

    #[test]
    fn delete_bumps_graph_version() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        col.insert(&rt, 1);
        col.insert(&rt, 2);
        assert_eq!(rt.get(col.version_node), 2);
        col.delete(&rt, &1);
        assert_eq!(rt.get(col.version_node), 3);
    }

    #[test]
    fn duplicate_insert_no_version_bump() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        col.insert(&rt, 1);
        assert_eq!(rt.get(col.version_node), 1);
        col.insert(&rt, 1); // duplicate
        assert_eq!(rt.get(col.version_node), 1); // unchanged
    }
}
