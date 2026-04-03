use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

use crate::collection::{CollectionLog, Delta, IncrCollection, VersionedDelta};
use crate::runtime::Runtime;
use crate::types::Incr;

#[derive(Clone, Debug)]
pub enum SortDelta<T> {
    Inserted { index: usize, value: T },
    Removed { index: usize, value: T },
}

pub struct SortedCollection<T: Clone + 'static> {
    pub(crate) ordered_values: Rc<RefCell<Vec<T>>>,
    pub(crate) pending_deltas: Rc<RefCell<Vec<SortDelta<T>>>>,
    pub(crate) version_node: Incr<u64>,
}

impl<T: Clone + 'static> SortedCollection<T> {
    /// Get a snapshot of the current sorted order.
    pub fn entries(&self) -> Vec<T> {
        self.ordered_values.borrow().clone()
    }

    pub fn version_node_id(&self) -> crate::types::NodeId {
        self.version_node.node_id()
    }

    pub fn version_node(&self) -> Incr<u64> {
        self.version_node
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Runtime;

    #[test]
    fn sort_basic_ordering() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);

        col.insert(&rt, 30);
        col.insert(&rt, 10);
        col.insert(&rt, 20);

        let _ = rt.get(sorted.version_node);
        assert_eq!(sorted.entries(), vec![10, 20, 30]);
    }

    #[test]
    fn sort_insert_maintains_order() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);

        col.insert(&rt, 10);
        col.insert(&rt, 30);
        let _ = rt.get(sorted.version_node);
        assert_eq!(sorted.entries(), vec![10, 30]);

        col.insert(&rt, 20);
        let _ = rt.get(sorted.version_node);
        assert_eq!(sorted.entries(), vec![10, 20, 30]);
    }

    #[test]
    fn sort_delete_maintains_order() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);

        col.insert(&rt, 10);
        col.insert(&rt, 20);
        col.insert(&rt, 30);
        let _ = rt.get(sorted.version_node);
        assert_eq!(sorted.entries(), vec![10, 20, 30]);

        col.delete(&rt, &20);
        let _ = rt.get(sorted.version_node);
        assert_eq!(sorted.entries(), vec![10, 30]);
    }

    #[test]
    fn sort_by_custom_key() {
        let rt = Runtime::new();
        let col = rt.create_collection::<(String, i64)>();
        let sorted = col.sort_by_key(&rt, |x: &(String, i64)| x.1);

        col.insert(&rt, ("bob".to_string(), 30));
        col.insert(&rt, ("alice".to_string(), 10));
        col.insert(&rt, ("carol".to_string(), 20));

        let _ = rt.get(sorted.version_node);
        let names: Vec<String> = sorted.entries().into_iter().map(|e| e.0).collect();
        assert_eq!(names, vec!["alice", "bob", "carol"]);
    }

    #[test]
    fn sort_duplicate_keys() {
        let rt = Runtime::new();
        let col = rt.create_collection::<(String, i64)>();
        let sorted = col.sort_by_key(&rt, |x: &(String, i64)| x.1);

        col.insert(&rt, ("a".to_string(), 10));
        col.insert(&rt, ("b".to_string(), 10));
        col.insert(&rt, ("c".to_string(), 20));

        let _ = rt.get(sorted.version_node);
        let entries = sorted.entries();
        assert_eq!(entries[2].1, 20);
        assert_eq!(entries[0].1, 10);
        assert_eq!(entries[1].1, 10);
    }

    #[test]
    fn sort_empty_collection() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);

        let _ = rt.get(sorted.version_node);
        assert_eq!(sorted.entries(), Vec::<i64>::new());
    }
}
