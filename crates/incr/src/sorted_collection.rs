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

impl<T: Any + Clone + Hash + Eq + 'static> SortedCollection<T> {
    pub fn window(&self, rt: &Runtime, size: usize) -> IncrCollection<Vec<T>>
    where
        T: Eq + Hash,
    {
        let ordered_values = self.ordered_values.clone();
        let sorted_ver = self.version_node;
        let output_log = Rc::new(RefCell::new(CollectionLog::<Vec<T>>::new()));
        let output_log_ref = output_log.clone();
        let prev_windows: Rc<RefCell<Vec<Vec<T>>>> = Rc::new(RefCell::new(Vec::new()));
        let prev_ref = prev_windows.clone();

        let version_node = rt.create_query(move |rt| -> u64 {
            let _sv = rt.get(sorted_ver);

            let vals = ordered_values.borrow();
            let mut output = output_log_ref.borrow_mut();
            let mut prev = prev_ref.borrow_mut();

            // Remove old windows
            for w in prev.drain(..) {
                output.delete(&w);
            }

            // Generate new windows
            if vals.len() >= size {
                for i in 0..=(vals.len() - size) {
                    let w: Vec<T> = vals[i..i + size].to_vec();
                    output.insert(w.clone());
                    prev.push(w);
                }
            }

            output.version
        });

        IncrCollection {
            log: output_log,
            version_node,
        }
    }

    pub fn pairwise(&self, rt: &Runtime) -> IncrCollection<(T, T)> {
        let sorted_deltas = self.pending_deltas.clone();
        let sorted_ver = self.version_node;
        let last_delta_idx = Rc::new(Cell::new(0_usize));

        // Shadow of the sorted values, maintained in lockstep by replaying SortDeltas
        let shadow: Rc<RefCell<Vec<T>>> = Rc::new(RefCell::new(Vec::new()));
        let shadow_ref = shadow.clone();

        let output_log = Rc::new(RefCell::new(CollectionLog::new()));
        let output_log_ref = output_log.clone();

        let version_node = rt.create_query(move |rt| -> u64 {
            let _sorted_v = rt.get(sorted_ver);

            let deltas = sorted_deltas.borrow();
            let start = last_delta_idx.get();
            if start >= deltas.len() {
                return output_log_ref.borrow().version;
            }

            let mut shadow = shadow_ref.borrow_mut();
            let mut output = output_log_ref.borrow_mut();

            for delta in &deltas[start..] {
                match delta {
                    SortDelta::Inserted { index, value } => {
                        let i = *index;
                        let n_before = shadow.len();

                        if n_before == 0 {
                            // First element, no pairs
                        } else if i == 0 {
                            // Inserting at front: new pair (new, old_first)
                            output.insert((value.clone(), shadow[0].clone()));
                        } else if i == n_before {
                            // Inserting at end: new pair (old_last, new)
                            output.insert((shadow[n_before - 1].clone(), value.clone()));
                        } else {
                            // Inserting in middle: remove old pair, add two new
                            let left = shadow[i - 1].clone();
                            let right = shadow[i].clone();
                            output.delete(&(left.clone(), right.clone()));
                            output.insert((left, value.clone()));
                            output.insert((value.clone(), right));
                        }

                        shadow.insert(i, value.clone());
                    }
                    SortDelta::Removed { index, value } => {
                        let i = *index;
                        shadow.remove(i);
                        let n_after = shadow.len();

                        if n_after == 0 {
                            // Was the only element; no pairs existed, nothing to remove
                        } else if i == 0 {
                            // Removed from front: delete pair (removed, new_first)
                            output.delete(&(value.clone(), shadow[0].clone()));
                        } else if i == n_after {
                            // Removed from end: delete pair (new_last, removed)
                            output.delete(&(shadow[n_after - 1].clone(), value.clone()));
                        } else {
                            // Removed from middle: delete two pairs, restore neighbor pair
                            let left = shadow[i - 1].clone();
                            let right = shadow[i].clone();
                            output.delete(&(left.clone(), value.clone()));
                            output.delete(&(value.clone(), right.clone()));
                            output.insert((left, right));
                        }
                    }
                }
            }

            last_delta_idx.set(deltas.len());
            output.version
        });

        IncrCollection {
            log: output_log,
            version_node,
        }
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
        // Sorted by key (.1) ascending: alice=10, carol=20, bob=30
        assert_eq!(names, vec!["alice", "carol", "bob"]);
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

    // ── pairwise tests ──────────────────────────────────────────────────────

    #[test]
    fn pairwise_basic() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let pairs = sorted.pairwise(&rt);

        col.insert(&rt, 10);
        col.insert(&rt, 20);
        col.insert(&rt, 30);

        let _ = rt.get(pairs.version_node);
        let elems = pairs.elements();
        assert_eq!(elems.len(), 2);
        assert!(elems.contains(&(10, 20)));
        assert!(elems.contains(&(20, 30)));
    }

    #[test]
    fn pairwise_single_element() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let pairs = sorted.pairwise(&rt);

        col.insert(&rt, 10);
        let _ = rt.get(pairs.version_node);
        assert_eq!(pairs.elements().len(), 0);
    }

    #[test]
    fn pairwise_empty() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let pairs = sorted.pairwise(&rt);

        let _ = rt.get(pairs.version_node);
        assert_eq!(pairs.elements().len(), 0);
    }

    #[test]
    fn pairwise_insert_middle() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let pairs = sorted.pairwise(&rt);

        col.insert(&rt, 10);
        col.insert(&rt, 30);
        let _ = rt.get(pairs.version_node);
        assert!(pairs.elements().contains(&(10, 30)));

        col.insert(&rt, 20);
        let _ = rt.get(pairs.version_node);
        let elems = pairs.elements();
        assert_eq!(elems.len(), 2);
        assert!(elems.contains(&(10, 20)));
        assert!(elems.contains(&(20, 30)));
        assert!(!elems.contains(&(10, 30)));
    }

    #[test]
    fn pairwise_delete_middle() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let pairs = sorted.pairwise(&rt);

        col.insert(&rt, 10);
        col.insert(&rt, 20);
        col.insert(&rt, 30);
        let _ = rt.get(pairs.version_node);

        col.delete(&rt, &20);
        let _ = rt.get(pairs.version_node);
        let elems = pairs.elements();
        assert_eq!(elems.len(), 1);
        assert!(elems.contains(&(10, 30)));
    }

    #[test]
    fn pairwise_delete_first() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let pairs = sorted.pairwise(&rt);

        col.insert(&rt, 10);
        col.insert(&rt, 20);
        col.insert(&rt, 30);
        let _ = rt.get(pairs.version_node);

        col.delete(&rt, &10);
        let _ = rt.get(pairs.version_node);
        let elems = pairs.elements();
        assert_eq!(elems.len(), 1);
        assert!(elems.contains(&(20, 30)));
    }

    #[test]
    fn pairwise_delete_last() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let pairs = sorted.pairwise(&rt);

        col.insert(&rt, 10);
        col.insert(&rt, 20);
        col.insert(&rt, 30);
        let _ = rt.get(pairs.version_node);

        col.delete(&rt, &30);
        let _ = rt.get(pairs.version_node);
        let elems = pairs.elements();
        assert_eq!(elems.len(), 1);
        assert!(elems.contains(&(10, 20)));
    }

    #[test]
    fn pairwise_delete_to_empty() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let pairs = sorted.pairwise(&rt);

        col.insert(&rt, 10);
        col.insert(&rt, 20);
        let _ = rt.get(pairs.version_node);
        assert_eq!(pairs.elements().len(), 1);

        col.delete(&rt, &10);
        col.delete(&rt, &20);
        let _ = rt.get(pairs.version_node);
        assert_eq!(pairs.elements().len(), 0);
    }

    #[test]
    fn pairwise_insert_at_front() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let pairs = sorted.pairwise(&rt);

        col.insert(&rt, 20);
        col.insert(&rt, 30);
        let _ = rt.get(pairs.version_node);
        assert!(pairs.elements().contains(&(20, 30)));

        col.insert(&rt, 10);
        let _ = rt.get(pairs.version_node);
        let elems = pairs.elements();
        assert_eq!(elems.len(), 2);
        assert!(elems.contains(&(10, 20)));
        assert!(elems.contains(&(20, 30)));
    }

    #[test]
    fn pairwise_insert_at_end() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let pairs = sorted.pairwise(&rt);

        col.insert(&rt, 10);
        col.insert(&rt, 20);
        let _ = rt.get(pairs.version_node);
        assert!(pairs.elements().contains(&(10, 20)));

        col.insert(&rt, 30);
        let _ = rt.get(pairs.version_node);
        let elems = pairs.elements();
        assert_eq!(elems.len(), 2);
        assert!(elems.contains(&(10, 20)));
        assert!(elems.contains(&(20, 30)));
    }

    // ── window tests ─────────────────────────────────────────────────────────

    #[test]
    fn window_basic() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let wins = sorted.window(&rt, 3);
        col.insert(&rt, 10);
        col.insert(&rt, 20);
        col.insert(&rt, 30);
        col.insert(&rt, 40);
        col.insert(&rt, 50);
        let _ = rt.get(wins.version_node);
        let elems = wins.elements();
        assert_eq!(elems.len(), 3);
        assert!(elems.contains(&vec![10, 20, 30]));
        assert!(elems.contains(&vec![20, 30, 40]));
        assert!(elems.contains(&vec![30, 40, 50]));
    }

    #[test]
    fn window_smaller_than_size() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let wins = sorted.window(&rt, 3);
        col.insert(&rt, 10);
        col.insert(&rt, 20);
        let _ = rt.get(wins.version_node);
        assert_eq!(wins.elements().len(), 0);
    }
}
