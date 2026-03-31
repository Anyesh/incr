use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
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

    pub fn filter<F>(&self, rt: &Runtime, predicate: F) -> IncrCollection<T>
    where
        F: Fn(&T) -> bool + 'static,
    {
        let upstream_log = self.log.clone();
        let output_log = Rc::new(RefCell::new(CollectionLog::new()));
        let output_log_ref = output_log.clone();
        let last_idx = Rc::new(Cell::new(0_usize));
        let upstream_ver = self.version_node;

        let version_node = rt.create_query(move |rt| -> u64 {
            let _upstream_v = rt.get(upstream_ver);

            let upstream = upstream_log.borrow();
            let start = last_idx.get();
            if start >= upstream.deltas.len() {
                return output_log_ref.borrow().version;
            }

            let mut output = output_log_ref.borrow_mut();

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

            last_idx.set(upstream.deltas.len());
            output.version
        });

        IncrCollection {
            log: output_log,
            version_node,
        }
    }

    pub fn map<U, F>(&self, rt: &Runtime, f: F) -> IncrCollection<U>
    where
        U: Any + Clone + Hash + Eq + 'static,
        F: Fn(&T) -> U + 'static,
    {
        let upstream_log = self.log.clone();
        let output_log = Rc::new(RefCell::new(CollectionLog::new()));
        let output_log_ref = output_log.clone();
        let last_idx = Rc::new(Cell::new(0_usize));
        let mapping: Rc<RefCell<HashMap<T, U>>> = Rc::new(RefCell::new(HashMap::new()));
        let mapping_ref = mapping.clone();
        let upstream_ver = self.version_node;

        let version_node = rt.create_query(move |rt| -> u64 {
            let _upstream_v = rt.get(upstream_ver);

            let upstream = upstream_log.borrow();
            let start = last_idx.get();
            if start >= upstream.deltas.len() {
                return output_log_ref.borrow().version;
            }

            let mut output = output_log_ref.borrow_mut();
            let mut map_state = mapping_ref.borrow_mut();

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

            last_idx.set(upstream.deltas.len());
            output.version
        });

        IncrCollection {
            log: output_log,
            version_node,
        }
    }

    pub fn elements(&self) -> std::collections::HashSet<T> {
        self.log.borrow().elements.clone()
    }

    pub fn count(&self, rt: &Runtime) -> Incr<usize> {
        let upstream_log = self.log.clone();
        let upstream_ver = self.version_node;
        let current_count = Rc::new(Cell::new(0_usize));
        let count_ref = current_count.clone();
        let last_idx = Rc::new(Cell::new(0_usize));

        rt.create_query(move |rt| -> usize {
            let _upstream_v = rt.get(upstream_ver);

            let upstream = upstream_log.borrow();
            let start = last_idx.get();
            if start >= upstream.deltas.len() {
                return count_ref.get();
            }

            let mut count = count_ref.get();

            for vd in &upstream.deltas[start..] {
                match &vd.delta {
                    Delta::Insert(_) => count += 1,
                    Delta::Delete(_) => count -= 1,
                }
            }

            last_idx.set(upstream.deltas.len());
            count_ref.set(count);
            count
        })
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
        assert_eq!(evens.log.borrow().elements.len(), 2);
    }

    #[test]
    fn filter_incremental_insert() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let evens = col.filter(&rt, |x| x % 2 == 0);

        col.insert(&rt, 2);
        let _ = rt.get(evens.version_node);
        assert_eq!(evens.log.borrow().elements.len(), 1);

        col.insert(&rt, 4);
        let _ = rt.get(evens.version_node);
        assert_eq!(evens.log.borrow().elements.len(), 2);

        col.insert(&rt, 3);
        let _ = rt.get(evens.version_node);
        assert_eq!(evens.log.borrow().elements.len(), 2);
    }

    #[test]
    fn filter_incremental_delete() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let evens = col.filter(&rt, |x| x % 2 == 0);

        col.insert(&rt, 2);
        col.insert(&rt, 4);
        let _ = rt.get(evens.version_node);
        assert_eq!(evens.log.borrow().elements.len(), 2);

        col.delete(&rt, &2);
        let _ = rt.get(evens.version_node);
        assert_eq!(evens.log.borrow().elements.len(), 1);
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
        assert_eq!(small.log.borrow().elements.len(), 2);
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
        let elements: Vec<i64> = {
            let log = doubled.log.borrow();
            log.elements.iter().cloned().collect()
        };
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
        assert_eq!(doubled.log.borrow().elements.len(), 2);

        col.delete(&rt, &1);
        let _ = rt.get(doubled.version_node);
        assert_eq!(doubled.log.borrow().elements.len(), 1);
        assert!(doubled.log.borrow().elements.contains(&4));
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
        let elements: Vec<i64> = {
            let log = doubled.log.borrow();
            log.elements.iter().cloned().collect()
        };
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

        col.delete(&rt, &2);
        assert_eq!(rt.get(count), 2);
    }

    #[test]
    fn count_early_cutoff() {
        use std::cell::Cell as StdCell;
        use std::rc::Rc as StdRc;

        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let evens = col.filter(&rt, |x| x % 2 == 0);
        let count = evens.count(&rt);

        let downstream_count = StdRc::new(StdCell::new(0_u32));
        let dc = downstream_count.clone();
        let label = rt.create_query(move |rt| {
            dc.set(dc.get() + 1);
            format!("{} evens", rt.get(count))
        });

        col.insert(&rt, 2);
        assert_eq!(rt.get(label), "1 evens");
        assert_eq!(downstream_count.get(), 1);

        col.insert(&rt, 3); // odd — count unchanged
        assert_eq!(rt.get(label), "1 evens");
        assert_eq!(downstream_count.get(), 1); // early cutoff!
    }
}
