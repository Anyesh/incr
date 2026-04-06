use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::rc::Rc;

use crate::runtime::Runtime;
use crate::sorted_collection::{SortDelta, SortedCollection};
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
    /// Counts each element. In set mode (multiset=false), counts are always 0 or 1.
    /// In multiset mode (multiset=true), counts can exceed 1 for duplicate values.
    pub elements: HashMap<T, usize>,
    pub deltas: Vec<VersionedDelta<T>>,
    pub version: u64,
    /// When true, allows duplicate values (reference-counted). Used by pipeline
    /// operators like `map` whose outputs may collide even when inputs are distinct.
    multiset: bool,
}

impl<T: Clone + Hash + Eq> CollectionLog<T> {
    /// Create a set-mode log: duplicate inserts are silently ignored.
    pub fn new() -> Self {
        CollectionLog {
            elements: HashMap::new(),
            deltas: Vec::new(),
            version: 0,
            multiset: false,
        }
    }

    /// Create a multiset-mode log: duplicate inserts increment a reference count
    /// and fire a delta each time; deletes decrement and fire a delta only when
    /// the count reaches zero.
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

    /// Returns the set of distinct elements present (regardless of multiplicity).
    pub fn distinct_elements(&self) -> HashSet<T> {
        self.elements.keys().cloned().collect()
    }

    /// Returns all elements expanded by multiplicity as a Vec.
    /// For set-mode logs (all counts 1), this is equivalent to iterating the set.
    /// For multiset-mode logs, duplicate values appear multiple times.
    pub fn elements_vec(&self) -> Vec<T> {
        self.elements
            .iter()
            .flat_map(|(v, &count)| std::iter::repeat(v.clone()).take(count))
            .collect()
    }
}

pub struct IncrCollection<T: Any + Clone + Hash + Eq + 'static> {
    pub(crate) log: Rc<RefCell<CollectionLog<T>>>,
    pub(crate) version_node: Incr<u64>,
}

impl<T: Any + Clone + Hash + Eq + 'static> IncrCollection<T> {
    pub fn version_node_id(&self) -> crate::types::NodeId {
        self.version_node.node_id()
    }

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
        let output_log = Rc::new(RefCell::new(CollectionLog::new_multiset()));
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
        self.log.borrow().distinct_elements()
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

    pub fn reduce<A, F>(&self, rt: &Runtime, fold_fn: F) -> Incr<A>
    where
        A: Any + Clone + PartialEq + 'static,
        F: Fn(&Vec<T>) -> A + 'static,
    {
        let upstream_log = self.log.clone();
        let upstream_ver = self.version_node;
        let last_idx = Rc::new(Cell::new(0_usize));

        rt.create_query(move |rt| -> A {
            let _upstream_v = rt.get(upstream_ver);

            let upstream = upstream_log.borrow();
            let start = last_idx.get();
            if start >= upstream.deltas.len() {
                // No new deltas, but we still need to return current value.
                // On first call with empty collection, fold over current elements.
                let elems = upstream.elements_vec();
                return fold_fn(&elems);
            }

            last_idx.set(upstream.deltas.len());
            let elems = upstream.elements_vec();
            fold_fn(&elems)
        })
    }

    pub fn sort_by_key<K, F>(&self, rt: &Runtime, key_fn: F) -> SortedCollection<T>
    where
        K: Ord + Clone + 'static,
        F: Fn(&T) -> K + 'static,
    {
        let upstream_log = self.log.clone();
        let upstream_ver = self.version_node;
        let last_idx = Rc::new(Cell::new(0_usize));

        // Internal state: keys vec lives inside the closure
        let keys: Rc<RefCell<Vec<K>>> = Rc::new(RefCell::new(Vec::new()));
        // Reverse lookup: value -> cached key (for delete)
        let key_cache: Rc<RefCell<HashMap<T, K>>> = Rc::new(RefCell::new(HashMap::new()));

        // Shared state: exposed to SortedCollection
        let ordered_values: Rc<RefCell<Vec<T>>> = Rc::new(RefCell::new(Vec::new()));
        let pending_deltas: Rc<RefCell<Vec<SortDelta<T>>>> = Rc::new(RefCell::new(Vec::new()));

        let keys_ref = keys.clone();
        let key_cache_ref = key_cache.clone();
        let ordered_values_ref = ordered_values.clone();
        let pending_deltas_ref = pending_deltas.clone();

        let version_counter: Rc<Cell<u64>> = Rc::new(Cell::new(0));
        let version_counter_ref = version_counter.clone();

        let version_node = rt.create_query(move |rt| -> u64 {
            let _upstream_v = rt.get(upstream_ver);

            let upstream = upstream_log.borrow();
            let start = last_idx.get();
            if start >= upstream.deltas.len() {
                return version_counter_ref.get();
            }

            let mut ks = keys_ref.borrow_mut();
            let mut kc = key_cache_ref.borrow_mut();
            let mut vals = ordered_values_ref.borrow_mut();
            let mut deltas = pending_deltas_ref.borrow_mut();

            for vd in &upstream.deltas[start..] {
                match &vd.delta {
                    Delta::Insert(x) => {
                        let k = key_fn(x);
                        let pos = ks
                            .binary_search_by(|probe| probe.cmp(&k))
                            .unwrap_or_else(|pos| pos);
                        ks.insert(pos, k.clone());
                        vals.insert(pos, x.clone());
                        kc.insert(x.clone(), k);
                        deltas.push(SortDelta::Inserted {
                            index: pos,
                            value: x.clone(),
                        });
                    }
                    Delta::Delete(x) => {
                        if let Some(k) = kc.remove(x) {
                            // Find the position: binary search for the key, then linear scan
                            // for the exact value in case of duplicate keys
                            let start_pos = ks
                                .binary_search_by(|probe| probe.cmp(&k))
                                .unwrap_or_else(|pos| pos);
                            let mut pos = start_pos;
                            while pos < vals.len() && ks[pos] == k {
                                if vals[pos] == *x {
                                    break;
                                }
                                pos += 1;
                            }
                            if pos < vals.len() && vals[pos] == *x {
                                ks.remove(pos);
                                vals.remove(pos);
                                deltas.push(SortDelta::Removed {
                                    index: pos,
                                    value: x.clone(),
                                });
                            }
                        }
                    }
                }
            }

            last_idx.set(upstream.deltas.len());
            let ver = version_counter_ref.get() + 1;
            version_counter_ref.set(ver);
            ver
        });

        SortedCollection {
            ordered_values,
            pending_deltas,
            version_node,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

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
        let elements: Vec<i64> = doubled.log.borrow().elements_vec();
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
        assert!(doubled.log.borrow().elements.contains_key(&4));
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
        let elements: Vec<i64> = doubled.log.borrow().elements_vec();
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

    // ── reduce tests ────────────────────────────────────────────────────────

    #[test]
    fn reduce_sum() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sum = col.reduce(&rt, |elements| -> i64 { elements.iter().sum() });

        assert_eq!(rt.get(sum), 0); // empty collection
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

        col.insert(&rt, 6);
        assert_eq!(rt.get(sum), 12);

        col.delete(&rt, &2);
        assert_eq!(rt.get(sum), 10);
    }

    #[test]
    fn reduce_early_cutoff() {
        use std::cell::Cell as StdCell;
        use std::rc::Rc as StdRc;

        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let max = col.reduce(&rt, |elements| -> Option<i64> {
            elements.iter().copied().max()
        });

        let downstream_count = StdRc::new(StdCell::new(0_u32));
        let dc = downstream_count.clone();
        let label = rt.create_query(move |rt| {
            dc.set(dc.get() + 1);
            format!("max={:?}", rt.get(max))
        });

        col.insert(&rt, 5);
        assert_eq!(rt.get(label), "max=Some(5)");
        assert_eq!(downstream_count.get(), 1);

        col.insert(&rt, 3); // doesn't change max
        assert_eq!(rt.get(label), "max=Some(5)");
        assert_eq!(downstream_count.get(), 1); // early cutoff!
    }
}
