use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use super::collection::{CollectionLog, Delta, IncrCollection};
use super::handle::Incr;
use super::runtime::Runtime;

#[derive(Clone, Debug)]
pub enum SortDelta<T> {
    Inserted { index: usize, value: T },
    Removed { index: usize, value: T },
}

pub struct SortedCollection<T: Clone + Send + Sync + 'static> {
    pub(crate) ordered_values: Arc<RwLock<Vec<T>>>,
    pub(crate) pending_deltas: Arc<RwLock<Vec<SortDelta<T>>>>,
    pub(crate) version_node: Incr<u64>,
}

impl<T: Clone + Send + Sync + 'static> SortedCollection<T> {
    pub fn entries(&self) -> Vec<T> {
        self.ordered_values.read().unwrap().clone()
    }

    pub fn version_node(&self) -> Incr<u64> {
        self.version_node
    }
}

impl<T> IncrCollection<T>
where
    T: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
{
    pub fn sort_by_key<K, F>(&self, rt: &Runtime, key_fn: F) -> SortedCollection<T>
    where
        K: Ord + Clone + Send + Sync + 'static,
        F: Fn(&T) -> K + Send + Sync + 'static,
    {
        let upstream_log = self.log.clone();
        let upstream_ver = self.version_node;
        let last_idx = Arc::new(AtomicUsize::new(0));

        let keys: Arc<RwLock<Vec<K>>> = Arc::new(RwLock::new(Vec::new()));
        let key_cache: Arc<RwLock<HashMap<T, K>>> = Arc::new(RwLock::new(HashMap::new()));

        let ordered_values: Arc<RwLock<Vec<T>>> = Arc::new(RwLock::new(Vec::new()));
        let pending_deltas: Arc<RwLock<Vec<SortDelta<T>>>> = Arc::new(RwLock::new(Vec::new()));

        let keys_ref = keys.clone();
        let key_cache_ref = key_cache.clone();
        let ordered_values_ref = ordered_values.clone();
        let pending_deltas_ref = pending_deltas.clone();

        let version_counter = Arc::new(AtomicU64::new(0));
        let version_counter_ref = version_counter.clone();

        let version_node = rt.create_query(move |rt| -> u64 {
            let _upstream_v = rt.get(upstream_ver);

            let upstream = upstream_log.read().unwrap();
            let start = last_idx.load(Ordering::Relaxed);
            if start >= upstream.deltas.len() {
                return version_counter_ref.load(Ordering::Relaxed);
            }

            let mut ks = keys_ref.write().unwrap();
            let mut kc = key_cache_ref.write().unwrap();
            let mut vals = ordered_values_ref.write().unwrap();
            let mut deltas = pending_deltas_ref.write().unwrap();

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

            last_idx.store(upstream.deltas.len(), Ordering::Relaxed);
            version_counter_ref.fetch_add(1, Ordering::Relaxed) + 1
        });

        SortedCollection {
            ordered_values,
            pending_deltas,
            version_node,
        }
    }
}

impl<T> SortedCollection<T>
where
    T: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
{
    pub fn window(&self, rt: &Runtime, size: usize) -> IncrCollection<Vec<T>>
    where
        T: Eq + Hash,
    {
        let ordered_values = self.ordered_values.clone();
        let sorted_ver = self.version_node;
        let output_log = Arc::new(RwLock::new(CollectionLog::<Vec<T>>::new()));
        let output_log_ref = output_log.clone();
        let prev_windows: Arc<RwLock<Vec<Vec<T>>>> = Arc::new(RwLock::new(Vec::new()));
        let prev_ref = prev_windows.clone();

        let version_node = rt.create_query(move |rt| -> u64 {
            let _sv = rt.get(sorted_ver);

            let vals = ordered_values.read().unwrap();
            let mut output = output_log_ref.write().unwrap();
            let mut prev = prev_ref.write().unwrap();

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
        let last_delta_idx = Arc::new(AtomicUsize::new(0));

        let shadow: Arc<RwLock<Vec<T>>> = Arc::new(RwLock::new(Vec::new()));
        let shadow_ref = shadow.clone();

        let output_log = Arc::new(RwLock::new(CollectionLog::new()));
        let output_log_ref = output_log.clone();

        let version_node = rt.create_query(move |rt| -> u64 {
            let _sorted_v = rt.get(sorted_ver);

            let deltas = sorted_deltas.read().unwrap();
            let start = last_delta_idx.load(Ordering::Relaxed);
            if start >= deltas.len() {
                return output_log_ref.read().unwrap().version;
            }

            let mut shadow = shadow_ref.write().unwrap();
            let mut output = output_log_ref.write().unwrap();

            for delta in &deltas[start..] {
                match delta {
                    SortDelta::Inserted { index, value } => {
                        let i = *index;
                        let n_before = shadow.len();

                        if n_before == 0 {
                            // first element, no pairs
                        } else if i == 0 {
                            output.insert((value.clone(), shadow[0].clone()));
                        } else if i == n_before {
                            output.insert((shadow[n_before - 1].clone(), value.clone()));
                        } else {
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
                            // was the only element
                        } else if i == 0 {
                            output.delete(&(value.clone(), shadow[0].clone()));
                        } else if i == n_after {
                            output.delete(&(shadow[n_after - 1].clone(), value.clone()));
                        } else {
                            let left = shadow[i - 1].clone();
                            let right = shadow[i].clone();
                            output.delete(&(left.clone(), value.clone()));
                            output.delete(&(value.clone(), right.clone()));
                            output.insert((left, right));
                        }
                    }
                }
            }

            last_delta_idx.store(deltas.len(), Ordering::Relaxed);
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
        assert_eq!(names, vec!["alice", "carol", "bob"]);
    }

    #[test]
    fn sort_empty_collection() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);

        let _ = rt.get(sorted.version_node);
        assert_eq!(sorted.entries(), Vec::<i64>::new());
    }

    // ── window tests ────────────────────────────────────────────────────────

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

    #[test]
    fn window_exact_size() {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let wins = sorted.window(&rt, 3);

        col.insert(&rt, 10);
        col.insert(&rt, 20);
        col.insert(&rt, 30);

        let _ = rt.get(wins.version_node);
        let elems = wins.elements();
        assert_eq!(elems.len(), 1);
        assert!(elems.contains(&vec![10, 20, 30]));
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
}
