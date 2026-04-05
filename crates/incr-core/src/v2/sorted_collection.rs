use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use super::collection::{Delta, IncrCollection};
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
            let ver = version_counter_ref.fetch_add(1, Ordering::Relaxed) + 1;
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
}
