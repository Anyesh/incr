//! Smoke tests for the `incr-compute` v0.2 wrapper. Proves the
//! re-exports compile and the basic API works end-to-end against
//! `incr_core::Runtime<Local>`.

use incr_compute::{IncrCollection, Runtime, SortedCollection};

#[test]
fn function_dag_chain_propagates() {
    let rt = Runtime::new();
    let a = rt.create_input(1_i64);
    let b = rt.create_query(move |rt| rt.get(a) + 1);
    let c = rt.create_query(move |rt| rt.get(b) * 2);
    assert_eq!(rt.get(c), 4);
    rt.set(a, 10);
    assert_eq!(rt.get(c), 22);
}

#[test]
fn diamond_with_early_cutoff() {
    let rt = Runtime::new();
    let a = rt.create_input(1_i64);
    let b = rt.create_query(move |rt| rt.get(a) + 10);
    let c = rt.create_query(move |rt| rt.get(a) + 100);
    let d = rt.create_query(move |rt| rt.get(b) + rt.get(c));
    assert_eq!(rt.get(d), 112);
    rt.set(a, 2);
    assert_eq!(rt.get(d), 114);
}

#[test]
fn collection_filter_map_reduce_pipeline() {
    let rt = Runtime::new();
    let scores: IncrCollection<i64> = rt.create_collection();
    let passing = scores.filter(&rt, |s| *s >= 50);
    let curved = passing.map(&rt, |s| s + 10);
    let total = curved.reduce(&rt, |xs| xs.iter().sum::<i64>());
    scores.insert(&rt, 80);
    scores.insert(&rt, 95);
    scores.insert(&rt, 60);
    scores.insert(&rt, 42);
    assert_eq!(rt.get(total), 265);
}

#[test]
fn sort_pairwise_count() {
    let rt = Runtime::new();
    let c: IncrCollection<i64> = rt.create_collection();
    let sorted: SortedCollection<i64, i64> = c.sort_by_key(&rt, |x| *x);
    let pairs = sorted.pairwise(&rt);
    c.insert(&rt, 5);
    c.insert(&rt, 1);
    c.insert(&rt, 3);
    let n = pairs.count(&rt);
    assert_eq!(rt.get(n), 2); // (1,3), (3,5)
}

#[test]
fn group_by_two_buckets() {
    let rt = Runtime::new();
    let c: IncrCollection<i64> = rt.create_collection();
    let groups = c.group_by(&rt, |x| x % 2);
    for i in 1..=6_i64 {
        c.insert(&rt, i);
    }
    let _ = rt.get(groups.version_node());
    assert_eq!(groups.group_count(), 2);
}

#[test]
fn join_two_collections() {
    let rt = Runtime::new();
    let left: IncrCollection<(i64, &'static str)> = rt.create_collection();
    let right: IncrCollection<(i64, i64)> = rt.create_collection();
    let j = left.join(&rt, &right, |l| l.0, |r| r.0);
    left.insert(&rt, (1, "alice"));
    right.insert(&rt, (1, 100));
    right.insert(&rt, (1, 200));
    let n = j.count(&rt);
    assert_eq!(rt.get(n), 2);
}

#[test]
fn graph_snapshot_returns_dependencies() {
    let rt = Runtime::new();
    let a = rt.create_input(1_i64);
    let _b = rt.create_query(move |rt| rt.get(a) + 1);
    // Force the query to run so its deps are recorded.
    let _ = rt.get(_b);
    let snap = rt.graph_snapshot();
    assert_eq!(snap.len(), 2);
    // The query (slot 1) should depend on the input (slot 0).
    assert_eq!(snap[1].dependencies.len(), 1);
}
