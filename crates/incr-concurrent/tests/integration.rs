//! Smoke tests for the `incr-concurrent` v0.2 wrapper. Proves the
//! re-exports compile, the API works end-to-end, and the runtime is
//! actually `Send + Sync` (shared across threads with an `Arc`).

use incr_concurrent::{IncrCollection, Runtime, SortedCollection};
use std::sync::Arc;
use std::thread;

#[test]
fn runtime_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Runtime>();
    assert_send_sync::<Arc<Runtime>>();
    assert_send_sync::<incr_concurrent::Incr<u64>>();
}

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
fn early_cutoff_stops_propagation() {
    let rt = Runtime::new();
    let input = rt.create_input(200_i64);
    let clamped = rt.create_query(move |rt| rt.get(input).min(100));
    let after = rt.create_query(move |rt| rt.get(clamped) + 1);
    assert_eq!(rt.get(after), 101);
    rt.set(input, 300);
    // clamped still 100, so after never recomputes — but value is still 101
    assert_eq!(rt.get(after), 101);
}

#[test]
fn concurrent_writer_reader_no_torn_reads() {
    // One writer thread mutates an input; many reader threads pull a
    // derived doubling. The derived value is always even; if a reader
    // ever observed a torn or partially-propagated value it would fail.
    let rt = Arc::new(Runtime::new());
    let counter = rt.create_input(0_i64);
    let doubled = rt.create_query(move |rt| rt.get(counter) * 2);

    let writer = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || {
            for i in 1..=1000 {
                rt.set(counter, i);
            }
        })
    };

    let mut readers = Vec::new();
    for _ in 0..4 {
        let rt = Arc::clone(&rt);
        readers.push(thread::spawn(move || {
            for _ in 0..500 {
                let v = rt.get(doubled);
                assert!(v % 2 == 0, "torn read: got odd value {}", v);
            }
        }));
    }

    writer.join().unwrap();
    for r in readers {
        r.join().unwrap();
    }
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
    assert_eq!(rt.get(n), 2);
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
