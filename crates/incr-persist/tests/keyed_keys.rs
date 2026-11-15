//! `PersistedKeyed::get_keys` depends on membership: a persisted write
//! reruns a listing query when it inserts or removes a key, and not when
//! it replaces a key's row, which the log records as one `Replace`.

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

use incr_persist::{Backend, MemoryBackend, Options, PersistedKeyed, Shared, Store};

type Rows = PersistedKeyed<u32, String, Shared>;

fn lister(
    store: &Store<Shared, MemoryBackend>,
    rows: &Rows,
) -> (incr_core::Incr<Vec<u32>>, Arc<AtomicUsize>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let (counted, read) = (Arc::clone(&runs), rows.clone());
    let query = store.runtime().create_query(move |rt| {
        counted.fetch_add(1, SeqCst);
        let mut keys = read.get_keys(rt);
        keys.sort_unstable();
        keys
    });
    (query, runs)
}

#[test]
fn a_listing_reruns_on_insert_and_remove_and_not_on_a_row_replacement() {
    let store = Store::<Shared, _>::open(MemoryBackend::new(), Options::default()).unwrap();
    let rows: Rows = store.keyed_collection("rows").unwrap();
    store.upsert(&rows, 1, "a".into()).unwrap();
    let (query, runs) = lister(&store, &rows);
    let rt = store.runtime();
    assert_eq!(rt.get(query), vec![1]);

    store.upsert(&rows, 1, "b".into()).unwrap();
    assert_eq!(rt.get(query), vec![1]);
    assert_eq!(runs.load(SeqCst), 1, "a row replacement");

    store.upsert(&rows, 2, "c".into()).unwrap();
    assert_eq!(rt.get(query), vec![1, 2]);
    assert_eq!(runs.load(SeqCst), 2, "an insert");

    store.remove(&rows, &2).unwrap();
    assert_eq!(rt.get(query), vec![1]);
    assert_eq!(runs.load(SeqCst), 3, "a remove");

    store
        .reconcile(&rows, [(1, "d".to_owned()), (3, "e".to_owned())])
        .unwrap();
    assert_eq!(rt.get(query), vec![1, 3]);
    assert_eq!(runs.load(SeqCst), 4, "a reconcile that inserts and removes");

    store
        .reconcile(&rows, [(1, "f".to_owned()), (3, "g".to_owned())])
        .unwrap();
    assert_eq!(rt.get(query), vec![1, 3]);
    assert_eq!(runs.load(SeqCst), 4, "a reconcile that replaces rows");
}

fn partition_reader(
    store: &Store<Shared, MemoryBackend>,
    rows: &Rows,
    parts: incr_core::KeyPartitions<u32>,
    part: u32,
) -> (incr_core::Incr<Vec<u32>>, Arc<AtomicUsize>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let (counted, read) = (Arc::clone(&runs), rows.clone());
    let query = store.runtime().create_query(move |rt| {
        counted.fetch_add(1, SeqCst);
        let mut keys = read.get_partition(rt, &parts, &part);
        keys.sort_unstable();
        keys
    });
    (query, runs)
}

#[test]
fn a_partition_reader_reruns_for_its_own_partition_and_not_on_a_row_replacement() {
    let store = Store::<Shared, _>::open(MemoryBackend::new(), Options::default()).unwrap();
    let rows: Rows = store.keyed_collection("rows").unwrap();
    store.upsert(&rows, 1, "a".into()).unwrap();
    store.upsert(&rows, 2, "b".into()).unwrap();
    let parts = rows.key_partitions(|k: &u32| k % 2);
    let (odd, odd_runs) = partition_reader(&store, &rows, parts, 1);
    let (even, even_runs) = partition_reader(&store, &rows, parts, 0);
    let rt = store.runtime();
    assert_eq!((rt.get(odd), rt.get(even)), (vec![1], vec![2]));

    store.upsert(&rows, 1, "c".into()).unwrap();
    store.upsert(&rows, 2, "d".into()).unwrap();
    assert_eq!((rt.get(odd), rt.get(even)), (vec![1], vec![2]));
    assert_eq!((odd_runs.load(SeqCst), even_runs.load(SeqCst)), (1, 1));

    store.upsert(&rows, 3, "e".into()).unwrap();
    assert_eq!((rt.get(odd), rt.get(even)), (vec![1, 3], vec![2]));
    assert_eq!((odd_runs.load(SeqCst), even_runs.load(SeqCst)), (2, 1));

    store.remove(&rows, &2).unwrap();
    assert_eq!((rt.get(odd), rt.get(even)), (vec![1, 3], Vec::new()));
    assert_eq!((odd_runs.load(SeqCst), even_runs.load(SeqCst)), (2, 2));
}

#[test]
fn partitions_register_over_a_restored_collection() {
    let log = MemoryBackend::new();
    {
        let store = Store::<Shared, _>::open(log.clone(), Options::default()).unwrap();
        let rows: Rows = store.keyed_collection("rows").unwrap();
        for k in 0..6 {
            store.upsert(&rows, k, format!("r{k}")).unwrap();
        }
    }
    let store = Store::<Shared, _>::open(log, Options::default()).unwrap();
    let rows: Rows = store.keyed_collection("rows").unwrap();
    let parts = rows.key_partitions(|k: &u32| k % 3);
    let (zero, runs) = partition_reader(&store, &rows, parts, 0);
    let rt = store.runtime();
    assert_eq!(rt.get(zero), vec![0, 3]);
    store.upsert(&rows, 6, "x".into()).unwrap();
    assert_eq!(rt.get(zero), vec![0, 3, 6]);
    assert_eq!(runs.load(SeqCst), 2);
}

#[test]
fn a_panicking_partition_fn_refuses_the_write_before_it_is_logged() {
    let log = MemoryBackend::new();
    let store = Store::<Shared, _>::open(log.clone(), Options::default()).unwrap();
    let rows: Rows = store.keyed_collection("rows").unwrap();
    store.upsert(&rows, 1, "a".into()).unwrap();
    let parts = rows.key_partitions(|k: &u32| {
        assert_ne!(*k, 9, "no partition for 9");
        k % 2
    });
    let (odd, runs) = partition_reader(&store, &rows, parts, 1);
    let rt = store.runtime();
    assert_eq!(rt.get(odd), vec![1]);
    let applied = store.upsert(&rows, 2, "b".into()).map(drop);
    assert!(applied.is_ok());
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        store.upsert(&rows, 9, "z".into())
    }));
    assert!(refused.is_err());
    let mut records = Vec::new();
    log.read_from(incr_persist::Position::ZERO, &mut records)
        .unwrap();
    let before = records.len();
    assert_eq!(rows.keys().len(), 2);
    store.upsert(&rows, 3, "c".into()).unwrap();
    assert_eq!(rt.get(odd), vec![1, 3]);
    assert_eq!(runs.load(SeqCst), 2);
    let mut after = Vec::new();
    log.read_from(incr_persist::Position::ZERO, &mut after)
        .unwrap();
    assert_eq!(after.len(), before + 1, "only the later write was logged");
}
