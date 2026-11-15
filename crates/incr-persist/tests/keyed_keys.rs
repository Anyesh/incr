//! `PersistedKeyed::get_keys` depends on membership: a persisted write
//! reruns a listing query when it inserts or removes a key, and not when
//! it replaces a key's row, which the log records as one `Replace`.

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

use incr_persist::{MemoryBackend, Options, PersistedKeyed, Shared, Store};

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
