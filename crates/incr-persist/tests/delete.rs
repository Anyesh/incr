use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use incr_persist::{
    Backend, Cells, Error, FileBackend, Local, MemoryBackend, Options, Position, ReadOutcome,
    Record, Shared, Store,
};

fn open<C: Cells, B: Backend>(backend: B) -> Store<C, B> {
    Store::open(backend, Options::default()).unwrap()
}

fn log_records(backend: &impl Backend) -> Vec<Record> {
    let mut out = Vec::new();
    assert_eq!(
        backend.read_from(Position::ZERO, &mut out).unwrap(),
        ReadOutcome::Records
    );
    out.into_iter().map(|(_, record)| record).collect()
}

/// Every value a store exposes for the keys the delete tests use, read
/// through queries so that recomputation is part of what is compared.
#[derive(Debug, PartialEq)]
struct View {
    x: u32,
    doubled: u32,
    label: String,
    items: Vec<u32>,
    count: u64,
}

fn view<C: Cells, B: Backend>(store: &Store<C, B>) -> View {
    let x = store.input("x", 0u32).unwrap();
    let label = store.input("label", String::new()).unwrap();
    let items = store.collection::<u32>("items").unwrap();
    let rt = store.runtime();
    let x = x.handle();
    let doubled = rt.create_query(move |rt| rt.get(x) * 2);
    let count = items.handle().count(rt);
    let mut values = items.handle().values(rt);
    values.sort();
    View {
        x: rt.get(x),
        doubled: rt.get(doubled),
        label: rt.get(label.handle()),
        items: values,
        count: rt.get(count),
    }
}

fn live_delete_matches_restart_and_refresh<C: Cells, B: Backend>(
    writer: impl Fn() -> B,
    reader: impl Fn() -> B,
) {
    let live = {
        let store = open::<C, B>(writer());
        let x = store.input("x", 3u32).unwrap();
        let label = store.input("label", "fresh".to_string()).unwrap();
        let items = store.collection::<u32>("items").unwrap();
        let second = open::<C, B>(reader());
        let before = view(&second);

        store.set(x, 9).unwrap();
        store.set(label, "edited".to_string()).unwrap();
        for item in [1, 1, 2] {
            items.insert(&store, item).unwrap();
        }
        let rt = store.runtime();
        let doubled = {
            let x = x.handle();
            rt.create_query(move |rt| rt.get(x) * 2)
        };
        let count = items.handle().count(rt);
        assert_eq!((rt.get(doubled), rt.get(count)), (18, 3));

        store.delete(x).unwrap();
        store.delete(&items).unwrap();
        store.delete("label").unwrap();
        assert_eq!((rt.get(doubled), rt.get(count)), (6, 0));

        let live = view(&store);
        assert_eq!(
            live,
            View {
                x: 3,
                doubled: 6,
                label: "fresh".into(),
                items: vec![],
                count: 0,
            }
        );
        assert_eq!(
            before, live,
            "the second store saw the keys at their defaults"
        );
        second.refresh().unwrap();
        assert_eq!(view(&second), live, "a refreshed second store agrees");

        items.insert(&store, 4).unwrap();
        store.set(x, 5).unwrap();
        second.refresh().unwrap();
        assert_eq!(view(&second), view(&store), "the keys stay usable");
        store.delete(x).unwrap();
        store.delete(&items).unwrap();
        second.refresh().unwrap();
        assert_eq!(view(&second), live);
        live
    };
    let restarted = open::<C, B>(writer());
    assert_eq!(
        view(&restarted),
        live,
        "a restart replays to the same state"
    );
}

#[test]
fn live_state_after_delete_matches_a_restart_and_a_refreshed_store_on_memory() {
    let log = MemoryBackend::new();
    live_delete_matches_restart_and_refresh::<Local, _>(|| log.clone(), || log.clone());
    let log = MemoryBackend::new();
    live_delete_matches_restart_and_refresh::<Shared, _>(|| log.clone(), || log.read_only());
}

#[test]
fn live_state_after_delete_matches_a_restart_and_a_refreshed_store_on_file() {
    let dir = tempfile::tempdir().unwrap();
    live_delete_matches_restart_and_refresh::<Local, _>(
        || FileBackend::open(dir.path()).unwrap(),
        || FileBackend::open_read_only(dir.path()).unwrap(),
    );
}

#[cfg(feature = "sqlite")]
#[test]
fn live_state_after_delete_matches_a_restart_and_a_refreshed_store_on_sqlite() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.sqlite");
    live_delete_matches_restart_and_refresh::<Shared, _>(
        || incr_persist::SqliteBackend::open(&path).unwrap(),
        || incr_persist::SqliteBackend::open(&path).unwrap(),
    );
}

#[test]
fn a_foreign_delete_recomputes_local_dependents() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let b = open::<Local, _>(log.clone());
    let x_a = a.input("x", 1u32).unwrap();
    let items_a = a.collection::<String>("items").unwrap();
    a.set(x_a, 7).unwrap();
    items_a.insert(&a, "one".into()).unwrap();

    let x_b = b.input("x", 1u32).unwrap();
    let items_b = b.collection::<String>("items").unwrap();
    let rt = b.runtime();
    let runs = Arc::new(AtomicUsize::new(0));
    let summary = {
        let (x, version) = (x_b.handle(), items_b.handle().version_node());
        let runs = Arc::clone(&runs);
        rt.create_query(move |rt| {
            runs.fetch_add(1, Ordering::SeqCst);
            (rt.get(x), rt.get(version))
        })
    };
    let count = items_b.handle().count(rt);
    assert_eq!((rt.get(summary).0, rt.get(count)), (7, 1));
    let runs_before = runs.load(Ordering::SeqCst);

    a.delete(x_a).unwrap();
    a.delete(&items_a).unwrap();
    b.refresh().unwrap();
    assert_eq!(rt.get(summary).0, 1);
    assert_eq!(rt.get(count), 0);
    assert!(items_b.handle().values(rt).is_empty());
    assert!(
        runs.load(Ordering::SeqCst) > runs_before,
        "the query recomputed"
    );
}

#[test]
fn deleting_by_name_reaches_keys_this_store_never_declared() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let b = open::<Local, _>(log.clone());
    let k = a.input("k", 2u32).unwrap();
    a.set(k, 8).unwrap();
    b.refresh().unwrap();
    assert_eq!(b.restored_keys("").unwrap(), ["k"]);

    b.delete("k").unwrap();
    b.delete("never/known").unwrap();
    a.refresh().unwrap();
    assert_eq!(a.runtime().get(k.handle()), 2);
    assert_eq!(
        b.restored_keys("").unwrap(),
        ["k"],
        "a delete of a key the log never knew leaves nothing behind"
    );
    assert!(a.restored_keys("").unwrap().is_empty());
    let k_b = b.input("k", 5u32).unwrap();
    assert_eq!(b.runtime().get(k_b.handle()), 2, "the logged default");
}

#[test]
fn batch_delete_key_resets_inside_one_append() {
    let log = MemoryBackend::new();
    let store = open::<Local, _>(log.clone());
    let x = store.input("x", 1u32).unwrap();
    let items = store.collection::<u32>("items").unwrap();
    items.insert(&store, 3).unwrap();
    let before = log_records(&log).len();
    store
        .batch(|b| {
            b.set(x, 5)
                .delete_key(x)
                .delete_key(&items)
                .insert(&items, 4);
        })
        .unwrap();
    assert_eq!(log_records(&log).len(), before + 4);
    assert_eq!(store.runtime().get(x.handle()), 1);
    assert_eq!(items.handle().values(store.runtime()), [4]);
}

#[test]
fn deleting_a_handle_from_another_store_is_rejected() {
    let a = open::<Local, _>(MemoryBackend::new());
    let b = open::<Local, _>(MemoryBackend::new());
    let x = a.input("x", 0u32).unwrap();
    let items = a.collection::<u32>("items").unwrap();
    b.input("x", 0u32).unwrap();
    b.collection::<u32>("items").unwrap();
    assert!(matches!(b.delete(x), Err(Error::UnknownHandle)));
    assert!(matches!(b.delete(&items), Err(Error::UnknownHandle)));
    let result = b.batch(|batch| {
        batch.delete_key(x);
    });
    assert!(matches!(result, Err(Error::UnknownHandle)));
}

fn undeclare_errors_while_dependents_remain_and_is_not_persisted<C: Cells>() {
    let log = MemoryBackend::new();
    let store = open::<C, _>(log.clone());
    let other = open::<C, _>(log.clone());
    let x = store.input("x", 1u32).unwrap();
    store.set(x, 4).unwrap();
    let rt = store.runtime();
    let doubled = {
        let x = x.handle();
        rt.create_query(move |rt| rt.get(x) * 2)
    };
    assert_eq!(rt.get(doubled), 8);
    let nodes = rt.node_count();

    assert!(matches!(
        store.undeclare(x),
        Err(Error::HasDependents { key }) if key == "x"
    ));
    store.set(x, 5).unwrap();
    assert_eq!(rt.get(doubled), 10, "a refused undeclare changes nothing");

    rt.delete_node(doubled);
    let records = log_records(&log).len();
    store.undeclare(x).unwrap();
    assert_eq!(rt.node_count(), nodes - 2);
    assert_eq!(
        log_records(&log).len(),
        records,
        "undeclare appends nothing"
    );
    assert!(matches!(store.set(x, 6), Err(Error::UnknownHandle)));
    assert!(matches!(store.delete(x), Err(Error::UnknownHandle)));
    assert!(matches!(store.undeclare(x), Err(Error::UnknownHandle)));
    assert_eq!(store.restored_keys("").unwrap(), ["x"]);

    let other_x = other.input("x", 0u32).unwrap();
    other.set(other_x, 7).unwrap();
    store.refresh().unwrap();
    let again = store.input("x", 0u32).unwrap();
    assert_ne!(again, x);
    assert_eq!(
        rt.get(again.handle()),
        7,
        "records applied while undeclared still fold"
    );
    assert!(store.restored_keys("").unwrap().is_empty());

    drop(store);
    let reopened = open::<C, _>(log);
    let x = reopened.input("x", 0u32).unwrap();
    assert_eq!(reopened.runtime().get(x.handle()), 7);
}

#[test]
fn local_undeclare_errors_while_dependents_remain_and_is_not_persisted() {
    undeclare_errors_while_dependents_remain_and_is_not_persisted::<Local>();
}

#[test]
fn shared_undeclare_errors_while_dependents_remain_and_is_not_persisted() {
    undeclare_errors_while_dependents_remain_and_is_not_persisted::<Shared>();
}

#[test]
fn undeclare_releases_an_input_whose_node_was_deleted_through_the_runtime() {
    let store = open::<Local, _>(MemoryBackend::new());
    let x = store.input("x", 1u32).unwrap();
    store.set(x, 3).unwrap();
    store.runtime().delete_node(x.handle());

    store.undeclare(x).unwrap();
    assert_eq!(store.restored_keys("").unwrap(), ["x"]);
    let again = store.input("x", 0u32).unwrap();
    assert_eq!(store.runtime().get(again.handle()), 3);
}

fn undeclare_a_collection<C: Cells>() {
    let log = MemoryBackend::new();
    let store = open::<C, _>(log.clone());
    let other = open::<C, _>(log.clone());
    let items = store.collection::<u32>("items").unwrap();
    for value in [1, 1, 2] {
        items.insert(&store, value).unwrap();
    }
    let rt = store.runtime();
    let count = items.handle().count(rt);
    assert_eq!(rt.get(count), 3);
    let nodes = rt.node_count();

    assert!(matches!(
        store.undeclare(&items),
        Err(Error::HasDependents { key }) if key == "items"
    ));
    items.insert(&store, 3).unwrap();
    assert_eq!(rt.get(count), 4, "a refused undeclare changes nothing");

    rt.delete_node(count);
    let records = log_records(&log).len();
    store.undeclare(&items).unwrap();
    assert_eq!(rt.node_count(), nodes - 2);
    assert_eq!(
        log_records(&log).len(),
        records,
        "undeclare appends nothing"
    );
    assert!(matches!(items.insert(&store, 9), Err(Error::UnknownHandle)));
    assert!(matches!(store.undeclare(&items), Err(Error::UnknownHandle)));
    assert_eq!(store.restored_keys("").unwrap(), ["items"]);

    let other_items = other.collection::<u32>("items").unwrap();
    other_items.insert(&other, 5).unwrap();
    other_items.delete(&other, &1).unwrap();
    store.refresh().unwrap();
    store.checkpoint().unwrap();
    let again = store.collection::<u32>("items").unwrap();
    let mut values = again.handle().values(rt);
    values.sort();
    assert_eq!(
        values,
        [1, 2, 3, 5],
        "elements held while declared and records applied while undeclared both fold"
    );
    assert!(store.restored_keys("").unwrap().is_empty());

    drop(store);
    let reopened = open::<C, _>(log);
    let items = reopened.collection::<u32>("items").unwrap();
    let mut values = items.handle().values(reopened.runtime());
    values.sort();
    assert_eq!(values, [1, 2, 3, 5]);
}

#[test]
fn local_undeclare_a_collection() {
    undeclare_a_collection::<Local>();
}

#[test]
fn shared_undeclare_a_collection() {
    undeclare_a_collection::<Shared>();
}

#[test]
fn undeclare_a_collection_errors_while_a_group_by_reads_it() {
    let store = open::<Shared, _>(MemoryBackend::new());
    let items = store.collection::<u32>("items").unwrap();
    items.insert(&store, 4).unwrap();
    let rt = store.runtime();
    let groups = items.handle().group_by(rt, |x| x % 2);
    let _ = rt.get(groups.version_node());

    assert!(matches!(
        store.undeclare(&items),
        Err(Error::HasDependents { .. })
    ));
    items.insert(&store, 7).unwrap();
    let _ = rt.get(groups.version_node());
    assert_eq!(groups.group_count(), 2);

    rt.delete_node(groups.version_node());
    store.undeclare(&items).unwrap();
    assert_eq!(rt.node_count(), 0);
}

#[test]
fn undeclare_by_name_releases_a_bound_key_and_ignores_unknown_ones() {
    let store = open::<Local, _>(MemoryBackend::new());
    let x = store.input("x", 1u32).unwrap();
    store.undeclare("never-declared").unwrap();
    store.undeclare("x").unwrap();
    assert!(matches!(store.set(x, 2), Err(Error::UnknownHandle)));
    assert_eq!(store.restored_keys("").unwrap(), ["x"]);
}

#[test]
fn undeclare_keeps_a_collection_whose_node_was_deleted_through_the_runtime() {
    let store = open::<Local, _>(MemoryBackend::new());
    let items = store.collection::<u32>("items").unwrap();
    items.insert(&store, 1).unwrap();
    store.runtime().delete_node(items.handle().version_node());

    assert!(matches!(store.undeclare(&items), Err(Error::UnknownHandle)));
    assert!(
        store.restored_keys("").unwrap().is_empty(),
        "the key stays bound rather than lose its elements"
    );
}
