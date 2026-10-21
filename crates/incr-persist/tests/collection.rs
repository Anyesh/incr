use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use incr_persist::{
    Backend, Cells, Error, FileBackend, Kind, Local, MemoryBackend, Options, Position, ReadOutcome,
    Record, Shared, Store,
};
use serde::{Deserialize, Serialize};

fn open<C: Cells, B: Backend>(backend: B) -> Store<C, B> {
    Store::open(backend, Options::default()).unwrap()
}

fn log_len(backend: &impl Backend) -> usize {
    let mut out = Vec::new();
    assert_eq!(
        backend.read_from(Position::ZERO, &mut out).unwrap(),
        ReadOutcome::Records
    );
    out.len()
}

fn sorted<T: Ord>(mut values: Vec<T>) -> Vec<T> {
    values.sort();
    values
}

fn s(value: &str) -> String {
    value.to_owned()
}

fn round_trip_through_a_restart<C: Cells, B: Backend>(backend: impl Fn() -> B) {
    let expected_count = {
        let store = open::<C, B>(backend());
        let tags = store.collection::<String>("tags").unwrap();
        let rt = store.runtime();
        let count = tags.handle().count(rt);
        assert_eq!(rt.get(count), 0);

        tags.insert(&store, s("a")).unwrap();
        tags.insert(&store, s("a")).unwrap();
        tags.insert(&store, s("b")).unwrap();
        tags.insert(&store, s("d")).unwrap();
        tags.delete(&store, &s("a")).unwrap();
        tags.delete(&store, &s("absent")).unwrap();
        tags.replace(&store, Some(&s("b")), s("c")).unwrap();
        tags.replace(&store, None, s("a")).unwrap();
        tags.replace(&store, Some(&s("absent")), s("d")).unwrap();

        assert_eq!(
            sorted(tags.handle().values(rt)),
            [s("a"), s("a"), s("c"), s("d"), s("d")]
        );
        assert_eq!(rt.get(count), 5);
        rt.get(count)
    };

    let store = open::<C, B>(backend());
    let tags = store.collection::<String>("tags").unwrap();
    let rt = store.runtime();
    assert_eq!(
        sorted(tags.handle().values(rt)),
        [s("a"), s("a"), s("c"), s("d"), s("d")]
    );
    let count = tags.handle().count(rt);
    assert_eq!(rt.get(count), expected_count);

    tags.delete(&store, &s("d")).unwrap();
    assert_eq!(rt.get(count), 4);
}

#[test]
fn collection_writes_round_trip_through_a_restart_on_memory() {
    let log = MemoryBackend::new();
    round_trip_through_a_restart::<Local, _>(|| log.clone());
    let log = MemoryBackend::new();
    round_trip_through_a_restart::<Shared, _>(|| log.clone());
}

#[test]
fn collection_writes_round_trip_through_a_restart_on_file() {
    let dir = tempfile::tempdir().unwrap();
    round_trip_through_a_restart::<Local, _>(|| FileBackend::open(dir.path()).unwrap());
    let dir = tempfile::tempdir().unwrap();
    round_trip_through_a_restart::<Shared, _>(|| FileBackend::open(dir.path()).unwrap());
}

#[test]
fn a_collection_is_declared_once_and_redeclaring_returns_the_same_collection() {
    let log = MemoryBackend::new();
    let store = open::<Local, _>(log.clone());
    let first = store.collection::<u32>("c").unwrap();
    let again = store.collection::<u32>("c").unwrap();
    first.insert(&store, 7).unwrap();
    assert_eq!(again.handle().values(store.runtime()), [7]);
    assert_eq!(
        log_len(&log),
        2,
        "one Declare and one Insert, no second Declare"
    );

    let other = open::<Local, _>(log.clone());
    other.collection::<u32>("c").unwrap();
    assert_eq!(log_len(&log), 2, "a live Declare is not appended again");
}

#[test]
fn a_batch_mixes_input_and_collection_writes_in_one_append() {
    let log = MemoryBackend::new();
    let store = open::<Local, _>(log.clone());
    let total = store.input("total", 0u32).unwrap();
    let items = store.collection::<u32>("items").unwrap();
    let before = log_len(&log);
    let last = store
        .batch(|b| {
            b.insert(&items, 1)
                .insert(&items, 1)
                .insert(&items, 2)
                .delete(&items, &2)
                .replace(&items, Some(&1), 5)
                .replace(&items, None, 6)
                .set(total, 3);
        })
        .unwrap();
    assert_eq!(log_len(&log), before + 7);
    assert_eq!(last.0 as usize, before + 7);
    let rt = store.runtime();
    assert_eq!(sorted(items.handle().values(rt)), [1, 5, 6]);
    assert_eq!(rt.get(total.handle()), 3);

    drop(store);
    let store = open::<Local, _>(log);
    let items = store.collection::<u32>("items").unwrap();
    assert_eq!(sorted(items.handle().values(store.runtime())), [1, 5, 6]);
}

#[test]
fn a_batch_with_a_foreign_collection_writes_nothing() {
    let other = open::<Local, _>(MemoryBackend::new());
    let foreign = other.collection::<u32>("c").unwrap();
    let log = MemoryBackend::new();
    let store = open::<Local, _>(log.clone());
    let mine = store.collection::<u32>("c").unwrap();
    let before = log_len(&log);
    let result = store.batch(|b| {
        b.insert(&mine, 1).insert(&foreign, 2);
    });
    assert!(matches!(result, Err(Error::UnknownHandle)));
    assert_eq!(log_len(&log), before);
    assert!(mine.handle().values(store.runtime()).is_empty());
}

#[test]
fn a_collection_from_another_store_is_rejected() {
    let a = open::<Local, _>(MemoryBackend::new());
    let b = open::<Local, _>(MemoryBackend::new());
    let foreign = a.collection::<u32>("c").unwrap();
    b.collection::<u32>("c").unwrap();
    assert!(matches!(foreign.insert(&b, 1), Err(Error::UnknownHandle)));
    assert!(matches!(foreign.delete(&b, &1), Err(Error::UnknownHandle)));
    assert!(matches!(
        foreign.replace(&b, None, 1),
        Err(Error::UnknownHandle)
    ));
}

#[test]
fn kinds_and_types_are_checked() {
    let log = MemoryBackend::new();
    let store = open::<Local, _>(log.clone());
    store.input("price", 0u32).unwrap();
    store.collection::<u32>("items").unwrap();
    assert!(matches!(
        store.collection::<u32>("price"),
        Err(Error::KindMismatch { key, declared: Kind::Input }) if key == "price"
    ));
    assert!(matches!(
        store.input("items", 0u32),
        Err(Error::KindMismatch { key, declared: Kind::Collection }) if key == "items"
    ));
    assert!(matches!(
        store.collection::<String>("items"),
        Err(Error::TypeMismatch { key, .. }) if key == "items"
    ));

    let other = open::<Local, _>(log);
    assert!(matches!(
        other.collection::<u32>("price"),
        Err(Error::KindMismatch { key, declared: Kind::Input }) if key == "price"
    ));
}

#[test]
fn logged_elements_of_another_type_are_a_type_mismatch() {
    let log = MemoryBackend::new();
    {
        let store = open::<Local, _>(log.clone());
        let names = store.collection::<String>("names").unwrap();
        names.insert(&store, s("a longer string")).unwrap();
    }
    let store = open::<Local, _>(log);
    assert!(matches!(
        store.collection::<u8>("names"),
        Err(Error::TypeMismatch { key, .. }) if key == "names"
    ));
    let names = store.collection::<String>("names").unwrap();
    assert_eq!(
        names.handle().values(store.runtime()),
        [s("a longer string")]
    );
}

#[test]
#[should_panic(expected = "derived collection")]
fn the_handle_is_a_view_that_cannot_bypass_the_log() {
    let store = open::<Local, _>(MemoryBackend::new());
    let items = store.collection::<u32>("items").unwrap();
    items.handle().insert(store.runtime(), 1);
}

#[test]
fn a_write_applies_foreign_collection_records_first() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let b = open::<Local, _>(log.clone());
    let items_a = a.collection::<u32>("items").unwrap();
    let items_b = b.collection::<u32>("items").unwrap();
    let count_b = items_b.handle().count(b.runtime());
    items_a.insert(&a, 1).unwrap();
    items_a.insert(&a, 1).unwrap();
    assert_eq!(b.runtime().get(count_b), 0);
    items_b.delete(&b, &1).unwrap();
    assert_eq!(b.runtime().get(count_b), 1);
    assert_eq!(items_b.handle().values(b.runtime()), [1]);
}

#[test]
fn foreign_raw_records_apply_to_a_declared_collection() {
    let log = MemoryBackend::new();
    let store = open::<Local, _>(log.clone());
    let items = store.collection::<u32>("items").unwrap();
    let element = |v: u32| postcard::to_stdvec(&v).unwrap();
    log.append(&[
        Record::Insert {
            key: "items".into(),
            element: element(4),
        },
        Record::Insert {
            key: "items".into(),
            element: element(4),
        },
        Record::Delete {
            key: "items".into(),
        },
        Record::Insert {
            key: "items".into(),
            element: element(9),
        },
    ])
    .unwrap();
    items.insert(&store, 1).unwrap();
    assert_eq!(sorted(items.handle().values(store.runtime())), [1, 9]);
}

/// Equal by `id` alone while `noise` is serialized too, so two equal values
/// can have different bytes, as a type with a `HashMap` field would.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Loose {
    id: u32,
    noise: u32,
}

impl PartialEq for Loose {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Loose {}

impl Hash for Loose {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

fn loose(id: u32, noise: u32) -> Loose {
    Loose { id, noise }
}

fn ids(values: Vec<Loose>) -> Vec<u32> {
    sorted(values.into_iter().map(|v| v.id).collect())
}

fn write_loose_history<C: Cells, B: Backend>(store: &Store<C, B>) -> Vec<u32> {
    let items = store.collection::<Loose>("loose").unwrap();
    items.insert(store, loose(1, 0)).unwrap();
    items.insert(store, loose(2, 0)).unwrap();
    items.delete(store, &loose(1, 7)).unwrap();
    items.delete(store, &loose(1, 8)).unwrap();
    items.insert(store, loose(1, 9)).unwrap();
    items
        .replace(store, Some(&loose(2, 5)), loose(3, 0))
        .unwrap();
    ids(items.handle().values(store.runtime()))
}

#[test]
fn declared_collections_fold_by_decoded_value_not_by_bytes() {
    let log = MemoryBackend::new();
    let live = {
        let store = open::<Local, _>(log.clone());
        write_loose_history(&store)
    };
    assert_eq!(live, [1, 3]);
    let store = open::<Local, _>(log.clone());
    let items = store.collection::<Loose>("loose").unwrap();
    assert_eq!(ids(items.handle().values(store.runtime())), live);
}

#[test]
fn a_collection_declared_while_records_arrive_folds_them_by_value() {
    let log = MemoryBackend::new();
    let writer = open::<Local, _>(log.clone());
    let reader = open::<Local, _>(log.clone());
    let live = write_loose_history(&writer);
    reader.input("unrelated", 0u32).unwrap();
    assert_eq!(reader.restored_keys("").unwrap(), ["loose"]);
    let items = reader.collection::<Loose>("loose").unwrap();
    assert_eq!(ids(items.handle().values(reader.runtime())), live);
}

#[test]
fn checkpoints_fold_loose_elements_by_value_whether_bound_or_not() {
    let log = MemoryBackend::new();
    let live = {
        let store = open::<Local, _>(log.clone());
        let live = write_loose_history(&store);
        store.checkpoint().unwrap();
        live
    };
    let (_, snapshot) = log.load_snapshot().unwrap().unwrap();
    let inserts = snapshot
        .iter()
        .filter(|r| matches!(r, Record::Insert { .. }))
        .count();
    assert_eq!(inserts, live.len(), "a bound collection folds by value");
    let store = open::<Local, _>(log.clone());
    let items = store.collection::<Loose>("loose").unwrap();
    assert_eq!(ids(items.handle().values(store.runtime())), live);

    let log = MemoryBackend::new();
    let live = {
        let store = open::<Local, _>(log.clone());
        write_loose_history(&store)
    };
    open::<Local, _>(log.clone()).checkpoint().unwrap();
    let store = open::<Local, _>(log);
    let items = store.collection::<Loose>("loose").unwrap();
    assert_eq!(ids(items.handle().values(store.runtime())), live);
}

fn reps(values: Vec<Loose>) -> Vec<(u32, u32)> {
    sorted(values.into_iter().map(|v| (v.id, v.noise)).collect())
}

/// A class keeps the first inserted of its equal elements: 1 keeps noise 1
/// after the occurrence written with it is deleted, 2 keeps 5 over the
/// later 6, 4 is reborn with 2 after it emptied, and 5 swaps to 3 through
/// a count-1 replace.
fn write_rep_history<C: Cells, B: Backend>(store: &Store<C, B>) -> Vec<(u32, u32)> {
    let items = store.collection::<Loose>("loose").unwrap();
    for value in [loose(1, 1), loose(2, 5), loose(1, 2), loose(2, 6)] {
        items.insert(store, value).unwrap();
    }
    items.delete(store, &loose(1, 1)).unwrap();
    items.insert(store, loose(4, 1)).unwrap();
    items.delete(store, &loose(4, 1)).unwrap();
    items.insert(store, loose(4, 2)).unwrap();
    items.insert(store, loose(5, 0)).unwrap();
    items
        .replace(store, Some(&loose(5, 0)), loose(5, 3))
        .unwrap();
    reps(items.handle().values(store.runtime()))
}

const REPS: [(u32, u32); 5] = [(1, 1), (2, 5), (2, 5), (4, 2), (5, 3)];

fn restored_reps(log: &MemoryBackend) -> Vec<(u32, u32)> {
    let store = open::<Local, _>(log.clone());
    let items = store.collection::<Loose>("loose").unwrap();
    reps(items.handle().values(store.runtime()))
}

// Each restart folds in a fresh hash order, so a fold that follows hash
// order rather than insertion order shows up within a few of them.
#[test]
fn a_restart_keeps_the_representatives_the_writer_had() {
    let log = MemoryBackend::new();
    assert_eq!(write_rep_history(&open::<Local, _>(log.clone())), REPS);
    for _ in 0..8 {
        assert_eq!(restored_reps(&log), REPS);
    }
}

#[test]
fn a_collection_declared_late_keeps_the_writers_representatives() {
    for _ in 0..8 {
        let log = MemoryBackend::new();
        let writer = open::<Local, _>(log.clone());
        let reader = open::<Local, _>(log.clone());
        write_rep_history(&writer);
        reader.refresh().unwrap();
        let items = reader.collection::<Loose>("loose").unwrap();
        assert_eq!(reps(items.handle().values(reader.runtime())), REPS);
    }
}

#[test]
fn checkpoints_keep_representatives_whether_bound_or_not() {
    for bound in [true, false] {
        for _ in 0..8 {
            let log = MemoryBackend::new();
            let writer = open::<Local, _>(log.clone());
            write_rep_history(&writer);
            if bound {
                writer.checkpoint().unwrap();
            } else {
                open::<Local, _>(log.clone()).checkpoint().unwrap();
            }
            drop(writer);
            assert_eq!(restored_reps(&log), REPS, "bound: {bound}");
        }
    }
}

// The other store's delete and insert land in its snapshot, so `a` takes
// the class's new representative from a fold whose count is unchanged.
#[test]
fn a_catch_up_across_a_compaction_takes_a_moved_representative() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let items_a = a.collection::<Loose>("loose").unwrap();
    items_a.insert(&a, loose(1, 1)).unwrap();
    let b = open::<Local, _>(log.clone());
    let items_b = b.collection::<Loose>("loose").unwrap();
    items_b.delete(&b, &loose(1, 1)).unwrap();
    items_b.insert(&b, loose(1, 2)).unwrap();
    b.checkpoint().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    a.subscribe(move |_, record, _| sink.lock().unwrap().push(record.clone()))
        .unwrap();
    a.refresh().unwrap();
    assert_eq!(reps(items_a.handle().values(a.runtime())), [(1, 2)]);
    let element = |value: Loose| postcard::to_stdvec(&value).unwrap();
    assert_eq!(
        *seen.lock().unwrap(),
        [
            Record::Remove {
                key: "loose".into(),
                element: element(loose(1, 1)),
            },
            Record::Insert {
                key: "loose".into(),
                element: element(loose(1, 2)),
            },
        ]
    );
}
