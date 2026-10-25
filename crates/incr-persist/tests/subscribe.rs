use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::thread;

use incr_persist::{
    Backend, Cells, Kind, Local, MemoryBackend, Options, Origin, Position, ReadOutcome, Record,
    Shared, Store,
};

fn open<C: Cells, B: Backend>(backend: B) -> Store<C, B> {
    Store::open(backend, Options::default()).unwrap()
}

fn enc<T: serde::Serialize>(value: &T) -> Vec<u8> {
    postcard::to_stdvec(value).unwrap()
}

fn log_after(backend: &impl Backend, after: Position) -> Vec<(Position, Record)> {
    let mut out = Vec::new();
    assert_eq!(
        backend.read_from(after, &mut out).unwrap(),
        ReadOutcome::Records
    );
    out
}

type Seen = Arc<Mutex<Vec<(Position, Record, Origin)>>>;

fn subscribe<C: Cells, B: Backend>(store: &Store<C, B>) -> Seen {
    let seen: Seen = Arc::default();
    let sink = Arc::clone(&seen);
    store
        .subscribe(move |position, record, origin| {
            sink.lock()
                .unwrap()
                .push((position, record.clone(), origin));
        })
        .unwrap();
    seen
}

#[derive(Debug, PartialEq)]
struct View {
    last: u64,
    events: Vec<(u8, u64)>,
}

fn view<C: Cells, B: Backend>(store: &Store<C, B>) -> View {
    let last = store.input("last", 0u64).unwrap();
    let events = store.collection::<(u8, u64)>("events").unwrap();
    let rt = store.runtime();
    let mut values = events.handle().values(rt);
    values.sort();
    View {
        last: rt.get(last.handle()),
        events: values,
    }
}

#[test]
fn two_writers_on_memory_converge_and_subscribers_see_every_record_in_order() {
    let log = MemoryBackend::new();
    let stores = [
        open::<Shared, _>(log.clone()),
        open::<Shared, _>(log.clone()),
    ];
    let mut subscriptions = Vec::new();
    for store in &stores {
        view(store);
        let from = store.refresh().unwrap();
        subscriptions.push((from, subscribe(store)));
    }

    let own: Vec<BTreeSet<Position>> = thread::scope(|s| {
        let workers: Vec<_> = stores
            .iter()
            .enumerate()
            .map(|(writer, store)| {
                s.spawn(move || {
                    let writer = writer as u8;
                    let last = store.input("last", 0u64).unwrap();
                    let events = store.collection::<(u8, u64)>("events").unwrap();
                    let mut own = BTreeSet::new();
                    for i in 0..100u64 {
                        own.insert(events.insert(store, (writer, i)).unwrap());
                        own.insert(store.set(last, i).unwrap());
                        if i % 10 == 0 {
                            own.insert(events.delete(store, &(writer, i)).unwrap());
                        }
                    }
                    let end = store
                        .batch(|b| {
                            b.set(last, 1000 + u64::from(writer))
                                .insert(&events, (writer, 1000))
                                .delete_key("scratch");
                        })
                        .unwrap();
                    own.extend((end.0 - 2..=end.0).map(Position));
                    own
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });

    let tail = log.append(&[]).unwrap();
    for store in &stores {
        assert_eq!(store.refresh().unwrap(), tail);
    }
    let converged = view(&stores[0]);
    assert_eq!(view(&stores[1]), converged);
    assert_eq!(converged.events.len(), 2 * (100 - 10 + 1));
    assert!(converged.last == 1000 || converged.last == 1001);
    assert_eq!(view(&open::<Local, _>(log.clone())), converged);

    for (writer, (from, seen)) in subscriptions.iter().enumerate() {
        let seen = seen.lock().unwrap();
        let expected = log_after(&log, *from);
        assert_eq!(
            seen.len(),
            expected.len(),
            "writer {writer} saw every record"
        );
        for ((position, record, origin), (at, logged)) in seen.iter().zip(&expected) {
            assert_eq!((position, record), (at, logged), "in position order");
            let want = if own[writer].contains(position) {
                Origin::Local
            } else {
                Origin::Foreign
            };
            assert_eq!(*origin, want, "origin of position {position}");
        }
    }
}

#[test]
fn unsubscribed_callbacks_see_nothing_more() {
    let store = open::<Local, _>(MemoryBackend::new());
    let x = store.input("x", 0u32).unwrap();
    let seen: Seen = Arc::default();
    let sink = Arc::clone(&seen);
    let id = store
        .subscribe(move |position, record, origin| {
            sink.lock()
                .unwrap()
                .push((position, record.clone(), origin));
        })
        .unwrap();
    let at = store.set(x, 1).unwrap();
    store.unsubscribe(id).unwrap();
    store.set(x, 2).unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(
        *seen,
        [(
            at,
            Record::Set {
                key: "x".into(),
                value: enc(&1u32),
            },
            Origin::Local,
        )]
    );
}

/// The records of one key in the order a subscriber saw them.
fn for_key(seen: &[(Position, Record, Origin)], key: &str) -> Vec<Record> {
    seen.iter()
        .filter(|(_, record, _)| record.key() == key)
        .map(|(_, record, _)| record.clone())
        .collect()
}

fn sorted_records(mut records: Vec<Record>) -> Vec<Vec<u8>> {
    let mut out: Vec<_> = records.drain(..).map(|r| r.encode().unwrap()).collect();
    out.sort();
    out
}

#[test]
fn catching_up_across_a_compaction_reports_each_change_with_snapshot_origin() {
    let log = MemoryBackend::new();
    let writer = open::<Local, _>(log.clone());
    let x = writer.input("x", 0u32).unwrap();
    let items = writer.collection::<u32>("items").unwrap();
    let gone = writer.input("gone", 0u32).unwrap();
    let raw = writer.collection::<u32>("raw").unwrap();
    writer.set(x, 5).unwrap();
    writer.set(gone, 9).unwrap();
    for item in [1, 1, 2] {
        items.insert(&writer, item).unwrap();
        raw.insert(&writer, item).unwrap();
    }
    writer.input("same", 4u32).unwrap();
    let kept = writer.input("kept", 0u32).unwrap();
    writer.set(kept, 2).unwrap();

    let reader = open::<Local, _>(log.read_only());
    let r_x = reader.input("x", 0u32).unwrap();
    let r_items = reader.collection::<u32>("items").unwrap();
    let r_gone = reader.input("gone", 0u32).unwrap();
    let seen = subscribe(&reader);

    writer.set(x, 7).unwrap();
    writer.delete(gone).unwrap();
    items.delete(&writer, &1).unwrap();
    items.insert(&writer, 3).unwrap();
    // A writer folds a collection it declared in its stored order, not its
    // insertion order, so `raw` ends with one distinct element to keep the
    // reader's byte diff from depending on hash order.
    raw.delete(&writer, &2).unwrap();
    raw.insert(&writer, 1).unwrap();
    let fresh = writer.input("fresh", 8u32).unwrap();
    writer.set(fresh, 3).unwrap();
    let at = writer.checkpoint().unwrap();

    reader.refresh().unwrap();
    let rt = reader.runtime();
    assert_eq!(rt.get(r_x.handle()), 7);
    assert_eq!(rt.get(r_gone.handle()), 0);
    let mut values = r_items.handle().values(rt);
    values.sort();
    assert_eq!(values, [1, 2, 3]);

    let seen = seen.lock().unwrap();
    assert!(seen
        .iter()
        .all(|(position, _, origin)| *position == at && *origin == Origin::Snapshot));
    assert_eq!(
        for_key(&seen, "x"),
        [Record::Set {
            key: "x".into(),
            value: enc(&7u32),
        }]
    );
    assert_eq!(
        for_key(&seen, "gone"),
        [Record::Forget { key: "gone".into() }],
        "a key deleted back to its default is forgotten"
    );
    assert_eq!(
        sorted_records(for_key(&seen, "items")),
        sorted_records(vec![
            Record::Remove {
                key: "items".into(),
                element: enc(&1u32),
            },
            Record::Insert {
                key: "items".into(),
                element: enc(&3u32),
            },
        ])
    );
    assert_eq!(
        sorted_records(for_key(&seen, "raw")),
        sorted_records(vec![
            Record::Remove {
                key: "raw".into(),
                element: enc(&2u32),
            },
            Record::Insert {
                key: "raw".into(),
                element: enc(&1u32),
            },
        ]),
        "a key this store never declared is diffed by bytes"
    );
    assert_eq!(
        for_key(&seen, "fresh"),
        [
            Record::Declare {
                key: "fresh".into(),
                kind: Kind::Input,
                default: enc(&8u32),
            },
            Record::Set {
                key: "fresh".into(),
                value: enc(&3u32),
            },
        ]
    );
    assert_eq!(
        for_key(&seen, "same"),
        [Record::Forget { key: "same".into() }],
        "a key left at its default is forgotten"
    );
    assert!(
        for_key(&seen, "kept").is_empty(),
        "unchanged keys are not reported"
    );
}

/// A multi-element collection the reader never declared, caught up across
/// a compaction, reports the exact `Remove`/`Insert` diff whatever hash seed
/// each store draws: the writer hands its elements to the folded log in byte
/// order. Each fresh store draws new hash keys, so the rounds cover several.
#[test]
fn an_undeclared_collection_catches_up_as_the_exact_diff() {
    for _ in 0..32 {
        let log = MemoryBackend::new();
        let writer = open::<Local, _>(log.clone());
        let raw = writer.collection::<u32>("raw").unwrap();
        for item in [1, 1, 2] {
            raw.insert(&writer, item).unwrap();
        }
        let reader = open::<Local, _>(log.read_only());
        let seen = subscribe(&reader);
        raw.delete(&writer, &2).unwrap();
        raw.insert(&writer, 6).unwrap();
        writer.checkpoint().unwrap();
        reader.refresh().unwrap();

        let records = for_key(&seen.lock().unwrap(), "raw");
        let diff = vec![
            Record::Remove {
                key: "raw".into(),
                element: enc(&2u32),
            },
            Record::Insert {
                key: "raw".into(),
                element: enc(&6u32),
            },
        ];
        assert_eq!(records, diff, "the exact diff, removes first");
        let mut values = reader
            .collection::<u32>("raw")
            .unwrap()
            .handle()
            .values(reader.runtime());
        values.sort();
        assert_eq!(values, [1, 1, 6], "the reader's state is the writer's");
    }
}

/// What a store that does not declare "raw" reports for it on catching up
/// across a compaction, after first loading an earlier snapshot, with the
/// writer's collection built by inserting `order`. With `undeclared`, the
/// store declares the key after loading and undeclares it again.
fn undeclared_catch_up(order: impl IntoIterator<Item = u32>, undeclared: bool) -> Vec<Record> {
    let log = MemoryBackend::new();
    let writer = open::<Local, _>(log.clone());
    let raw = writer.collection::<u32>("raw").unwrap();
    for value in order {
        raw.insert(&writer, value).unwrap();
    }
    writer.checkpoint().unwrap();
    let reader = open::<Local, _>(log.read_only());
    if undeclared {
        let r_raw = reader.collection::<u32>("raw").unwrap();
        reader.undeclare(&r_raw).unwrap();
    }
    let seen = subscribe(&reader);
    raw.delete(&writer, &63).unwrap();
    raw.insert(&writer, 64).unwrap();
    writer.checkpoint().unwrap();
    reader.refresh().unwrap();
    let seen = seen.lock().unwrap();
    for_key(&seen, "raw")
}

// Each collection hashes with its own seed, so two of them would list the
// same 64 elements in the same order only by chance.
#[test]
fn an_undeclared_catch_up_reports_the_same_changes_however_the_writer_hashes() {
    let expected = [
        Record::Remove {
            key: "raw".into(),
            element: enc(&63u32),
        },
        Record::Insert {
            key: "raw".into(),
            element: enc(&64u32),
        },
    ];
    for undeclared in [false, true] {
        assert_eq!(undeclared_catch_up(0..64, undeclared), expected);
        assert_eq!(undeclared_catch_up((0..64).rev(), undeclared), expected);
    }
}

#[derive(Debug, Default, PartialEq)]
struct Mirrored {
    declared: Option<(Kind, Vec<u8>)>,
    value: Option<Vec<u8>>,
    elements: BTreeMap<Vec<u8>, usize>,
}

fn take_one(elements: &mut BTreeMap<Vec<u8>, usize>, element: &[u8]) {
    if let Some(count) = elements.get_mut(element) {
        *count -= 1;
        if *count == 0 {
            elements.remove(element);
        }
    }
}

/// Folds records by the log's own rules: the first live `Declare` wins, a
/// `Delete` resets the key to it, and a `Forget` drops the key, `Declare`
/// included.
fn mirror(keys: &mut BTreeMap<String, Mirrored>, records: impl IntoIterator<Item = Record>) {
    for record in records {
        let key = record.key().to_owned();
        let entry = keys.entry(key.clone()).or_default();
        match record {
            Record::Declare { kind, default, .. } => {
                entry.declared.get_or_insert((kind, default));
            }
            Record::Set { value, .. } => entry.value = Some(value),
            Record::Insert { element, .. } => *entry.elements.entry(element).or_default() += 1,
            Record::Remove { element, .. } => take_one(&mut entry.elements, &element),
            Record::Replace { old, new, .. } => {
                if let Some(old) = old {
                    take_one(&mut entry.elements, &old);
                }
                *entry.elements.entry(new).or_default() += 1;
            }
            Record::Delete { .. } => {
                entry.value = None;
                entry.elements.clear();
            }
            Record::Forget { .. } => *entry = Mirrored::default(),
        }
        if *entry == Mirrored::default() {
            keys.remove(&key);
        }
    }
}

/// What a store opened on `backend` now would fold: its snapshot, then the
/// records after it.
fn restart_state(backend: &impl Backend) -> BTreeMap<String, Mirrored> {
    let (at, snapshot) = backend
        .load_snapshot()
        .unwrap()
        .unwrap_or((Position::ZERO, Vec::new()));
    let later = log_after(backend, at).into_iter().map(|(_, record)| record);
    let mut keys = BTreeMap::new();
    mirror(&mut keys, snapshot.into_iter().chain(later));
    keys
}

/// Folds everything a subscriber registered on a store over an empty log
/// saw.
fn mirrored(seen: &Seen) -> BTreeMap<String, Mirrored> {
    let mut keys = BTreeMap::new();
    mirror(
        &mut keys,
        seen.lock()
            .unwrap()
            .iter()
            .map(|(_, record, _)| record.clone()),
    );
    keys
}

#[test]
fn folding_what_a_subscriber_saw_across_a_compaction_reaches_the_restart_state() {
    let log = MemoryBackend::new();
    let reader = open::<Local, _>(log.read_only());
    let seen = subscribe(&reader);
    let writer = open::<Local, _>(log.clone());
    let k = writer.input("k", 1u32).unwrap();
    let s = writer.input("s", 1u32).unwrap();
    let items = writer.collection::<u32>("items").unwrap();
    let loose = writer.collection::<u32>("loose").unwrap();
    writer.set(k, 3).unwrap();
    writer.set(s, 4).unwrap();
    items.insert(&writer, 1).unwrap();
    loose.insert(&writer, 2).unwrap();
    reader.refresh().unwrap();
    reader.input("k", 0u32).unwrap();
    reader.collection::<u32>("items").unwrap();

    writer.delete(k).unwrap();
    items.delete(&writer, &1).unwrap();
    loose.delete(&writer, &2).unwrap();
    writer.set(s, 6).unwrap();
    writer.checkpoint().unwrap();
    let other = open::<Local, _>(log.clone());
    let k2 = other.input("k", 9u32).unwrap();
    other.set(k2, 10).unwrap();
    other
        .collection::<u32>("items")
        .unwrap()
        .insert(&other, 5)
        .unwrap();
    let loose_input = other.input("loose", 2u32).unwrap();
    other.set(loose_input, 3).unwrap();
    other.checkpoint().unwrap();

    reader.refresh().unwrap();
    assert!(seen
        .lock()
        .unwrap()
        .iter()
        .any(
            |(_, record, origin)| matches!(record, Record::Forget { .. })
                && *origin == Origin::Snapshot
        ));
    assert_eq!(mirrored(&seen), restart_state(&log));
}

/// A store already at the position another writer compacts to reads no
/// record of the compaction; its subscribers must still be told what it
/// forgot, so that folding what they saw reaches the restart state.
#[test]
fn folding_what_a_subscriber_saw_across_a_compaction_at_its_position_reaches_the_restart_state() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let seen = subscribe(&a);
    let k = a.input("k", 1u32).unwrap();
    let items = a.collection::<u32>("items").unwrap();
    a.set(k, 3).unwrap();
    items.insert(&a, 1).unwrap();
    let b = open::<Local, _>(log.clone());
    b.input("u", 1u32).unwrap();
    b.input("v", 1u32).unwrap();
    b.delete("k").unwrap();
    b.delete("items").unwrap();
    let applied = a.refresh().unwrap();
    assert_eq!(b.checkpoint().unwrap(), applied);

    let c = open::<Local, _>(log.clone());
    let c_k = c.input("k", 9u32).unwrap();
    c.set(c_k, 10).unwrap();
    c.collection::<u32>("items").unwrap().insert(&c, 5).unwrap();
    c.input("u", 4u32).unwrap();
    a.refresh().unwrap();

    let forgotten: BTreeSet<String> = seen
        .lock()
        .unwrap()
        .iter()
        .filter(|(_, record, origin)| {
            matches!(record, Record::Forget { .. }) && *origin == Origin::Snapshot
        })
        .map(|(_, record, _)| record.key().to_owned())
        .collect();
    assert_eq!(
        forgotten,
        BTreeSet::from(["items", "k", "u", "v"].map(str::to_owned))
    );
    assert_eq!(mirrored(&seen), restart_state(&log));
    assert_eq!(a.runtime().get(k.handle()), 10);
    assert_eq!(items.handle().values(a.runtime()), [5]);
}
