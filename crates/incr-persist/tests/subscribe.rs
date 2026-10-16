use std::collections::BTreeSet;
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

    let reader = open::<Local, _>(log.read_only());
    let r_x = reader.input("x", 0u32).unwrap();
    let r_items = reader.collection::<u32>("items").unwrap();
    let r_gone = reader.input("gone", 0u32).unwrap();
    let seen = subscribe(&reader);

    writer.set(x, 7).unwrap();
    writer.delete(gone).unwrap();
    items.delete(&writer, &1).unwrap();
    items.insert(&writer, 3).unwrap();
    raw.delete(&writer, &2).unwrap();
    raw.insert(&writer, 6).unwrap();
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
        [Record::Delete { key: "gone".into() }]
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
                element: enc(&6u32),
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
    assert!(
        for_key(&seen, "same").is_empty(),
        "unchanged keys are not reported"
    );
}
