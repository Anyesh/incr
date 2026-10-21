use std::collections::BTreeMap;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use incr_persist::{
    Backend, Cells, Error, FileBackend, Kind, Local, MemoryBackend, Options, Position, ReadOutcome,
    Record, Shared, Store,
};

fn open<C: Cells, B: Backend>(backend: B) -> Store<C, B> {
    Store::open(backend, Options::default()).unwrap()
}

fn enc<T: serde::Serialize>(value: &T) -> Vec<u8> {
    postcard::to_stdvec(value).unwrap()
}

fn sorted<T: Ord>(mut values: Vec<T>) -> Vec<T> {
    values.sort();
    values
}

fn records_after(backend: &impl Backend, after: Position) -> Vec<Record> {
    let mut out = Vec::new();
    assert_eq!(
        backend.read_from(after, &mut out).unwrap(),
        ReadOutcome::Records
    );
    out.into_iter().map(|(_, record)| record).collect()
}

/// The snapshot's records grouped by key, each group in a canonical order
/// so that element order, which follows hash iteration, does not matter.
fn snapshot_by_key(backend: &impl Backend) -> (Position, BTreeMap<String, Vec<Vec<u8>>>) {
    let (at, records) = backend.load_snapshot().unwrap().expect("a snapshot");
    let mut keys: BTreeMap<String, Vec<Vec<u8>>> = BTreeMap::new();
    for record in records {
        keys.entry(record.key().to_owned())
            .or_default()
            .push(record.encode().unwrap());
    }
    for group in keys.values_mut() {
        group.sort();
    }
    (at, keys)
}

fn group(records: &[Record]) -> Vec<Vec<u8>> {
    sorted(records.iter().map(|r| r.encode().unwrap()).collect())
}

fn declare(key: &str, kind: Kind, default: Vec<u8>) -> Record {
    Record::Declare {
        key: key.into(),
        kind,
        default,
    }
}

fn set(key: &str, value: Vec<u8>) -> Record {
    Record::Set {
        key: key.into(),
        value,
    }
}

fn insert(key: &str, element: Vec<u8>) -> Record {
    Record::Insert {
        key: key.into(),
        element,
    }
}

#[test]
fn checkpoint_folds_each_key_per_the_folding_rules() {
    let log = MemoryBackend::new();
    let store = open::<Local, _>(log.clone());
    let price = store.input("price", 0.0f64).unwrap();
    store.set(price, 1.0).unwrap();
    store.set(price, 2.5).unwrap();
    let back = store.input("back", 5u32).unwrap();
    store.set(back, 6).unwrap();
    store.set(back, 5).unwrap();
    store.input("idle", 1u32).unwrap();
    let tags = store.collection::<String>("tags").unwrap();
    for tag in ["a", "a", "b"] {
        tags.insert(&store, tag.to_owned()).unwrap();
    }
    tags.delete(&store, &"b".to_owned()).unwrap();
    let emptied = store.collection::<u32>("emptied").unwrap();
    emptied.insert(&store, 1).unwrap();
    emptied.delete(&store, &1).unwrap();

    log.append(&[
        declare("raw/in", Kind::Input, enc(&0u32)),
        set("raw/in", enc(&1u32)),
        set("raw/in", enc(&2u32)),
        declare("raw/del", Kind::Input, enc(&0u32)),
        set("raw/del", enc(&3u32)),
        Record::Delete {
            key: "raw/del".into(),
        },
        set("raw/undeclared", enc(&4u32)),
        declare("raw/col", Kind::Collection, vec![]),
        insert("raw/col", vec![1]),
        insert("raw/col", vec![1]),
        Record::Remove {
            key: "raw/col".into(),
            element: vec![1],
        },
        insert("raw/col", vec![2]),
    ])
    .unwrap();

    let at = store.checkpoint().unwrap();
    assert_eq!(at, log.append(&[]).unwrap(), "folds through the tail");
    assert!(records_after(&log, at).is_empty());

    let (snapshot_at, keys) = snapshot_by_key(&log);
    assert_eq!(snapshot_at, at);
    let expected = BTreeMap::from([
        (
            "price".to_owned(),
            group(&[
                declare("price", Kind::Input, enc(&0.0f64)),
                set("price", enc(&2.5f64)),
            ]),
        ),
        (
            "tags".to_owned(),
            group(&[
                declare("tags", Kind::Collection, vec![]),
                insert("tags", enc(&"a")),
                insert("tags", enc(&"a")),
            ]),
        ),
        (
            "raw/in".to_owned(),
            group(&[
                declare("raw/in", Kind::Input, enc(&0u32)),
                set("raw/in", enc(&2u32)),
            ]),
        ),
        (
            "raw/undeclared".to_owned(),
            group(&[set("raw/undeclared", enc(&4u32))]),
        ),
        (
            "raw/col".to_owned(),
            group(&[
                declare("raw/col", Kind::Collection, vec![]),
                insert("raw/col", vec![1]),
                insert("raw/col", vec![2]),
            ]),
        ),
    ]);
    assert_eq!(keys, expected);
    assert_eq!(
        store.restored_keys("").unwrap(),
        ["raw/col", "raw/in", "raw/undeclared"]
    );

    drop(store);
    let store = open::<Local, _>(log);
    let rt = store.runtime();
    assert_eq!(rt.get(store.input("price", 9.0f64).unwrap().handle()), 2.5);
    assert_eq!(rt.get(store.input("back", 9u32).unwrap().handle()), 9);
    assert_eq!(rt.get(store.input("idle", 9u32).unwrap().handle()), 9);
    let tags = store.collection::<String>("tags").unwrap();
    assert_eq!(tags.handle().values(rt), ["a", "a"]);
    assert_eq!(
        store.restored_keys("").unwrap(),
        ["raw/col", "raw/in", "raw/undeclared"]
    );
}

fn reader_behind_compaction<B: Backend>(writer_log: B, reader_log: B) {
    let writer = open::<Local, _>(writer_log);
    let items = writer.collection::<u32>("items").unwrap();
    let x = writer.input("x", 0u32).unwrap();
    for item in [1, 1, 2, 3] {
        items.insert(&writer, item).unwrap();
    }
    writer.set(x, 5).unwrap();

    let reader = open::<Local, _>(reader_log);
    let r_items = reader.collection::<u32>("items").unwrap();
    let r_x = reader.input("x", 0u32).unwrap();
    let rt = reader.runtime();
    let count = r_items.handle().count(rt);
    let sum = r_items
        .handle()
        .reduce(rt, |elements: &[u32]| elements.iter().sum::<u32>());
    let doubled = rt.create_query(move |rt| rt.get(sum) * 2);
    assert_eq!(sorted(r_items.handle().values(rt)), [1, 1, 2, 3]);
    assert_eq!(
        (rt.get(count), rt.get(doubled), rt.get(r_x.handle())),
        (4, 14, 5)
    );
    let version = rt.get(r_items.handle().version_node());

    items.delete(&writer, &1).unwrap();
    items.insert(&writer, 2).unwrap();
    items.insert(&writer, 4).unwrap();
    items.delete(&writer, &3).unwrap();
    writer.set(x, 0).unwrap();
    let at = writer.checkpoint().unwrap();

    reader.refresh().unwrap();
    assert!(reader.refresh().unwrap() >= at);
    assert_eq!(sorted(r_items.handle().values(rt)), [1, 2, 2, 4]);
    assert_eq!(
        rt.get(r_items.handle().version_node()) - version,
        4,
        "only the four differing occurrences change"
    );
    assert_eq!(
        (rt.get(count), rt.get(doubled), rt.get(r_x.handle())),
        (4, 18, 0)
    );

    items.insert(&writer, 7).unwrap();
    reader.refresh().unwrap();
    assert_eq!(sorted(r_items.handle().values(rt)), [1, 2, 2, 4, 7]);
    assert_eq!(rt.get(doubled), 32);
}

#[test]
fn a_reader_behind_compaction_ends_with_the_snapshot_multiset_on_memory() {
    let log = MemoryBackend::new();
    reader_behind_compaction(log.clone(), log.read_only());
}

#[test]
fn a_reader_behind_compaction_ends_with_the_snapshot_multiset_on_file() {
    let dir = tempfile::tempdir().unwrap();
    let writer = FileBackend::open(dir.path()).unwrap();
    let reader = FileBackend::open_read_only(dir.path()).unwrap();
    reader_behind_compaction(writer, reader);
}

#[cfg(feature = "sqlite")]
#[test]
fn a_reader_behind_compaction_ends_with_the_snapshot_multiset_on_sqlite() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.sqlite");
    let writer = incr_persist::SqliteBackend::open(&path).unwrap();
    let reader = incr_persist::SqliteBackend::open_read_only(&path).unwrap();
    reader_behind_compaction(writer, reader);
}

#[test]
fn a_checkpoint_on_file_restarts_to_the_same_state() {
    let dir = tempfile::tempdir().unwrap();
    {
        let store = open::<Local, _>(FileBackend::open(dir.path()).unwrap());
        let x = store.input("x", 0u32).unwrap();
        let items = store.collection::<String>("items").unwrap();
        for value in 1..=5 {
            store.set(x, value).unwrap();
            items.insert(&store, format!("v{}", value % 2)).unwrap();
        }
        store.checkpoint().unwrap();
        items.delete(&store, &"v0".to_owned()).unwrap();
    }
    let store = open::<Local, _>(FileBackend::open(dir.path()).unwrap());
    let x = store.input("x", 0u32).unwrap();
    let items = store.collection::<String>("items").unwrap();
    assert_eq!(store.runtime().get(x.handle()), 5);
    assert_eq!(
        sorted(items.handle().values(store.runtime())),
        ["v0", "v1", "v1", "v1"]
    );
}

#[test]
fn a_reader_replaces_a_collection_over_a_shared_runtime() {
    let log = MemoryBackend::new();
    let writer = open::<Shared, _>(log.clone());
    let reader = open::<Shared, _>(log.read_only());
    let items = writer.collection::<u32>("items").unwrap();
    for item in [1, 1, 2] {
        items.insert(&writer, item).unwrap();
    }
    reader.refresh().unwrap();
    let r_items = reader.collection::<u32>("items").unwrap();
    let count = r_items.handle().count(reader.runtime());
    assert_eq!(reader.runtime().get(count), 3);
    items.delete(&writer, &1).unwrap();
    items.delete(&writer, &1).unwrap();
    writer.checkpoint().unwrap();
    reader.refresh().unwrap();
    assert_eq!(r_items.handle().values(reader.runtime()), [2]);
    assert_eq!(reader.runtime().get(count), 1);
}

#[test]
fn a_key_deleted_back_to_its_default_is_forgotten_and_can_be_declared_afresh() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let k = a.input("k", 3u32).unwrap();
    a.set(k, 9).unwrap();
    {
        let b = open::<Local, _>(log.clone());
        b.input("j", 0u32).unwrap();
        let gone = b.collection::<u32>("gone").unwrap();
        gone.insert(&b, 1).unwrap();
    }
    log.append(&[
        Record::Delete { key: "k".into() },
        Record::Delete { key: "gone".into() },
    ])
    .unwrap();
    a.refresh().unwrap();
    assert_eq!(a.runtime().get(k.handle()), 3);
    assert_eq!(a.restored_keys("").unwrap(), ["gone", "j"]);

    a.checkpoint().unwrap();
    let (_, snapshot) = snapshot_by_key(&log);
    assert!(snapshot.is_empty(), "every key is at its default");
    assert!(a.restored_keys("").unwrap().is_empty());

    let c = open::<Local, _>(log.clone());
    assert!(c.restored_keys("").unwrap().is_empty());
    let c_k = c.input("k", 42u32).unwrap();
    assert_eq!(c.runtime().get(c_k.handle()), 42);
    let c_j = c.input("j", 7u32).unwrap();
    assert_eq!(c.runtime().get(c_j.handle()), 7);
    let gone = c.collection::<String>("gone").unwrap();
    assert!(gone.handle().values(c.runtime()).is_empty());
    let (snapshot_at, _) = log.load_snapshot().unwrap().unwrap();
    assert_eq!(
        records_after(&log, snapshot_at)
            .iter()
            .filter(|r| matches!(r, Record::Declare { .. }))
            .count(),
        3,
        "each forgotten key gets a fresh Declare"
    );

    a.refresh().unwrap();
    assert_eq!(
        a.runtime().get(k.handle()),
        42,
        "the fresh first-wins Declare is adopted by every store"
    );
}

/// The values of `k`, `s` and `u`, declared with a default no store uses
/// elsewhere, so a store that missed a live `Declare` would show it.
fn kus<C: Cells, B: Backend>(store: &Store<C, B>) -> [u32; 3] {
    ["k", "s", "u"].map(|key| {
        let input = store.input(key, 99u32).unwrap();
        store.runtime().get(input.handle())
    })
}

/// A store already at the position another writer compacts to reads
/// nothing of the compaction, yet when a forgotten key is declared afresh
/// it must adopt the new `Declare`, as a restart replaying the snapshot
/// does, rather than keep the old one the snapshot dropped.
#[test]
fn a_store_at_the_compacted_position_adopts_a_fresh_declare_of_a_key_forgotten_there() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let k = a.input("k", 1u32).unwrap();
    let s = a.input("s", 1u32).unwrap();
    a.set(s, 1).unwrap();
    let b = open::<Local, _>(log.clone());
    b.input("u", 1u32).unwrap();
    let applied = a.refresh().unwrap();
    assert_eq!(a.restored_keys("").unwrap(), ["u"]);
    assert_eq!(b.checkpoint().unwrap(), applied);
    let (_, snapshot) = snapshot_by_key(&log);
    assert!(snapshot.is_empty(), "every key was at its default");

    let c = open::<Local, _>(log.clone());
    for key in ["k", "s", "u"] {
        c.input(key, 5u32).unwrap();
    }
    a.refresh().unwrap();
    assert_eq!(kus(&a), [5; 3], "the store at the compacted position");
    assert_eq!(kus(&b), [5; 3], "the store that compacted");
    assert_eq!(kus(&c), [5; 3], "the store that declared afresh");
    assert_eq!(kus(&open::<Local, _>(log.clone())), [5; 3], "a restart");

    a.delete(k).unwrap();
    a.set(s, 8).unwrap();
    c.delete("s").unwrap();
    let expected = [5, 5, 5];
    for store in [&a, &b, &c] {
        store.refresh().unwrap();
        assert_eq!(kus(store), expected);
    }
    assert_eq!(kus(&open::<Local, _>(log)), expected);
}

/// A store at the position another writer compacts to reads no record of
/// the compaction, yet it must stop listing the keys the snapshot forgot,
/// as a restart does.
#[test]
fn a_store_at_the_compacted_position_stops_listing_keys_forgotten_there() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let refreshed = open::<Local, _>(log.clone());
    let b = open::<Local, _>(log.clone());
    let gone = b.input("visit/gone", 1u32).unwrap();
    let kept = b.input("visit/kept", 1u32).unwrap();
    b.set(gone, 2).unwrap();
    b.set(kept, 2).unwrap();
    b.delete(gone).unwrap();
    let applied = a.refresh().unwrap();
    refreshed.refresh().unwrap();
    assert_eq!(
        a.restored_keys("visit/").unwrap(),
        ["visit/gone", "visit/kept"]
    );

    assert_eq!(b.checkpoint().unwrap(), applied);
    assert_eq!(a.restored_keys("visit/").unwrap(), ["visit/kept"]);
    assert_eq!(refreshed.refresh().unwrap(), applied);
    assert_eq!(refreshed.restored_keys("visit/").unwrap(), ["visit/kept"]);
    let restart = open::<Local, _>(log);
    assert_eq!(restart.restored_keys("visit/").unwrap(), ["visit/kept"]);
}

/// Declaring a key a compaction at the store's position forgot appends a
/// fresh `Declare` with the caller's default, as a restart would, instead
/// of binding the default the snapshot dropped.
#[test]
fn declaring_a_key_forgotten_at_the_store_position_appends_a_fresh_declare() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let b = open::<Local, _>(log.clone());
    let b_k = b.input("k", 1u32).unwrap();
    let b_items = b.collection::<u32>("items").unwrap();
    b_items.insert(&b, 3).unwrap();
    b_items.delete(&b, &3).unwrap();
    let applied = a.refresh().unwrap();
    assert_eq!(b.checkpoint().unwrap(), applied);

    let k = a.input("k", 5u32).unwrap();
    assert_eq!(a.runtime().get(k.handle()), 5);
    let items = a.collection::<u32>("items").unwrap();
    items.insert(&a, 4).unwrap();
    assert_eq!(
        records_after(&log, applied),
        [
            declare("k", Kind::Input, enc(&5u32)),
            declare("items", Kind::Collection, vec![]),
            insert("items", enc(&4u32)),
        ]
    );

    let restart = open::<Local, _>(log.clone());
    let r_k = restart.input("k", 7u32).unwrap();
    assert_eq!(restart.runtime().get(r_k.handle()), 5);
    b.refresh().unwrap();
    assert_eq!(b.runtime().get(b_k.handle()), 5, "the fresh Declare wins");
    b.delete(b_k).unwrap();
    a.refresh().unwrap();
    assert_eq!(a.runtime().get(k.handle()), 5);
    assert_eq!(b.runtime().get(b_k.handle()), 5);
}

/// A file reader behind a compaction may read through it record by record
/// instead of loading the snapshot; it still learns what the snapshot
/// forgot, including a key declared afresh right after it.
#[test]
fn a_file_reader_that_reads_through_a_compaction_learns_what_it_forgot() {
    let dir = tempfile::tempdir().unwrap();
    let writer = open::<Local, _>(FileBackend::open(dir.path()).unwrap());
    let reader = open::<Local, _>(FileBackend::open_read_only(dir.path()).unwrap());
    let k = writer.input("k", 1u32).unwrap();
    let x = writer.input("x", 0u32).unwrap();
    writer.input("j", 2u32).unwrap();
    reader.refresh().unwrap();
    let r_k = reader.input("k", 0u32).unwrap();
    assert_eq!(reader.restored_keys("").unwrap(), ["j", "x"]);
    writer.set(x, 1).unwrap();
    let at = writer.checkpoint().unwrap();
    writer.undeclare(k).unwrap();
    writer.input("k", 6u32).unwrap();

    assert!(reader.refresh().unwrap() > at);
    assert_eq!(reader.runtime().get(r_k.handle()), 6);
    assert_eq!(reader.restored_keys("").unwrap(), ["x"]);
    let restart = open::<Local, _>(FileBackend::open_read_only(dir.path()).unwrap());
    let restarted = restart.input("k", 0u32).unwrap();
    assert_eq!(restart.runtime().get(restarted.handle()), 6);
}

/// A snapshot replaced at the same position may forget more than the one it
/// replaced: a store that declared a key compares its value with the default
/// by decoded value, one that did not by bytes, and `-0.0` equals `0.0` only
/// by value. A store that accounted for the first snapshot must still learn
/// what the second forgot.
#[test]
fn a_snapshot_replaced_at_the_same_position_is_taken_into_account_too() {
    let log = MemoryBackend::new();
    let declarer = open::<Local, _>(log.clone());
    let z = declarer.input("z", 0.0f64).unwrap();
    declarer.set(z, -0.0).unwrap();
    let other = open::<Local, _>(log.clone());
    let at = other.checkpoint().unwrap();
    let (_, snapshot) = snapshot_by_key(&log);
    assert!(snapshot.contains_key("z"), "bytes differ from the default");
    assert_eq!(other.restored_keys("").unwrap(), ["z"]);

    assert_eq!(declarer.checkpoint().unwrap(), at);
    let (_, snapshot) = snapshot_by_key(&log);
    assert!(!snapshot.contains_key("z"), "equal to the default by value");
    assert!(other.restored_keys("").unwrap().is_empty());
    assert!(open::<Local, _>(log).restored_keys("").unwrap().is_empty());
}

/// Runs `race` inside the next `compact`, before forwarding it, the way
/// another writer compacting between this store's catch-up and its
/// compaction would.
struct Racing {
    inner: MemoryBackend,
    race: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl Backend for Racing {
    fn append(&self, records: &[Record]) -> incr_persist::Result<Position> {
        self.inner.append(records)
    }

    fn sync(&self, upto: Position) -> incr_persist::Result<()> {
        self.inner.sync(upto)
    }

    fn read_from(
        &self,
        after: Position,
        out: &mut Vec<(Position, Record)>,
    ) -> incr_persist::Result<ReadOutcome> {
        self.inner.read_from(after, out)
    }

    fn load_snapshot(&self) -> incr_persist::Result<Option<(Position, Vec<Record>)>> {
        self.inner.load_snapshot()
    }

    fn snapshot_id(&self) -> incr_persist::Result<Option<incr_persist::SnapshotId>> {
        self.inner.snapshot_id()
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> incr_persist::Result<()> {
        let race = self.race.lock().unwrap().take();
        if let Some(race) = race {
            race();
        }
        self.inner.compact(upto, folded)
    }

    fn writable(&self) -> bool {
        self.inner.writable()
    }
}

#[test]
fn a_compaction_by_another_writer_is_caught_up_and_retried() {
    let log = MemoryBackend::new();
    let other = Arc::new(open::<Shared, _>(log.clone()));
    let other_x = other.input("x", 0u32).unwrap();
    other.set(other_x, 1).unwrap();

    let racer = Arc::clone(&other);
    let store = open::<Local, _>(Racing {
        inner: log.clone(),
        // Another writer is another process: it never runs under this
        // store's mutex on this thread.
        race: Mutex::new(Some(Box::new(move || {
            std::thread::spawn(move || {
                racer.set(other_x, 2).unwrap();
                racer.checkpoint().unwrap();
            })
            .join()
            .unwrap();
        }))),
    });
    let x = store.input("x", 0u32).unwrap();
    let doubled = {
        let x = x.handle();
        store.runtime().create_query(move |rt| rt.get(x) * 2)
    };
    assert_eq!(store.runtime().get(doubled), 2);

    let at = store.checkpoint().unwrap();
    let (snapshot_at, snapshot) = snapshot_by_key(&log);
    assert_eq!(at, snapshot_at);
    assert_eq!(
        snapshot["x"],
        group(&[declare("x", Kind::Input, enc(&0u32)), set("x", enc(&2u32))])
    );
    assert_eq!(store.runtime().get(doubled), 4);
    store.set(x, 3).unwrap();
    assert_eq!(store.runtime().get(doubled), 6);
}

/// Fails `compact` with an I/O error, either before forwarding it or after
/// the inner backend already compacted, the two outcomes a caller cannot
/// tell apart.
struct FailingCompact {
    inner: MemoryBackend,
    after_compacting: bool,
    armed: AtomicBool,
}

impl Backend for FailingCompact {
    fn append(&self, records: &[Record]) -> incr_persist::Result<Position> {
        self.inner.append(records)
    }

    fn sync(&self, upto: Position) -> incr_persist::Result<()> {
        self.inner.sync(upto)
    }

    fn read_from(
        &self,
        after: Position,
        out: &mut Vec<(Position, Record)>,
    ) -> incr_persist::Result<ReadOutcome> {
        self.inner.read_from(after, out)
    }

    fn load_snapshot(&self) -> incr_persist::Result<Option<(Position, Vec<Record>)>> {
        self.inner.load_snapshot()
    }

    fn snapshot_id(&self) -> incr_persist::Result<Option<incr_persist::SnapshotId>> {
        self.inner.snapshot_id()
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> incr_persist::Result<()> {
        if !self.armed.swap(false, Ordering::SeqCst) {
            return self.inner.compact(upto, folded);
        }
        if self.after_compacting {
            self.inner.compact(upto, folded)?;
        }
        Err(Error::Io(io::Error::other("injected compaction failure")))
    }

    fn writable(&self) -> bool {
        self.inner.writable()
    }
}

#[test]
fn a_failed_compaction_leaves_the_store_in_step_with_the_log() {
    for after_compacting in [false, true] {
        let log = MemoryBackend::new();
        open::<Local, _>(log.clone()).input("idle", 0u32).unwrap();
        let store = open::<Local, _>(FailingCompact {
            inner: log.clone(),
            after_compacting,
            armed: AtomicBool::new(true),
        });
        let x = store.input("x", 0u32).unwrap();
        store.set(x, 4).unwrap();

        assert!(matches!(store.checkpoint(), Err(Error::Io(_))));
        let expected: &[&str] = if after_compacting { &[] } else { &["idle"] };
        assert_eq!(store.restored_keys("").unwrap(), expected);
        store.set(x, 5).unwrap();
        assert_eq!(store.runtime().get(x.handle()), 5);

        store.checkpoint().unwrap();
        assert!(store.restored_keys("").unwrap().is_empty());
        let reopened = open::<Local, _>(log);
        let x = reopened.input("x", 0u32).unwrap();
        assert_eq!(reopened.runtime().get(x.handle()), 5);
    }
}

#[test]
fn checkpoint_on_a_read_only_store_fails_without_poisoning() {
    let log = MemoryBackend::new();
    let writer = open::<Local, _>(log.clone());
    let x = writer.input("x", 0u32).unwrap();
    writer.set(x, 1).unwrap();
    let reader = open::<Local, _>(log.read_only());
    assert!(matches!(reader.checkpoint(), Err(Error::ReadOnly)));
    let x = reader.input("x", 0u32).unwrap();
    assert_eq!(reader.runtime().get(x.handle()), 1);
}

#[test]
fn auto_checkpoint_compacts_at_the_next_write_past_the_threshold() {
    let log = MemoryBackend::new();
    let options = Options {
        auto_checkpoint_records: Some(4),
        ..Options::default()
    };
    {
        let store = Store::<Local, _>::open(log.clone(), options.clone()).unwrap();
        let x = store.input("x", 0u32).unwrap();
        for value in 1..=10 {
            store.set(x, value).unwrap();
        }
    }
    let (at, _) = log.load_snapshot().unwrap().expect("a checkpoint ran");
    assert!(records_after(&log, at).len() < 4);

    let store = Store::<Local, _>::open(log.clone(), options).unwrap();
    let x = store.input("x", 0u32).unwrap();
    assert_eq!(store.runtime().get(x.handle()), 10);
}

#[test]
fn checkpoint_without_auto_threshold_never_runs_on_its_own() {
    let log = MemoryBackend::new();
    let store = open::<Local, _>(log.clone());
    let x = store.input("x", 0u32).unwrap();
    for value in 1..=100 {
        store.set(x, value).unwrap();
    }
    assert!(log.load_snapshot().unwrap().is_none());
}
