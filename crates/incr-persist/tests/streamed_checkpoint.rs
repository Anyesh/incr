//! A checkpoint hands the backend its snapshot through `compact_streamed`,
//! folding each record as the backend pulls it.

use std::cell::Cell;
use std::sync::{Arc, Mutex};

use incr_persist::{
    Backend, Error, Local, MemoryBackend, Options, PersistedKeyed, Position, ReadOutcome, Record,
    Result, Shared, SnapshotId, Store,
};
use serde::{Deserialize, Serialize, Serializer};

thread_local! {
    static ENCODES: Cell<usize> = const { Cell::new(0) };
    static FAIL_ON: Cell<Option<u32>> = const { Cell::new(None) };
}

/// A row whose encodes are counted, and whose encode fails for the value
/// `FAIL_ON` names.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Deserialize)]
struct Counted(u32);

impl Serialize for Counted {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        ENCODES.with(|n| n.set(n.get() + 1));
        if FAIL_ON.with(Cell::get) == Some(self.0) {
            return Err(serde::ser::Error::custom("refused to encode"));
        }
        s.serialize_u32(self.0)
    }
}

fn encodes() -> usize {
    ENCODES.with(Cell::get)
}

type Pull = dyn Fn(&MemoryBackend, Position, &mut dyn Iterator<Item = Result<Record>>) -> Result<()>
    + Send
    + Sync;

/// A [`MemoryBackend`] whose `compact_streamed` is `pull`.
struct Pulling {
    inner: MemoryBackend,
    pull: Box<Pull>,
}

impl Pulling {
    fn new(
        inner: &MemoryBackend,
        pull: impl Fn(&MemoryBackend, Position, &mut dyn Iterator<Item = Result<Record>>) -> Result<()>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            inner: inner.clone(),
            pull: Box::new(pull),
        }
    }
}

impl Backend for Pulling {
    fn append(&self, records: &[Record]) -> Result<Position> {
        self.inner.append(records)
    }

    fn sync(&self, upto: Position) -> Result<()> {
        self.inner.sync(upto)
    }

    fn read_from(&self, after: Position, out: &mut Vec<(Position, Record)>) -> Result<ReadOutcome> {
        self.inner.read_from(after, out)
    }

    fn load_snapshot(&self) -> Result<Option<(Position, Vec<Record>)>> {
        self.inner.load_snapshot()
    }

    fn snapshot_id(&self) -> Result<Option<SnapshotId>> {
        self.inner.snapshot_id()
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> Result<()> {
        self.compact_streamed(upto, &mut folded.into_iter().map(Ok))
    }

    fn compact_streamed(
        &self,
        upto: Position,
        records: &mut dyn Iterator<Item = Result<Record>>,
    ) -> Result<()> {
        (self.pull)(&self.inner, upto, records)
    }

    fn writable(&self) -> bool {
        self.inner.writable()
    }
}

type Rows<C> = PersistedKeyed<u32, Counted, C>;

/// A store over `backend` holding an input and `n` keyed rows.
fn populated<B: Backend>(backend: B, n: u32) -> (Store<Local, B>, Rows<Local>) {
    let store = Store::<Local, _>::open(backend, Options::default()).unwrap();
    let x = store.input("x", 0u32).unwrap();
    store.set(x, 1).unwrap();
    let rows: Rows<Local> = store.keyed_collection("rows").unwrap();
    for key in 0..n {
        store.upsert(&rows, key, Counted(key)).unwrap();
    }
    (store, rows)
}

/// A log's snapshot and the records after it.
type LogState = (Option<(Position, Vec<Record>)>, Vec<(Position, Record)>);

fn log_state(log: &MemoryBackend) -> LogState {
    let snapshot = log.load_snapshot().unwrap();
    let mut out = Vec::new();
    let after = snapshot.as_ref().map_or(Position::ZERO, |(at, _)| *at);
    assert_eq!(
        log.read_from(after, &mut out).unwrap(),
        ReadOutcome::Records
    );
    (snapshot, out)
}

/// Each keyed row is encoded as the backend pulls it, not all of them
/// before the backend sees the first.
#[test]
fn a_checkpoint_encodes_keyed_rows_as_the_backend_pulls_them() {
    let log = MemoryBackend::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let pulled = Arc::clone(&seen);
    let (store, _rows) = populated(
        Pulling::new(&log, move |inner, upto, records| {
            let mut kept = Vec::new();
            for record in records {
                let record = record?;
                if matches!(&record, Record::Insert { key, .. } if key == "rows") {
                    pulled.lock().unwrap().push(encodes());
                }
                kept.push(record);
            }
            inner.compact(upto, kept)
        }),
        100,
    );
    let before = encodes();
    store.checkpoint().unwrap();
    let seen: Vec<usize> = seen.lock().unwrap().iter().map(|n| n - before).collect();
    assert_eq!(
        seen,
        (1..=100).collect::<Vec<_>>(),
        "one encode per pulled row"
    );
}

#[test]
fn a_backend_returning_with_records_left_poisons_the_store() {
    let log = MemoryBackend::new();
    let (store, rows) = populated(
        Pulling::new(&log, |inner, upto, records| {
            let first = records.take(2).collect::<Result<Vec<_>>>()?;
            inner.compact(upto, first)
        }),
        10,
    );
    let err = store.checkpoint().unwrap_err();
    assert!(matches!(err, Error::Corrupt(_)), "{err:?}");
    assert!(matches!(
        store.upsert(&rows, 1, Counted(1)),
        Err(Error::Poisoned)
    ));
}

#[test]
fn a_backend_swallowing_a_pulled_error_poisons_the_store() {
    let log = MemoryBackend::new();
    let (store, rows) = populated(
        Pulling::new(&log, |inner, upto, records| {
            let kept = records.filter_map(Result::ok).collect();
            inner.compact(upto, kept)
        }),
        10,
    );
    FAIL_ON.with(|f| f.set(Some(5)));
    let err = store.checkpoint().unwrap_err();
    FAIL_ON.with(|f| f.set(None));
    assert!(matches!(err, Error::Corrupt(_)), "{err:?}");
    assert!(matches!(
        store.upsert(&rows, 1, Counted(1)),
        Err(Error::Poisoned)
    ));
}

/// A row that fails to encode part-way through the backend's write fails
/// the checkpoint with that error and changes neither the log nor the
/// store, and nothing is forgotten.
#[test]
fn an_encode_failing_mid_checkpoint_changes_nothing() {
    let log = MemoryBackend::new();
    let (store, rows) = populated(log.clone(), 10);
    let gone = store.input("a/gone", 3u32).unwrap();
    store.set(gone, 4).unwrap();
    store.delete(gone).unwrap();
    store.checkpoint().unwrap();
    {
        let other = Store::<Local, _>::open(log.clone(), Options::default()).unwrap();
        let back = other.input("a/back", 3u32).unwrap();
        other.set(back, 4).unwrap();
        other.delete(back).unwrap();
    }
    store.refresh().unwrap();
    store.upsert(&rows, 3, Counted(30)).unwrap();
    let before = log_state(&log);
    let listed = store.restored_keys("a/").unwrap();
    assert_eq!(listed, ["a/back"]);

    FAIL_ON.with(|f| f.set(Some(7)));
    let err = store.checkpoint().unwrap_err();
    FAIL_ON.with(|f| f.set(None));
    assert!(matches!(err, Error::Encode(_)), "{err:?}");
    assert_eq!(log_state(&log), before);
    assert_eq!(store.restored_keys("a/").unwrap(), listed);

    store.upsert(&rows, 4, Counted(40)).unwrap();
    store.checkpoint().unwrap();
    assert!(store.restored_keys("a/").unwrap().is_empty());
    let reopened = Store::<Local, _>::open(log, Options::default()).unwrap();
    let rows: Rows<Local> = reopened.keyed_collection("rows").unwrap();
    let rt = reopened.runtime();
    assert_eq!(rows.get_key(rt, &3), Some(Counted(30)));
    assert_eq!(rows.get_key(rt, &4), Some(Counted(40)));
    assert_eq!(rows.get_key(rt, &9), Some(Counted(9)));
}

/// A compaction another writer overtook mid-stream is retried from a fresh
/// fold, so the snapshot holds every record, not the ones left after the
/// first attempt's pulls.
#[test]
fn a_compaction_overtaken_mid_stream_is_retried_from_a_fresh_fold() {
    let log = MemoryBackend::new();
    let other = Arc::new(Store::<Shared, _>::open(log.clone(), Options::default()).unwrap());
    let other_y = other.input("y", 0u32).unwrap();
    let race = Mutex::new(Some(move || {
        std::thread::spawn(move || {
            other.set(other_y, 2).unwrap();
            other.checkpoint().unwrap();
        })
        .join()
        .unwrap();
    }));
    let (store, _rows) = populated(
        Pulling::new(&log, move |inner, upto, records| {
            if let Some(race) = race.lock().unwrap().take() {
                records.take(3).for_each(drop);
                race();
            }
            inner.compact_streamed(upto, records)
        }),
        10,
    );
    store.checkpoint().unwrap();
    let reopened = Store::<Local, _>::open(log, Options::default()).unwrap();
    let rows: Rows<Local> = reopened.keyed_collection("rows").unwrap();
    let rt = reopened.runtime();
    for key in 0..10 {
        assert_eq!(rows.get_key(rt, &key), Some(Counted(key)));
    }
    let x = reopened.input("x", 0u32).unwrap();
    assert_eq!(rt.get(x.handle()), 1);
    let y = reopened.input("y", 0u32).unwrap();
    assert_eq!(rt.get(y.handle()), 2);
}

/// A write around the store between the fold listing a keyed collection's
/// pairs and pulling one of them fails the checkpoint instead of yielding
/// whatever pair the listing's handle now names.
#[test]
fn a_keyed_write_around_the_store_mid_checkpoint_fails_it() {
    type Bypass = Box<dyn FnOnce() + Send>;
    let log = MemoryBackend::new();
    let hook: Arc<Mutex<Option<Bypass>>> = Arc::new(Mutex::new(None));
    let fire = Arc::clone(&hook);
    let backend = Pulling::new(&log, move |inner, upto, records| {
        let mut kept = Vec::new();
        for record in records {
            let record = record?;
            if matches!(&record, Record::Insert { key, .. } if key == "rows") {
                if let Some(write) = fire.lock().unwrap().take() {
                    write();
                }
            }
            kept.push(record);
        }
        inner.compact(upto, kept)
    });
    let store = Arc::new(Store::<Shared, _>::open(backend, Options::default()).unwrap());
    let rows: Rows<Shared> = store.keyed_collection("rows").unwrap();
    for key in 0..10 {
        store.upsert(&rows, key, Counted(key)).unwrap();
    }
    let core = {
        let scope = store.unlogged_scope().unwrap();
        let core = scope.keyed(&rows).unwrap();
        scope.close().unwrap();
        core
    };
    let weak = Arc::downgrade(&store);
    *hook.lock().unwrap() = Some(Box::new(move || {
        let store = weak.upgrade().expect("the store outlives its checkpoint");
        core.upsert(store.runtime(), 99, Counted(99));
    }));
    let before = log_state(&log);
    let err = store.checkpoint().unwrap_err();
    assert!(
        matches!(&err, Error::Bypassed { key } if key == "rows"),
        "{err:?}"
    );
    assert_eq!(log_state(&log), before);
}
