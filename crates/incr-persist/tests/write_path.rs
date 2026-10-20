use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use incr_persist::{
    Backend, Cells, Local, MemoryBackend, Options, Origin, Position, ReadOutcome, Record, Shared,
    Store,
};
use serde::{Deserialize, Serialize};

type Seen = Arc<Mutex<Vec<(Position, Record, Origin)>>>;
type Hook = Box<dyn FnOnce() + Send>;

fn open<C: Cells, B: Backend>(backend: B) -> Store<C, B> {
    Store::open(backend, Options::default()).unwrap()
}

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

fn log_after(backend: &impl Backend, after: Position) -> Vec<(Position, Record)> {
    let mut out = Vec::new();
    assert_eq!(
        backend.read_from(after, &mut out).unwrap(),
        ReadOutcome::Records
    );
    out
}

/// A memory log whose `append` and `sync` first run a hook, standing in for
/// another writer that acts between a store's catch-up and its append, or
/// between its append and its apply. It counts the calls that query the
/// log, its snapshot or the snapshot's id.
#[derive(Clone, Default)]
struct Hooked {
    inner: MemoryBackend,
    before_append: Arc<Mutex<Option<Hook>>>,
    before_sync: Arc<Mutex<Option<Hook>>>,
    queries: Arc<AtomicUsize>,
}

impl Hooked {
    fn before_append(&self, hook: impl FnOnce() + Send + 'static) {
        *self.before_append.lock().unwrap() = Some(Box::new(hook));
    }

    fn before_sync(&self, hook: impl FnOnce() + Send + 'static) {
        *self.before_sync.lock().unwrap() = Some(Box::new(hook));
    }
}

/// Runs the hook on a thread of its own: the other writer it plays is
/// another process, which never runs under this store's mutex.
fn run(hook: &Mutex<Option<Hook>>) {
    let hook = hook.lock().unwrap().take();
    if let Some(hook) = hook {
        std::thread::spawn(hook).join().unwrap();
    }
}

impl Backend for Hooked {
    fn append(&self, records: &[Record]) -> incr_persist::Result<Position> {
        run(&self.before_append);
        self.inner.append(records)
    }

    fn sync(&self, upto: Position) -> incr_persist::Result<()> {
        run(&self.before_sync);
        self.inner.sync(upto)
    }

    fn read_from(
        &self,
        after: Position,
        out: &mut Vec<(Position, Record)>,
    ) -> incr_persist::Result<ReadOutcome> {
        self.queries.fetch_add(1, Ordering::SeqCst);
        self.inner.read_from(after, out)
    }

    fn load_snapshot(&self) -> incr_persist::Result<Option<(Position, Vec<Record>)>> {
        self.queries.fetch_add(1, Ordering::SeqCst);
        self.inner.load_snapshot()
    }

    fn snapshot_id(&self) -> incr_persist::Result<Option<incr_persist::SnapshotId>> {
        self.queries.fetch_add(1, Ordering::SeqCst);
        self.inner.snapshot_id()
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> incr_persist::Result<()> {
        self.inner.compact(upto, folded)
    }

    fn writable(&self) -> bool {
        self.inner.writable()
    }
}

#[derive(Debug, PartialEq)]
struct View {
    x: u64,
    y: u64,
    items: Vec<u64>,
}

fn view<C: Cells, B: Backend>(store: &Store<C, B>) -> View {
    let x = store.input("x", 0u64).unwrap();
    let y = store.input("y", 0u64).unwrap();
    let items = store.collection::<u64>("items").unwrap();
    let rt = store.runtime();
    let mut values = items.handle().values(rt);
    values.sort();
    View {
        x: rt.get(x.handle()),
        y: rt.get(y.handle()),
        items: values,
    }
}

/// Asserts that `seen` is exactly the log after `from`, in position order,
/// with `own` positions reported as local and every other one as foreign.
fn assert_saw_log(seen: &Seen, log: &impl Backend, from: Position, own: &BTreeSet<Position>) {
    let seen = seen.lock().unwrap();
    let expected = log_after(log, from);
    assert_eq!(seen.len(), expected.len(), "every record is reported once");
    for ((position, record, origin), (at, logged)) in seen.iter().zip(&expected) {
        assert_eq!((position, record), (at, logged), "in position order");
        let want = if own.contains(position) {
            Origin::Local
        } else {
            Origin::Foreign
        };
        assert_eq!(*origin, want, "origin of position {position}");
    }
}

#[test]
fn foreign_records_between_own_writes_apply_in_log_order_with_their_origin() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let b = open::<Local, _>(log.clone());
    view(&a);
    view(&b);
    let from = a.refresh().unwrap();
    let seen = subscribe(&a);
    let (a_x, a_y) = (a.input("x", 0u64).unwrap(), a.input("y", 0u64).unwrap());
    let a_items = a.collection::<u64>("items").unwrap();
    let (b_x, b_items) = (
        b.input("x", 0u64).unwrap(),
        b.collection::<u64>("items").unwrap(),
    );

    let mut own = BTreeSet::new();
    own.insert(a.set(a_x, 1).unwrap());
    b.set(b_x, 2).unwrap();
    b_items.insert(&b, 10).unwrap();
    own.insert(a.set(a_y, 3).unwrap());
    assert_eq!(view(&a).x, 2, "the write applied the foreign set first");
    own.insert(a_items.insert(&a, 11).unwrap());
    b_items.delete(&b, &10).unwrap();
    let end = a
        .batch(|batch| {
            batch.set(a_x, 4).insert(&a_items, 12).delete(&a_items, &11);
        })
        .unwrap();
    own.extend((end.0 - 2..=end.0).map(Position));
    b.set(b_x, 5).unwrap();
    own.insert(a.set(a_y, 6).unwrap());

    assert_saw_log(&seen, &log, from, &own);
    let expected = View {
        x: 5,
        y: 6,
        items: vec![12],
    };
    assert_eq!(view(&a), expected);
    assert_eq!(view(&open::<Local, _>(log.clone())), expected);
}

#[test]
fn a_foreign_append_racing_the_append_applies_before_own_records() {
    let backend = Hooked::default();
    let a = open::<Local, _>(backend.clone());
    let b = Arc::new(open::<Shared, _>(backend.inner.clone()));
    view(&a);
    view(&*b);
    let from = a.refresh().unwrap();
    let seen = subscribe(&a);
    let a_y = a.input("y", 0u64).unwrap();

    let other = Arc::clone(&b);
    backend.before_append(move || {
        let x = other.input("x", 0u64).unwrap();
        other.set(x, 7).unwrap();
        let items = other.collection::<u64>("items").unwrap();
        items.insert(&other, 8).unwrap();
    });
    let at = a.set(a_y, 9).unwrap();

    assert_eq!(at, Position(from.0 + 3));
    assert_saw_log(&seen, &backend.inner, from, &BTreeSet::from([at]));
    let expected = View {
        x: 7,
        y: 9,
        items: vec![8],
    };
    assert_eq!(view(&a), expected);
    assert_eq!(view(&open::<Local, _>(backend.inner.clone())), expected);
}

#[test]
fn a_compaction_past_applied_racing_the_append_is_crossed_before_own_records() {
    let backend = Hooked::default();
    let a = open::<Local, _>(backend.clone());
    let b = Arc::new(open::<Shared, _>(backend.inner.clone()));
    view(&a);
    view(&*b);
    let seen = subscribe(&a);
    let a_y = a.input("y", 0u64).unwrap();
    let a_items = a.collection::<u64>("items").unwrap();
    a_items.insert(&a, 1).unwrap();
    seen.lock().unwrap().clear();

    let other = Arc::clone(&b);
    let snapshot_at = Arc::new(Mutex::new(Position::ZERO));
    let compacted = Arc::clone(&snapshot_at);
    backend.before_append(move || {
        let x = other.input("x", 0u64).unwrap();
        other.set(x, 7).unwrap();
        let items = other.collection::<u64>("items").unwrap();
        items.insert(&other, 2).unwrap();
        *compacted.lock().unwrap() = other.checkpoint().unwrap();
    });
    let at = a.set(a_y, 9).unwrap();
    let snapshot_at = *snapshot_at.lock().unwrap();

    assert_eq!(at, Position(snapshot_at.0 + 1));
    let seen = seen.lock().unwrap();
    let (last, crossed) = seen.split_last().unwrap();
    assert_eq!(
        *last,
        (
            at,
            Record::Set {
                key: "y".into(),
                value: postcard::to_stdvec(&9u64).unwrap(),
            },
            Origin::Local,
        )
    );
    assert!(!crossed.is_empty());
    assert!(crossed
        .iter()
        .all(|(position, _, origin)| *position == snapshot_at && *origin == Origin::Snapshot));
    let expected = View {
        x: 7,
        y: 9,
        items: vec![1, 2],
    };
    assert_eq!(view(&a), expected);
    assert_eq!(view(&open::<Local, _>(backend.inner.clone())), expected);
}

#[test]
fn a_compaction_at_applied_racing_the_append_leaves_the_state_a_restart_reaches() {
    let backend = Hooked::default();
    let a = open::<Local, _>(backend.clone());
    let b = Arc::new(open::<Shared, _>(backend.inner.clone()));
    let (a_x, a_y) = (a.input("x", 0u64).unwrap(), a.input("y", 0u64).unwrap());
    let a_items = a.collection::<u64>("items").unwrap();
    a.set(a_x, 3).unwrap();
    a_items.insert(&a, 1).unwrap();
    a.set(a_y, 4).unwrap();
    a.delete(a_y).unwrap();
    let applied = a.refresh().unwrap();
    let seen = subscribe(&a);

    let other = Arc::clone(&b);
    backend.before_append(move || {
        assert_eq!(other.checkpoint().unwrap(), applied);
    });
    let at = a.set(a_x, 5).unwrap();

    assert_eq!(at, Position(applied.0 + 1));
    assert_saw_log(&seen, &backend.inner, applied, &BTreeSet::from([at]));
    let expected = View {
        x: 5,
        y: 0,
        items: vec![1],
    };
    assert_eq!(view(&a), expected);
    assert_eq!(view(&open::<Local, _>(backend.inner.clone())), expected);
    a.checkpoint().unwrap();
    assert_eq!(view(&open::<Local, _>(backend.inner.clone())), expected);
}

#[test]
fn a_compaction_covering_own_records_before_they_apply_leaves_the_state_a_restart_reaches() {
    let backend = Hooked::default();
    let a = open::<Local, _>(backend.clone());
    let b = Arc::new(open::<Shared, _>(backend.inner.clone()));
    view(&a);
    view(&*b);
    let a_x = a.input("x", 0u64).unwrap();
    let a_items = a.collection::<u64>("items").unwrap();
    a_items.insert(&a, 1).unwrap();

    let other = Arc::clone(&b);
    backend.before_sync(move || {
        let y = other.input("y", 0u64).unwrap();
        other.set(y, 6).unwrap();
        other.checkpoint().unwrap();
    });
    let at = a.set(a_x, 5).unwrap();
    assert_eq!(view(&a).x, 5);

    let expected = View {
        x: 5,
        y: 6,
        items: vec![1],
    };
    assert_eq!(a.refresh().unwrap(), Position(at.0 + 1));
    assert_eq!(view(&a), expected);
    assert_eq!(view(&open::<Local, _>(backend.inner.clone())), expected);
}

/// Serializes its map in iteration order, which differs between equal maps,
/// and drops `cache`, which its `Eq` still compares, so a value read back
/// from the log can differ from the one written.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct Lossy {
    tags: HashMap<String, u32>,
    #[serde(skip)]
    cache: u32,
}

#[test]
fn a_written_value_reads_back_as_replaying_the_log_decodes_it() {
    let log = MemoryBackend::new();
    let store = open::<Local, _>(log.clone());
    let doc = store.input("doc", Lossy::default()).unwrap();
    let tags: HashMap<String, u32> = (0..32).map(|i| (format!("tag{i}"), i)).collect();
    let written = Lossy {
        tags: tags.clone(),
        cache: 7,
    };
    store.set(doc, written).unwrap();
    let live = store.runtime().get(doc.handle());
    assert_eq!(
        live,
        Lossy { tags, cache: 0 },
        "the runtime holds the logged value"
    );

    let restarted = open::<Local, _>(log.clone());
    let replayed = restarted.input("doc", Lossy::default()).unwrap();
    assert_eq!(restarted.runtime().get(replayed.handle()), live);

    let mut next = live.clone();
    next.tags.insert("extra".into(), 99);
    store.set(doc, next).unwrap();
    store.checkpoint().unwrap();
    let restarted = open::<Local, _>(log);
    let replayed = restarted.input("doc", Lossy::default()).unwrap();
    assert_eq!(
        restarted.runtime().get(replayed.handle()),
        store.runtime().get(doc.handle())
    );
}

/// Own writes with nothing foreign before them apply as held, without
/// querying the backend, even once another writer has compacted at the
/// store's position; the next catch-up deals with that compaction.
#[test]
fn own_writes_with_nothing_foreign_before_them_query_the_backend_for_nothing() {
    let backend = Hooked::default();
    let a = open::<Local, _>(backend.clone());
    let x = a.input("x", 0u64).unwrap();
    let items = a.collection::<u64>("items").unwrap();
    let applied = a.refresh().unwrap();
    assert_eq!(
        open::<Local, _>(backend.inner.clone())
            .checkpoint()
            .unwrap(),
        applied
    );

    backend.queries.store(0, Ordering::SeqCst);
    a.set(x, 1).unwrap();
    items.insert(&a, 2).unwrap();
    a.batch(|batch| {
        batch.set(x, 3).delete_key("items");
    })
    .unwrap();
    a.delete(x).unwrap();
    assert_eq!(backend.queries.load(Ordering::SeqCst), 0);
}
