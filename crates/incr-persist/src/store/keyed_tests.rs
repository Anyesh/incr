use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use incr_core::PairOp;
use serde::{Deserialize, Serialize};

use super::collection::fp_of_elements;
use super::dep_index::{DepFp, DepIndex};
use super::{Options, PendingWrite, PersistedKeyed, Store};
use crate::{
    Backend, Cells, Error, Kind, Local, MemoryBackend, Position, ReadOutcome, Record, Result,
    Shared, SnapshotId,
};

type Kc<C> = PersistedKeyed<u32, String, C>;

macro_rules! local_and_shared {
    ($($name:ident),* $(,)?) => {
        $(
            mod $name {
                #[test]
                fn local() {
                    super::$name::<crate::Local>();
                }

                #[test]
                fn shared() {
                    super::$name::<crate::Shared>();
                }
            }
        )*
    };
}

local_and_shared!(
    records_match_the_reference_encoding,
    foreign_insert_remove_and_replace_reach_the_index_via_refresh,
    a_write_staged_before_a_caught_up_foreign_write_repairs_to_one_pair,
    a_write_staged_before_a_refreshed_foreign_write_repairs_to_one_pair,
    contention_beyond_three_rounds_is_reported,
    delete_and_forget_empty_the_key,
    a_reader_crossing_a_compaction_ends_with_the_snapshot_rows,
    a_scope_write_and_its_undo_leave_the_log_and_rows_unchanged,
    a_foreign_decode_failure_leaves_index_and_collection_equal,
    the_dep_index_fingerprint_is_the_fingerprint_of_the_pairs,
    a_plain_and_a_keyed_collection_on_one_key_mismatch,
    user_drops_run_after_the_dep_index_hold,
    undeclare_releases_the_key_and_a_redeclare_reads_the_logged_rows,
    undeclare_is_refused_while_a_query_reads_the_rows,
    undeclare_is_refused_while_a_query_reads_the_pairs,
    counts_name_the_collection_asked_for,
    an_upsert_back_to_the_held_row_still_wins_over_a_foreign_write,
    a_reconcile_overtaken_by_a_foreign_insert_removes_it,
    a_reconcile_keeps_keys_staged_earlier_in_the_write,
);

fn open<C: Cells, B: Backend>(backend: B) -> Store<C, B> {
    Store::open(backend, Options::default()).unwrap()
}

fn keyed<C: Cells, B: Backend>(store: &Store<C, B>) -> Kc<C> {
    store.keyed_collection("k").unwrap()
}

fn s(value: &str) -> String {
    value.to_owned()
}

fn enc(k: u32, v: &str) -> Vec<u8> {
    postcard::to_stdvec(&(k, s(v))).unwrap()
}

fn ins(k: u32, v: &str) -> Record {
    Record::Insert {
        key: s("k"),
        element: enc(k, v),
    }
}

fn rem(k: u32, v: &str) -> Record {
    Record::Remove {
        key: s("k"),
        element: enc(k, v),
    }
}

fn records(backend: &impl Backend) -> Vec<Record> {
    let mut out = Vec::new();
    assert_eq!(
        backend.read_from(Position::ZERO, &mut out).unwrap(),
        ReadOutcome::Records
    );
    out.into_iter().map(|(_, record)| record).collect()
}

fn sorted<T: Ord>(mut values: Vec<T>) -> Vec<T> {
    values.sort();
    values
}

fn items<C: Cells>(kc: &Kc<C>) -> Vec<(u32, String)> {
    sorted(kc.items())
}

fn pairs<C: Cells, B: Backend>(store: &Store<C, B>, kc: &Kc<C>) -> Vec<(u32, String)> {
    sorted(kc.pairs().values(store.runtime()))
}

fn pair(k: u32, v: &str) -> (u32, String) {
    (k, s(v))
}

fn records_match_the_reference_encoding<C: Cells>() {
    let log = MemoryBackend::new();
    let store = open::<C, _>(log.clone());
    let x = store.input("x", 0u32).unwrap();
    let kc = keyed(&store);
    let start = records(&log).len();
    assert_eq!(
        records(&log)[start - 1],
        Record::Declare {
            key: s("k"),
            kind: Kind::Collection,
            default: Vec::new(),
        }
    );

    store.upsert(&kc, 1, s("a")).unwrap();
    store.upsert(&kc, 1, s("a")).unwrap();
    store.upsert(&kc, 1, s("b")).unwrap();
    store.remove(&kc, &1).unwrap();
    let mut pending = PendingWrite::new();
    pending.set(x, 7);
    pending.upsert(&kc, 2, s("c"));
    pending.upsert(&kc, 3, s("d"));
    pending.upsert(&kc, 2, s("e"));
    store.write_pending(pending).unwrap();
    store.reconcile(&kc, [(3, s("d")), (4, s("f"))]).unwrap();

    assert_eq!(
        records(&log)[start..],
        [
            ins(1, "a"),
            rem(1, "a"),
            ins(1, "b"),
            rem(1, "b"),
            Record::Set {
                key: s("x"),
                value: postcard::to_stdvec(&7u32).unwrap(),
            },
            ins(2, "e"),
            ins(3, "d"),
            ins(4, "f"),
            rem(2, "e"),
        ]
    );
    assert_eq!(items(&kc), [pair(3, "d"), pair(4, "f")]);
}

fn foreign_insert_remove_and_replace_reach_the_index_via_refresh<C: Cells>() {
    let log = MemoryBackend::new();
    let a = open::<C, _>(log.clone());
    let ka = keyed(&a);
    let b = open::<C, _>(log.clone());
    let kb = keyed(&b);
    a.upsert(&ka, 1, s("a")).unwrap();
    a.upsert(&ka, 2, s("b")).unwrap();
    b.refresh().unwrap();
    assert_eq!(kb.row(&1), Some(s("a")));
    assert_eq!(items(&kb), [pair(1, "a"), pair(2, "b")]);

    a.upsert(&ka, 1, s("c")).unwrap();
    a.remove(&ka, &2).unwrap();
    log.append(&[Record::Replace {
        key: s("k"),
        old: Some(enc(1, "c")),
        new: enc(3, "d"),
    }])
    .unwrap();
    b.refresh().unwrap();
    assert_eq!(items(&kb), [pair(3, "d")]);
    assert_eq!(pairs(&b, &kb), [pair(3, "d")]);
    a.refresh().unwrap();
    assert_eq!(items(&ka), [pair(3, "d")]);
}

fn staged_then_overtaken<C: Cells>(refresh_first: bool) {
    let log = MemoryBackend::new();
    let a = open::<C, _>(log.clone());
    let ka = keyed(&a);
    let b = open::<C, _>(log.clone());
    let kb = keyed(&b);
    a.upsert(&ka, 1, s("x")).unwrap();
    b.refresh().unwrap();

    let mut pending = PendingWrite::new();
    pending.upsert(&kb, 1, s("b"));
    a.upsert(&ka, 1, s("a")).unwrap();
    if refresh_first {
        b.refresh().unwrap();
    }
    b.write_pending(pending).unwrap();

    assert_eq!(pairs(&b, &kb), [pair(1, "b")]);
    assert_eq!(kb.row(&1), Some(s("b")));
    a.refresh().unwrap();
    assert_eq!(pairs(&a, &ka), [pair(1, "b")]);
    drop((a, b));
    let c = open::<C, _>(log.clone());
    let kc = keyed(&c);
    assert_eq!(pairs(&c, &kc), [pair(1, "b")]);
}

fn a_write_staged_before_a_caught_up_foreign_write_repairs_to_one_pair<C: Cells>() {
    staged_then_overtaken::<C>(false);
}

fn a_write_staged_before_a_refreshed_foreign_write_repairs_to_one_pair<C: Cells>() {
    staged_then_overtaken::<C>(true);
}

/// Appends an intruding pair for key 1 before each of the next `armed`
/// appends, as another writer racing every write would.
struct Intruding {
    inner: MemoryBackend,
    armed: Arc<AtomicUsize>,
    count: AtomicUsize,
}

impl Backend for Intruding {
    fn append(&self, records: &[Record]) -> Result<Position> {
        let armed = self.armed.load(Ordering::SeqCst);
        if armed > 0 {
            self.armed.store(armed - 1, Ordering::SeqCst);
            let n = self.count.fetch_add(1, Ordering::SeqCst);
            self.inner.append(&[ins(1, &format!("intruder {n}"))])?;
        }
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
        self.inner.compact(upto, folded)
    }

    fn writable(&self) -> bool {
        self.inner.writable()
    }
}

fn contention_beyond_three_rounds_is_reported<C: Cells>() {
    let log = MemoryBackend::new();
    let armed = Arc::new(AtomicUsize::new(0));
    let store = open::<C, _>(Intruding {
        inner: log.clone(),
        armed: Arc::clone(&armed),
        count: AtomicUsize::new(0),
    });
    let kc = keyed(&store);
    store.upsert(&kc, 1, s("x")).unwrap();

    armed.store(4, Ordering::SeqCst);
    match store.upsert(&kc, 1, s("b")) {
        Err(Error::KeyedContention { key }) => assert_eq!(key, "k"),
        other => panic!("expected KeyedContention, got {other:?}"),
    }
    assert_eq!(armed.load(Ordering::SeqCst), 0);

    armed.store(3, Ordering::SeqCst);
    store.upsert(&kc, 1, s("c")).unwrap();
    assert_eq!(pairs(&store, &kc), [pair(1, "c")]);
}

fn delete_and_forget_empty_the_key<C: Cells>() {
    let log = MemoryBackend::new();
    let store = open::<C, _>(log.clone());
    let kc = keyed(&store);
    store.upsert(&kc, 1, s("a")).unwrap();
    store.upsert(&kc, 2, s("b")).unwrap();
    store.delete(&kc).unwrap();
    assert!(kc.is_empty());
    assert_eq!(pairs(&store, &kc), []);

    store.upsert(&kc, 1, s("c")).unwrap();
    assert_eq!(kc.row(&1), Some(s("c")));
    log.append(&[Record::Forget { key: s("k") }]).unwrap();
    store.refresh().unwrap();
    assert!(kc.is_empty());
    assert_eq!(pairs(&store, &kc), []);

    store.upsert(&kc, 2, s("d")).unwrap();
    assert_eq!(items(&kc), [pair(2, "d")]);
}

fn a_reader_crossing_a_compaction_ends_with_the_snapshot_rows<C: Cells>() {
    let log = MemoryBackend::new();
    let a = open::<C, _>(log.clone());
    let ka = keyed(&a);
    let b = open::<C, _>(log.clone());
    let kb = keyed(&b);
    a.upsert(&ka, 1, s("a")).unwrap();
    a.upsert(&ka, 2, s("b")).unwrap();
    b.refresh().unwrap();

    a.upsert(&ka, 1, s("c")).unwrap();
    a.remove(&ka, &2).unwrap();
    a.upsert(&ka, 3, s("d")).unwrap();
    a.checkpoint().unwrap();
    b.refresh().unwrap();

    assert_eq!(items(&kb), [pair(1, "c"), pair(3, "d")]);
    assert_eq!(pairs(&b, &kb), pairs(&a, &ka));
}

fn a_scope_write_and_its_undo_leave_the_log_and_rows_unchanged<C: Cells>() {
    let log = MemoryBackend::new();
    let store = open::<C, _>(log.clone());
    let kc = keyed(&store);
    store.upsert(&kc, 1, s("a")).unwrap();
    store.upsert(&kc, 2, s("b")).unwrap();
    let logged = records(&log);
    let rt = store.runtime();

    let scope = store.unlogged_scope().unwrap();
    let overlay = scope.keyed(&kc).unwrap();
    let mut journal = Vec::new();
    overlay.upsert_journaled(rt, 1, s("z"), Some(&mut journal));
    overlay.remove_journaled(rt, &2, Some(&mut journal));
    overlay.upsert_journaled(rt, 3, s("w"), Some(&mut journal));
    assert_eq!(items(&kc), [pair(1, "z"), pair(3, "w")]);
    overlay.apply_pairs_with(
        rt,
        journal.into_iter().rev().map(PairOp::inverse),
        &mut (),
        None,
    );
    scope.close().unwrap();

    assert_eq!(records(&log), logged);
    assert_eq!(items(&kc), [pair(1, "a"), pair(2, "b")]);
    assert_eq!(pairs(&store, &kc), [pair(1, "a"), pair(2, "b")]);
    store.upsert(&kc, 1, s("q")).unwrap();
    assert_eq!(items(&kc), [pair(1, "q"), pair(2, "b")]);
}

fn a_foreign_decode_failure_leaves_index_and_collection_equal<C: Cells>() {
    let log = MemoryBackend::new();
    let a = open::<C, _>(log.clone());
    let ka = keyed(&a);
    let b = open::<C, _>(log.clone());
    let kb = keyed(&b);
    a.upsert(&ka, 1, s("a")).unwrap();
    b.refresh().unwrap();
    log.append(&[
        ins(2, "b"),
        Record::Insert {
            key: s("k"),
            element: vec![0xff],
        },
        ins(3, "c"),
    ])
    .unwrap();

    assert!(matches!(b.refresh(), Err(Error::TypeMismatch { .. })));
    assert_eq!(items(&kb), [pair(1, "a"), pair(2, "b")]);
    assert_eq!(items(&kb), pairs(&b, &kb));
}

fn known(deps: &DepIndex, key: &str) -> u128 {
    match deps.fp(key) {
        Some(DepFp::Known(fp)) => fp,
        other => panic!("{key}: expected a known fingerprint, found {other:?}"),
    }
}

fn assert_fp<C: Cells, B: Backend>(store: &Store<C, B>, kc: &Kc<C>, deps: &DepIndex) {
    let held = kc.pairs().values(store.runtime());
    assert_eq!(known(deps, "k"), fp_of_elements("k", &held).unwrap());
}

fn the_dep_index_fingerprint_is_the_fingerprint_of_the_pairs<C: Cells>() {
    let log = MemoryBackend::new();
    let store = open::<C, _>(log.clone());
    let kc = keyed(&store);
    store.upsert(&kc, 1, s("a")).unwrap();
    let deps = store.open_dep_index().unwrap();
    assert_fp(&store, &kc, &deps);
    store.upsert(&kc, 2, s("b")).unwrap();
    store.upsert(&kc, 1, s("c")).unwrap();
    store.remove(&kc, &2).unwrap();
    assert_fp(&store, &kc, &deps);
    log.append(&[ins(1, "two"), ins(4, "d")]).unwrap();
    store.refresh().unwrap();
    assert_eq!(pairs(&store, &kc).len(), 3);
    assert_fp(&store, &kc, &deps);
    log.append(&[Record::Replace {
        key: s("k"),
        old: Some(enc(4, "d")),
        new: enc(5, "e"),
    }])
    .unwrap();
    store.refresh().unwrap();
    assert_fp(&store, &kc, &deps);
    store.delete(&kc).unwrap();
    assert_eq!(known(&deps, "k"), 0);

    store.upsert(&kc, 6, s("f")).unwrap();
    let reopened = open::<C, _>(log.clone());
    let deps = reopened.open_dep_index().unwrap();
    let kc = keyed(&reopened);
    assert_fp(&reopened, &kc, &deps);
    assert_eq!(items(&kc), [pair(6, "f")]);
}

fn a_plain_and_a_keyed_collection_on_one_key_mismatch<C: Cells>() {
    let store = open::<C, _>(MemoryBackend::new());
    store.collection::<(u32, String)>("k").unwrap();
    assert!(matches!(
        store.keyed_collection::<u32, String, (u32, String)>("k"),
        Err(Error::TypeMismatch { .. })
    ));
}

#[test]
fn a_keyed_collection_then_a_plain_one_on_its_key_mismatch() {
    let store = open::<Local, _>(MemoryBackend::new());
    keyed(&store);
    assert!(matches!(
        store.collection::<(u32, String)>("k"),
        Err(Error::TypeMismatch { .. })
    ));
    store.input("i", 0u32).unwrap();
    assert!(matches!(
        store.keyed_collection::<u32, String, (u32, String)>("i"),
        Err(Error::KindMismatch { .. })
    ));
    let shared = open::<Shared, _>(MemoryBackend::new());
    keyed(&shared);
    assert!(shared.collection::<(u32, String)>("k").is_err());
}

thread_local! {
    static WATCHED: RefCell<Option<Arc<DepIndex>>> = const { RefCell::new(None) };
    static DROPS: Cell<(u32, u32)> = const { Cell::new((0, 0)) };
}

/// Counts its drops while an index is watched, and those that run while
/// the watched index is held for writing.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Watched(String);

impl Drop for Watched {
    fn drop(&mut self) {
        WATCHED.with(|watched| {
            if let Some(deps) = watched.borrow().as_ref() {
                let (all, held) = DROPS.get();
                DROPS.set((all + 1, held + u32::from(deps.is_held())));
            }
        });
    }
}

fn w(value: &str) -> Watched {
    Watched(s(value))
}

fn user_drops_run_after_the_dep_index_hold<C: Cells>() {
    let log = MemoryBackend::new();
    let a = open::<C, _>(log.clone());
    let ka = a
        .keyed_collection::<u32, Watched, (u32, Watched)>("k")
        .unwrap();
    a.upsert(&ka, 1, w("a")).unwrap();
    a.upsert(&ka, 2, w("b")).unwrap();
    let b = open::<C, _>(log.clone());
    let deps = b.open_dep_index().unwrap();
    WATCHED.with(|watched| *watched.borrow_mut() = Some(Arc::clone(&deps)));
    DROPS.set((0, 0));

    let kb = b
        .keyed_collection::<u32, Watched, (u32, Watched)>("k")
        .unwrap();
    a.upsert(&ka, 1, w("c")).unwrap();
    a.remove(&ka, &2).unwrap();
    b.refresh().unwrap();
    b.upsert(&kb, 1, w("d")).unwrap();
    b.upsert(&kb, 3, w("e")).unwrap();
    b.remove(&kb, &3).unwrap();
    a.refresh().unwrap();
    a.upsert(&ka, 1, w("f")).unwrap();
    a.upsert(&ka, 4, w("g")).unwrap();
    a.checkpoint().unwrap();
    b.refresh().unwrap();
    assert_eq!(kb.row(&1), Some(w("f")));
    b.delete(&kb).unwrap();

    WATCHED.with(|watched| *watched.borrow_mut() = None);
    let (all, held) = DROPS.get();
    assert!(all > 0, "the writes dropped no Watched value");
    assert_eq!(held, 0, "{held} of {all} drops ran under the hold");
}

fn undeclare_releases_the_key_and_a_redeclare_reads_the_logged_rows<C: Cells>() {
    let log = MemoryBackend::new();
    let store = open::<C, _>(log.clone());
    let nodes = store.runtime().node_count();
    let kc = keyed(&store);
    store.upsert(&kc, 1, s("a")).unwrap();
    store.upsert(&kc, 2, s("b")).unwrap();
    let logged = records(&log);
    assert_eq!(sorted(kc.rows().values(store.runtime())), [s("a"), s("b")]);

    store.undeclare(&kc).unwrap();
    assert_eq!(store.runtime().node_count(), nodes);
    assert_eq!(records(&log), logged, "undeclare appends nothing");
    assert!(matches!(
        store.upsert(&kc, 3, s("c")),
        Err(Error::UnknownHandle)
    ));

    let kc = keyed(&store);
    assert_eq!(items(&kc), [pair(1, "a"), pair(2, "b")]);
    assert_eq!(kc.rows().values(store.runtime()).len(), 2);
    store.upsert(&kc, 3, s("c")).unwrap();
    assert_eq!(kc.row(&3), Some(s("c")));
}

fn refused_while_read<C: Cells>(read_pairs: bool) {
    let log = MemoryBackend::new();
    let store = open::<C, _>(log.clone());
    let kc = keyed(&store);
    store.upsert(&kc, 1, s("a")).unwrap();
    let rt = store.runtime();
    let read = if read_pairs {
        kc.pairs().version_node()
    } else {
        kc.rows().version_node()
    };
    let query = rt.create_query(move |rt| rt.get(read));
    let before = rt.get(query);
    let nodes = rt.node_count();

    assert!(matches!(
        store.undeclare(&kc),
        Err(Error::HasDependents { key }) if key == "k"
    ));
    assert_eq!(
        rt.node_count(),
        nodes,
        "a refused undeclare deletes nothing"
    );
    assert_eq!(kc.row(&1), Some(s("a")));
    store.upsert(&kc, 2, s("b")).unwrap();
    assert_eq!(items(&kc), [pair(1, "a"), pair(2, "b")]);
    assert!(rt.get(query) > before);
    assert_eq!(
        keyed(&store).row(&2),
        Some(s("b")),
        "the key stays declared"
    );

    rt.delete_node(query);
    store.undeclare(&kc).unwrap();
    assert_eq!(items(&keyed(&store)), [pair(1, "a"), pair(2, "b")]);
}

fn undeclare_is_refused_while_a_query_reads_the_rows<C: Cells>() {
    refused_while_read::<C>(false);
}

fn undeclare_is_refused_while_a_query_reads_the_pairs<C: Cells>() {
    refused_while_read::<C>(true);
}

fn counts_name_the_collection_asked_for<C: Cells>() {
    let store = open::<C, _>(MemoryBackend::new());
    let first = keyed(&store);
    let second: Kc<C> = store.keyed_collection("k2").unwrap();
    let mut pending = PendingWrite::new();
    pending.upsert(&first, 1, s("a"));
    pending.upsert(&second, 2, s("b"));
    pending.upsert(&second, 3, s("c"));
    assert_eq!(pending.counts(&first).inserted, 1);
    assert_eq!(pending.counts(&second).inserted, 2);
}

fn an_upsert_back_to_the_held_row_still_wins_over_a_foreign_write<C: Cells>() {
    let log = MemoryBackend::new();
    let a = open::<C, _>(log.clone());
    let ka = keyed(&a);
    let b = open::<C, _>(log.clone());
    let kb = keyed(&b);
    a.upsert(&ka, 1, s("x")).unwrap();
    b.refresh().unwrap();

    let mut pending = PendingWrite::new();
    pending.upsert(&kb, 1, s("y"));
    pending.upsert(&kb, 1, s("x"));
    a.upsert(&ka, 1, s("a")).unwrap();
    b.write_pending(pending).unwrap();

    assert_eq!(pairs(&b, &kb), [pair(1, "x")]);
    a.refresh().unwrap();
    assert_eq!(pairs(&a, &ka), [pair(1, "x")]);
}

fn a_reconcile_overtaken_by_a_foreign_insert_removes_it<C: Cells>() {
    let log = MemoryBackend::new();
    let a = open::<C, _>(log.clone());
    let ka = keyed(&a);
    let b = open::<C, _>(log.clone());
    let kb = keyed(&b);
    a.upsert(&ka, 1, s("x")).unwrap();
    b.refresh().unwrap();

    let mut pending = PendingWrite::new();
    pending.reconcile(&kb, [(1, s("b"))]);
    a.upsert(&ka, 2, s("z")).unwrap();
    b.write_pending(pending).unwrap();

    assert_eq!(pairs(&b, &kb), [pair(1, "b")]);
    a.refresh().unwrap();
    assert_eq!(pairs(&a, &ka), [pair(1, "b")]);
}

fn a_reconcile_keeps_keys_staged_earlier_in_the_write<C: Cells>() {
    let store = open::<C, _>(MemoryBackend::new());
    let kc = keyed(&store);
    store.upsert(&kc, 1, s("x")).unwrap();
    let mut pending = PendingWrite::new();
    pending.upsert(&kc, 2, s("y"));
    pending.reconcile(&kc, [(3, s("z"))]);
    store.write_pending(pending).unwrap();
    assert_eq!(pairs(&store, &kc), [pair(2, "y"), pair(3, "z")]);
}
