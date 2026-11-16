//! A memo fingerprints a store collection only when it first checks an
//! entry that depends on the whole collection: opening encodes no element,
//! a `get_key` row check encodes none, and a collection is encoded once.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

use incr_persist::memo::{
    BuildId, Memo, MemoBuilder, MemoCells, MemoOptions, MemoTable, MemoryMemo,
};
use incr_persist::{
    current_value_context, Local, MemoryBackend, Options, PersistedCollection, PersistedKeyed,
    Shared, Store, ValueContext,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const BUILD: u128 = 0x1a2f;
const LEN: u32 = 40;

thread_local! {
    static ENCODES: RefCell<HashMap<String, usize>> = RefCell::new(HashMap::new());
    static FAILING: Cell<Option<u32>> = const { Cell::new(None) };
}

/// Counts its encodes by the store key they run for, and fails to encode
/// while it is the value [`FAILING`] names.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Counted(u32);

impl Serialize for Counted {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if let Some(ValueContext::Key(key)) = current_value_context() {
            ENCODES.with(|e| *e.borrow_mut().entry(key).or_default() += 1);
        }
        if FAILING.with(Cell::get) == Some(self.0) {
            return Err(serde::ser::Error::custom("failing on purpose"));
        }
        s.serialize_u32(self.0)
    }
}

impl<'de> Deserialize<'de> for Counted {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        u32::deserialize(d).map(Counted)
    }
}

fn reset() {
    ENCODES.with(|e| e.borrow_mut().clear());
}

fn encodes(key: &str) -> usize {
    ENCODES.with(|e| e.borrow().get(key).copied().unwrap_or(0))
}

fn encoded_nothing() -> bool {
    ENCODES.with(|e| e.borrow().values().all(|&n| n == 0))
}

struct Backends {
    log: MemoryBackend,
    memo: MemoryMemo,
}

struct Keys<C: MemoCells> {
    store: Arc<Store<C, MemoryBackend>>,
    xs: PersistedCollection<Counted, C>,
    ys: PersistedCollection<Counted, C>,
    rows: PersistedKeyed<u32, Counted, C>,
}

fn keys<C: MemoCells>(backends: &Backends) -> Keys<C> {
    let store = Arc::new(Store::open(backends.log.clone(), Options::default()).unwrap());
    Keys {
        xs: store.collection("xs").unwrap(),
        ys: store.collection("ys").unwrap(),
        rows: store.keyed_collection("rows").unwrap(),
        store,
    }
}

/// A collection handle captured by a compute.
struct OneThread<T>(T);

// SAFETY: under `Local` the memo runs every compute on the thread that
// opened it, which owns the handle; under `Shared` the handle is already
// Send + Sync.
unsafe impl<T> Send for OneThread<T> {}
// SAFETY: see `Send`.
unsafe impl<T> Sync for OneThread<T> {}

impl<T> OneThread<T> {
    fn get(&self) -> &T {
        &self.0
    }
}

#[derive(Default)]
struct Runs {
    sum_x: AtomicUsize,
    len_x: AtomicUsize,
    sum_y: AtomicUsize,
    row: AtomicUsize,
}

impl Runs {
    fn take(&self) -> [usize; 4] {
        [&self.sum_x, &self.len_x, &self.sum_y, &self.row].map(|n| n.swap(0, SeqCst))
    }
}

struct Opened<C: MemoCells> {
    memo: Memo<C, MemoryBackend>,
    sum_x: MemoTable<(), u64>,
    len_x: MemoTable<(), u64>,
    sum_y: MemoTable<(), u64>,
    row: MemoTable<u32, Option<u32>>,
}

impl<C: MemoCells> Opened<C> {
    fn get<K: incr_persist::memo::MemoKey, T: incr_persist::memo::MemoValue>(
        &self,
        table: MemoTable<K, T>,
        key: K,
    ) -> T {
        let node = self.memo.get_or_insert(table, key);
        self.memo.runtime().get(node)
    }
}

fn open_memo<C: MemoCells>(
    keys: &Keys<C>,
    backends: &Backends,
    runs: &Arc<Runs>,
) -> incr_persist::Result<Opened<C>> {
    let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(BUILD)));
    let sum_x = builder.table::<(), u64>("sum_x", 1);
    let len_x = builder.table::<(), u64>("len_x", 1);
    let sum_y = builder.table::<(), u64>("sum_y", 1);
    let row = builder.table::<u32, Option<u32>>("row", 1);
    let sum = |values: Vec<Counted>| values.iter().map(|c| u64::from(c.0)).sum::<u64>();
    let (view, r) = (OneThread(keys.xs.handle().clone()), Arc::clone(runs));
    builder.define(sum_x, move |ctx, _| {
        r.sum_x.fetch_add(1, SeqCst);
        sum(view.get().values(ctx.rt()))
    });
    let (view, r) = (OneThread(keys.xs.handle().clone()), Arc::clone(runs));
    builder.define(len_x, move |ctx, _| {
        r.len_x.fetch_add(1, SeqCst);
        view.get().values(ctx.rt()).len() as u64
    });
    let (view, r) = (OneThread(keys.ys.handle().clone()), Arc::clone(runs));
    builder.define(sum_y, move |ctx, _| {
        r.sum_y.fetch_add(1, SeqCst);
        sum(view.get().values(ctx.rt()))
    });
    let (rows, r) = (OneThread(keys.rows.clone()), Arc::clone(runs));
    builder.define(row, move |ctx, k: &u32| {
        r.row.fetch_add(1, SeqCst);
        rows.get().get_key(ctx.rt(), k).map(|c| c.0)
    });
    let memo = builder.open(&keys.store, backends.memo.clone())?;
    Ok(Opened {
        memo,
        sum_x,
        len_x,
        sum_y,
        row,
    })
}

fn populate<C: MemoCells>(backends: &Backends, runs: &Arc<Runs>) {
    let keys = keys::<C>(backends);
    for i in 0..LEN {
        keys.xs.insert(&keys.store, Counted(i)).unwrap();
        keys.ys.insert(&keys.store, Counted(2 * i)).unwrap();
    }
    keys.store
        .reconcile(&keys.rows, (0..LEN).map(|k| (k, Counted(10 * k))))
        .unwrap();
    let memo = open_memo(&keys, backends, runs).unwrap();
    memo.get(memo.sum_x, ());
    memo.get(memo.len_x, ());
    memo.get(memo.sum_y, ());
    memo.get(memo.row, 3);
    memo.get(memo.row, 4);
    memo.memo.checkpoint().unwrap();
    runs.take();
}

fn collection_fingerprints_are_computed_on_first_check<C: MemoCells>() {
    let backends = Backends {
        log: MemoryBackend::new(),
        memo: MemoryMemo::new(),
    };
    let runs = Arc::new(Runs::default());
    populate::<C>(&backends, &runs);
    let sum_x: u64 = (0..LEN).map(u64::from).sum();

    {
        let keys = keys::<C>(&backends);
        reset();
        let memo = open_memo(&keys, &backends, &runs).unwrap();
        assert!(
            encoded_nothing(),
            "opening the memo encoded collection rows"
        );

        assert_eq!(memo.get(memo.row, 3), Some(30));
        assert!(
            encoded_nothing(),
            "a get_key row check encoded a collection"
        );

        assert_eq!(memo.get(memo.sum_x, ()), sum_x);
        assert_eq!(memo.get(memo.len_x, ()), u64::from(LEN));
        assert_eq!(encodes("xs"), LEN as usize, "xs is encoded exactly once");
        assert_eq!((encodes("ys"), encodes("rows")), (0, 0));
        assert_eq!(runs.take(), [0, 0, 0, 0], "every entry is reused");

        keys.xs.insert(&keys.store, Counted(100)).unwrap();
        assert_eq!(memo.get(memo.sum_x, ()), sum_x + 100);
        memo.memo.checkpoint().unwrap();
        runs.take();
    }

    let keys = keys::<C>(&backends);
    keys.store.upsert(&keys.rows, 4, Counted(999)).unwrap();
    reset();
    let memo = open_memo(&keys, &backends, &runs).unwrap();
    assert!(
        encoded_nothing(),
        "opening the memo encoded collection rows"
    );
    keys.ys.insert(&keys.store, Counted(1)).unwrap();
    reset();

    assert_eq!(memo.get(memo.row, 3), Some(30));
    assert_eq!(memo.get(memo.row, 4), Some(999));
    assert!(
        encoded_nothing(),
        "a get_key row check encoded a collection"
    );
    assert_eq!(memo.get(memo.sum_x, ()), sum_x + 100);
    assert_eq!(runs.take(), [0, 0, 0, 1], "only the changed row reruns");
    assert_eq!(encodes("xs"), LEN as usize + 1);

    let sum_y: u64 = (0..LEN).map(|i| u64::from(2 * i)).sum();
    assert_eq!(memo.get(memo.sum_y, ()), sum_y + 1);
    assert_eq!(
        runs.take(),
        [0, 0, 1, 0],
        "a collection written while its fingerprint was pending misses"
    );
    assert_eq!(encodes("ys"), LEN as usize + 1);
}

#[test]
fn collection_fingerprints_are_computed_on_first_check_local() {
    collection_fingerprints_are_computed_on_first_check::<Local>();
}

#[test]
fn collection_fingerprints_are_computed_on_first_check_shared() {
    collection_fingerprints_are_computed_on_first_check::<Shared>();
}

/// A stored element that no longer encodes makes the collection's
/// dependents miss instead of failing the memo's open.
fn a_collection_that_fails_to_encode_misses<C: MemoCells>() {
    let backends = Backends {
        log: MemoryBackend::new(),
        memo: MemoryMemo::new(),
    };
    let runs = Arc::new(Runs::default());
    populate::<C>(&backends, &runs);
    let keys = keys::<C>(&backends);
    FAILING.with(|f| f.set(Some(5)));
    let memo = open_memo(&keys, &backends, &runs);
    let memo = match memo {
        Ok(memo) => memo,
        Err(e) => {
            FAILING.with(|f| f.set(None));
            panic!("opening the memo failed: {e}");
        }
    };
    assert_eq!(memo.get(memo.row, 3), Some(30));
    assert_eq!(
        memo.get(memo.sum_y, ()),
        (0..LEN).map(|i| u64::from(2 * i)).sum()
    );
    assert_eq!(runs.take(), [0, 0, 0, 0]);
    assert_eq!(memo.get(memo.sum_x, ()), (0..LEN).map(u64::from).sum());
    assert_eq!(memo.get(memo.len_x, ()), u64::from(LEN));
    FAILING.with(|f| f.set(None));
    assert_eq!(
        runs.take(),
        [1, 1, 0, 0],
        "an untrusted collection's dependents miss"
    );
}

#[test]
fn a_collection_that_fails_to_encode_misses_local() {
    a_collection_that_fails_to_encode_misses::<Local>();
}

#[test]
fn a_collection_that_fails_to_encode_misses_shared() {
    a_collection_that_fails_to_encode_misses::<Shared>();
}
