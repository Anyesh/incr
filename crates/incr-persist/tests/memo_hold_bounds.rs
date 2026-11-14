//! A checkpoint's holds of the store's value scope stay bounded and are
//! never empty: no hold covers more than a batch of value calls, and a
//! hold that would run none is not taken.

use std::cell::Cell;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use incr_persist::memo::{BuildId, Memo, MemoBuilder, MemoOptions, MemoryMemo};
use incr_persist::{MemoryBackend, Options, PersistedKeyed, Shared, Store, ValueScope};

struct ScopeMarker;

static SERIAL: Mutex<()> = Mutex::new(());
static MOST_IN_A_HOLD: AtomicUsize = AtomicUsize::new(0);
static EMPTY_HOLDS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

fn hold(f: &mut dyn FnMut()) {
    DEPTH.with(|d| d.set(d.get() + 1));
    f();
    DEPTH.with(|d| d.set(d.get() - 1));
    let calls = CALLS.with(|c| c.replace(0));
    MOST_IN_A_HOLD.fetch_max(calls, Ordering::SeqCst);
    if calls == 0 {
        EMPTY_HOLDS.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Debug, Eq)]
struct Probe(u32);

fn note() {
    if DEPTH.with(Cell::get) > 0 {
        CALLS.with(|c| c.set(c.get() + 1));
    }
}

impl PartialEq for Probe {
    fn eq(&self, other: &Self) -> bool {
        note();
        self.0 == other.0
    }
}

impl Hash for Probe {
    fn hash<H: Hasher>(&self, state: &mut H) {
        note();
        self.0.hash(state);
    }
}

impl Clone for Probe {
    fn clone(&self) -> Self {
        note();
        Probe(self.0)
    }
}

impl serde::Serialize for Probe {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        note();
        s.serialize_u32(self.0)
    }
}

impl<'de> serde::Deserialize<'de> for Probe {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        note();
        u32::deserialize(d).map(Probe)
    }
}

struct Fixture {
    store: Arc<Store<Shared, MemoryBackend>>,
    rows: PersistedKeyed<Probe, Probe, Shared>,
    _serial: MutexGuard<'static, ()>,
}

fn fixture() -> Fixture {
    let serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    MOST_IN_A_HOLD.store(0, Ordering::SeqCst);
    EMPTY_HOLDS.store(0, Ordering::SeqCst);
    let options = Options {
        value_scope: Some(ValueScope::new(Arc::new(ScopeMarker)).holding(hold)),
        ..Options::default()
    };
    let store = Arc::new(Store::<Shared, _>::open(MemoryBackend::new(), options).unwrap());
    let rows = store.keyed_collection("rows").unwrap();
    Fixture {
        store,
        rows,
        _serial: serial,
    }
}

fn open(fx: &Fixture, build: impl FnOnce(&mut MemoBuilder<Shared>)) -> Memo<Shared, MemoryBackend> {
    let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(0x401d)));
    build(&mut builder);
    builder.open(&fx.store, MemoryMemo::new()).unwrap()
}

/// One entry reads 1,000 keys of a persisted collection, so the
/// checkpoint maps 1,000 row deps. The bound is a batch of 256 keys at
/// two calls each (hash and compare while ranking); unchunked, the red
/// run covered all 1,000 keys in one hold.
#[test]
fn a_thousand_row_deps_map_in_bounded_holds() {
    let fx = fixture();
    let keys = 1000_u32;
    fx.store
        .reconcile(&fx.rows, (0..keys).map(|k| (Probe(k), Probe(k * 2))))
        .unwrap();
    let mut table = None;
    let read = fx.rows.clone();
    let memo = open(&fx, |builder| {
        let sum = builder.table::<Probe, Probe>("sum", 1);
        builder.define(sum, move |ctx, _: &Probe| {
            Probe(
                (0..keys)
                    .map(|k| read.get_key(ctx.rt(), &Probe(k)).unwrap().0)
                    .fold(0_u32, u32::wrapping_add),
            )
        });
        table = Some(sum);
    });
    let node = memo.get_or_insert(table.unwrap(), Probe(0));
    memo.runtime().get(node);
    memo.checkpoint().unwrap();
    let most = MOST_IN_A_HOLD.load(Ordering::SeqCst);
    assert!(most <= 512, "{most} value calls in one hold");
}

/// A hold that runs no value call was taken for nothing. Every entry in
/// the tests below has a key and a value that some hold encodes, so an
/// empty hold is a mapping or inspect hold that had no value to cover.
fn assert_no_hold_was_empty() {
    let empty = EMPTY_HOLDS.load(Ordering::SeqCst);
    assert_eq!(empty, 0, "{empty} holds ran no value call");
}

/// Entries read only queries, so none maps a row dep, while a collection
/// is declared and one of its keys has a version input: the row source
/// finds no row for any dep and takes no hold.
#[test]
fn query_deps_next_to_a_keyed_collection_take_no_mapping_hold() {
    let fx = fixture();
    fx.store
        .reconcile(&fx.rows, (0..4).map(|k| (Probe(k), Probe(k))))
        .unwrap();
    let read = fx.rows.clone();
    let reader = fx
        .store
        .runtime()
        .create_query(move |rt| read.get_key(rt, &Probe(0)));
    fx.store.runtime().get(reader);
    assert!(fx.rows.version_input(&Probe(0)).is_some());
    let mut tables = None;
    let memo = open(&fx, |builder| {
        let leaf = builder.table::<Probe, Probe>("leaf", 1);
        let top = builder.table::<Probe, Probe>("top", 1);
        builder.define(leaf, |_, k: &Probe| Probe(k.0 + 1));
        builder.define(top, move |ctx, k: &Probe| ctx.get(leaf, Probe(k.0)));
        tables = Some(top);
    });
    for k in 0..4 {
        let node = memo.get_or_insert(tables.unwrap(), Probe(k));
        memo.runtime().get(node);
    }
    memo.checkpoint().unwrap();
    assert_no_hold_was_empty();
}

/// Every entry of the batch was invalidated after it ran, so the
/// checkpoint's inspect finds none Clean and takes no hold.
#[test]
fn a_batch_of_dirty_entries_takes_no_inspect_hold() {
    let fx = fixture();
    let mut table = None;
    let memo = open(&fx, |builder| {
        let entry = builder.table::<Probe, Probe>("entry", 1);
        builder.define(entry, |ctx, k: &Probe| {
            Probe(k.0.wrapping_add(ctx.external("n") as u32))
        });
        table = Some(entry);
    });
    memo.set_external("n", 1);
    for k in 0..4 {
        let node = memo.get_or_insert(table.unwrap(), Probe(k));
        memo.runtime().get(node);
    }
    memo.set_external("n", 2);
    memo.checkpoint().unwrap();
    assert_no_hold_was_empty();
}
