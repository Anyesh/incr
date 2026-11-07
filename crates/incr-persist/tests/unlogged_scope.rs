//! `Store::unlogged_scope`: unlogged runtime changes that the caller
//! restores exactly, with the store and the memo inert meanwhile.

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

use incr_persist::memo::{
    BuildId, Memo, MemoBuilder, MemoCells, MemoOptions, MemoStats, MemoTable, MemoryMemo,
};
use incr_persist::{Backend, Error, Input, Local, MemoryBackend, Options, Record, Shared, Store};

const BUILD: u128 = 0xb1d;

fn open<C: incr_persist::Cells>(log: &MemoryBackend) -> Store<C, MemoryBackend> {
    Store::open(log.clone(), Options::default()).unwrap()
}

fn snapshot(log: &MemoryBackend) -> Vec<Record> {
    log.load_snapshot().unwrap().expect("a snapshot").1
}

fn assert_refused<T: std::fmt::Debug>(what: &str, result: incr_persist::Result<T>) {
    match result {
        Err(e @ Error::UnloggedScopeOpen) => {
            assert!(e.to_string().contains("unlogged scope"), "{what}: {e}");
        }
        other => panic!("{what}: expected UnloggedScopeOpen, got {other:?}"),
    }
}

#[test]
fn an_unlogged_insert_then_delete_leaves_the_next_checkpoint_equal_to_the_logged_state() {
    let log = MemoryBackend::new();
    let store = open::<Local>(&log);
    let c = store.collection::<u32>("c").unwrap();
    let x = store.input("x", 1_u32).unwrap();
    c.insert(&store, 1).unwrap();
    c.insert(&store, 2).unwrap();
    store.set(x, 5).unwrap();
    store.checkpoint().unwrap();
    let base = snapshot(&log);

    let mut scope = store.unlogged_scope().unwrap();
    let handle = scope.collection(&c).unwrap();
    let rt = store.runtime();
    let count = handle.count(rt);
    handle.insert(rt, 9);
    handle.delete(rt, &1);
    assert_eq!(rt.get(count), 2);
    assert_eq!(sorted(handle.values(rt)), vec![2, 9]);
    scope.touched(x).unwrap();
    rt.set(scope.input(x).unwrap(), 77);
    handle.delete(rt, &9);
    handle.insert(rt, 1);
    rt.set(x.handle(), 5);
    scope.close().unwrap();

    assert_eq!(sorted(c.handle().values(rt)), vec![1, 2]);
    store.checkpoint().unwrap();
    assert_eq!(snapshot(&log), base);
    c.insert(&store, 3).unwrap();
    store.set(x, 6).unwrap();
}

fn sorted(mut values: Vec<u32>) -> Vec<u32> {
    values.sort();
    values
}

struct Counted<C: MemoCells> {
    store: Arc<Store<C, MemoryBackend>>,
    memo: Memo<C, MemoryBackend>,
    x: Input<u64>,
    scaled: MemoTable<u32, u64>,
}

type Declared<C> = (
    Arc<Store<C, MemoryBackend>>,
    Input<u64>,
    MemoBuilder<C>,
    MemoTable<u32, u64>,
);

fn declare<C: MemoCells>(log: &MemoryBackend, runs: &Arc<AtomicUsize>) -> Declared<C> {
    let store = Arc::new(open::<C>(log));
    let x = store.input("x", 1_u64).unwrap();
    let xh = x.handle();
    let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(BUILD)));
    let scaled = builder.table::<u32, u64>("scaled", 1);
    let runs = Arc::clone(runs);
    builder.define(scaled, move |ctx, k| {
        runs.fetch_add(1, SeqCst);
        ctx.rt().get(xh) * u64::from(*k)
    });
    (store, x, builder, scaled)
}

fn counted<C: MemoCells>(
    log: &MemoryBackend,
    memo: &MemoryMemo,
    runs: &Arc<AtomicUsize>,
) -> Counted<C> {
    let (store, x, builder, scaled) = declare::<C>(log, runs);
    let memo = builder.open(&store, memo.clone()).unwrap();
    Counted {
        store,
        memo,
        x,
        scaled,
    }
}

fn read<C: MemoCells>(w: &Counted<C>, k: u32) -> u64 {
    let q = w.memo.get_or_insert(w.scaled, k);
    w.memo.runtime().get(q)
}

/// Shared only: a Local collection can't be captured by a closure.
#[test]
fn a_memo_lookup_in_a_scope_computes_the_candidate_and_records_nothing() {
    let log = MemoryBackend::new();
    let memo_log = MemoryMemo::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let open_memo = || {
        let store = Arc::new(open::<Shared>(&log));
        let c = store.collection::<u32>("c").unwrap();
        let ch = c.handle().clone();
        let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(BUILD)));
        let count = builder.table::<u32, u64>("count", 1);
        let runs = Arc::clone(&runs);
        builder.define(count, move |ctx, _| {
            runs.fetch_add(1, SeqCst);
            ch.values(ctx.rt()).len() as u64
        });
        let memo = builder.open(&store, memo_log.clone()).unwrap();
        (store, c, memo, count)
    };
    {
        let (store, c, memo, count) = open_memo();
        c.insert(&store, 1).unwrap();
        c.insert(&store, 2).unwrap();
        let q = memo.get_or_insert(count, 0);
        assert_eq!(memo.runtime().get(q), 2);
        assert_eq!(memo.checkpoint().unwrap().written, 1);
    }
    runs.store(0, SeqCst);
    let (store, c, memo, count) = open_memo();
    let rt = store.runtime();
    let scope = store.unlogged_scope().unwrap();
    let handle = scope.collection(&c).unwrap();
    handle.insert(rt, 3);
    let q = memo.get_or_insert(count, 0);
    assert_eq!(rt.get(q), 3);
    assert_eq!(runs.load(SeqCst), 1);
    assert_eq!(memo.stats(), MemoStats::default());
    handle.delete(rt, &3);
    scope.close().unwrap();
    assert_eq!(rt.get(q), 2);
    assert_eq!(memo.stats(), MemoStats::default());
}

fn a_memo_entry_over_a_touched_input_survives_a_checkpoint_after_close<C: MemoCells>() {
    let log = MemoryBackend::new();
    let memo = MemoryMemo::new();
    let runs = Arc::new(AtomicUsize::new(0));
    {
        let w = counted::<C>(&log, &memo, &runs);
        w.store.set(w.x, 3).unwrap();
        assert_eq!(read(&w, 2), 6);
        let mut scope = w.store.unlogged_scope().unwrap();
        scope.touched(w.x).unwrap();
        let rt = w.store.runtime();
        rt.set(scope.input(w.x).unwrap(), 50);
        assert_eq!(read(&w, 2), 100);
        rt.set(w.x.handle(), 3);
        scope.close().unwrap();
        assert_eq!(read(&w, 2), 6);
        assert_eq!(w.memo.checkpoint().unwrap().written, 1);
    }
    runs.store(0, SeqCst);
    let w = counted::<C>(&log, &memo, &runs);
    assert_eq!(read(&w, 2), 6);
    assert_eq!(runs.load(SeqCst), 0);
    assert_eq!(w.memo.stats().reused, 1);
}

fn a_memo_opened_after_a_scope_vouches_for_the_inputs_it_touched<C: MemoCells>() {
    let log = MemoryBackend::new();
    let memo = MemoryMemo::new();
    let runs = Arc::new(AtomicUsize::new(0));
    {
        let (store, x, builder, scaled) = declare::<C>(&log, &runs);
        store.set(x, 3).unwrap();
        let mut scope = store.unlogged_scope().unwrap();
        scope.touched(x).unwrap();
        let rt = store.runtime();
        rt.set(x.handle(), 50);
        rt.set(x.handle(), 3);
        scope.close().unwrap();
        let memo_handle = builder.open(&store, memo.clone()).unwrap();
        let q = memo_handle.get_or_insert(scaled, 2);
        assert_eq!(rt.get(q), 6);
        assert_eq!(memo_handle.checkpoint().unwrap().written, 1);
    }
    runs.store(0, SeqCst);
    let w = counted::<C>(&log, &memo, &runs);
    assert_eq!(read(&w, 2), 6);
    assert_eq!(runs.load(SeqCst), 0);
}

fn an_input_changed_before_the_scope_is_never_vouched<C: MemoCells>() {
    let log = MemoryBackend::new();
    let memo = MemoryMemo::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let w = counted::<C>(&log, &memo, &runs);
    assert_eq!(read(&w, 2), 2);
    w.store.runtime().set(w.x.handle(), 8);
    w.store.runtime().set(w.x.handle(), 1);
    let mut scope = w.store.unlogged_scope().unwrap();
    scope.touched(w.x).unwrap();
    scope.close().unwrap();
    assert_eq!(w.memo.checkpoint().unwrap().written, 0);
}

fn each_choke_point_refuses_while_a_scope_is_open<C: MemoCells>() {
    let log = MemoryBackend::new();
    let memo = MemoryMemo::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let w = counted::<C>(&log, &memo, &runs);
    assert_eq!(read(&w, 2), 2);
    let c = w.store.collection::<u32>("c").unwrap();
    w.store.input("y", 0_u8).unwrap();
    let scope = w.store.unlogged_scope().unwrap();
    assert_refused("set", w.store.set(w.x, 2));
    assert_refused(
        "batch",
        w.store.batch(|b| {
            b.set(w.x, 2);
        }),
    );
    assert_refused("collection insert", c.insert(&w.store, 1));
    assert_refused("delete", w.store.delete("x"));
    assert_refused("undeclare", w.store.undeclare("y"));
    assert_refused("declare input", w.store.input("fresh", 0_u8));
    assert_refused("declare collection", w.store.collection::<u8>("z"));
    assert_refused("checkpoint", w.store.checkpoint());
    assert_refused("refresh", w.store.refresh());
    assert_refused("second scope", w.store.unlogged_scope().map(|_| ()));
    assert_refused("memo checkpoint", w.memo.checkpoint());
    scope.close().unwrap();
    w.store.set(w.x, 2).unwrap();
    assert_eq!(read(&w, 2), 4);
    assert_eq!(w.memo.checkpoint().unwrap().written, 1);
}

fn opening_a_memo_refuses_while_a_scope_is_open<C: MemoCells>() {
    let log = MemoryBackend::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let (store, _x, builder, _) = declare::<C>(&log, &runs);
    let scope = store.unlogged_scope().unwrap();
    assert_refused(
        "open_dep_index",
        builder.open(&store, MemoryMemo::new()).map(|_| ()),
    );
    scope.close().unwrap();
}

fn opening_a_second_memo_refuses_while_a_scope_is_open<C: MemoCells>() {
    let log = MemoryBackend::new();
    let memo = MemoryMemo::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let w = counted::<C>(&log, &memo, &runs);
    let scope = w.store.unlogged_scope().unwrap();
    let mut builder = MemoBuilder::<C>::new(MemoOptions::new(BuildId::Fixed(BUILD)));
    let t = builder.table::<u32, u64>("t", 1);
    builder.define(t, |_, _| 0);
    assert_refused(
        "open_dep_index with an existing index",
        builder.open(&w.store, MemoryMemo::new()).map(|_| ()),
    );
    scope.close().unwrap();
}

fn a_scope_dropped_without_close_poisons_the_store<C: MemoCells>() {
    let log = MemoryBackend::new();
    let memo = MemoryMemo::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let w = counted::<C>(&log, &memo, &runs);
    {
        let _scope = w.store.unlogged_scope().unwrap();
    }
    assert!(matches!(w.store.set(w.x, 2), Err(Error::Poisoned)));
    assert!(matches!(w.store.checkpoint(), Err(Error::Poisoned)));
    assert!(matches!(w.store.refresh(), Err(Error::Poisoned)));
}

fn a_closed_scope_does_not_poison<C: MemoCells>() {
    let log = MemoryBackend::new();
    let store = open::<C>(&log);
    let x = store.input("x", 1_u8).unwrap();
    store.unlogged_scope().unwrap().close().unwrap();
    store.set(x, 2).unwrap();
}

fn a_scope_over_a_foreign_handle_fails<C: MemoCells>() {
    let log = MemoryBackend::new();
    let store = open::<C>(&log);
    let other = open::<C>(&MemoryBackend::new());
    let _ = store.input("x", 1_u8).unwrap();
    let foreign = other.input("x", 1_u8).unwrap();
    let mut scope = store.unlogged_scope().unwrap();
    assert!(matches!(scope.touched(foreign), Err(Error::UnknownHandle)));
    assert!(matches!(scope.input(foreign), Err(Error::UnknownHandle)));
    scope.close().unwrap();
}

#[allow(clippy::arc_with_non_send_sync)]
fn a_shared_scope_holds_the_store_and_closes_or_poisons_like_a_borrowed_one<C: MemoCells>() {
    let log = MemoryBackend::new();
    let store = Arc::new(open::<C>(&log));
    let x = store.input("x", 1_u8).unwrap();
    let mut scope = store.unlogged_scope_shared().unwrap();
    assert!(matches!(
        store.unlogged_scope(),
        Err(Error::UnloggedScopeOpen)
    ));
    scope.touched(x).unwrap();
    store.runtime().set(scope.input(x).unwrap(), 9);
    store.runtime().set(x.handle(), 1);
    scope.close().unwrap();
    store.set(x, 2).unwrap();

    drop(store.unlogged_scope_shared().unwrap());
    assert!(matches!(store.set(x, 3), Err(Error::Poisoned)));
}

macro_rules! both {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(#[test] fn $name() { super::$name::<super::Local>(); })*
        }
        mod shared {
            $(#[test] fn $name() { super::$name::<super::Shared>(); })*
        }
    };
}

both!(
    a_shared_scope_holds_the_store_and_closes_or_poisons_like_a_borrowed_one,
    a_memo_entry_over_a_touched_input_survives_a_checkpoint_after_close,
    a_memo_opened_after_a_scope_vouches_for_the_inputs_it_touched,
    an_input_changed_before_the_scope_is_never_vouched,
    each_choke_point_refuses_while_a_scope_is_open,
    opening_a_memo_refuses_while_a_scope_is_open,
    opening_a_second_memo_refuses_while_a_scope_is_open,
    a_scope_dropped_without_close_poisons_the_store,
    a_closed_scope_does_not_poison,
    a_scope_over_a_foreign_handle_fails,
);
