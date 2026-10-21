//! `MemoBuilder`, `Memo` and `MemoTable` over a store, with "restarts"
//! that reopen a fresh store and memo on the same backends.

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;
use std::time::{Duration, Instant};

use incr_core::lock_order::{acquire, LockKind};

use incr_persist::memo::{
    BuildId, Memo, MemoBackend, MemoBuilder, MemoCells, MemoOptions, MemoTable, MemoryMemo,
};
use incr_persist::{Error, Local, MemoryBackend, Options, Shared, Store};
use serde::{Deserialize, Serialize};

macro_rules! both {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Local>();
                }
            )*
        }
        mod shared {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Shared>();
                }
            )*
        }
    };
}

both!(
    a_restart_reuses_entries_without_running,
    a_changed_input_reruns_the_entry,
    a_recompute_first_cutoff_spares_its_dependents,
    another_build_salt_or_version_drops_the_entries,
    externals_gate_reuse,
    a_corrupt_frame_is_created_cold_and_counted,
    mutually_recursive_tables_restore_whole,
    a_restore_inside_a_compute_reuses_the_entry,
    get_finds_only_live_nodes,
    an_input_written_around_the_store_is_never_trusted,
    an_entry_over_a_bypassed_input_is_not_checkpointed,
    externals_belong_to_their_memo,
    an_external_written_around_set_external_is_not_checkpointed,
    another_value_type_drops_the_table,
    a_checkpoint_removes_the_generation_it_replaces,
    an_undefined_table_fails_open,
    a_restore_drops_no_key_under_the_memo_lock,
    a_checkpoint_drops_no_key_under_the_memo_lock,
);

const BUILD: u128 = 0xb1d;

type St<C> = Arc<Store<C, MemoryBackend>>;

struct Backends {
    log: MemoryBackend,
    memo: MemoryMemo,
}

impl Backends {
    fn new() -> Self {
        Backends {
            log: MemoryBackend::new(),
            memo: MemoryMemo::new(),
        }
    }

    fn store<C: MemoCells>(&self) -> St<C> {
        Arc::new(Store::open(self.log.clone(), Options::default()).unwrap())
    }
}

/// `scaled(k) = x * k`, counting runs.
struct Scaled<C: MemoCells> {
    store: St<C>,
    memo: Memo<C, MemoryBackend>,
    x: incr_persist::Input<u64>,
    scaled: MemoTable<u32, u64>,
}

fn scaled<C: MemoCells>(
    backends: &Backends,
    options: MemoOptions,
    version: u32,
    runs: &Arc<AtomicUsize>,
) -> Scaled<C> {
    let store = backends.store::<C>();
    let x = store.input("x", 1_u64).unwrap();
    let xh = x.handle();
    let mut builder = MemoBuilder::new(options);
    let scaled = builder.table::<u32, u64>("scaled", version);
    let runs = Arc::clone(runs);
    builder.define(scaled, move |ctx, k| {
        runs.fetch_add(1, SeqCst);
        ctx.rt().get(xh) * u64::from(*k)
    });
    let memo = builder.open(&store, backends.memo.clone()).unwrap();
    Scaled {
        store,
        memo,
        x,
        scaled,
    }
}

fn options() -> MemoOptions {
    MemoOptions::new(BuildId::Fixed(BUILD))
}

fn read<C: MemoCells>(w: &Scaled<C>, k: u32) -> u64 {
    let q = w.memo.get_or_insert(w.scaled, k);
    w.memo.runtime().get(q)
}

fn a_restart_reuses_entries_without_running<C: MemoCells>() {
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    {
        let w = scaled::<C>(&backends, options(), 1, &runs);
        w.store.set(w.x, 3).unwrap();
        assert_eq!(read(&w, 2), 6);
        assert_eq!(w.memo.checkpoint().unwrap().written, 1);
    }
    let w = scaled::<C>(&backends, options(), 1, &runs);
    assert_eq!(read(&w, 2), 6);
    assert_eq!(runs.load(SeqCst), 1);
    assert_eq!(w.memo.stats().reused, 1);
}

fn a_changed_input_reruns_the_entry<C: MemoCells>() {
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    {
        let w = scaled::<C>(&backends, options(), 1, &runs);
        assert_eq!(read(&w, 2), 2);
        w.memo.checkpoint().unwrap();
    }
    let w = scaled::<C>(&backends, options(), 1, &runs);
    w.store.set(w.x, 5).unwrap();
    assert_eq!(read(&w, 2), 10);
    assert_eq!(runs.load(SeqCst), 2);
    let stats = w.memo.stats();
    assert_eq!((stats.reused, stats.revalidated), (0, 1));
}

/// `label(k) = parity(k) + 10`, `parity(k) = x % 2`. After a restart with
/// x moved from 1 to 3, parity runs again and keeps its value, so label,
/// which trusts it, verifies without running.
fn a_recompute_first_cutoff_spares_its_dependents<C: MemoCells>() {
    let backends = Backends::new();
    let parity_runs = Arc::new(AtomicUsize::new(0));
    let label_runs = Arc::new(AtomicUsize::new(0));
    let open = || {
        let store = backends.store::<C>();
        let x = store.input("x", 1_u64).unwrap();
        let xh = x.handle();
        let mut builder = MemoBuilder::new(options());
        let parity = builder.table::<u32, u64>("parity", 1);
        let label = builder.table::<u32, u64>("label", 1);
        let runs = Arc::clone(&parity_runs);
        builder.define(parity, move |ctx, _| {
            runs.fetch_add(1, SeqCst);
            ctx.rt().get(xh) % 2
        });
        let runs = Arc::clone(&label_runs);
        builder.define(label, move |ctx, k| {
            runs.fetch_add(1, SeqCst);
            ctx.get(parity, *k) + 10
        });
        let memo = builder.open(&store, backends.memo.clone()).unwrap();
        (store, x, memo, label)
    };
    {
        let (_store, _x, memo, label) = open();
        let q = memo.get_or_insert(label, 0);
        assert_eq!(memo.runtime().get(q), 11);
        assert_eq!(memo.checkpoint().unwrap().written, 2);
    }
    let (store, x, memo, label) = open();
    store.set(x, 3).unwrap();
    let q = memo.get_or_insert(label, 0);
    assert_eq!(memo.runtime().get(q), 11);
    assert_eq!(parity_runs.load(SeqCst), 2);
    assert_eq!(label_runs.load(SeqCst), 1);
    let stats = memo.stats();
    assert_eq!((stats.reused, stats.revalidated), (1, 1));
}

fn another_build_salt_or_version_drops_the_entries<C: MemoCells>() {
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    {
        let w = scaled::<C>(&backends, options(), 1, &runs);
        assert_eq!(read(&w, 2), 2);
        w.memo.checkpoint().unwrap();
    }
    let reopened = [
        (MemoOptions::new(BuildId::Fixed(BUILD + 1)), 1),
        (options().salt("other config"), 1),
        (options(), 2),
    ];
    for (i, (options, version)) in reopened.into_iter().enumerate() {
        let w = scaled::<C>(&backends, options, version, &runs);
        assert_eq!(read(&w, 2), 2);
        assert_eq!(runs.load(SeqCst), 2 + i, "reopen {i} reused an entry");
        assert_eq!(w.memo.stats().reused, 0);
    }
    let w = scaled::<C>(&backends, options(), 1, &runs);
    assert_eq!(read(&w, 2), 2);
    assert_eq!(
        runs.load(SeqCst),
        4,
        "the original identity still has its entry"
    );
}

fn externals_gate_reuse<C: MemoCells>() {
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let open = |fp: Option<u128>| {
        let store = backends.store::<C>();
        let mut builder = MemoBuilder::new(options());
        let t = builder.table::<u32, u128>("t", 1);
        let runs = Arc::clone(&runs);
        builder.define(t, move |ctx, k| {
            runs.fetch_add(1, SeqCst);
            ctx.external("cfg") + u128::from(*k)
        });
        let memo = builder.open(&store, backends.memo.clone()).unwrap();
        if let Some(fp) = fp {
            memo.set_external("cfg", fp);
        }
        (store, memo, t)
    };
    {
        let (_store, memo, t) = open(Some(7));
        let q = memo.get_or_insert(t, 1);
        assert_eq!(memo.runtime().get(q), 8);
        memo.checkpoint().unwrap();
    }
    for (fp, value, total) in [(Some(7), 8, 1), (Some(9), 10, 2)] {
        let (_store, memo, t) = open(fp);
        let q = memo.get_or_insert(t, 1);
        assert_eq!(memo.runtime().get(q), value);
        assert_eq!(runs.load(SeqCst), total);
    }
    // An unset external reads 0, as a stored entry over it recorded.
    {
        let (_store, memo, t) = open(None);
        let q = memo.get_or_insert(t, 2);
        assert_eq!(memo.runtime().get(q), 2);
        memo.checkpoint().unwrap();
    }
    let (_store, memo, t) = open(None);
    let q = memo.get_or_insert(t, 2);
    assert_eq!(memo.runtime().get(q), 2);
    assert_eq!(runs.load(SeqCst), 3);
}

/// Shared only: a Local collection can't be captured by a closure.
#[test]
fn an_entry_over_a_collection_is_reused() {
    type C = Shared;
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let open = || {
        let store = backends.store::<C>();
        let items = store.collection::<u64>("items").unwrap();
        let view = items.handle().clone();
        let mut builder = MemoBuilder::new(options());
        let sum = builder.table::<(), u64>("sum", 1);
        let runs = Arc::clone(&runs);
        builder.define(sum, move |ctx, _| {
            runs.fetch_add(1, SeqCst);
            view.values(ctx.rt()).iter().sum()
        });
        let memo = builder.open(&store, backends.memo.clone()).unwrap();
        (store, items, memo, sum)
    };
    {
        let (store, items, memo, sum) = open();
        items.insert(&store, 4).unwrap();
        items.insert(&store, 5).unwrap();
        let q = memo.get_or_insert(sum, ());
        assert_eq!(memo.runtime().get(q), 9);
        memo.checkpoint().unwrap();
    }
    {
        let (_store, _items, memo, sum) = open();
        let q = memo.get_or_insert(sum, ());
        assert_eq!(memo.runtime().get(q), 9);
        assert_eq!(runs.load(SeqCst), 1);
    }
    let (store, items, memo, sum) = open();
    items.insert(&store, 1).unwrap();
    let q = memo.get_or_insert(sum, ());
    assert_eq!(memo.runtime().get(q), 10);
    assert_eq!(runs.load(SeqCst), 2);
}

fn a_corrupt_frame_is_created_cold_and_counted<C: MemoCells>() {
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    {
        let w = scaled::<C>(&backends, options(), 1, &runs);
        assert_eq!(read(&w, 2), 2);
        w.memo.checkpoint().unwrap();
    }
    let generation = backends.memo.load().unwrap().unwrap();
    let item = generation.items()[0];
    backends
        .memo
        .replace_frame(generation.id(), &item, vec![0xff; item.len as usize]);
    let w = scaled::<C>(&backends, options(), 1, &runs);
    assert_eq!(read(&w, 2), 2);
    assert_eq!(runs.load(SeqCst), 2);
    let stats = w.memo.stats();
    assert_eq!((stats.corrupt, stats.cold, stats.reused), (1, 1, 0));
}

/// `even(n)` and `odd(n)` read each other down to 0.
fn mutually_recursive_tables_restore_whole<C: MemoCells>() {
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let open = || {
        let store = backends.store::<C>();
        let mut builder = MemoBuilder::new(options());
        let even = builder.table::<u32, bool>("even", 1);
        let odd = builder.table::<u32, bool>("odd", 1);
        let r = Arc::clone(&runs);
        builder.define(even, move |ctx, &n| {
            r.fetch_add(1, SeqCst);
            n == 0 || ctx.get(odd, n - 1)
        });
        let r = Arc::clone(&runs);
        builder.define(odd, move |ctx, &n| {
            r.fetch_add(1, SeqCst);
            n != 0 && ctx.get(even, n - 1)
        });
        let memo = builder.open(&store, backends.memo.clone()).unwrap();
        (store, memo, even)
    };
    {
        let (_store, memo, even) = open();
        let q = memo.get_or_insert(even, 10);
        assert!(memo.runtime().get(q));
        assert_eq!(memo.checkpoint().unwrap().written, 11);
    }
    let (_store, memo, even) = open();
    let q = memo.get_or_insert(even, 10);
    assert!(memo.runtime().get(q));
    assert_eq!(runs.load(SeqCst), 11);
    assert_eq!(memo.stats().reused, 11);
}

/// Only `inner` was checkpointed; `outer` runs cold and restores `inner`
/// from inside its compute.
fn a_restore_inside_a_compute_reuses_the_entry<C: MemoCells>() {
    let backends = Backends::new();
    let inner_runs = Arc::new(AtomicUsize::new(0));
    let open = || {
        let store = backends.store::<C>();
        let mut builder = MemoBuilder::new(options());
        let inner = builder.table::<u32, u64>("inner", 1);
        let outer = builder.table::<u32, u64>("outer", 1);
        let runs = Arc::clone(&inner_runs);
        builder.define(inner, move |_, &k| {
            runs.fetch_add(1, SeqCst);
            u64::from(k) * 100
        });
        builder.define(outer, move |ctx, &k| ctx.get(inner, k) + 1);
        let memo = builder.open(&store, backends.memo.clone()).unwrap();
        (store, memo, inner, outer)
    };
    {
        let (_store, memo, inner, _outer) = open();
        let q = memo.get_or_insert(inner, 4);
        assert_eq!(memo.runtime().get(q), 400);
        memo.checkpoint().unwrap();
    }
    let (_store, memo, _inner, outer) = open();
    let q = memo.get_or_insert(outer, 4);
    assert_eq!(memo.runtime().get(q), 401);
    assert_eq!(inner_runs.load(SeqCst), 1);
    assert_eq!(memo.stats().reused, 1);
}

/// A direct `rt.set` leaves the input's fp at the logged bytes, so only
/// its stamp shows the entry over it is stale.
fn an_input_written_around_the_store_is_never_trusted<C: MemoCells>() {
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    {
        let w = scaled::<C>(&backends, options(), 1, &runs);
        assert_eq!(read(&w, 2), 2);
        w.memo.checkpoint().unwrap();
    }
    let w = scaled::<C>(&backends, options(), 1, &runs);
    w.memo.runtime().set(w.x.handle(), 5);
    assert_eq!(read(&w, 2), 10);
    assert_eq!(runs.load(SeqCst), 2);
}

/// A bypass stamps the input past its mapped stamp, and an equal mapped
/// write after it is a no-op that leaves the stamp behind, so nothing
/// over the input is checkpointed.
fn an_entry_over_a_bypassed_input_is_not_checkpointed<C: MemoCells>() {
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let w = scaled::<C>(&backends, options(), 1, &runs);
    w.memo.runtime().set(w.x.handle(), 5);
    assert_eq!(read(&w, 2), 10);
    assert_eq!(w.memo.checkpoint().unwrap().written, 0);
    // Debug builds refuse the write as a bypass; release builds apply it.
    match w.store.set(w.x, 5) {
        Ok(_) | Err(Error::Bypassed { .. }) => {}
        Err(other) => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(w.memo.checkpoint().unwrap().written, 0);
}

/// One memo sets an external and is dropped; another memo on the same
/// store restores its entry before setting the external, so it must not
/// trust the first memo's node.
fn externals_belong_to_their_memo<C: MemoCells>() {
    let backends = Backends::new();
    let store = backends.store::<C>();
    let runs = Arc::new(AtomicUsize::new(0));
    let open = || {
        let mut builder = MemoBuilder::new(options());
        let t = builder.table::<u32, u128>("t", 1);
        let runs = Arc::clone(&runs);
        builder.define(t, move |ctx, k| {
            runs.fetch_add(1, SeqCst);
            ctx.external("cfg") + u128::from(*k)
        });
        (builder.open(&store, backends.memo.clone()).unwrap(), t)
    };
    {
        let (memo, t) = open();
        memo.set_external("cfg", 1);
        let q = memo.get_or_insert(t, 1);
        assert_eq!(memo.runtime().get(q), 2);
        memo.checkpoint().unwrap();
    }
    let (memo, t) = open();
    let q = memo.get_or_insert(t, 1);
    assert_eq!(memo.runtime().get(q), 1);
    assert_eq!(runs.load(SeqCst), 2);
    memo.set_external("cfg", 5);
    assert_eq!(memo.runtime().get(q), 6);
}

fn an_external_written_around_set_external_is_not_checkpointed<C: MemoCells>() {
    let backends = Backends::new();
    let store = backends.store::<C>();
    let mut builder = MemoBuilder::new(options());
    let t = builder.table::<u32, u128>("t", 1);
    builder.define(t, |ctx, k| ctx.external("cfg") + u128::from(*k));
    let memo = builder.open(&store, backends.memo.clone()).unwrap();
    let cfg = memo.set_external("cfg", 1);
    memo.runtime().set(cfg, 2);
    let q = memo.get_or_insert(t, 1);
    assert_eq!(memo.runtime().get(q), 3);
    assert_eq!(memo.checkpoint().unwrap().written, 0);
}

fn another_value_type_drops_the_table<C: MemoCells>() {
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    {
        let w = scaled::<C>(&backends, options(), 1, &runs);
        assert_eq!(read(&w, 2), 2);
        w.memo.checkpoint().unwrap();
    }
    let store = backends.store::<C>();
    let mut builder = MemoBuilder::new(options());
    let scaled = builder.table::<u32, u32>("scaled", 1);
    let r = Arc::clone(&runs);
    builder.define(scaled, move |_, &k| {
        r.fetch_add(1, SeqCst);
        k
    });
    let memo = builder.open(&store, backends.memo.clone()).unwrap();
    let q = memo.get_or_insert(scaled, 2);
    assert_eq!(memo.runtime().get(q), 2);
    assert_eq!(runs.load(SeqCst), 2);
    assert_eq!(memo.stats().reused, 0);
}

fn a_checkpoint_removes_the_generation_it_replaces<C: MemoCells>() {
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let w = scaled::<C>(&backends, options(), 1, &runs);
    assert_eq!(read(&w, 2), 2);
    w.memo.checkpoint().unwrap();
    let first = backends.memo.load().unwrap().unwrap();
    let item = first.items()[0];
    assert!(backends
        .memo
        .read_entry(first.id(), &item)
        .unwrap()
        .is_some());
    w.memo.checkpoint().unwrap();
    let second = backends.memo.load().unwrap().unwrap();
    assert_ne!(second.id(), first.id());
    assert_eq!(backends.memo.read_entry(first.id(), &item).unwrap(), None);
    assert_eq!(w.memo.stats().stale_generations, 0);
}

fn get_finds_only_live_nodes<C: MemoCells>() {
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let w = scaled::<C>(&backends, options(), 1, &runs);
    assert_eq!(w.memo.get(w.scaled, &2), None);
    let q = w.memo.get_or_insert(w.scaled, 2);
    assert_eq!(w.memo.get(w.scaled, &2), Some(q));
}

fn an_undefined_table_fails_open<C: MemoCells>() {
    let backends = Backends::new();
    let store = backends.store::<C>();
    let mut builder = MemoBuilder::<C>::new(options());
    builder.table::<u32, u64>("declared only", 1);
    match builder.open(&store, backends.memo.clone()) {
        Err(Error::UndefinedTable(name)) => assert_eq!(name, "declared only"),
        other => panic!("expected UndefinedTable, got {:?}", other.err()),
    }
}

#[test]
fn automatic_checkpoints_need_a_shared_runtime() {
    let backends = Backends::new();
    let store = backends.store::<Local>();
    let builder = MemoBuilder::<Local>::new(options().auto_checkpoint_secs(1));
    assert!(matches!(
        builder.open(&store, backends.memo.clone()),
        Err(Error::AutoCheckpointNeedsShared)
    ));
}

#[test]
fn automatic_checkpoints_write_and_stop_on_drop() {
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    {
        let w = scaled::<Shared>(&backends, options().auto_checkpoint_secs(1), 1, &runs);
        assert_eq!(read(&w, 3), 3);
        let deadline = Instant::now() + Duration::from_secs(30);
        while w.memo.stats().checkpoints == 0 {
            assert!(Instant::now() < deadline, "no automatic checkpoint ran");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    let w = scaled::<Shared>(&backends, options(), 1, &runs);
    assert_eq!(read(&w, 3), 3);
    assert_eq!(runs.load(SeqCst), 1);
}

thread_local! {
    static WATCHING: Cell<bool> = const { Cell::new(false) };
    static DROPS: Cell<usize> = const { Cell::new(0) };
    static UNDER_M: Cell<usize> = const { Cell::new(0) };
}

/// A key whose drop, while watched, counts itself and whether the memo
/// lock M is held, which the debug lock-order checker shows by refusing
/// to take M again.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
struct Watched(u32);

impl Drop for Watched {
    fn drop(&mut self) {
        if !WATCHING.with(Cell::get) {
            return;
        }
        DROPS.with(|n| n.set(n.get() + 1));
        if catch_unwind(AssertUnwindSafe(|| drop(acquire(LockKind::Memo)))).is_err() {
            UNDER_M.with(|n| n.set(n.get() + 1));
        }
    }
}

/// `f`'s watched key drops on this thread, and how many of them ran under M.
fn watch(f: impl FnOnce()) -> (usize, usize) {
    DROPS.with(|n| n.set(0));
    UNDER_M.with(|n| n.set(0));
    WATCHING.with(|w| w.set(true));
    f();
    WATCHING.with(|w| w.set(false));
    (DROPS.with(Cell::get), UNDER_M.with(Cell::get))
}

/// Enough entries that the restore's table grows while M is held, each
/// dep decoded by the walk.
fn a_restore_drops_no_key_under_the_memo_lock<C: MemoCells>() {
    const N: u32 = 40;
    let backends = Backends::new();
    let open = || {
        let store = backends.store::<C>();
        let mut builder = MemoBuilder::new(options());
        let chain = builder.table::<Watched, u64>("chain", 1);
        builder.define(chain, move |ctx, k| match k.0 {
            0 => 0,
            n => ctx.get(chain, Watched(n - 1)) + 1,
        });
        let memo = builder.open(&store, backends.memo.clone()).unwrap();
        (store, memo, chain)
    };
    {
        let (_store, memo, chain) = open();
        let q = memo.get_or_insert(chain, Watched(N));
        assert_eq!(memo.runtime().get(q), u64::from(N));
        memo.checkpoint().unwrap();
    }
    let (_store, memo, chain) = open();
    let (drops, under_m) = watch(|| {
        let q = memo.get_or_insert(chain, Watched(N));
        assert_eq!(memo.runtime().get(q), u64::from(N));
    });
    assert_eq!(memo.stats().reused, u64::from(N) + 1);
    assert!(drops > 0);
    assert_eq!(under_m, 0);
}

/// A Dirty entry is skipped by the snapshot, which holds M.
fn a_checkpoint_drops_no_key_under_the_memo_lock<C: MemoCells>() {
    let backends = Backends::new();
    let store = backends.store::<C>();
    let x = store.input("x", 1_u64).unwrap();
    let xh = x.handle();
    let mut builder = MemoBuilder::new(options());
    let t = builder.table::<Watched, u64>("t", 1);
    builder.define(t, move |ctx, k| ctx.rt().get(xh) + u64::from(k.0));
    let memo = builder.open(&store, backends.memo.clone()).unwrap();
    let q = memo.get_or_insert(t, Watched(1));
    assert_eq!(memo.runtime().get(q), 2);
    store.set(x, 2).unwrap();
    let (drops, under_m) = watch(|| {
        assert_eq!(memo.checkpoint().unwrap().written, 0);
    });
    assert!(drops > 0);
    assert_eq!(under_m, 0);
}
