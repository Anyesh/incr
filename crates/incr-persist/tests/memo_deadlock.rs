//! The memo deadlock suite. Each scenario runs in a child process that is
//! killed after 10 s, and the parent asserts how the child exited: cleanly
//! where nothing may wait forever, or with the contract panic where value
//! code calls into a lock its caller may hold.

#[path = "../../incr-core/tests/common/child.rs"]
mod child;

use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Barrier, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use child::run_in_child;
use incr_core::persist_hooks::assert_invariants;
use incr_core::{CycleStrategy, Fixpoint, Incr, QueryOpts};
use incr_persist::memo::{
    BuildId, Memo, MemoBuilder, MemoCells, MemoOptions, MemoTable, MemoValue, MemoryMemo,
};
use incr_persist::{Local, MemoryBackend, Options, Shared, Store};
use serde::{Deserialize, Serialize};

const CONTRACT_MESSAGE: &str = "value and key methods must not call into the runtime";
const FRAME_MESSAGE: &str = "Store methods must not be called from inside a query's compute";
const FAILED_MESSAGE: &str = "read but its last compute panicked";
const BUILD: u128 = 0x0dea_d10c;
const ROUNDS: u32 = 8;
/// How long a thread that must be inside a wait before another acts gives
/// it to get there, after a flag says it is about to.
const SETTLE: Duration = Duration::from_millis(20);

/// Runs each generic scenario under Local and Shared in a child, and checks
/// the child's exit with the named `Outcome` method.
macro_rules! in_child {
    ($($name:ident: $check:ident $(($arg:expr))?),* $(,)?) => {
        mod local {
            $(
                #[test]
                #[cfg_attr(miri, ignore)]
                fn $name() {
                    super::run_in_child(
                        concat!("local::", stringify!($name)),
                        super::$name::<super::Local>,
                    )
                    .$check($($arg)?);
                }
            )*
        }
        mod shared {
            $(
                #[test]
                #[cfg_attr(miri, ignore)]
                fn $name() {
                    super::run_in_child(
                        concat!("shared::", stringify!($name)),
                        super::$name::<super::Shared>,
                    )
                    .$check($($arg)?);
                }
            )*
        }
    };
}

in_child!(
    insert_whose_element_hash_checkpoints: assert_panicked_with(super::CONTRACT_MESSAGE),
    insert_whose_element_hash_sets_an_external: assert_panicked_with(super::CONTRACT_MESSAGE),
    insert_whose_element_hash_restores_an_entry: assert_panicked_with(super::CONTRACT_MESSAGE),
    a_declare_catches_up_a_foreign_insert_under_a_memo: assert_clean,
);

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

fn options() -> MemoOptions {
    MemoOptions::new(BuildId::Fixed(BUILD))
}

fn read<T: MemoValue>(memo: &Memo<Shared, MemoryBackend>, table: MemoTable<u32, T>, key: u32) -> T {
    memo.runtime().get(memo.get_or_insert(table, key))
}

fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
    match err.downcast::<String>() {
        Ok(s) => *s,
        Err(err) => err.downcast::<&str>().map(|s| s.to_string()).unwrap(),
    }
}

fn wait_for(flag: &AtomicBool) {
    while !flag.load(SeqCst) {
        thread::yield_now();
    }
}

thread_local! {
    static HOOK: RefCell<Option<Box<dyn Fn()>>> = const { RefCell::new(None) };
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

fn arm(f: impl Fn() + 'static) {
    HOOK.with(|h| *h.borrow_mut() = Some(Box::new(f)));
}

/// Runs the armed hook once, on the thread that armed it.
fn fire() {
    if let Some(f) = HOOK.with(|h| h.borrow_mut().take()) {
        f();
    }
}

/// An element whose `Hash` fires this thread's hook.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
struct Hooked(u32);

impl Hash for Hooked {
    fn hash<H: Hasher>(&self, state: &mut H) {
        fire();
        self.0.hash(state);
    }
}

/// A value whose `PartialEq` fires this thread's hook.
#[derive(Clone, Debug)]
struct Compared(u32);

impl PartialEq for Compared {
    fn eq(&self, other: &Self) -> bool {
        fire();
        self.0 == other.0
    }
}

/// `even(n)` and `odd(n)` read each other down to 0.
struct Parity {
    memo: Memo<Shared, MemoryBackend>,
    even: MemoTable<u32, bool>,
    odd: MemoTable<u32, bool>,
}

fn parity(backends: &Backends, runs: &Arc<AtomicUsize>) -> Parity {
    let store = backends.store::<Shared>();
    let mut builder = MemoBuilder::new(options());
    let even = builder.table::<u32, bool>("even", 1);
    let odd = builder.table::<u32, bool>("odd", 1);
    let r = Arc::clone(runs);
    builder.define(even, move |ctx, &n| {
        r.fetch_add(1, SeqCst);
        n == 0 || ctx.get(odd, n - 1)
    });
    let r = Arc::clone(runs);
    builder.define(odd, move |ctx, &n| {
        r.fetch_add(1, SeqCst);
        n != 0 && ctx.get(even, n - 1)
    });
    let memo = builder.open(&store, backends.memo.clone()).unwrap();
    Parity { memo, even, odd }
}

const PARITY_KEYS: u32 = 48;

/// One thread restores `even` then `odd` upwards while another restores
/// `odd` then `even` downwards, so their walks cross the two tables in
/// opposite orders. Every entry is still restored once, and only one
/// seeded while the other thread held its dep's claim runs, once
/// (`a_restore_over_a_claimed_dep_reruns_the_entry_once` in restore_tests).
#[test]
#[cfg_attr(miri, ignore)]
fn mutually_recursive_tables_restored_in_opposite_orders() {
    run_in_child(
        "mutually_recursive_tables_restored_in_opposite_orders",
        || {
            let backends = Backends::new();
            let runs = Arc::new(AtomicUsize::new(0));
            {
                let p = parity(&backends, &runs);
                for k in 0..PARITY_KEYS {
                    read(&p.memo, p.even, k);
                    read(&p.memo, p.odd, k);
                }
                let written = p.memo.checkpoint().unwrap().written;
                assert_eq!(written, 2 * PARITY_KEYS as usize);
            }
            for _ in 0..ROUNDS {
                let before = runs.load(SeqCst);
                let p = parity(&backends, &runs);
                let start = Barrier::new(2);
                thread::scope(|s| {
                    s.spawn(|| {
                        start.wait();
                        for k in 0..PARITY_KEYS {
                            assert_eq!(read(&p.memo, p.even, k), k.is_multiple_of(2));
                            assert_eq!(read(&p.memo, p.odd, k), k % 2 == 1);
                        }
                    });
                    s.spawn(|| {
                        start.wait();
                        for k in (0..PARITY_KEYS).rev() {
                            assert_eq!(read(&p.memo, p.odd, k), k % 2 == 1);
                            assert_eq!(read(&p.memo, p.even, k), k.is_multiple_of(2));
                        }
                    });
                });
                let stats = p.memo.stats();
                assert_eq!(stats.reused + stats.revalidated, 2 * u64::from(PARITY_KEYS));
                assert_eq!((runs.load(SeqCst) - before) as u64, stats.revalidated);
                assert_invariants(p.memo.runtime());
            }
        },
    )
    .assert_clean();
}

const NESTED_KEYS: u32 = 32;

/// `inner(k) = 100 k`, checkpointed alone, and `outer(k) = inner(k) + x`,
/// whose compute restores `inner(k)` after giving way for a while, so a
/// reader of `outer(k)` arriving meanwhile waits on its claim.
struct Nested {
    store: St<Shared>,
    memo: Arc<Memo<Shared, MemoryBackend>>,
    tick: incr_persist::Input<u32>,
    inner: MemoTable<u32, u64>,
    outer: MemoTable<u32, u64>,
}

fn nested(backends: &Backends, inner_runs: &Arc<AtomicUsize>) -> Nested {
    let store = backends.store::<Shared>();
    let x = store.input("x", 2_u64).unwrap().handle();
    let tick = store.input("tick", 0_u32).unwrap();
    let mut builder = MemoBuilder::new(options());
    let inner = builder.table::<u32, u64>("inner", 1);
    let outer = builder.table::<u32, u64>("outer", 1);
    let runs = Arc::clone(inner_runs);
    builder.define(inner, move |_, &k| {
        runs.fetch_add(1, SeqCst);
        u64::from(k) * 100
    });
    builder.define(outer, move |ctx, &k| {
        for _ in 0..64 {
            thread::yield_now();
        }
        ctx.get(inner, k) + ctx.rt().get(x)
    });
    let memo = Arc::new(builder.open(&store, backends.memo.clone()).unwrap());
    Nested {
        store,
        memo,
        tick,
        inner,
        outer,
    }
}

/// Two threads each compute an `outer` entry whose compute restores its
/// `inner` entry under M, while a third thread's store write runs a
/// subscriber that, under S, looks both entries up with `Memo::get` and
/// reads them with `rt.get`, so it waits on the computing threads' claims.
/// Restores never request S and the subscriber requests no memo lock while
/// it waits, so every round completes.
#[test]
#[cfg_attr(miri, ignore)]
fn restores_inside_computes_while_a_subscriber_reads_under_the_store_mutex() {
    run_in_child(
        "restores_inside_computes_while_a_subscriber_reads_under_the_store_mutex",
        || {
            let backends = Backends::new();
            let inner_runs = Arc::new(AtomicUsize::new(0));
            {
                let n = nested(&backends, &inner_runs);
                for k in 0..2 * NESTED_KEYS {
                    read(&n.memo, n.inner, k);
                }
                n.memo.checkpoint().unwrap();
            }
            let n = nested(&backends, &inner_runs);
            let targets: Arc<[AtomicU32; 2]> = Arc::new([AtomicU32::new(0), AtomicU32::new(0)]);
            let subscriber_reads = Arc::new(AtomicUsize::new(0));
            let (memo, wanted, reads, outer) = (
                Arc::clone(&n.memo),
                Arc::clone(&targets),
                Arc::clone(&subscriber_reads),
                n.outer,
            );
            n.store
                .subscribe(move |_, _, _| {
                    for target in wanted.iter() {
                        let k = target.load(SeqCst);
                        if let Some(q) = memo.get(outer, &k) {
                            assert_eq!(memo.runtime().get(q), u64::from(k) * 100 + 2);
                            reads.fetch_add(1, SeqCst);
                        }
                    }
                })
                .unwrap();
            for k in 0..NESTED_KEYS {
                targets[0].store(k, SeqCst);
                targets[1].store(k + NESTED_KEYS, SeqCst);
                let start = Barrier::new(3);
                thread::scope(|s| {
                    for target in targets.iter() {
                        let (n, start) = (&n, &start);
                        s.spawn(move || {
                            let k = target.load(SeqCst);
                            start.wait();
                            assert_eq!(read(&n.memo, n.outer, k), u64::from(k) * 100 + 2);
                        });
                    }
                    start.wait();
                    n.store.set(n.tick, k + 1).unwrap();
                });
            }
            assert!(
                subscriber_reads.load(SeqCst) > 0,
                "the subscriber never read"
            );
            assert_eq!(inner_runs.load(SeqCst), 2 * NESTED_KEYS as usize);
            assert_eq!(n.memo.stats().reused, 2 * u64::from(NESTED_KEYS));
            assert_invariants(n.memo.runtime());
        },
    )
    .assert_clean();
}

const LEAVES: u32 = 64;

/// `leaf(k) = 10 k`, checkpointed for even `k` only, and
/// `pair(k) = leaf(k) + leaf(k + 1)`, never checkpointed.
struct Pairs {
    memo: Memo<Shared, MemoryBackend>,
    leaf: MemoTable<u32, u64>,
    pair: MemoTable<u32, u64>,
}

fn pairs(backends: &Backends, leaf_runs: &Arc<AtomicUsize>) -> Pairs {
    let store = backends.store::<Shared>();
    let mut builder = MemoBuilder::new(options());
    let leaf = builder.table::<u32, u64>("leaf", 1);
    let pair = builder.table::<u32, u64>("pair", 1);
    let runs = Arc::clone(leaf_runs);
    builder.define(leaf, move |_, &k| {
        runs.fetch_add(1, SeqCst);
        u64::from(k) * 10
    });
    builder.define(pair, move |ctx, &k| ctx.get(leaf, k) + ctx.get(leaf, k + 1));
    let memo = builder.open(&store, backends.memo.clone()).unwrap();
    Pairs { memo, leaf, pair }
}

/// A restorer walks the even leaves down while a cold misser walks the
/// pairs up: each `pair` and odd `leaf` misses the index and is created
/// without M, and each pair's compute reaches an even leaf the restorer
/// may be seeding at that moment. Each even leaf is restored exactly once
/// and each odd leaf runs exactly once.
#[test]
#[cfg_attr(miri, ignore)]
fn a_cold_misser_racing_a_restorer() {
    run_in_child("a_cold_misser_racing_a_restorer", || {
        let backends = Backends::new();
        let leaf_runs = Arc::new(AtomicUsize::new(0));
        let evens = (0..=LEAVES).filter(|k| k.is_multiple_of(2)).count();
        {
            let p = pairs(&backends, &leaf_runs);
            for k in (0..=LEAVES).filter(|k| k.is_multiple_of(2)) {
                read(&p.memo, p.leaf, k);
            }
            assert_eq!(p.memo.checkpoint().unwrap().written, evens);
        }
        for _ in 0..ROUNDS {
            leaf_runs.store(0, SeqCst);
            let p = pairs(&backends, &leaf_runs);
            let start = Barrier::new(2);
            thread::scope(|s| {
                s.spawn(|| {
                    start.wait();
                    for k in (0..=LEAVES).rev().filter(|k| k.is_multiple_of(2)) {
                        assert_eq!(read(&p.memo, p.leaf, k), u64::from(k) * 10);
                    }
                });
                s.spawn(|| {
                    start.wait();
                    for k in 0..LEAVES {
                        if k % 2 == 1 {
                            assert_eq!(read(&p.memo, p.leaf, k), u64::from(k) * 10);
                        }
                        assert_eq!(read(&p.memo, p.pair, k), u64::from(k) * 20 + 10);
                    }
                });
            });
            assert_eq!(leaf_runs.load(SeqCst), LEAVES as usize + 1 - evens);
            assert_eq!(p.memo.stats().reused, evens as u64);
            assert_invariants(p.memo.runtime());
        }
    })
    .assert_clean();
}

const AGGREGATE_ROUNDS: u64 = 300;

/// `below` aggregates a store collection with a `combine` that reads `m`,
/// and `m`, a Fixpoint query, reads `below`. Once both hold values, each
/// round changes the collection and reads the two on two threads at once,
/// so each compute waits on the other's claim, one of them under the
/// aggregate's state lock. Wound-wait resolves the cycle on one thread by
/// the single-thread rules, under which either read order gives
/// `below = min(|S|, m + 1) = 6` and `m = 5` for ten or more elements.
#[test]
#[cfg_attr(miri, ignore)]
fn an_aggregate_whose_combine_reads_a_node_computed_on_another_thread() {
    run_in_child(
        "an_aggregate_whose_combine_reads_a_node_computed_on_another_thread",
        || {
            let store: St<Shared> =
                Arc::new(Store::open(MemoryBackend::new(), Options::default()).unwrap());
            let items = store.collection::<u64>("items").unwrap();
            let rt = store.runtime();
            let m_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
            let (combining, m) = (Arc::clone(&store), Arc::clone(&m_cell));
            // Built while the collection is empty: `combine` runs only once
            // `m` exists.
            let below = items.handle().aggregate(
                rt,
                0_u64,
                |_| 1,
                move |a, b| (a + b).min(combining.runtime().get(*m.get().unwrap()) + 1),
            );
            let m = rt.create_query_with(
                move |rt| (rt.get(below) + 1).min(5),
                QueryOpts {
                    cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_u64).max_iterations(20)),
                    ..QueryOpts::default()
                },
            );
            m_cell.set(m).unwrap();
            for x in 0..10 {
                items.insert(&store, x).unwrap();
            }
            assert_eq!((rt.get(m), rt.get(below)), (5, 6));
            for round in 0..AGGREGATE_ROUNDS {
                if round > 0 {
                    items.delete(&store, &(9 + round)).unwrap();
                }
                items.insert(&store, 10 + round).unwrap();
                let start = Barrier::new(2);
                let read = |node: Incr<u64>| {
                    start.wait();
                    catch_unwind(AssertUnwindSafe(|| rt.get(node))).map_err(panic_message)
                };
                let (b, m) = thread::scope(|s| {
                    let b = s.spawn(|| read(below));
                    let m = s.spawn(|| read(m));
                    (b.join().unwrap(), m.join().unwrap())
                });
                assert_eq!(
                    (b, m),
                    (Ok(6), Ok(5)),
                    "round {round}: the concurrent reads differ from every single-thread order"
                );
                assert_invariants(rt);
            }
        },
    )
    .assert_clean();
}

/// A subscriber, under S, reads `q` while another thread computes it, and
/// that compute calls `store.restored_keys`. The frame check refuses the
/// call before S is requested, so the compute panics instead of waiting
/// on the subscriber, whose read then reports `q` Failed.
#[test]
#[cfg_attr(miri, ignore)]
fn a_compute_calling_restored_keys_while_a_subscriber_waits_on_it() {
    run_in_child(
        "a_compute_calling_restored_keys_while_a_subscriber_waits_on_it",
        || {
            let store: St<Shared> =
                Arc::new(Store::open(MemoryBackend::new(), Options::default()).unwrap());
            let tick = store.input("tick", 0_u32).unwrap();
            let target: Arc<Mutex<Option<Incr<u32>>>> = Arc::new(Mutex::new(None));
            let computing = Arc::new(AtomicBool::new(false));
            let waiting = Arc::new(AtomicBool::new(false));
            let seen = Arc::new(Mutex::new(Vec::new()));
            let (subscriber_store, wanted, flag, out) = (
                Arc::clone(&store),
                Arc::clone(&target),
                Arc::clone(&waiting),
                Arc::clone(&seen),
            );
            store
                .subscribe(move |_, _, _| {
                    let Some(q) = *wanted.lock().unwrap() else {
                        return;
                    };
                    flag.store(true, SeqCst);
                    let read = catch_unwind(AssertUnwindSafe(|| subscriber_store.runtime().get(q)));
                    out.lock().unwrap().push(read.map_err(panic_message));
                })
                .unwrap();
            for round in 0..ROUNDS {
                computing.store(false, SeqCst);
                waiting.store(false, SeqCst);
                let (calling, entered, waited) = (
                    Arc::clone(&store),
                    Arc::clone(&computing),
                    Arc::clone(&waiting),
                );
                let q = store.runtime().create_query(move |_| {
                    if ARMED.with(Cell::get) {
                        entered.store(true, SeqCst);
                        wait_for(&waited);
                        thread::sleep(SETTLE);
                        let _ = calling.restored_keys("");
                    }
                    round
                });
                *target.lock().unwrap() = Some(q);
                thread::scope(|s| {
                    let computer = s.spawn(|| {
                        ARMED.with(|a| a.set(true));
                        catch_unwind(AssertUnwindSafe(|| store.runtime().get(q)))
                    });
                    wait_for(&computing);
                    store.set(tick, round + 1).unwrap();
                    let message = panic_message(
                        computer
                            .join()
                            .unwrap()
                            .expect_err("the compute's store call was allowed"),
                    );
                    assert!(
                        message.contains(FRAME_MESSAGE),
                        "expected the frame panic, got: {message}"
                    );
                });
                match seen.lock().unwrap().pop() {
                    Some(Err(message)) if message.contains(FAILED_MESSAGE) => {}
                    other => panic!("expected the subscriber to read q Failed, got {other:?}"),
                }
            }
            assert!(store.restored_keys("").unwrap().is_empty());
        },
    )
    .assert_clean();
}

/// A subscriber, under S, reads input `x` and then sets it, while another
/// thread's depth-0 `set(x)` holds x's claim and its `PartialEq` calls
/// `store.batch`. Reads never wait on a set's claim, but the subscriber's
/// own `set` waits on it. The setting thread has no frame, so only the
/// value-code check in `Store::state()` stops it before it requests S:
/// its `set` panics with the contract message before writing, and the
/// subscriber's read and `set` complete.
#[test]
#[cfg_attr(miri, ignore)]
fn a_set_whose_partial_eq_batches_while_a_subscriber_sets_it() {
    run_in_child(
        "a_set_whose_partial_eq_batches_while_a_subscriber_sets_it",
        || {
            let store: St<Shared> =
                Arc::new(Store::open(MemoryBackend::new(), Options::default()).unwrap());
            let tick = store.input("tick", 0_u32).unwrap();
            let x = store.runtime().create_input(Compared(0));
            let waiting = Arc::new(AtomicBool::new(false));
            let seen = Arc::new(Mutex::new(Vec::new()));
            let (subscriber_store, flag, out) =
                (Arc::clone(&store), Arc::clone(&waiting), Arc::clone(&seen));
            store
                .subscribe(move |_, _, _| {
                    flag.store(true, SeqCst);
                    let rt = subscriber_store.runtime();
                    let value = rt.get(x).0;
                    rt.set(x, Compared(value + 1));
                    out.lock().unwrap().push(value);
                })
                .unwrap();
            for round in 0..ROUNDS {
                waiting.store(false, SeqCst);
                let comparing = Arc::new(AtomicBool::new(false));
                thread::scope(|s| {
                    let setter = s.spawn(|| {
                        let (batching, entered, waited) = (
                            Arc::clone(&store),
                            Arc::clone(&comparing),
                            Arc::clone(&waiting),
                        );
                        arm(move || {
                            entered.store(true, SeqCst);
                            wait_for(&waited);
                            thread::sleep(SETTLE);
                            let _ = batching.batch(|_| {});
                        });
                        catch_unwind(AssertUnwindSafe(|| {
                            store.runtime().set(x, Compared(u32::MAX))
                        }))
                    });
                    wait_for(&comparing);
                    store.set(tick, round + 1).unwrap();
                    let message = panic_message(
                        setter
                            .join()
                            .unwrap()
                            .expect_err("the PartialEq's store call was allowed"),
                    );
                    assert!(
                        message.contains(CONTRACT_MESSAGE),
                        "expected the contract panic, got: {message}"
                    );
                });
                assert_eq!(seen.lock().unwrap().pop(), Some(round));
                assert_eq!(store.runtime().get(x).0, round + 1);
            }
        },
    )
    .assert_clean();
}

/// A store with collection `tags` and a memo whose table `t(k) = k` holds
/// a checkpointed entry for 1 and whose external `cfg` exists, and the
/// insert into `tags` whose first element `Hash`, run under the insert's
/// `DepIndex` write hold, calls `call` on the memo.
fn insert_whose_element_hash_calls<C: MemoCells>(
    call: impl Fn(&Memo<C, MemoryBackend>, MemoTable<u32, u32>) + 'static,
) {
    let backends = Backends::new();
    let open = || {
        let store = backends.store::<C>();
        let tags = store.collection::<Hooked>("tags").unwrap();
        let mut builder = MemoBuilder::new(options());
        let t = builder.table::<u32, u32>("t", 1);
        builder.define(t, |_, &k| k);
        let memo = Rc::new(builder.open(&store, backends.memo.clone()).unwrap());
        (store, tags, memo, t)
    };
    {
        let (_store, _tags, memo, t) = open();
        assert_eq!(memo.runtime().get(memo.get_or_insert(t, 1)), 1);
        memo.checkpoint().unwrap();
    }
    let (store, tags, memo, t) = open();
    memo.set_external("cfg", 0);
    let hooked = Rc::clone(&memo);
    arm(move || call(&hooked, t));
    tags.insert(&store, Hooked(1)).unwrap();
}

/// Checkpoint's snapshot would request a `DepIndex` read hold on the lock
/// its own thread holds for writing; the value-code check refuses it
/// before lock 1.
fn insert_whose_element_hash_checkpoints<C: MemoCells>() {
    insert_whose_element_hash_calls::<C>(|memo, _| {
        let _ = memo.checkpoint();
    });
}

/// `set_external` would request the `DepIndex` write hold its own thread
/// already holds; the value-code check refuses it first. `cfg` exists, so
/// no table miss checks the word on the way.
fn insert_whose_element_hash_sets_an_external<C: MemoCells>() {
    insert_whose_element_hash_calls::<C>(|memo, _| {
        memo.set_external("cfg", 1);
    });
}

/// The entry for 1 is in the index and not live, so `get_or_insert` would
/// restore it under M, below the hold in the lock order; the value-code
/// check refuses it before M.
fn insert_whose_element_hash_restores_an_entry<C: MemoCells>() {
    insert_whose_element_hash_calls::<C>(|memo, t| {
        memo.get_or_insert(t, 1);
    });
}

/// Store A, with a memo open, declares a new collection after store B on
/// the same log inserted into A's collection `items`. The declare catches
/// up B's insert as a mapped write, which takes the `DepIndex` hold
/// itself, so it must run before the declare takes its own.
fn a_declare_catches_up_a_foreign_insert_under_a_memo<C: MemoCells>() {
    let backends = Backends::new();
    let a = backends.store::<C>();
    let items = a.collection::<u64>("items").unwrap();
    let mut builder = MemoBuilder::new(options());
    let t = builder.table::<u32, u32>("t", 1);
    builder.define(t, |_, &k| k);
    let _memo = builder.open(&a, backends.memo.clone()).unwrap();

    let b = backends.store::<C>();
    b.collection::<u64>("items").unwrap().insert(&b, 7).unwrap();

    a.collection::<u64>("other").unwrap();
    assert_eq!(items.handle().values(a.runtime()), vec![7]);
}
