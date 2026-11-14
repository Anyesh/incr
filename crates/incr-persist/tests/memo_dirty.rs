//! Memo dirty tracking: a checkpoint encodes only entries computed since
//! the last checkpoint that saved them, and `MemoStats` reports what is
//! not yet in the backend.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

use incr_persist::memo::{
    BuildId, Memo, MemoBackend, MemoBuilder, MemoCells, MemoOptions, MemoTable, MemoryMemo,
};
use incr_persist::{Error, Input, Local, MemoryBackend, Options, Shared, Store};
use serde::{Deserialize, Serialize, Serializer};

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
    a_second_checkpoint_encodes_only_the_dirty_entry,
    unsaved_entries_counts_entries_computed_since_the_last_checkpoint,
    entries_and_bytes_describe_the_current_generation,
    a_clean_dep_keeps_the_fingerprint_it_was_stored_with,
    an_unsaveable_entry_stays_dirty_and_is_counted,
    evicted_entries_are_dirtied_again_and_a_reopen_recomputes_them,
    a_failed_checkpoint_leaves_its_entries_unsaved,
);

const BUILD: u128 = 0xd1;

thread_local! {
    static ENCODES: Cell<usize> = const { Cell::new(0) };
}

/// Counts its serializations on this thread; a checkpoint encodes values
/// on the thread that calls it.
#[derive(Clone, Debug, PartialEq, Deserialize)]
struct Counted(u64);

impl Serialize for Counted {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        ENCODES.with(|n| n.set(n.get() + 1));
        self.0.serialize(s)
    }
}

/// Encodes after the checkpoint's `f`, by count of values serialized.
fn encodes(f: impl FnOnce()) -> usize {
    ENCODES.with(|n| n.set(0));
    f();
    ENCODES.with(Cell::get)
}

static NOISE: AtomicU64 = AtomicU64::new(0);

/// A value that encodes differently every time and decodes equal, as a
/// pickled set can after a restart.
#[derive(Clone, Debug, Deserialize)]
struct Noisy {
    v: u64,
    noise: u64,
}

impl PartialEq for Noisy {
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

impl Serialize for Noisy {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (self.v, NOISE.fetch_add(1, SeqCst) + self.noise).serialize(s)
    }
}

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

/// `leaf(0) = x`, `leaf(k) = y + k` otherwise, counting runs.
struct Leaves<C: MemoCells> {
    store: St<C>,
    memo: Memo<C, MemoryBackend>,
    x: Input<u64>,
    y: Input<u64>,
    leaf: MemoTable<u32, Counted>,
    runs: Arc<AtomicUsize>,
}

impl<C: MemoCells> Leaves<C> {
    fn open(backends: &Backends, options: MemoOptions) -> Self {
        let store = backends.store::<C>();
        let x = store.input("x", 1_u64).unwrap();
        let y = store.input("y", 100_u64).unwrap();
        let (xh, yh) = (x.handle(), y.handle());
        let runs = Arc::new(AtomicUsize::new(0));
        let mut builder = MemoBuilder::new(options);
        let leaf = builder.table::<u32, Counted>("leaf", 1);
        let counter = Arc::clone(&runs);
        builder.define(leaf, move |ctx, &k| {
            counter.fetch_add(1, SeqCst);
            if k == 0 {
                Counted(ctx.rt().get(xh))
            } else {
                Counted(ctx.rt().get(yh) + u64::from(k))
            }
        });
        let memo = builder.open(&store, backends.memo.clone()).unwrap();
        Leaves {
            store,
            memo,
            x,
            y,
            leaf,
            runs,
        }
    }

    fn read(&self, k: u32) -> u64 {
        let q = self.memo.get_or_insert(self.leaf, k);
        self.memo.runtime().get(q).0
    }

    fn runs(&self) -> usize {
        self.runs.load(SeqCst)
    }
}

fn a_second_checkpoint_encodes_only_the_dirty_entry<C: MemoCells>() {
    let backends = Backends::new();
    {
        let w = Leaves::<C>::open(&backends, options());
        for k in 0..3 {
            w.read(k);
        }
        assert_eq!(
            encodes(|| assert_eq!(w.memo.checkpoint().unwrap().written, 3)),
            3
        );
        w.store.set(w.x, 7).unwrap();
        assert_eq!(w.read(0), 7);
        let mut written = 0;
        assert_eq!(
            encodes(|| written = w.memo.checkpoint().unwrap().written),
            1
        );
        assert_eq!(written, 3);
    }
    let w = Leaves::<C>::open(&backends, options());
    w.store.set(w.x, 7).unwrap();
    assert_eq!((w.read(0), w.read(1), w.read(2)), (7, 101, 102));
    assert_eq!(w.runs(), 0);
}

fn unsaved_entries_counts_entries_computed_since_the_last_checkpoint<C: MemoCells>() {
    let backends = Backends::new();
    let w = Leaves::<C>::open(&backends, options());
    assert_eq!(w.memo.stats().unsaved_entries, 0);
    w.read(1);
    w.read(2);
    w.read(2);
    assert_eq!(w.memo.stats().unsaved_entries, 2);
    w.memo.checkpoint().unwrap();
    assert_eq!(w.memo.stats().unsaved_entries, 0);
    w.read(1);
    assert_eq!(w.memo.stats().unsaved_entries, 0);
    w.store.set(w.y, 200).unwrap();
    w.read(1);
    w.read(2);
    assert_eq!(w.memo.stats().unsaved_entries, 2);
}

fn entries_and_bytes_describe_the_current_generation<C: MemoCells>() {
    let backends = Backends::new();
    let w = Leaves::<C>::open(&backends, options());
    w.read(1);
    w.read(2);
    let stats = w.memo.stats();
    assert_eq!((stats.entries, stats.bytes), (0, 0));
    w.memo.checkpoint().unwrap();
    let generation = incr_persist::memo::MemoBackend::load(&backends.memo)
        .unwrap()
        .unwrap();
    let stats = w.memo.stats();
    assert_eq!(stats.entries, 2);
    assert_eq!(stats.bytes, generation.entry_bytes());
    assert!(stats.bytes > 0);
}

/// `parent(k) = child(k).v + y`, `child(k) = x * k` with a value that
/// re-encodes differently each time. After `y` moves, only the parent is
/// dirty; its stored dep must carry the fp of the child's stored bytes, so
/// a reopen reuses both, once before and once after a restart.
fn a_clean_dep_keeps_the_fingerprint_it_was_stored_with<C: MemoCells>() {
    struct Tree<C: MemoCells> {
        store: St<C>,
        memo: Memo<C, MemoryBackend>,
        y: Input<u64>,
        parent: MemoTable<u32, u64>,
        runs: Arc<AtomicUsize>,
    }
    fn open<C: MemoCells>(backends: &Backends) -> Tree<C> {
        let store = backends.store::<C>();
        let x = store.input("x", 3_u64).unwrap();
        let y = store.input("y", 10_u64).unwrap();
        let (xh, yh) = (x.handle(), y.handle());
        let runs = Arc::new(AtomicUsize::new(0));
        let mut builder = MemoBuilder::new(options());
        let child = builder.table::<u32, Noisy>("child", 1);
        let parent = builder.table::<u32, u64>("parent", 1);
        let counter = Arc::clone(&runs);
        builder.define(child, move |ctx, &k| {
            counter.fetch_add(1, SeqCst);
            Noisy {
                v: ctx.rt().get(xh) * u64::from(k),
                noise: 0,
            }
        });
        let counter = Arc::clone(&runs);
        builder.define(parent, move |ctx, &k| {
            counter.fetch_add(1, SeqCst);
            ctx.get(child, k).v + ctx.rt().get(yh)
        });
        let memo = builder.open(&store, backends.memo.clone()).unwrap();
        Tree {
            store,
            memo,
            y,
            parent,
            runs,
        }
    }
    fn read<C: MemoCells>(t: &Tree<C>) -> u64 {
        let q = t.memo.get_or_insert(t.parent, 2);
        t.memo.runtime().get(q)
    }

    let backends = Backends::new();
    {
        let t = open::<C>(&backends);
        assert_eq!(read(&t), 16);
        t.memo.checkpoint().unwrap();
        t.store.set(t.y, 20).unwrap();
        assert_eq!(read(&t), 26);
        assert_eq!(t.memo.checkpoint().unwrap().written, 2);
    }
    {
        let t = open::<C>(&backends);
        t.store.set(t.y, 20).unwrap();
        assert_eq!(read(&t), 26);
        assert_eq!(t.runs.load(SeqCst), 0);
        assert_eq!(t.memo.stats().reused, 2);
        t.store.set(t.y, 30).unwrap();
        assert_eq!(read(&t), 36);
        assert_eq!(t.runs.load(SeqCst), 1);
        t.memo.checkpoint().unwrap();
    }
    let t = open::<C>(&backends);
    t.store.set(t.y, 30).unwrap();
    assert_eq!(read(&t), 36);
    assert_eq!(t.runs.load(SeqCst), 0);
    assert_eq!(t.memo.stats().reused, 2);
}

/// An entry over a bypassed input never maps, so every checkpoint tries
/// it again without counting it as unsaved.
fn an_unsaveable_entry_stays_dirty_and_is_counted<C: MemoCells>() {
    let backends = Backends::new();
    let w = Leaves::<C>::open(&backends, options());
    w.memo.runtime().set(w.x.handle(), 5);
    assert_eq!(w.read(0), 5);
    assert_eq!(w.read(1), 101);
    assert_eq!(
        encodes(|| assert_eq!(w.memo.checkpoint().unwrap().written, 1)),
        2
    );
    let stats = w.memo.stats();
    assert_eq!((stats.unsaveable, stats.unsaved_entries), (1, 0));
    assert_eq!(
        encodes(|| assert_eq!(w.memo.checkpoint().unwrap().written, 1)),
        1
    );
    let stats = w.memo.stats();
    assert_eq!((stats.unsaveable, stats.unsaved_entries), (1, 0));
}

fn evicted_entries_are_dirtied_again_and_a_reopen_recomputes_them<C: MemoCells>() {
    const N: u32 = 8;
    let sizing = Backends::new();
    let per_entry = {
        let w = Leaves::<C>::open(&sizing, options());
        for k in 1..=N {
            w.read(k);
        }
        w.memo.checkpoint().unwrap();
        w.memo.stats().bytes / u64::from(N)
    };
    let budget = options().max_bytes(per_entry * 5);
    let backends = Backends::new();
    let evicted = {
        let w = Leaves::<C>::open(&backends, budget.clone());
        for k in 1..=N {
            w.read(k);
        }
        w.memo.checkpoint().unwrap();
        let stats = w.memo.stats();
        assert_eq!(stats.evicted_last_checkpoint, 3);
        assert_eq!((stats.unsaved_entries, stats.unsaveable), (0, 0));
        let again = encodes(|| {
            let done = w.memo.checkpoint().unwrap();
            assert_eq!((done.written, done.evicted), (5, 3));
        });
        assert_eq!(again, 3);
        w.memo.stats().evicted_last_checkpoint
    };
    let w = Leaves::<C>::open(&backends, budget);
    for k in 1..=N {
        assert_eq!(w.read(k), 100 + u64::from(k));
    }
    assert_eq!(w.runs() as u64, evicted);
}

fn a_failed_checkpoint_leaves_its_entries_unsaved<C: MemoCells>() {
    let backends = Backends::new();
    let w = Leaves::<C>::open(&backends, options());
    w.read(1);
    w.read(2);
    {
        let _other = MemoBackend::lock_checkpoint(&backends.memo).unwrap();
        assert!(matches!(w.memo.checkpoint(), Err(Error::Busy)));
    }
    assert_eq!(w.memo.stats().unsaved_entries, 2);
    w.read(3);
    assert_eq!(w.memo.stats().unsaved_entries, 3);
    assert_eq!(w.memo.checkpoint().unwrap().written, 3);
    assert_eq!(w.memo.stats().unsaved_entries, 0);
}
