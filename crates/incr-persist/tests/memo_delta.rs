//! Delta memo checkpoints: a few dirty entries append a small layer, and a
//! chain restores exactly like the full generation it stands for.

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

use incr_persist::memo::{
    key_hash, CheckpointStats, DeltaHeader, EncodedEntry, GenHeader, GenId, Generation, IndexItem,
    Tombstone,
};
use incr_persist::memo::{
    BuildId, Memo, MemoBackend, MemoBuilder, MemoCells, MemoOptions, MemoStats, MemoTable,
    MemoryMemo,
};
use incr_persist::{Input, Local, MemoryBackend, Options, Result, Shared, Store};
use proptest::prelude::*;
use proptest::test_runner::{Config as RunnerConfig, TestRunner};

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
    a_one_percent_dirty_checkpoint_writes_a_delta_of_the_dirty_entries,
    restore_then_checkpoint_is_unchanged,
    a_chain_restores_like_a_full_generation,
    a_tombstoned_entrys_stored_dependents_restore_cold,
);

const BUILD: u128 = 0xde;
const WIDE: u32 = 2000;

type St<C> = Arc<Store<C, MemoryBackend>>;

#[derive(Clone, Default)]
struct PlainMemo(MemoryMemo);

impl MemoBackend for PlainMemo {
    type Guard = <MemoryMemo as MemoBackend>::Guard;

    fn lock_checkpoint(&self) -> Result<Self::Guard> {
        self.0.lock_checkpoint()
    }

    fn load(&self) -> Result<Option<Generation>> {
        self.0.load()
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        self.0.read_entry(generation, item)
    }

    fn write_generation(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &[EncodedEntry],
    ) -> Result<Generation> {
        self.0.write_generation(base, id, header, entries)
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        self.0.remove_generation(generation)
    }
}

struct Backends<M> {
    log: MemoryBackend,
    memo: M,
}

impl<M: MemoBackend + Clone + 'static> Backends<M> {
    fn new(memo: M) -> Self {
        Backends {
            log: MemoryBackend::new(),
            memo,
        }
    }

    fn store<C: MemoCells>(&self) -> St<C> {
        Arc::new(Store::open(self.log.clone(), Options::default()).unwrap())
    }
}

fn options() -> MemoOptions {
    MemoOptions::new(BuildId::Fixed(BUILD))
}

struct Wide<C: MemoCells> {
    store: St<C>,
    memo: Memo<C, MemoryBackend>,
    inputs: Vec<Input<u64>>,
    table: MemoTable<u32, u64>,
    runs: Arc<AtomicUsize>,
}

impl<C: MemoCells> Wide<C> {
    fn open(backends: &Backends<MemoryMemo>) -> Self {
        let store = backends.store::<C>();
        let inputs: Vec<Input<u64>> = (0..WIDE)
            .map(|i| store.input(&format!("w{i}"), u64::from(i)).unwrap())
            .collect();
        let handles: Arc<Vec<_>> = Arc::new(inputs.iter().map(|i| i.handle()).collect());
        let runs = Arc::new(AtomicUsize::new(0));
        let mut builder = MemoBuilder::new(options());
        let table = builder.table::<u32, u64>("wide", 1);
        let counter = Arc::clone(&runs);
        builder.define(table, move |ctx, &k| {
            counter.fetch_add(1, SeqCst);
            ctx.rt().get(handles[k as usize]) + 1
        });
        let memo = builder.open(&store, backends.memo.clone()).unwrap();
        Wide {
            store,
            memo,
            inputs,
            table,
            runs,
        }
    }

    fn read_all(&self) {
        for k in 0..WIDE {
            let q = self.memo.get_or_insert(self.table, k);
            assert_eq!(self.memo.runtime().get(q), u64::from(k) + 1);
        }
    }

    fn runs(&self) -> usize {
        self.runs.load(SeqCst)
    }
}

fn a_one_percent_dirty_checkpoint_writes_a_delta_of_the_dirty_entries<C: MemoCells>() {
    let backends = Backends::new(MemoryMemo::new());
    let w = Wide::<C>::open(&backends);
    w.read_all();
    let first = w.memo.checkpoint().unwrap();
    assert!(!first.delta);
    assert!(first.bytes_appended > 0);

    for i in 0..20 {
        w.store.set(w.inputs[i * 100], 1_000_000).unwrap();
    }
    for k in 0..WIDE {
        let q = w.memo.get_or_insert(w.table, k);
        let want = if k % 100 == 0 {
            1_000_001
        } else {
            u64::from(k) + 1
        };
        assert_eq!(w.memo.runtime().get(q), want);
    }
    let second = w.memo.checkpoint().unwrap();
    assert!(second.delta, "{second:?}");
    assert_eq!(second.fresh_written, 20);
    assert!(!second.compacted);
    assert_eq!(second.chain_depth, 1);
    assert!(second.bytes_appended > 0);
    assert!(
        second.bytes_appended * 10 < first.bytes_appended,
        "{} vs {}",
        second.bytes_appended,
        first.bytes_appended
    );

    let stats = w.memo.stats();
    assert_eq!(stats.chain_depth, 1);
    assert_eq!(stats.delta_checkpoints, 1);
    assert_eq!(stats.last_bytes_appended, second.bytes_appended);
    assert_eq!(stats.chain_bytes, second.bytes_appended);
    assert_eq!(stats.entries, u64::from(WIDE));
}

fn restore_then_checkpoint_is_unchanged<C: MemoCells>() {
    let backends = Backends::new(MemoryMemo::new());
    {
        let w = Wide::<C>::open(&backends);
        w.read_all();
        assert!(!w.memo.checkpoint().unwrap().delta);
        for i in 0..20 {
            w.store.set(w.inputs[i * 7], 1_000_000).unwrap();
        }
        for k in 0..WIDE {
            let q = w.memo.get_or_insert(w.table, k);
            w.memo.runtime().get(q);
        }
        assert!(w.memo.checkpoint().unwrap().delta);
    }
    let before = backends.memo.load_chain(&[]).unwrap().unwrap();
    assert_eq!(before.depth(), 1);

    let w = Wide::<C>::open(&backends);
    for k in 0..WIDE {
        let q = w.memo.get_or_insert(w.table, k);
        w.memo.runtime().get(q);
    }
    assert_eq!(w.runs(), 0);
    let stats = w.memo.stats();
    assert_eq!((stats.reused, stats.revalidated, stats.cold), (2000, 0, 0));
    let done = w.memo.checkpoint().unwrap();
    assert!(done.unchanged, "{done:?}");
    assert_eq!(done.bytes_appended, 0);
    let after = backends.memo.load_chain(&[]).unwrap().unwrap();
    assert_eq!(after.depth(), before.depth());
    assert_eq!(after.tip().id(), before.tip().id());
    assert_eq!(after.tip().items(), before.tip().items());
}

const INPUTS: usize = 64;
const KEYS: u32 = 64;
const TABLES: usize = 3;
const NAMES: [&str; TABLES] = ["base", "top", "side"];
const IN_ORDER: [usize; 3] = [0, 1, 2];

struct Rig<C: MemoCells> {
    store: St<C>,
    memo: Memo<C, MemoryBackend>,
    inputs: Vec<Input<u64>>,
    tables: [Option<MemoTable<u32, u64>>; TABLES],
    runs: Arc<AtomicUsize>,
}

impl<C: MemoCells> Rig<C> {
    fn open<M: MemoBackend + Clone + 'static>(backends: &Backends<M>, order: &[usize]) -> Self {
        let store = backends.store::<C>();
        let inputs: Vec<Input<u64>> = (0..INPUTS)
            .map(|i| store.input(&format!("in{i}"), i as u64).unwrap())
            .collect();
        let handles: Arc<Vec<_>> = Arc::new(inputs.iter().map(|i| i.handle()).collect());
        let runs = Arc::new(AtomicUsize::new(0));
        let mut builder = MemoBuilder::new(options());
        let mut tables: [Option<MemoTable<u32, u64>>; TABLES] = [None; TABLES];
        for &t in order {
            tables[t] = Some(builder.table::<u32, u64>(NAMES[t], 1));
        }
        let n = INPUTS as u32;
        if let Some(base) = tables[0] {
            let (counter, h) = (Arc::clone(&runs), Arc::clone(&handles));
            builder.define(base, move |ctx, &k| {
                counter.fetch_add(1, SeqCst);
                ctx.rt().get(h[(k % n) as usize]) * 10 + u64::from(k)
            });
        }
        if let (Some(top), Some(base)) = (tables[1], tables[0]) {
            let (counter, h) = (Arc::clone(&runs), Arc::clone(&handles));
            builder.define(top, move |ctx, &k| {
                counter.fetch_add(1, SeqCst);
                ctx.get(base, k) + 1 + ctx.rt().get(h[((k + 1) % n) as usize])
            });
        }
        if let Some(side) = tables[2] {
            let (counter, h) = (Arc::clone(&runs), Arc::clone(&handles));
            builder.define(side, move |ctx, &k| {
                counter.fetch_add(1, SeqCst);
                ctx.rt().get(h[((k + 3) % n) as usize]) + u64::from(k)
            });
        }
        let memo = builder.open(&store, backends.memo.clone()).unwrap();
        Rig {
            store,
            memo,
            inputs,
            tables,
            runs,
        }
    }

    fn set(&self, input: usize, value: u64) {
        self.store.set(self.inputs[input], value).unwrap();
    }

    fn read(&self, table: usize, key: u32) -> Option<u64> {
        let table = self.tables[table]?;
        let q = self.memo.get_or_insert(table, key);
        Some(self.memo.runtime().get(q))
    }

    fn read_all(&self) -> Vec<Option<u64>> {
        (0..TABLES)
            .flat_map(|t| (0..KEYS).map(move |k| (t, k)))
            .map(|(t, k)| self.read(t, k))
            .collect()
    }

    fn runs(&self) -> usize {
        self.runs.load(SeqCst)
    }
}

#[derive(Debug)]
struct Round {
    sets: Vec<usize>,
    reads: Vec<(usize, u32)>,
}

fn rounds() -> impl Strategy<Value = Vec<Round>> {
    let round = (
        prop::collection::vec(0..INPUTS, 0..4),
        prop::collection::vec((0..TABLES, 0..KEYS), 0..40),
    )
        .prop_map(|(sets, reads)| Round { sets, reads });
    prop::collection::vec(round, 1..=3)
}

struct Pair<C: MemoCells> {
    delta: Backends<MemoryMemo>,
    full: Backends<PlainMemo>,
    live: Option<(Rig<C>, Rig<C>)>,
}

impl<C: MemoCells> Pair<C> {
    fn new() -> Self {
        Pair {
            delta: Backends::new(MemoryMemo::new()),
            full: Backends::new(PlainMemo::default()),
            live: None,
        }
    }

    fn reopen(&mut self, order: &[usize]) {
        self.live = None;
        self.live = Some((Rig::open(&self.delta, order), Rig::open(&self.full, order)));
    }

    fn rigs(&self) -> &(Rig<C>, Rig<C>) {
        self.live.as_ref().unwrap()
    }

    fn set(&self, input: usize, value: u64) {
        let (d, f) = self.rigs();
        d.set(input, value);
        f.set(input, value);
    }

    fn checkpoint(&self) -> (CheckpointStats, CheckpointStats) {
        let (d, f) = self.rigs();
        (d.memo.checkpoint().unwrap(), f.memo.checkpoint().unwrap())
    }

    fn checkpoint_expecting_delta(&self) {
        let (d, f) = self.checkpoint();
        assert!(d.delta, "{d:?}");
        assert!(!f.delta, "{f:?}");
    }

    fn read_all_tables(&self, tables: &[usize]) {
        let (d, f) = self.rigs();
        for &t in tables {
            for k in 0..KEYS {
                assert_eq!(d.read(t, k), f.read(t, k));
            }
        }
    }

    fn assert_restores_alike(&mut self) {
        self.reopen(&IN_ORDER);
        let (d, f) = self.rigs();
        assert_eq!(d.read_all(), f.read_all());
        let counts = |s: MemoStats| (s.reused, s.revalidated, s.cold, s.corrupt);
        assert_eq!(counts(d.memo.stats()), counts(f.memo.stats()));
        assert_eq!(d.runs(), f.runs());
    }
}

fn a_chain_restores_like_a_full_generation<C: MemoCells>() {
    let mut runner = TestRunner::new(RunnerConfig::with_cases(24));
    runner
        .run(&rounds(), |rounds| {
            let mut pair = Pair::<C>::new();
            pair.reopen(&IN_ORDER);
            pair.read_all_tables(&[0, 1, 2]);
            let (d, f) = pair.checkpoint();
            prop_assert!(!d.delta && !f.delta);

            pair.set(0, 5000);
            pair.read_all_tables(&[0, 1, 2]);
            pair.checkpoint_expecting_delta();

            for (i, round) in rounds.iter().enumerate() {
                for (j, &input) in round.sets.iter().enumerate() {
                    pair.set(input, 1000 + (i * 100 + j) as u64);
                }
                let (d, f) = pair.rigs();
                for &(t, k) in &round.reads {
                    prop_assert_eq!(d.read(t, k), f.read(t, k));
                }
                pair.checkpoint();
            }
            pair.assert_restores_alike();

            pair.reopen(&[0, 1]);
            pair.set(1, 6000);
            pair.read_all_tables(&[0, 1]);
            pair.checkpoint_expecting_delta();
            pair.set(2, 6001);
            pair.read_all_tables(&[0, 1]);
            pair.checkpoint_expecting_delta();
            pair.assert_restores_alike();

            pair.reopen(&[1, 2, 0]);
            pair.set(3, 7000);
            pair.read_all_tables(&[0, 1, 2]);
            pair.checkpoint_expecting_delta();
            pair.assert_restores_alike();

            Ok(())
        })
        .unwrap();
}

/// `top(k) = leaf(k) * 10`, `leaf(k) = x + k`. A delta that tombstones
/// `leaf(1)`, as one does for an oversized rewrite, leaves the base's
/// `top(1)` naming a dep no layer shows: the restore recomputes `leaf(1)`
/// and reruns `top(1)` once as revalidated, to the right value, where a
/// full write would have dropped `top(1)`.
fn a_tombstoned_entrys_stored_dependents_restore_cold<C: MemoCells>() {
    let backends = Backends::new(MemoryMemo::new());
    let runs = Arc::new(AtomicUsize::new(0));
    let open = |backends: &Backends<MemoryMemo>| {
        let store = backends.store::<C>();
        let x = store.input("x", 4_u64).unwrap();
        let xh = x.handle();
        let mut builder = MemoBuilder::new(options());
        let leaf = builder.table::<u32, u64>("leaf", 1);
        let top = builder.table::<u32, u64>("top", 1);
        let counter = Arc::clone(&runs);
        builder.define(leaf, move |ctx, &k| {
            counter.fetch_add(1, SeqCst);
            ctx.rt().get(xh) + u64::from(k)
        });
        let counter = Arc::clone(&runs);
        builder.define(top, move |ctx, &k| {
            counter.fetch_add(1, SeqCst);
            ctx.get(leaf, k) * 10
        });
        let memo = builder.open(&store, backends.memo.clone()).unwrap();
        (store, memo, top)
    };
    {
        let (_store, memo, top) = open(&backends);
        assert_eq!(memo.runtime().get(memo.get_or_insert(top, 1)), 50);
        assert_eq!(memo.checkpoint().unwrap().written, 2);
    }
    let base = backends.memo.load().unwrap().unwrap();
    let leaf_table = base
        .header()
        .tables
        .iter()
        .position(|desc| desc.name == "leaf")
        .unwrap() as u32;
    let key = postcard::to_stdvec(&1_u32).unwrap();
    let delta = DeltaHeader {
        header: GenHeader {
            seq: base.header().seq + 1,
            ..base.header().clone()
        },
        base: base.id(),
        parent: base.id(),
    };
    backends
        .memo
        .write_delta(
            base.id(),
            GenId(base.id().0 + 1),
            &delta,
            &[],
            &[Tombstone {
                table: leaf_table,
                key_hash: key_hash(&key),
            }],
        )
        .unwrap();

    let before = runs.load(SeqCst);
    let (_store, memo, top) = open(&backends);
    assert_eq!(memo.stats().entries, 1);
    assert_eq!(memo.runtime().get(memo.get_or_insert(top, 1)), 50);
    assert_eq!(runs.load(SeqCst), before + 2);
    let stats = memo.stats();
    assert_eq!((stats.reused, stats.revalidated, stats.cold), (0, 1, 0));
}
