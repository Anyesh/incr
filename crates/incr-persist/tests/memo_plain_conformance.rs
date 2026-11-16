//! A backend that implements only the required `MemoBackend` methods gets
//! the provided defaults: no deltas, and full checkpoints as before.

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

use incr_persist::memo::{
    conformance::MemoHarness, write_checkpoint, BuildId, CheckpointInput, CheckpointStats,
    CheckpointTimes, DeltaHeader, EncodedEntry, Entry, GenHeader, GenId, Generation, IndexItem,
    MemoBackend, MemoBuilder, MemoOptions, MemoryMemo, StoredDep, TableDesc,
};
use incr_persist::{Error, MemoryBackend, Options, Result, Shared, Store};

/// The required methods only, counting the frames `read_entry` returns.
#[derive(Clone, Default)]
struct PlainMemo(MemoryMemo, Arc<AtomicUsize>);

impl MemoBackend for PlainMemo {
    type Guard = <MemoryMemo as MemoBackend>::Guard;

    fn lock_checkpoint(&self) -> Result<Self::Guard> {
        self.0.lock_checkpoint()
    }

    fn load(&self) -> Result<Option<Generation>> {
        self.0.load()
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        self.1.fetch_add(1, SeqCst);
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

struct PlainHarness;

impl MemoHarness for PlainHarness {
    type Backend = PlainMemo;
    type Store = PlainMemo;

    fn new_store(&mut self) -> PlainMemo {
        PlainMemo::default()
    }

    fn open(&mut self, store: &PlainMemo) -> PlainMemo {
        store.clone()
    }

    fn exclusive_lock(&self) -> bool {
        true
    }
}

incr_persist::memo_backend_conformance!(PlainHarness);

fn desc(name: &str) -> TableDesc {
    TableDesc {
        name: name.into(),
        version: 1,
        key_type: "u8".into(),
        value_type: "u8".into(),
    }
}

fn header(seq: u64) -> GenHeader {
    GenHeader {
        build: 3,
        seq,
        tables: vec![desc("a"), desc("b")],
    }
}

fn input(keys: std::ops::Range<u8>) -> CheckpointInput {
    CheckpointInput {
        build: 3,
        tables: vec![desc("a"), desc("b")],
        fresh: keys
            .map(|key| Entry {
                table: u32::from(key % 2),
                key: vec![key],
                value: vec![key; 5],
                deps: if key > 0 {
                    vec![StoredDep::Query {
                        table: u32::from((key - 1) % 2),
                        key: vec![key - 1],
                        fp: 0,
                    }]
                } else {
                    Vec::new()
                },
                last_used: 0,
            })
            .collect(),
        max_bytes: 1 << 20,
    }
}

#[test]
fn the_defaults_offer_no_delta() {
    let plain = PlainMemo::default();
    assert!(!plain.supports_delta());
    assert!(plain.load_chain(&[]).unwrap().is_none());
    let generation = plain
        .write_generation(None, GenId(1), &header(1), &[])
        .unwrap();
    let chain = plain.load_chain(&[]).unwrap().unwrap();
    assert_eq!(chain.depth(), 0);
    assert_eq!(chain.tip().id(), generation.id());
    assert_eq!(chain.tip().items(), generation.items());

    let delta = DeltaHeader {
        header: header(2),
        base: GenId(1),
        parent: GenId(1),
    };
    let refused = plain.write_delta(GenId(1), GenId(2), &delta, &[], &[]);
    assert!(
        matches!(refused, Err(Error::Unsupported(_))),
        "got {refused:?}"
    );
    assert_eq!(plain.load_chain(&[]).unwrap().unwrap().tip().id(), GenId(1));
}

#[test]
fn the_default_read_entries_reads_each_item() {
    let plain = PlainMemo::default();
    let done = write_checkpoint(&plain, input(0..6)).unwrap();
    let items = done.generation.items();
    let batched = plain.read_entries(done.generation.id(), items).unwrap();
    let single: Vec<_> = items
        .iter()
        .map(|item| plain.read_entry(done.generation.id(), item).unwrap())
        .collect();
    assert_eq!(batched, single);
    assert!(batched.iter().all(Option::is_some));
}

#[test]
fn full_checkpoints_match_a_memory_store_without_deltas() {
    let plain = PlainMemo::default();
    let memory = MemoryMemo::without_deltas();
    for keys in [0..6, 3..9, 9..10] {
        let a = write_checkpoint(&plain, input(keys.clone())).unwrap();
        let b = write_checkpoint(&memory, input(keys)).unwrap();
        let untimed = |stats: &CheckpointStats| CheckpointStats {
            times: CheckpointTimes::default(),
            ..stats.clone()
        };
        assert_eq!(untimed(&a.stats), untimed(&b.stats));
        assert!(!a.stats.delta);
        assert_eq!(a.superseded, b.superseded);
        assert_eq!(a.generation.header(), b.generation.header());
        assert_eq!(a.generation.items(), b.generation.items());
        for item in a.generation.items() {
            assert_eq!(
                plain.read_entry(a.generation.id(), item).unwrap(),
                memory.read_entry(b.generation.id(), item).unwrap()
            );
        }
        for &old in &a.superseded {
            plain.remove_generation(old).unwrap();
            memory.remove_generation(old).unwrap();
        }
    }
}

#[test]
fn a_memo_at_its_cap_reads_each_stored_entry_at_most_once_per_checkpoint() {
    let plain = PlainMemo::default();
    let store =
        Arc::new(Store::<Shared, _>::open(MemoryBackend::new(), Options::default()).unwrap());
    let x = store.input("x", 1_u64).unwrap().handle();
    let max_bytes = 16 << 10;
    let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(7)).max_bytes(max_bytes));
    let t = builder.table::<u32, u64>("t", 1);
    let u = builder.table::<u32, u64>("u", 1);
    builder.define(t, move |ctx, &k| ctx.rt().get(x) + u64::from(k));
    builder.define(u, move |ctx, &k| ctx.get(t, k) + 1);
    let memo = builder.open(&store, plain.clone()).unwrap();
    let read = |table, k| memo.runtime().get(memo.get_or_insert(table, k));
    let mut evicting = 0;
    for round in 0..200_u32 {
        let keys = 10 * round..10 * round + 10;
        for k in keys.clone() {
            read(t, k);
            if round > 0 {
                read(u, k - 10);
            }
        }
        let stored = memo.stats().entries as usize;
        plain.1.store(0, SeqCst);
        let done = memo.checkpoint().unwrap();
        let reads = plain.1.load(SeqCst);
        assert!(
            reads <= stored,
            "round {round}: {reads} reads for {stored} stored entries"
        );
        assert!(memo.stats().bytes <= max_bytes, "round {round}");
        if done.evicted > 0 {
            evicting += 1;
            if evicting == 5 {
                return;
            }
        }
    }
    panic!("the memo never reached its cap");
}
