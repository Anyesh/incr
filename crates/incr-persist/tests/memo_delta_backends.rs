//! A delta chain on the real memo backends: a store checkpoints deltas and
//! a compaction, reopens after each checkpoint, and restores the entries a
//! full checkpoint of the same state restores.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

use incr_persist::memo::{
    BuildId, FileMemo, Memo, MemoBackend, MemoBuilder, MemoCells, MemoOptions, MemoTable,
    MAX_CHAIN_DEPTH,
};
use incr_persist::{Input, Local, MemoryBackend, Options, Shared, Store};

const BUILD: u128 = 0xdb;
const ENTRIES: u32 = 300;

macro_rules! backends {
    ($($name:ident: $open:expr),* $(,)?) => {
        $(
            mod $name {
                use super::*;

                #[test]
                fn local_chain_restores_like_a_full_generation_across_a_compaction() {
                    chain_restores_like_a_full_generation_across_a_compaction::<Local, _>($open);
                }

                #[test]
                fn shared_chain_restores_like_a_full_generation_across_a_compaction() {
                    chain_restores_like_a_full_generation_across_a_compaction::<Shared, _>($open);
                }
            }
        )*
    };
}

fn file(dir: &Path) -> FileMemo {
    FileMemo::open(dir).unwrap()
}

#[cfg(feature = "sqlite")]
fn sqlite(dir: &Path) -> incr_persist::memo::SqliteMemo {
    incr_persist::memo::SqliteMemo::open(dir.join("memo.db")).unwrap()
}

backends!(file_memo: file);
#[cfg(feature = "sqlite")]
backends!(sqlite_memo: sqlite);

struct Rig<C: MemoCells> {
    store: Arc<Store<C, MemoryBackend>>,
    memo: Memo<C, MemoryBackend>,
    inputs: Vec<Input<u64>>,
    table: MemoTable<u32, u64>,
    runs: Arc<AtomicUsize>,
}

impl<C: MemoCells> Rig<C> {
    fn open(log: &MemoryBackend, memo: impl MemoBackend + 'static) -> Self {
        let store = Arc::new(Store::open(log.clone(), Options::default()).unwrap());
        let inputs: Vec<Input<u64>> = (0..ENTRIES)
            .map(|i| store.input(&format!("w{i}"), u64::from(i)).unwrap())
            .collect();
        let handles: Arc<Vec<_>> = Arc::new(inputs.iter().map(|i| i.handle()).collect());
        let runs = Arc::new(AtomicUsize::new(0));
        let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(BUILD)));
        let table = builder.table::<u32, u64>("wide", 1);
        let counter = Arc::clone(&runs);
        builder.define(table, move |ctx, &k| {
            counter.fetch_add(1, SeqCst);
            ctx.rt().get(handles[k as usize]) + 1
        });
        let memo = builder.open(&store, memo).unwrap();
        Rig {
            store,
            memo,
            inputs,
            table,
            runs,
        }
    }

    fn read_all(&self) -> Vec<u64> {
        (0..ENTRIES)
            .map(|k| {
                let q = self.memo.get_or_insert(self.table, k);
                self.memo.runtime().get(q)
            })
            .collect()
    }

    fn runs(&self) -> usize {
        self.runs.load(SeqCst)
    }

    fn assert_restored_whole(&self, want: &[u64]) {
        assert_eq!(self.read_all(), want);
        assert_eq!(self.runs(), 0, "an entry recomputed instead of restoring");
        let stats = self.memo.stats();
        assert_eq!(
            (stats.reused, stats.revalidated, stats.cold, stats.corrupt),
            (u64::from(ENTRIES), 0, 0, 0)
        );
    }
}

fn chain_restores_like_a_full_generation_across_a_compaction<
    C: MemoCells,
    M: MemoBackend + 'static,
>(
    open: fn(&Path) -> M,
) {
    let dir = tempfile::tempdir().unwrap();
    let log = MemoryBackend::new();
    let mut model: Vec<u64> = (0..ENTRIES).map(|i| u64::from(i) + 1).collect();

    let mut rig = Rig::<C>::open(&log, open(dir.path()));
    assert_eq!(rig.read_all(), model);
    let first = rig.memo.checkpoint().unwrap();
    assert!(!first.delta && !first.compacted, "{first:?}");

    let (mut deltas, mut compactions, mut depth) = (0, 0, 0);
    for round in 1..=(2 * MAX_CHAIN_DEPTH + 3) {
        let changed = (round * 37) % ENTRIES as usize;
        model[changed] = 10_000 + round as u64 + 1;
        rig.store
            .set(rig.inputs[changed], 10_000 + round as u64)
            .unwrap();
        assert_eq!(rig.read_all(), model);
        let done = rig.memo.checkpoint().unwrap();
        if depth < MAX_CHAIN_DEPTH {
            assert!(done.delta, "round {round}: {done:?}");
            assert_eq!(done.fresh_written, 1);
            depth += 1;
            deltas += 1;
        } else {
            assert!(!done.delta && done.compacted, "round {round}: {done:?}");
            assert_eq!(done.written, ENTRIES as usize);
            depth = 0;
            compactions += 1;
        }
        assert_eq!(done.chain_depth, depth);
        assert_eq!(rig.memo.stats().chain_depth, depth as u64);

        drop(rig);
        rig = Rig::<C>::open(&log, open(dir.path()));
        rig.assert_restored_whole(&model);
    }
    assert_eq!(compactions, 2);
    assert_eq!(deltas, 2 * MAX_CHAIN_DEPTH + 1);
    drop(rig);

    let full_dir = tempfile::tempdir().unwrap();
    let full_log = MemoryBackend::new();
    {
        let full = Rig::<C>::open(&full_log, open(full_dir.path()));
        for (input, want) in full.inputs.iter().zip(&model) {
            full.store.set(*input, want - 1).unwrap();
        }
        assert_eq!(full.read_all(), model);
        assert!(!full.memo.checkpoint().unwrap().delta);
    }
    let full = Rig::<C>::open(&full_log, open(full_dir.path()));
    full.assert_restored_whole(&model);
}
