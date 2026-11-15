//! A memo entry reading a keyed collection's key set depends on its
//! membership input, which a checkpoint cannot store: the entry and its
//! dependents are unsaveable and recompute after each restore.

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

use incr_persist::memo::{BuildId, Memo, MemoBuilder, MemoOptions, MemoTable, MemoryMemo};
use incr_persist::{MemoryBackend, Options, PersistedKeyed, Shared, Store};

struct World {
    store: Arc<Store<Shared, MemoryBackend>>,
    rows: PersistedKeyed<u32, String, Shared>,
    memo: Memo<Shared, MemoryBackend>,
    names: MemoTable<u32, Vec<u32>>,
    count: MemoTable<u32, usize>,
    row: MemoTable<u32, Option<String>>,
    name_runs: Arc<AtomicUsize>,
}

impl World {
    fn open(log: &MemoryBackend, memo: &MemoryMemo) -> Self {
        let store = Arc::new(Store::open(log.clone(), Options::default()).unwrap());
        let rows: PersistedKeyed<u32, String, Shared> = store.keyed_collection("rows").unwrap();
        let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(0x15)));
        let names = builder.table::<u32, Vec<u32>>("names", 1);
        let count = builder.table::<u32, usize>("count", 1);
        let row = builder.table::<u32, Option<String>>("row", 1);
        let name_runs = Arc::new(AtomicUsize::new(0));
        let (runs, read) = (Arc::clone(&name_runs), rows.clone());
        builder.define(names, move |ctx, _| {
            runs.fetch_add(1, SeqCst);
            let mut keys = read.get_keys(ctx.rt());
            keys.sort_unstable();
            keys
        });
        builder.define(count, move |ctx, _| ctx.get(names, 0).len());
        let read = rows.clone();
        builder.define(row, move |ctx, k| read.get_key(ctx.rt(), k));
        let memo = builder.open(&store, memo.clone()).unwrap();
        World {
            store,
            rows,
            memo,
            names,
            count,
            row,
            name_runs,
        }
    }

    fn names(&self) -> Vec<u32> {
        let node = self.memo.get_or_insert(self.names, 0);
        self.memo.runtime().get(node)
    }

    fn count(&self) -> usize {
        let node = self.memo.get_or_insert(self.count, 0);
        self.memo.runtime().get(node)
    }

    fn row(&self, k: u32) -> Option<String> {
        let node = self.memo.get_or_insert(self.row, k);
        self.memo.runtime().get(node)
    }

    fn upsert(&self, k: u32) {
        self.store.upsert(&self.rows, k, format!("r{k}")).unwrap();
    }
}

#[test]
fn a_memo_entry_reading_the_key_set_is_unsaveable_with_its_dependents() {
    let (log, memo) = (MemoryBackend::new(), MemoryMemo::new());
    let w = World::open(&log, &memo);
    w.upsert(1);
    w.upsert(2);
    assert_eq!(w.names(), vec![1, 2]);
    assert_eq!(w.count(), 2);
    assert_eq!(w.row(1).as_deref(), Some("r1"));
    assert_eq!(w.memo.checkpoint().unwrap().written, 1);
    assert_eq!(w.memo.stats().unsaveable, 2);
}

#[test]
fn a_key_set_entry_recomputes_after_a_restore_and_follows_inserts() {
    let (log, memo) = (MemoryBackend::new(), MemoryMemo::new());
    {
        let w = World::open(&log, &memo);
        w.upsert(1);
        assert_eq!(w.names(), vec![1]);
        assert_eq!(w.count(), 1);
        w.memo.checkpoint().unwrap();
    }
    let w = World::open(&log, &memo);
    w.upsert(2);
    assert_eq!(w.names(), vec![1, 2]);
    assert_eq!(w.name_runs.load(SeqCst), 1);
    assert_eq!(w.count(), 2);
    w.upsert(3);
    assert_eq!(w.names(), vec![1, 2, 3]);
    assert_eq!(w.count(), 3);
    assert_eq!(w.name_runs.load(SeqCst), 2);
    w.upsert(3);
    assert_eq!(w.names(), vec![1, 2, 3]);
    assert_eq!(w.name_runs.load(SeqCst), 2);
}
