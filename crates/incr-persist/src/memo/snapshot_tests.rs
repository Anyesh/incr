use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use super::{CheckpointPark, CHECKPOINT_PARK, KEYS_TOUCHED};
use crate::memo::{BuildId, Memo, MemoBuilder, MemoOptions, MemoTable, MemoryMemo};
use crate::{Backend, MemoryBackend, Options, PersistedKeyed, Position, Shared, Store};

/// A park point that is never reached would otherwise hang the test.
const PARK_TIMEOUT: Duration = Duration::from_secs(10);

/// `top(1) = leaf(1) + y` over 200 stored leaves `leaf(k) = k`. After `y`
/// moves only `top(1)` is dirty, with one clean dep: the checkpoint reads
/// those two keys, not every live entry's.
#[test]
fn a_checkpoint_touches_only_the_keys_of_dirty_entries_and_their_clean_deps() {
    let store =
        Arc::new(Store::<Shared, _>::open(MemoryBackend::new(), Options::default()).unwrap());
    let y = store.input("y", 10_u64).unwrap();
    let yh = y.handle();
    let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(0x5a)));
    let leaf = builder.table::<u32, u64>("leaf", 1);
    let top = builder.table::<u32, u64>("top", 1);
    builder.define(leaf, |_, &k| u64::from(k));
    builder.define(top, move |ctx, &k| ctx.get(leaf, k) + ctx.rt().get(yh));
    let memo = builder.open(&store, MemoryMemo::new()).unwrap();
    let rt = memo.runtime();
    let q = memo.get_or_insert(top, 1);
    assert_eq!(rt.get(q), 11);
    for k in 2..=200 {
        rt.get(memo.get_or_insert(leaf, k));
    }
    assert_eq!(memo.checkpoint().unwrap().written, 201);
    store.set(y, 20).unwrap();
    assert_eq!(rt.get(q), 21);
    KEYS_TOUCHED.with(|n| n.set(0));
    assert_eq!(memo.checkpoint().unwrap().written, 201);
    assert_eq!(KEYS_TOUCHED.with(std::cell::Cell::get), 2);
}

type Rows = PersistedKeyed<u32, String, Shared>;

/// `price(k)` reads row `k` of the keyed collection "k" on `Shared`.
struct Priced {
    store: Arc<Store<Shared, MemoryBackend>>,
    rows: Rows,
    memo: Memo<Shared, MemoryBackend>,
    price: MemoTable<u32, Option<String>>,
    runs: Arc<AtomicUsize>,
}

impl Priced {
    fn open(log: &MemoryBackend, memo: &MemoryMemo) -> Self {
        let store = Arc::new(Store::open(log.clone(), Options::default()).unwrap());
        let rows: Rows = store.keyed_collection("k").unwrap();
        let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(0x5b)));
        let price = builder.table::<u32, Option<String>>("price", 1);
        let runs = Arc::new(AtomicUsize::new(0));
        let (counted, read) = (Arc::clone(&runs), rows.clone());
        builder.define(price, move |ctx, k| {
            counted.fetch_add(1, SeqCst);
            read.get_key(ctx.rt(), k)
        });
        let memo = builder.open(&store, memo.clone()).unwrap();
        Priced {
            store,
            rows,
            memo,
            price,
            runs,
        }
    }

    fn read(&self, k: u32) -> Option<String> {
        self.memo
            .runtime()
            .get(self.memo.get_or_insert(self.price, k))
    }
}

fn copy_of(log: &MemoryBackend) -> MemoryBackend {
    let mut read = Vec::new();
    log.read_from(Position(0), &mut read).unwrap();
    let copy = MemoryBackend::new();
    let records: Vec<_> = read.into_iter().map(|(_, record)| record).collect();
    copy.append(&records).unwrap();
    copy
}

/// Checkpoints `price(1)` over row "a", parks the checkpoint at `point`
/// and upserts row 1 to "x" from another thread. Inside the `DepIndex`
/// hold the upsert must wait for the checkpoint; after it, it lands. The
/// stored entry must pair "a" with the fingerprint of row "a": a reopen
/// over the current log reruns it, and one over the log as it was before
/// the upsert reuses it.
fn a_keyed_write_racing_a_checkpoint_never_stores_a_newer_row_fp(point: CheckpointPark) {
    let (log, memo) = (MemoryBackend::new(), MemoryMemo::new());
    let before = {
        let p = Priced::open(&log, &memo);
        p.store.upsert(&p.rows, 1, "a".to_owned()).unwrap();
        assert_eq!(p.read(1).as_deref(), Some("a"));
        let before = copy_of(&log);
        let (parked_tx, parked_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let p = &p;
        thread::scope(|s| {
            let checkpointer = s.spawn(move || {
                let armed = Cell::new(true);
                CHECKPOINT_PARK.set(Some(Box::new(move |at| {
                    if at == point && armed.replace(false) {
                        parked_tx.send(()).unwrap();
                        release_rx
                            .recv_timeout(PARK_TIMEOUT)
                            .expect("the test releases the checkpoint");
                    }
                })));
                let stats = p.memo.checkpoint();
                CHECKPOINT_PARK.set(None);
                stats
            });
            parked_rx
                .recv_timeout(PARK_TIMEOUT)
                .expect("the checkpoint reaches its park point");
            let writer = s.spawn(|| p.store.upsert(&p.rows, 1, "x".to_owned()));
            if point == CheckpointPark::Inspected {
                thread::sleep(Duration::from_millis(100));
                assert!(
                    !writer.is_finished(),
                    "the upsert lands inside the checkpoint's DepIndex hold"
                );
                release_tx.send(()).unwrap();
                writer.join().unwrap().unwrap();
            } else {
                writer.join().unwrap().unwrap();
                release_tx.send(()).unwrap();
            }
            assert_eq!(checkpointer.join().unwrap().unwrap().written, 1);
        });
        assert_eq!(p.read(1).as_deref(), Some("x"));
        before
    };
    let live = Priced::open(&log, &memo);
    assert_eq!(live.read(1).as_deref(), Some("x"));
    assert_eq!(
        (live.memo.stats().reused, live.runs.load(SeqCst)),
        (0, 1),
        "the entry verified against the newer row"
    );
    drop(live);
    let old = Priced::open(&before, &memo);
    assert_eq!(old.read(1).as_deref(), Some("a"));
    assert_eq!(
        (old.memo.stats().reused, old.runs.load(SeqCst)),
        (1, 0),
        "the entry did not verify against the row it was computed from"
    );
}

#[test]
fn a_keyed_write_parked_against_an_inspected_entry_never_stores_a_newer_row_fp() {
    a_keyed_write_racing_a_checkpoint_never_stores_a_newer_row_fp(CheckpointPark::Inspected);
}

#[test]
fn a_keyed_write_landing_before_the_encode_never_stores_a_newer_row_fp() {
    a_keyed_write_racing_a_checkpoint_never_stores_a_newer_row_fp(CheckpointPark::Snapshotted);
}
