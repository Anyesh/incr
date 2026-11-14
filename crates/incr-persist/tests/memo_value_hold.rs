//! An automatic checkpoint runs its value code in batches inside the
//! store's value-scope hold, a few holds per checkpoint rather than one
//! per value, so a binding whose value code takes a lock per call (the
//! GIL) takes it a few times instead of several times per entry.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use incr_persist::memo::{BuildId, MemoBuilder, MemoOptions, MemoryMemo};
use incr_persist::{MemoryBackend, Options, PersistedKeyed, Shared, Store, ValueScope};

/// The memo module's batch size: each batch of entries takes about five
/// holds (one to encode, one to inspect, three to map its row deps).
const BATCH: usize = 256;

struct ScopeMarker;

static HOLDS: AtomicUsize = AtomicUsize::new(0);
static HELD: AtomicUsize = AtomicUsize::new(0);
static OUTSIDE: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    static CALLER: Cell<bool> = const { Cell::new(false) };
}

fn hold(f: &mut dyn FnMut()) {
    HOLDS.fetch_add(1, Ordering::SeqCst);
    DEPTH.with(|d| d.set(d.get() + 1));
    f();
    DEPTH.with(|d| d.set(d.get() - 1));
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct Probe(u32);

impl Probe {
    fn note() {
        if CALLER.with(Cell::get) {
            return;
        }
        let counter = if DEPTH.with(Cell::get) > 0 {
            &HELD
        } else {
            &OUTSIDE
        };
        counter.fetch_add(1, Ordering::SeqCst);
    }
}

impl Clone for Probe {
    fn clone(&self) -> Self {
        Probe::note();
        Probe(self.0)
    }
}

impl serde::Serialize for Probe {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        Probe::note();
        s.serialize_u32(self.0)
    }
}

impl<'de> serde::Deserialize<'de> for Probe {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Probe::note();
        u32::deserialize(d).map(Probe)
    }
}

/// `price(k)` reads row `k` of a keyed collection, so the checkpoint
/// clones and encodes each entry's key and value and maps its row dep,
/// whose key it clones, hashes and encodes.
#[test]
fn an_automatic_checkpoint_runs_its_value_code_in_a_few_holds() {
    CALLER.with(|c| c.set(true));
    let options = Options {
        value_scope: Some(ValueScope::new(Arc::new(ScopeMarker)).holding(hold)),
        ..Options::default()
    };
    let store = Arc::new(Store::<Shared, _>::open(MemoryBackend::new(), options).unwrap());
    let rows: PersistedKeyed<Probe, Probe, Shared> = store.keyed_collection("rows").unwrap();
    let entries = 1024_u32;
    store
        .reconcile(&rows, (0..entries).map(|k| (Probe(k), Probe(k * 2))))
        .unwrap();
    let options =
        MemoOptions::new(BuildId::Fixed(0x401d)).checkpoint_after_entries(u64::from(entries));
    let mut builder = MemoBuilder::new(options);
    let price = builder.table::<Probe, Option<Probe>>("price", 1);
    let read = rows.clone();
    builder.define(price, move |ctx, key: &Probe| read.get_key(ctx.rt(), key));
    let memo = builder.open(&store, MemoryMemo::new()).unwrap();
    // Store calls before this drop their spent values inside a hold too.
    let setup = HOLDS.load(Ordering::SeqCst);
    for k in 0..entries {
        let node = memo.get_or_insert(price, Probe(k));
        assert_eq!(memo.runtime().get(node), Some(Probe(k * 2)));
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    while memo.stats().checkpoints == 0 {
        assert!(Instant::now() < deadline, "no automatic checkpoint ran");
        std::thread::sleep(Duration::from_millis(10));
    }
    let checkpoints = memo.stats().checkpoints as usize;
    drop(memo);
    assert_eq!(OUTSIDE.load(Ordering::SeqCst), 0);
    assert!(HELD.load(Ordering::SeqCst) >= 6 * entries as usize);
    let holds = HOLDS.load(Ordering::SeqCst) - setup;
    assert!(
        holds <= checkpoints * (entries as usize).div_ceil(BATCH) * 6,
        "{holds} holds for {checkpoints} checkpoints of {entries} entries"
    );
}
