//! A memo's encodes and decodes see the store's value scope on every
//! thread: the automatic-checkpoint thread, and a caller that reads an
//! entry outside any store call.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use incr_persist::memo::{BuildId, Memo, MemoBuilder, MemoOptions, MemoTable, MemoryMemo};
use incr_persist::{MemoryBackend, Options, Shared, Store, ValueScope};

struct ScopeMarker;

static SCOPED: AtomicUsize = AtomicUsize::new(0);
static UNSCOPED: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Probe(u32);

impl Probe {
    fn note() {
        let scoped = incr_persist::current_value_scope()
            .and_then(|scope| scope.downcast::<ScopeMarker>())
            .is_some();
        let counter = if scoped { &SCOPED } else { &UNSCOPED };
        counter.fetch_add(1, Ordering::SeqCst);
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

type World = (Memo<Shared, MemoryBackend>, MemoTable<Probe, Probe>);

fn open(store: &Arc<Store<Shared, MemoryBackend>>, backend: MemoryMemo) -> World {
    let options = MemoOptions::new(BuildId::Fixed(0x5c09e)).checkpoint_after_entries(1);
    let mut builder = MemoBuilder::new(options);
    let table = builder.table::<Probe, Probe>("t", 1);
    builder.define(table, |_, key: &Probe| Probe(key.0 + 1));
    (builder.open(store, backend).unwrap(), table)
}

#[test]
fn memo_value_code_runs_in_the_store_scope_on_every_thread() {
    let options = Options {
        value_scope: Some(ValueScope::new(Arc::new(ScopeMarker))),
        ..Options::default()
    };
    let store = Arc::new(Store::<Shared, _>::open(MemoryBackend::new(), options).unwrap());
    let backend = MemoryMemo::new();
    let (memo, table) = open(&store, backend.clone());
    for k in 0..4 {
        let node = memo.get_or_insert(table, Probe(k));
        assert_eq!(memo.runtime().get(node), Probe(k + 1));
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    while memo.stats().checkpoints == 0 {
        assert!(Instant::now() < deadline, "no automatic checkpoint ran");
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(memo);
    let restored = Arc::new(
        Store::<Shared, _>::open(
            MemoryBackend::new(),
            Options {
                value_scope: Some(ValueScope::new(Arc::new(ScopeMarker))),
                ..Options::default()
            },
        )
        .unwrap(),
    );
    let (memo, table) = open(&restored, backend);
    let node = memo.get_or_insert(table, Probe(2));
    assert_eq!(memo.runtime().get(node), Probe(3));
    assert!(SCOPED.load(Ordering::SeqCst) > 0);
    assert_eq!(UNSCOPED.load(Ordering::SeqCst), 0);
}
