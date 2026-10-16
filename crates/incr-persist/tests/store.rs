use std::io;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use incr_persist::{
    Backend, Cells, Durability, Error, FileBackend, Kind, Local, MemoryBackend, Options, Position,
    ReadOutcome, Record, Shared, Store,
};

fn open<C: Cells, B: Backend>(backend: B) -> Store<C, B> {
    Store::open(backend, Options::default()).unwrap()
}

fn log_records(backend: &impl Backend) -> Vec<Record> {
    let mut out = Vec::new();
    assert_eq!(
        backend.read_from(Position::ZERO, &mut out).unwrap(),
        ReadOutcome::Records
    );
    out.into_iter().map(|(_, record)| record).collect()
}

fn restart_reproduces_values_and_queries<C: Cells, B: Backend>(backend: impl Fn() -> B) {
    let expected = {
        let store = open::<C, B>(backend());
        let price = store.input("price", 0.0f64).unwrap();
        let qty = store.input("qty", 1u32).unwrap();
        let label = store.input("label", String::new()).unwrap();
        store.set(price, 10.5).unwrap();
        store.set(qty, 3).unwrap();
        store.set(label, "spring".to_string()).unwrap();
        store.set(price, 2.25).unwrap();
        let (p, q) = (price.handle(), qty.handle());
        let total = store
            .runtime()
            .create_query(move |rt| rt.get(p) * f64::from(rt.get(q)));
        assert_eq!(store.runtime().get(total), 6.75);
        store.runtime().get(total)
    };
    let store = open::<C, B>(backend());
    let price = store.input("price", 99.0f64).unwrap();
    let qty = store.input("qty", 99u32).unwrap();
    let label = store.input("label", "other".to_string()).unwrap();
    let rt = store.runtime();
    assert_eq!(rt.get(price.handle()), 2.25);
    assert_eq!(rt.get(qty.handle()), 3);
    assert_eq!(rt.get(label.handle()), "spring");
    let (p, q) = (price.handle(), qty.handle());
    let total = rt.create_query(move |rt| rt.get(p) * f64::from(rt.get(q)));
    assert_eq!(rt.get(total), expected);

    store.set(qty, 4).unwrap();
    assert_eq!(rt.get(total), 9.0);
}

#[test]
fn restart_reproduces_values_and_queries_on_memory() {
    let log = MemoryBackend::new();
    restart_reproduces_values_and_queries::<Local, _>(|| log.clone());
    let log = MemoryBackend::new();
    restart_reproduces_values_and_queries::<Shared, _>(|| log.clone());
}

#[test]
fn restart_reproduces_values_and_queries_on_file() {
    let dir = tempfile::tempdir().unwrap();
    restart_reproduces_values_and_queries::<Local, _>(|| FileBackend::open(dir.path()).unwrap());
    let dir = tempfile::tempdir().unwrap();
    restart_reproduces_values_and_queries::<Shared, _>(|| FileBackend::open(dir.path()).unwrap());
}

#[test]
fn restart_replays_from_a_snapshot_and_the_records_after_it() {
    let log = MemoryBackend::new();
    {
        let store = open::<Local, _>(log.clone());
        let a = store.input("a", 0u32).unwrap();
        store.set(a, 1).unwrap();
        store.set(a, 2).unwrap();
    }
    let records = log_records(&log);
    let tail = log.append(&[]).unwrap();
    let folded = vec![records[0].clone(), records[2].clone()];
    log.compact(tail, folded).unwrap();
    {
        let store = open::<Local, _>(log.clone());
        let a = store.input("a", 0u32).unwrap();
        assert_eq!(store.runtime().get(a.handle()), 2);
        store.set(a, 3).unwrap();
    }
    let store = open::<Local, _>(log);
    let a = store.input("a", 0u32).unwrap();
    assert_eq!(store.runtime().get(a.handle()), 3);
}

#[test]
fn a_replayed_delete_resets_the_key_to_its_logged_default() {
    let log = MemoryBackend::new();
    {
        let store = open::<Local, _>(log.clone());
        let k = store.input("k", 3u32).unwrap();
        store.set(k, 9).unwrap();
    }
    log.append(&[Record::Delete { key: "k".into() }]).unwrap();
    let store = open::<Local, _>(log);
    let k = store.input("k", 100u32).unwrap();
    assert_eq!(store.runtime().get(k.handle()), 3);
}

#[test]
fn first_declare_wins_across_stores_and_later_stores_adopt_it() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let b = open::<Local, _>(log.clone());

    let limit_a = a.input("limit", 10u32).unwrap();
    let limit_b = b.input("limit", 20u32).unwrap();
    assert_eq!(a.runtime().get(limit_a.handle()), 10);
    assert_eq!(
        b.runtime().get(limit_b.handle()),
        10,
        "the earlier Declare in log order must win over this store's own"
    );
    let declares = log_records(&log)
        .iter()
        .filter(|r| matches!(r, Record::Declare { .. }))
        .count();
    assert_eq!(declares, 2);

    let c = open::<Local, _>(log.clone());
    let limit_c = c.input("limit", 30u32).unwrap();
    assert_eq!(c.runtime().get(limit_c.handle()), 10);
    assert_eq!(
        log_records(&log).len(),
        2,
        "a key with a live Declare must not be declared again"
    );
}

#[test]
fn a_write_applies_foreign_records_before_its_own() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let b = open::<Local, _>(log.clone());
    let x_a = a.input("x", 0u32).unwrap();
    let x_b = b.input("x", 0u32).unwrap();
    a.set(x_a, 5).unwrap();
    assert_eq!(b.runtime().get(x_b.handle()), 0);

    let y_b = b.input("y", 0u32).unwrap();
    b.set(y_b, 1).unwrap();
    assert_eq!(b.runtime().get(x_b.handle()), 5);

    b.set(x_b, 6).unwrap();
    a.set(x_a, 7).unwrap();
    assert_eq!(a.runtime().get(x_a.handle()), 7);
}

#[test]
fn a_type_mismatch_is_an_error_not_a_panic() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let n = a.input("n", 5u64).unwrap();
    a.set(n, 7).unwrap();
    assert!(matches!(
        a.input("n", String::new()),
        Err(Error::TypeMismatch { key, .. }) if key == "n"
    ));
    assert_eq!(a.input("n", 0u64).unwrap(), n);

    let b = open::<Local, _>(log);
    assert!(matches!(
        b.input("n", String::from("x")),
        Err(Error::TypeMismatch { key, .. }) if key == "n"
    ));
    let n = b.input("n", 0u64).unwrap();
    assert_eq!(b.runtime().get(n.handle()), 7);
}

#[test]
fn an_input_declared_as_a_collection_is_a_kind_mismatch() {
    let log = MemoryBackend::new();
    log.append(&[Record::Declare {
        key: "visits".into(),
        kind: Kind::Collection,
        default: vec![],
    }])
    .unwrap();
    let store = open::<Local, _>(log);
    assert!(matches!(
        store.input("visits", 0u32),
        Err(Error::KindMismatch { key, declared: Kind::Collection }) if key == "visits"
    ));
}

#[test]
fn a_handle_from_another_store_is_rejected() {
    let a = open::<Local, _>(MemoryBackend::new());
    let b = open::<Local, _>(MemoryBackend::new());
    let x = a.input("x", 0u32).unwrap();
    b.input("x", 0u32).unwrap();
    assert!(matches!(b.set(x, 1), Err(Error::UnknownHandle)));
}

#[test]
fn restored_keys_lists_undeclared_keys_by_prefix() {
    let log = MemoryBackend::new();
    {
        let store = open::<Local, _>(log.clone());
        for key in ["visit/2", "visit/1", "price", "visitor"] {
            store.input(key, 0u32).unwrap();
        }
    }
    let store = open::<Local, _>(log);
    assert_eq!(
        store.restored_keys("visit/").unwrap(),
        ["visit/1", "visit/2"]
    );
    assert_eq!(
        store.restored_keys("visit").unwrap(),
        ["visit/1", "visit/2", "visitor"]
    );
    store.input("visit/1", 0u32).unwrap();
    assert_eq!(store.restored_keys("visit/").unwrap(), ["visit/2"]);
    assert_eq!(
        store.restored_keys("").unwrap(),
        ["price", "visit/2", "visitor"]
    );
}

#[derive(Clone, Default)]
struct Faulty {
    inner: MemoryBackend,
    fail_append: Arc<AtomicBool>,
    fail_sync: Arc<AtomicBool>,
    fail_read: Arc<AtomicBool>,
    appends: Arc<AtomicUsize>,
    syncs: Arc<AtomicUsize>,
}

fn injected(flag: &AtomicBool) -> Result<(), Error> {
    if flag.load(Ordering::SeqCst) {
        Err(Error::Io(io::Error::other("injected failure")))
    } else {
        Ok(())
    }
}

impl Backend for Faulty {
    fn append(&self, records: &[Record]) -> incr_persist::Result<Position> {
        injected(&self.fail_append)?;
        self.appends.fetch_add(1, Ordering::SeqCst);
        self.inner.append(records)
    }

    fn sync(&self, upto: Position) -> incr_persist::Result<()> {
        injected(&self.fail_sync)?;
        self.syncs.fetch_add(1, Ordering::SeqCst);
        self.inner.sync(upto)
    }

    fn read_from(
        &self,
        after: Position,
        out: &mut Vec<(Position, Record)>,
    ) -> incr_persist::Result<ReadOutcome> {
        injected(&self.fail_read)?;
        self.inner.read_from(after, out)
    }

    fn load_snapshot(&self) -> incr_persist::Result<Option<(Position, Vec<Record>)>> {
        self.inner.load_snapshot()
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> incr_persist::Result<()> {
        self.inner.compact(upto, folded)
    }

    fn writable(&self) -> bool {
        self.inner.writable()
    }
}

fn assert_poisoned<T: std::fmt::Debug>(result: incr_persist::Result<T>) {
    assert!(
        matches!(result, Err(Error::Poisoned)),
        "expected Poisoned, got {result:?}"
    );
}

#[test]
fn a_failure_after_append_poisons_the_store_until_reopened() {
    for fail_sync in [true, false] {
        let backend = Faulty::default();
        let store = open::<Local, _>(backend.clone());
        let x = store.input("x", 0u32).unwrap();
        let flag = if fail_sync {
            &backend.fail_sync
        } else {
            &backend.fail_read
        };
        flag.store(true, Ordering::SeqCst);
        assert!(matches!(store.set(x, 5), Err(Error::Io(_))));
        flag.store(false, Ordering::SeqCst);

        assert_poisoned(store.set(x, 6));
        assert_poisoned(store.input("y", 0u32));
        assert_poisoned(store.restored_keys(""));
        assert_poisoned(store.batch(|b| {
            b.set(x, 7);
        }));
        drop(store);

        let store = open::<Local, _>(backend);
        let x = store.input("x", 0u32).unwrap();
        assert_eq!(
            store.runtime().get(x.handle()),
            5,
            "the appended record is in the log and replays"
        );
        store.set(x, 8).unwrap();
        assert_eq!(store.runtime().get(x.handle()), 8);
    }
}

#[test]
fn a_failed_append_does_not_poison() {
    let backend = Faulty::default();
    let store = open::<Local, _>(backend.clone());
    let x = store.input("x", 0u32).unwrap();
    backend.fail_append.store(true, Ordering::SeqCst);
    assert!(matches!(store.set(x, 5), Err(Error::Io(_))));
    backend.fail_append.store(false, Ordering::SeqCst);
    store.set(x, 6).unwrap();
    assert_eq!(store.runtime().get(x.handle()), 6);
}

#[test]
fn durability_decides_whether_writes_sync() {
    let backend = Faulty::default();
    let store = open::<Local, _>(backend.clone());
    let x = store.input("x", 0u32).unwrap();
    let syncs = backend.syncs.load(Ordering::SeqCst);
    store.set(x, 1).unwrap();
    assert_eq!(backend.syncs.load(Ordering::SeqCst), syncs + 1);

    let buffered = Store::<Local, _>::open(
        backend.clone(),
        Options {
            durability: Durability::Buffered,
            ..Options::default()
        },
    )
    .unwrap();
    let x = buffered.input("x", 0u32).unwrap();
    let syncs = backend.syncs.load(Ordering::SeqCst);
    buffered.set(x, 2).unwrap();
    assert_eq!(backend.syncs.load(Ordering::SeqCst), syncs);
    assert_eq!(buffered.runtime().get(x.handle()), 2);
}

#[test]
fn a_batch_is_one_append_and_one_sync() {
    let backend = Faulty::default();
    let store = open::<Local, _>(backend.clone());
    let x = store.input("x", 0u32).unwrap();
    let y = store.input("y", String::new()).unwrap();
    let (appends, syncs) = (
        backend.appends.load(Ordering::SeqCst),
        backend.syncs.load(Ordering::SeqCst),
    );
    let last = store
        .batch(|b| {
            b.set(x, 1).set(y, "one".to_string()).set(x, 2);
        })
        .unwrap();
    assert_eq!(backend.appends.load(Ordering::SeqCst), appends + 1);
    assert_eq!(backend.syncs.load(Ordering::SeqCst), syncs + 1);
    assert_eq!(store.runtime().get(x.handle()), 2);
    assert_eq!(store.runtime().get(y.handle()), "one");
    assert_eq!(log_records(&backend).len() as u64, last.0);

    let before = backend.appends.load(Ordering::SeqCst);
    store.batch(|_| {}).unwrap();
    assert_eq!(backend.appends.load(Ordering::SeqCst), before);
}

#[test]
fn a_batch_with_a_foreign_handle_writes_nothing() {
    let other = open::<Local, _>(MemoryBackend::new());
    let foreign = other.input("x", 0u32).unwrap();
    let backend = Faulty::default();
    let store = open::<Local, _>(backend.clone());
    let x = store.input("x", 0u32).unwrap();
    let appends = backend.appends.load(Ordering::SeqCst);
    let result = store.batch(|b| {
        b.set(x, 1).set(foreign, 2);
    });
    assert!(matches!(result, Err(Error::UnknownHandle)));
    assert_eq!(backend.appends.load(Ordering::SeqCst), appends);
    assert_eq!(store.runtime().get(x.handle()), 0);
}

#[test]
fn a_shared_store_is_usable_across_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Store<Shared, MemoryBackend>>();
    assert_send_sync::<Store<Shared, FileBackend>>();

    let log = MemoryBackend::new();
    let store = Arc::new(open::<Shared, _>(log.clone()));
    let inputs: Vec<_> = (0..4)
        .map(|i| store.input(&format!("k{i}"), 0u64).unwrap())
        .collect();
    let sum = {
        let handles: Vec<_> = inputs.iter().map(|i| i.handle()).collect();
        store
            .runtime()
            .create_query(move |rt| handles.iter().map(|&h| rt.get(h)).sum::<u64>())
    };
    thread::scope(|s| {
        for &input in &inputs {
            let store = Arc::clone(&store);
            s.spawn(move || {
                for value in 1..=50 {
                    store.set(input, value).unwrap();
                }
            });
        }
    });
    assert_eq!(store.runtime().get(sum), 200);

    let reopened = open::<Shared, _>(log);
    for i in 0..4 {
        let k = reopened.input(&format!("k{i}"), 0u64).unwrap();
        assert_eq!(reopened.runtime().get(k.handle()), 50);
    }
}
