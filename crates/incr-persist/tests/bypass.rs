//! Debug builds catch an input written through the runtime instead of the
//! store, which the log never sees and a restart would lose.
#![cfg(debug_assertions)]

use std::collections::HashMap;

use incr_persist::{Backend, Cells, Error, Local, MemoryBackend, Options, Position, Store};

fn open<C: Cells, B: Backend>(backend: B) -> Store<C, B> {
    Store::open(backend, Options::default()).unwrap()
}

fn assert_bypassed<T: std::fmt::Debug>(result: incr_persist::Result<T>, expected: &str) {
    assert!(
        matches!(&result, Err(Error::Bypassed { key }) if key == expected),
        "expected Bypassed for {expected:?}, got {result:?}"
    );
}

#[test]
fn set_and_checkpoint_reject_a_runtime_write_the_log_never_saw() {
    let log = MemoryBackend::new();
    let store = open::<Local, _>(log.clone());
    let x = store.input("x", 1u32).unwrap();
    let y = store.input("y", 0u32).unwrap();
    store.set(x, 2).unwrap();
    let tail = log.append(&[]).unwrap();

    store.runtime().set(x.handle(), 9);
    assert_bypassed(store.set(x, 3), "x");
    assert_bypassed(
        store.batch(|b| {
            b.set(x, 3);
        }),
        "x",
    );
    assert_bypassed(store.checkpoint(), "x");
    assert_eq!(log.append(&[]).unwrap(), tail, "nothing was appended");
    assert!(
        log.load_snapshot().unwrap().is_none(),
        "nothing was compacted"
    );
    store.set(y, 4).unwrap();

    store.runtime().set(x.handle(), 2);
    store.set(x, 3).unwrap();
    store.checkpoint().unwrap();
}

#[test]
fn values_the_log_holds_are_never_reported() {
    let log = MemoryBackend::new();
    let store = open::<Local, _>(log.clone());
    let nan = store.input("nan", 0.0f64).unwrap();
    store.set(nan, f64::NAN).unwrap();
    store.set(nan, 1.0).unwrap();

    let map = store.input("map", HashMap::<u32, u32>::new()).unwrap();
    store.set(map, (0..64).map(|i| (i, i)).collect()).unwrap();
    store.set(map, HashMap::from([(1, 1)])).unwrap();

    let deleted = store.input("deleted", 5u32).unwrap();
    store.set(deleted, 6).unwrap();
    store.delete(deleted).unwrap();
    store.checkpoint().unwrap();
    store.set(deleted, 7).unwrap();
    store.checkpoint().unwrap();
}

#[test]
fn a_key_forgotten_and_declared_afresh_elsewhere_is_not_reported() {
    let log = MemoryBackend::new();
    let a = open::<Local, _>(log.clone());
    let k = a.input("k", 3u32).unwrap();
    a.checkpoint().unwrap();
    assert!(a.restored_keys("").unwrap().is_empty());

    let b = open::<Local, _>(log.clone());
    b.input("k", 7u32).unwrap();
    a.refresh().unwrap();
    assert_eq!(a.runtime().get(k.handle()), 7, "the fresh Declare wins");
    a.checkpoint().unwrap();
    assert_eq!(a.set(k, 8).unwrap(), log.append(&[]).unwrap());
    a.checkpoint().unwrap();
    a.delete(k).unwrap();
    assert_eq!(a.runtime().get(k.handle()), 7);
    let at: Position = a.checkpoint().unwrap();
    assert!(at > Position::ZERO);
    a.set(k, 9).unwrap();
}
