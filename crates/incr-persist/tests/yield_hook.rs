//! A store's restore, catch-up and checkpoint call its runtime's yield
//! hook every `YIELD_EVERY` records or rows, on a `Local` store.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use incr_core::{Local, YieldHook, YIELD_EVERY};
use incr_persist::{MemoryBackend, Options, Store};

const N: u64 = 8 * YIELD_EVERY as u64;
const BATCHES: u64 = N / YIELD_EVERY as u64;

fn counted(calls: &Arc<AtomicU64>) -> Options {
    let seen = Arc::clone(calls);
    Options {
        yield_hook: Some(YieldHook::new(move || {
            seen.fetch_add(1, Ordering::Relaxed);
        })),
        ..Options::default()
    }
}

fn calls_during<R>(calls: &AtomicU64, f: impl FnOnce() -> R) -> (u64, R) {
    let before = calls.load(Ordering::Relaxed);
    let result = f();
    (calls.load(Ordering::Relaxed) - before, result)
}

type Rows = incr_persist::PersistedKeyed<u64, u64, Local>;

#[test]
fn restore_catch_up_and_checkpoint_call_the_hook() {
    let calls = Arc::new(AtomicU64::new(0));
    let log = MemoryBackend::new();
    let store = Store::<Local, _>::open(log.clone(), counted(&calls)).unwrap();
    let rows: Rows = store.keyed_collection("rows").unwrap();
    store.reconcile(&rows, (0..N).map(|k| (k, k))).unwrap();
    let (checkpoint, _) = calls_during(&calls, || store.checkpoint().unwrap());
    assert!(checkpoint >= BATCHES, "{checkpoint} calls folding {N} rows");
    store.reconcile(&rows, (0..N).map(|k| (k, k + 1))).unwrap();
    drop(rows);
    drop(store);

    let (opened, store) = calls_during(&calls, || {
        Store::<Local, _>::open(log.clone(), counted(&calls)).unwrap()
    });
    assert!(opened >= BATCHES, "{opened} calls catching up {N} records");
    let (declared, rows) = calls_during(&calls, || -> Rows {
        store.keyed_collection("rows").unwrap()
    });
    assert!(declared >= BATCHES, "{declared} calls restoring {N} rows");
    assert_eq!(rows.row(&7), Some(8));
}

#[test]
fn a_store_without_a_hook_runs_the_same() {
    let log = MemoryBackend::new();
    let store = Store::<Local, _>::open(log.clone(), Options::default()).unwrap();
    let rows: Rows = store.keyed_collection("rows").unwrap();
    store.reconcile(&rows, (0..N).map(|k| (k, k))).unwrap();
    store.checkpoint().unwrap();
    drop(rows);
    drop(store);
    let store = Store::<Local, _>::open(log, Options::default()).unwrap();
    let rows: Rows = store.keyed_collection("rows").unwrap();
    assert_eq!(rows.len(), N as usize);
}
