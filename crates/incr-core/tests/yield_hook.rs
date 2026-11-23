//! Where the engine calls a runtime's yield hook: at every compute, and
//! every `YIELD_EVERY` iterations of walks, marking, batch writes and
//! operator passes, under internal locks only on a `Local` runtime.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use incr_core::{
    with_ambient_hook, Cells, KeyedCollection, Local, Runtime, Shared, YieldHook, Yielder,
    YIELD_EVERY,
};

const N: u64 = 8 * YIELD_EVERY as u64;
const BATCHES: u64 = N / YIELD_EVERY as u64;

struct Counted<C: Cells> {
    rt: Runtime<C>,
    calls: Arc<AtomicU64>,
}

impl<C: Cells> Counted<C> {
    fn new() -> Self {
        let calls = Arc::new(AtomicU64::new(0));
        let seen = Arc::clone(&calls);
        let mut rt = Runtime::new();
        rt.set_yield_hook(Some(YieldHook::new(move || {
            seen.fetch_add(1, Ordering::Relaxed);
        })));
        Counted { rt, calls }
    }

    fn calls_during<R>(&self, f: impl FnOnce(&Runtime<C>) -> R) -> (u64, R) {
        let before = self.calls.load(Ordering::Relaxed);
        let result = f(&self.rt);
        (self.calls.load(Ordering::Relaxed) - before, result)
    }
}

#[test]
fn every_compute_calls_the_hook() {
    let c = Counted::<Local>::new();
    let a = c.rt.create_input(1_u64);
    let reads: Vec<_> = (0..100)
        .map(|_| c.rt.create_query(move |rt| rt.get(a)))
        .collect();
    let total =
        c.rt.create_query(move |rt| reads.iter().map(|&q| rt.get(q)).sum::<u64>());
    let (calls, value) = c.calls_during(|rt| rt.get(total));
    assert_eq!(value, 100);
    assert!(calls >= 101, "{calls} calls for 101 computes");
}

fn fan_out<C: Cells>(c: &Counted<C>) -> u64 {
    let a = c.rt.create_input(0_u64);
    let reads: Vec<_> = (0..N)
        .map(|_| c.rt.create_query(move |rt| rt.get(a)))
        .collect();
    for &read in &reads {
        c.rt.get(read);
    }
    c.calls_during(|rt| rt.set(a, 1)).0
}

#[test]
fn marking_a_local_fan_out_calls_the_hook_every_yield_every_nodes() {
    let calls = fan_out(&Counted::<Local>::new());
    assert!(calls >= BATCHES, "{calls} calls marking {N} nodes");
}

#[test]
fn marking_a_shared_fan_out_never_calls_the_hook() {
    assert_eq!(fan_out(&Counted::<Shared>::new()), 0);
}

#[test]
fn a_walk_verifying_many_nodes_without_computing_them_calls_the_hook() {
    let c = Counted::<Local>::new();
    let a = c.rt.create_input(0_u64);
    let cut = c.rt.create_query(move |rt| u64::from(rt.get(a) > 10));
    let reads: Vec<_> = (0..N)
        .map(|_| c.rt.create_query(move |rt| rt.get(cut)))
        .collect();
    let total =
        c.rt.create_query(move |rt| reads.iter().map(|&q| rt.get(q)).sum::<u64>());
    c.rt.get(total);
    c.rt.set(a, 1);
    let (calls, value) = c.calls_during(|rt| rt.get(total));
    assert_eq!(value, 0);
    assert!(calls >= BATCHES, "{calls} calls verifying {N} nodes");
}

#[test]
fn a_collection_extend_and_its_count_call_the_hook() {
    let c = Counted::<Local>::new();
    let col = c.rt.create_collection::<u64>();
    let count = col.count(&c.rt);
    c.rt.get(count);
    let (extend, _) = c.calls_during(|rt| col.extend(rt, 0..N));
    assert!(
        extend >= 2 * BATCHES,
        "{extend} calls staging and applying {N}"
    );
    let (counted, value) = c.calls_during(|rt| rt.get(count));
    assert_eq!(value, N);
    assert!(counted >= BATCHES, "{counted} calls counting {N} deltas");
}

#[test]
fn a_filter_rejecting_every_delta_calls_the_hook() {
    let c = Counted::<Local>::new();
    let col = c.rt.create_collection::<u64>();
    let none = col.filter(&c.rt, |_| false).count(&c.rt);
    c.rt.get(none);
    col.extend(&c.rt, 0..N);
    let (calls, value) = c.calls_during(|rt| rt.get(none));
    assert_eq!(value, 0);
    assert!(calls >= BATCHES, "{calls} calls filtering {N} deltas");
}

#[test]
fn a_count_partition_matching_nothing_calls_the_hook() {
    let c = Counted::<Local>::new();
    let col = c.rt.create_collection::<u64>();
    col.extend(&c.rt, 0..N);
    let (calls, split) = c.calls_during(|rt| {
        let split = col.count_partition(rt, |_| false);
        rt.get(split)
    });
    assert_eq!(split, (0, N));
    assert!(calls >= BATCHES, "{calls} calls counting {N} elements");
}

#[test]
fn a_keyed_reconcile_calls_the_hook() {
    let c = Counted::<Local>::new();
    let keyed: KeyedCollection<u64, u64, Local> = KeyedCollection::new(&c.rt);
    let (calls, change) = c.calls_during(|rt| keyed.reconcile(rt, (0..N).map(|k| (k, k))));
    assert_eq!(change.counts.inserted, N as usize);
    assert!(calls >= 2 * BATCHES, "{calls} calls reconciling {N} rows");
}

#[test]
fn a_removed_hook_is_never_called() {
    let mut c = Counted::<Local>::new();
    c.rt.set_yield_hook(None);
    let col = c.rt.create_collection::<u64>();
    let count = col.count(&c.rt);
    let (calls, value) = c.calls_during(|rt| {
        col.extend(rt, 0..N);
        rt.get(count)
    });
    assert_eq!((calls, value), (0, N));
}

fn counting(calls: &Arc<AtomicU64>, by: u64) -> YieldHook {
    let calls = Arc::clone(calls);
    YieldHook::new(move || {
        calls.fetch_add(by, Ordering::Relaxed);
    })
}

fn tick_ambient_batch() {
    let yielder = Yielder::ambient();
    yielder.tick_by(YIELD_EVERY as usize);
}

#[test]
fn a_nested_or_panicking_ambient_scope_restores_the_outer_hook() {
    let calls = Arc::new(AtomicU64::new(0));
    let outer = counting(&calls, 1);
    let inner = counting(&calls, 1000);
    with_ambient_hook(Some(&outer), || {
        with_ambient_hook(Some(&inner), tick_ambient_batch);
        assert_eq!(calls.load(Ordering::Relaxed), 1000);
        tick_ambient_batch();
        assert_eq!(calls.load(Ordering::Relaxed), 1001);
        let unwound = catch_unwind(AssertUnwindSafe(|| {
            with_ambient_hook(Some(&inner), || panic!("inside the inner scope"))
        }));
        assert!(unwound.is_err());
        tick_ambient_batch();
        assert_eq!(calls.load(Ordering::Relaxed), 1002);
        with_ambient_hook(None, tick_ambient_batch);
        assert_eq!(calls.load(Ordering::Relaxed), 1002);
    });
    tick_ambient_batch();
    assert_eq!(calls.load(Ordering::Relaxed), 1002);
}
