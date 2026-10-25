//! Park-point tests for restore and checkpoint under `Shared`: one thread
//! parks at a named point, or inside a value's `Clone`, while another
//! writes, and the restored or stored result must not reflect the stale
//! value.

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::thread;
use std::time::Duration;

use incr_core::persist_hooks::{assert_invariants, changed_at, inspect};
use incr_core::ErasedIncr;
use serde::{Deserialize, Serialize};

use super::{Park, PARK};
use crate::memo::{
    decode_entry, BuildId, Entry, Memo, MemoBackend, MemoBuilder, MemoOptions, MemoTable,
    MemoryMemo, StoredDep,
};
use crate::{Input, MemoryBackend, Options, Shared, Store};

const BUILD: u128 = 0x9a7c;

/// A park point that is never reached would otherwise hang the test.
const PARK_TIMEOUT: Duration = Duration::from_secs(10);

/// One-shot rendezvous: once armed, the first [`Gate::park`] blocks until
/// [`Gate::release`], and [`Gate::wait_parked`] returns once it blocks.
#[derive(Default)]
struct Gate {
    state: Mutex<GateState>,
    changed: Condvar,
}

#[derive(Default)]
struct GateState {
    armed: bool,
    parked: bool,
    released: bool,
}

impl Gate {
    fn arm(&self) {
        self.state.lock().unwrap().armed = true;
    }

    fn park(&self) {
        let mut state = self.state.lock().unwrap();
        if !std::mem::take(&mut state.armed) {
            return;
        }
        state.parked = true;
        self.changed.notify_all();
        let (state, timeout) = self
            .changed
            .wait_timeout_while(state, PARK_TIMEOUT, |s| !s.released)
            .unwrap();
        drop(state);
        assert!(!timeout.timed_out(), "parked thread was never released");
    }

    fn wait_parked(&self) {
        let state = self.state.lock().unwrap();
        let (state, timeout) = self
            .changed
            .wait_timeout_while(state, PARK_TIMEOUT, |s| !s.parked)
            .unwrap();
        drop(state);
        assert!(!timeout.timed_out(), "thread never reached its park point");
    }

    fn release(&self) {
        self.state.lock().unwrap().released = true;
        self.changed.notify_all();
    }
}

struct Backends {
    log: MemoryBackend,
    memo: MemoryMemo,
}

impl Backends {
    fn new() -> Self {
        Backends {
            log: MemoryBackend::new(),
            memo: MemoryMemo::new(),
        }
    }

    fn store(&self) -> Arc<Store<Shared, MemoryBackend>> {
        Arc::new(Store::open(self.log.clone(), Options::default()).unwrap())
    }
}

/// How `scaled` reads `x`: a Store input or an external, the two mapped
/// deps whose writes take the `DepIndex` hold.
#[derive(Clone, Copy, Debug)]
enum Dep {
    Input,
    External,
}

/// `scaled(k) = x * k` and `plus(k) = scaled(k) + 1`, counting runs.
struct World {
    store: Arc<Store<Shared, MemoryBackend>>,
    memo: Memo<Shared, MemoryBackend>,
    kind: Dep,
    x: Input<u64>,
    scaled: MemoTable<u32, u64>,
    plus: MemoTable<u32, u64>,
}

impl World {
    fn open(backends: &Backends, kind: Dep, runs: &Arc<AtomicUsize>) -> Self {
        let store = backends.store();
        let x = store.input("x", 1_u64).unwrap();
        let xh = x.handle();
        let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(BUILD)));
        let scaled = builder.table::<u32, u64>("scaled", 1);
        let plus = builder.table::<u32, u64>("plus", 1);
        let counted = Arc::clone(runs);
        builder.define(scaled, move |ctx, k| {
            counted.fetch_add(1, SeqCst);
            let x = match kind {
                Dep::Input => ctx.rt().get(xh),
                Dep::External => u64::try_from(ctx.external("x")).unwrap(),
            };
            x * u64::from(*k)
        });
        let counted = Arc::clone(runs);
        builder.define(plus, move |ctx, k| {
            counted.fetch_add(1, SeqCst);
            ctx.get(scaled, *k) + 1
        });
        let memo = builder.open(&store, backends.memo.clone()).unwrap();
        if let Dep::External = kind {
            memo.set_external("x", 1);
        }
        World {
            store,
            memo,
            kind,
            x,
            scaled,
            plus,
        }
    }

    fn set_x(&self, value: u64) {
        match self.kind {
            Dep::Input => {
                self.store.set(self.x, value).unwrap();
            }
            Dep::External => {
                self.memo.set_external("x", u128::from(value));
            }
        }
    }

    fn read(&self, table: MemoTable<u32, u64>, key: u32) -> u64 {
        let q = self.memo.get_or_insert(table, key);
        self.memo.runtime().get(q)
    }
}

/// Computes `table`'s entry for key 2 at `x = 1` and checkpoints it, so the
/// next `World::open` on `backends` restores it.
fn checkpointed(backends: &Backends, kind: Dep, table: fn(&World) -> MemoTable<u32, u64>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let w = World::open(backends, kind, &runs);
    w.read(table(&w), 2);
    assert!(w.memo.checkpoint().unwrap().written > 0);
}

/// Reads `table`'s entry for `key` on a second thread that parks at the
/// first `at` its restore reaches, sets `x` to `value` while it is parked,
/// and returns what the read got.
fn race(w: &World, at: Park, value: u64, table: MemoTable<u32, u64>, key: u32) -> u64 {
    let gate = Arc::new(Gate::default());
    gate.arm();
    thread::scope(|s| {
        let restorer = {
            let gate = Arc::clone(&gate);
            s.spawn(move || {
                PARK.set(Some(Box::new(move |point| {
                    if point == at {
                        gate.park();
                    }
                })));
                w.read(table, key)
            })
        };
        gate.wait_parked();
        w.set_x(value);
        gate.release();
        restorer.join().unwrap()
    })
}

fn a_set_between_the_base_and_the_fp_read_reruns_the_entry(kind: Dep) {
    let backends = Backends::new();
    checkpointed(&backends, kind, |w| w.scaled);
    let runs = Arc::new(AtomicUsize::new(0));
    let w = World::open(&backends, kind, &runs);
    assert_eq!(race(&w, Park::AfterBase, 5, w.scaled, 2), 10);
    let stats = w.memo.stats();
    assert_eq!(
        (stats.reused, stats.revalidated),
        (0, 1),
        "the fp read saw the set"
    );
    assert_eq!(runs.load(SeqCst), 1);
    assert_invariants(w.memo.runtime());
}

fn a_set_between_the_fp_read_and_the_seed_reruns_the_entry(kind: Dep) {
    let backends = Backends::new();
    checkpointed(&backends, kind, |w| w.scaled);
    let runs = Arc::new(AtomicUsize::new(0));
    let w = World::open(&backends, kind, &runs);
    assert_eq!(race(&w, Park::BeforeSeed, 5, w.scaled, 2), 10);
    let stats = w.memo.stats();
    assert_eq!(
        (stats.reused, stats.revalidated),
        (1, 0),
        "the fp read came before the set"
    );
    assert_eq!(runs.load(SeqCst), 1);
    assert_invariants(w.memo.runtime());
}

// The first `BeforeSeed` is `scaled`'s, inside the walk; `plus` then
// trusts it at its seed stamp, so only the set's stamp can rerun either.
fn a_set_between_a_deps_fp_read_and_its_seed_reruns_both(kind: Dep) {
    let backends = Backends::new();
    checkpointed(&backends, kind, |w| w.plus);
    let runs = Arc::new(AtomicUsize::new(0));
    let w = World::open(&backends, kind, &runs);
    assert_eq!(race(&w, Park::BeforeSeed, 5, w.plus, 2), 11);
    let stats = w.memo.stats();
    assert_eq!((stats.reused, stats.revalidated), (2, 0));
    assert_eq!(runs.load(SeqCst), 2);
    assert_invariants(w.memo.runtime());
}

macro_rules! kinds {
    ($($name:ident),* $(,)?) => {
        mod input {
            $(
                #[test]
                fn $name() {
                    super::$name(super::Dep::Input);
                }
            )*
        }
        mod external {
            $(
                #[test]
                fn $name() {
                    super::$name(super::Dep::External);
                }
            )*
        }
    };
}

kinds!(
    a_set_between_the_base_and_the_fp_read_reruns_the_entry,
    a_set_between_the_fp_read_and_the_seed_reruns_the_entry,
    a_set_between_a_deps_fp_read_and_its_seed_reruns_both,
);

type Clamped = (
    Arc<Store<Shared, MemoryBackend>>,
    Input<u64>,
    Memo<Shared, MemoryBackend>,
    MemoTable<u32, u64>,
    MemoTable<u32, u64>,
);

/// `scaled(k) = min(x, 1) * k`, parking at `gate` after it reads `x`, and
/// `plus(k) = scaled(k) + 1`, counting runs.
fn clamped(backends: &Backends, gate: &Arc<Gate>, runs: &Arc<AtomicUsize>) -> Clamped {
    let store = backends.store();
    let x = store.input("x", 1_u64).unwrap();
    let xh = x.handle();
    let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(BUILD)));
    let scaled = builder.table::<u32, u64>("scaled", 1);
    let plus = builder.table::<u32, u64>("plus", 1);
    let (counted, gate) = (Arc::clone(runs), Arc::clone(gate));
    builder.define(scaled, move |ctx, k| {
        counted.fetch_add(1, SeqCst);
        let x = ctx.rt().get(xh).min(1);
        gate.park();
        x * u64::from(*k)
    });
    let counted = Arc::clone(runs);
    builder.define(plus, move |ctx, k| {
        counted.fetch_add(1, SeqCst);
        ctx.get(scaled, *k) + 1
    });
    let memo = builder.open(&store, backends.memo.clone()).unwrap();
    (store, x, memo, scaled, plus)
}

/// `plus(2)` is restored while another thread holds `scaled(2)`'s claim,
/// running it after a set its value does not depend on. The trust check
/// can't tell that run from one that moves the value, and can't wait on
/// it under the memo lock, so `plus` seeds `RecomputeFirst`: it counts as
/// revalidated and runs once, cut off at its stored value.
#[test]
fn a_restore_over_a_claimed_dep_reruns_the_entry_once() {
    let backends = Backends::new();
    let gate = Arc::new(Gate::default());
    {
        let runs = Arc::new(AtomicUsize::new(0));
        let (_store, _x, memo, _, plus) = clamped(&backends, &gate, &runs);
        assert_eq!(memo.runtime().get(memo.get_or_insert(plus, 2)), 3);
        assert_eq!(memo.checkpoint().unwrap().written, 2);
    }
    let runs = Arc::new(AtomicUsize::new(0));
    let (store, x, memo, scaled, plus) = clamped(&backends, &gate, &runs);
    store.set(x, 3).unwrap();
    gate.arm();
    thread::scope(|s| {
        let reader = s.spawn(|| memo.runtime().get(memo.get_or_insert(scaled, 2)));
        gate.wait_parked();
        let q = memo.get_or_insert(plus, 2);
        let stats = memo.stats();
        assert_eq!((stats.reused, stats.revalidated), (0, 2));
        gate.release();
        assert_eq!(reader.join().unwrap(), 2);
        assert_eq!(memo.runtime().get(q), 3);
    });
    assert_eq!(runs.load(SeqCst), 2);
    assert_invariants(memo.runtime());
}

/// Armed by the one test that stores `Parked`; `inspect` clones the value
/// between its two holds, so its first clone parks the checkpoint there.
static CLONE_GATE: LazyLock<Gate> = LazyLock::new(Gate::default);

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Parked(u64);

impl Clone for Parked {
    fn clone(&self) -> Self {
        CLONE_GATE.park();
        Parked(self.0)
    }
}

fn stored(memo: &MemoryMemo) -> Vec<Entry> {
    let Some(generation) = memo.load().unwrap() else {
        return Vec::new();
    };
    generation
        .items()
        .iter()
        .map(|item| {
            let frame = memo.read_entry(generation.id(), item).unwrap().unwrap();
            decode_entry(&frame, generation.frame_tag()).unwrap()
        })
        .collect()
}

// A mapped write would wait on the snapshot's `DepIndex` hold, so the dep
// change mid-`inspect` comes from a write around the store. That write also
// unmaps x, which the stamp check in `mapping.rs` catches whatever `inspect`
// returns, so this guards what a checkpoint writes; `inspect`'s own recheck
// is covered by incr-core's seed tests.
#[test]
fn a_checkpoint_never_stores_deps_a_cutoff_moved_mid_inspect() {
    let backends = Backends::new();
    let store = backends.store();
    let x = store.input("x", false).unwrap().handle();
    let a = store.input("a", 5_u64).unwrap().handle();
    let b = store.input("b", 5_u64).unwrap().handle();
    let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(BUILD)));
    let table = builder.table::<u32, Parked>("q", 1);
    builder.define(table, move |ctx, _| {
        let side = if ctx.rt().get(x) { b } else { a };
        Parked(ctx.rt().get(side))
    });
    let memo = builder.open(&store, backends.memo.clone()).unwrap();
    let rt = memo.runtime();
    let q = memo.get_or_insert(table, 0);
    assert_eq!(rt.get(q), Parked(5));
    let stamp = changed_at(rt, q);
    CLONE_GATE.arm();
    let checkpointed = thread::scope(|s| {
        let checkpoint = s.spawn(|| memo.checkpoint().unwrap());
        CLONE_GATE.wait_parked();
        rt.set(x, true);
        assert_eq!(rt.get(q), Parked(5));
        CLONE_GATE.release();
        checkpoint.join().unwrap()
    });
    assert_eq!(changed_at(rt, q), stamp, "not a cutoff");
    let deps = inspect(rt, q).expect("Clean").deps;
    assert_eq!(deps, vec![x.erase(), b.erase()] as Vec<ErasedIncr>);
    assert_eq!(checkpointed.written, 0);
    let over_a = stored(&backends.memo).into_iter().any(|entry| {
        entry
            .deps
            .iter()
            .any(|dep| matches!(dep, StoredDep::Input { key, .. } if key == "a"))
    });
    assert!(!over_a, "a checkpoint stored q over a after it moved to b");
    assert_invariants(rt);
}
