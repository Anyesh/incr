//! Loom models for the `Shared` verification protocol (plan fix 8). Each
//! model runs the real runtime on the [`Loom`] strategy and, at the end
//! of every explored execution, checks the structural invariant (I) and
//! the value invariant (V) against an oracle.
//!
//! Run one model with:
//! `RUSTFLAGS="--cfg loom" LOOM_MAX_PREEMPTIONS=2 LOOM_MAX_BRANCHES=100000 cargo test -p incr-core --release --lib -- --exact loom_models::loom_fan_out`
//!
//! Each preemption of bound costs roughly 30x, so CI runs every model to
//! completion at bound 2 on each push and the models that finish at bound
//! 3 within the hour (`loom.yml`) weekly. A bound run to completion beats a
//! larger one cut off by a timeout: B7 sat at bound 2 while a bound-3 run
//! of the same model spent three hours elsewhere in the search.

use std::cell::{Cell, RefCell};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Once;

use loom::sync::atomic::{AtomicU64, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;

use crate::changes::{Changes, Count};
use crate::generic_arena::GenericArena;
use crate::handle::Incr;
use crate::loom_cells::Loom;
use crate::node::NodeData;
use crate::runtime::Runtime;
use crate::segmented_nodes::SegmentedNodes;
use crate::state::NodeState;

type Rt = Runtime<Loom>;

loom::thread_local! {
    static EXPECTING_UNWIND: Cell<bool> = Cell::new(false);
}

/// Run `f` on a racing thread where the runtime may legitimately unwind
/// (stale handle, read of a deleted node, read of a cleared slot).
/// Returns `None` on unwind.
fn may_unwind<R>(f: impl FnOnce() -> R) -> Option<R> {
    EXPECTING_UNWIND.with(|e| e.set(true));
    let out = catch_unwind(AssertUnwindSafe(f)).ok();
    EXPECTING_UNWIND.with(|e| e.set(false));
    out
}

/// Panics inside [`may_unwind`] would otherwise print once per explored
/// execution. Every other panic, including a runtime panic on a model's
/// final reads, still reaches the default hook.
fn silence_expected_panics() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if IN_MODEL.with(Cell::get) {
                if fatal(info) {
                    report_raw(info);
                } else if EXPECTING_UNWIND.try_with(Cell::get).unwrap_or(false) {
                    return;
                } else {
                    LAST_SHOWN.with(|l| *l.borrow_mut() = Some(describe(info)));
                }
            }
            default(info);
        }));
    });
}

/// Loom's own panics (a branch limit, a deadlock, its state touched
/// outside an execution) and core's non-unwinding ones. While these run
/// loom's thread-locals panic on access, which in the hook aborts: loom
/// may hold its state borrowed, or a failed model's teardown drops threads
/// whose locals hold loom objects outside any execution.
fn fatal(info: &std::panic::PanicHookInfo<'_>) -> bool {
    info.location()
        .is_some_and(|l| l.file().contains("/loom-") || l.file().ends_with("core/src/panicking.rs"))
}

fn describe(info: &std::panic::PanicHookInfo<'_>) -> String {
    let payload = info.payload();
    let msg = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("");
    let at = info.location().map(ToString::to_string).unwrap_or_default();
    format!("panicked at {at}: {msg}")
}

/// A fatal panic usually ends in an abort, which loses the test harness's
/// captured output, so it and the last panic shown before it, often the
/// model's failed assertion, also go to the process's stderr.
fn report_raw(info: &std::panic::PanicHookInfo<'_>) {
    use std::io::Write;
    let mut err = std::io::stderr().lock();
    if let Some(last) = LAST_SHOWN.with(|l| l.take()) {
        let _ = writeln!(err, "loom model, last panic shown: {last}");
    }
    let _ = writeln!(err, "loom model, fatal: {}", describe(info));
}

std::thread_local! {
    static IN_MODEL: Cell<bool> = const { Cell::new(false) };
    static LAST_SHOWN: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Marks the thread as inside [`loom::model`] until dropped, unwinding
/// included.
struct InModel;

impl InModel {
    fn enter() -> Self {
        IN_MODEL.with(|m| m.set(true));
        LAST_SHOWN.with(|l| *l.borrow_mut() = None);
        Self
    }
}

impl Drop for InModel {
    fn drop(&mut self) {
        IN_MODEL.with(|m| m.set(false));
    }
}

pub(crate) fn run(f: impl Fn() + Sync + Send + 'static) {
    silence_expected_panics();
    let _in = InModel::enter();
    loom::model(f);
}

/// x -> Q (early cutoff) -> D. Two setters race a reader of D, and their
/// walks can finish out of order. Targets B4, B5 and B6: a reader that
/// lands D `Clean` past a concurrent set leaves D `Clean` over a
/// non-Clean Q, or stale against x. Slot reuse in the ring is covered by
/// `walk_ring::loom_tests` instead, because a third set here puts the
/// model out of reach at preemption bound 3.
#[test]
fn loom_chain_cutoff() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(0_u64);
        let q = rt.create_query(move |rt: &Rt| rt.get(x) / 2);
        let d = rt.create_query(move |rt: &Rt| rt.get(q) + 100);
        assert_eq!(rt.get(d), 100);

        let setter_a = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(x, 1))
        };
        let setter_b = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(x, 2))
        };
        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.get(d);
            })
        };
        setter_a.join().unwrap();
        setter_b.join().unwrap();
        reader.join().unwrap();

        rt.assert_invariant_i();
        rt.assert_edges_complete();
        rt.assert_walks_complete();
        let xv = rt.get(x);
        assert_eq!(
            rt.get(d),
            xv / 2 + 100,
            "(V) violated: D stale against x = {xv}"
        );
    });
}

/// x -> Q (early cutoff) -> {D1, D2}, one setter and one reader per
/// dependent. The shape that reaches fix 1's state-then-changed_at load
/// order: a verifier of D1 observes Q while the reader of D2 recomputes
/// it. The first set cuts off at Q and the second changes it, which is
/// what B7 needs: one reader observes Q `Dirty` from the first set and
/// claims it on the second, while the other cleans Q and verifies D1 in
/// between. Found at preemption bound 2.
#[test]
fn loom_fan_out() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(0_u64);
        let q = rt.create_query(move |rt: &Rt| rt.get(x) / 2);
        let d1 = rt.create_query(move |rt: &Rt| rt.get(q) + 100);
        let d2 = rt.create_query(move |rt: &Rt| rt.get(q) * 10);
        assert_eq!(rt.get(d1), 100);
        assert_eq!(rt.get(d2), 0);

        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.set(x, 1);
                rt.set(x, 2);
            })
        };
        let r1 = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.get(d1);
            })
        };
        let r2 = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.get(d2);
            })
        };
        setter.join().unwrap();
        r1.join().unwrap();
        r2.join().unwrap();

        rt.assert_invariant_i();
        rt.assert_edges_complete();
        rt.assert_walks_complete();
        let qv = rt.get(x) / 2;
        assert_eq!(rt.get(d1), qv + 100, "(V) violated: D1 stale");
        assert_eq!(rt.get(d2), qv * 10, "(V) violated: D2 stale");
    });
}

/// x -> P -> D with D New, so D's edge to P is not published yet. A
/// setter races D's first compute and a reader of P. Targets bug l: D
/// reads P Clean after the set's bump and before its walk, P recomputes
/// on the other thread, and D's publish must not count P settled.
#[test]
fn loom_new_dependent_races_set() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(1_u64);
        let p = rt.create_query(move |rt: &Rt| rt.get(x) + 1);
        let d = rt.create_query(move |rt: &Rt| rt.get(p) * 10);
        assert_eq!(rt.get(p), 2);

        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(x, 5))
        };
        let reader_d = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.get(d);
            })
        };
        let reader_p = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.get(p);
            })
        };
        setter.join().unwrap();
        reader_d.join().unwrap();
        reader_p.join().unwrap();

        rt.assert_invariant_i();
        rt.assert_edges_complete();
        rt.assert_walks_complete();
        assert_eq!(rt.get(d), 60, "(V) violated: D stale against x = 5");
    });
}

/// Q's first compute reads v while another thread deletes v and recycles
/// its slot into a new input. Targets B1 and B2. The reader may unwind
/// (stale handle, `Dead` arm, cleared slot); when its `get` returns, the
/// value must be the oracle's. No edge may land on the new occupant, and
/// a successful delete means Q can't be `Clean` over v.
#[test]
fn loom_compute_races_delete() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let v = rt.create_input(7_u64);
        let q = rt.create_query(move |rt: &Rt| rt.get(v) + 1);

        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || may_unwind(|| rt.get(q)))
        };
        let deleter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                let deleted = may_unwind(|| rt.delete_node(v)).is_some();
                let recycled = deleted.then(|| rt.create_input(99_u64));
                (deleted, recycled)
            })
        };
        let read = reader.join().unwrap();
        let (deleted, recycled) = deleter.join().unwrap();

        if let Some(value) = read {
            assert_eq!(
                value, 8,
                "(V) violated: Q returned a value not computed from v"
            );
        }
        rt.assert_invariant_i();
        rt.assert_edges_complete();
        rt.assert_walks_complete();
        if let Some(new) = recycled {
            assert!(
                !rt.has_reverse_edge(new, q),
                "phantom edge: Q landed on the recycled slot's dependents"
            );
        }
        if deleted {
            assert_ne!(
                rt.state_of(q),
                NodeState::Clean,
                "Q is Clean over a deleted dep"
            );
        }
    });
}

/// A dependent P over a query Q whose first compute races a delete of v,
/// while a setter moves x so that Q stops reading v. Exercises the stale
/// recovery (fix 5) and the walk through a recovered node (fix 6): once
/// x = 1 every path must recover, so the final read of P is exact.
#[test]
fn loom_recovered_node_under_set() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(0_u64);
        let v = rt.create_input(7_u64);
        let q = rt.create_query(move |rt: &Rt| {
            let xv = rt.get(x);
            if xv == 0 {
                xv + rt.get(v)
            } else {
                xv
            }
        });
        let p = rt.create_query(move |rt: &Rt| rt.get(q) + 100);

        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                may_unwind(|| rt.get(p));
            })
        };
        let deleter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                let deleted = may_unwind(|| rt.delete_node(v)).is_some();
                deleted.then(|| rt.create_input(99_u64))
            })
        };
        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(x, 1))
        };
        reader.join().unwrap();
        let recycled = deleter.join().unwrap();
        setter.join().unwrap();

        rt.assert_invariant_i();
        rt.assert_edges_complete();
        rt.assert_walks_complete();
        if let Some(new) = recycled {
            assert!(
                !rt.has_reverse_edge(new, q),
                "phantom edge: Q landed on the recycled slot's dependents"
            );
        }
        assert_eq!(
            rt.get(p),
            101,
            "(V) violated: P did not recover after x = 1"
        );
    });
}

/// Q reads whichever input `pick` names, m until a thread points it at a
/// node it created. Main dirties Q and P through m, then thread A creates
/// n in a fresh segment (segments hold four nodes under loom) and
/// recomputes Q, installing [n] over [m]: the same dep count. Thread B
/// walks P and scans Q's deps without a claim, so it can read the old
/// count and the new id; it must not index n's segment before it has
/// happens-before with n's creation. Targets bug i's inline half.
#[test]
fn loom_walk_scan_after_same_count_install() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let m = rt.create_input(1_u64);
        let pick: Arc<Mutex<Option<Incr<u64>>>> = Arc::new(Mutex::new(None));
        let q = {
            let pick = Arc::clone(&pick);
            rt.create_query(move |rt: &Rt| {
                let chosen = pick.lock().unwrap().unwrap_or(m);
                rt.get(chosen)
            })
        };
        let p = rt.create_query(move |rt: &Rt| rt.get(q) + 100);
        let _fills_segment_zero = rt.create_input(0_u64);
        assert_eq!(rt.get(p), 101);
        rt.set(m, 2);

        let installer = {
            let rt = Arc::clone(&rt);
            let pick = Arc::clone(&pick);
            thread::spawn(move || {
                let n = rt.create_input(5_u64);
                *pick.lock().unwrap() = Some(n);
                rt.get(q)
            })
        };
        let walker = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(p))
        };
        installer.join().unwrap();
        walker.join().unwrap();

        rt.assert_invariant_i();
        rt.assert_edges_complete();
        rt.assert_walks_complete();
        let pv = rt.get(p);
        assert!(
            pv == 102 || pv == 105,
            "(V) violated: P = {pv} matches neither pick"
        );
    });
}

/// Q reads x and, while x is 0, seven more inputs, so its deps overflow
/// the inline array. Main sets x to 1, which dirties Q and P. Thread A
/// recomputes Q, whose deps shrink to [x, i0] and move back inline.
/// Thread B walks P and scans Q's deps without a claim, so it can read
/// the overflow marker after the installer has swapped the list out and
/// before it has stored the inline count. Targets bug i's overflow half.
#[test]
fn loom_walk_scan_after_overflow_to_inline_install() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(0_u64);
        let inputs: Vec<Incr<u64>> = (1..=7).map(|v| rt.create_input(v)).collect();
        let q = {
            let inputs = inputs.clone();
            rt.create_query(move |rt: &Rt| {
                if rt.get(x) == 0 {
                    inputs.iter().map(|&i| rt.get(i)).sum()
                } else {
                    rt.get(inputs[0])
                }
            })
        };
        let p = rt.create_query(move |rt: &Rt| rt.get(q) + 100);
        assert_eq!(rt.get(p), 128);
        rt.set(x, 1);

        let installer = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(q))
        };
        let walker = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(p))
        };
        assert_eq!(installer.join().unwrap(), 1);
        assert_eq!(walker.join().unwrap(), 101);

        rt.assert_invariant_i();
        rt.assert_edges_complete();
        rt.assert_walks_complete();
        assert_eq!(rt.get(p), 101, "(V) violated: P stale after x = 1");
    });
}

/// A Clean read of v races a delete of v and a recycle of its slot by an
/// input of the same type, which reuses v's arena slot too. The read
/// returns v's value or unwinds as stale; it never returns the new
/// occupant's value. Targets bug (d)'s read half.
#[test]
fn loom_clean_get_vs_recycle() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let v = rt.create_input(7_u64);

        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || may_unwind(|| rt.get(v)))
        };
        let recycler = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.delete_node(v);
                rt.create_input(99_u64)
            })
        };
        let read = reader.join().unwrap();
        let new = recycler.join().unwrap();

        assert_eq!(new.slot(), v.slot(), "the recycle reused v's slot");
        if let Some(value) = read {
            assert_eq!(
                value, 7,
                "(V) violated: the read returned the new occupant's value"
            );
        }
        assert_eq!(rt.get(new), 99);
        rt.assert_invariant_i();
    });
}

/// As above, but the recycler first reuses v's node slot for an input of
/// another type, so v's arena slot is taken by a node in the segment the
/// recycler allocates (segments hold four nodes under loom). The stale
/// read's node slot and its arena slot then belong to different nodes.
#[test]
fn loom_clean_get_vs_recycle_new_segment() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let v = rt.create_input(7_u64);
        let _fills_segment_zero: Vec<Incr<u64>> = (1..=3).map(|k| rt.create_input(k)).collect();

        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || may_unwind(|| rt.get(v)))
        };
        let recycler = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.delete_node(v);
                let other = rt.create_input("s".to_string());
                let new = rt.create_input(99_u64);
                (other, new)
            })
        };
        let read = reader.join().unwrap();
        let (other, new) = recycler.join().unwrap();

        assert_eq!(other.slot(), v.slot(), "the recycle reused v's node slot");
        assert_eq!(
            new.slot(),
            4,
            "the u64 occupant lives in the second segment"
        );
        if let Some(value) = read {
            assert_eq!(
                value, 7,
                "(V) violated: the read returned the new occupant's value"
            );
        }
        assert_eq!(rt.get(new), 99);
        assert_eq!(rt.get(other), "s");
        rt.assert_invariant_i();
    });
}

/// A set of v races a delete of v and a recycle of its slot by an input
/// of the same type. The set's claim on v's `verified_at` either lands
/// before the delete marks v, which then waits for its release, or is
/// refused and the set unwinds as stale. It never writes into the new
/// occupant and never leaves it claimed or marked. Targets bug (g).
#[test]
fn loom_set_vs_recycle() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let v = rt.create_input(7_u64);

        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || may_unwind(|| rt.set(v, 8)))
        };
        let recycler = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.delete_node(v);
                rt.create_input(99_u64)
            })
        };
        setter.join().unwrap();
        let new = recycler.join().unwrap();

        assert_eq!(new.slot(), v.slot(), "the recycle reused v's slot");
        assert_eq!(
            rt.get(new),
            99,
            "(V) violated: the set wrote into the occupant"
        );
        rt.set(new, 100);
        assert_eq!(rt.get(new), 100);
        rt.assert_invariant_i();
    });
}

/// A reader of an arena slot it received through a Release store races
/// the reserves that allocate the next segments and double the segment
/// directory twice (one-entry directories and four-slot segments under
/// loom). Whichever directory the reader loads holds the slot's segment,
/// so the read finds the value. The lock-free arena read behind every
/// Clean `get` relies on this.
#[test]
fn loom_arena_read_during_growth() {
    use loom::sync::atomic::{AtomicU32, Ordering};
    run(|| {
        let arena: Arc<GenericArena<u64, Loom>> = Arc::new(GenericArena::new());
        for k in 0..4 {
            arena.reserve_with(k);
        }
        let handoff = Arc::new(AtomicU32::new(u32::MAX));

        let writer = {
            let arena = Arc::clone(&arena);
            let handoff = Arc::clone(&handoff);
            thread::spawn(move || {
                let s = arena.reserve_with(4);
                handoff.store(s, Ordering::Release);
                for k in 5..9 {
                    arena.reserve_with(k);
                }
            })
        };
        let reader = {
            let arena = Arc::clone(&arena);
            let handoff = Arc::clone(&handoff);
            thread::spawn(move || {
                let s = handoff.load(Ordering::Acquire);
                if s != u32::MAX {
                    assert_eq!(arena.try_read(s), Some(4), "a handed-off slot was missing");
                }
                assert_eq!(arena.try_read(0), Some(0));
            })
        };
        writer.join().unwrap();
        reader.join().unwrap();
        for k in 0..9 {
            assert_eq!(arena.read(k), k as u64);
        }
    });
}

/// A reader of a node slot it received through a Release store races the
/// pushes that allocate the next segments and double the node directory
/// twice (one-entry directories and four-node segments under loom).
/// Whichever directory the reader loads holds the slot's segment, even
/// though it loads the segment pointer Relaxed. Every node read relies on
/// this once the store outgrows its first directory.
#[test]
fn loom_reader_during_directory_growth() {
    use loom::sync::atomic::{AtomicU32, Ordering};
    run(|| {
        let nodes: Arc<SegmentedNodes<Loom>> = Arc::new(SegmentedNodes::new());
        for k in 0..4 {
            nodes.push(NodeData::new_input(0, k, 2));
        }
        let handoff = Arc::new(AtomicU32::new(u32::MAX));

        let writer = {
            let nodes = Arc::clone(&nodes);
            let handoff = Arc::clone(&handoff);
            thread::spawn(move || {
                let s = nodes.push(NodeData::new_input(0, 4, 2));
                handoff.store(s, Ordering::Release);
                for k in 5..9 {
                    nodes.push(NodeData::new_input(0, k, 2));
                }
            })
        };
        let reader = {
            let nodes = Arc::clone(&nodes);
            let handoff = Arc::clone(&handoff);
            thread::spawn(move || {
                let s = handoff.load(Ordering::Acquire);
                if s != u32::MAX {
                    assert_eq!(
                        nodes.get(s).arena_slot(),
                        4,
                        "a handed-off node was missing"
                    );
                }
                assert_eq!(nodes.get(0).arena_slot(), 0);
            })
        };
        writer.join().unwrap();
        reader.join().unwrap();
        for k in 0..9 {
            assert_eq!(nodes.get(k).arena_slot(), k);
        }
    });
}

/// A (Fallback) ↔ B, where B also reads x. A reader lands the cycle while
/// a setter changes x: the group lands with x's old value and is walked
/// Dirty, or lands with the new one.
#[test]
fn loom_cycle_land_under_set() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(0_u64);
        let bc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
        let a = {
            let bc = bc.clone();
            rt.create_query_with_fallback(
                move |rt: &Rt| rt.get(*bc.get().unwrap()) + 1,
                |_c: &crate::cycle::Cycle| 100,
            )
        };
        let b = rt.create_query(move |rt: &Rt| rt.get(a) + rt.get(x));
        bc.set(b).unwrap();

        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.get(a);
            })
        };
        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(x, 1))
        };
        reader.join().unwrap();
        setter.join().unwrap();

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(rt.get(a), 100);
        assert_eq!(rt.get(b), 101, "(V) violated: B stale against x = 1");
    });
}

/// A (Fallback) ↔ B, landed. A reads y through a cutoff, so a `set` of y
/// leaves the group Dirty with nothing changed, and a reader's group
/// verify re-lands it while a second setter changes x, which B reads.
#[test]
fn loom_group_verify_under_set() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(0_u64);
        let y = rt.create_input(0_u64);
        let cut = rt.create_query(move |rt: &Rt| rt.get(y) / 4);
        let bc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
        let a = {
            let bc = bc.clone();
            rt.create_query_with_fallback(
                move |rt: &Rt| rt.get(*bc.get().unwrap()) + rt.get(cut),
                |_c: &crate::cycle::Cycle| 100,
            )
        };
        let b = rt.create_query(move |rt: &Rt| rt.get(a) + rt.get(x));
        bc.set(b).unwrap();
        assert_eq!(rt.get(a), 100);
        assert_eq!(rt.get(b), 100);
        rt.set(y, 1);

        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.get(b);
            })
        };
        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(x, 1))
        };
        reader.join().unwrap();
        setter.join().unwrap();

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(rt.get(b), 101, "(V) violated: B stale against x = 1");
        assert_eq!(rt.get(a), 100);
    });
}

/// A (Fallback) ↔ B, landed and then made Dirty with no change. Two
/// readers enter the group at different members, so one group verify can
/// find the other member claimed and wait on it, and a wound in that wait
/// releases the younger reader's claim, after which a walk may reach the
/// cycle at either member. The members read each
/// other with `try_get`, so every entry lands the same values.
#[test]
fn loom_group_claim_partial() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let y = rt.create_input(0_u64);
        let cut = rt.create_query(move |rt: &Rt| rt.get(y) / 4);
        let bc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
        let a = {
            let bc = bc.clone();
            rt.create_query_with_fallback(
                move |rt: &Rt| rt.try_get(*bc.get().unwrap()).unwrap_or(0) + rt.get(cut),
                |_c: &crate::cycle::Cycle| 100,
            )
        };
        let b = rt.create_query(move |rt: &Rt| rt.try_get(a).map_or(0, |v| v + 1));
        bc.set(b).unwrap();
        assert_eq!(rt.get(b), 101);
        rt.set(y, 1);

        let first = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(a))
        };
        let second = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(b))
        };
        assert_eq!(first.join().unwrap(), 100);
        assert_eq!(second.join().unwrap(), 101);

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(rt.get(a), 100);
        assert_eq!(rt.get(b), 101);
        assert_eq!(crate::waits::registered(), 0);
    });
}

/// A (Fallback) and B, which has no strategy, read each other with `get`,
/// landed and then made Dirty. One reader enters the group at each member,
/// so each can hold its member while it waits for the other's. B alone on
/// a stack is a cycle error, so the result is the single-thread one only
/// if the reader left holding the group verifies it as a group.
#[test]
fn loom_group_claim_contended() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let y = rt.create_input(0_u64);
        let bc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
        let a = {
            let bc = bc.clone();
            rt.create_query_with_fallback(
                move |rt: &Rt| rt.get(*bc.get().unwrap()) + rt.get(y),
                |_c: &crate::cycle::Cycle| 100,
            )
        };
        let b = rt.create_query(move |rt: &Rt| rt.get(a) + 1);
        bc.set(b).unwrap();
        assert_eq!(rt.get(a), 100);
        rt.set(y, 1);

        let first = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(a))
        };
        let second = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(b))
        };
        assert_eq!(first.join().unwrap(), 100);
        assert_eq!(second.join().unwrap(), 101);

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(crate::waits::registered(), 0);
    });
}

/// X, which has no strategy, keeps a record of the group {X, M}, and M
/// (Fallback) has left it, as a single-node landing of M leaves it; once
/// k is 1, X reads M, and M reads X and then W (Fallback), which reads X.
/// The reader of M computes it alone, and its group verify of X finds M
/// on its stack and computes X alone. A wound of that reader must leave X
/// in its group: out of it, the reader of W computes X alone, a cycle
/// error that neither single-thread order raises.
#[test]
fn loom_wound_keeps_a_member_computed_alone_in_its_group() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let k = rt.create_input(0_u64);
        let mc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
        let wc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
        let x = {
            let mc = mc.clone();
            rt.create_query(move |rt: &Rt| match rt.get(k) {
                0 => 1,
                _ => rt.get(*mc.get().unwrap()) + 1,
            })
        };
        let w = rt.create_query_with_fallback(
            move |rt: &Rt| match rt.get(k) {
                0 => 0,
                _ => rt.get(x) + 1000,
            },
            |_c: &crate::cycle::Cycle| 500,
        );
        let m = {
            let wc = wc.clone();
            rt.create_query_with_fallback(
                move |rt: &Rt| match rt.get(k) {
                    0 => 0,
                    _ => rt.get(x) + rt.get(*wc.get().unwrap()),
                },
                |_c: &crate::cycle::Cycle| 100,
            )
        };
        mc.set(m).unwrap();
        wc.set(w).unwrap();
        assert_eq!((rt.get(m), rt.get(x), rt.get(w)), (0, 1, 0));
        rt.force_group_for_test(&[x.erase(), m.erase()]);
        let (node, flags) = rt.nodes.get_with_flags(m.slot());
        rt.leave_group(
            crate::node::NodeId(m.slot()),
            node,
            flags,
            crate::state::NodeState::Clean,
        );
        rt.set(k, 1);

        let reads_w = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(w))
        };
        let reads_m = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(m))
        };
        assert_eq!(reads_w.join().unwrap(), 500);
        assert_eq!(reads_m.join().unwrap(), 100);
        assert_eq!(rt.get(x), 101);

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(crate::waits::registered(), 0);
    });
}

/// The landed group {r, x, y} after k is set to 1: r reads x, x
/// (Fallback) reads y, and y reads x, so r is out of the cycle. A's
/// group verify of r runs an episode that releases x and y; until A's
/// read ends unwounded the other thread still sees them in the group, so
/// B's read of y verifies the group, or waits for A, as a single-thread
/// order does. Seeing them out, B computes y alone and enters the cycle
/// at the unseeded y: a cycle error neither single-thread order raises.
fn released_group(rt: &Rt) -> (Incr<u64>, Incr<u64>) {
    let k = rt.create_input(0_u64);
    let yc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
    let x = {
        let yc = yc.clone();
        rt.create_query_with_fallback(
            move |rt: &Rt| match rt.get(k) {
                0 => 0,
                _ => rt.get(*yc.get().unwrap()) + 1,
            },
            |_c: &crate::cycle::Cycle| 10,
        )
    };
    let y = rt.create_query(move |rt: &Rt| match rt.get(k) {
        0 => 0,
        _ => rt.get(x) + 100,
    });
    let r = rt.create_query(move |rt: &Rt| match rt.get(k) {
        0 => 1,
        _ => rt.get(x) + 1000,
    });
    yc.set(y).unwrap();
    assert_eq!((rt.get(r), rt.get(x), rt.get(y)), (1, 0, 0));
    rt.force_group_for_test(&[r.erase(), x.erase(), y.erase()]);
    rt.set(k, 1);
    (r, y)
}

type SlotRead = Result<u64, Vec<u32>>;

/// A read's result with its cycle as the slots it names, once every
/// member is checked to be `rt`'s: the serial orders run on runtimes built
/// alike, whose handles differ from the model's only by runtime.
fn by_slots(read: Result<u64, crate::cycle::Cycle>, rt: &Rt) -> SlotRead {
    read.map_err(|c| {
        assert!(
            c.participants()
                .iter()
                .all(|p| p.runtime_id() == rt.runtime_id),
            "the cycle {c:?} names another runtime's node"
        );
        c.participants().iter().map(|p| p.slot()).collect()
    })
}

#[test]
fn loom_released_group_members_stay_grouped_until_the_read_ends() {
    run(|| {
        let orders: Vec<(SlotRead, SlotRead)> = [true, false]
            .into_iter()
            .map(|r_first| {
                let rt = Rt::new();
                let (r, y) = released_group(&rt);
                let (a, b) = if r_first {
                    let a = rt.try_get(r);
                    (a, rt.try_get(y))
                } else {
                    let b = rt.try_get(y);
                    (rt.try_get(r), b)
                };
                (by_slots(a, &rt), by_slots(b, &rt))
            })
            .collect();
        let rt = Arc::new(Rt::new());
        let (r, y) = released_group(&rt);
        let reads_r = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.try_get(r))
        };
        let reads_y = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.try_get(y))
        };
        let got = (
            by_slots(reads_r.join().unwrap(), &rt),
            by_slots(reads_y.join().unwrap(), &rt),
        );
        assert!(orders.contains(&got), "{got:?}, single-thread {orders:?}");

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(crate::waits::registered(), 0);
    });
}

/// After k is set to 1, x (Fallback) and y, a landed group, form a cycle;
/// q reads y and o reads x and then q. A's read of o lands the episode
/// {x, y} and then waits on B's claim of q, while B's read of q waits on
/// y: the waits-for cycle wounds one of them. A wounded A retracts the
/// episode, so B never builds on a value of an attempt that gave its claims
/// back, both reads match a single-thread order, and no group record names
/// a node that left it.
fn held_episode(rt: &Rt) -> (Incr<u64>, Incr<u64>, Incr<u64>, Incr<u64>) {
    let k = rt.create_input(0_u64);
    let yc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
    let x = {
        let yc = yc.clone();
        rt.create_query_with_fallback(
            move |rt: &Rt| match rt.get(k) {
                0 => 0,
                _ => rt.get(*yc.get().unwrap()) + 1,
            },
            |_c: &crate::cycle::Cycle| 10,
        )
    };
    let y = rt.create_query(move |rt: &Rt| match rt.get(k) {
        0 => 0,
        _ => rt.get(x) + 100,
    });
    let q = rt.create_query(move |rt: &Rt| match rt.get(k) {
        0 => 0,
        _ => rt.get(y) + 1000,
    });
    let o = rt.create_query(move |rt: &Rt| match rt.get(k) {
        0 => 0,
        _ => rt.get(x) + rt.get(q),
    });
    yc.set(y).unwrap();
    assert_eq!((rt.get(o), rt.get(q)), (0, 0));
    rt.force_group_for_test(&[x.erase(), y.erase()]);
    rt.set(k, 1);
    (o, q, x, y)
}

#[test]
fn loom_held_episode_wound_retracts_values() {
    run(|| {
        let orders: Vec<(SlotRead, SlotRead)> = [true, false]
            .into_iter()
            .map(|o_first| {
                let rt = Rt::new();
                let (o, q, _, _) = held_episode(&rt);
                let (a, b) = if o_first {
                    let a = rt.try_get(o);
                    (a, rt.try_get(q))
                } else {
                    let b = rt.try_get(q);
                    (rt.try_get(o), b)
                };
                (by_slots(a, &rt), by_slots(b, &rt))
            })
            .collect();
        let rt = Arc::new(Rt::new());
        let (o, q, x, y) = held_episode(&rt);
        let reads_o = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.try_get(o))
        };
        let reads_q = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.try_get(q))
        };
        let got = (
            by_slots(reads_o.join().unwrap(), &rt),
            by_slots(reads_q.join().unwrap(), &rt),
        );
        assert!(orders.contains(&got), "{got:?}, single-thread {orders:?}");
        assert_eq!(rt.stale_group_records(), Vec::<String>::new());
        assert!(
            rt.in_group_of(x.erase(), y.erase()),
            "x and y left their group"
        );

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(crate::waits::registered(), 0);
    });
}

/// A (Fallback) and B, where B also reads x, and o, which reads A and
/// then B. A reader of o lands the episode {A, B} and is still in o when a
/// setter changes x: the held members publish Dirty if the set came
/// first, and the set's walk reaches o either way.
#[test]
fn loom_held_landing_publishes_under_set() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(0_u64);
        let bc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
        let a = {
            let bc = bc.clone();
            rt.create_query_with_fallback(
                move |rt: &Rt| rt.get(*bc.get().unwrap()) + 1,
                |_c: &crate::cycle::Cycle| 100,
            )
        };
        let b = rt.create_query(move |rt: &Rt| rt.get(a) + rt.get(x));
        let o = rt.create_query(move |rt: &Rt| rt.get(a) + rt.get(b));
        bc.set(b).unwrap();

        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(o))
        };
        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(x, 1))
        };
        let read = reader.join().unwrap();
        setter.join().unwrap();
        assert!(read == 200 || read == 201, "o read {read}");

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(rt.get(b), 101, "(V) violated: B stale against x = 1");
        assert_eq!(rt.get(o), 201, "(V) violated: o stale against x = 1");
        assert_eq!(rt.stale_group_records(), Vec::<String>::new());
    });
}

/// A (Fallback) and B, a landed group made Dirty by a set of x, which B
/// reads; o reads A and then B. While A's thread holds the episode it ran
/// for o, B's thread reads B and waits on a member: once A publishes, B's
/// group verify finds every member under one record, never a group half
/// published, and both reads match a single-thread order.
fn republished_group(rt: &Rt) -> (Incr<u64>, Incr<u64>, Incr<u64>) {
    let x = rt.create_input(0_u64);
    let bc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
    let a = {
        let bc = bc.clone();
        rt.create_query_with_fallback(
            move |rt: &Rt| rt.get(*bc.get().unwrap()) + 1,
            |_c: &crate::cycle::Cycle| 100,
        )
    };
    let b = rt.create_query(move |rt: &Rt| rt.get(a) + rt.get(x));
    let o = rt.create_query(move |rt: &Rt| rt.get(a) + rt.get(b));
    bc.set(b).unwrap();
    assert_eq!(rt.get(o), 200);
    rt.set(x, 1);
    (o, a, b)
}

#[test]
fn loom_held_group_record_precedes_state() {
    run(|| {
        let orders: Vec<(SlotRead, SlotRead)> = [true, false]
            .into_iter()
            .map(|o_first| {
                let rt = Rt::new();
                let (o, _, b) = republished_group(&rt);
                let (a, b) = if o_first {
                    let first = rt.try_get(o);
                    (first, rt.try_get(b))
                } else {
                    let second = rt.try_get(b);
                    (rt.try_get(o), second)
                };
                (by_slots(a, &rt), by_slots(b, &rt))
            })
            .collect();
        let rt = Arc::new(Rt::new());
        let (o, a, b) = republished_group(&rt);
        let reads_o = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.try_get(o))
        };
        let reads_b = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.try_get(b))
        };
        let got = (
            by_slots(reads_o.join().unwrap(), &rt),
            by_slots(reads_b.join().unwrap(), &rt),
        );
        assert!(orders.contains(&got), "{got:?}, single-thread {orders:?}");
        assert_eq!(rt.stale_group_records(), Vec::<String>::new());
        assert!(
            rt.in_group_of(a.erase(), b.erase()),
            "A and B left their group"
        );

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(crate::waits::registered(), 0);
    });
}

/// c (Fallback) reads a and b, and a (while i is 0) and b (Fallback) read
/// c: a landed group. Once i is 1, a no longer reads c, so the episode at c
/// that o's read runs shrinks to {c, b} and o re-reads a, released from it,
/// giving a fresh graph's (21, 1). Counts every closure run. Without b's
/// fallback the shrunk episode panics, as on main.
fn shrunk_cut<C: crate::cells::Cells>(
    rt: &Runtime<C>,
    runs: &std::sync::Arc<std::sync::atomic::AtomicUsize>,
) -> (Incr<i64>, Incr<i64>) {
    let i = rt.create_input(0_i64);
    let ac: Arc<std::sync::OnceLock<Incr<i64>>> = Arc::new(std::sync::OnceLock::new());
    let bc: Arc<std::sync::OnceLock<Incr<i64>>> = Arc::new(std::sync::OnceLock::new());
    let run = move |runs: &std::sync::Arc<std::sync::atomic::AtomicUsize>| {
        runs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    };
    let c = {
        let (ac, bc, runs) = (ac.clone(), bc.clone(), runs.clone());
        rt.create_query_with_fallback(
            move |rt: &Runtime<C>| {
                run(&runs);
                rt.get(*ac.get().unwrap()) + rt.get(*bc.get().unwrap())
            },
            |cy: &crate::cycle::Cycle| 10 * cy.len() as i64,
        )
    };
    let a = {
        let runs = runs.clone();
        rt.create_query(move |rt: &Runtime<C>| {
            run(&runs);
            if rt.get(i) == 0 {
                rt.get(c)
            } else {
                1
            }
        })
    };
    let b = {
        let runs = runs.clone();
        rt.create_query_with_fallback(
            move |rt: &Runtime<C>| {
                run(&runs);
                rt.get(c) + 2
            },
            |cy: &crate::cycle::Cycle| 100 * cy.len() as i64,
        )
    };
    let o = {
        let runs = runs.clone();
        rt.create_query(move |rt: &Runtime<C>| {
            run(&runs);
            rt.get(c) + rt.get(a)
        })
    };
    ac.set(a).unwrap();
    bc.set(b).unwrap();
    assert_eq!((rt.get(o), rt.get(a)), (60, 30));
    rt.set(i, 1);
    (o, a)
}

#[test]
fn loom_own_landed_reread_matches_local() {
    run(|| {
        let local_runs = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let local = Runtime::<crate::cells::Local>::new();
        let (o, a) = shrunk_cut(&local, &local_runs);
        let want = (
            panic_message(|| local.get(o)),
            panic_message(|| local.get(a)),
        );
        assert_eq!(want, (Ok(21), Ok(1)), "Local differs from a fresh graph");
        let local_runs = local_runs.load(std::sync::atomic::Ordering::Relaxed);
        let local_b_first = {
            let rt = Runtime::<crate::cells::Local>::new();
            let (o, a) = shrunk_cut(&rt, &std::sync::Arc::default());
            let b = panic_message(|| rt.get(a));
            (panic_message(|| rt.get(o)), b)
        };
        assert!(
            local_b_first.0.is_ok() && local_b_first.1.is_ok(),
            "Local panics reading a first: {local_b_first:?}"
        );

        let runs = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let rt = Rt::new();
        let (o, a) = shrunk_cut(&rt, &runs);
        assert_eq!(
            (panic_message(|| rt.get(o)), panic_message(|| rt.get(a))),
            want
        );
        assert_eq!(
            runs.load(std::sync::atomic::Ordering::Relaxed),
            local_runs,
            "closure runs differ from Local's"
        );

        let rt = Arc::new(Rt::new());
        let (o, a) = shrunk_cut(&rt, &std::sync::Arc::default());
        let reads_o = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || panic_message(|| rt.get(o)))
        };
        let reads_a = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || panic_message(|| rt.get(a)))
        };
        let got = (reads_o.join().unwrap(), reads_a.join().unwrap());
        assert!(
            got == want || got == local_b_first,
            "{got:?}, Local {want:?} or {local_b_first:?}"
        );
        assert_eq!(rt.stale_group_records(), Vec::<String>::new());
        assert_eq!(rt.nodes.get(a.slot()).state(), NodeState::Clean);

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(crate::waits::registered(), 0);
    });
}

/// A (Fallback) reads B while s is 0, and B reads A. A reader enters the
/// cycle while a setter breaks it: the final values are the acyclic ones.
#[test]
fn loom_cycle_break_race() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let s = rt.create_input(0_u64);
        let bc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
        let a = {
            let bc = bc.clone();
            rt.create_query_with_fallback(
                move |rt: &Rt| {
                    if rt.get(s) == 0 {
                        rt.get(*bc.get().unwrap()) + 1
                    } else {
                        7
                    }
                },
                |_c: &crate::cycle::Cycle| 100,
            )
        };
        let b = rt.create_query(move |rt: &Rt| rt.get(a) + 1);
        bc.set(b).unwrap();

        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.get(a);
            })
        };
        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(s, 1))
        };
        reader.join().unwrap();
        setter.join().unwrap();

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(rt.get(b), 8, "(V) violated: B stale after the break");
        assert_eq!(rt.get(a), 7);
        assert!(rt.cycle_of(a).is_none());
    });
}

fn fixpoint_pair(rt: &Rt, x: Option<Incr<u64>>) -> (Incr<u64>, Incr<u64>) {
    let bc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
    let opts = || crate::cycle::QueryOpts {
        cycle: crate::cycle::CycleStrategy::Fixpoint(crate::cycle::Fixpoint::new(|| 0_u64)),
        ..Default::default()
    };
    let a = {
        let bc = bc.clone();
        rt.create_query_with(
            move |rt: &Rt| (rt.get(*bc.get().unwrap()) + 1).min(2),
            opts(),
        )
    };
    let b = rt.create_query_with(
        move |rt: &Rt| rt.get(a) + x.map_or(0, |x| rt.get(x)),
        opts(),
    );
    bc.set(b).unwrap();
    (a, b)
}

/// A ↔ B, both Fixpoint, entered at A and at B on two threads at once.
/// Their claims wait on each other, and wound-wait leaves one thread to
/// run the whole episode.
#[test]
fn loom_fixpoint_two_entries() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let (a, b) = fixpoint_pair(&rt, None);
        let first = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(a))
        };
        let second = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(b))
        };
        assert_eq!(first.join().unwrap(), 2);
        assert_eq!(second.join().unwrap(), 2);

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(crate::waits::registered(), 0);
    });
}

/// A ↔ B, both Fixpoint, with B reading x. A reader iterates the episode
/// while a setter changes x.
#[test]
fn loom_fixpoint_iterate_under_set() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(0_u64);
        let (a, b) = fixpoint_pair(&rt, Some(x));
        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.get(a);
            })
        };
        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(x, 1))
        };
        reader.join().unwrap();
        setter.join().unwrap();

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(rt.get(a), 2, "(V) violated: A");
        assert_eq!(rt.get(b), 3, "(V) violated: B stale against x = 1");
    });
}

/// A (Fallback) ↔ B, entered at A and at B on two threads: each claims
/// its entry and waits on the other's, a waits-for cycle that the second
/// registration closes and wounds.
#[test]
fn loom_cross_thread_cycle() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let bc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
        let a = {
            let bc = bc.clone();
            rt.create_query_with_fallback(
                move |rt: &Rt| rt.get(*bc.get().unwrap()) + 1,
                |_c: &crate::cycle::Cycle| 100,
            )
        };
        let b = rt.create_query(move |rt: &Rt| rt.get(a) + 1);
        bc.set(b).unwrap();
        let first = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(a))
        };
        let second = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || may_unwind(|| rt.get(b)))
        };
        assert_eq!(first.join().unwrap(), 100);
        // Entered alone at B, a `get` closes the cycle at an unseeded
        // member and panics by R3; any other order reads the landing.
        if let Some(v) = second.join().unwrap() {
            assert_eq!(v, 101);
        }

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(rt.get(b), 101);
        assert_eq!(crate::waits::registered(), 0);
    });
}

/// The cross-thread cycle with A's read of B wrapped in `catch_unwind`,
/// so a wound raised inside it is swallowed: the pop treats it as an
/// unwind, and the retry lands the same values.
#[test]
fn loom_wound_retry() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let bc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
        let a = {
            let bc = bc.clone();
            rt.create_query_with_fallback(
                move |rt: &Rt| {
                    let b = *bc.get().unwrap();
                    may_unwind(|| rt.get(b)).map_or(50, |v| v + 1)
                },
                |_c: &crate::cycle::Cycle| 100,
            )
        };
        let b = rt.create_query(move |rt: &Rt| rt.try_get(a).map_or(0, |v| v + 1));
        bc.set(b).unwrap();
        let first = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(a))
        };
        let second = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(b))
        };
        assert_eq!(first.join().unwrap(), 100);
        assert_eq!(second.join().unwrap(), 101);

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(crate::waits::registered(), 0);
    });
}

/// D reads Q, which a second thread is computing: D's thread registers
/// its wait while holding D's claim, and deregisters once Q lands, with
/// no cycle and no wound.
#[test]
fn loom_wait_deregister() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(1_u64);
        let q = rt.create_query(move |rt: &Rt| rt.get(x) + 1);
        let d = rt.create_query(move |rt: &Rt| rt.get(q) * 10);
        let computer = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(q))
        };
        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(d))
        };
        assert_eq!(computer.join().unwrap(), 2);
        assert_eq!(reader.join().unwrap(), 20);

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(crate::waits::registered(), 0);
    });
}

/// Two threads miss on one input key while a third sets another input:
/// one node is created, both get it, and the losing `init` value is
/// dropped, not kept or leaked.
#[test]
fn table_miss_race_under_set() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let table = Arc::new(rt.create_table::<u64, std::sync::Arc<u64>>());
        let x = rt.create_input(0_u64);
        let made = Arc::new(std::sync::Mutex::new(Vec::new()));
        let nodes = rt.node_count();
        let miss = |n: u64| {
            let (rt, table, made) = (Arc::clone(&rt), Arc::clone(&table), Arc::clone(&made));
            thread::spawn(move || {
                table.get_or_insert_input(&rt, 1, || {
                    let value = std::sync::Arc::new(n);
                    made.lock().unwrap().push(std::sync::Arc::downgrade(&value));
                    value
                })
            })
        };
        let (a, b) = (miss(1), miss(2));
        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(x, 1))
        };
        let (a, b) = (a.join().unwrap(), b.join().unwrap());
        setter.join().unwrap();
        assert_eq!(a, b, "both missers get the one node");
        assert_eq!(rt.node_count(), nodes + 1);
        let alive = made
            .lock()
            .unwrap()
            .iter()
            .filter(|w| w.upgrade().is_some())
            .count();
        assert_eq!(alive, 1, "the loser's value was dropped");
        rt.assert_invariant_i();
    });
}

/// A reader looks a key up while another thread's miss publishes it: it
/// sees a miss, or the published node with its value.
#[test]
fn table_hit_vs_publish() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let table = Arc::new(rt.create_table::<u64, u64>());
        let writer = {
            let (rt, table) = (Arc::clone(&rt), Arc::clone(&table));
            thread::spawn(move || table.get_or_insert_input(&rt, 7, || 70))
        };
        let reader = {
            let (rt, table) = (Arc::clone(&rt), Arc::clone(&table));
            thread::spawn(move || table.get(&rt, &7).map(|h| (h, rt.get(h))))
        };
        let h = writer.join().unwrap();
        if let Some((seen, value)) = reader.join().unwrap() {
            assert_eq!((seen, value), (h, 70));
        }
        rt.assert_invariant_i();
    });
}

/// A reader looks a key up while another thread removes a second key
/// and then inserts a third, which rehashes: the reader gets the live
/// handle, and loom checks the bucket cells for races.
#[test]
fn table_hit_vs_remove_rehash() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let table = Arc::new(rt.create_table::<u64, u64>());
        let kept = table.get_or_insert_input(&rt, 1, || 10);
        let removed = table.get_or_insert_input(&rt, 2, || 20);
        table.get_or_insert_input(&rt, 3, || 30);
        let writer = {
            let (rt, table) = (Arc::clone(&rt), Arc::clone(&table));
            thread::spawn(move || {
                table.remove(&rt, &2).unwrap();
                table.get_or_insert_input(&rt, 4, || 40)
            })
        };
        let reader = {
            let (rt, table) = (Arc::clone(&rt), Arc::clone(&table));
            thread::spawn(move || (table.get(&rt, &1), table.get(&rt, &2)))
        };
        let fourth = writer.join().unwrap();
        let (one, two) = reader.join().unwrap();
        assert_eq!(one, Some(kept));
        assert!(two.is_none() || two == Some(removed));
        assert_eq!(table.get(&rt, &2), None);
        assert_eq!(rt.get(fourth), 40);
        rt.assert_invariant_i();
    });
}

/// A strict delete of an owner races a compute of the query it owns:
/// `claim_set` finds the child `Computing` and rolls back, or claims the
/// whole set first and the compute finds its node deleted.
#[test]
fn tree_claim_vs_compute() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(2_u64);
        let owner = rt.create_input(0_u64);
        let child = rt.create_query(move |rt| rt.get(x) * 10);
        rt.adopt(owner.erase(), child.erase());
        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || may_unwind(|| rt.get(child)))
        };
        let deleted = rt.try_delete_node(owner);
        let read = reader.join().unwrap();
        assert_eq!(deleted, Ok(()));
        assert!(read.is_none() || read == Some(20));
        assert!(!rt.is_alive(owner) && !rt.is_alive(child));
        assert!(rt.is_alive(x));
        rt.assert_invariant_i();
    });
}

/// A strict delete of an owner finds one of its inputs held by a `set`:
/// every member it took rolls back, and the delete retries once the set
/// releases, or it claims first and the set finds the input deleted.
#[test]
fn tree_claim_rollback_under_set() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let owner = rt.create_input(0_u64);
        let a = rt.create_input(1_u64);
        let b = rt.create_input(2_u64);
        rt.adopt(owner.erase(), a.erase());
        rt.adopt(owner.erase(), b.erase());
        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || may_unwind(|| rt.set(b, 5)))
        };
        let deleted = rt.try_delete_node(owner);
        setter.join().unwrap();
        assert_eq!(deleted, Ok(()));
        assert!(!rt.is_alive(owner) && !rt.is_alive(a) && !rt.is_alive(b));
        rt.assert_invariant_i();
    });
}

/// A creator's rerun sweeps a child that another thread reads, and that
/// thread computes the creator too: the child's read returns its value
/// or panics as stale, the creator's runs reach its index only under its
/// claim, and the child is gone at the end.
#[test]
fn child_sweep_vs_reader() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(true);
        let creator = rt.create_query(move |rt| {
            if rt.get(x) {
                Some(rt.child_input(1_u32, || 10_u64))
            } else {
                None
            }
        });
        let child = rt.get(creator).unwrap();
        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                let read = may_unwind(|| rt.get(child));
                let seen = rt.get(creator);
                (read, seen)
            })
        };
        rt.set(x, false);
        assert_eq!(rt.get(creator), None);
        let (read, seen) = reader.join().unwrap();
        assert!(read.is_none() || read == Some(10));
        assert!(seen.is_none() || seen == Some(child));
        assert!(!rt.is_alive(child));
        rt.assert_invariant_i();
    });
}

/// Two threads run the same creator, one after a `set`: the second run
/// to claim it finds the index the first created and reuses the child,
/// whose init runs once.
#[test]
fn child_create_and_reuse() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(0_u64);
        let inits = Arc::new(Mutex::new(0_u32));
        let creator = {
            let inits = Arc::clone(&inits);
            rt.create_query(move |rt| {
                let _ = rt.get(x);
                let inits = Arc::clone(&inits);
                rt.child_input("k", move || {
                    *inits.lock().unwrap() += 1;
                    7_u64
                })
            })
        };
        let other = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(creator))
        };
        rt.set(x, 1);
        let mine = rt.get(creator);
        let theirs = other.join().unwrap();
        assert_eq!(mine, theirs);
        assert_eq!(*inits.lock().unwrap(), 1);
        assert_eq!(rt.get(mine), 7);
        rt.assert_invariant_i();
    });
}

fn replayable<T>() -> crate::cycle::QueryOpts<T> {
    crate::cycle::QueryOpts {
        replay: crate::cycle::Replay::Allowed,
        ..Default::default()
    }
}

/// T reads X at budget 1, so T's read dooms and releases X's claim before
/// T gets the placeholder, and the driver runs X as an entry while
/// another thread reads X. X lands once with its value, and T's retry
/// reads it.
#[test]
fn replay_release_vs_reader() {
    run(|| {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(1);
        let x = rt.create_query_with(|_: &Rt| 7_u64, replayable());
        let t = rt.create_query_with(move |rt: &Rt| rt.get(x) + 1, replayable());
        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(x))
        };
        assert_eq!(rt.get(t), 8);
        assert_eq!(reader.join().unwrap(), 7);

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(rt.state_of(x), NodeState::Clean);
        assert_eq!(rt.state_of(t), NodeState::Clean);
    });
}

/// A Fallback cycle F <-> X at budget 1, entered at F on two threads. On
/// each, F's read of X dooms, and X's entry reads F: a virtual hit whose
/// claim of F can find the other thread's claim, roll back and wait,
/// while that thread's F waits on X. Both reads return the landing.
#[test]
fn virtual_hit_claim_race() {
    run(|| {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(1);
        let fc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
        let x = {
            let fc = fc.clone();
            rt.create_query_with(move |rt: &Rt| rt.get(*fc.get().unwrap()) + 1, replayable())
        };
        let f = rt.create_query_with(
            move |rt: &Rt| rt.get(x) + 1,
            crate::cycle::QueryOpts {
                cycle: crate::cycle::CycleStrategy::Fallback(Box::new(|_| 100)),
                replay: crate::cycle::Replay::Allowed,
            },
        );
        fc.set(f).unwrap();
        let other = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(f))
        };
        assert_eq!(rt.get(f), 100);
        assert_eq!(other.join().unwrap(), 100);

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(rt.get(x), 101);
        assert_eq!(crate::waits::registered(), 0);
    });
}

/// A Fallback cycle F -> X -> ... -> F of `len` nodes at budget
/// `frames`, entered at F on one thread and at X on the other: each
/// thread's chain dooms on nodes the other's entries need, and its
/// replay keeps every node the doom released reserved until its outermost
/// context is removed. Each thread can wait on the other's reservation while
/// holding its own, across its context's retries; wound-wait resolves
/// any such wait, nothing stays reserved, and the pair of reads is one
/// serial order's.
fn doom_reservations_cross(len: usize, frames: u32) {
    fn shape(len: usize, frames: u32) -> (Arc<Rt>, Incr<u64>, Incr<u64>) {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(frames);
        let fc: Arc<std::sync::OnceLock<Incr<u64>>> = Arc::new(std::sync::OnceLock::new());
        let mut next = {
            let fc = fc.clone();
            rt.create_query_with(move |rt: &Rt| rt.get(*fc.get().unwrap()) + 1, replayable())
        };
        for _ in 2..len {
            let n = next;
            next = rt.create_query_with(move |rt: &Rt| rt.get(n) + 1, replayable());
        }
        let x = next;
        let f = rt.create_query_with(
            move |rt: &Rt| rt.get(x) + 1,
            crate::cycle::QueryOpts {
                cycle: crate::cycle::CycleStrategy::Fallback(Box::new(|_| 100)),
                replay: crate::cycle::Replay::Allowed,
            },
        );
        fc.set(f).unwrap();
        (rt, f, x)
    }
    run(move || {
        let serial = |f_first: bool| {
            let (rt, f, x) = shape(len, frames);
            if f_first {
                let vf = panic_message(|| rt.get(f));
                (vf, panic_message(|| rt.get(x)))
            } else {
                let vx = panic_message(|| rt.get(x));
                (panic_message(|| rt.get(f)), vx)
            }
        };
        let orders = [serial(true), serial(false)];
        let (rt, f, x) = shape(len, frames);
        let other = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || panic_message(|| rt.get(x)))
        };
        let vf = panic_message(|| rt.get(f));
        let got = (vf, other.join().unwrap());
        assert!(orders.contains(&got), "{got:?} not in {orders:?}");

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(crate::waits::registered(), 0);
        assert!(!crate::reserve::any(), "a reservation outlived both reads");
    });
}

/// A thread's wait on another thread's claim of a node it reserves
/// (`given_back_state`) ends under every interleaving with the state the
/// claim leaves: X is Dirty and reserved here while the other thread
/// either reads X, claiming it and giving it back unrun to the
/// reservation, or claims and lands it Clean as a claim that skips the
/// reservation check does.
fn give_back_wait(lands: bool) {
    run(move || {
        let rt = Arc::new(Rt::new());
        let i = rt.create_input(1_u64);
        let x = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        assert_eq!(rt.get(x), 2);
        rt.set(i, 2);
        let (slot, generation) = (x.slot(), x.generation());
        {
            let _scope = crate::reserve::Scope::new(Some(1));
            crate::reserve::note((rt.runtime_id, slot), generation);
        }
        let other = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                if lands {
                    let cell = rt.nodes.get(slot).state_cell();
                    crate::state::try_transition::<Loom>(
                        cell,
                        NodeState::Dirty,
                        NodeState::Computing,
                    )
                    .unwrap();
                    crate::state::try_transition::<Loom>(
                        cell,
                        NodeState::Computing,
                        NodeState::Clean,
                    )
                    .unwrap();
                    None
                } else {
                    Some(rt.get(x))
                }
            })
        };
        let left = rt.given_back_state(slot, generation);
        let want = if lands {
            [NodeState::Dirty, NodeState::Clean]
        } else {
            [NodeState::Dirty; 2]
        };
        assert!(left.is_some_and(|s| want.contains(&s)), "{left:?}");
        crate::reserve::end_all();
        let read = other.join().unwrap();
        assert!(read.is_none() || read == Some(3), "{read:?}");
    });
}

#[test]
fn give_back_wait_ends_on_a_give_back() {
    give_back_wait(false);
}

#[test]
fn give_back_wait_ends_on_a_landing() {
    give_back_wait(true);
}

/// [`give_back_wait`] where the other thread takes over: this thread's
/// reservation of X is held by an attempt whose discard is in flight, and
/// the other thread wounds that attempt, claims X and keeps the claim until
/// this thread's wait has returned. The wait must end once the wound takes
/// the reservation away, not wait out a claim that outlives it.
#[test]
fn give_back_wait_ends_when_a_take_over_keeps_the_claim() {
    use loom::sync::atomic::{AtomicBool, Ordering};
    run(|| {
        let rt = Arc::new(Rt::new());
        let i = rt.create_input(1_u64);
        let x = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        assert_eq!(rt.get(x), 2);
        rt.set(i, 2);
        let (slot, generation) = (x.slot(), x.generation());
        crate::attempt::begin();
        {
            let _scope = crate::reserve::Scope::new(Some(1));
            crate::reserve::note((rt.runtime_id, slot), generation);
        }
        crate::attempt::enter_flight();
        let attempt = crate::attempt::mine();
        let returned = Arc::new(AtomicBool::new(false));
        let taker = {
            let rt = Arc::clone(&rt);
            let returned = Arc::clone(&returned);
            thread::spawn(move || {
                if !attempt.wound_in_flight((rt.runtime_id, slot)) {
                    return false;
                }
                let cell = rt.nodes.get(slot).state_cell();
                crate::state::try_transition::<Loom>(cell, NodeState::Dirty, NodeState::Computing)
                    .unwrap();
                while !returned.load(Ordering::Acquire) {
                    thread::yield_now();
                }
                crate::state::try_transition::<Loom>(cell, NodeState::Computing, NodeState::Clean)
                    .unwrap();
                true
            })
        };
        let left = rt.given_back_state(slot, generation);
        returned.store(true, Ordering::Release);
        assert!(left.is_none() || left == Some(NodeState::Dirty), "{left:?}");
        let wounded = crate::attempt::leave_flight();
        crate::reserve::end_all();
        let took = taker.join().unwrap();
        assert_eq!(took, wounded);
        assert!(left.is_some() || took, "the wait ended with no take-over");
    });
}

/// [`doom_reservations_cross`] on F <-> X at budget 1: each read dooms on
/// the other thread's entry node.
#[test]
fn doom_reservations_cross_direct() {
    doom_reservations_cross(2, 1);
}

/// [`doom_reservations_cross`] on a cycle of three at budget 2: each doom
/// discards the frame below it, whose node stays reserved while the
/// context's entries run.
#[test]
fn doom_reservations_cross_held() {
    doom_reservations_cross(3, 2);
}

/// T catches its read of X, which dooms at budget 1, and joins a thread
/// that reads X before its pop. The reader meets X reserved for T's
/// retry while T's discard is in flight, takes it over, which wounds T's
/// attempt, and both reads give the unbounded run's values.
#[test]
fn doom_caught_joins_reader() {
    run(|| {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(1);
        let i = rt.create_input(3_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 10, replayable());
        let me: Arc<Mutex<Option<Arc<Rt>>>> = Arc::new(Mutex::new(Some(Arc::clone(&rt))));
        let t = {
            let me = Arc::clone(&me);
            rt.create_query_with(
                move |rt: &Rt| {
                    let v = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_or(0);
                    let other = me.lock().unwrap().clone().unwrap();
                    v + thread::spawn(move || other.get(x)).join().unwrap()
                },
                replayable(),
            )
        };
        assert_eq!(rt.get(t), 28);
        *me.lock().unwrap() = None;
        rt.assert_invariant_i();
        assert_eq!(crate::waits::registered(), 0);
        assert!(!crate::reserve::any(), "a reservation outlived the read");
    });
}

/// T reads X, which dooms at budget 1, while another thread reads X. The
/// reader can meet X's reservation while T's discard is in flight, before
/// the engine's pop catches it, and take it over, which restarts T's
/// read; or the pop ends the flight first and the reader waits for T's
/// context. Either way both reads give X's value and nothing stays
/// reserved.
#[test]
fn doom_in_flight_race() {
    run(|| {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(1);
        let i = rt.create_input(3_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 10, replayable());
        let t = rt.create_query_with(move |rt: &Rt| rt.get(x) + 100, replayable());
        let other = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(x))
        };
        assert_eq!(rt.get(t), 114);
        assert_eq!(other.join().unwrap(), 14);
        rt.assert_invariant_i();
        assert_eq!(crate::waits::registered(), 0);
        assert!(!crate::reserve::any(), "a reservation outlived both reads");
    });
}

/// T reads X, which dooms at budget 1, so T takes the placeholder and runs
/// on with its doom's flight open, into a join of a thread that reads X.
/// The reader meets X reserved for T's retry, takes it over and computes
/// X; T's run ends as the wound its read retries from, and both reads give
/// the unbounded run's values.
#[test]
fn doom_lazy_joins_reader() {
    run(|| {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(1);
        let i = rt.create_input(3_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 10, replayable());
        let me: Arc<Mutex<Option<Arc<Rt>>>> = Arc::new(Mutex::new(Some(Arc::clone(&rt))));
        let t = {
            let me = Arc::clone(&me);
            rt.create_query_with(
                move |rt: &Rt| {
                    let v = rt.get(x);
                    let other = me.lock().unwrap().clone().unwrap();
                    v + thread::spawn(move || other.get(x)).join().unwrap()
                },
                replayable(),
            )
        };
        assert_eq!(rt.get(t), 28);
        *me.lock().unwrap() = None;
        rt.assert_invariant_i();
        assert_eq!(crate::waits::registered(), 0);
        assert!(!crate::reserve::any(), "a reservation outlived the read");
    });
}

/// T reads X, which dooms at budget 1, then Z, while another thread reads
/// X. T runs on past X's placeholder with its flight open and raises the
/// discard at its read of Z; the reader may take X over in that window or
/// wait for T's context. Both reads give the unbounded run's values and
/// nothing stays reserved.
#[test]
fn doom_lazy_reads_again() {
    run(|| {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(1);
        let i = rt.create_input(3_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 10, replayable());
        let z = rt.create_query_with(move |rt: &Rt| rt.get(i) + 20, replayable());
        let t = rt.create_query_with(move |rt: &Rt| rt.get(x) + rt.get(z) + 100, replayable());
        let other = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(x))
        };
        assert_eq!(rt.get(t), 137);
        assert_eq!(other.join().unwrap(), 14);
        rt.assert_invariant_i();
        assert_eq!(crate::waits::registered(), 0);
        assert!(!crate::reserve::any(), "a reservation outlived both reads");
    });
}

/// The orders a lazy doom model's executions reached, kept outside the
/// model in plain atomics and checked once every execution has run.
struct Reached {
    take_over: std::sync::atomic::AtomicBool,
    no_take_over: std::sync::atomic::AtomicBool,
    /// The other thread waited on X's reservation and did not take over.
    waited_untaken: std::sync::atomic::AtomicBool,
}

impl Reached {
    const fn new() -> Self {
        use std::sync::atomic::AtomicBool;
        Self {
            take_over: AtomicBool::new(false),
            no_take_over: AtomicBool::new(false),
            waited_untaken: AtomicBool::new(false),
        }
    }

    fn note(&self, (took, waited): (bool, bool)) {
        use std::sync::atomic::Ordering::Relaxed;
        if took {
            self.take_over.store(true, Relaxed);
        } else {
            self.no_take_over.store(true, Relaxed);
            if waited {
                self.waited_untaken.store(true, Relaxed);
            }
        }
    }

    fn assert_both(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        assert!(
            self.take_over.load(Relaxed),
            "no execution took the doom over"
        );
        assert!(
            self.no_take_over.load(Relaxed),
            "every execution took the doom over"
        );
    }

    fn assert_waited_untaken(&self) {
        assert!(
            self.waited_untaken
                .load(std::sync::atomic::Ordering::Relaxed),
            "no execution waited on X without taking it over"
        );
    }
}

/// A lazy doom world at budget 1: T's closure is `t_run` over X (= Y + 10,
/// Y = I + 1, I = 3) and Z (= I + 20). T's runs push the attempt epoch
/// they ran in, which moves on only when the outermost entry retries
/// after a wound.
struct LazyWorld {
    rt: Arc<Rt>,
    x: Incr<u64>,
    t: Incr<u64>,
    epochs: Arc<std::sync::Mutex<Vec<u32>>>,
}

impl LazyWorld {
    fn new(t_run: impl Fn(&Rt, Incr<u64>, Incr<u64>) -> u64 + Send + Sync + 'static) -> Self {
        // Shared by every model in the process: run one model per process.
        crate::attempt::TAKE_OVERS.store(0, std::sync::atomic::Ordering::Relaxed);
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(1);
        let i = rt.create_input(3_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 10, replayable());
        let z = rt.create_query_with(move |rt: &Rt| rt.get(i) + 20, replayable());
        let epochs = Arc::new(std::sync::Mutex::new(Vec::new()));
        let t = {
            let epochs = Arc::clone(&epochs);
            rt.create_query_with(
                move |rt: &Rt| {
                    epochs.lock().unwrap().push(crate::attempt::epoch());
                    t_run(rt, x, z)
                },
                replayable(),
            )
        };
        Self { rt, x, t, epochs }
    }

    /// Reads T against another thread's read of X; checks both values, that
    /// T's read retried exactly when a waiter took the doom over.
    /// Returns whether one did, and whether the other thread waited on X.
    fn race(&self, want: u64, user_panics: bool) -> (bool, bool) {
        let other = {
            let (rt, x) = (Arc::clone(&self.rt), self.x);
            thread::spawn(move || rt.get(x))
        };
        EXPECTING_UNWIND.with(|e| e.set(user_panics));
        let got = self.rt.get(self.t);
        EXPECTING_UNWIND.with(|e| e.set(false));
        assert_eq!(got, want);
        assert_eq!(other.join().unwrap(), 14);
        let took = crate::attempt::TAKE_OVERS.load(std::sync::atomic::Ordering::Relaxed);
        let epochs = self.epochs.lock().unwrap().clone();
        let retried = epochs.first() != epochs.last();
        assert!(took <= 1, "{took} take-overs of one doom");
        assert_eq!(
            retried,
            took == 1,
            "T's read retried {retried} with {took} take-overs (epochs {epochs:?})"
        );
        self.rt.assert_invariant_i();
        assert_eq!(crate::waits::registered(), 0);
        assert!(!crate::reserve::any(), "a reservation outlived both reads");
        let waited = self
            .rt
            .reservation_waits
            .load(std::sync::atomic::Ordering::Acquire)
            != 0;
        (took == 1, waited)
    }
}

static END_RACE: Reached = Reached::new();

/// T reads X, which dooms at budget 1, takes the placeholder and returns
/// at once, so its run's end closes the lazy doom's flight while another
/// thread waits on X's reservation. The waiter's wound and the end's
/// `leave_flight` race in both orders, each reached with the waiter
/// waiting: T's read retries exactly when the wound came first, and both
/// reads give the unbounded run's values.
#[test]
fn doom_lazy_end_races_take_over() {
    run(|| {
        let world = LazyWorld::new(|rt, x, _| rt.get(x) + 100);
        END_RACE.note(world.race(114, false));
    });
    END_RACE.assert_both();
    END_RACE.assert_waited_untaken();
}

static RAISE_RACE: Reached = Reached::new();

/// T reads X, which dooms at budget 1, runs on past the placeholder and
/// reads Z, which raises the doom's discard, while another thread waits on
/// X's reservation. A take-over in any order around that raise leaves T's
/// read retried exactly when it happened, and both reads give the
/// unbounded run's values. `settle_discard`'s wound only stops the attempt
/// sooner: the sticky wound turns the discard into a wound at its catch
/// anyway, so its red is the unit test
/// `a_discard_the_engine_raises_again_is_back_in_flight`.
#[test]
fn doom_lazy_wound_before_raise() {
    run(|| {
        let world = LazyWorld::new(|rt, x, z| rt.get(x) + rt.get(z) + 100);
        RAISE_RACE.note(world.race(137, false));
    });
    RAISE_RACE.assert_both();
}

/// On drop, reads W from the frame it runs in, under a catch, and notes
/// whether the read unwound.
struct ReadOnDrop<'a> {
    rt: &'a Rt,
    w: Incr<u64>,
    unwound: &'a std::sync::atomic::AtomicBool,
}

impl Drop for ReadOnDrop<'_> {
    fn drop(&mut self) {
        if catch_unwind(AssertUnwindSafe(|| self.rt.get(self.w))).is_err() {
            self.unwound
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

static PANIC_RACE: Reached = Reached::new();

/// T reads X, which dooms at budget 1, and on the placeholder panics in
/// user code while another thread waits on X's reservation; a destructor
/// in that panic reads W. The destructor's read never unwinds and the
/// waiter can still take X over; T's read retries exactly when it does,
/// and both reads give the unbounded run's values. A doomed frame takes
/// placeholders while panicking with or without `KeptUnwind` keeping the
/// lazy doom, so that guard's red is the unit test
/// `a_destructors_discard_leaves_a_lazy_doom_in_flight`.
#[test]
fn doom_lazy_take_over_during_user_panic() {
    run(|| {
        let unwound = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let world = {
            let unwound = Arc::clone(&unwound);
            LazyWorld::new(move |rt, x, z| {
                let v = rt.get(x);
                if v == 0 {
                    let _read = ReadOnDrop {
                        rt,
                        w: z,
                        unwound: &unwound,
                    };
                    panic!("a doomed run panics past its placeholder");
                }
                v + 100
            })
        };
        PANIC_RACE.note(world.race(114, true));
        assert!(
            !unwound.load(std::sync::atomic::Ordering::Relaxed),
            "the destructor's read raised the lazy doom's discard"
        );
    });
    PANIC_RACE.assert_both();
}

/// F, Clean at s = 0, reads a New query N once s is set, so at budget 1
/// its run claimed from Dirty dooms on N and is discarded, while another
/// thread sets s again. The discard landing keeps that set's dirtying:
/// the read returns either revision's value and the next read the last.
#[test]
fn discard_under_set() {
    run(|| {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        let s = rt.create_input(0_u64);
        let n = rt.create_query_with(|_: &Rt| 1_u64, replayable());
        let f = rt.create_query_with(
            move |rt: &Rt| match rt.get(s) {
                0 => 0,
                v => v * 10 + rt.get(n),
            },
            replayable(),
        );
        assert_eq!(rt.get(f), 0);
        rt.set(s, 1);
        rt.set_max_frames(1);
        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(s, 2))
        };
        let read = rt.get(f);
        setter.join().unwrap();
        assert!(read == 11 || read == 21, "{read}");

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        assert_eq!(rt.get(f), 21, "(V) violated: F stale against s = 2");
    });
}

/// A read's panic message, its panic expected.
fn panic_message<R>(f: impl FnOnce() -> R) -> Result<R, String> {
    EXPECTING_UNWIND.with(|e| e.set(true));
    let out = catch_unwind(AssertUnwindSafe(f)).map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    });
    EXPECTING_UNWIND.with(|e| e.set(false));
    out
}

/// X reads Y, which panics "y boom", at budget 1, and two threads read X.
/// A thread whose read of Y dooms X runs Y as an entry, which lands Failed
/// with the payload held for X's retry, while the other thread may compute
/// X, read Y Failed and land X Failed, letting that failure out. The
/// record published before Y's landing is what the other thread's Failed
/// read finds, so the retry takes the payload: one read gets "y boom" and
/// the other "last compute panicked", as in either serial order.
#[test]
fn vframe_failed_provenance() {
    run(|| {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(1);
        let y = rt.create_query_with(|_: &Rt| -> u64 { panic!("y boom") }, replayable());
        let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 1, replayable());
        let other = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || panic_message(|| rt.get(x)))
        };
        let mine = panic_message(|| rt.get(x));
        let other = other.join().unwrap();
        let boom = Err("y boom".to_string());
        let failed = |r: &Result<u64, String>| {
            r.as_ref()
                .is_err_and(|m| m.contains("last compute panicked"))
        };
        assert!(
            (mine == boom && failed(&other)) || (other == boom && failed(&mine)),
            "no serial order: {mine:?} {other:?}"
        );
        assert_eq!(rt.provenance_records(), 0);
        rt.assert_invariant_i();
    });
}

/// As `vframe_failed_provenance`, but once s is 1 Y returns 7 and X
/// panics "x fresh" on it, and the other thread reads X, sets s and reads
/// X again, so X's own panic can reland X Failed over the landing that let
/// the held payload's failure out before this thread's retry reads X. X
/// reads only Y, so the set's walk reaches X once. The three outcomes must
/// be those of a serial order of this thread's read among the other's.
#[test]
fn vframe_failed_provenance_relanded() {
    run(|| {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(1);
        let s = rt.create_input(0_u64);
        let y = rt.create_query_with(
            move |rt: &Rt| -> u64 {
                if rt.get(s) == 1 {
                    return 7;
                }
                panic!("y boom")
            },
            replayable(),
        );
        let x = rt.create_query_with(
            move |rt: &Rt| {
                let v = rt.get(y);
                if v == 7 {
                    panic!("x fresh")
                }
                v + 1
            },
            replayable(),
        );
        let other = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                let before = panic_message(|| rt.get(x));
                rt.set(s, 1);
                (before, panic_message(|| rt.get(x)))
            })
        };
        let mine = panic_message(|| rt.get(x));
        let (before, after) = other.join().unwrap();
        let kind = |r: &Result<u64, String>| match r {
            Err(m) if m == "y boom" => 'Y',
            Err(m) if m == "x fresh" => 'F',
            Err(m) if m.contains("last compute panicked") => 'L',
            _ => '?',
        };
        let got = [kind(&mine), kind(&before), kind(&after)];
        assert!(
            [['Y', 'L', 'F'], ['L', 'Y', 'F'], ['F', 'Y', 'L']].contains(&got),
            "no serial order: {mine:?} {before:?} {after:?}"
        );
        assert_eq!(rt.provenance_records(), 0);
        rt.assert_invariant_i();
    });
}

/// x -> Q, both Clean, with x watched. One thread sets x while another
/// watches Q and drains. The walk's flip and `watch`'s state load are
/// ordered only through `inner`, with no SeqCst: whichever section comes
/// second sees the other's store, so Q is reported, and x once.
#[test]
fn drain_vs_set() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(1_u64);
        let q = rt.create_query(move |rt: &Rt| rt.get(x) + 1);
        assert_eq!(rt.get(q), 2);
        rt.watch(x);

        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(x, 2))
        };
        let drained = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.watch(q);
                rt.drain_dirtied_watched()
            })
        };
        setter.join().unwrap();
        let mut seen = drained.join().unwrap();
        let rest = rt.drain_dirtied_watched();
        for e in &rest {
            assert!(!seen.contains(e) || *e == q.erase(), "{e:?} drained twice");
        }
        seen.extend(rest);

        assert_eq!(seen.iter().filter(|&&e| e == x.erase()).count(), 1);
        let reports = seen.iter().filter(|&&e| e == q.erase()).count();
        assert!(reports >= 1, "Q's dirtying was never reported");
        assert!(reports <= 2, "Q was reported {reports} times");
        assert!(rt.drain_dirtied_watched().is_empty());
        assert_eq!(rt.get(q), 3);
        rt.assert_invariant_i();
    });
}

/// A contained panic payload that counts the live copies.
struct Boom(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl Drop for Boom {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// As `unread_contained_panic_is_released`, with the set racing the read:
/// x's walk contains F's panic and holds it in a ticket after F's Failed
/// store, while a set of j flips F from Failed to Dirty. The flip must
/// release F's ticket in every order, so once F recomputes Clean no
/// payload is held.
#[test]
fn flip_racing_a_contained_hold_releases_the_ticket() {
    run(|| {
        let live = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let rt = Arc::new(Rt::new());
        let sel = rt.create_input(0_u64);
        let j = rt.create_input(1_u64);
        let f = {
            let live = live.clone();
            rt.create_query(move |rt: &Rt| {
                let v = rt.get(j);
                if v == 0 {
                    live.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    std::panic::panic_any(Boom(live.clone()));
                }
                v
            })
        };
        let x = rt.create_query(move |rt: &Rt| if rt.get(sel) == 0 { rt.get(f) } else { 100 });
        assert_eq!(rt.get(x), 1);
        rt.set(j, 0);
        rt.set(sel, 1);

        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(j, 3))
        };
        assert_eq!(may_unwind(|| rt.get(x)), Some(100));
        setter.join().unwrap();

        assert_eq!(rt.get(f), 3);
        assert_eq!(rt.get(x), 100);
        assert_eq!(
            live.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "F is Clean but its contained payload is still held"
        );
        rt.assert_invariant_i();
    });
}
/// A Clean check and a read made against a runtime's `Changes` while a
/// change stores a value and then dirties it, as a set's swap and walk do.
/// A count found before the change moves once the reader has seen any part
/// of it, and one found after it ended comes with all of it, so a stash
/// adopted while its count holds read what a read at the adoption would.
#[test]
fn loom_checked_read_vs_change() {
    loom::model(|| {
        let changes = Arc::new(Changes::new());
        let value = Arc::new(AtomicU64::new(0));
        let dirty = Arc::new(AtomicU64::new(0));
        let writer = {
            let (changes, value, dirty) = (changes.clone(), value.clone(), dirty.clone());
            thread::spawn(move || {
                let _change = changes.change(false);
                value.store(1, Ordering::Release);
                dirty.store(1, Ordering::Release);
            })
        };
        if let Some(seen) = changes.count() {
            let clean = dirty.load(Ordering::Acquire) == 0;
            let read = value.load(Ordering::Acquire);
            let later = (dirty.load(Ordering::Acquire), value.load(Ordering::Acquire));
            if seen.before_any() {
                if !clean || read == 1 || later != (0, 0) {
                    assert!(!seen.holds());
                }
            } else {
                assert!(!clean && read == 1);
            }
        }
        writer.join().unwrap();
    });
}

/// A speculative and a non-speculative change race a checked read: once the
/// reader has seen any part of the non-speculative one, or it has ended, a
/// count found before both no longer passes as moved by speculative runs
/// only.
#[test]
fn loom_speculative_changes_vs_check() {
    loom::model(|| {
        let changes = Arc::new(Changes::new());
        let real = Arc::new(AtomicU64::new(0));
        let writers = [true, false].map(|speculative| {
            let (changes, real) = (changes.clone(), real.clone());
            thread::spawn(move || {
                let _change = changes.change(speculative);
                if !speculative {
                    real.store(1, Ordering::Release);
                }
            })
        });
        let seen = changes.count();
        if let Some(seen) = &seen {
            if seen.before_any() && real.load(Ordering::Acquire) == 1 {
                assert!(!seen.only_speculative_since());
            }
        }
        for w in writers {
            w.join().unwrap();
        }
        if let Some(seen) = seen.filter(Count::before_any) {
            assert!(!seen.only_speculative_since());
        }
    });
}

/// [`loom_speculative_changes_vs_check`]'s other side: with only a
/// speculative change, a count found before it passes once it has ended.
#[test]
fn loom_speculative_change_passes_check() {
    loom::model(|| {
        let changes = Arc::new(Changes::new());
        let writer = {
            let changes = changes.clone();
            thread::spawn(move || drop(changes.change(true)))
        };
        let seen = changes.count();
        writer.join().unwrap();
        if let Some(seen) = seen.filter(Count::before_any) {
            assert!(seen.only_speculative_since());
        }
    });
}

/// Which outcomes a hand-over model's executions reached.
struct HandOvers {
    taken: std::sync::atomic::AtomicBool,
    kept: std::sync::atomic::AtomicBool,
}

impl HandOvers {
    const fn new() -> Self {
        Self {
            taken: std::sync::atomic::AtomicBool::new(false),
            kept: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn note(&self, taken: bool) {
        let flag = if taken { &self.taken } else { &self.kept };
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn assert_both(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        assert!(
            self.taken.load(Relaxed),
            "no execution handed the landing over"
        );
        assert!(
            self.kept.load(Relaxed),
            "every execution handed the landing over"
        );
    }
}

/// A waiter's claim wait on the held landing of `CELL`, for as many checks
/// as a stuck owner needs, under loom's `STUCK_AFTER`.
fn wait_to_hand_over(rt: crate::handle::RuntimeId) -> bool {
    let mut patience = crate::attempt::Patience::default();
    for _ in 0..4 {
        if crate::hold::hand_over(HELD_CELL, (rt, HELD_SLOT), &mut patience).is_some() {
            return true;
        }
        thread::yield_now();
    }
    false
}

const HELD_CELL: usize = 0x1000;
const HELD_SLOT: u32 = 7;

/// The owner opens and lands a held entry and makes no more progress, as
/// one parked in a join, then publishes.
fn hold_one(rt: crate::handle::RuntimeId) {
    crate::attempt::begin();
    let own = crate::hold::open(HELD_CELL, rt, HELD_SLOT, 0, true, None, None);
    assert!(own.is_none());
    crate::hold::land(HELD_CELL, NodeState::Clean, NodeState::Clean, false);
}

static PUBLISH_RACE: HandOvers = HandOvers::new();

/// A waiter on a held landing races the owner's publication of it
/// (`hold::hand_over` against `hold::take`): the entry is published or
/// taken, never both and never neither, and the owner's take reports the
/// wound exactly when the waiter took it, so a publication never rests on
/// a landing a waiter retracted.
#[test]
fn hand_over_races_publication() {
    run(|| {
        let rt = crate::handle::RuntimeId::allocate();
        hold_one(rt);
        let waiter = thread::spawn(move || wait_to_hand_over(rt));
        let taken = crate::hold::take(rt);
        let handed = waiter.join().unwrap();
        assert_eq!(taken.wounded, handed);
        assert_eq!(taken.entries.len(), usize::from(!handed));
        assert!(!crate::hold::active());
        PUBLISH_RACE.note(handed);
    });
    PUBLISH_RACE.assert_both();
}

static RECLAIM_RACE: HandOvers = HandOvers::new();

/// A waiter on a held landing races the owner claiming it again
/// (`hold::reclaim`): an entry the owner computes again is never taken,
/// and one taken is gone from the owner's view.
#[test]
fn hand_over_races_reclaim() {
    run(|| {
        let rt = crate::handle::RuntimeId::allocate();
        hold_one(rt);
        let waiter = thread::spawn(move || wait_to_hand_over(rt));
        let reclaimed = crate::hold::reclaim(HELD_CELL, NodeState::Clean);
        let handed = waiter.join().unwrap();
        assert!(!(reclaimed && handed), "a reclaimed entry was taken");
        assert_eq!(crate::hold::holds(HELD_CELL), !handed);
        let taken = crate::hold::take(rt);
        assert_eq!(taken.wounded, handed);
        RECLAIM_RACE.note(handed);
    });
    RECLAIM_RACE.assert_both();
}

static PROGRESS_RACE: HandOvers = HandOvers::new();

/// A waiter's patience against a holder that keeps landing: each landing
/// is engine progress and restarts it, so only a holder that stops is
/// wounded, and the owner learns of it at the take that ends its hold.
#[test]
fn hand_over_waits_out_a_holder_that_progresses() {
    run(|| {
        let rt = crate::handle::RuntimeId::allocate();
        hold_one(rt);
        let waiter = thread::spawn(move || wait_to_hand_over(rt));
        crate::hold::land(HELD_CELL, NodeState::Clean, NodeState::Clean, false);
        let wounded = crate::attempt::hold_wounded();
        let handed = waiter.join().unwrap();
        let taken = crate::hold::take(rt);
        assert_eq!(taken.wounded, handed);
        assert!(!wounded || handed);
        PROGRESS_RACE.note(handed);
    });
    PROGRESS_RACE.assert_both();
}

/// A read `r` that, before reading `c`, joins a thread reading `n`, where
/// `n` and `c` read each other with `try_get` below `r`. The unbounded run
/// finishes: the joined reader computes `n` itself, as `r` has not read
/// below itself yet. With
/// `reaches_r`, `n` also reads `r`, and the joined reader waits on `r`'s
/// claim while `r` waits on the join, a deadlock in any engine.
fn cycle_join(reaches_r: bool) {
    run(move || {
        type Handles = (std::sync::Weak<Rt>, Incr<u64>, Incr<u64>, Incr<u64>);
        let cell: std::sync::Arc<std::sync::OnceLock<Handles>> = std::sync::Arc::default();
        let rt = std::sync::Arc::new(Rt::new());
        let i = rt.create_input(1_u64);
        let n = {
            let cell = std::sync::Arc::clone(&cell);
            rt.create_query_with(
                move |rt: &Rt| {
                    let (_, _, c, r) = cell.get().unwrap().clone();
                    let up = if reaches_r {
                        rt.try_get(r).unwrap_or(0)
                    } else {
                        0
                    };
                    rt.try_get(c).unwrap_or(0) + up + rt.get(i)
                },
                replayable(),
            )
        };
        let c = rt.create_query_with(
            move |rt: &Rt| rt.try_get(n).map_or(0, |v| v + 1),
            replayable(),
        );
        let r = {
            let cell = std::sync::Arc::clone(&cell);
            rt.create_query_with(
                move |rt: &Rt| {
                    let v = rt.get(i);
                    if v == 2 {
                        let (w, n, _, _) = cell.get().unwrap().clone();
                        let rt = w.upgrade().unwrap();
                        let _ = thread::spawn(move || rt.try_get(n)).join().unwrap();
                    }
                    rt.try_get(c).unwrap_or(0) + v
                },
                replayable(),
            )
        };
        assert!(cell.set((std::sync::Arc::downgrade(&rt), n, c, r)).is_ok());
        let _ = rt.try_get(r);
        rt.set(i, 2);
        let _ = rt.try_get(r);
    });
}

/// A held landing of `n` would make the joined reader wait on a holder
/// blocked in the join; the hand-over gives the reader the landing, as the
/// unbounded run lets it compute `n` itself.
#[test]
fn a_read_joining_a_reader_of_a_cycle_below_it_finishes() {
    cycle_join(false);
}

/// Not a CI model: loom reports the branch limit, because the unbounded run
/// deadlocks too. Kept as a probe of the deadlocking shape.
#[test]
#[ignore]
fn a_read_joining_a_reader_of_a_cycle_through_it_deadlocks() {
    cycle_join(true);
}
