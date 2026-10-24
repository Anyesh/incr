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

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Once;

use loom::sync::{Arc, Mutex};
use loom::thread;

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
            let expected = EXPECTING_UNWIND.try_with(Cell::get).unwrap_or(false);
            if !expected {
                default(info);
            }
        }));
    });
}

pub(crate) fn run(f: impl Fn() + Sync + Send + 'static) {
    silence_expected_panics();
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

#[test]
#[ignore = "T8-44e3-clean-landing-variant"]
fn loom_released_group_members_stay_grouped_until_the_read_ends() {
    run(|| {
        let orders: Vec<(
            Result<u64, crate::cycle::Cycle>,
            Result<u64, crate::cycle::Cycle>,
        )> = [true, false]
            .into_iter()
            .map(|r_first| {
                let rt = Rt::new();
                let (r, y) = released_group(&rt);
                if r_first {
                    let a = rt.try_get(r);
                    (a, rt.try_get(y))
                } else {
                    let b = rt.try_get(y);
                    (rt.try_get(r), b)
                }
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
        let got = (reads_r.join().unwrap(), reads_y.join().unwrap());
        assert!(orders.contains(&got), "{got:?}, single-thread {orders:?}");

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
fn panic_message(f: impl FnOnce() -> u64) -> Result<u64, String> {
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
