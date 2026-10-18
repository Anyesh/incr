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
use crate::runtime::Runtime;
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

fn run(f: impl Fn() + Sync + Send + 'static) {
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
