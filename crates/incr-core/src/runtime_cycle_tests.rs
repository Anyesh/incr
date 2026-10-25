//! Cycles as values: fallbacks, fixpoints, unseeded cycles that land
//! `Cyclic`, validation, restarts, group verification and the unwinds
//! that cycles raise under `get`. Each case runs under `Local` and
//! `Shared`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{Cycle, CycleStrategy, Fixpoint, QueryOpts};
use crate::handle::Incr;
use crate::state::NodeState;

macro_rules! both {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(
                #[test]
                #[cfg_attr(miri, ignore)]
                fn $name() {
                    super::$name::<super::Local>();
                }
            )*
        }
        mod shared {
            $(
                #[test]
                #[cfg_attr(miri, ignore)]
                fn $name() {
                    super::$name::<super::Shared>();
                }
            )*
        }
    };
}

fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = err.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        String::from("<non-string panic>")
    }
}

/// A handle cell, so a closure can read a query created after it.
fn cell<T>() -> Arc<OnceLock<Incr<T>>> {
    Arc::new(OnceLock::new())
}

fn slots(cycle: &Cycle) -> Vec<u32> {
    cycle.participants().iter().map(|p| p.slot()).collect()
}

fn fixpoint<T: crate::Value>(fp: Fixpoint<T>) -> QueryOpts<T> {
    QueryOpts {
        cycle: CycleStrategy::Fixpoint(fp),
        ..Default::default()
    }
}

/// A query that reads itself, with a fallback, lands the fallback.
fn self_cycle_fallback<C: Cells>() {
    let rt = Runtime::<C>::new();
    let me = cell::<i64>();
    let q = {
        let me = me.clone();
        rt.create_query_with_fallback(
            move |rt| rt.get(*me.get().unwrap()) + 1,
            |c: &Cycle| 100 + c.len() as i64,
        )
    };
    me.set(q).unwrap();
    assert_eq!(rt.get(q), 101);
    assert_eq!(rt.try_get(q), Ok(101));
    assert_eq!(rt.state_of(q), NodeState::Clean);
    assert_eq!(slots(&rt.cycle_of(q).unwrap()), vec![q.slot()]);
    rt.assert_invariant_i();
}

/// A query that reads itself iterates to its fixed point.
fn self_cycle_fixpoint<C: Cells>() {
    let rt = Runtime::<C>::new();
    let me = cell::<i64>();
    let runs = Arc::new(AtomicU32::new(0));
    let q = {
        let (me, runs) = (me.clone(), runs.clone());
        rt.create_query_with(
            move |rt| {
                runs.fetch_add(1, Ordering::Relaxed);
                (rt.get(*me.get().unwrap()) + 1).min(10)
            },
            fixpoint(Fixpoint::new(|| 0_i64)),
        )
    };
    me.set(q).unwrap();
    assert_eq!(rt.get(q), 10);
    assert!(runs.load(Ordering::Relaxed) <= 13);
    rt.assert_invariant_i();
}

/// A query with no strategy that reads itself through `try_get` lands
/// `Cyclic`, and `get` of it panics naming the cycle.
fn self_cycle_unseeded<C: Cells>() {
    let rt = Runtime::<C>::new();
    let me = cell::<i64>();
    let q = {
        let me = me.clone();
        rt.create_query(move |rt| rt.try_get(*me.get().unwrap()).map_or(-1, |v| v + 1))
    };
    me.set(q).unwrap();
    let err = rt.try_get(q).unwrap_err();
    assert_eq!(slots(&err), vec![q.slot()]);
    assert_eq!(rt.state_of(q), NodeState::Cyclic);
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(q))).unwrap_err());
    assert!(msg.contains("dependency cycle"), "{msg}");
    assert_eq!(
        crate::take_last_cycle().map(|c| slots(&c)),
        Some(vec![q.slot()])
    );
    rt.assert_invariant_i();
}

/// a (Fallback) and b (unseeded) read each other: a lands its fallback
/// and b its value against it.
fn two_cycle_fallback<C: Cells>() {
    let rt = Runtime::<C>::new();
    let bc = cell::<i64>();
    let a = {
        let bc = bc.clone();
        rt.create_query_with_fallback(move |rt| rt.get(*bc.get().unwrap()) + 1, |_c: &Cycle| 100)
    };
    let b = rt.create_query(move |rt| rt.get(a) * 2);
    bc.set(b).unwrap();
    assert_eq!(rt.get(a), 100);
    assert_eq!(rt.get(b), 200);
    let cycle = rt.cycle_of(b).unwrap();
    assert_eq!(slots(&cycle), vec![a.slot(), b.slot()]);
    assert_eq!(cycle.head(), a.erase());
    rt.assert_invariant_i();
}

/// The same cycle entered at the unseeded member: `get` panics there
/// (R3 during discovery), and `try_get` lands the same values.
fn two_cycle_entered_at_unseeded<C: Cells>() {
    let build = |rt: &Runtime<C>| {
        let bc = cell::<i64>();
        let a = {
            let bc = bc.clone();
            rt.create_query_with_fallback(
                move |rt| rt.try_get(*bc.get().unwrap()).map_or(-7, |v| v + 1),
                |_c: &Cycle| 100,
            )
        };
        let b = rt.create_query(move |rt| rt.try_get(a).map_or(-9, |v| v * 2));
        bc.set(b).unwrap();
        (a, b)
    };
    let rt = Runtime::<C>::new();
    let (a, b) = build(&rt);
    assert_eq!(rt.try_get(b), Ok(200));
    assert_eq!(rt.get(a), 100);

    let rt = Runtime::<C>::new();
    let bc = cell::<i64>();
    let a = {
        let bc = bc.clone();
        rt.create_query_with_fallback(move |rt| rt.get(*bc.get().unwrap()) + 1, |_c: &Cycle| 100)
    };
    let b = rt.create_query(move |rt| rt.get(a) * 2);
    bc.set(b).unwrap();
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(b))).unwrap_err());
    assert!(msg.contains("dependency cycle"), "{msg}");
    // Every frame the cycle unwound took the discard landing, so a read
    // that enters at the strategy member lands the group.
    assert_eq!(rt.state_of(b), NodeState::New);
    assert_eq!(rt.get(a), 100);
    assert_eq!(rt.get(b), 200);
    rt.assert_invariant_i();
}

/// Two Fixpoint members converge together.
fn two_cycle_fixpoint<C: Cells>() {
    let rt = Runtime::<C>::new();
    let bc = cell::<i64>();
    let a = {
        let bc = bc.clone();
        rt.create_query_with(
            move |rt| (rt.get(*bc.get().unwrap()) + 1).min(5),
            fixpoint(Fixpoint::new(|| 0_i64)),
        )
    };
    let b = rt.create_query_with(move |rt| rt.get(a), fixpoint(Fixpoint::new(|| 0_i64)));
    bc.set(b).unwrap();
    assert_eq!(rt.get(a), 5);
    assert_eq!(rt.get(b), 5);
    rt.assert_invariant_i();
}

/// An oscillating member stops at its limit with `on_limit`'s value.
fn oscillation_reaches_the_limit<C: Cells>() {
    let rt = Runtime::<C>::new();
    let me = cell::<i64>();
    let runs = Arc::new(AtomicU32::new(0));
    let q = {
        let (me, runs) = (me.clone(), runs.clone());
        rt.create_query_with(
            move |rt| {
                runs.fetch_add(1, Ordering::Relaxed);
                1 - rt.get(*me.get().unwrap())
            },
            fixpoint(
                Fixpoint::new(|| 0_i64)
                    .max_iterations(10)
                    .on_limit(|c: &Cycle, last| 1000 * c.len() as i64 + last),
            ),
        )
    };
    me.set(q).unwrap();
    // s_10 = 0 after ten flips from 0.
    assert_eq!(rt.get(q), 1000);
    assert!(
        runs.load(Ordering::Relaxed) <= 12,
        "{} runs",
        runs.load(Ordering::Relaxed)
    );
}

/// A tolerance test ends Newton's iteration for the square root of 2.
fn tolerance_convergence<C: Cells>() {
    let rt = Runtime::<C>::new();
    let me = cell::<f64>();
    let q = {
        let me = me.clone();
        rt.create_query_with(
            move |rt| {
                let x = rt.get(*me.get().unwrap());
                (x + 2.0 / x) / 2.0
            },
            fixpoint(Fixpoint::new(|| 1.0_f64).converged(|a: &f64, b: &f64| (a - b).abs() < 1e-12)),
        )
    };
    me.set(q).unwrap();
    assert!((rt.get(q) - std::f64::consts::SQRT_2).abs() < 1e-12);
}

/// A step function that damps the closure's output.
fn fixpoint_step<C: Cells>() {
    let rt = Runtime::<C>::new();
    let me = cell::<i64>();
    let q = {
        let me = me.clone();
        rt.create_query_with(
            move |rt| rt.get(*me.get().unwrap()) + 8,
            fixpoint(
                Fixpoint::new(|| 0_i64)
                    .step(|prev: &i64, out: i64, _k: u32| (prev + out) / 2)
                    .max_iterations(64),
            ),
        )
    };
    me.set(q).unwrap();
    // s = (s + s + 8) / 2 = s + 4 until the limit's last value.
    assert_eq!(rt.get(q), 4 * 64);
}

/// An unseeded 2-cycle lands Cyclic, recomputes to a value once an
/// input breaks it, and forms again.
fn unseeded_cycle_forms_breaks_reforms<C: Cells>() {
    let rt = Runtime::<C>::new();
    let g = rt.create_input(0_i64);
    let bc = cell::<i64>();
    let a = {
        let bc = bc.clone();
        rt.create_query(move |rt| {
            if rt.get(g) == 0 {
                rt.try_get(*bc.get().unwrap()).map_or(-1, |v| v + 1)
            } else {
                7
            }
        })
    };
    let b = rt.create_query(move |rt| rt.try_get(a).map_or(-1, |v| v * 2));
    bc.set(b).unwrap();
    let d = rt.create_query(move |rt| rt.try_get(b).unwrap_or(-100));

    assert_eq!(rt.get(d), -100);
    assert_eq!(rt.state_of(a), NodeState::Cyclic);
    assert_eq!(rt.state_of(b), NodeState::Cyclic);
    assert_eq!(slots(&rt.try_get(a).unwrap_err()), vec![a.slot(), b.slot()]);
    rt.assert_invariant_i();

    rt.set(g, 1);
    assert_eq!(rt.get(a), 7);
    assert_eq!(rt.get(b), 14);
    assert_eq!(rt.get(d), 14);
    assert_eq!(rt.cycle_of(a), None);
    rt.assert_invariant_i();

    rt.set(g, 0);
    assert_eq!(rt.get(d), -100);
    assert_eq!(rt.state_of(a), NodeState::Cyclic);
    rt.assert_invariant_i();
}

/// A strategy member that reads an unseeded member gets the value the
/// unseeded member computed against the strategy member's value.
fn seeded_reads_unseeded<C: Cells>() {
    let rt = Runtime::<C>::new();
    let uc = cell::<i64>();
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let s = {
        let (uc, seen) = (uc.clone(), seen.clone());
        rt.create_query_with(
            move |rt| {
                let u = rt.get(*uc.get().unwrap());
                seen.lock().unwrap().push(u);
                (u / 10 + 1).min(4)
            },
            fixpoint(Fixpoint::new(|| 0_i64)),
        )
    };
    let u = rt.create_query(move |rt| rt.get(s) * 10);
    uc.set(u).unwrap();
    assert_eq!(rt.get(s), 4);
    assert_eq!(rt.get(u), 40);
    // Iteration k reads u_k = 10 * s_{k-1}.
    let seen = seen.lock().unwrap().clone();
    assert!(seen.ends_with(&[0, 10, 20, 30, 40]), "{seen:?}");
}

/// Every entry member gives the same landed values on a static graph:
/// C (Fallback) reads X, X reads A, A reads C.
fn entry_member_does_not_matter<C: Cells>() {
    for entry in 0..3 {
        let rt = Runtime::<C>::new();
        let xc = cell::<i64>();
        let c = {
            let xc = xc.clone();
            rt.create_query_with_fallback(
                move |rt| rt.try_get(*xc.get().unwrap()).map_or(-5, |v| v + 1),
                |cy: &Cycle| 10 * cy.len() as i64,
            )
        };
        let a = rt.create_query(move |rt| rt.try_get(c).map_or(-6, |v| v + 2));
        let x = rt.create_query(move |rt| rt.try_get(a).map_or(-7, |v| v + 3));
        xc.set(x).unwrap();
        let nodes = [c, x, a];
        rt.try_get(nodes[entry]).unwrap();
        assert_eq!(rt.get(c), 30, "entry {entry}");
        assert_eq!(rt.get(a), 32, "entry {entry}");
        assert_eq!(rt.get(x), 35, "entry {entry}");
        assert_eq!(slots(&rt.cycle_of(a).unwrap()).len(), 3);
        rt.assert_invariant_i();
    }
}

/// The plan's group C (Fallback) -> X -> A -> C entered at X with `get`:
/// R3 panics, and every frame it unwinds lands New, so a later `get` of
/// C lands C's fallback.
fn get_at_unseeded_member_panics_then_fallback_lands<C: Cells>() {
    let rt = Runtime::<C>::new();
    let xc = cell::<i64>();
    let c = {
        let xc = xc.clone();
        rt.create_query_with_fallback(
            move |rt| rt.get(*xc.get().unwrap()) + 1,
            |cy: &Cycle| 10 * cy.len() as i64,
        )
    };
    let a = rt.create_query(move |rt| rt.get(c) + 2);
    let x = rt.create_query(move |rt| rt.get(a) + 3);
    xc.set(x).unwrap();
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_err());
    assert!(msg.contains("dependency cycle"), "{msg}");
    for n in [x, a] {
        assert_eq!(rt.state_of(n), NodeState::New);
    }
    assert_eq!(rt.get(c), 30);
    assert_eq!(rt.get(x), 35);
    rt.assert_invariant_i();
}

/// Validation finds the root alone: the episode dissolves and the root
/// computes as an ordinary node.
fn validation_dissolves<C: Cells>() {
    let rt = Runtime::<C>::new();
    let ac = cell::<i64>();
    let a_runs = Arc::new(AtomicU32::new(0));
    let c = {
        let ac = ac.clone();
        rt.create_query_with_fallback(move |rt| rt.get(*ac.get().unwrap()) + 1, |_c: &Cycle| 100)
    };
    let a = {
        let a_runs = a_runs.clone();
        rt.create_query(move |rt| {
            if a_runs.fetch_add(1, Ordering::Relaxed) == 0 {
                rt.get(c)
            } else {
                5
            }
        })
    };
    ac.set(a).unwrap();
    assert_eq!(rt.get(c), 6);
    assert_eq!(rt.get(a), 5);
    assert_eq!(rt.cycle_of(c), None);
    assert_eq!(rt.state_of(c), NodeState::Clean);
    rt.assert_invariant_i();
}

/// The plan's validation-release test: X is read only in discovery, so
/// validation releases it and the episode restarts with M = {C, A, B}.
fn validation_releases_and_restarts<C: Cells>() {
    let rt = Runtime::<C>::new();
    let (ac, bc, xc) = (cell::<i64>(), cell::<i64>(), cell::<i64>());
    let runs: Arc<[AtomicU32; 4]> = Arc::new(Default::default());
    let c = {
        let (ac, bc, runs) = (ac.clone(), bc.clone(), runs.clone());
        rt.create_query_with_fallback(
            move |rt| {
                runs[0].fetch_add(1, Ordering::Relaxed);
                rt.get(*ac.get().unwrap()) + rt.get(*bc.get().unwrap())
            },
            |cy: &Cycle| i64::from(cy.len() == 2),
        )
    };
    let a = {
        let (xc, runs) = (xc.clone(), runs.clone());
        rt.create_query(move |rt| {
            runs[1].fetch_add(1, Ordering::Relaxed);
            if rt.get(c) == 1 {
                rt.get(*xc.get().unwrap());
            }
            0
        })
    };
    let b = {
        let runs = runs.clone();
        rt.create_query(move |rt| {
            runs[2].fetch_add(1, Ordering::Relaxed);
            rt.get(c);
            0
        })
    };
    let x = {
        let runs = runs.clone();
        rt.create_query(move |rt| {
            runs[3].fetch_add(1, Ordering::Relaxed);
            rt.get(c)
        })
    };
    ac.set(a).unwrap();
    bc.set(b).unwrap();
    xc.set(x).unwrap();

    assert_eq!(rt.get(c), 0);
    let r: Vec<u32> = runs.iter().map(|r| r.load(Ordering::Relaxed)).collect();
    assert_eq!(r, vec![3, 3, 3, 2], "runs of C, A, B, X");
    assert_eq!(rt.state_of(x), NodeState::New);
    assert_eq!(
        slots(&rt.cycle_of(c).unwrap()),
        vec![c.slot(), a.slot(), b.slot()]
    );
    assert_eq!(rt.get(x), 0);
    assert_eq!(runs[3].load(Ordering::Relaxed), 3);
    assert_eq!(rt.try_get(x), Ok(0));
    rt.assert_invariant_i();
}

/// A landed group whose upstream input changes by a cutoff re-lands Clean
/// through group verification, without running a closure.
fn group_verify_relands_clean<C: Cells>() {
    let rt = Runtime::<C>::new();
    let i = rt.create_input(1_i64);
    let bc = cell::<i64>();
    let runs = Arc::new(AtomicU32::new(0));
    let cut = rt.create_query(move |rt| rt.get(i) % 2);
    let a = {
        let (bc, runs) = (bc.clone(), runs.clone());
        rt.create_query_with_fallback(
            move |rt| {
                runs.fetch_add(1, Ordering::Relaxed);
                rt.get(*bc.get().unwrap()) + rt.get(cut)
            },
            |_c: &Cycle| 100,
        )
    };
    let b = {
        let runs = runs.clone();
        rt.create_query(move |rt| {
            runs.fetch_add(1, Ordering::Relaxed);
            rt.get(a) + 1
        })
    };
    bc.set(b).unwrap();
    assert_eq!(rt.get(a), 100);
    assert_eq!(rt.get(b), 101);
    let before = runs.load(Ordering::Relaxed);
    rt.set(i, 3);
    assert_eq!(rt.state_of(a), NodeState::Dirty);
    assert_eq!(rt.get(a), 100);
    assert_eq!(rt.get(b), 101);
    assert_eq!(
        runs.load(Ordering::Relaxed),
        before,
        "group verify ran no closure"
    );
    assert_eq!(rt.state_of(b), NodeState::Clean);
    rt.assert_invariant_i();

    rt.set(i, 2);
    assert_eq!(rt.get(b), 101);
    assert!(runs.load(Ordering::Relaxed) > before);
    rt.assert_invariant_i();
}

/// Validation shrinks a group recomputed through group verification, and
/// the released member's record goes with its landing.
fn validation_shrinks_group<C: Cells>() {
    let rt = Runtime::<C>::new();
    let i = rt.create_input(0_i64);
    let (ac, bc) = (cell::<i64>(), cell::<i64>());
    let c = {
        let (ac, bc) = (ac.clone(), bc.clone());
        rt.create_query_with_fallback(
            move |rt| rt.get(*ac.get().unwrap()) + rt.get(*bc.get().unwrap()),
            |cy: &Cycle| 10 * cy.len() as i64,
        )
    };
    let a = rt.create_query(move |rt| if rt.get(i) == 0 { rt.get(c) } else { 1 });
    let b = rt.create_query(move |rt| rt.get(c) + 2);
    ac.set(a).unwrap();
    bc.set(b).unwrap();
    assert_eq!(rt.get(c), 30);
    assert_eq!(rt.get(a), 30);
    assert_eq!(slots(&rt.cycle_of(a).unwrap()).len(), 3);

    rt.set(i, 1);
    assert_eq!(rt.get(c), 20);
    assert_eq!(rt.get(b), 22);
    assert_eq!(rt.get(a), 1);
    assert_eq!(rt.cycle_of(a), None);
    assert_eq!(slots(&rt.cycle_of(b).unwrap()), vec![c.slot(), b.slot()]);
    rt.assert_invariant_i();
}

/// Y's episode releases R, M and X still holding their group's values,
/// which their closures need not give: M's is its fallback 4, not X + 1.
/// Their deps verifying unchanged must not land them Clean on those.
fn released_member_reruns_off_its_group_value<C: Cells>() {
    let rt = Runtime::<C>::new();
    let s = rt.create_input(0_i64);
    let (mc, rc) = (cell::<i64>(), cell::<i64>());
    let y = {
        let rc = rc.clone();
        rt.create_query(move |rt| {
            if rt.get(s) == 0 {
                rt.get(*rc.get().unwrap())
            } else {
                10
            }
        })
    };
    let x = rt.create_query(move |rt| rt.get(y));
    let m = rt.create_query_with_fallback(move |rt| rt.get(x) + 1, |_c: &Cycle| 4);
    let r = {
        let mc = mc.clone();
        rt.create_query_with_fallback(move |rt| rt.get(*mc.get().unwrap()), |_c: &Cycle| 10)
    };
    mc.set(m).unwrap();
    rc.set(r).unwrap();
    assert_eq!(rt.get(r), 10);
    assert_eq!((rt.get(m), rt.get(x)), (4, 10));
    assert_eq!(slots(&rt.cycle_of(m).unwrap()).len(), 4);

    rt.set(s, 1);
    assert_eq!(rt.get(y), 10);
    assert_eq!(rt.get(r), 11);
    assert_eq!(rt.get(m), 11);
    assert_eq!(rt.cycle_of(m), None);
    rt.assert_invariant_i();
}

/// Two disjoint cycles, the second nested inside the first's discovery.
fn two_disjoint_cycles_on_one_stack<C: Cells>() {
    let rt = Runtime::<C>::new();
    let (bc, dc) = (cell::<i64>(), cell::<i64>());
    let c2 = {
        let dc = dc.clone();
        rt.create_query_with_fallback(move |rt| rt.get(*dc.get().unwrap()), |_c: &Cycle| 7)
    };
    let d2 = rt.create_query(move |rt| rt.get(c2) + 1);
    dc.set(d2).unwrap();
    let a1 = {
        let bc = bc.clone();
        rt.create_query_with_fallback(
            move |rt| rt.get(*bc.get().unwrap()) + rt.get(c2),
            |_c: &Cycle| 50,
        )
    };
    let b1 = rt.create_query(move |rt| rt.get(a1) + rt.get(c2));
    bc.set(b1).unwrap();
    assert_eq!(rt.get(a1), 50);
    assert_eq!(rt.get(b1), 57);
    assert_eq!(rt.get(d2), 8);
    assert_eq!(rt.get(c2), 7);
    assert_eq!(slots(&rt.cycle_of(c2).unwrap()), vec![c2.slot(), d2.slot()]);
    assert_eq!(slots(&rt.cycle_of(a1).unwrap()), vec![a1.slot(), b1.slot()]);
    rt.assert_invariant_i();
}

/// Callbacks run as value code under a barrier frame: a read of a Clean
/// node returns its value and records nothing, a read of anything else
/// and a `set` panic.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_callback_barrier() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let weak = Arc::downgrade(&rt);
    let i = rt.create_input(3_i64);
    let t = rt.create_input(0_i64);
    let late = rt.create_query(move |rt| rt.get(i) * 2);
    let mode = Arc::new(AtomicU32::new(0));
    let me = cell::<i64>();
    let q = {
        let (me, mode) = (me.clone(), mode.clone());
        rt.create_query_with_fallback(
            move |rt| rt.get(t) + rt.get(*me.get().unwrap()),
            move |_c: &Cycle| {
                let rt = weak.upgrade().unwrap();
                match mode.load(Ordering::Relaxed) {
                    0 => rt.get(i),
                    1 => rt.get(late),
                    _ => {
                        rt.set(i, 4);
                        0
                    }
                }
            },
        )
    };
    me.set(q).unwrap();
    assert_eq!(rt.get(q), 3);
    assert!(
        !rt.has_reverse_edge(i, q),
        "a callback's read records nothing"
    );

    for m in [1, 2] {
        mode.store(m, Ordering::Relaxed);
        rt.set(t, i64::from(m));
        let msg = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(q))).unwrap_err());
        assert!(
            msg.contains("must not call into the runtime"),
            "mode {m}: {msg}"
        );
        rt.assert_invariant_i();
    }
    mode.store(0, Ordering::Relaxed);
    rt.set(t, 5);
    assert_eq!(rt.get(q), 3);
}

/// A 1000-member ring of unseeded members reading their successor with
/// `try_get` lands Cyclic; breaking it recomputes every member.
fn thousand_member_ring<C: Cells>() {
    let rt = Runtime::<C>::new();
    let n = crate::test_size::test_size(1000);
    let g = rt.create_input(0_i64);
    let ring: Arc<OnceLock<Vec<Incr<i64>>>> = Arc::new(OnceLock::new());
    let members: Vec<Incr<i64>> = (0..n)
        .map(|k| {
            let ring = ring.clone();
            rt.create_query(move |rt| {
                if k == n / 2 && rt.get(g) != 0 {
                    return 0;
                }
                rt.try_get(ring.get().unwrap()[(k + 1) % n])
                    .map_or(-1, |v| v + 1)
            })
        })
        .collect();
    ring.set(members.clone()).unwrap();
    assert_eq!(rt.try_get(members[0]).unwrap_err().len(), n);
    for &m in &members {
        assert_eq!(rt.state_of(m), NodeState::Cyclic);
    }
    rt.set(g, 1);
    for (k, &m) in members.iter().enumerate() {
        let expected = if k <= n / 2 { n / 2 - k } else { n / 2 + n - k };
        assert_eq!(rt.get(m), expected as i64, "member {k}");
    }
    rt.assert_invariant_i();
}

/// A → B → C → A with a Fallback on A: B and C land over A's fallback.
fn three_cycle_fallback<C: Cells>() {
    let rt = Runtime::<C>::new();
    let bc = cell::<i64>();
    let cc = cell::<i64>();
    let a = {
        let bc = bc.clone();
        rt.create_query_with_fallback(move |rt| rt.get(*bc.get().unwrap()) + 1, |_c: &Cycle| 100)
    };
    let b = {
        let cc = cc.clone();
        rt.create_query(move |rt| rt.get(*cc.get().unwrap()) * 2)
    };
    let c = rt.create_query(move |rt| rt.get(a) + 3);
    bc.set(b).unwrap();
    cc.set(c).unwrap();
    assert_eq!(rt.get(a), 100);
    assert_eq!(rt.get(b), 206);
    assert_eq!(rt.get(c), 103);
    assert_eq!(
        slots(&rt.cycle_of(c).unwrap()),
        vec![a.slot(), b.slot(), c.slot()]
    );
    rt.assert_invariant_i();
}

/// Three Fixpoint members converge together.
fn three_cycle_fixpoint<C: Cells>() {
    let rt = Runtime::<C>::new();
    let bc = cell::<i64>();
    let cc = cell::<i64>();
    let a = {
        let bc = bc.clone();
        rt.create_query_with(
            move |rt| (rt.get(*bc.get().unwrap()) + 1).min(10),
            fixpoint(Fixpoint::new(|| 0_i64)),
        )
    };
    let b = {
        let cc = cc.clone();
        rt.create_query_with(
            move |rt| rt.get(*cc.get().unwrap()),
            fixpoint(Fixpoint::new(|| 0_i64)),
        )
    };
    let c = rt.create_query_with(move |rt| rt.get(a), fixpoint(Fixpoint::new(|| 0_i64)));
    bc.set(b).unwrap();
    cc.set(c).unwrap();
    assert_eq!(rt.get(b), 10);
    assert_eq!(rt.get(a), 10);
    assert_eq!(rt.get(c), 10);
    rt.assert_invariant_i();
}

/// Three unseeded members reading each other with `try_get` land Cyclic.
fn three_cycle_unseeded<C: Cells>() {
    let rt = Runtime::<C>::new();
    let bc = cell::<i64>();
    let cc = cell::<i64>();
    let a = {
        let bc = bc.clone();
        rt.create_query(move |rt| rt.try_get(*bc.get().unwrap()).map_or(-1, |v| v + 1))
    };
    let b = {
        let cc = cc.clone();
        rt.create_query(move |rt| rt.try_get(*cc.get().unwrap()).map_or(-2, |v| v + 1))
    };
    let c = rt.create_query(move |rt| rt.try_get(a).map_or(-3, |v| v + 1));
    bc.set(b).unwrap();
    cc.set(c).unwrap();
    let cycle = rt.try_get(b).unwrap_err();
    assert_eq!(slots(&cycle), vec![a.slot(), b.slot(), c.slot()]);
    for n in [a, b, c] {
        assert!(rt.try_get(n).is_err());
        assert_eq!(rt.state_of(n), NodeState::Cyclic);
    }
    rt.assert_invariant_i();
}

/// The group C (Fallback) → X → A → C. With `try_get` reads, entering at
/// any member lands the same values; with `get`, entering at X panics by
/// R3 on C's read of X, and a later read at C lands the group.
fn group_c_x_a<C: Cells>() {
    let build = |rt: &Runtime<C>, tolerant: bool| {
        let xc = cell::<i64>();
        let ac = cell::<i64>();
        let c = {
            let xc = xc.clone();
            rt.create_query_with_fallback(
                move |rt| {
                    let x = *xc.get().unwrap();
                    if tolerant {
                        rt.try_get(x).map_or(-7, |v| v + 1)
                    } else {
                        rt.get(x) + 1
                    }
                },
                |_c: &Cycle| 100,
            )
        };
        let x = {
            let ac = ac.clone();
            rt.create_query(move |rt| {
                let a = *ac.get().unwrap();
                if tolerant {
                    rt.try_get(a).map_or(-8, |v| v * 2)
                } else {
                    rt.get(a) * 2
                }
            })
        };
        let a = rt.create_query(move |rt| {
            if tolerant {
                rt.try_get(c).map_or(-9, |v| v + 3)
            } else {
                rt.get(c) + 3
            }
        });
        xc.set(x).unwrap();
        ac.set(a).unwrap();
        [c, x, a]
    };
    for entry in 0..3 {
        let rt = Runtime::<C>::new();
        let nodes = build(&rt, true);
        rt.try_get(nodes[entry]).unwrap();
        let landed: Vec<i64> = nodes.iter().map(|&n| rt.try_get(n).unwrap()).collect();
        assert_eq!(landed, vec![100, 206, 103], "entered at {entry}");
        rt.assert_invariant_i();
    }

    let rt = Runtime::<C>::new();
    let [c, x, a] = build(&rt, false);
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_err());
    assert!(msg.contains("dependency cycle"), "{msg}");
    assert_eq!(rt.get(c), 100);
    assert_eq!(rt.get(x), 206);
    assert_eq!(rt.get(a), 103);
    rt.assert_invariant_i();
}

/// A Fixpoint member that catches a failing query's panic in the runs
/// `catching` picks from its run count and the value it read of itself,
/// with `step` adding each output to the previous value when `adds`.
fn caught_in_some_runs<C: Cells>(
    catching: impl Fn(u32, i64) -> bool + Send + Sync + 'static,
    adds: bool,
) -> (Runtime<C>, Incr<i64>) {
    let rt = Runtime::<C>::new();
    let failing = rt.create_query(|_: &Runtime<C>| -> i64 { panic!("boom") });
    let me = cell::<i64>();
    let runs = Arc::new(AtomicU32::new(0));
    let f = {
        let me = me.clone();
        rt.create_query_with(
            move |rt| {
                let prev = rt.get(*me.get().unwrap());
                if catching(runs.fetch_add(1, Ordering::Relaxed), prev) {
                    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(failing))).is_err());
                }
                1
            },
            fixpoint(if adds {
                Fixpoint::new(|| 0_i64).step(|prev: &i64, out, _| (prev + out).min(3))
            } else {
                Fixpoint::new(|| 0_i64)
            }),
        )
    };
    me.set(f).unwrap();
    (rt, f)
}

/// Iteration 1 catches and iteration 2 does not, but reads the value
/// iteration 1 built: the landing keeps the poison through that read.
fn poison_reaches_the_landing_through_a_member_read<C: Cells>() {
    let (rt, f) = caught_in_some_runs::<C>(|_, prev| prev == 0, false);
    assert_eq!(rt.get(f), 1);
    assert_eq!(rt.state_of(f), NodeState::Dirty);
}

/// Only the discovery run and iteration 1 catch, and the last run reads
/// nothing poisoned, but `step` builds every later value on iteration 1's.
fn poison_reaches_the_landing_through_step<C: Cells>() {
    let (rt, f) = caught_in_some_runs::<C>(|run, _| run < 2, true);
    assert_eq!(rt.get(f), 3);
    assert_eq!(rt.state_of(f), NodeState::Dirty);
}

both!(
    self_cycle_fallback,
    self_cycle_fixpoint,
    self_cycle_unseeded,
    two_cycle_fallback,
    two_cycle_entered_at_unseeded,
    two_cycle_fixpoint,
    three_cycle_fallback,
    three_cycle_fixpoint,
    three_cycle_unseeded,
    group_c_x_a,
    oscillation_reaches_the_limit,
    tolerance_convergence,
    fixpoint_step,
    unseeded_cycle_forms_breaks_reforms,
    seeded_reads_unseeded,
    entry_member_does_not_matter,
    get_at_unseeded_member_panics_then_fallback_lands,
    validation_dissolves,
    validation_releases_and_restarts,
    group_verify_relands_clean,
    validation_shrinks_group,
    released_member_reruns_off_its_group_value,
    two_disjoint_cycles_on_one_stack,
    poison_reaches_the_landing_through_a_member_read,
    poison_reaches_the_landing_through_step,
);

#[test]
#[cfg_attr(miri, ignore)]
fn local_thousand_member_ring() {
    on_deep_stack(thousand_member_ring::<Local>);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_thousand_member_ring() {
    on_deep_stack(thousand_member_ring::<Shared>);
}

fn on_deep_stack(body: fn()) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}
