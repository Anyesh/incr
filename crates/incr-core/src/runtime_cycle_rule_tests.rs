//! Cycles: the `verified_at = 0` rules, landings around panics, the empty
//! slot of a `Cyclic` node, group verification's claim failures, merges,
//! recovery bounds and cross-runtime reads. Each case runs under `Local`
//! and `Shared`.

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

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

fn cell<T>() -> Arc<OnceLock<Incr<T>>> {
    Arc::new(OnceLock::new())
}

fn slots(cycle: &Cycle) -> Vec<u32> {
    cycle.participants().iter().map(|p| p.slot()).collect()
}

fn quiet<R>(f: impl FnOnce() -> R) -> Result<R, Box<dyn std::any::Any + Send>> {
    catch_unwind(AssertUnwindSafe(f))
}

/// P reads j and, while j is 0, itself through `try_get`, so it lands
/// Cyclic alone. `set(j, 1)` breaks the cycle and moves P to New, and its
/// recompute panics before writing: it lands Failed with installed deps
/// {j, P}, and rule (a) leaves its `verified_at` at 0, not at the
/// revision its Cyclic landing raised it to.
fn rule_a_setup<C: Cells>(rt: &Runtime<C>) -> (Incr<i64>, Incr<i64>) {
    let j = rt.create_input(0_i64);
    let me = cell::<i64>();
    let p = {
        let me = me.clone();
        rt.create_query(move |rt| match rt.get(j) {
            0 => rt.try_get(*me.get().unwrap()).map_or(-1, |v| v + 1),
            1 => panic!("P fails before writing"),
            v => v * 10,
        })
    };
    me.set(p).unwrap();
    assert!(rt.try_get(p).is_err());
    assert_eq!(rt.state_of(p), NodeState::Cyclic);
    assert_ne!(rt.verified_at_of(p), 0);
    rt.set(j, 1);
    assert_eq!(rt.state_of(p), NodeState::New);
    assert!(quiet(|| rt.get(p)).is_err());
    assert_eq!(rt.state_of(p), NodeState::Failed);
    assert!(rt.has_reverse_edge(p, p));
    assert!(rt.has_reverse_edge(j, p));
    (j, p)
}

fn rule_a_stores_zero_on_a_new_claim_that_wrote_nothing<C: Cells>() {
    let rt = Runtime::<C>::new();
    let (_, p) = rule_a_setup(&rt);
    assert_eq!(rt.verified_at_of(p), 0);
}

/// A never-run query moved to Dirty by hand: without rule (b), verify
/// finds no changed dep among zero and lands it Clean over an empty slot.
fn rule_b_fails_a_verify_at_zero<C: Cells>() {
    let rt = Runtime::<C>::new();
    let q = rt.create_query(|_| 42_i64);
    rt.force_dirty_for_test(q);
    assert_eq!(rt.get(q), 42);
}

/// Rule (b) in group verify: a group member at `verified_at = 0` fails
/// the group, which recomputes through an episode that dissolves.
fn rule_b_fails_a_group_at_zero<C: Cells>() {
    let rt = Runtime::<C>::new();
    let (j, p) = rule_a_setup(&rt);
    rt.set(j, 2);
    assert_eq!(rt.state_of(p), NodeState::Dirty);
    assert_eq!(rt.verified_at_of(p), 0);

    let k = rt.create_input(0_i64);
    let c = rt.create_query(move |rt| {
        rt.get(k);
        7_i64
    });
    let w = rt.create_query(move |rt| rt.get(c) + 1);
    assert_eq!(rt.get(w), 8);
    rt.set(k, 1);
    assert_eq!(rt.state_of(w), NodeState::Dirty);
    rt.force_group_for_test(&[p.erase(), w.erase()]);

    assert_eq!(rt.get(w), 8);
    assert_eq!(rt.group_zero_fails.load(Ordering::Relaxed), 1);
    assert_eq!(rt.state_of(p), NodeState::Dirty);
    assert_eq!(rt.verified_at_of(p), 0);
    assert_eq!(rt.get(p), 20);
    rt.assert_invariant_i();
}

/// Scenario A: Q's first compute reads i, sets it, and panics before
/// writing. The set supersedes Q's read, so Q lands Dirty with
/// `verified_at = 0`, and the next read recomputes it.
fn scenario_a<C: Cells>() {
    let rt = Runtime::<C>::new();
    let i = rt.create_input(1_i64);
    let first = Arc::new(AtomicBool::new(true));
    let q = {
        let first = first.clone();
        rt.create_query(move |rt| {
            let v = rt.get(i);
            if first.swap(false, Ordering::Relaxed) {
                rt.set(i, v + 1);
                panic!("Q fails after its set");
            }
            v * 100
        })
    };
    assert!(quiet(|| rt.get(q)).is_err());
    assert_eq!(rt.state_of(q), NodeState::Dirty);
    assert_eq!(rt.verified_at_of(q), 0);
    assert_eq!(rt.get(q), 200);
}

/// Scenario B: Q's first compute panics and lands Failed; a set of a
/// recorded dep walks it Dirty; a cutoff-only change upstream follows.
/// The next read recomputes Q.
fn scenario_b<C: Cells>() {
    let rt = Runtime::<C>::new();
    let i = rt.create_input(1_i64);
    let k = rt.create_input(0_i64);
    let c = rt.create_query(move |rt| rt.get(k) % 2);
    let fail = Arc::new(AtomicBool::new(true));
    let q = {
        let fail = fail.clone();
        rt.create_query(move |rt| {
            let v = rt.get(i) + rt.get(c);
            if fail.load(Ordering::Relaxed) {
                panic!("Q fails");
            }
            v
        })
    };
    assert!(quiet(|| rt.get(q)).is_err());
    assert_eq!(rt.state_of(q), NodeState::Failed);
    assert_eq!(rt.verified_at_of(q), 0);
    fail.store(false, Ordering::Relaxed);
    rt.set(i, 5);
    rt.set(k, 2);
    assert_eq!(rt.get(q), 5);
}

/// A former CU member whose recompute from New panics while a set flags
/// it lands Dirty with `verified_at = 0`, and a cutoff-only set later does
/// not let it verify Clean over its empty slot.
fn former_member_flagged_panic<C: Cells>() {
    let rt = Runtime::<C>::new();
    let j = rt.create_input(0_i64);
    let t = rt.create_input(0_i64);
    let k = rt.create_input(0_i64);
    let c = rt.create_query(move |rt| rt.get(k) % 2);
    let me = cell::<i64>();
    let p = {
        let me = me.clone();
        rt.create_query(move |rt| {
            let v = rt.get(j) + rt.get(c);
            match v {
                0 => rt.try_get(*me.get().unwrap()).map_or(-1, |v| v + 1),
                1 => {
                    let tv = rt.get(t);
                    if tv == 0 {
                        rt.set(t, 1);
                        panic!("P fails while flagged");
                    }
                    v * 3
                }
                v => v,
            }
        })
    };
    me.set(p).unwrap();
    assert!(rt.try_get(p).is_err());
    rt.set(j, 1);
    assert!(quiet(|| rt.get(p)).is_err());
    assert_eq!(rt.state_of(p), NodeState::Dirty);
    assert_eq!(rt.verified_at_of(p), 0);
    rt.set(k, 2);
    assert_eq!(rt.get(p), 3);
}

/// A member panics in the episode's first iteration: it lands Failed
/// with `verified_at = 0`, the other member takes the discard landing,
/// and after a cutoff-only set neither verifies Clean over an empty slot.
fn panic_in_an_iteration<C: Cells>() {
    let rt = Runtime::<C>::new();
    let k = rt.create_input(0_i64);
    let c = rt.create_query(move |rt| rt.get(k) % 2);
    let bc = cell::<i64>();
    let b_runs = Arc::new(AtomicU32::new(0));
    let a = {
        let bc = bc.clone();
        rt.create_query_with_fallback(move |rt| rt.get(*bc.get().unwrap()) + 1, |_c: &Cycle| 100)
    };
    let b = {
        let b_runs = b_runs.clone();
        rt.create_query(move |rt| {
            let v = rt.get(a) + rt.get(c);
            if b_runs.fetch_add(1, Ordering::Relaxed) == 1 {
                panic!("b fails in iteration 1");
            }
            v
        })
    };
    bc.set(b).unwrap();
    assert!(quiet(|| rt.get(a)).is_err());
    assert_eq!(rt.state_of(b), NodeState::Failed);
    assert_eq!(rt.verified_at_of(b), 0);
    assert_eq!(rt.state_of(a), NodeState::New);
    assert_eq!(rt.verified_at_of(a), 0);
    rt.set(k, 2);
    assert_eq!(rt.state_of(b), NodeState::Dirty);
    assert_eq!(rt.get(a), 100);
    assert_eq!(rt.get(b), 100);
    rt.assert_invariant_i();
}

thread_local! {
    static BOMBS: Cell<u32> = const { Cell::new(0) };
}

/// A value whose original instance panics in its `Drop` when armed;
/// clones never panic.
#[derive(Debug)]
struct Bomb {
    v: i64,
    armed: bool,
}

impl Clone for Bomb {
    fn clone(&self) -> Self {
        Bomb {
            v: self.v,
            armed: false,
        }
    }
}

impl PartialEq for Bomb {
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

impl Drop for Bomb {
    fn drop(&mut self) {
        if self.armed && !std::thread::panicking() {
            BOMBS.with(|b| b.set(b.get() + 1));
            panic!("Bomb {} dropped", self.v);
        }
    }
}

/// A Fixpoint member's s_1 panics when the episode drops it between
/// iterations: the member lands Failed, the other New, none Computing,
/// and after an upstream set the cycle converges.
fn fixpoint_value_drop_panics<C: Cells>() {
    let rt = Runtime::<C>::new();
    let g = rt.create_input(0_i64);
    let bc = cell::<Bomb>();
    let arm = Arc::new(AtomicBool::new(true));
    let a = {
        let (bc, arm) = (bc.clone(), arm.clone());
        rt.create_query_with(
            move |rt| Bomb {
                v: (rt.try_get(*bc.get().unwrap()).map_or(0, |b| b.v) + 1 + rt.get(g)).min(4),
                armed: false,
            },
            QueryOpts {
                cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| Bomb { v: 0, armed: false }).step(
                    move |_prev: &Bomb, out: Bomb, k: u32| Bomb {
                        v: out.v,
                        armed: k == 1 && arm.swap(false, Ordering::Relaxed),
                    },
                )),
                ..Default::default()
            },
        )
    };
    let b = rt.create_query(move |rt| {
        rt.try_get(a).unwrap_or(Bomb {
            v: -1,
            armed: false,
        })
    });
    bc.set(b).unwrap();
    let err = quiet(|| rt.get(a)).unwrap_err();
    assert!(err
        .downcast_ref::<String>()
        .is_some_and(|m| m.contains("Bomb 1")));
    assert_eq!(rt.state_of(a), NodeState::Failed);
    assert_eq!(rt.verified_at_of(a), 0);
    assert_eq!(rt.state_of(b), NodeState::New);
    rt.set(g, 1);
    assert_eq!(rt.get(a).v, 4);
    assert_eq!(rt.get(b).v, 4);
    rt.assert_invariant_i();
}

/// A landing from a value to Cyclic is a stamped change: D, Clean on P's
/// value, recomputes from the `Err`, and its observer fires once. After
/// the cycle breaks, D recomputes from P's value again.
fn cyclic_landing_is_a_change<C: Cells>() {
    let rt = Runtime::<C>::new();
    let g = rt.create_input(0_i64);
    let qc = cell::<i64>();
    let p = {
        let qc = qc.clone();
        rt.create_query(move |rt| {
            if rt.get(g) == 1 {
                rt.try_get(*qc.get().unwrap()).map_or(-1, |v| v)
            } else {
                5
            }
        })
    };
    let q = rt.create_query(move |rt| rt.try_get(p).map_or(-1, |v| v));
    qc.set(q).unwrap();
    let d = rt.create_query(move |rt| rt.try_get(p).map_or(-100, |v| v * 2));
    assert_eq!(rt.get(d), 10);
    let fired = Arc::new(Mutex::new(Vec::new()));
    {
        let fired = fired.clone();
        rt.observe(d, move |v| fired.lock().unwrap().push(*v));
    }
    rt.stabilize();
    fired.lock().unwrap().clear();

    rt.set(g, 1);
    assert_eq!(rt.try_get(d), Ok(-100));
    assert_eq!(rt.state_of(p), NodeState::Cyclic);
    rt.stabilize();
    assert_eq!(*fired.lock().unwrap(), vec![-100]);

    rt.set(g, 0);
    assert_eq!(rt.get(d), 10);
    rt.stabilize();
    assert_eq!(*fired.lock().unwrap(), vec![-100, 10]);
    rt.assert_invariant_i();
}

/// P reads S, which panics on its own cycle, through `try_get`, then
/// itself: P is a lone unseeded member, and S landed New under it, so the
/// episode lands P New with no value. Its reader still gets the cycle.
fn dirty_cyclic_landing_reads_as_the_cycle<C: Cells>() {
    let rt = Runtime::<C>::new();
    let sc = cell::<i64>();
    let s = {
        let sc = sc.clone();
        rt.create_query(move |rt| rt.get(*sc.get().unwrap()) + 1)
    };
    sc.set(s).unwrap();
    let pc = cell::<i64>();
    let p = {
        let pc = pc.clone();
        rt.create_query(move |rt| {
            rt.try_get(s).unwrap_or(0) + rt.try_get(*pc.get().unwrap()).unwrap_or(0)
        })
    };
    pc.set(p).unwrap();
    let err = rt.try_get(p).unwrap_err();
    assert_eq!(slots(&err), vec![p.slot()]);
    assert_eq!(rt.state_of(p), NodeState::New);
    rt.assert_invariant_i();
}

/// R catches the cycle S raises, then reads P, which panics on its own:
/// P lands Failed, its panic no part of the cycle R swallowed.
fn a_panic_after_a_swallowed_cycle_lands_failed<C: Cells>() {
    let rt = Runtime::<C>::new();
    let sc = cell::<i64>();
    let s = {
        let sc = sc.clone();
        rt.create_query(move |rt| rt.get(*sc.get().unwrap()) + 1)
    };
    sc.set(s).unwrap();
    let p = rt.create_query(|_| -> i64 { panic!("P fails") });
    let r = rt.create_query(move |rt| quiet(|| rt.get(s)).unwrap_or(0) + rt.get(p));
    assert!(quiet(|| rt.get(r)).is_err());
    assert_eq!(rt.state_of(p), NodeState::Failed);
}

/// R reads A, which catches B's panic. B reads C, which reads R through
/// `try_get`, so C defers to R and B inherits R's position, then B panics.
/// The panic takes C's claim with it, though C is anchored below B, and B
/// lands Dirty, since C landed New under it.
fn a_panic_releases_the_claims_its_frame_deferred<C: Cells>() {
    let rt = Runtime::<C>::new();
    let rc = cell::<i64>();
    let c = {
        let rc = rc.clone();
        rt.create_query(move |rt| rt.try_get(*rc.get().unwrap()).unwrap_or(1))
    };
    let b = rt.create_query(move |rt| {
        rt.get(c);
        panic!("B fails after reading C")
    });
    let a = rt.create_query(move |rt| quiet(|| rt.get(b)).unwrap_or(9));
    let r = rt.create_query(move |rt| rt.get(a));
    rc.set(r).unwrap();
    assert_eq!(rt.get(r), 9);
    assert_eq!(rt.state_of(c), NodeState::New);
    assert_eq!(rt.state_of(b), NodeState::Dirty);
    rt.assert_invariant_i();
}

/// S, under R, reads Dirty Q, whose walk computes its dep A first. A reads
/// R, a Fallback on the stack, and then Q, whose nested walk computes it:
/// Q reads A, a Fallback on the stack, and defers to it, and A defers to R.
/// The outer walk finds its root Q held by this thread, so the read is a
/// read of a held node, not a wait on its own claim.
fn a_walk_root_deferred_under_its_walk<C: Cells>() {
    let rt = Runtime::<C>::new();
    let g = rt.create_input(0_i64);
    let rc = cell::<i64>();
    let qc = cell::<i64>();
    let a = {
        let (rc, qc) = (rc.clone(), qc.clone());
        rt.create_query_with_fallback(
            move |rt| {
                if rt.get(g) == 0 {
                    1
                } else {
                    rt.try_get(*rc.get().unwrap()).unwrap_or(0)
                        + rt.try_get(*qc.get().unwrap()).unwrap_or(0)
                }
            },
            |_c: &Cycle| 5,
        )
    };
    let q = rt.create_query(move |rt| rt.get(a) + 1);
    qc.set(q).unwrap();
    assert_eq!(rt.get(q), 2);
    rt.set(g, 1);
    let s = rt.create_query(move |rt| rt.get(q));
    let r = rt.create_query_with_fallback(move |rt| rt.get(s), |_c: &Cycle| 7);
    rc.set(r).unwrap();
    rt.get(r);
    for n in [a, q, s, r] {
        assert!(!matches!(
            rt.state_of(n),
            NodeState::Computing | NodeState::ComputingDirty
        ));
    }
    rt.assert_invariant_i();
}

/// P is Cyclic in {P, Q}; a set makes it land Cyclic in {P, Q, R}, and D,
/// which computed the participant count from `try_get(P)`, recomputes.
fn cyclic_group_grows<C: Cells>() {
    let rt = Runtime::<C>::new();
    let g = rt.create_input(0_i64);
    let (qc, rc) = (cell::<i64>(), cell::<i64>());
    let p = {
        let (qc, rc) = (qc.clone(), rc.clone());
        rt.create_query(move |rt| {
            let target = if rt.get(g) == 0 {
                *qc.get().unwrap()
            } else {
                *rc.get().unwrap()
            };
            rt.try_get(target).map_or(-1, |v| v)
        })
    };
    let r = {
        let qc = qc.clone();
        rt.create_query(move |rt| rt.try_get(*qc.get().unwrap()).map_or(-1, |v| v))
    };
    let q = rt.create_query(move |rt| rt.try_get(p).map_or(-1, |v| v));
    qc.set(q).unwrap();
    rc.set(r).unwrap();
    let d = rt.create_query(move |rt| rt.try_get(p).map_or_else(|c| c.len() as i64, |_| 0));
    assert_eq!(rt.get(d), 2);
    rt.set(g, 1);
    assert_eq!(rt.get(d), 3);
    assert_eq!(
        slots(&rt.cycle_of(p).unwrap()),
        vec![p.slot(), r.slot(), q.slot()]
    );
    rt.assert_invariant_i();
}

/// A recompute of a group whose values do not change cuts off its
/// dependents: D verifies Clean.
fn early_cutoff_through_a_group<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_i64);
    let bc = cell::<i64>();
    let a = {
        let bc = bc.clone();
        rt.create_query_with_fallback(move |rt| rt.get(*bc.get().unwrap()), |_c: &Cycle| 100)
    };
    let b = rt.create_query(move |rt| rt.get(a) + rt.get(x) % 2);
    bc.set(b).unwrap();
    let d = rt.create_query(move |rt| rt.get(b) * 10);
    assert_eq!(rt.get(a), 100);
    assert_eq!(rt.get(d), 1010);
    rt.set(x, 3);
    let (v, trace) = rt.get_traced(d);
    assert_eq!(v, 1010);
    let d_recomputed = trace.node_traces.iter().any(|t| {
        t.id.0 == d.slot() && matches!(t.action, crate::trace::TraceAction::Recomputed { .. })
    });
    assert!(!d_recomputed, "{trace:?}");
    assert!(trace.node_traces.iter().any(|t| t.id.0 == b.slot()));
}

/// A Fixpoint member reads input i only in iteration 1; a later set of i
/// still reaches the landed value.
fn input_read_only_in_iteration_one<C: Cells>() {
    let rt = Runtime::<C>::new();
    let i = rt.create_input(3_i64);
    let me = cell::<i64>();
    let q = {
        let me = me.clone();
        rt.create_query_with(
            move |rt| {
                let s = rt.get(*me.get().unwrap());
                if s == 0 {
                    rt.get(i)
                } else {
                    s
                }
            },
            QueryOpts {
                cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64)),
                ..Default::default()
            },
        )
    };
    me.set(q).unwrap();
    assert_eq!(rt.get(q), 3);
    rt.set(i, 9);
    assert_eq!(rt.get(q), 9);
}

/// Membership grows mid-episode with a lower-slot node, which becomes
/// the head: every member lands against the final cycle.
fn head_changes_mid_episode<C: Cells>() {
    let rt = Runtime::<C>::new();
    let cc = cell::<i64>();
    let z = {
        let cc = cc.clone();
        rt.create_query(move |rt| rt.get(*cc.get().unwrap()))
    };
    let (ac, bc) = (cell::<i64>(), cell::<i64>());
    let c = {
        let (ac, bc) = (ac.clone(), bc.clone());
        rt.create_query_with_fallback(
            move |rt| rt.get(*ac.get().unwrap()) + rt.get(*bc.get().unwrap()),
            |cy: &Cycle| cy.len() as i64,
        )
    };
    cc.set(c).unwrap();
    let a = rt.create_query(move |rt| {
        if rt.get(c) == 3 {
            rt.get(z);
        }
        0
    });
    let b = rt.create_query(move |rt| {
        rt.get(c);
        0
    });
    ac.set(a).unwrap();
    bc.set(b).unwrap();
    assert_eq!(rt.get(c), 4);
    let cycle = rt.cycle_of(c).unwrap();
    assert_eq!(cycle.head(), z.erase());
    assert_eq!(cycle.len(), 4);
    assert_eq!(rt.get(z), 4);
    rt.assert_invariant_i();
}

/// A group rooted at B merges into the lower cycle through A that one of
/// its member runs reaches, and lands as one group.
fn scc_merge<C: Cells>() {
    let rt = Runtime::<C>::new();
    let (bc, cc) = (cell::<i64>(), cell::<i64>());
    let c_runs = Arc::new(AtomicU32::new(0));
    let a = {
        let bc = bc.clone();
        rt.create_query_with_fallback(move |rt| rt.get(*bc.get().unwrap()) + 1, |_c: &Cycle| 1000)
    };
    let b = {
        let cc = cc.clone();
        rt.create_query_with_fallback(
            move |rt| rt.get(*cc.get().unwrap()) + 1,
            |cy: &Cycle| 10 * cy.len() as i64,
        )
    };
    let c = {
        let c_runs = c_runs.clone();
        rt.create_query(move |rt| {
            let v = rt.get(b);
            if c_runs.fetch_add(1, Ordering::Relaxed) >= 1 {
                v + rt.get(a)
            } else {
                v
            }
        })
    };
    bc.set(b).unwrap();
    cc.set(c).unwrap();
    assert_eq!(rt.get(a), 1000);
    assert_eq!(
        slots(&rt.cycle_of(a).unwrap()),
        vec![a.slot(), b.slot(), c.slot()]
    );
    assert_eq!(rt.get(b), 30);
    assert_eq!(rt.get(c), 1030);
    rt.assert_invariant_i();
}

/// Group verification fails to claim a member in each state but Dirty,
/// and the node it started from computes as an ordinary node.
fn group_verify_claim_failures<C: Cells>() {
    for forced in [
        NodeState::New,
        NodeState::Clean,
        NodeState::Failed,
        NodeState::Cyclic,
    ] {
        let rt = Runtime::<C>::new();
        let i = rt.create_input(0_i64);
        let pc = cell::<i64>();
        let x = {
            let pc = pc.clone();
            rt.create_query_with_fallback(
                move |rt| rt.get(*pc.get().unwrap()) + rt.get(i),
                |_c: &Cycle| 50,
            )
        };
        let p = rt.create_query(move |rt| rt.get(x) + 1);
        pc.set(p).unwrap();
        assert_eq!(rt.get(x), 50);
        rt.set(i, 1);
        assert_eq!(rt.state_of(p), NodeState::Dirty);
        rt.force_state_for_test(p, forced);
        if forced == NodeState::Clean || forced == NodeState::Failed || forced == NodeState::Cyclic
        {
            // X reads P as it stands, so P's record stays and X leaves
            // the group.
            let got = quiet(|| rt.get(x));
            if forced == NodeState::Clean {
                assert_eq!(got.unwrap(), 52, "{forced:?}");
                assert_eq!(rt.cycle_of(x), None, "{forced:?}");
            } else {
                assert!(
                    got.is_err() || rt.state_of(x) != NodeState::Computing,
                    "{forced:?}"
                );
            }
            continue;
        }
        assert_eq!(rt.get(x), 50, "{forced:?}");
        assert_eq!(rt.get(p), 51, "{forced:?}");
        assert_eq!(slots(&rt.cycle_of(x).unwrap()).len(), 2, "{forced:?}");
        rt.assert_invariant_i();
    }
}

/// Group verification reaches a member this thread is computing: the
/// claim fails, and the cycle is rediscovered through the stack.
fn group_verify_member_on_this_stack<C: Cells>() {
    let rt = Runtime::<C>::new();
    let i = rt.create_input(0_i64);
    let pc = cell::<i64>();
    let x = {
        let pc = pc.clone();
        rt.create_query_with_fallback(move |rt| rt.get(*pc.get().unwrap()), |_c: &Cycle| 50)
    };
    let p = rt.create_query_with_fallback(move |rt| rt.get(x) + rt.get(i), |_c: &Cycle| 60);
    pc.set(p).unwrap();
    assert_eq!(rt.get(x), 50);
    assert_eq!(rt.get(p), 60);
    rt.set(i, 1);
    assert_eq!(rt.get(p), 60);
    assert_eq!(rt.get(x), 50);
    rt.assert_invariant_i();
}

/// A member's slot is deleted and recycled: group verification finds its
/// generation moved and gives up the group.
fn group_verify_recycled_member<C: Cells>() {
    let rt = Runtime::<C>::new();
    let i = rt.create_input(0_i64);
    let x = rt.create_query(move |rt| rt.get(i) + 1);
    let p = rt.create_query(|_| 5_i64);
    assert_eq!(rt.get(x), 1);
    assert_eq!(rt.get(p), 5);
    rt.force_group_for_test(&[x.erase(), p.erase()]);
    rt.delete_node(p);
    let p2 = rt.create_query(|_| 6_i64);
    assert_eq!(p2.slot(), p.slot());
    rt.set(i, 1);
    assert_eq!(rt.get(x), 2);
    assert_eq!(rt.cycle_of(x), None);
    assert_eq!(rt.get(p2), 6);
}

/// The expansions and pops the first read after a break makes, against
/// the edges and members of the cycle it broke.
fn recovery_pops_are_bounded<C: Cells>() {
    // A 2-member CU {P, U} and a self-reading P, each broken by a set.
    let rt = Runtime::<C>::new();
    let g = rt.create_input(0_i64);
    let uc = cell::<i64>();
    let p = {
        let uc = uc.clone();
        rt.create_query(move |rt| {
            if rt.get(g) == 0 {
                rt.try_get(*uc.get().unwrap()).map_or(-1, |v| v)
            } else {
                1
            }
        })
    };
    let u = rt.create_query(move |rt| rt.try_get(p).map_or(-1, |v| v + 1));
    uc.set(u).unwrap();
    assert!(rt.try_get(p).is_err());
    rt.set(g, 1);
    let before = rt.walk_pops();
    assert_eq!(rt.get(p), 1);
    assert!(rt.walk_pops() - before <= 5, "{}", rt.walk_pops() - before);
    assert_eq!(rt.get(u), 2);
}

/// A 1000-member ring of `try_get` readers that stays cyclic after an
/// upstream set flips it to New, and the same ring broken mid-way: the
/// first read pops at most edges + members walk entries.
fn ring_recovery_pops<C: Cells>() {
    for broken in [false, true] {
        let rt = Runtime::<C>::new();
        let n = crate::test_size::test_size(1000);
        let g = rt.create_input(0_i64);
        let ring: Arc<OnceLock<Vec<Incr<i64>>>> = Arc::new(OnceLock::new());
        let members: Vec<Incr<i64>> = (0..n)
            .map(|k| {
                let ring = ring.clone();
                rt.create_query(move |rt| {
                    let gv = rt.get(g);
                    if broken && k == n / 2 && gv != 0 {
                        return 0;
                    }
                    rt.try_get(ring.get().unwrap()[(k + 1) % n])
                        .map_or(-1, |v| v + 1)
                })
            })
            .collect();
        ring.set(members.clone()).unwrap();
        assert!(rt.try_get(members[0]).is_err());
        rt.set(g, 1);
        for &m in &members {
            assert_eq!(rt.state_of(m), NodeState::New);
        }
        let before = rt.walk_pops();
        let got = rt.try_get(members[0]);
        let pops = rt.walk_pops() - before;
        let bound = (2 * n + n) as u64;
        assert!(pops <= bound, "broken {broken}: {pops} pops for {bound}");
        if broken {
            assert_eq!(got, Ok((n / 2) as i64));
        } else {
            assert_eq!(got.unwrap_err().len(), n);
        }
        rt.assert_invariant_i();
    }
}

/// A closure that catches a cycle unwind and returns a value lands that
/// value, as a `try_get` would have: X catches the cycle that Y and Z
/// close under it, lands its own value, and leaves the unwind kind clear.
fn swallowed_cycle_is_an_ordinary_result<C: Cells>() {
    let rt = Runtime::<C>::new();
    let zc = cell::<i64>();
    let y = {
        let zc = zc.clone();
        rt.create_query(move |rt| rt.get(*zc.get().unwrap()) + 1)
    };
    let z = rt.create_query(move |rt| rt.get(y) + 1);
    zc.set(z).unwrap();
    let x = rt.create_query(move |rt| quiet(|| rt.get(y)).unwrap_or(-5));
    assert_eq!(rt.get(x), -5);
    // X read Y, which the unwind left New, so X lands Dirty.
    assert_eq!(rt.state_of(x), NodeState::Dirty);
    assert_eq!(rt.state_of(y), NodeState::New);
    assert_eq!(rt.state_of(z), NodeState::New);
    assert_eq!(crate::cycle_state::unwind_kind(), None);
    let i = rt.create_input(1_i64);
    let r = rt.create_query(move |rt| rt.get(i) + 1);
    assert_eq!(rt.get(r), 2);
}

/// The plan's G1 = {X, P, U}: X has a Fallback and reads P with
/// `try_get`; P reads `i`, then U and X when `i` is 0 and R when it is 1;
/// U and R read P with `try_get`, and so does P read U and R, because
/// once an episode puts them in CU a `get` between them panics by R0. With `i` = 0, `get(X)` lands X Clean and
/// P and U Cyclic. After `set(i, 1)` a read of P (or, `at_x`, of X) runs
/// G1's group verify from X, whose claim fails on New P or U, so X takes
/// an ordinary compute that lands P and R Cyclic in a new group, and U
/// and X land Clean through single-node landings that remove their G1
/// records.
fn g1_group_verify_falls_back<C: Cells>(at_x: bool) {
    let rt = Runtime::<C>::new();
    let i = rt.create_input(0_i64);
    let pc = cell::<i64>();
    let rc = cell::<i64>();
    let x = {
        let pc = pc.clone();
        rt.create_query_with_fallback(
            move |rt| rt.try_get(*pc.get().unwrap()).unwrap_or(-1),
            |_c: &Cycle| 100,
        )
    };
    let u = {
        let pc = pc.clone();
        rt.create_query(move |rt| rt.try_get(*pc.get().unwrap()).unwrap_or(-2))
    };
    let r = {
        let pc = pc.clone();
        rt.create_query(move |rt| rt.try_get(*pc.get().unwrap()).unwrap_or(-3))
    };
    let p = {
        let rc = rc.clone();
        rt.create_query(move |rt| {
            if rt.get(i) == 0 {
                rt.try_get(u).unwrap_or(-4) + rt.get(x)
            } else {
                rt.try_get(*rc.get().unwrap()).unwrap_or(-5)
            }
        })
    };
    pc.set(p).unwrap();
    rc.set(r).unwrap();

    assert_eq!(rt.get(x), 100);
    assert_eq!(rt.state_of(x), NodeState::Clean);
    assert_eq!(rt.state_of(p), NodeState::Cyclic);
    assert_eq!(rt.state_of(u), NodeState::Cyclic);
    rt.assert_invariant_i();

    rt.set(i, 1);
    assert_eq!(rt.state_of(p), NodeState::New);
    assert_eq!(rt.state_of(u), NodeState::New);
    assert_eq!(rt.state_of(x), NodeState::Dirty);
    rt.group_verifies.lock().unwrap().clear();
    if at_x {
        assert_eq!(rt.get(x), -1);
    } else {
        assert!(rt.try_get(p).is_err());
        assert_eq!(rt.get(x), -1);
    }
    assert!(rt.group_verifies.lock().unwrap().contains(&x.slot()));
    assert!(rt.try_get(p).is_err());
    assert_eq!(rt.get(u), -2);
    assert_eq!(rt.state_of(p), NodeState::Cyclic);
    assert_eq!(rt.state_of(r), NodeState::Cyclic);
    assert_eq!(rt.state_of(u), NodeState::Clean);
    assert_eq!(rt.state_of(x), NodeState::Clean);
    rt.assert_invariant_i();
}

fn g1_group_verify_falls_back_from_p<C: Cells>() {
    g1_group_verify_falls_back::<C>(false);
}

fn g1_group_verify_falls_back_from_x<C: Cells>() {
    g1_group_verify_falls_back::<C>(true);
}

/// R reads P with `try_get` and P reads R with `get`. In the episode's
/// iteration R's `try_get(P)` runs P for its value (R5), and P's `get(R)`
/// meets R on the stack (R3) and unwinds P's run. That unwind is R's
/// `try_get`'s `Err`, as a nested compute's would be, so every run of R
/// returns from its read.
fn member_run_unwind_is_the_readers_err<C: Cells>() {
    let rt = Runtime::<C>::new();
    let pc = cell::<i64>();
    let started = Arc::new(AtomicU32::new(0));
    let returned = Arc::new(AtomicU32::new(0));
    let r = {
        let (pc, started, returned) = (pc.clone(), started.clone(), returned.clone());
        rt.create_query(move |rt| {
            started.fetch_add(1, Ordering::Relaxed);
            let v = rt.try_get(*pc.get().unwrap());
            returned.fetch_add(1, Ordering::Relaxed);
            v.unwrap_or(-3)
        })
    };
    let p = rt.create_query(move |rt| rt.get(r));
    pc.set(p).unwrap();
    assert!(quiet(|| rt.try_get(p)).unwrap().is_err());
    assert!(started.load(Ordering::Relaxed) >= 2);
    assert_eq!(
        started.load(Ordering::Relaxed),
        returned.load(Ordering::Relaxed)
    );
}

both!(
    a_panic_after_a_swallowed_cycle_lands_failed,
    member_run_unwind_is_the_readers_err,
    g1_group_verify_falls_back_from_p,
    g1_group_verify_falls_back_from_x,
    rule_a_stores_zero_on_a_new_claim_that_wrote_nothing,
    rule_b_fails_a_verify_at_zero,
    rule_b_fails_a_group_at_zero,
    scenario_a,
    scenario_b,
    former_member_flagged_panic,
    panic_in_an_iteration,
    fixpoint_value_drop_panics,
    cyclic_landing_is_a_change,
    dirty_cyclic_landing_reads_as_the_cycle,
    a_panic_releases_the_claims_its_frame_deferred,
    a_walk_root_deferred_under_its_walk,
    cyclic_group_grows,
    early_cutoff_through_a_group,
    input_read_only_in_iteration_one,
    head_changes_mid_episode,
    scc_merge,
    group_verify_claim_failures,
    group_verify_member_on_this_stack,
    group_verify_recycled_member,
    recovery_pops_are_bounded,
    swallowed_cycle_is_an_ordinary_result,
);

fn on_deep_stack(body: fn()) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_ring_recovery_pops() {
    on_deep_stack(ring_recovery_pops::<Local>);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_ring_recovery_pops() {
    on_deep_stack(ring_recovery_pops::<Shared>);
}

/// A read of a node of runtime R from another runtime's closure, while
/// that node is on this thread's R stack, is a cycle no group resolves:
/// `try_get` returns `Err` and `get` panics.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_cross_runtime_reentry() {
    for use_get in [false, true] {
        let r1 = Arc::new(Runtime::<Shared>::new());
        let r2 = Arc::new(Runtime::<Shared>::new());
        let ac = cell::<i64>();
        let b = {
            let (r1, ac) = (Arc::downgrade(&r1), ac.clone());
            r2.create_query(move |_| {
                let r1 = r1.upgrade().unwrap();
                let a = *ac.get().unwrap();
                if use_get {
                    r1.get(a)
                } else {
                    r1.try_get(a).map_or(-1, |v| v)
                }
            })
        };
        let i = r1.create_input(10_i64);
        let a = {
            let r2 = Arc::downgrade(&r2);
            r1.create_query_with_fallback(
                move |rt| rt.get(i) + r2.upgrade().unwrap().get(b),
                |_c: &Cycle| 77,
            )
        };
        ac.set(a).unwrap();
        if use_get {
            let err = quiet(|| r1.get(a)).unwrap_err();
            let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
            assert!(msg.contains("dependency cycle"), "{msg}");
            assert_eq!(r1.state_of(a), NodeState::New);
            assert_eq!(r2.state_of(b), NodeState::New);
        } else {
            assert_eq!(r1.get(a), 9);
            assert_eq!(r1.cycle_of(a), None);
        }
    }
}

/// Group verify decides on `IN_GROUP` through a load after its claim:
/// `n`'s `compute_one` parks between its hold and its claim while another
/// thread changes `n`'s group membership, each test in one direction.
mod flags_reload {
    use std::sync::Arc;
    use std::thread;

    use super::super::race_tests::{id, park_at, Gate};
    use super::super::{ParkPoint, Runtime};
    use super::cell;
    use crate::cells::Shared;
    use crate::cycle::Cycle;
    use crate::state::NodeState;

    /// While the first thread is parked, a second computes `n` as the root
    /// of a 2-member cycle, and a `set` of the cycle's external dep flags
    /// the episode, so `n` lands Dirty with a record. The first thread
    /// claims `n` from Dirty and runs group verify.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn shared_in_group_set_between_hold_and_claim() {
        let rt = Arc::new(Runtime::<Shared>::new());
        let e = rt.create_input(0_i64);
        let sw = rt.create_input(0_i64);
        let mc = cell::<i64>();
        let m_gate = Gate::new();
        let n = {
            let mc = mc.clone();
            rt.create_query_with_fallback(
                move |rt| {
                    if rt.get(sw) == 0 {
                        rt.get(e)
                    } else {
                        rt.get(*mc.get().unwrap()) + rt.get(e)
                    }
                },
                |_c: &Cycle| 100,
            )
        };
        let m = {
            let m_gate = m_gate.clone();
            rt.create_query(move |rt| {
                let v = rt.get(n);
                m_gate.park();
                v + 1
            })
        };
        mc.set(m).unwrap();
        assert_eq!(rt.get(n), 0);
        rt.set(sw, 1);
        assert_eq!(rt.state_of(n), NodeState::Dirty);

        let claim = Gate::new();
        park_at(&rt, vec![(ParkPoint::BeforeClaim(id(n)), claim.clone())]);
        let first = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(n))
        };
        claim.wait_parked();
        m_gate.arm();
        let second = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(n))
        };
        m_gate.wait_parked();
        rt.set(e, 5);
        m_gate.release();
        assert_eq!(second.join().unwrap(), 100);
        assert_eq!(rt.state_of(n), NodeState::Dirty);
        assert_eq!(rt.state_of(m), NodeState::Dirty);
        rt.group_verifies.lock().unwrap().clear();
        claim.release();
        assert_eq!(first.join().unwrap(), 100);
        assert_eq!(rt.group_verifies.lock().unwrap().first(), Some(&id(n).0));
        assert_eq!(rt.get(m), 101);
        rt.assert_invariant_i();
    }

    /// `n` starts Dirty as a group member, after a `set` that makes its
    /// closure stop reading the cycle. While the first thread is parked, a
    /// second claims `n`, group verify finds the change, the episode
    /// dissolves and `n` lands through the single-node path, which removes
    /// its record, and a `set` walks it Dirty again. The first thread then
    /// takes the single-node path; group verify's lookup asserts if a stale
    /// bit reaches it.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn shared_in_group_cleared_between_hold_and_claim() {
        let rt = Arc::new(Runtime::<Shared>::new());
        let e = rt.create_input(0_i64);
        let sw = rt.create_input(1_i64);
        let mc = cell::<i64>();
        let n = {
            let mc = mc.clone();
            rt.create_query_with_fallback(
                move |rt| {
                    if rt.get(sw) == 1 {
                        rt.get(*mc.get().unwrap()) + rt.get(e)
                    } else {
                        rt.get(e)
                    }
                },
                |_c: &Cycle| 100,
            )
        };
        let m = rt.create_query(move |rt| rt.get(n) + 1);
        mc.set(m).unwrap();
        assert_eq!(rt.get(n), 100);
        assert_eq!(rt.get(m), 101);
        rt.set(sw, 0);
        assert_eq!(rt.state_of(n), NodeState::Dirty);

        let claim = Gate::new();
        park_at(&rt, vec![(ParkPoint::BeforeClaim(id(n)), claim.clone())]);
        let first = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(n))
        };
        claim.wait_parked();
        let second = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(n))
        };
        assert_eq!(second.join().unwrap(), 0);
        rt.set(e, 7);
        assert_eq!(rt.state_of(n), NodeState::Dirty);
        rt.group_verifies.lock().unwrap().clear();
        claim.release();
        assert_eq!(first.join().unwrap(), 7);
        assert!(rt.group_verifies.lock().unwrap().is_empty());
        assert_eq!(rt.get(m), 8);
        rt.assert_invariant_i();
    }
}
