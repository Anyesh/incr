//! Cycles: the `verified_at = 0` rules, landings around panics, the empty
//! slot of a `Cyclic` node, group verification's claim failures, merges,
//! recovery bounds and cross-runtime reads. Each case runs under `Local`
//! and `Shared`.

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{Cycle, CycleStrategy, Fixpoint, QueryOpts, Replay};
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

/// A group verify's episode panics in member b: b lands Failed, released
/// with the Fixpoint co-member a it reads. A later set that reaches only a
/// must still let b recompute, so with j back at
/// 1 the cycle converges to a = 2, b = 3 as it first did.
fn wave6_failed_member_over_dirty_co_member<C: Cells>() {
    let rt = Runtime::<C>::new();
    let j = rt.create_input(1_i64);
    let bc = cell::<i64>();
    let a = {
        let bc = bc.clone();
        rt.create_query_with(
            move |rt| {
                let v = rt.get(*bc.get().unwrap());
                if rt.get(j) == 0 {
                    v
                } else {
                    v.min(2)
                }
            },
            QueryOpts {
                cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64)),
                ..Default::default()
            },
        )
    };
    let b = rt.create_query(move |rt| {
        let v = rt.get(a);
        assert!(v < 3, "b boom");
        v + 1
    });
    bc.set(b).unwrap();
    assert_eq!(rt.get(a), 2);
    assert_eq!(rt.get(b), 3);

    rt.set(j, 0);
    let err = quiet(|| rt.get(b)).unwrap_err();
    assert_eq!(err.downcast_ref::<&str>(), Some(&"b boom"));
    assert_eq!(rt.state_of(b), NodeState::Failed);

    rt.set(j, 1);
    assert_ne!(rt.state_of(b), NodeState::Failed);
    assert_eq!(quiet(|| rt.get(b)).ok(), Some(3));
    assert_eq!(rt.get(a), 2);
    rt.assert_invariant_i();
}

/// As [`wave6_failed_member_over_dirty_co_member`], with the set reaching
/// b only through two co-members: j is read by c, c by a, a by b.
fn failed_member_two_hops_from_the_set<C: Cells>() {
    let rt = Runtime::<C>::new();
    let j = rt.create_input(1_i64);
    let cc = cell::<i64>();
    let a = {
        let cc = cc.clone();
        rt.create_query_with(
            move |rt| rt.get(*cc.get().unwrap()),
            QueryOpts {
                cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64)),
                ..Default::default()
            },
        )
    };
    let b = rt.create_query(move |rt| {
        let v = rt.get(a);
        assert!(v < 3, "b boom");
        v + 1
    });
    let c = rt.create_query(move |rt| {
        let v = rt.get(b);
        if rt.get(j) == 0 {
            v
        } else {
            v.min(2)
        }
    });
    cc.set(c).unwrap();
    assert_eq!(rt.get(a), 2);
    assert_eq!(rt.get(b), 3);

    rt.set(j, 0);
    let err = quiet(|| rt.get(b)).unwrap_err();
    assert_eq!(err.downcast_ref::<&str>(), Some(&"b boom"));
    assert_eq!(rt.state_of(b), NodeState::Failed);

    rt.set(j, 1);
    assert_ne!(rt.state_of(b), NodeState::Failed);
    assert_eq!(quiet(|| rt.get(b)).ok(), Some(3));
    assert_eq!(rt.get(c), 2);
    rt.assert_invariant_i();
}

/// A co-member reads k only in the episode that fails b, so k's edge is
/// only in its run's record; setting k must still let b recompute.
fn failed_member_cured_through_a_recorded_dep<C: Cells>() {
    let rt = Runtime::<C>::new();
    let j = rt.create_input(1_i64);
    let k = rt.create_input(0_i64);
    let bc = cell::<i64>();
    let a = {
        let bc = bc.clone();
        rt.create_query_with(
            move |rt| {
                let v = rt.get(*bc.get().unwrap());
                if rt.get(j) == 0 && rt.get(k) == 0 {
                    v
                } else {
                    v.min(2)
                }
            },
            QueryOpts {
                cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64)),
                ..Default::default()
            },
        )
    };
    let b = rt.create_query(move |rt| {
        let v = rt.get(a);
        assert!(v < 3, "b boom");
        v + 1
    });
    bc.set(b).unwrap();
    assert_eq!(rt.get(a), 2);
    assert_eq!(rt.get(b), 3);

    rt.set(j, 0);
    let err = quiet(|| rt.get(b)).unwrap_err();
    assert_eq!(err.downcast_ref::<&str>(), Some(&"b boom"));
    assert_eq!(rt.state_of(b), NodeState::Failed);

    rt.set(k, 1);
    assert_ne!(rt.state_of(b), NodeState::Failed);
    assert_eq!(quiet(|| rt.get(b)).ok(), Some(3));
    rt.assert_invariant_i();
}

/// The cycle forms inside root x's own frame, which panics: x lands
/// Failed through its compute's panic path, over the member y it read.
fn failed_frame_node_over_its_released_member<C: Cells>() {
    let rt = Runtime::<C>::new();
    let j = rt.create_input(1_i64);
    let yc = cell::<i64>();
    let x = {
        let yc = yc.clone();
        rt.create_query_with(
            move |rt| {
                let v = rt.get(*yc.get().unwrap());
                assert!(v < 3, "x boom");
                v + 1
            },
            QueryOpts {
                cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64)),
                ..Default::default()
            },
        )
    };
    let y = rt.create_query(move |rt| if rt.get(j) == 1 { 1 } else { rt.get(x) + 5 });
    yc.set(y).unwrap();
    assert_eq!(rt.get(x), 2);

    rt.set(j, 0);
    let err = quiet(|| rt.get(x)).unwrap_err();
    assert_eq!(err.downcast_ref::<&str>(), Some(&"x boom"));
    assert_eq!(rt.state_of(x), NodeState::Failed);

    rt.set(j, 1);
    assert_ne!(rt.state_of(x), NodeState::Failed);
    assert_eq!(quiet(|| rt.get(x)).ok(), Some(2));
    rt.assert_invariant_i();
}

/// An `i64` whose compare panics only between 3 and 6.
#[derive(Clone, Debug)]
struct EqBomb(i64);

impl PartialEq for EqBomb {
    fn eq(&self, other: &Self) -> bool {
        assert!(
            (self.0.min(other.0), self.0.max(other.0)) != (3, 6),
            "EqBomb"
        );
        self.0 == other.0
    }
}

/// Step 3's write of b compares b's stored 3 with its converged 6 and
/// panics: b lands Failed, and a set reaching it through the co-members
/// must still let it recompute.
fn failed_compare_over_a_co_member<C: Cells>() {
    let rt = Runtime::<C>::new();
    let j = rt.create_input(1_i64);
    let cc = cell::<i64>();
    let a = {
        let cc = cc.clone();
        rt.create_query_with(
            move |rt| rt.get(*cc.get().unwrap()),
            QueryOpts {
                cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64)),
                ..Default::default()
            },
        )
    };
    let b = rt.create_query(move |rt| EqBomb(rt.get(a) + 1));
    let c = rt.create_query(move |rt| {
        let v = rt.get(b).0;
        if rt.get(j) == 0 {
            v.min(5)
        } else {
            v.min(2)
        }
    });
    cc.set(c).unwrap();
    assert_eq!(rt.get(a), 2);
    assert_eq!(rt.get(b).0, 3);

    rt.set(j, 0);
    let err = quiet(|| rt.get(b)).unwrap_err();
    assert_eq!(err.downcast_ref::<&str>(), Some(&"EqBomb"));
    assert_eq!(rt.state_of(b), NodeState::Failed);

    rt.set(j, 1);
    assert_ne!(rt.state_of(b), NodeState::Failed);
    assert_eq!(quiet(|| rt.get(b)).ok().map(|v| v.0), Some(3));
    rt.assert_invariant_i();
}

both!(
    wave6_failed_member_over_dirty_co_member,
    failed_member_two_hops_from_the_set,
    failed_member_cured_through_a_recorded_dep,
    failed_frame_node_over_its_released_member,
    failed_compare_over_a_co_member,
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

macro_rules! pairs {
    ($($name:ident),* $(,)?) => {
        $(
            mod $name {
                use super::{Local, Shared};

                #[test]
                #[cfg_attr(miri, ignore)]
                fn local_local() {
                    super::$name::<Local, Local>();
                }

                #[test]
                #[cfg_attr(miri, ignore)]
                fn local_shared() {
                    super::$name::<Local, Shared>();
                }

                #[test]
                #[cfg_attr(miri, ignore)]
                fn shared_local() {
                    super::$name::<Shared, Local>();
                }

                #[test]
                #[cfg_attr(miri, ignore)]
                fn shared_shared() {
                    super::$name::<Shared, Shared>();
                }
            }
        )*
    };
}

pairs!(
    cross_runtime_reentry,
    cross_runtime_read_records_nothing,
    foreign_reader_is_doomed_and_retried,
    foreign_ring_read_is_not_rerun,
);

/// A runtime that another runtime's closures read. Test-only: it outlives
/// every call into those closures, and a Local one never leaves its thread.
struct OtherRt<C: Cells>(*const Runtime<C>);

// SAFETY: see `OtherRt`.
unsafe impl<C: Cells> Send for OtherRt<C> {}
unsafe impl<C: Cells> Sync for OtherRt<C> {}

impl<C: Cells> std::ops::Deref for OtherRt<C> {
    type Target = Runtime<C>;

    fn deref(&self) -> &Runtime<C> {
        // SAFETY: see `OtherRt`.
        unsafe { &*self.0 }
    }
}

/// A read of a node of runtime R from another runtime's closure, while
/// that node is on this thread's stack, is a cycle no group resolves:
/// `try_get` returns `Err` and `get` panics.
fn cross_runtime_reentry<A: Cells, B: Cells>() {
    for use_get in [false, true] {
        let r1 = Box::new(Runtime::<A>::new());
        let r2 = Box::new(Runtime::<B>::new());
        let ac = cell::<i64>();
        let b = {
            let (r1, ac) = (OtherRt(&*r1), ac.clone());
            r2.create_query(move |_| {
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
            let r2 = OtherRt(&*r2);
            r1.create_query_with_fallback(move |rt| rt.get(i) + r2.get(b), |_c: &Cycle| 77)
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

/// A read of R's node from another runtime's closure nested in R's compute
/// records no edge for R's computing node, and R's own reads after that
/// closure returns still record.
fn cross_runtime_read_records_nothing<A: Cells, B: Cells>() {
    let r1 = Box::new(Runtime::<A>::new());
    let r2 = Box::new(Runtime::<B>::new());
    let x = r1.create_input(1_i64);
    let y = r1.create_input(100_i64);
    let b = {
        let r1 = OtherRt(&*r1);
        r2.create_query(move |_| r1.get(x))
    };
    let runs = Arc::new(AtomicU32::new(0));
    let a = {
        let (r2, runs) = (OtherRt(&*r2), runs.clone());
        r1.create_query(move |rt| {
            runs.fetch_add(1, Ordering::Relaxed);
            r2.get(b) + rt.get(y)
        })
    };
    assert_eq!(r1.get(a), 101);
    r1.set(x, 2);
    assert_eq!(r1.get(a), 101);
    assert_eq!(runs.load(Ordering::Relaxed), 1);
    r1.set(y, 200);
    assert_eq!(r1.get(a), 201);
    assert_eq!(runs.load(Ordering::Relaxed), 2);
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

/// What a replayable frame reads first in [`foreign_reader_is_doomed_and_retried`].
#[derive(Clone, Copy, Debug)]
enum FirstRead {
    Own,
    /// Another runtime's input, Clean when read.
    ForeignClean,
    /// Another runtime's query over that input, New when read, so the
    /// read computes it.
    ForeignComputed,
    /// That query, which the last leaf reads again after its set, so the
    /// retry finds it landed Clean on the set's value.
    ForeignLandedClean,
}

/// A replayable frame at budget 1 that reads 64 leaves is doomed once and
/// retried, whatever it first reads; see [`foreign_reader_is_doomed_with`].
/// The last leaf sets another runtime's input when it runs, which is after the first read in the unbounded run
/// and between the doomed run and its retry here. A retry takes its doomed
/// run's foreign read outcome again, Clean or not, so it agrees with the
/// unbounded run.
fn foreign_reader_is_doomed_and_retried<A: Cells, B: Cells>() {
    foreign_reader_is_doomed_with::<A, B>(64);
}

/// [`foreign_reader_is_doomed_and_retried`] with `n` leaves.
fn foreign_reader_is_doomed_with<A: Cells, B: Cells>(n: u64) {
    for first in [
        FirstRead::Own,
        FirstRead::ForeignClean,
        FirstRead::ForeignComputed,
        FirstRead::ForeignLandedClean,
    ] {
        let runs = doomed_reader_runs::<A, B>(n, first);
        // `landed_foreign_reader_runs_twice` holds the landed case's count.
        if !matches!(first, FirstRead::ForeignLandedClean) {
            assert_eq!(runs, 2, "{first:?}");
        }
    }
}

/// One case of [`foreign_reader_is_doomed_with`], asserting the reader's
/// value and its doom and returning how many times its closure ran.
fn doomed_reader_runs<A: Cells, B: Cells>(n: u64, first: FirstRead) -> u32 {
    let r1 = Box::new(Runtime::<A>::new());
    let r2 = Box::new(Runtime::<B>::new());
    r1.set_placeholder(0_u64);
    r1.set_max_frames(1);
    let x = r2.create_input(1000_u64);
    let q = r2.create_query(move |rt| rt.get(x));
    let mut leaves: Vec<Incr<u64>> = (0..n - 1).map(|i| r1.create_query(move |_| i)).collect();
    let r2_ = OtherRt(&*r2);
    leaves.push(r1.create_query(move |_| {
        r2_.set(x, 2000);
        if matches!(first, FirstRead::ForeignLandedClean) {
            r2_.get(q);
        }
        n - 1
    }));
    let runs = Arc::new(AtomicU32::new(0));
    let wide = {
        let (r2, runs) = (OtherRt(&*r2), runs.clone());
        r1.create_query_with(
            move |rt| {
                runs.fetch_add(1, Ordering::Relaxed);
                let base = match first {
                    FirstRead::Own => 1000,
                    FirstRead::ForeignClean => r2.get(x),
                    FirstRead::ForeignComputed | FirstRead::ForeignLandedClean => r2.get(q),
                };
                base + leaves.iter().map(|&l| rt.get(l)).sum::<u64>()
            },
            QueryOpts {
                replay: Replay::Allowed,
                ..Default::default()
            },
        )
    };
    assert_eq!(r1.get(wide), 1000 + (0..n).sum::<u64>(), "{first:?}");
    assert!(!r1.dooms.lock().unwrap().is_empty(), "{first:?}");
    runs.load(Ordering::Relaxed)
}

#[test]
#[ignore = "T10-speculative-landing-third-foreign-reader-run"]
fn landed_foreign_reader_runs_twice() {
    let first = FirstRead::ForeignLandedClean;
    assert_eq!(doomed_reader_runs::<Local, Local>(64, first), 2);
    assert_eq!(doomed_reader_runs::<Local, Shared>(64, first), 2);
    assert_eq!(doomed_reader_runs::<Shared, Local>(64, first), 2);
    assert_eq!(doomed_reader_runs::<Shared, Shared>(64, first), 2);
}

/// A value whose `Clone` reads another runtime's input, as the value-code
/// contract allows of a Clean node.
struct ReadsOther<B: Cells> {
    rt: OtherRt<B>,
    x: Incr<u64>,
    seen: u64,
}

impl<B: Cells> Clone for ReadsOther<B> {
    fn clone(&self) -> Self {
        Self {
            rt: OtherRt(self.rt.0),
            x: self.x,
            seen: self.rt.get(self.x),
        }
    }
}

impl<B: Cells> PartialEq for ReadsOther<B> {
    fn eq(&self, other: &Self) -> bool {
        self.seen == other.seen
    }
}

/// A replayable reader within a budget of 8 frames whose read of another
/// runtime's input is in value code, where nothing is kept for a retry,
/// over a chain of 20 replayable queries whose deepest sets that input.
/// The chain runs past the budget and is doomed, but the reader is held,
/// so it runs once and agrees with the unbounded run.
fn value_code_foreign_read_holds_its_frame<A: Cells, B: Cells>() {
    let r1 = Box::new(Runtime::<A>::new());
    let r2 = Box::new(Runtime::<B>::new());
    r1.set_placeholder(0_u64);
    r1.set_max_frames(8);
    let x = r2.create_input(1000_u64);
    let v = r1.create_input(ReadsOther {
        rt: OtherRt(&*r2),
        x,
        seen: 0,
    });
    let replayable = || QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    };
    let r2_ = OtherRt(&*r2);
    let mut below = r1.create_query_with(
        move |_| {
            r2_.set(x, 2000);
            1
        },
        replayable(),
    );
    for _ in 0..19 {
        below = r1.create_query_with(move |rt| rt.get(below) + 1, replayable());
    }
    let runs = Arc::new(AtomicU32::new(0));
    let reader = {
        let runs = runs.clone();
        r1.create_query_with(
            move |rt| {
                runs.fetch_add(1, Ordering::Relaxed);
                rt.get(v).seen + rt.get(below)
            },
            replayable(),
        )
    };
    assert_eq!(r1.get(reader), 1020);
    assert!(!r1.dooms.lock().unwrap().is_empty());
    assert_eq!(runs.load(Ordering::Relaxed), 1);
}

#[test]
fn local_shared_value_code_foreign_read_holds_its_frame() {
    value_code_foreign_read_holds_its_frame::<Local, Shared>();
}

#[test]
fn shared_shared_value_code_foreign_read_holds_its_frame() {
    value_code_foreign_read_holds_its_frame::<Shared, Shared>();
}

/// Seed 0x3f39 shrunk: rt1's ring 7 -> 8 -> 9 -> 11 -> 7 of replayable
/// queries with no strategy, 11 reading 7 by `try_get`, and rt2's X =
/// rt1 8 + Y. Read once, rt1 8 is 7777, but read again after the first
/// read left the ring Dirty, the walk enters at 7 and closes a cycle. X,
/// doomed by its read of Y past the budget, would read 8 again on its
/// retry; having read rt1 it is never doomed, so every budget agrees
/// with the unbounded run.
fn foreign_ring_read_is_not_rerun<A: Cells, B: Cells>() {
    let run = |frames: u32| {
        let rt1 = Box::new(Runtime::<A>::new());
        let rt2 = Box::new(Runtime::<B>::new());
        rt1.set_max_frames(frames);
        rt1.set_placeholder(0_u64);
        rt2.set_max_frames(frames);
        rt2.set_placeholder(0_u64);
        let c7 = cell::<u64>();
        let replayable = || QueryOpts {
            replay: Replay::Allowed,
            ..Default::default()
        };
        let n11 = {
            let c7 = c7.clone();
            rt1.create_query_with(
                move |rt| rt.try_get(*c7.get().unwrap()).unwrap_or(7777),
                replayable(),
            )
        };
        let n9 = rt1.create_query_with(move |rt| rt.get(n11), replayable());
        let n8 = rt1.create_query_with(move |rt| rt.get(n9), replayable());
        let n7 = rt1.create_query_with(move |rt| rt.get(n8), replayable());
        c7.set(n7).unwrap();
        let y = rt2.create_query_with(|_| 0_u64, replayable());
        let x = {
            let r1 = OtherRt(&*rt1);
            rt2.create_query_with(move |rt| r1.get(n8) + rt.get(y), replayable())
        };
        rt2.try_get(x).map_err(|c| c.participants().len())
    };
    assert_eq!(run(u32::MAX), Ok(7777));
    for frames in [1, 2] {
        assert_eq!(run(frames), Ok(7777), "budget {frames}");
    }
}

/// A replayable chain of `n` levels read top-down in runtime A, where every
/// level reads an input of runtime B and then the next level. Run on a
/// spawned thread's default stack, which the frame budget must bound, as
/// it does a chain that reads only its own runtime.
fn foreign_reading_chain_stays_within_the_stack<A: Cells, B: Cells>() {
    let n = crate::test_size::test_size(100_000);
    let got = std::thread::spawn(move || {
        let r1 = Box::new(Runtime::<A>::new());
        let r2 = Box::new(Runtime::<B>::new());
        r1.set_placeholder(0_i64);
        let x = r2.create_input(1_i64);
        let mut below: Option<Incr<i64>> = None;
        for _ in 0..n {
            let (r2, next) = (OtherRt(&*r2), below);
            below = Some(r1.create_query_with(
                move |rt| r2.get(x) + next.map_or(0, |b| rt.get(b)),
                QueryOpts {
                    replay: Replay::Allowed,
                    ..Default::default()
                },
            ));
        }
        r1.get(below.unwrap())
    })
    .join()
    .unwrap();
    assert_eq!(got, n as i64);
}

mod foreign_chain {
    use crate::cells::{Local, Shared};

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "a debug build's frames at the default budget overflow a 2 MiB stack"
    )]
    #[cfg_attr(miri, ignore)]
    fn local_local() {
        super::foreign_reading_chain_stays_within_the_stack::<Local, Local>();
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "a debug build's frames at the default budget overflow a 2 MiB stack"
    )]
    #[cfg_attr(miri, ignore)]
    fn local_shared() {
        super::foreign_reading_chain_stays_within_the_stack::<Local, Shared>();
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "a debug build's frames at the default budget overflow a 2 MiB stack"
    )]
    #[cfg_attr(miri, ignore)]
    fn shared_local() {
        super::foreign_reading_chain_stays_within_the_stack::<Shared, Local>();
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "a debug build's frames at the default budget overflow a 2 MiB stack"
    )]
    #[cfg_attr(miri, ignore)]
    fn shared_shared() {
        super::foreign_reading_chain_stays_within_the_stack::<Shared, Shared>();
    }
}

/// A panic payload no copy can stand for.
struct Uncopyable;

/// Silences the panics of [`Uncopyable`] payloads, one per level of
/// `foreign_panicking_chain_stays_within_the_stack`, and leaves every
/// other panic to the default hook.
fn quiet_uncopyable() {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !info.payload().is::<Uncopyable>() {
                default(info);
            }
        }));
    });
}

/// As [`foreign_reading_chain_stays_within_the_stack`], with every level
/// catching the panic of its own node of runtime B, whose payload no kept
/// read can carry. Such a reader stays undoomed only within the budget, so
/// the chain is bounded as one with a ticket payload is.
/// `b_frames` is B's frame budget, if not the default: the reader is A's
/// frame, so A's budget bounds it whatever B's is.
fn foreign_panicking_chain_stays_within_the_stack<A: Cells, B: Cells>(b_frames: Option<u32>) {
    let n = crate::test_size::test_size(100_000);
    assert_eq!(
        foreign_panicking_chain::<A, B>(n, None, b_frames).0,
        n as i64
    );
}

/// The chain of [`foreign_panicking_chain_stays_within_the_stack`], `n`
/// levels with the given frame budgets, giving A's result and how many
/// times each level ran, the top level first.
fn foreign_panicking_chain<A: Cells, B: Cells>(
    n: usize,
    a_frames: Option<u32>,
    b_frames: Option<u32>,
) -> (i64, Vec<u32>) {
    quiet_uncopyable();
    std::thread::spawn(move || {
        let r1 = Box::new(Runtime::<A>::new());
        let r2 = Box::new(Runtime::<B>::new());
        r1.set_placeholder(0_i64);
        if let Some(frames) = a_frames {
            r1.set_max_frames(frames);
        }
        if let Some(frames) = b_frames {
            r2.set_max_frames(frames);
        }
        let runs: Arc<Vec<AtomicU32>> = Arc::new((0..n).map(|_| AtomicU32::new(0)).collect());
        let mut below: Option<Incr<i64>> = None;
        for level in (0..n).rev() {
            let bq = r2.create_query(|_| -> i64 { std::panic::panic_any(Uncopyable) });
            let (r2, next, runs) = (OtherRt(&*r2), below, runs.clone());
            below = Some(r1.create_query_with(
                move |rt| {
                    runs[level].fetch_add(1, Ordering::Relaxed);
                    let caught = quiet(|| r2.get(bq)).is_err();
                    i64::from(caught) + next.map_or(0, |b| rt.get(b))
                },
                QueryOpts {
                    replay: Replay::Allowed,
                    ..Default::default()
                },
            ));
        }
        let got = r1.get(below.unwrap());
        (
            got,
            runs.iter().map(|r| r.load(Ordering::Relaxed)).collect(),
        )
    })
    .join()
    .unwrap()
}

mod foreign_panic_chain {
    use crate::cells::{Local, Shared};

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "a debug build's frames at the default budget overflow a 2 MiB stack"
    )]
    #[cfg_attr(miri, ignore)]
    fn local_shared() {
        super::foreign_panicking_chain_stays_within_the_stack::<Local, Shared>(None);
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "a debug build's frames at the default budget overflow a 2 MiB stack"
    )]
    #[cfg_attr(miri, ignore)]
    fn shared_shared() {
        super::foreign_panicking_chain_stays_within_the_stack::<Shared, Shared>(None);
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "a debug build's frames at the default budget overflow a 2 MiB stack"
    )]
    #[cfg_attr(miri, ignore)]
    fn local_shared_with_b_unbudgeted() {
        super::foreign_panicking_chain_stays_within_the_stack::<Local, Shared>(Some(u32::MAX));
    }
}

/// Small foreign-read cases that run under Miri: foreign reads, Clean or
/// not, kept for a doomed frame's retry, and
/// an uncopyable foreign panic holding its reader within the budget only.
mod foreign_miri {
    use crate::cells::{Local, Shared};

    #[test]
    fn doomed_reader_local_shared() {
        super::foreign_reader_is_doomed_with::<Local, Shared>(4);
    }

    #[test]
    fn doomed_reader_shared_local() {
        super::foreign_reader_is_doomed_with::<Shared, Local>(4);
    }

    #[test]
    fn panic_chain_past_the_budget_local_shared() {
        panic_chain_past_the_budget::<Local, Shared>();
    }

    #[test]
    fn panic_chain_past_the_budget_shared_shared() {
        panic_chain_past_the_budget::<Shared, Shared>();
    }

    /// At A's budget of 4, the three levels within it took an uncopyable
    /// foreign payload and are held, so they run once; levels past it are
    /// doomed and retried.
    fn panic_chain_past_the_budget<A: crate::cells::Cells, B: crate::cells::Cells>() {
        let (got, runs) = super::foreign_panicking_chain::<A, B>(12, Some(4), None);
        assert_eq!(got, 12);
        assert_eq!(runs[..3], [1, 1, 1]);
        assert!(runs[3..].contains(&2), "{runs:?}");
    }
}

/// A chain of readers of runtime A, each catching the panic of its own
/// node of runtime B, whose payload no kept read can carry, under A's
/// frame budget `frames` and byte budget `bytes`. The outermost reader is
/// at the bottom of the stack, within any budget, so its foreign read's
/// hold keeps it undoomed and it sees its own payload; a retry would read
/// B's node again and see the Failed node's.
fn outermost_foreign_reader_keeps_its_payload<A: Cells, B: Cells>(budget: fn(&Runtime<A>)) {
    quiet_uncopyable();
    let outer = std::thread::spawn(move || {
        let r1 = Box::new(Runtime::<A>::new());
        let r2 = Box::new(Runtime::<B>::new());
        r1.set_placeholder(0_i64);
        budget(&r1);
        let mut below: Option<Incr<i64>> = None;
        for level in 0..200 {
            let bq = r2.create_query(|_| -> i64 { std::panic::panic_any(Uncopyable) });
            let (r2, next) = (OtherRt(&*r2), below);
            let weight = if level == 199 { 1_000_000 } else { 1 };
            below = Some(r1.create_query_with(
                move |rt| {
                    let own = quiet(|| r2.get(bq)).is_err_and(|p| p.is::<Uncopyable>());
                    i64::from(own) * weight + next.map_or(0, |b| rt.get(b))
                },
                QueryOpts {
                    replay: Replay::Allowed,
                    ..Default::default()
                },
            ));
        }
        r1.get(below.unwrap()) / 1_000_000
    })
    .join()
    .unwrap();
    assert_eq!(outer, 1, "the outermost reader saw a retry's payload");
}

/// Runs `read` with about `kib` KiB more of this thread's stack in use.
#[inline(never)]
fn deeper<R>(kib: usize, read: impl FnOnce() -> R) -> R {
    if kib == 0 {
        return read();
    }
    let pad = std::hint::black_box([0_u8; 1024]);
    let out = deeper(kib - 1, read);
    std::hint::black_box(&pad);
    out
}

/// The outermost reader takes its first uncopyable foreign read within
/// the byte budget, then a second one from deep in its own closure's
/// stack, past it. The first read still holds the frame, which is within
/// the budget, so the outermost reader sees its own payload.
fn a_later_deep_hold_keeps_the_first<A: Cells, B: Cells>() {
    quiet_uncopyable();
    let outer = std::thread::spawn(move || {
        let r1 = Box::new(Runtime::<A>::new());
        let r2 = Box::new(Runtime::<B>::new());
        r1.set_placeholder(0_i64);
        r1.set_max_frames(u32::MAX);
        r1.set_max_stack_bytes(64 * 1024);
        let mut below: Option<Incr<i64>> = None;
        for level in 0..200 {
            let bq = r2.create_query(|_| -> i64 { std::panic::panic_any(Uncopyable) });
            let deep = r2.create_query(|_| -> i64 { std::panic::panic_any(Uncopyable) });
            let (r2, next) = (OtherRt(&*r2), below);
            let outermost = level == 199;
            below = Some(r1.create_query_with(
                move |rt| {
                    if !outermost {
                        return next.map_or(0, |b| rt.get(b));
                    }
                    let own = quiet(|| r2.get(bq)).is_err_and(|p| p.is::<Uncopyable>());
                    let _ = deeper(96, || quiet(|| r2.get(deep)));
                    i64::from(own) + next.map_or(0, |b| rt.get(b))
                },
                QueryOpts {
                    replay: Replay::Allowed,
                    ..Default::default()
                },
            ));
        }
        r1.get(below.unwrap())
    })
    .join()
    .unwrap();
    assert_eq!(outer, 1, "the outermost reader saw a retry's payload");
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_shared_a_later_deep_hold_keeps_the_first() {
    a_later_deep_hold_keeps_the_first::<Local, Shared>();
}

mod foreign_hold_within_budget {
    use crate::cells::{Local, Shared};

    #[test]
    #[cfg_attr(miri, ignore)]
    fn frame_budget() {
        super::outermost_foreign_reader_keeps_its_payload::<Local, Shared>(|r| {
            r.set_max_frames(8);
            r.set_max_stack_bytes(0);
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn byte_budget() {
        super::outermost_foreign_reader_keeps_its_payload::<Local, Shared>(|r| {
            r.set_max_frames(u32::MAX);
            r.set_max_stack_bytes(64 * 1024);
        });
    }

    /// The byte budget set by another thread while `set_max_frames` lists
    /// the runtime.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn byte_budget_set_by_a_racing_thread() {
        super::outermost_foreign_reader_keeps_its_payload::<Shared, Shared>(|r| {
            crate::runtime::replay_tests::race_budget_setters(r, 64 * 1024);
        });
    }
}

thread_local! {
    static CLONES: Cell<u32> = const { Cell::new(0) };
}

/// A value that counts its clones on this thread.
#[derive(PartialEq)]
struct Counted(i64);

impl Clone for Counted {
    fn clone(&self) -> Self {
        CLONES.with(|c| c.set(c.get() + 1));
        Counted(self.0)
    }
}

fn clones_of(read: impl FnOnce()) -> u32 {
    CLONES.with(|c| c.set(0));
    read();
    CLONES.with(Cell::get)
}

/// With no budget no read is retried, so a foreign read keeps no copy:
/// a frame's read of another runtime's New node clones its value as often
/// as a read from outside any frame does.
fn unbudgeted_foreign_read_keeps_no_clone<A: Cells, B: Cells>() {
    let direct = {
        let r2 = Runtime::<B>::new();
        let bq = r2.create_query(|_| Counted(7));
        clones_of(|| {
            r2.get(bq);
        })
    };
    let r1 = Box::new(Runtime::<A>::new());
    let r2 = Box::new(Runtime::<B>::new());
    r1.set_max_frames(u32::MAX);
    let bq = r2.create_query(|_| Counted(7));
    let other = OtherRt(&*r2);
    let q = r1.create_query(move |_| other.get(bq).0);
    let framed = clones_of(|| {
        r1.get(q);
    });
    assert_eq!(
        framed, direct,
        "clones of a framed foreign read, unbudgeted"
    );
}

#[test]
fn local_shared_unbudgeted_foreign_read_keeps_no_clone() {
    unbudgeted_foreign_read_keeps_no_clone::<Local, Shared>();
}

#[test]
fn shared_shared_unbudgeted_foreign_read_keeps_no_clone() {
    unbudgeted_foreign_read_keeps_no_clone::<Shared, Shared>();
}

/// The same for a read of another runtime's Clean node.
fn unbudgeted_foreign_clean_read_keeps_no_clone<A: Cells, B: Cells>() {
    let r2 = Box::new(Runtime::<B>::new());
    let bq = r2.create_query(|_| Counted(7));
    r2.get(bq);
    let direct = clones_of(|| {
        r2.get(bq);
    });
    let r1 = Box::new(Runtime::<A>::new());
    r1.set_max_frames(u32::MAX);
    let other = OtherRt(&*r2);
    let q = r1.create_query_with(
        move |_| other.get(bq).0,
        QueryOpts {
            replay: Replay::Allowed,
            ..Default::default()
        },
    );
    let framed = clones_of(|| {
        r1.get(q);
    });
    assert_eq!(
        framed, direct,
        "clones of a framed foreign Clean read, unbudgeted"
    );
}

#[test]
fn local_shared_unbudgeted_foreign_clean_read_keeps_no_clone() {
    unbudgeted_foreign_clean_read_keeps_no_clone::<Local, Shared>();
}

#[test]
fn shared_shared_unbudgeted_foreign_clean_read_keeps_no_clone() {
    unbudgeted_foreign_clean_read_keeps_no_clone::<Shared, Shared>();
}

/// A budgeted replayable frame keeps its read of another runtime's Clean
/// node for a retry, and the kept copy goes when the frame pops, so no
/// copy outlives the read that could take it.
fn budgeted_foreign_clean_read_frees_its_clone<A: Cells, B: Cells>() {
    let probe = Arc::new(());
    let r2 = Box::new(Runtime::<B>::new());
    let p = probe.clone();
    let bq = r2.create_query(move |_| p.clone());
    drop(r2.get(bq));
    let held = Arc::strong_count(&probe);
    let r1 = Box::new(Runtime::<A>::new());
    let other = OtherRt(&*r2);
    let q = r1.create_query_with(
        move |_| Arc::strong_count(&other.get(bq)) as u64,
        QueryOpts {
            replay: Replay::Allowed,
            ..Default::default()
        },
    );
    assert!(r1.get(q) as usize > held + 1, "the read kept no copy");
    assert_eq!(
        Arc::strong_count(&probe),
        held,
        "a kept copy outlived its frame"
    );
}

#[test]
fn local_shared_budgeted_foreign_clean_read_frees_its_clone() {
    budgeted_foreign_clean_read_frees_its_clone::<Local, Shared>();
}

#[test]
fn shared_shared_budgeted_foreign_clean_read_frees_its_clone() {
    budgeted_foreign_clean_read_frees_its_clone::<Shared, Shared>();
}

/// X reads B's uncopyable panic `held_at` KiB deep, then `during` runs,
/// then X reads its own runtime's Y, which reads Z, `deep` KiB deep.
/// Returns X's value: 10 for its own payload, plus Y's 2.
fn hold_then_own_read<A: Cells, B: Cells>(
    before: fn(&Runtime<A>),
    during: fn(&Runtime<A>),
    held_at: usize,
    deep: usize,
) -> i64 {
    quiet_uncopyable();
    std::thread::spawn(move || {
        let r1 = Box::new(Runtime::<A>::new());
        let r2 = Box::new(Runtime::<B>::new());
        r1.set_placeholder(0_i64);
        before(&r1);
        let opts = || QueryOpts {
            replay: Replay::Allowed,
            ..Default::default()
        };
        let bq = r2.create_query(|_| -> i64 { std::panic::panic_any(Uncopyable) });
        let z = r1.create_query_with(|_| 1_i64, opts());
        let y = r1.create_query_with(move |rt| rt.get(z) + 1, opts());
        let r2 = OtherRt(&*r2);
        let x = r1.create_query_with(
            move |rt| {
                let own =
                    deeper(held_at, || quiet(|| r2.get(bq))).is_err_and(|p| p.is::<Uncopyable>());
                during(rt);
                10 * i64::from(own) + deeper(deep, || rt.get(y))
            },
            opts(),
        );
        r1.get(x)
    })
    .join()
    .unwrap()
}

/// [`hold_then_own_read`] with X's first read of its own runtime's A,
/// which panics with a message, in place of B's uncopyable panic.
fn own_panic_then_own_read<A: Cells>(before: fn(&Runtime<A>), during: fn(&Runtime<A>)) -> i64 {
    std::thread::spawn(move || {
        let r1 = Runtime::<A>::new();
        r1.set_placeholder(0_i64);
        before(&r1);
        let opts = || QueryOpts {
            replay: Replay::Allowed,
            ..Default::default()
        };
        let a = r1.create_query(|_| -> i64 { std::panic::panic_any("own") });
        let z = r1.create_query_with(|_| 1_i64, opts());
        let y = r1.create_query_with(move |rt| rt.get(z) + 1, opts());
        let x = r1.create_query_with(
            move |rt| {
                let own =
                    quiet(|| rt.get(a)).is_err_and(|p| p.downcast_ref::<&str>() == Some(&"own"));
                during(rt);
                10 * i64::from(own) + rt.get(y)
            },
            opts(),
        );
        r1.get(x)
    })
    .join()
    .unwrap()
}

/// X reads Y1 and then Y2, siblings that each take an uncopyable foreign
/// read `held_at` KiB deep and then read their own runtime 96 KiB deep,
/// under a 64 KiB byte budget. Returns how many times each ran.
fn sibling_holds<A: Cells, B: Cells>(held_at: [usize; 2]) -> [usize; 2] {
    quiet_uncopyable();
    std::thread::spawn(move || {
        let runs: &'static [AtomicUsize; 2] = Box::leak(Box::default());
        let r1 = Box::new(Runtime::<A>::new());
        let r2 = Box::new(Runtime::<B>::new());
        r1.set_placeholder(0_i64);
        r1.set_max_frames(u32::MAX);
        r1.set_max_stack_bytes(64 * 1024);
        let opts = || QueryOpts {
            replay: Replay::Allowed,
            ..Default::default()
        };
        let ys = [0, 1].map(|i| {
            let z = r1.create_query_with(|_| 1_i64, opts());
            let bq = r2.create_query(|_| -> i64 { std::panic::panic_any(Uncopyable) });
            let r2 = OtherRt(&*r2);
            r1.create_query_with(
                move |rt| {
                    runs[i].fetch_add(1, Ordering::SeqCst);
                    let _ = deeper(held_at[i], || quiet(|| r2.get(bq)));
                    deeper(96, || rt.get(z))
                },
                opts(),
            )
        });
        let x = r1.create_query_with(move |rt| rt.get(ys[0]) + rt.get(ys[1]), opts());
        r1.get(x);
        runs.each_ref().map(|n| n.load(Ordering::SeqCst))
    })
    .join()
    .unwrap()
}

/// A hold Y1 took and gave up with its frame does not hold Y2, a later
/// frame at the same place: Y2's own hold, past the budget, leaves it
/// doomed and run again when it reads past the budget.
#[test]
#[cfg_attr(miri, ignore)]
fn a_popped_siblings_hold_does_not_hold_the_next_frame() {
    let (both, second) = (
        sibling_holds::<Local, Shared>([96, 96]),
        sibling_holds::<Local, Shared>([0, 96]),
    );
    assert!(both[1] > 1 && second[1] > 1, "{both:?} {second:?}");
}

mod review_8b_held_deep_read {
    use super::{hold_then_own_read, Runtime};
    use crate::cells::{Cells, Local, Shared};

    fn none<C: Cells>(_: &Runtime<C>) {}

    fn unbudgeted<C: Cells>(r: &Runtime<C>) {
        r.set_max_frames(u32::MAX);
    }

    fn bytes_64k<C: Cells>(r: &Runtime<C>) {
        r.set_max_frames(u32::MAX);
        r.set_max_stack_bytes(64 * 1024);
    }

    /// X's hold, taken within the byte budget, keeps X undoomed as
    /// `NO_DOOM` would when X's own later read is past the budget, so X
    /// sees its own payload as the unbudgeted run does.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_held_frame_reading_deep_keeps_its_payload() {
        assert_eq!(
            hold_then_own_read::<Local, Shared>(unbudgeted, none, 0, 96),
            12
        );
        assert_eq!(
            hold_then_own_read::<Local, Shared>(bytes_64k, none, 0, 96),
            12
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shared_a_held_frame_reading_deep_keeps_its_payload() {
        assert_eq!(
            hold_then_own_read::<Shared, Shared>(unbudgeted, none, 0, 96),
            12
        );
        assert_eq!(
            hold_then_own_read::<Shared, Shared>(bytes_64k, none, 0, 96),
            12
        );
    }
}

mod review_8b_mid_compute_budget {
    use super::{hold_then_own_read, own_panic_then_own_read, Runtime};
    use crate::cells::{Cells, Local, Shared};

    fn none<C: Cells>(_: &Runtime<C>) {}

    fn unbudgeted<C: Cells>(r: &Runtime<C>) {
        r.set_max_frames(u32::MAX);
    }

    fn frames_2<C: Cells>(r: &Runtime<C>) {
        r.set_max_frames(2);
    }

    /// A frame budget set while X runs applies to reads made after it, so
    /// X's foreign read made before it is not retried: X sees its own
    /// payload, as with no budget and with the budget set beforehand.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_budget_set_mid_compute_keeps_the_earlier_payload() {
        assert_eq!(
            hold_then_own_read::<Local, Shared>(unbudgeted, none, 0, 0),
            12
        );
        assert_eq!(
            hold_then_own_read::<Local, Shared>(frames_2, none, 0, 0),
            12
        );
        assert_eq!(
            hold_then_own_read::<Local, Shared>(unbudgeted, frames_2, 0, 0),
            12
        );
    }

    fn bytes_64k<C: Cells>(r: &Runtime<C>) {
        r.set_max_frames(u32::MAX);
        r.set_max_stack_bytes(64 * 1024);
    }

    fn bytes_1m<C: Cells>(r: &Runtime<C>) {
        r.set_max_frames(u32::MAX);
        r.set_max_stack_bytes(1024 * 1024);
    }

    /// X's hold, taken 96 KiB deep where no budget or a 1 MiB one held
    /// it, still holds X once a 64 KiB budget is set mid-compute.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_budget_tightened_mid_compute_keeps_an_earlier_hold() {
        assert_eq!(
            hold_then_own_read::<Local, Shared>(unbudgeted, none, 96, 96),
            12
        );
        assert_eq!(
            hold_then_own_read::<Local, Shared>(unbudgeted, bytes_64k, 96, 96),
            12
        );
        assert_eq!(
            hold_then_own_read::<Local, Shared>(bytes_1m, bytes_64k, 96, 96),
            12
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shared_a_budget_tightened_mid_compute_keeps_an_earlier_hold() {
        assert_eq!(
            hold_then_own_read::<Shared, Shared>(bytes_1m, bytes_64k, 96, 96),
            12
        );
        assert_eq!(
            hold_then_own_read::<Shared, Shared>(unbudgeted, bytes_64k, 96, 96),
            12
        );
    }

    /// A frame budget set while X runs keeps X's earlier read of its own
    /// runtime's panic, which nothing kept while X's runtime had no budget.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_budget_set_mid_compute_keeps_an_earlier_own_panic() {
        assert_eq!(own_panic_then_own_read::<Local>(unbudgeted, none), 12);
        assert_eq!(own_panic_then_own_read::<Local>(frames_2, none), 12);
        assert_eq!(own_panic_then_own_read::<Local>(unbudgeted, frames_2), 12);
        assert_eq!(own_panic_then_own_read::<Shared>(unbudgeted, frames_2), 12);
    }
}
