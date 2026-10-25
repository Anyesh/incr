//! Owned children: `child_input` and `child_query` reuse, the sweep and
//! retention at a Clean landing, rollback at every discarding landing,
//! grandchildren, fixpoint groups, and deletes of owned trees through
//! `claim_set`. Each generic case runs under `Local` and `Shared`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::SeqCst};
use std::sync::{Arc, Barrier, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use super::race_tests::{id, park_at, Gate};
use super::{DeleteError, ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::child_test::run_in_child;
use crate::cycle::{CycleStrategy, Fixpoint, QueryOpts};
use crate::handle::{Incr, RuntimeId};
use crate::reclaim_probe::{self, RetireSite};
use crate::state::NodeState;

macro_rules! local_and_shared {
    ($body:ident, $local:ident, $shared:ident) => {
        #[test]
        fn $local() {
            $body(&Runtime::<Local>::new());
        }

        #[test]
        fn $shared() {
            $body(&Runtime::<Shared>::new());
        }
    };
}

fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
    match err.downcast::<String>() {
        Ok(s) => *s,
        Err(err) => err.downcast::<&str>().map(|s| s.to_string()).unwrap(),
    }
}

type Made = Arc<Mutex<Vec<Incr<u64>>>>;

/// Counts its drops, so a test sees a payload dropped on a hit.
struct Dropped(Arc<AtomicU32>);

impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.fetch_add(1, SeqCst);
    }
}

/// A creator over `x` with one child input per key in `0..x`, valued
/// `10 * key`, returning their handles. `inits` counts the inits run.
fn keyed_creator<C: Cells>(
    rt: &Runtime<C>,
    x: Incr<u64>,
    inits: Arc<AtomicU32>,
) -> Incr<Vec<Incr<u64>>> {
    rt.create_query(move |rt| {
        (0..rt.get(x))
            .map(|k| {
                let inits = inits.clone();
                rt.child_input(k, move || {
                    inits.fetch_add(1, SeqCst);
                    k * 10
                })
            })
            .collect()
    })
}

fn reuse<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(2_u64);
    let inits = Arc::new(AtomicU32::new(0));
    let creator = keyed_creator(rt, x, inits.clone());
    let first = rt.get(creator);
    assert_eq!(inits.load(SeqCst), 2);
    rt.set(x, 3);
    let second = rt.get(creator);
    assert_eq!(second[..2], first[..]);
    assert_eq!(inits.load(SeqCst), 3, "only the new key ran its init");
    let values: Vec<u64> = second.iter().map(|&c| rt.get(c)).collect();
    assert_eq!(values, vec![0, 10, 20]);
    rt.assert_invariant_i();
}
local_and_shared!(reuse, local_children_reuse, shared_children_reuse);

/// Identity includes the value type and the kind: the same key names a
/// different child for each.
fn identity_includes_type_and_kind<C: Cells>(rt: &Runtime<C>) {
    let q = rt.create_query(|rt| {
        let a = rt.child_input(1_u32, || 1_u64).erase();
        let b = rt.child_input(1_u32, || 1_i64).erase();
        let c = rt.child_query(1_u32, |_| 1_u64).erase();
        let d = rt.child_input(1_u64, || 1_u64).erase();
        vec![a, b, c, d]
    });
    let mut kids = rt.get(q);
    kids.sort_by_key(|h| h.slot());
    kids.dedup();
    assert_eq!(kids.len(), 4);
}
local_and_shared!(
    identity_includes_type_and_kind,
    local_children_identity_includes_type_and_kind,
    shared_children_identity_includes_type_and_kind
);

/// On a hit the init and the closure are dropped unrun, and the child
/// keeps the closure it was created with.
fn hit_drops_payload<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(0_u64);
    let dropped = Arc::new(AtomicU32::new(0));
    let ran = Arc::new(AtomicU32::new(0));
    let creator = {
        let (dropped, ran) = (dropped.clone(), ran.clone());
        rt.create_query(move |rt| {
            let _ = rt.get(x);
            let guard = Dropped(dropped.clone());
            let ran = ran.clone();
            rt.child_query("q", move |_| {
                let _ = &guard;
                ran.fetch_add(1, SeqCst);
                5_u64
            })
        })
    };
    let q = rt.get(creator);
    assert_eq!(rt.get(q), 5);
    assert_eq!(dropped.load(SeqCst), 0);
    rt.set(x, 1);
    assert_eq!(rt.get(creator), q);
    assert_eq!(dropped.load(SeqCst), 1, "the hit dropped the new closure");
    assert_eq!(rt.get(q), 5);
    assert_eq!(ran.load(SeqCst), 1);
}
local_and_shared!(
    hit_drops_payload,
    local_children_hit_drops_payload,
    shared_children_hit_drops_payload
);

/// A creator whose rerun returns the same child cuts off: its static
/// deps publish through the fast path and its dependent stays Clean.
fn early_cutoff_kept<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(0_u64);
    let creator = rt.create_query(move |rt| {
        let _ = rt.get(x);
        rt.child_input("k", || 7_u64)
    });
    let runs = Arc::new(AtomicU32::new(0));
    let dependent = {
        let runs = runs.clone();
        rt.create_query(move |rt| {
            runs.fetch_add(1, SeqCst);
            rt.get(rt.get(creator))
        })
    };
    assert_eq!(rt.get(dependent), 7);
    rt.set(x, 1);
    assert_eq!(rt.get(dependent), 7);
    assert_eq!(
        runs.load(SeqCst),
        1,
        "the creator's unchanged value cut off"
    );
    assert_eq!(rt.state_of(creator), NodeState::Clean);
    rt.assert_invariant_i();
}
local_and_shared!(
    early_cutoff_kept,
    local_children_early_cutoff_kept,
    shared_children_early_cutoff_kept
);

fn sweep<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(3_u64);
    let creator = keyed_creator(rt, x, Arc::default());
    let kids = rt.get(creator);
    let before = rt.node_count();
    rt.set(x, 1);
    assert_eq!(rt.get(creator), kids[..1]);
    assert!(rt.is_alive(kids[0]));
    assert!(!rt.is_alive(kids[1]) && !rt.is_alive(kids[2]));
    assert_eq!(rt.node_count(), before - 2);
    rt.set(x, 3);
    let again = rt.get(creator);
    assert_eq!(again[0], kids[0]);
    assert_ne!(again[1], kids[1], "a swept key gets a new child");
    rt.assert_invariant_i();
}
local_and_shared!(sweep, local_children_sweep, shared_children_sweep);

/// A verification landing ran no closure, so it sweeps nothing.
fn verified_landing_sweeps_nothing<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(2_u64);
    let y = rt.create_input(0_u64);
    let mid = rt.create_query(move |rt| rt.get(y) / 10);
    let creator = rt.create_query(move |rt| {
        let _ = rt.get(mid);
        (0..rt.get(x))
            .map(|k| rt.child_input(k, || k))
            .collect::<Vec<_>>()
    });
    let kids = rt.get(creator);
    rt.set(y, 1);
    assert_eq!(rt.get(creator), kids);
    assert!(kids.iter().all(|&k| rt.is_alive(k)));
}
local_and_shared!(
    verified_landing_sweeps_nothing,
    local_children_verified_landing_sweeps_nothing,
    shared_children_verified_landing_sweeps_nothing
);

/// A child with a dependent outside its owned set survives the sweep,
/// stays bound to its key and owned, and goes at the first Clean
/// landing after the dependent does.
fn retention<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(2_u64);
    let creator = keyed_creator(rt, x, Arc::default());
    let kids = rt.get(creator);
    let held = kids[1];
    let reader = rt.create_query(move |rt| rt.get(held) + 1);
    assert_eq!(rt.get(reader), 11);
    rt.set(x, 1);
    assert_eq!(rt.get(creator).len(), 1);
    assert!(
        rt.is_alive(held),
        "a child with an outside dependent is retained"
    );
    assert_eq!(rt.get(reader), 11);
    rt.set(x, 2);
    assert_eq!(rt.get(creator)[1], held, "a retained child keeps its key");
    rt.set(x, 1);
    let _ = rt.get(creator);
    rt.delete_node(reader);
    assert!(rt.is_alive(held));
    rt.set(x, 0);
    assert!(rt.get(creator).is_empty());
    assert!(!rt.is_alive(held) && !rt.is_alive(kids[0]));
    rt.assert_invariant_i();
}
local_and_shared!(
    retention,
    local_children_retention,
    shared_children_retention
);

fn reads_own_child<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(true);
    let made: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let creator = {
        let made = made.clone();
        rt.create_query(move |rt| {
            if !rt.get(x) {
                return 0;
            }
            let c = rt.child_input("c", || 4_u64);
            let _ = made.set(c);
            rt.get(c) + 1
        })
    };
    assert_eq!(rt.get(creator), 5);
    let c = *made.get().unwrap();
    rt.set(c, 9);
    assert_eq!(rt.get(creator), 10);
    rt.set(x, false);
    assert_eq!(rt.get(creator), 0);
    assert!(
        !rt.is_alive(c),
        "the run that stopped reading its child swept it"
    );
    rt.assert_invariant_i();
}
local_and_shared!(
    reads_own_child,
    local_children_reads_own_child,
    shared_children_reads_own_child
);

/// Units {a, a1} and {b, b1}: a reads b1 and b reads a1, so each unit
/// has a dependent in the other. The cascade merges them and deletes
/// all four.
fn cascade_unit_cycle<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(0_u64);
    let a_kid: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let b_kid: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let a = {
        let (a_kid, b_kid) = (a_kid.clone(), b_kid.clone());
        rt.create_query(move |rt| {
            let _ = rt.get(x);
            let k = rt.child_input("a", || 1_u64);
            let _ = a_kid.set(k);
            b_kid.get().map_or(0, |&b| rt.get(b)) + 1
        })
    };
    let b = {
        let (a_kid, b_kid) = (a_kid.clone(), b_kid.clone());
        rt.create_query(move |rt| {
            let _ = rt.get(x);
            let k = rt.child_input("b", || 2_u64);
            let _ = b_kid.set(k);
            a_kid.get().map_or(0, |&a| rt.get(a)) + 2
        })
    };
    rt.get(a);
    rt.get(b);
    rt.set(x, 1);
    assert_eq!(rt.get(a), 3);
    assert_eq!(rt.get(b), 3);
    assert_eq!(rt.delete_with_dependents(a), 4);
    let (a1, b1) = (*a_kid.get().unwrap(), *b_kid.get().unwrap());
    assert!(!rt.is_alive(a) && !rt.is_alive(b) && !rt.is_alive(a1) && !rt.is_alive(b1));
    assert_eq!(rt.node_count(), 1);
    rt.assert_invariant_i();
}
local_and_shared!(
    cascade_unit_cycle,
    local_children_cascade_unit_cycle,
    shared_children_cascade_unit_cycle
);

/// Two queries in a dependency cycle with no ownership: the cascade
/// from either merges both into one segment.
fn cascade_plain_cycle<C: Cells>(rt: &Runtime<C>) {
    let p_cell: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let q_cell: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let fallback = |opts_value: u64| QueryOpts {
        cycle: CycleStrategy::fallback(move |_| opts_value),
        ..Default::default()
    };
    let p = {
        let q_cell = q_cell.clone();
        rt.create_query_with(move |rt| rt.get(*q_cell.get().unwrap()) + 1, fallback(10))
    };
    let q = {
        let p_cell = p_cell.clone();
        rt.create_query_with(move |rt| rt.get(*p_cell.get().unwrap()) + 1, fallback(20))
    };
    p_cell.set(p).unwrap();
    q_cell.set(q).unwrap();
    rt.get(p);
    assert_eq!(rt.delete_with_dependents(p), 2);
    assert_eq!(rt.node_count(), 0);
    rt.assert_invariant_i();
}
local_and_shared!(
    cascade_plain_cycle,
    local_children_cascade_plain_cycle,
    shared_children_cascade_plain_cycle
);

/// A creator that panics after creating a child lands Failed, and the
/// child is deleted; the recovered run creates a fresh one.
fn rollback_failed<C: Cells>(rt: &Runtime<C>) {
    let fail = rt.create_input(true);
    let made: Made = Arc::default();
    let creator = {
        let made = made.clone();
        rt.create_query(move |rt| {
            let c = rt.child_input("k", || 1_u64);
            made.lock().unwrap().push(c);
            if rt.get(fail) {
                panic!("the creator fails");
            }
            c
        })
    };
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(creator))).is_err());
    let first = made.lock().unwrap()[0];
    assert!(!rt.is_alive(first), "a failed attempt's child is deleted");
    assert_eq!(rt.state_of(creator), NodeState::Failed);
    rt.set(fail, false);
    let c = rt.get(creator);
    assert_ne!(c, first);
    assert_eq!(rt.get(c), 1);
    rt.assert_invariant_i();
}
local_and_shared!(
    rollback_failed,
    local_children_rollback_failed,
    shared_children_rollback_failed
);

/// The failed attempt read its child, so the union publish leaves the
/// child a dependent and the rollback cannot delete it. It is unbound
/// from its key, so the recovered run creates a fresh child, and the
/// orphan goes at that run's Clean landing.
fn rollback_unbinds_a_retained_child<C: Cells>(rt: &Runtime<C>) {
    let fail = rt.create_input(true);
    let made: Made = Arc::default();
    let creator = {
        let made = made.clone();
        rt.create_query(move |rt| {
            let c = rt.child_input("k", || 1_u64);
            made.lock().unwrap().push(c);
            let v = rt.get(c);
            if rt.get(fail) {
                panic!("the creator fails");
            }
            v
        })
    };
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(creator))).is_err());
    let first = made.lock().unwrap()[0];
    assert!(rt.is_alive(first), "the creator's own edge retains it");
    rt.set(fail, false);
    assert_eq!(rt.get(creator), 1);
    let made = made.lock().unwrap();
    assert_ne!(made[1], first, "an unbound child is never reused");
    assert!(!rt.is_alive(first), "the orphan went at the Clean landing");
    rt.assert_invariant_i();
}
local_and_shared!(
    rollback_unbinds_a_retained_child,
    local_children_rollback_unbinds_a_retained_child,
    shared_children_rollback_unbinds_a_retained_child
);

/// The first attempt reads a dep that is deleted and recycled under it,
/// creates a child and lands New; the child is deleted and the retry's
/// child is the one that lives.
#[test]
fn local_children_rollback_stale() {
    let rt = Runtime::<Local>::new();
    let v = rt.create_input(7_u64);
    let first = AtomicBool::new(true);
    let made: Made = Arc::default();
    let q = {
        let made = made.clone();
        rt.create_query(move |rt| {
            if first.swap(false, SeqCst) {
                let a = rt.get(v);
                rt.delete_node(v);
                let _ = rt.create_input_unchecked(99_u64);
                made.lock().unwrap().push(rt.child_input("k", || 1_u64));
                return a;
            }
            made.lock().unwrap().push(rt.child_input("k", || 2_u64));
            42
        })
    };
    assert_eq!(rt.get(q), 42);
    let made = made.lock().unwrap();
    assert_eq!(made.len(), 2);
    assert!(
        !rt.is_alive(made[0]),
        "the stale attempt's child is deleted"
    );
    assert_eq!(rt.get(made[1]), 2);
}

/// A creator unwound by a cycle read takes the discard landing, and the
/// child its attempt created is deleted.
fn rollback_cycle_discard<C: Cells>(rt: &Runtime<C>) {
    let d_cell: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let made: Made = Arc::default();
    let c = {
        let (d_cell, made) = (d_cell.clone(), made.clone());
        rt.create_query(move |rt| {
            made.lock().unwrap().push(rt.child_input("k", || 1_u64));
            rt.get(*d_cell.get().unwrap())
        })
    };
    let d = rt.create_query(move |rt| rt.get(c) + 1);
    d_cell.set(d).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(c))).is_err());
    assert_ne!(rt.state_of(c), NodeState::Clean);
    for &k in made.lock().unwrap().iter() {
        assert!(!rt.is_alive(k), "a discarded attempt's child is deleted");
    }
    rt.assert_invariant_i();
}
local_and_shared!(
    rollback_cycle_discard,
    local_children_rollback_cycle_discard,
    shared_children_rollback_cycle_discard
);

/// A fixpoint member whose third run panics aborts the episode; every
/// child its claim created, in any run, is rolled back.
fn rollback_group_abort<C: Cells>(rt: &Runtime<C>) {
    let me: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let made: Made = Arc::default();
    let runs = Arc::new(AtomicU32::new(0));
    let q = {
        let (me, made, runs) = (me.clone(), made.clone(), runs.clone());
        rt.create_query_with(
            move |rt| {
                let v = rt.get(*me.get().unwrap());
                made.lock().unwrap().push(rt.child_input(v, move || v));
                if runs.fetch_add(1, SeqCst) == 2 {
                    panic!("the member fails");
                }
                (v + 1).min(5)
            },
            QueryOpts {
                cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_u64)),
                ..Default::default()
            },
        )
    };
    me.set(q).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(q))).is_err());
    assert!(made.lock().unwrap().len() >= 3);
    for &k in made.lock().unwrap().iter() {
        assert!(!rt.is_alive(k), "an aborted episode's children are deleted");
    }
    rt.assert_invariant_i();
}
local_and_shared!(
    rollback_group_abort,
    local_children_rollback_group_abort,
    shared_children_rollback_group_abort
);

/// Only the children the final pass touched survive the group's Clean
/// landing.
fn fixpoint_keeps_the_final_pass<C: Cells>(rt: &Runtime<C>) {
    let me: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let made: Made = Arc::default();
    let q = {
        let (me, made) = (me.clone(), made.clone());
        rt.create_query_with(
            move |rt| {
                let v = rt.get(*me.get().unwrap());
                made.lock().unwrap().push(rt.child_input(v, move || v));
                (v + 1).min(3)
            },
            QueryOpts {
                cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_u64)),
                ..Default::default()
            },
        )
    };
    me.set(q).unwrap();
    assert_eq!(rt.get(q), 3);
    let mut live: Vec<Incr<u64>> = made
        .lock()
        .unwrap()
        .iter()
        .copied()
        .filter(|&k| rt.is_alive(k))
        .collect();
    live.dedup();
    assert_eq!(live.len(), 1, "only the final pass's child is kept");
    assert_eq!(rt.get(live[0]), 3);
    rt.assert_invariant_i();
}
local_and_shared!(
    fixpoint_keeps_the_final_pass,
    local_children_fixpoint_keeps_the_final_pass,
    shared_children_fixpoint_keeps_the_final_pass
);

fn grandchildren<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(true);
    let creator = rt.create_query(move |rt| {
        if !rt.get(x) {
            return None;
        }
        Some(rt.child_query("k", |rt| {
            let g = rt.child_input("g", || 3_u64);
            rt.get(g) * 2
        }))
    });
    let k = rt.get(creator).unwrap();
    assert_eq!(rt.get(k), 6);
    assert_eq!(rt.node_count(), 4);
    rt.set(x, false);
    assert_eq!(rt.get(creator), None);
    assert!(!rt.is_alive(k));
    assert_eq!(
        rt.node_count(),
        2,
        "the sweep took the grandchild with its owner"
    );

    rt.set(x, true);
    let k = rt.get(creator).unwrap();
    assert_eq!(rt.get(k), 6);
    rt.delete_node(creator);
    assert_eq!(rt.node_count(), 1);
    rt.assert_invariant_i();
}
local_and_shared!(
    grandchildren,
    local_children_grandchildren,
    shared_children_grandchildren
);

/// A child query that reads its creator: deleting the creator takes the
/// child, whose edge stays inside the owned set.
fn child_reads_creator<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(1_u64);
    let me: Arc<OnceLock<Incr<Incr<u64>>>> = Arc::default();
    let creator = {
        let me = me.clone();
        rt.create_query(move |rt| {
            let _ = rt.get(x);
            let me = me.clone();
            rt.child_query("k", move |rt| {
                let _ = rt.get(*me.get().unwrap());
                5_u64
            })
        })
    };
    me.set(creator).unwrap();
    let k = rt.get(creator);
    assert_eq!(rt.get(k), 5);
    assert_eq!(rt.try_delete_node(creator), Ok(()));
    assert!(!rt.is_alive(k));
    rt.assert_invariant_i();
}
local_and_shared!(
    child_reads_creator,
    local_children_child_reads_creator,
    shared_children_child_reads_creator
);

fn outside_compute_panics<C: Cells>(rt: &Runtime<C>) {
    let err = catch_unwind(AssertUnwindSafe(|| rt.child_input(1_u32, || 0_u64))).unwrap_err();
    assert!(panic_message(err).contains("child_input called outside a compute"));
    let err = catch_unwind(AssertUnwindSafe(|| rt.child_query(1_u32, |_| 0_u64))).unwrap_err();
    assert!(panic_message(err).contains("child_query called outside a compute"));
}
local_and_shared!(
    outside_compute_panics,
    local_children_outside_compute_panics,
    shared_children_outside_compute_panics
);

/// In a doomed frame a hit returns the child and a miss the pending
/// handle, which creates nothing.
fn doomed_frame<C: Cells>(rt: &Runtime<C>) {
    rt.set_placeholder(0_u64);
    let gate = rt.doom_gate();
    let x = rt.create_input(0_u64);
    let seen = Arc::new(Mutex::new(None));
    let out = seen.clone();
    let creator = rt.create_query_with(
        move |rt| {
            let n = rt.get(x);
            let a = rt.child_input("a", || 1_u64);
            if n == 0 {
                return vec![a];
            }
            if rt.get(gate).0 {
                return vec![a];
            }
            let doomed = vec![rt.child_input("a", || 2_u64), rt.child_input("b", || 3_u64)];
            *out.lock().unwrap() = Some((doomed.clone(), rt.node_count()));
            doomed
        },
        QueryOpts {
            replay: crate::cycle::Replay::Allowed,
            ..Default::default()
        },
    );
    let first = rt.get(creator)[0];
    let before = rt.node_count();
    rt.set(x, 1);
    assert_eq!(rt.get(creator), vec![first]);
    if !C::SINGLE_THREADED {
        // A Shared doomed frame unwinds at its placeholder read, so the
        // code past the gate never runs.
        assert!(seen.lock().unwrap().is_none());
        return;
    }
    let (doomed, nodes) = seen.lock().unwrap().take().expect("the run was doomed");
    assert_eq!(doomed[0], first);
    assert_eq!(doomed[1].runtime_id(), RuntimeId::PENDING);
    assert_eq!(nodes, before, "a doomed miss creates nothing");
}
local_and_shared!(
    doomed_frame,
    local_children_doomed_frame,
    shared_children_doomed_frame
);

fn adopt_accepts_a_query<C: Cells>(rt: &Runtime<C>) {
    let owner = rt.create_input(0_u64);
    let q = rt.create_query(|_| 1_u64);
    rt.adopt(owner.erase(), q.erase());
    assert_eq!(rt.get(q), 1);
    rt.delete_node(owner);
    assert!(!rt.is_alive(q));
    rt.assert_invariant_i();
}
local_and_shared!(
    adopt_accepts_a_query,
    local_children_adopt_accepts_a_query,
    shared_children_adopt_accepts_a_query
);

fn adopt_refuses_an_ownership_cycle<C: Cells>(rt: &Runtime<C>) {
    let a = rt.create_input(0_u64);
    let b = rt.create_input(0_u64);
    rt.adopt(a.erase(), b.erase());
    let err = catch_unwind(AssertUnwindSafe(|| rt.adopt(b.erase(), a.erase()))).unwrap_err();
    assert!(panic_message(err).contains("would make it own itself"));
}
local_and_shared!(
    adopt_refuses_an_ownership_cycle,
    local_children_adopt_refuses_an_ownership_cycle,
    shared_children_adopt_refuses_an_ownership_cycle
);

fn strict_delete_of_an_owner_in_compute_panics<C: Cells>(rt: &Runtime<C>) {
    let owner = rt.create_input(0_u64);
    let child = rt.create_input(1_u64);
    rt.adopt(owner.erase(), child.erase());
    let q = rt.create_query(move |rt| {
        catch_unwind(AssertUnwindSafe(|| rt.delete_node(owner)))
            .err()
            .map(panic_message)
    });
    let message = rt.get(q).expect("the delete panicked");
    assert!(message.contains("owns nodes"), "{message}");
    assert!(rt.is_alive(owner) && rt.is_alive(child));
}
local_and_shared!(
    strict_delete_of_an_owner_in_compute_panics,
    local_children_strict_delete_of_an_owner_in_compute_panics,
    shared_children_strict_delete_of_an_owner_in_compute_panics
);

/// A deleted child is not reused: the next run creates a fresh one.
fn deleted_child_is_recreated<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(0_u64);
    let creator = rt.create_query(move |rt| {
        let _ = rt.get(x);
        rt.child_input("k", || 1_u64)
    });
    let first = rt.get(creator);
    rt.delete_node(first);
    rt.set(x, 1);
    let second = rt.get(creator);
    assert_ne!(second, first);
    assert_eq!(rt.get(second), 1);
    rt.assert_invariant_i();
}
local_and_shared!(
    deleted_child_is_recreated,
    local_children_deleted_child_is_recreated,
    shared_children_deleted_child_is_recreated
);

/// A value whose `Drop` panics while `armed`.
#[derive(Clone)]
struct Bomb(Arc<AtomicBool>);

impl PartialEq for Bomb {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Drop for Bomb {
    fn drop(&mut self) {
        if self.0.swap(false, SeqCst) {
            panic!("a swept child's drop panicked on purpose");
        }
    }
}

/// The sweep defers the swept child's closure drop past the creator's
/// landing, so its panic leaves the creator Clean and the next read
/// returns at once.
fn swept_child_drop_panic<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(2_u64);
    let armed: Arc<Vec<Arc<AtomicBool>>> = Arc::new((0..2).map(|_| Arc::default()).collect());
    let creator = {
        let armed = armed.clone();
        rt.create_query(move |rt| {
            let n = rt.get(x);
            for k in 0..n {
                let bomb = Bomb(armed[k as usize].clone());
                rt.child_query(k, move |_| {
                    let _ = &bomb;
                    k
                });
            }
            n
        })
    };
    assert_eq!(rt.get(creator), 2);
    armed[1].store(true, SeqCst);
    rt.set(x, 1);
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(creator))).unwrap_err();
    assert!(panic_message(err).contains("on purpose"));
    assert_eq!(rt.state_of(creator), NodeState::Clean);
    assert_eq!(rt.get(creator), 1);
    rt.assert_invariant_i();
}
local_and_shared!(
    swept_child_drop_panic,
    local_children_swept_child_drop_panic,
    shared_children_swept_child_drop_panic
);

/// A retired value whose `Drop` panics is reclaimed at the sweep's
/// deferred retire, or at a rollback's, after the creator landed.
fn forced_reclaim_at_child_delete(rollback: bool) {
    let rt = Runtime::<Shared>::new();
    super::drop_panic_tests::prime_reclaim_timer(&rt);
    let x = rt.create_input(2_u64);
    let creator = rt.create_query(move |rt| {
        let n = rt.get(x);
        for k in 0..n.max(1) {
            rt.child_input(k, move || k);
        }
        if n == 9 {
            panic!("the creator fails");
        }
        n
    });
    assert_eq!(rt.get(creator), 2);
    super::drop_panic_tests::retire_armed_boom(&rt);
    reclaim_probe::force(RetireSite::DeleteRetire);
    rt.set(x, if rollback { 9 } else { 1 });
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(creator))).unwrap_err();
    if !rollback {
        assert!(panic_message(err).contains("on purpose"));
        assert_eq!(rt.state_of(creator), NodeState::Clean);
        assert_eq!(rt.get(creator), 1);
    } else {
        assert_eq!(rt.state_of(creator), NodeState::Failed);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_children_forced_reclaim_at_sweep() {
    run_in_child(
        "runtime::children_tests::shared_children_forced_reclaim_at_sweep",
        || forced_reclaim_at_child_delete(false),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_children_forced_reclaim_at_rollback() {
    run_in_child(
        "runtime::children_tests::shared_children_forced_reclaim_at_rollback",
        || forced_reclaim_at_child_delete(true),
    )
    .assert_clean();
}

/// Two threads strictly delete overlapping nested owned trees at once:
/// `a` owns `b`, which owns `c` and `d`. Under the old tree claim each
/// could wait on the other's members while holding `Dead` claims.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_children_overlapping_owned_tree_deletes_complete() {
    run_in_child(
        "runtime::children_tests::shared_children_overlapping_owned_tree_deletes_complete",
        || {
            for _ in 0..300 {
                let rt = Arc::new(Runtime::<Shared>::new());
                let [a, b, c, d] = [0, 1, 2, 3].map(|v| rt.create_input(v as u64));
                rt.adopt(a.erase(), b.erase());
                rt.adopt(b.erase(), c.erase());
                rt.adopt(b.erase(), d.erase());
                let start = Arc::new(Barrier::new(2));
                let racers: Vec<_> = [a, b]
                    .into_iter()
                    .map(|root| {
                        let (rt, start) = (rt.clone(), start.clone());
                        thread::spawn(move || {
                            start.wait();
                            rt.try_delete_node(root)
                        })
                    })
                    .collect();
                for racer in racers {
                    let result = racer.join().unwrap();
                    assert!(matches!(result, Ok(()) | Err(DeleteError::NotAlive)));
                }
                assert_eq!(rt.node_count(), 0);
                rt.assert_invariant_i();
            }
        },
    )
    .assert_clean();
}

/// Value code reads an input that a parked `claim_set` holds
/// `Deleting`. It spins unregistered, and once the claim rolls back it
/// reads the input as Clean.
#[test]
fn shared_children_value_code_read_spins_through_a_rollback() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let owner = rt.create_input(0_u64);
    let held = rt.create_input(7_u64);
    let busy = rt.create_input(0_u64);
    rt.adopt(owner.erase(), held.erase());
    rt.adopt(owner.erase(), busy.erase());
    let set_claimed = Gate::new();
    let before_rollback = Gate::new();
    park_at(
        &rt,
        vec![
            (ParkPoint::SetAfterClaim(id(busy)), set_claimed.clone()),
            (
                ParkPoint::ClaimSetBeforeRollback(id(owner)),
                before_rollback.clone(),
            ),
        ],
    );
    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(busy, 1))
    };
    set_claimed.wait_parked();
    let deleter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.try_delete_node(owner))
    };
    before_rollback.wait_parked();
    assert_eq!(rt.state_of(held), NodeState::Deleting);
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || crate::with_value_code(|| rt.get(held)))
    };
    thread::sleep(Duration::from_millis(50));
    assert!(
        !reader.is_finished(),
        "the read spins while the input is Deleting"
    );
    before_rollback.release();
    assert_eq!(reader.join().unwrap(), 7);
    set_claimed.release();
    setter.join().unwrap();
    assert_eq!(deleter.join().unwrap(), Ok(()));
    assert!(!rt.is_alive(held) && !rt.is_alive(busy));
    rt.assert_invariant_i();
}
