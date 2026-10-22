//! The `ensure_clean` walk: its cost on wide DAGs and its recovery after a
//! cycle panic, each case run under `Local` and `Shared`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::Duration;

use super::race_tests::{id, park_at, Gate};
use super::{ParkPoint, Runtime, EXACT_MODE_AT};
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{CycleStrategy, Fixpoint, QueryOpts, Replay};
use crate::handle::Incr;
use crate::test_size::test_size;
use crate::value_code;

/// Nested first computes overflow the default test thread stack somewhere
/// between 600 and 700 levels in debug builds.
const DEEP_STACK: usize = 256 << 20;

fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
    match err.downcast::<String>() {
        Ok(s) => *s,
        Err(err) => err.downcast::<&str>().map(|s| s.to_string()).unwrap(),
    }
}

/// Node k reads s_0 .. s_{k-1} in ascending order and computes
/// `x + max(s_j)`, so s_k = (k + 1) * x and a set on x changes every node.
/// Built bottom-up, so no first compute nests deeper than one level and
/// the whole walk after the set is iterative.
fn prefix_sum_dag<C: Cells>(rt: &Runtime<C>) {
    let n = test_size(2000);
    let x = rt.create_input(1_i64);
    let mut nodes: Vec<Incr<i64>> = Vec::with_capacity(n);
    for _ in 0..n {
        let below = nodes.clone();
        let q = rt.create_query(move |rt| {
            let mut best = 0;
            for &s in &below {
                best = best.max(rt.get(s));
            }
            rt.get(x) + best
        });
        nodes.push(q);
        rt.get(q);
    }
    let last = *nodes.last().unwrap();
    assert_eq!(rt.get(last), n as i64);

    rt.set(x, 2);
    assert_eq!(rt.get(last), 2 * n as i64);
    assert_eq!(rt.get(nodes[n / 2]), 2 * (n as i64 / 2 + 1));
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_prefix_sum_dag_is_not_a_cycle() {
    prefix_sum_dag(&Runtime::<Local>::new());
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_prefix_sum_dag_is_not_a_cycle() {
    prefix_sum_dag(&Runtime::<Shared>::new());
}

/// A ring of n members, each reading the shared gate `g` and, while it is
/// 0, its successor. The first read nests n levels before the read rules
/// find the ring's head on the stack with no strategy and raise the
/// cycle, and every member it unwinds takes the discard landing: New,
/// with no edges installed. Setting `g` breaks the cycle, and the next
/// walk visits each member once.
fn cycle_recovery_is_bounded<C: Cells>() {
    let rt = Runtime::<C>::new();
    let n = test_size(700);
    let g = rt.create_input(0_i64);
    let ring: Arc<OnceLock<Vec<Incr<i64>>>> = Arc::new(OnceLock::new());
    let members: Vec<Incr<i64>> = (0..n)
        .map(|k| {
            let ring = ring.clone();
            rt.create_query(move |rt| {
                if rt.get(g) == 0 {
                    rt.get(ring.get().unwrap()[(k + 1) % n]) + 1
                } else {
                    k as i64
                }
            })
        })
        .collect();
    ring.set(members.clone()).unwrap();

    let err = catch_unwind(AssertUnwindSafe(|| rt.get(members[0]))).unwrap_err();
    assert!(
        panic_message(err).contains("dependency cycle"),
        "the ring must panic as a cycle on its first read"
    );

    rt.set(g, 1);
    let before = rt.first_visits();
    assert_eq!(rt.get(members[0]), 0);
    let visits = rt.first_visits() - before;
    let edges = 2 * n as u64;
    assert!(
        visits <= edges + n as u64,
        "recovery walk made {visits} first visits for {edges} edges"
    );
    for (k, &m) in members.iter().enumerate() {
        assert_eq!(rt.get(m), k as i64);
    }
}

fn on_deep_stack(body: fn()) {
    thread::Builder::new()
        .stack_size(DEEP_STACK)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_cycle_recovery_is_bounded() {
    on_deep_stack(cycle_recovery_is_bounded::<Local>);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_cycle_recovery_is_bounded() {
    on_deep_stack(cycle_recovery_is_bounded::<Shared>);
}

/// The same ring broken at a single member: with `g` set, member 0 reads
/// only `g`, so its recorded edge to member 1 is stale. The recovery walk
/// re-expands the ring through that edge until the switch, and the first
/// recompute's nested walk reaches the recomputing member through it
/// again, as a Computing node on this thread's own frame.
fn single_point_break_recovery_is_bounded<C: Cells>() {
    let rt = Runtime::<C>::new();
    let n = test_size(700);
    let g = rt.create_input(0_i64);
    let ring: Arc<OnceLock<Vec<Incr<i64>>>> = Arc::new(OnceLock::new());
    let members: Vec<Incr<i64>> = (0..n)
        .map(|k| {
            let ring = ring.clone();
            rt.create_query(move |rt| {
                if k == 0 && rt.get(g) != 0 {
                    0
                } else {
                    rt.get(ring.get().unwrap()[(k + 1) % n]) + 1
                }
            })
        })
        .collect();
    ring.set(members.clone()).unwrap();

    let err = catch_unwind(AssertUnwindSafe(|| rt.get(members[0]))).unwrap_err();
    assert!(
        panic_message(err).contains("dependency cycle"),
        "the ring must panic as a cycle on its first read"
    );

    rt.set(g, 1);
    let before = rt.first_visits();
    assert_eq!(rt.get(members[0]), 0);
    let visits = rt.first_visits() - before;
    let edges = 2 * n as u64;
    assert!(
        visits <= u64::from(EXACT_MODE_AT) + edges,
        "recovery walk made {visits} first visits for {edges} edges"
    );
    for (k, &m) in members.iter().enumerate() {
        let expected = if k == 0 { 0 } else { (n - k) as i64 };
        assert_eq!(rt.get(m), expected);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_single_point_break_recovery_is_bounded() {
    on_deep_stack(single_point_break_recovery_is_bounded::<Local>);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_single_point_break_recovery_is_bounded() {
    on_deep_stack(single_point_break_recovery_is_bounded::<Shared>);
}

/// A ring a -> b -> c -> d -> e -> a that the first read lands with its
/// edges recorded, where `e` also reads `t1` above `a`. After `x` changes,
/// the walk of `t0` re-expands the Dirty ring until the switch to exact
/// mode, so it holds about `EXACT_MODE_AT / 5` pending entries for each
/// member. The members' computes don't land Clean, and each of those
/// entries would run them, and `e`'s closure, again.
fn dirty_ring_computes_each_member_once_per_walk<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(3_u64);
    let e_runs = Arc::new(AtomicU64::new(0));
    let a_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let t1_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let e = {
        let (a_cell, t1_cell, e_runs) = (a_cell.clone(), t1_cell.clone(), e_runs.clone());
        rt.create_query_with_fallback(
            move |rt| {
                e_runs.fetch_add(1, Ordering::Relaxed);
                let a = catch_unwind(AssertUnwindSafe(|| rt.get(*a_cell.get().unwrap())))
                    .unwrap_or(9999);
                if a % 2 == 0 {
                    0
                } else {
                    catch_unwind(AssertUnwindSafe(|| rt.get(*t1_cell.get().unwrap())))
                        .unwrap_or(9999)
                }
            },
            |_| 3,
        )
    };
    let d = rt.create_query(move |rt| rt.get(e));
    let c = rt.create_query(move |rt| rt.get(d));
    let b = rt.create_query_with(
        move |rt| rt.get(c),
        QueryOpts {
            cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 4).max_iterations(3)),
            replay: Replay::Native,
        },
    );
    let xq = rt.create_query(move |rt| rt.get(x));
    let a = rt.create_query(move |rt| {
        if rt.get(xq).wrapping_add(rt.get(b)) % 2 == 0 {
            0
        } else {
            rt.get(b)
        }
    });
    a_cell.set(a).unwrap();
    let t0 = rt.create_query(move |rt| rt.get(a));
    let t1 = rt.create_query(move |rt| rt.get(t0));
    t1_cell.set(t1).unwrap();
    assert_eq!(rt.get(t1), 0);

    rt.set(x, 2);
    e_runs.store(0, Ordering::Relaxed);
    assert_eq!(rt.get(t0), 3);
    let runs = e_runs.load(Ordering::Relaxed);
    assert!(runs <= 32, "e's closure ran {runs} times in one read");
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_dirty_ring_computes_each_member_once_per_walk() {
    dirty_ring_computes_each_member_once_per_walk::<Local>();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_dirty_ring_computes_each_member_once_per_walk() {
    dirty_ring_computes_each_member_once_per_walk::<Shared>();
}

/// `l` reads itself and then `n`, `n` reads `s` (twice when it is odd),
/// and `s` reads `l` once `x` is odd. After `x` changes, the walk of `l`
/// re-expands `l`'s self edge until the switch to exact mode, and each
/// lap reaches `n` and `s` with nothing to wait on. Their computes land
/// Dirty, so each lap would run their closures, and the nested walks
/// those start, again.
fn non_clean_landing_computes_once_per_walk<C: Cells>() {
    let rt = Runtime::<C>::new();
    let runs = Arc::new(AtomicU64::new(0));
    let l_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let x = rt.create_input(2_u64);
    let s = {
        let (runs, l_cell) = (runs.clone(), l_cell.clone());
        rt.create_query(move |rt| {
            runs.fetch_add(1, Ordering::Relaxed);
            if rt.get(x) % 2 == 0 {
                3
            } else {
                rt.get(*l_cell.get().unwrap())
            }
        })
    };
    let n = {
        let runs = runs.clone();
        rt.create_query(move |rt| {
            runs.fetch_add(1, Ordering::Relaxed);
            if rt.get(s) % 2 == 0 {
                0
            } else {
                rt.get(s)
            }
        })
    };
    let l = {
        let (runs, l_cell) = (runs.clone(), l_cell.clone());
        rt.create_query_with_fallback(
            move |rt| {
                runs.fetch_add(1, Ordering::Relaxed);
                rt.get(*l_cell.get().unwrap())
                    .wrapping_add(rt.try_get(n).unwrap_or(7777))
            },
            |_| 1,
        )
    };
    l_cell.set(l).unwrap();
    assert_eq!(rt.get(l), 1);

    rt.set(x, 3);
    runs.store(0, Ordering::Relaxed);
    assert_eq!(rt.get(l), 1);
    let runs = runs.load(Ordering::Relaxed);
    assert!(runs <= 32, "closures ran {runs} times in one read");
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_non_clean_landing_computes_once_per_walk() {
    non_clean_landing_computes_once_per_walk::<Local>();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_non_clean_landing_computes_once_per_walk() {
    non_clean_landing_computes_once_per_walk::<Shared>();
}

/// A recomputing `a` reads `c`, whose recorded edge to `a` is stale: `c`
/// stopped reading `a` when `s` changed, but has not recomputed yet. The
/// walk for `c` finds `a` Computing on this thread's own frame through
/// that edge.
fn direct_stale_edge_is_not_a_cycle<C: Cells>(rt: &Runtime<C>) {
    let s = rt.create_input(0_i64);
    let t = rt.create_input(0_i64);
    let i = rt.create_input(10_i64);
    let j = rt.create_input(20_i64);
    let cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let cell = cell.clone();
        rt.create_query(move |rt| {
            if rt.get(t) == 0 {
                rt.get(i)
            } else {
                rt.get(*cell.get().unwrap()) + 1
            }
        })
    };
    let c = rt.create_query(move |rt| {
        if rt.get(s) == 0 {
            rt.get(a) * 2
        } else {
            rt.get(j)
        }
    });
    cell.set(c).unwrap();
    assert_eq!(rt.get(c), 20);

    rt.set(s, 1);
    rt.set(t, 1);
    assert_eq!(rt.get(a), 21);
    assert_eq!(rt.get(c), 20);
    rt.set(j, 30);
    assert_eq!(rt.get(a), 31);
    rt.assert_invariant_i();
}

#[test]
fn local_direct_stale_edge_is_not_a_cycle() {
    direct_stale_edge_is_not_a_cycle(&Runtime::<Local>::new());
}

#[test]
fn shared_direct_stale_edge_is_not_a_cycle() {
    direct_stale_edge_is_not_a_cycle(&Runtime::<Shared>::new());
}

/// The stale edge through an intermediate node: `c` read `x`, `x` reads
/// `a`, `a` read `i`. Setting `j` makes `c` read only `j`, and setting
/// `i` makes `a` read `c`; neither `c` nor `x` has recomputed, so the walk
/// for `c` expands `x` and finds `a` Computing through `x`'s edge. The
/// stale edges are `c -> x` and `x -> a`; `x` is never recomputed.
fn intermediate_stale_edge_is_not_a_cycle<C: Cells>(rt: &Runtime<C>) {
    let i = rt.create_input(0_i64);
    let j = rt.create_input(0_i64);
    let cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let cell = cell.clone();
        rt.create_query(move |rt| {
            if rt.get(i) == 0 {
                100
            } else {
                rt.get(*cell.get().unwrap()) + 1
            }
        })
    };
    let x = rt.create_query(move |rt| rt.get(a) * 2);
    let c = rt.create_query(move |rt| {
        if rt.get(j) == 0 {
            rt.get(x) + 1
        } else {
            rt.get(j)
        }
    });
    cell.set(c).unwrap();
    assert_eq!(rt.get(c), 201);

    rt.set(j, 5);
    rt.set(i, 1);
    assert_eq!(rt.get(a), 6);
    assert_eq!(rt.get(c), 5);
    assert_eq!(rt.get(x), 12);
    rt.assert_invariant_i();
}

#[test]
fn local_intermediate_stale_edge_is_not_a_cycle() {
    intermediate_stale_edge_is_not_a_cycle(&Runtime::<Local>::new());
}

#[test]
fn shared_intermediate_stale_edge_is_not_a_cycle() {
    intermediate_stale_edge_is_not_a_cycle(&Runtime::<Shared>::new());
}

/// A Dirty chain whose bottom has a stale edge to `p`, which this thread
/// is recomputing and which now reads the chain's top. The top's walk
/// skips `p`, taints the whole chain and verifies each member; the top's
/// closure then recomputes the chain top-down through nested gets, each
/// of which must compute its node directly instead of expanding the rest
/// of the chain again. Returns (s, t, p, top).
fn taint_chain_graph<C: Cells>(
    rt: &Runtime<C>,
    n: usize,
) -> (Incr<i64>, Incr<i64>, Incr<i64>, Incr<i64>) {
    let s = rt.create_input(0_i64);
    let t = rt.create_input(0_i64);
    let i = rt.create_input(10_i64);
    let j = rt.create_input(20_i64);
    let top: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let p = {
        let top = top.clone();
        rt.create_query(move |rt| {
            if rt.get(t) == 0 {
                rt.get(i)
            } else {
                rt.get(*top.get().unwrap())
            }
        })
    };
    let mut prev = rt.create_query(move |rt| if rt.get(s) == 0 { rt.get(p) } else { rt.get(j) });
    for _ in 1..n {
        let below = prev;
        prev = rt.create_query(move |rt| rt.get(below) + 1);
    }
    top.set(prev).unwrap();
    (s, t, p, prev)
}

fn taint_chain_recomputes_in_linear_expansions<C: Cells>() {
    let n = test_size(10_000);
    let rt = Runtime::<C>::new();
    let (s, t, p, top) = taint_chain_graph(&rt, n);
    assert_eq!(rt.get(p), 10);
    assert_eq!(rt.get(top), 10 + n as i64 - 1);
    rt.set(s, 1);
    rt.set(t, 1);
    let before = rt.expansions();
    let value = rt.get(p);
    let expansions = rt.expansions() - before;

    let fresh = Runtime::<C>::new();
    let (fs, ft, fp, _) = taint_chain_graph(&fresh, n);
    fresh.set(fs, 1);
    fresh.set(ft, 1);
    assert_eq!(value, fresh.get(fp));
    assert_eq!(value, 20 + n as i64 - 1);
    assert!(
        expansions <= 2 * n as u64 + 8,
        "the taint chain took {expansions} expansions for {n} nodes"
    );
    rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_taint_chain_recomputes_in_linear_expansions() {
    on_deep_stack(taint_chain_recomputes_in_linear_expansions::<Local>);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_taint_chain_recomputes_in_linear_expansions() {
    on_deep_stack(taint_chain_recomputes_in_linear_expansions::<Shared>);
}

/// A reader's thread stack when the chain is recomputed through the walk.
const READER_STACK: usize = 256 << 10;

/// x -> q_0 -> q_1 -> ... -> q_{n-1}, made Clean by ascending gets. After
/// a set on x, one thread recomputes q_0 with a closure parked inside it
/// while a second thread, on a small stack, gets the top: its walk waits
/// on the first thread's claim, then recomputes the chain iteratively.
fn deep_contended_chain(nested: bool) {
    let n = test_size(100_000);
    let rt = Arc::new(Runtime::<Shared>::new());
    let x = rt.create_input(0_i64);
    let slow = Gate::new();
    let bottom = {
        let slow = slow.clone();
        rt.create_query(move |rt| {
            let v = rt.get(x);
            slow.park();
            v
        })
    };
    let mut chain = vec![bottom];
    for k in 1..n {
        let below = chain[k - 1];
        chain.push(rt.create_query(move |rt| rt.get(below) + 1));
    }
    for &q in &chain {
        rt.get(q);
    }
    let top = chain[n - 1];
    let outer = rt.create_query(move |rt| rt.get(top) * 2);
    assert_eq!(rt.get(outer), 2 * (n as i64 - 1));

    rt.set(x, 5);
    slow.arm();
    let recompute = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(bottom))
    };
    slow.wait_parked();

    let wait = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::BeforeComputeOne(id(bottom)), wait.clone())],
    );
    let reader = {
        let rt = Arc::clone(&rt);
        thread::Builder::new()
            .stack_size(READER_STACK)
            .spawn(move || if nested { rt.get(outer) } else { rt.get(top) })
            .unwrap()
    };
    wait.wait_parked();
    wait.release();
    thread::sleep(Duration::from_millis(50));
    slow.release();

    assert_eq!(recompute.join().unwrap(), 5);
    let expected = 5 + n as i64 - 1;
    assert_eq!(
        reader.join().unwrap(),
        if nested { 2 * expected } else { expected }
    );
    rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_deep_contended_chain_reader_waits_at_depth_0() {
    deep_contended_chain(false);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_deep_contended_chain_reader_waits_inside_a_compute() {
    deep_contended_chain(true);
}

/// `x` reads `sel` first and `f` only while `sel` is 0. After `f`'s input
/// makes it panic and `sel` moves to 1, `x`'s rerun reads `sel` alone, so
/// `get(x)` is 100 whether or not `f` was read, and landed Failed, before.
fn stale_dep_that_panics_is_not_read<C: Cells>(read_f_first: bool) {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let j = rt.create_input(1_i64);
    let f = rt.create_query(move |rt| {
        let v = rt.get(j);
        assert!(v != 0, "f boom");
        v
    });
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(f) } else { 100 });
    assert_eq!(rt.get(x), 1);

    rt.set(j, 0);
    rt.set(sel, 1);
    if read_f_first {
        let err = catch_unwind(AssertUnwindSafe(|| rt.get(f))).unwrap_err();
        assert_eq!(panic_message(err), "f boom");
    }
    assert_eq!(rt.get(x), 100);
    rt.set(sel, 0);
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_err();
    let expected = if read_f_first {
        "last compute panicked"
    } else {
        "f boom"
    };
    assert!(panic_message(err).contains(expected));
}

#[test]
fn local_stale_dep_that_panics_is_not_read() {
    stale_dep_that_panics_is_not_read::<Local>(false);
    stale_dep_that_panics_is_not_read::<Local>(true);
}

#[test]
fn shared_stale_dep_that_panics_is_not_read() {
    stale_dep_that_panics_is_not_read::<Shared>(false);
    stale_dep_that_panics_is_not_read::<Shared>(true);
}

/// `x` read `a` while `sel` was 0; setting `k` closes a cycle `a -> b -> a`
/// and setting `sel` stops `x` reading `a`. A run from scratch never meets
/// the cycle, so `get(x)` does not raise it, though the walk may run `a`.
fn stale_dep_in_a_cycle_is_not_read<C: Cells>() {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let k = rt.create_input(0_i64);
    let cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let cell = cell.clone();
        rt.create_query(move |rt| {
            if rt.get(k) == 0 {
                1
            } else {
                rt.get(*cell.get().unwrap()) + 1
            }
        })
    };
    let b = rt.create_query(move |rt| rt.get(a) + 1);
    cell.set(b).unwrap();
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(a) } else { 100 });
    assert_eq!(rt.get(x), 1);
    assert_eq!(rt.get(b), 2);

    rt.set(k, 1);
    rt.set(sel, 1);
    assert_eq!(rt.get(x), 100);
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(a))).unwrap_err();
    assert!(panic_message(err).contains("cycle"));
}

#[test]
fn local_stale_dep_in_a_cycle_is_not_read() {
    stale_dep_in_a_cycle_is_not_read::<Local>();
}

#[test]
fn shared_stale_dep_in_a_cycle_is_not_read() {
    stale_dep_in_a_cycle_is_not_read::<Shared>();
}

/// As above, but `a` first reads `s`, which `try_get`s `a` and so becomes a
/// held claim anchored at `a`'s own position before `b` closes the cycle.
/// `a`'s unwind releases that claim, so the walk still contains the cycle:
/// `get(x)` gives 100, as from scratch, and only a direct `get(a)` raises it.
fn wave6_containable_counts_claim_anchored_at_walk_entry<C: Cells>() {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let k = rt.create_input(0_i64);
    let s_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let b_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let (s_cell, b_cell) = (s_cell.clone(), b_cell.clone());
        rt.create_query(move |rt| {
            if rt.get(k) == 0 {
                1
            } else {
                rt.get(*s_cell.get().unwrap()) + rt.get(*b_cell.get().unwrap()) + 1
            }
        })
    };
    let s = rt.create_query(move |rt| rt.try_get(a).unwrap_or(5));
    let b = rt.create_query(move |rt| rt.get(a) + 1);
    s_cell.set(s).unwrap();
    b_cell.set(b).unwrap();
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(a) } else { 100 });
    assert_eq!(rt.get(x), 1);

    rt.set(k, 1);
    rt.set(sel, 1);
    assert_eq!(rt.get(x), 100);
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(a))).unwrap_err();
    assert!(panic_message(err).contains("cycle"));
}

#[test]
fn wave6_local_containable_counts_claim_anchored_at_walk_entry() {
    wave6_containable_counts_claim_anchored_at_walk_entry::<Local>();
}

#[test]
fn wave6_shared_containable_counts_claim_anchored_at_walk_entry() {
    wave6_containable_counts_claim_anchored_at_walk_entry::<Shared>();
}

/// The walk cleans every recorded dep before a node's rerun, so it runs
/// `e` after `sel` has stopped `x` reading it: a cost beta.5 has too,
/// which the result must not show.
fn stale_dep_gives_the_rerun_result<C: Cells>() {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let u = rt.create_input(0_i64);
    let j = rt.create_input(1_i64);
    let e = rt.create_query(move |rt| rt.get(j) * 10);
    let d = rt.create_query(move |rt| rt.get(j) + 1);
    let x = rt.create_query(move |rt| {
        if rt.get(sel) == 0 {
            rt.get(e)
        } else {
            rt.get(u) + rt.get(d)
        }
    });
    assert_eq!(rt.get(x), 10);
    rt.set(j, 2);
    rt.set(sel, 1);
    assert_eq!(rt.get(x), 3);
    rt.set(j, 5);
    assert_eq!(rt.get(x), 6);
}

#[test]
fn local_stale_dep_gives_the_rerun_result() {
    stale_dep_gives_the_rerun_result::<Local>();
}

#[test]
fn shared_stale_dep_gives_the_rerun_result() {
    stale_dep_gives_the_rerun_result::<Shared>();
}

/// A dep the walk cleans panics, and the rerun still reads it: the rerun
/// raises the dep's own payload, as a run from scratch does, and only a
/// later read sees "last compute panicked". Nothing is held once the dep
/// recovers.
fn walked_dep_panic_reaches_the_rerun_that_reads_it<C: Cells>() {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let j = rt.create_input(1_i64);
    let f = rt.create_query(move |rt| {
        let v = rt.get(j);
        assert!(v != 0, "f boom");
        v
    });
    let x = rt.create_query(move |rt| rt.get(sel) + rt.get(f));
    assert_eq!(rt.get(x), 1);

    rt.set(j, 0);
    rt.set(sel, 1);
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_err();
    assert_eq!(panic_message(err), "f boom");
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(f))).unwrap_err();
    assert!(panic_message(err).contains("last compute panicked"));

    rt.set(j, 2);
    assert_eq!(rt.get(x), 3);
    assert_eq!(rt.contained_payloads(), 0);
    assert_no_notes();
}

/// No unwind is in flight, so this thread keeps no landing for one.
fn assert_no_notes() {
    assert_eq!(value_code::thread_word() & value_code::NOTES_BIT, 0);
}

/// A contained panic nobody reads is held only while its node stays in
/// that Failed landing: a set that flips the node Dirty releases it, and
/// so does a delete of the node.
fn unread_contained_panic_is_released<C: Cells>() {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let j = rt.create_input(1_i64);
    let f = rt.create_query(move |rt| {
        let v = rt.get(j);
        assert!(v != 0, "f boom");
        v
    });
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(f) } else { 100 });
    let held = |rt: &Runtime<C>| {
        rt.set(j, 1);
        rt.set(sel, 0);
        assert_eq!(rt.get(x), 1);
        rt.set(j, 0);
        rt.set(sel, 1);
        assert_eq!(rt.get(x), 100);
        assert_eq!(rt.contained_payloads(), 1);
        assert_no_notes();
    };
    held(&rt);
    rt.set(j, 3);
    assert_eq!(rt.contained_payloads(), 0);

    held(&rt);
    rt.delete_node(x);
    rt.delete_node(f);
    assert_eq!(rt.contained_payloads(), 0);
    assert_no_notes();
}

#[test]
fn local_unread_contained_panic_is_released() {
    unread_contained_panic_is_released::<Local>();
}

#[test]
fn shared_unread_contained_panic_is_released() {
    unread_contained_panic_is_released::<Shared>();
}

#[test]
fn local_walked_dep_panic_reaches_the_rerun_that_reads_it() {
    walked_dep_panic_reaches_the_rerun_that_reads_it::<Local>();
}

#[test]
fn shared_walked_dep_panic_reaches_the_rerun_that_reads_it() {
    walked_dep_panic_reaches_the_rerun_that_reads_it::<Shared>();
}

/// How `x`'s stale dep `a` comes to run an episode that panics.
#[derive(Clone, Copy, PartialEq)]
enum EpisodeShape {
    /// `a` and `b` are a landed group, which the walk group-verifies.
    Group,
    /// The set of `k` closes the cycle, so `a`'s compute roots it.
    NewCycle,
}

/// `a`, a fixpoint from 0 capped at 5, reads `b` while `k` is 1, and `b`
/// reads `a`. `b` panics on an `a` of 3 or more while `j` is 0, which only
/// a fixpoint iteration reaches. `x` reads `sel`, and `a` while `sel` is 0
/// or always if `always_reads_a`. After `j` drops to 0 and `sel` moves to
/// 1, the walk of `x` runs `a`'s episode, whose member run of `b` panics.
/// Returns what `get(x)` gives then.
fn episode_panic_under_walk<C: Cells>(
    shape: EpisodeShape,
    always_reads_a: bool,
) -> std::thread::Result<i64> {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let j = rt.create_input(1_i64);
    let k = rt.create_input(i64::from(shape == EpisodeShape::Group));
    let cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let cell = cell.clone();
        rt.create_query_with(
            move |rt| {
                if rt.get(k) == 0 {
                    1
                } else {
                    rt.get(*cell.get().unwrap()).min(5)
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
        assert!(v < 3 || rt.get(j) != 0, "b boom");
        v + 1
    });
    cell.set(b).unwrap();
    let x = rt.create_query(move |rt| {
        let s = rt.get(sel);
        if always_reads_a || s == 0 {
            s + rt.get(a)
        } else {
            100
        }
    });
    let before = if shape == EpisodeShape::Group { 5 } else { 1 };
    assert_eq!(rt.get(x), before);
    assert_eq!(rt.get(b), before + 1);

    if shape == EpisodeShape::NewCycle {
        rt.set(k, 1);
    }
    rt.set(j, 0);
    rt.set(sel, 1);
    rt.group_verifies.lock().unwrap().clear();
    let episodes = rt.episodes.load(Ordering::Relaxed);
    let got = catch_unwind(AssertUnwindSafe(|| rt.get(x)));
    assert!(rt.episodes.load(Ordering::Relaxed) > episodes);
    assert_eq!(
        rt.group_verifies.lock().unwrap().contains(&a.slot()),
        shape == EpisodeShape::Group
    );
    if got.is_ok() {
        // The episode's panic stays with its Failed member for a reader.
        let err = catch_unwind(AssertUnwindSafe(|| rt.get(a))).unwrap_err();
        assert_eq!(panic_message(err), "b boom");
    }
    assert_no_notes();
    got
}

#[test]
fn local_stale_group_dep_whose_verify_panics_is_not_read() {
    let got = episode_panic_under_walk::<Local>(EpisodeShape::Group, false);
    assert_eq!(got.ok(), Some(100));
}

#[test]
fn shared_stale_group_dep_whose_verify_panics_is_not_read() {
    let got = episode_panic_under_walk::<Shared>(EpisodeShape::Group, false);
    assert_eq!(got.ok(), Some(100));
}

#[test]
fn local_stale_dep_whose_episode_panics_is_not_read() {
    let got = episode_panic_under_walk::<Local>(EpisodeShape::NewCycle, false);
    assert_eq!(got.ok(), Some(100));
}

#[test]
fn shared_stale_dep_whose_episode_panics_is_not_read() {
    let got = episode_panic_under_walk::<Shared>(EpisodeShape::NewCycle, false);
    assert_eq!(got.ok(), Some(100));
}

/// The rerun still reads `a`, so the read raises the episode's panic, as a
/// run from scratch does.
fn walked_episode_panic_reaches_the_rerun_that_reads_it<C: Cells>() {
    for shape in [EpisodeShape::Group, EpisodeShape::NewCycle] {
        let err = episode_panic_under_walk::<C>(shape, true).unwrap_err();
        assert_eq!(panic_message(err), "b boom");
    }
}

#[test]
fn local_walked_episode_panic_reaches_the_rerun_that_reads_it() {
    walked_episode_panic_reaches_the_rerun_that_reads_it::<Local>();
}

#[test]
fn shared_walked_episode_panic_reaches_the_rerun_that_reads_it() {
    walked_episode_panic_reaches_the_rerun_that_reads_it::<Shared>();
}

/// A running total `a_k = b + a_{k-1}` reads the changed `b` before its
/// predecessor; the walk still cleans the chain bottom-up, so a set of `b`
/// recomputes 100k nodes without nesting their closures.
fn running_total_chain<C: Cells>() -> i64 {
    let rt = Runtime::<C>::new();
    let b = rt.create_input(1_i64);
    let mut prev: Option<Incr<i64>> = None;
    for _ in 0..100_000 {
        let p = prev;
        let q = rt.create_query(move |rt| rt.get(b) + p.map_or(0, |p| rt.get(p)));
        rt.get(q);
        prev = Some(q);
    }
    rt.set(b, 2);
    rt.get(prev.unwrap())
}

fn on_8_mib_stack(f: fn() -> i64) -> i64 {
    thread::Builder::new()
        .stack_size(8 << 20)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap()
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_running_total_chain_recomputes_without_nesting() {
    assert_eq!(on_8_mib_stack(running_total_chain::<Local>), 200_000);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_running_total_chain_recomputes_without_nesting() {
    assert_eq!(on_8_mib_stack(running_total_chain::<Shared>), 200_000);
}

struct Counted(Arc<AtomicU64>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// A child swept while it holds a contained panic drops that payload.
fn swept_child_drops_its_contained_panic<C: Cells>() {
    let rt = Runtime::<C>::new();
    let drops = Arc::new(AtomicU64::new(0));
    let sel = rt.create_input(0_i64);
    let dc = drops.clone();
    let x = rt.create_query(move |rt| {
        let s = rt.get(sel);
        let dc = dc.clone();
        let c = rt.child_query(s, move |rt| {
            if rt.get(sel) != s {
                std::panic::panic_any(Counted(dc.clone()));
            }
            s
        });
        rt.get(c)
    });
    assert_eq!(rt.get(x), 0);
    for k in 1..=8 {
        rt.set(sel, k);
        assert_eq!(rt.get(x), k);
    }
    assert_eq!(drops.load(Ordering::SeqCst), 8);
}

#[test]
fn local_swept_child_drops_its_contained_panic() {
    swept_child_drops_its_contained_panic::<Local>();
}

#[test]
fn shared_swept_child_drops_its_contained_panic() {
    swept_child_drops_its_contained_panic::<Shared>();
}

/// A closure that catches a nested read's panic and raises a fresh box
/// must not hand the caught node's stale note to its own contained ticket,
/// even when the allocator gives the new box the freed address.
fn caught_note_stays_out_of_a_reused_address_ticket<C: Cells>() {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let t = rt.create_input(0_i64);
    let x = rt.create_query(|_: &Runtime<C>| -> i64 { panic!("pX") });
    let reused = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let r = reused.clone();
    let d = rt.create_query(move |rt| {
        if rt.get(t) == 1 {
            let caught = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_err();
            let freed = &*caught as *const _ as *const u8 as usize;
            drop(caught);
            let fresh: Box<dyn std::any::Any + Send> = Box::new("pD");
            r.store(
                &*fresh as *const _ as *const u8 as usize == freed,
                Ordering::SeqCst,
            );
            std::panic::resume_unwind(fresh);
        }
        0
    });
    let n = rt.create_query(move |rt| {
        if rt.get(sel) == 1 {
            rt.get(x)
        } else {
            rt.get(d)
        }
    });
    assert_eq!(rt.get(n), 0);
    rt.set(t, 1);
    rt.set(sel, 1);
    let got = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(n))).unwrap_err());
    // Whether the allocator reused the address is its choice (Miri often
    // does not): either way d's catch dropped x's payload, so x's read has
    // none to raise.
    let path = if reused.load(Ordering::SeqCst) {
        "at the freed address"
    } else {
        "at a fresh address"
    };
    assert!(
        got.contains("last compute panicked"),
        "x's read raised {got:?} with d's box {path}"
    );
    let got = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(d))).unwrap_err());
    assert_eq!(got, "pD");
}

#[test]
fn local_caught_note_stays_out_of_a_reused_address_ticket() {
    caught_note_stays_out_of_a_reused_address_ticket::<Local>();
}

#[test]
fn shared_caught_note_stays_out_of_a_reused_address_ticket() {
    caught_note_stays_out_of_a_reused_address_ticket::<Shared>();
}

struct XZst;
struct DZst;

/// As [`caught_note_stays_out_of_a_reused_address_ticket`] with
/// zero-sized payloads, whose boxes all share one dangling address and
/// carry no text: x's note must stay out of the ticket for d's fresh
/// payload of another type.
fn caught_zst_note_stays_out_of_another_types_ticket<C: Cells>() {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let t = rt.create_input(0_i64);
    let x = rt.create_query(|_: &Runtime<C>| -> i64 { std::panic::panic_any(XZst) });
    let d = rt.create_query(move |rt| {
        if rt.get(t) == 1 {
            drop(catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_err());
            std::panic::panic_any(DZst);
        }
        0
    });
    let n = rt.create_query(move |rt| {
        if rt.get(sel) == 1 {
            rt.get(x)
        } else {
            rt.get(d)
        }
    });
    assert_eq!(rt.get(n), 0);
    rt.set(t, 1);
    rt.set(sel, 1);
    let got = catch_unwind(AssertUnwindSafe(|| rt.get(n))).unwrap_err();
    assert!(!got.is::<DZst>(), "x's read took d's contained payload");
    assert!(
        panic_message(got).contains("last compute panicked"),
        "x's read raised a payload"
    );
    let got = catch_unwind(AssertUnwindSafe(|| rt.get(d))).unwrap_err();
    assert!(got.is::<DZst>(), "d's read lost its payload");
}

#[test]
fn local_caught_zst_note_stays_out_of_another_types_ticket() {
    caught_zst_note_stays_out_of_another_types_ticket::<Local>();
}

#[test]
fn shared_caught_zst_note_stays_out_of_another_types_ticket() {
    caught_zst_note_stays_out_of_another_types_ticket::<Shared>();
}

/// f catches t's held payload, then reads r2, whose walk entry e runs at the
/// cidx the read was noted at, then re-raises the payload, which q's walk
/// contains. Whether e panics must not change who holds t's payload: t's
/// read keeps raising "pT" and f's read reports a panicked last compute.
fn wave6_ticket_note_survives_a_sibling_entry<C: Cells>(sibling_panics: bool) {
    let rt = Runtime::<C>::new();
    let i = rt.create_input(0_i64);
    let sel = rt.create_input(0_i64);
    let j = rt.create_input(1_i64);
    let m = rt.create_input(0_i64);
    let k = rt.create_input(0_i64);
    let qs = rt.create_input(0_i64);
    let t = rt.create_query(move |rt: &Runtime<C>| -> i64 {
        if rt.get(i) == 1 {
            panic!("pT")
        }
        0
    });
    let rec = rt.create_query(move |rt: &Runtime<C>| if rt.get(sel) == 0 { rt.get(t) } else { 5 });
    let e = rt.create_query(move |rt: &Runtime<C>| {
        let v = rt.get(j);
        assert!(v != 0, "pE");
        v
    });
    let r2 = rt.create_query(move |rt: &Runtime<C>| if rt.get(m) == 0 { rt.get(e) } else { 1 });
    let f = rt.create_query(move |rt: &Runtime<C>| {
        if rt.get(k) == 0 {
            return 0;
        }
        let c = catch_unwind(AssertUnwindSafe(|| rt.get(t)));
        let w = rt.get(r2);
        match c {
            Ok(v) => v + w,
            Err(p) => std::panic::resume_unwind(p),
        }
    });
    let q = rt.create_query(move |rt: &Runtime<C>| if rt.get(qs) == 0 { rt.get(f) } else { 7 });
    assert_eq!(rt.get(rec), 0);
    assert_eq!(rt.get(r2), 1);
    assert_eq!(rt.get(q), 0);
    rt.set(i, 1);
    rt.set(sel, 1);
    assert_eq!(rt.get(rec), 5);
    assert_eq!(rt.contained_payloads(), 1);
    rt.set(j, if sibling_panics { 0 } else { 2 });
    rt.set(m, 1);
    rt.set(k, 1);
    rt.set(qs, 1);
    assert_eq!(rt.get(q), 7);
    assert_no_notes();
    let got = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(t))).unwrap_err());
    assert_eq!(got, "pT", "sibling_panics={sibling_panics}");
    let got = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(f))).unwrap_err());
    assert!(
        got.contains("last compute panicked"),
        "sibling_panics={sibling_panics}: f raised {got:?}"
    );
    assert_no_notes();
}

#[test]
fn wave6_contained_sibling_entry_keeps_a_caught_ticket_note() {
    wave6_ticket_note_survives_a_sibling_entry::<Local>(false);
    wave6_ticket_note_survives_a_sibling_entry::<Shared>(false);
    wave6_ticket_note_survives_a_sibling_entry::<Local>(true);
    wave6_ticket_note_survives_a_sibling_entry::<Shared>(true);
}

/// As `wave6_ticket_note_survives_a_sibling_entry`, with f's second read a
/// New s that lands Failed at the position t's read was noted at, and f
/// catches s's panic too before it re-raises t's payload.
fn ticket_note_survives_a_sibling_landing<C: Cells>(sibling_panics: bool) {
    let rt = Runtime::<C>::new();
    let i = rt.create_input(0_i64);
    let sel = rt.create_input(0_i64);
    let j = rt.create_input(i64::from(!sibling_panics));
    let k = rt.create_input(0_i64);
    let qs = rt.create_input(0_i64);
    let t = rt.create_query(move |rt: &Runtime<C>| -> i64 {
        if rt.get(i) == 1 {
            panic!("pT")
        }
        0
    });
    let rec = rt.create_query(move |rt: &Runtime<C>| if rt.get(sel) == 0 { rt.get(t) } else { 5 });
    let s = rt.create_query(move |rt: &Runtime<C>| {
        let v = rt.get(j);
        assert!(v != 0, "pS");
        v
    });
    let f = rt.create_query(move |rt: &Runtime<C>| {
        if rt.get(k) == 0 {
            return 0;
        }
        let c = catch_unwind(AssertUnwindSafe(|| rt.get(t)));
        let w = catch_unwind(AssertUnwindSafe(|| rt.get(s))).unwrap_or(1);
        match c {
            Ok(v) => v + w,
            Err(p) => std::panic::resume_unwind(p),
        }
    });
    let q = rt.create_query(move |rt: &Runtime<C>| if rt.get(qs) == 0 { rt.get(f) } else { 7 });
    assert_eq!(rt.get(rec), 0);
    assert_eq!(rt.get(q), 0);
    rt.set(i, 1);
    rt.set(sel, 1);
    assert_eq!(rt.get(rec), 5);
    assert_eq!(rt.contained_payloads(), 1);
    rt.set(k, 1);
    rt.set(qs, 1);
    assert_eq!(rt.get(q), 7);
    assert_no_notes();
    let got = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(t))).unwrap_err());
    assert_eq!(got, "pT", "sibling_panics={sibling_panics}");
    let got = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(f))).unwrap_err());
    assert!(
        got.contains("last compute panicked"),
        "sibling_panics={sibling_panics}: f raised {got:?}"
    );
    if sibling_panics {
        let got = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(s))).unwrap_err());
        assert!(got.contains("last compute panicked"), "s raised {got:?}");
    }
    assert_no_notes();
}

#[test]
fn caught_sibling_landing_keeps_a_caught_ticket_note() {
    ticket_note_survives_a_sibling_landing::<Local>(false);
    ticket_note_survives_a_sibling_landing::<Shared>(false);
    ticket_note_survives_a_sibling_landing::<Local>(true);
    ticket_note_survives_a_sibling_landing::<Shared>(true);
}
