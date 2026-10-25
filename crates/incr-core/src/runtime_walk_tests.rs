//! The `ensure_clean` walk: its cost on wide DAGs and its recovery after a
//! cycle panic, each case run under `Local` and `Shared`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use super::race_tests::{id, park_at, Gate};
use super::{ParkPoint, Runtime, Walk, EXACT_MODE_AT, NO_TAINT_NEW};
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{CycleStrategy, Fixpoint, QueryOpts, Replay};
use crate::handle::Incr;
use crate::node::NodeId;
use crate::state::NodeState;
use crate::test_size::test_size;
use crate::value_code;

/// Nested first computes overflow the default test thread stack somewhere
/// between 600 and 700 levels in debug builds. The taint chain's 10,000
/// native nested gets need 240 to 248 MiB in debug, about 25 KiB each, so
/// this leaves them twice that.
const DEEP_STACK: usize = 512 << 20;

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
/// and setting `sel` stops `x` reading `a`. The walk of `x` cleans its
/// recorded dep `a` before `x` reruns, and the cycle `a`'s run closes
/// reaches `get(x)`, as on main, though a run from scratch never meets it:
/// a known limitation until a walk can tell a dep its reader still reads.
fn stale_dep_in_a_cycle_reaches_the_reader<C: Cells>() {
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
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_err();
    assert_eq!(panic_message(err), "incr-core: dependency cycle: 2, 3");
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(a))).unwrap_err();
    assert!(panic_message(err).contains("cycle"));
}

#[test]
fn local_stale_dep_in_a_cycle_reaches_the_reader() {
    stale_dep_in_a_cycle_reaches_the_reader::<Local>();
}

#[test]
fn shared_stale_dep_in_a_cycle_reaches_the_reader() {
    stale_dep_in_a_cycle_reaches_the_reader::<Shared>();
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

/// As `stale_dep_in_a_cycle_is_not_read`, but `a` catches the cycle unwind
/// and then panics on its own. The walk entry must contain that panic as it
/// contains the cycle, so `get(x)` gives 100, as from scratch, and only a
/// direct `get(a)` raises "a own".
fn wave7_walk_contains_own_panic_after_swallowed_cycle<C: Cells>() {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let k = rt.create_input(0_i64);
    let cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let cell = cell.clone();
        rt.create_query(move |rt| -> i64 {
            if rt.get(k) == 0 {
                1
            } else {
                let _ = catch_unwind(AssertUnwindSafe(|| rt.get(*cell.get().unwrap())));
                panic!("a own")
            }
        })
    };
    let b = rt.create_query(move |rt| rt.get(a) + 1);
    cell.set(b).unwrap();
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(a) } else { 100 });
    assert_eq!(rt.get(x), 1);

    rt.set(k, 1);
    rt.set(sel, 1);
    let got = catch_unwind(AssertUnwindSafe(|| rt.get(x)));
    assert_eq!(got.map_err(panic_message), Ok(100));
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(a))).unwrap_err();
    assert_eq!(panic_message(err), "a own");
}

#[test]
fn wave7_local_walk_contains_own_panic_after_swallowed_cycle() {
    wave7_walk_contains_own_panic_after_swallowed_cycle::<Local>();
}

#[test]
fn wave7_shared_walk_contains_own_panic_after_swallowed_cycle() {
    wave7_walk_contains_own_panic_after_swallowed_cycle::<Shared>();
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

/// `y` read `e` while `sel` was 0; setting `k` makes `e` catch a cycle and
/// land Dirty, and `sel` is set to `sel_after`. `z`'s first run reads `y`,
/// whose walk cleans `e` first. Returns `z`'s and `e`'s states after it.
fn walk_entry_reader<C: Cells>(frames: Option<u32>, sel_after: i64) -> (NodeState, NodeState) {
    let rt = Runtime::<C>::new();
    if let Some(frames) = frames {
        rt.set_max_frames(frames);
    }
    let sel = rt.create_input(0_i64);
    let k = rt.create_input(0_i64);
    let cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let c1 = {
        let cell = cell.clone();
        rt.create_query(move |rt| rt.get(*cell.get().unwrap()) + 1)
    };
    let c2 = rt.create_query(move |rt| rt.get(c1) + 1);
    cell.set(c2).unwrap();
    let e = rt.create_query(move |rt| {
        if rt.get(k) == 0 {
            1
        } else {
            catch_unwind(AssertUnwindSafe(|| rt.get(c1))).unwrap_or(7)
        }
    });
    let y = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(e) } else { 5 });
    assert_eq!(rt.get(y), 1);

    rt.set(k, 1);
    rt.set(sel, sel_after);
    let z = rt.create_query(move |rt| rt.get(y) + 1);
    assert_eq!(rt.get(z), if sel_after == 0 { 8 } else { 6 });
    (rt.state_of(z), rt.state_of(e))
}

/// When `y`'s rerun no longer reads `e`, `z` read nothing that did not land
/// Clean and lands Clean, as from scratch, at every frame budget (seed
/// 0x77d).
fn unread_walk_entry_leaves_the_reader_clean<C: Cells>() {
    for frames in [None, Some(1), Some(2), Some(3)] {
        let (z, e) = walk_entry_reader::<C>(frames, 1);
        assert_ne!(e, NodeState::Clean, "frames {frames:?}");
        assert_eq!(z, NodeState::Clean, "frames {frames:?}");
    }
}

#[test]
fn local_unread_walk_entry_leaves_the_reader_clean() {
    unread_walk_entry_leaves_the_reader_clean::<Local>();
}

#[test]
fn shared_unread_walk_entry_leaves_the_reader_clean() {
    unread_walk_entry_leaves_the_reader_clean::<Shared>();
}

/// When `y`'s rerun still reads `e`, `z` read a value built on `e`'s Dirty
/// landing and lands Dirty, at every frame budget.
fn read_walk_entry_leaves_the_reader_dirty<C: Cells>() {
    for frames in [None, Some(1), Some(2), Some(3)] {
        let (z, e) = walk_entry_reader::<C>(frames, 0);
        assert_ne!(e, NodeState::Clean, "frames {frames:?}");
        assert_eq!(z, NodeState::Dirty, "frames {frames:?}");
    }
}

#[test]
fn local_read_walk_entry_leaves_the_reader_dirty() {
    read_walk_entry_leaves_the_reader_dirty::<Local>();
}

#[test]
fn shared_read_walk_entry_leaves_the_reader_dirty() {
    read_walk_entry_leaves_the_reader_dirty::<Shared>();
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

/// A read outside any compute takes a contained panic, which no frame
/// below can contain again: the ticket goes with it, so none is held or
/// lent after, however often the read repeats.
fn top_level_read_takes_the_ticket<C: Cells>() {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let j = rt.create_input(1_i64);
    let f = rt.create_query(move |rt| {
        let v = rt.get(j);
        assert!(v != 0, "f boom");
        v
    });
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(f) } else { 100 });
    for round in 0..3 {
        rt.set(j, 1);
        rt.set(sel, 0);
        assert_eq!(rt.get(x), 1);
        rt.set(j, 0);
        rt.set(sel, 1);
        assert_eq!(rt.get(x), 100);
        assert_eq!(rt.contained_payloads(), 1);
        let err = catch_unwind(AssertUnwindSafe(|| rt.get(f))).unwrap_err();
        assert!(panic_message(err).contains("f boom"));
        assert_eq!(rt.tickets_kept(), (0, false), "round {round}");
        assert_no_notes();
    }
}

#[test]
fn local_top_level_read_takes_the_ticket() {
    top_level_read_takes_the_ticket::<Local>();
}

#[test]
fn shared_top_level_read_takes_the_ticket() {
    top_level_read_takes_the_ticket::<Shared>();
}

#[test]
fn local_unread_contained_panic_is_released() {
    unread_contained_panic_is_released::<Local>();
}

#[test]
fn shared_unread_contained_panic_is_released() {
    unread_contained_panic_is_released::<Shared>();
}

/// A payload no engine copy can stand for.
struct NotClone(i64);

/// Builds the held "f boom" ticket of `unread_contained_panic_is_released`
/// (a `NotClone(6)` payload if `custom`), then has a replayable h catch f's
/// panic, fold its text length or number into its value and read a New g,
/// at frame budget `budget` (or none).
fn wave6_doomed_run_after_taking_a_ticket<C: Cells>(budget: Option<u32>, custom: bool) -> i64 {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_i64);
    let sel = rt.create_input(0_i64);
    let j = rt.create_input(1_i64);
    let f = rt.create_query(move |rt| {
        let v = rt.get(j);
        if v == 0 && custom {
            std::panic::panic_any(NotClone(6));
        }
        assert!(v != 0, "f boom");
        v
    });
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(f) } else { 100 });
    assert_eq!(rt.get(x), 1);
    rt.set(j, 0);
    rt.set(sel, 1);
    assert_eq!(rt.get(x), 100);
    assert_eq!(rt.contained_payloads(), 1);
    let g = rt.create_query(move |rt| rt.get(j) + 10);
    let h = rt.create_query_with(
        move |rt: &Runtime<C>| {
            let n = match catch_unwind(AssertUnwindSafe(|| rt.get(f))) {
                Ok(v) => v,
                Err(p) => match p.downcast::<NotClone>() {
                    Ok(n) => n.0,
                    Err(p) => panic_message(p).len() as i64,
                },
            };
            n + rt.get(g)
        },
        QueryOpts {
            replay: Replay::Allowed,
            ..Default::default()
        },
    );
    rt.set_max_frames(budget.unwrap_or(u32::MAX));
    rt.get(h)
}

/// A replayable run that takes a contained ticket, drops the payload and is
/// doomed by a later read past the budget must leave its retry the same
/// payload: h is `6 + g` = 16, as unbounded. A payload no copy can stand
/// for keeps h from being doomed, past the budget too since f is not
/// replayable and so can't run again to regenerate it (`hold_uncopied`).
fn wave6_doomed_run_keeps_the_ticket_it_took<C: Cells>() {
    for custom in [false, true] {
        let unbounded = wave6_doomed_run_after_taking_a_ticket::<C>(None, custom);
        assert_eq!(unbounded, 6 + 10, "custom {custom}");
        for b in 1..=3 {
            assert_eq!(
                wave6_doomed_run_after_taking_a_ticket::<C>(Some(b), custom),
                unbounded,
                "budget {b}, custom {custom}"
            );
        }
    }
}

#[test]
fn wave6_local_doomed_run_keeps_the_ticket_it_took() {
    wave6_doomed_run_keeps_the_ticket_it_took::<Local>();
}

#[test]
fn wave6_shared_doomed_run_keeps_the_ticket_it_took() {
    wave6_doomed_run_keeps_the_ticket_it_took::<Shared>();
}

/// Under x's walk, a replayable g catches the panic of a never-read r, reads
/// the replayable chain dd1..dd4 and re-raises, at frame budget `budget` (or
/// none), holding the caught box across a doom so the retry's copy
/// cannot reuse its address. With `deep`, r re-raises the panic of its dep d,
/// whose landing sits a frame deeper. With `replay_r`, r is replayable too.
/// Returns x, then the messages of d (empty without `deep`), r and g.
fn wave7_kept_read_after_fresh_failed_read<C: Cells>(
    budget: Option<u32>,
    deep: bool,
    replay_r: bool,
) -> (i64, String, String, String) {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_i64);
    let sel = rt.create_input(0_i64);
    let k = rt.create_input(0_i64);
    let replay = || QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    };
    let d = rt.create_query(move |_rt: &Runtime<C>| -> i64 { panic!("r boom") });
    let r = rt.create_query_with(
        move |rt: &Runtime<C>| -> i64 {
            if deep {
                rt.get(d)
            } else {
                panic!("r boom")
            }
        },
        if replay_r {
            replay()
        } else {
            QueryOpts::default()
        },
    );
    let dd4 = rt.create_query_with(move |_rt: &Runtime<C>| 1_i64, replay());
    let dd3 = rt.create_query_with(move |rt: &Runtime<C>| rt.get(dd4) + 1, replay());
    let dd2 = rt.create_query_with(move |rt: &Runtime<C>| rt.get(dd3) + 1, replay());
    let dd1 = rt.create_query_with(move |rt: &Runtime<C>| rt.get(dd2) + 1, replay());
    let held: Arc<Mutex<Option<Box<dyn std::any::Any + Send>>>> = Arc::new(Mutex::new(None));
    let g = rt.create_query_with(
        move |rt: &Runtime<C>| {
            if rt.get(k) == 0 {
                return 0;
            }
            let c = catch_unwind(AssertUnwindSafe(|| rt.get(r))).map_err(|p| {
                *held.lock().unwrap() = Some(p);
            });
            let v = rt.get(dd1);
            match c {
                Ok(n) => n + v,
                Err(()) => {
                    let p = held.lock().unwrap().take().unwrap();
                    std::panic::resume_unwind(p)
                }
            }
        },
        replay(),
    );
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(g) } else { 100 });
    assert_eq!(rt.get(x), 0);
    rt.set(k, 1);
    rt.set(sel, 1);
    rt.set_max_frames(budget.unwrap_or(u32::MAX));
    let v = rt.get(x);
    let md = if deep {
        panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(d))).unwrap_err())
    } else {
        String::new()
    };
    let mr = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(r))).unwrap_err());
    let mg = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(g))).unwrap_err());
    assert_no_notes();
    (v, md, mr, mg)
}

/// The walk contains g's re-raised panic with r's own Failed landing, so r
/// raises "r boom" and g reports its last compute panicked. With `deep`, g's
/// read of dd1 retired d's deeper landing first, so d reports its last
/// compute panicked. A doomed run's retry of g, or of a replayable r, must
/// leave the same ticket at every budget.
fn wave7_kept_read_keeps_the_read_nodes_landing<C: Cells>(deep: bool) {
    let unbounded = wave7_kept_read_after_fresh_failed_read::<C>(None, deep, false);
    assert_eq!(unbounded.0, 100);
    assert_eq!(
        unbounded.1.contains("last compute panicked"),
        deep,
        "{}",
        unbounded.1
    );
    assert_eq!(unbounded.2, "r boom");
    assert!(
        unbounded.3.contains("last compute panicked"),
        "{}",
        unbounded.3
    );
    for replay_r in [false, true] {
        for b in 1..=8 {
            assert_eq!(
                wave7_kept_read_after_fresh_failed_read::<C>(Some(b), deep, replay_r),
                unbounded,
                "budget {b}, deep {deep}, replay_r {replay_r}"
            );
        }
    }
}

#[test]
fn wave7_local_kept_read_keeps_the_read_nodes_landing() {
    wave7_kept_read_keeps_the_read_nodes_landing::<Local>(false);
}

#[test]
fn wave7_shared_kept_read_keeps_the_read_nodes_landing() {
    wave7_kept_read_keeps_the_read_nodes_landing::<Shared>(false);
}

#[test]
fn wave7_local_kept_read_keeps_the_read_nodes_deeper_landing_retired() {
    wave7_kept_read_keeps_the_read_nodes_landing::<Local>(true);
}

#[test]
fn wave7_shared_kept_read_keeps_the_read_nodes_deeper_landing_retired() {
    wave7_kept_read_keeps_the_read_nodes_landing::<Shared>(true);
}

/// Under x's walk, a replayable g catches the panic of a never-read a, sets
/// k and catches a's second panic through b, which lands a Failed again,
/// then reads the replayable chain dd1..dd4 and re-raises a's first panic,
/// at frame budget `budget` (or none). Returns x, then the messages
/// of a and g.
fn wave7_kept_read_after_relanded_read<C: Cells>(budget: Option<u32>) -> (i64, String, String) {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_i64);
    let sel = rt.create_input(0_i64);
    let on = rt.create_input(0_i64);
    let k = rt.create_input(0_i64);
    let a = rt.create_query(move |rt: &Runtime<C>| -> i64 {
        let kk = rt.get(k);
        panic!("a boom {kk}")
    });
    let b = rt.create_query(move |rt: &Runtime<C>| rt.get(a));
    let replay = || QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    };
    let dd4 = rt.create_query_with(move |_rt: &Runtime<C>| 1_i64, replay());
    let dd3 = rt.create_query_with(move |rt: &Runtime<C>| rt.get(dd4) + 1, replay());
    let dd2 = rt.create_query_with(move |rt: &Runtime<C>| rt.get(dd3) + 1, replay());
    let dd1 = rt.create_query_with(move |rt: &Runtime<C>| rt.get(dd2) + 1, replay());
    let held: Arc<Mutex<Option<Box<dyn std::any::Any + Send>>>> = Arc::new(Mutex::new(None));
    let g = rt.create_query_with(
        move |rt: &Runtime<C>| {
            if rt.get(on) == 0 {
                return 0;
            }
            let c = catch_unwind(AssertUnwindSafe(|| rt.get(a))).map_err(|p| {
                *held.lock().unwrap() = Some(p);
            });
            if c.is_err() {
                rt.set(k, 1);
                let _ = catch_unwind(AssertUnwindSafe(|| rt.get(b)));
            }
            let v = rt.get(dd1);
            match c {
                Ok(n) => n + v,
                Err(()) => {
                    let p = held.lock().unwrap().take().unwrap();
                    std::panic::resume_unwind(p)
                }
            }
        },
        replay(),
    );
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(g) } else { 100 });
    assert_eq!(rt.get(x), 0);
    rt.set(on, 1);
    rt.set(sel, 1);
    rt.set_max_frames(budget.unwrap_or(u32::MAX));
    let v = rt.get(x);
    let ma = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(a))).unwrap_err());
    let mg = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(g))).unwrap_err());
    assert_no_notes();
    (v, ma, mg)
}

/// a's second Failed landing replaces its first, so no ticket holds the
/// payload g re-raises for a, at any budget: a kept copy of g's read must
/// lose a as the live note does. With `rerun`, g's message is compared
/// too: x's walk reruns g after its flagged Dirty landing, and that run
/// must read a afresh.
fn wave7_kept_read_drops_a_relanded_node<C: Cells>(rerun: bool) {
    let unbounded = wave7_kept_read_after_relanded_read::<C>(None);
    assert!(
        unbounded.1.contains("last compute panicked"),
        "{unbounded:?}"
    );
    for b in 1..=8 {
        let got = wave7_kept_read_after_relanded_read::<C>(Some(b));
        if rerun {
            assert_eq!(got, unbounded, "budget {b}");
        } else {
            assert_eq!((got.0, &got.1), (unbounded.0, &unbounded.1), "budget {b}");
        }
    }
}

#[test]
fn wave7_local_kept_read_drops_a_relanded_node() {
    wave7_kept_read_drops_a_relanded_node::<Local>(false);
}

#[test]
fn wave7_shared_kept_read_drops_a_relanded_node() {
    wave7_kept_read_drops_a_relanded_node::<Shared>(false);
}

#[test]
fn wave7_local_rerun_reads_a_relanded_node_afresh() {
    wave7_kept_read_drops_a_relanded_node::<Local>(true);
}

#[test]
fn wave7_shared_rerun_reads_a_relanded_node_afresh() {
    wave7_kept_read_drops_a_relanded_node::<Shared>(true);
}

/// Under x's walk, a replayable g catches the panic of a never-read r, which
/// re-raises its dep s's, then reads the replayable chain dd1..dd4, where
/// dd1 panics after reading dd2; g catches that and re-raises r's panic, at
/// frame budget `budget` (or none). Returns x, then the messages of
/// s, r and g.
fn wave7_kept_read_after_another_landing<C: Cells>(
    budget: Option<u32>,
) -> (i64, String, String, String) {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_i64);
    let sel = rt.create_input(0_i64);
    let on = rt.create_input(0_i64);
    let s = rt.create_query(move |_rt: &Runtime<C>| -> i64 { panic!("s boom") });
    let r = rt.create_query(move |rt: &Runtime<C>| rt.get(s));
    let replay = || QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    };
    let dd4 = rt.create_query_with(move |_rt: &Runtime<C>| 1_i64, replay());
    let dd3 = rt.create_query_with(move |rt: &Runtime<C>| rt.get(dd4) + 1, replay());
    let dd2 = rt.create_query_with(move |rt: &Runtime<C>| rt.get(dd3) + 1, replay());
    let dd1 = rt.create_query_with(
        move |rt: &Runtime<C>| -> i64 {
            let v = rt.get(dd2);
            panic!("dd boom {v}")
        },
        replay(),
    );
    let held: Arc<Mutex<Option<Box<dyn std::any::Any + Send>>>> = Arc::new(Mutex::new(None));
    let g = rt.create_query_with(
        move |rt: &Runtime<C>| {
            if rt.get(on) == 0 {
                return 0;
            }
            let c = catch_unwind(AssertUnwindSafe(|| rt.get(r))).map_err(|p| {
                *held.lock().unwrap() = Some(p);
            });
            let v = catch_unwind(AssertUnwindSafe(|| rt.get(dd1))).unwrap_or(7);
            match c {
                Ok(n) => n + v,
                Err(()) => {
                    let p = held.lock().unwrap().take().unwrap();
                    std::panic::resume_unwind(p)
                }
            }
        },
        replay(),
    );
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(g) } else { 100 });
    assert_eq!(rt.get(x), 0);
    rt.set(on, 1);
    rt.set(sel, 1);
    rt.set_max_frames(budget.unwrap_or(u32::MAX));
    let v = rt.get(x);
    let ms = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(s))).unwrap_err());
    let mr = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(r))).unwrap_err());
    let mg = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(g))).unwrap_err());
    assert_no_notes();
    (v, ms, mr, mg)
}

/// dd1's Failed landing with its own panic, just above g, retires s's
/// deeper note: the walk contains g's panic with r and g only, at every
/// budget.
fn wave7_kept_read_follows_another_landing<C: Cells>() {
    let unbounded = wave7_kept_read_after_another_landing::<C>(None);
    assert!(
        unbounded.1.contains("last compute panicked"),
        "{unbounded:?}"
    );
    assert_eq!(unbounded.2, "s boom");
    for b in 1..=8 {
        assert_eq!(
            wave7_kept_read_after_another_landing::<C>(Some(b)),
            unbounded,
            "budget {b}"
        );
    }
}

#[test]
fn wave7_local_kept_read_follows_another_landing() {
    wave7_kept_read_follows_another_landing::<Local>();
}

#[test]
fn wave7_shared_kept_read_follows_another_landing() {
    wave7_kept_read_follows_another_landing::<Shared>();
}

/// Under x's walk, the entry w (replayable with `replay_w`) catches the
/// panic of c, which re-raises its dep r's "r boom" (r lands a frame deeper
/// than c), then catches the cycle unwind out of y, whose replayable chain
/// z1..z4 reads y back, and re-raises "r boom" while the unwind kind is
/// still Cycle. Returns x, then the messages of r and c.
fn wave8_cycle_after_caught_deep_landing<C: Cells>(
    budget: Option<u32>,
    replay_w: bool,
) -> (i64, String, String) {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_i64);
    let sel = rt.create_input(0_i64);
    let on = rt.create_input(0_i64);
    let replay = || QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    };
    let r = rt.create_query(move |_rt: &Runtime<C>| -> i64 { panic!("r boom") });
    let c = rt.create_query(move |rt: &Runtime<C>| rt.get(r));
    let y_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let z4 = {
        let y_cell = y_cell.clone();
        rt.create_query_with(
            move |rt: &Runtime<C>| rt.get(*y_cell.get().unwrap()) + 1,
            replay(),
        )
    };
    let z3 = rt.create_query_with(move |rt: &Runtime<C>| rt.get(z4) + 1, replay());
    let z2 = rt.create_query_with(move |rt: &Runtime<C>| rt.get(z3) + 1, replay());
    let z1 = rt.create_query_with(move |rt: &Runtime<C>| rt.get(z2) + 1, replay());
    let y = rt.create_query_with(move |rt: &Runtime<C>| rt.get(z1) + 1, replay());
    y_cell.set(y).unwrap();
    let w = rt.create_query_with(
        move |rt: &Runtime<C>| -> i64 {
            if rt.get(on) == 0 {
                return 0;
            }
            let caught = catch_unwind(AssertUnwindSafe(|| rt.get(c))).unwrap_err();
            let _ = catch_unwind(AssertUnwindSafe(|| rt.get(y)));
            std::panic::resume_unwind(caught)
        },
        QueryOpts {
            replay: if replay_w {
                Replay::Allowed
            } else {
                Replay::Native
            },
            ..Default::default()
        },
    );
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(w) } else { 100 });
    assert_eq!(rt.get(x), 0);
    rt.set(on, 1);
    rt.set(sel, 1);
    rt.set_max_frames(budget.unwrap_or(u32::MAX));
    let v = rt.get(x);
    let mr = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(r))).unwrap_err());
    let mc = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(c))).unwrap_err());
    (v, mr, mc)
}

/// The walk contains w's re-raised "r boom" with both landings it was noted
/// for, r's included: y's run ends in a cycle unwind, which retires no
/// note, so no frame pops above w with a value between the catch and the
/// re-raise. Every budget must leave the unbounded run's ticket, though
/// there y's doomed run pops with a value.
fn wave8_cycle_after_caught_deep_landing_matches_unbounded<C: Cells>(replay_w: bool) {
    let unbounded = wave8_cycle_after_caught_deep_landing::<C>(None, replay_w);
    assert_eq!(unbounded.1, "r boom", "{unbounded:?}");
    let differ: Vec<u32> = (1..=8)
        .filter(|&b| wave8_cycle_after_caught_deep_landing::<C>(Some(b), replay_w) != unbounded)
        .collect();
    let first = differ
        .first()
        .map(|&b| wave8_cycle_after_caught_deep_landing::<C>(Some(b), replay_w));
    assert!(
        differ.is_empty(),
        "replay_w {replay_w}: budgets {differ:?} give {first:?}, unbounded {unbounded:?}"
    );
}

/// A doomed pop above w must not retire w's live note of r's landing.
#[test]
fn wave8_local_cycle_after_caught_deep_landing_matches_unbounded() {
    wave8_cycle_after_caught_deep_landing_matches_unbounded::<Local>(false);
}

/// A doomed pop above a doomed w must not retire r's landing from the
/// kept copy of w's read of c.
#[test]
fn wave8_local_cycle_after_kept_deep_landing_matches_unbounded() {
    wave8_cycle_after_caught_deep_landing_matches_unbounded::<Local>(true);
}

/// A doomed pop above w must not retire w's live note of r's landing.
#[test]
fn wave8_shared_cycle_after_caught_deep_landing_matches_unbounded() {
    wave8_cycle_after_caught_deep_landing_matches_unbounded::<Shared>(false);
}

/// A doomed pop above a doomed w must not retire r's landing from the
/// kept copy of w's read of c.
#[test]
fn wave8_shared_cycle_after_kept_deep_landing_matches_unbounded() {
    wave8_cycle_after_caught_deep_landing_matches_unbounded::<Shared>(true);
}

/// g takes a's held ticket, sets k (flipping a Dirty) and rereads a, directly
/// or through b, so a lands Failed again with "a boom 1"; g drops that and
/// re-raises the taken "a boom 0". No ticket may hold the superseded payload:
/// get(a) reports "last compute panicked".
fn wave7_relanded_node_drops_its_taken_ticket_note<C: Cells>(via_b: bool) {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let j = rt.create_input(1_i64);
    let k = rt.create_input(0_i64);
    let a = rt.create_query(move |rt: &Runtime<C>| -> i64 {
        let v = rt.get(j);
        let kk = rt.get(k);
        if v == 0 {
            panic!("a boom {kk}")
        }
        v
    });
    let b = rt.create_query(move |rt: &Runtime<C>| rt.get(a));
    let g = rt.create_query(move |rt: &Runtime<C>| -> i64 {
        match catch_unwind(AssertUnwindSafe(|| rt.get(a))) {
            Ok(v) => v,
            Err(p) => {
                rt.set(k, 1);
                let _ = catch_unwind(AssertUnwindSafe(
                    || {
                        if via_b {
                            rt.get(b)
                        } else {
                            rt.get(a)
                        }
                    },
                ));
                std::panic::resume_unwind(p)
            }
        }
    });
    let w = rt.create_query(move |rt: &Runtime<C>| rt.get(g));
    let x = rt.create_query(move |rt: &Runtime<C>| if rt.get(sel) == 0 { rt.get(w) } else { 100 });
    assert_eq!(rt.get(x), 1);
    rt.set(j, 0);
    rt.set(sel, 1);
    assert_eq!(rt.get(x), 100);
    assert_no_notes();
    let got = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(a))).unwrap_err());
    assert!(
        got.contains("last compute panicked"),
        "via_b={via_b}: a raised {got:?}, a superseded landing's payload"
    );
    // w's walk reran g after its flagged Dirty landing, and that run
    // re-raised what its read of a raised.
    let got = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(g))).unwrap_err());
    assert!(
        got.contains("last compute panicked"),
        "via_b={via_b}: g raised {got:?}"
    );
    assert_eq!(rt.contained_payloads(), 0, "via_b={via_b}");
    assert_no_notes();
}

#[test]
fn wave7_local_relanded_node_drops_its_taken_ticket_note() {
    wave7_relanded_node_drops_its_taken_ticket_note::<Local>(false);
}

#[test]
fn wave7_shared_relanded_node_drops_its_taken_ticket_note() {
    wave7_relanded_node_drops_its_taken_ticket_note::<Shared>(false);
}

#[test]
fn wave7_local_relanded_node_via_dependent_drops_its_taken_ticket_note() {
    wave7_relanded_node_drops_its_taken_ticket_note::<Local>(true);
}

#[test]
fn wave7_shared_relanded_node_via_dependent_drops_its_taken_ticket_note() {
    wave7_relanded_node_drops_its_taken_ticket_note::<Shared>(true);
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

struct SameZst;

/// d catches x's zero-sized payload, drops it, reads an input and a query
/// child, then raises a fresh payload of the same zero-sized type. Both
/// boxes share one dangling address, type and empty text, yet x never had
/// its payload contained, so x's read must raise "last compute panicked"
/// and d's read must keep d's payload.
fn wave6_same_type_zst_note_joins_ticket_after_engine_call<C: Cells>() {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let t = rt.create_input(0_i64);
    let o = rt.create_input(0_i64);
    let x = rt.create_query(|_: &Runtime<C>| -> i64 { std::panic::panic_any(SameZst) });
    let c = rt.create_query(move |rt| rt.get(o));
    let d = rt.create_query(move |rt| {
        if rt.get(t) == 1 {
            drop(catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_err());
            let _ = rt.get(o);
            let _ = rt.get(c);
            std::panic::panic_any(SameZst);
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
    assert!(!got.is::<SameZst>(), "x's read took d's contained payload");
    assert!(
        panic_message(got).contains("last compute panicked"),
        "x's read raised a payload"
    );
    let got = catch_unwind(AssertUnwindSafe(|| rt.get(d))).unwrap_err();
    assert!(got.is::<SameZst>(), "d's read lost its payload");
}

#[test]
#[ignore = "payload identity residual (decision shared-panic-payload-contract): this fresh box is indistinguishable from the re-raised one in zst_reraised_after_engine_calls_joins_the_caught_ticket"]
fn wave6_local_same_type_zst_note_joins_ticket_after_engine_call() {
    wave6_same_type_zst_note_joins_ticket_after_engine_call::<Local>();
}

#[test]
#[ignore = "payload identity residual (decision shared-panic-payload-contract): this fresh box is indistinguishable from the re-raised one in zst_reraised_after_engine_calls_joins_the_caught_ticket"]
fn wave6_shared_same_type_zst_note_joins_ticket_after_engine_call() {
    wave6_same_type_zst_note_joins_ticket_after_engine_call::<Shared>();
}
/// The control of `wave6_same_type_zst_note_joins_ticket_after_engine_call`:
/// d re-raises the box it caught after the same engine calls, so the
/// payload failed x too and x's read takes it. The engine sees the same
/// address, type and text as for the fresh box there.
fn zst_reraised_after_engine_calls_joins_the_caught_ticket<C: Cells>() {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let t = rt.create_input(0_i64);
    let o = rt.create_input(0_i64);
    let x = rt.create_query(|_: &Runtime<C>| -> i64 { std::panic::panic_any(SameZst) });
    let c = rt.create_query(move |rt| rt.get(o));
    let d = rt.create_query(move |rt| {
        if rt.get(t) == 1 {
            let caught = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_err();
            let _ = rt.get(o);
            let _ = rt.get(c);
            std::panic::resume_unwind(caught);
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
    assert!(got.is::<SameZst>(), "x's read lost the payload d re-raised");
    let got = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(d))).unwrap_err());
    assert!(got.contains("last compute panicked"), "d raised {got:?}");
}

#[test]
fn local_zst_reraised_after_engine_calls_joins_the_caught_ticket() {
    zst_reraised_after_engine_calls_joins_the_caught_ticket::<Local>();
}

#[test]
fn shared_zst_reraised_after_engine_calls_joins_the_caught_ticket() {
    zst_reraised_after_engine_calls_joins_the_caught_ticket::<Shared>();
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

/// Silences the panics of `ticket_catching_chain`, one per level, and
/// leaves every other panic to the default hook.
fn quiet_ticket_booms() {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let payload = info.payload();
            if payload.downcast_ref::<&str>() != Some(&"ticket boom") && !payload.is::<NotClone>() {
                default(info);
            }
        }));
    });
}

/// A replayable chain of `n` levels read top-down, where level i catches
/// f_i's panic and then reads level i + 1. With `failed`, every f_i is
/// Failed with its own contained ticket, a `NotClone` payload if `custom`,
/// so every level takes a ticket payload. Returns the count of caught
/// panics. A `NotClone` f_i is replayable, so a retry past the budget
/// regenerates its payload; with a non-replayable one the readers would be
/// held past the budget and the stack would not be bounded (CHANGELOG,
/// Known limitations).
fn ticket_catching_chain<C: Cells>(n: usize, failed: bool, custom: bool) -> i64 {
    quiet_ticket_booms();
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_i64);
    let j = rt.create_input(1_i64);
    let sel = rt.create_input(0_i64);
    let fs: Vec<Incr<i64>> = (0..n)
        .map(|_| {
            rt.create_query_with(
                move |rt: &Runtime<C>| {
                    if rt.get(j) == 0 && custom {
                        std::panic::panic_any(NotClone(1));
                    }
                    assert!(rt.get(j) != 0, "ticket boom");
                    0
                },
                QueryOpts {
                    replay: if custom {
                        Replay::Allowed
                    } else {
                        Replay::Native
                    },
                    ..Default::default()
                },
            )
        })
        .collect();
    let all = fs.clone();
    let x = rt.create_query(move |rt: &Runtime<C>| {
        if rt.get(sel) == 0 {
            all.iter().map(|&f| rt.get(f)).sum()
        } else {
            0
        }
    });
    rt.get(x);
    if failed {
        rt.set(j, 0);
        rt.set(sel, 1);
        assert_eq!(rt.get(x), 0);
        assert_eq!(rt.contained_payloads(), n);
    }
    let mut below: Option<Incr<i64>> = None;
    for &f in fs.iter().rev() {
        let next = below;
        below = Some(rt.create_query_with(
            move |rt: &Runtime<C>| {
                let caught = catch_unwind(AssertUnwindSafe(|| rt.get(f))).is_err();
                i64::from(caught) + next.map_or(0, |b| rt.get(b))
            },
            QueryOpts {
                replay: Replay::Allowed,
                ..Default::default()
            },
        ));
    }
    rt.get(below.unwrap())
}

/// Runs the chain on a thread with the default 2 MiB spawn stack, which
/// the default frame budget must bound in a debug build too.
fn ticket_catching_chain_on_a_thread<C: Cells>(failed: bool, custom: bool) {
    // Frames that are never doomed overflow this stack at 1000 to 2000
    // levels.
    let n = test_size(100_000);
    let got = thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || ticket_catching_chain::<C>(n, failed, custom))
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(got, if failed { n as i64 } else { 0 });
}

/// The chain at 2000 to 64000 with a ticket per caught read, every read's
/// ticket taken while the frames below hold theirs: the tickets' work
/// stays linear in the reads. A retry's read of a regenerable payload keeps
/// a copy of its notes, about 20 units a read.
fn ticket_catching_chain_work<C: Cells>() {
    for n in [2000, 4000, 16000, test_size(64_000)] {
        let work = thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(move || {
                crate::replay::take_note_work();
                assert_eq!(ticket_catching_chain::<C>(n, true, true), n as i64);
                crate::replay::take_note_work()
            })
            .unwrap()
            .join()
            .unwrap();
        assert!(
            work <= 24 * n as u64,
            "{work} units of note work for {n} reads"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_ticket_catching_chain_work_stays_linear() {
    ticket_catching_chain_work::<Local>();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_ticket_catching_chain_work_stays_linear() {
    ticket_catching_chain_work::<Shared>();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_catching_chain_stays_within_the_stack() {
    ticket_catching_chain_on_a_thread::<Local>(false, false);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_catching_chain_stays_within_the_stack() {
    ticket_catching_chain_on_a_thread::<Shared>(false, false);
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_ticket_catching_chain_stays_within_the_stack() {
    ticket_catching_chain_on_a_thread::<Local>(true, false);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_ticket_catching_chain_stays_within_the_stack() {
    ticket_catching_chain_on_a_thread::<Shared>(true, false);
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_uncopyable_ticket_catching_chain_stays_within_the_stack() {
    ticket_catching_chain_on_a_thread::<Local>(true, true);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_uncopyable_ticket_catching_chain_stays_within_the_stack() {
    ticket_catching_chain_on_a_thread::<Shared>(true, true);
}

/// Fixpoint `a` reads unseeded `b` with `try_get` and panics on its `Err`;
/// once `k` is 1, `b` reads `a`. A fresh `get(r)` enters the cycle at `a`,
/// so `b`'s read of `a` gets `a`'s initial value and the episode converges.
/// The walk computes `b` as a walk entry first instead, which puts `b`
/// below `a`, so `a`'s read of `b` is R3's `Err` and `a` panics. A cycle
/// whose member branches on a cycle error depends on which member computes
/// first, and an incremental read enters it where its walk does (CHANGELOG,
/// Known limitations).
fn walk_cycle_entry<C: Cells>(k0: i64) -> (Runtime<C>, Incr<i64>, Incr<i64>) {
    let rt = Runtime::<C>::new();
    let k = rt.create_input(k0);
    let a_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let b = {
        let a_cell = a_cell.clone();
        rt.create_query(move |rt| {
            if rt.get(k) == 0 {
                0
            } else {
                rt.get(*a_cell.get().unwrap()) + 1
            }
        })
    };
    let a = rt.create_query_with(
        move |rt| match rt.try_get(b) {
            Ok(v) => (v + 1).min(5),
            Err(_) => panic!("boom"),
        },
        QueryOpts {
            cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64).max_iterations(20)),
            ..QueryOpts::default()
        },
    );
    a_cell.set(a).unwrap();
    let r = rt.create_query(move |rt| rt.get(a) * 10);
    (rt, k, r)
}

fn walk_enters_a_cycle_where_a_fresh_read_does<C: Cells>() {
    let (fresh, _, fresh_r) = walk_cycle_entry::<C>(1);
    let expected = fresh.get(fresh_r);
    assert_eq!(expected, 50);

    let (rt, k, r) = walk_cycle_entry::<C>(0);
    assert_eq!(rt.get(r), 10);
    rt.set(k, 1);
    assert_eq!(rt.try_get(r), Ok(expected));
}

#[test]
#[ignore = "a walk enters a cycle where it recomputes first, not where a fresh read does"]
fn local_walk_enters_a_cycle_where_a_fresh_read_does() {
    walk_enters_a_cycle_where_a_fresh_read_does::<Local>();
}

#[test]
#[ignore = "a walk enters a cycle where it recomputes first, not where a fresh read does"]
fn shared_walk_enters_a_cycle_where_a_fresh_read_does() {
    walk_enters_a_cycle_where_a_fresh_read_does::<Shared>();
}

fn walk_enters_a_cycle_at_its_first_recomputed_member<C: Cells>() {
    let (rt, k, r) = walk_cycle_entry::<C>(0);
    assert_eq!(rt.get(r), 10);
    rt.set(k, 1);
    let read = catch_unwind(AssertUnwindSafe(|| rt.try_get(r)));
    assert_eq!(
        read.map_err(panic_message),
        Err("boom".to_string()),
        "the walk's cycle entry changed; check the ignored fresh-order test"
    );
}

#[test]
fn local_walk_enters_a_cycle_at_its_first_recomputed_member() {
    walk_enters_a_cycle_at_its_first_recomputed_member::<Local>();
}

#[test]
fn shared_walk_enters_a_cycle_at_its_first_recomputed_member() {
    walk_enters_a_cycle_at_its_first_recomputed_member::<Shared>();
}

/// `a`, a fixpoint from 0 capped at 5, reads `b` once `k` is 1; `b` reads
/// `a`, and on an `a` of 2 or more, which only a member run reaches, reads
/// `c`, which reads `d`, which reads `b` back: R3 in the member run of `b`,
/// a cycle over the held member `b`, `c` and `d` that aborts the episode.
/// `x` reads `a` only while `sel` is 0. After `k` and `sel` go to 1, the
/// walk of `x` recomputes its stale dep `a` and contains the abort, so the
/// rerun, which no longer reads `a`, gives 100 as a fresh read of `x` does.
fn stale_dep_whose_episode_cycles<C: Cells>() -> Result<i64, String> {
    let rt = Runtime::<C>::new();
    let sel = rt.create_input(0_i64);
    let k = rt.create_input(0_i64);
    let b_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let b_cell = b_cell.clone();
        rt.create_query_with(
            move |rt| {
                if rt.get(k) == 0 {
                    1
                } else {
                    rt.get(*b_cell.get().unwrap()).min(5)
                }
            },
            QueryOpts {
                cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64)),
                ..Default::default()
            },
        )
    };
    let c_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let b = {
        let c_cell = c_cell.clone();
        rt.create_query(move |rt| {
            let v = rt.get(a);
            if v >= 2 {
                rt.get(*c_cell.get().unwrap())
            } else {
                v + 1
            }
        })
    };
    let d_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let c = {
        let d_cell = d_cell.clone();
        rt.create_query(move |rt| rt.get(*d_cell.get().unwrap()) + 1)
    };
    let d = rt.create_query(move |rt| rt.get(b) + 1);
    b_cell.set(b).unwrap();
    c_cell.set(c).unwrap();
    d_cell.set(d).unwrap();
    let x = rt.create_query(move |rt| {
        let s = rt.get(sel);
        if s == 0 {
            s + rt.get(a)
        } else {
            100
        }
    });
    assert_eq!(rt.get(x), 1);
    rt.set(k, 1);
    rt.set(sel, 1);
    let got = catch_unwind(AssertUnwindSafe(|| rt.get(x))).map_err(panic_message);
    assert_no_notes();
    got
}

#[test]
fn local_stale_dep_whose_episode_cycles_is_not_read() {
    assert_eq!(stale_dep_whose_episode_cycles::<Local>(), Ok(100));
}

#[test]
fn shared_stale_dep_whose_episode_cycles_is_not_read() {
    assert_eq!(stale_dep_whose_episode_cycles::<Shared>(), Ok(100));
}

#[test]
fn a_pruned_plan_forgets_the_visits_and_expansions_it_dropped() {
    let (r, q, p, x) = (NodeId(1), NodeId(2), NodeId(3), NodeId(4));
    let mut walk = Walk::new(r, 0, NodeState::Dirty);
    walk.visited = Some(Default::default());
    // R's deps Q and P, P's dep X, a New node with installed deps.
    walk.expand(q, 0, NodeState::Dirty);
    walk.visit(p, 0, false);
    walk.visit(x, 0, true);
    assert!(walk.first_new_expansion(x));
    walk.prune(3);
    assert_eq!(walk.work.len(), 3);
    assert!(walk.is_visited(p));
    assert!(!walk.is_visited(x));
    assert!(walk.first_new_expansion(x));
}

#[test]
fn an_untainted_entry_merged_into_its_parent_keeps_the_new_mark() {
    let (r, p, x) = (NodeId(1), NodeId(2), NodeId(3));
    let mut walk = Walk::new(r, 0, NodeState::Dirty);
    walk.work.clear();
    walk.visit(p, 0, true);
    walk.visit(p, 0, false);
    walk.drop_repeats(walk.work.len());
    assert_eq!(walk.work.len(), 1);
    assert_eq!(walk.work[0].taint, NO_TAINT_NEW);
    walk.visit(x, 0, false);
    walk.unsettled(x);
    assert_eq!(walk.work.len(), 1);
    assert_eq!(walk.work[0].taint, NO_TAINT_NEW);
}
