//! The `ensure_clean` walk: its cost on wide DAGs and its recovery after a
//! cycle panic, each case run under `Local` and `Shared`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::Duration;

use super::race_tests::{id, park_at, Gate};
use super::{ParkPoint, Runtime, EXACT_MODE_AT};
use crate::cells::{Cells, Local, Shared};
use crate::handle::Incr;
use crate::test_size::test_size;

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

/// Thread A computes `a`, which now reads `q`; thread B computes `q`,
/// whose closure reads `r`, and the walk for `r` reaches `a` through
/// `x`'s stale edge, so B waits on A's claim while A waits on B's. The
/// walk cannot tell this stale edge from a real one across threads, so
/// the two threads hang; the test records that.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_stale_edge_closes_a_waits_for_cycle_across_threads() {
    crate::child_test::run_in_child(
        "runtime::walk_tests::shared_stale_edge_closes_a_waits_for_cycle_across_threads",
        || {
            let rt = Arc::new(Runtime::<Shared>::new());
            let s = rt.create_input(0_i64);
            let t = rt.create_input(0_i64);
            let u = rt.create_input(0_i64);
            let i = rt.create_input(1_i64);
            let k = rt.create_input(2_i64);
            let v = rt.create_input(3_i64);
            let q_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
            let a_gate = Gate::new();
            let a = {
                let q_cell = q_cell.clone();
                let a_gate = a_gate.clone();
                rt.create_query(move |rt| {
                    if rt.get(t) == 0 {
                        rt.get(i)
                    } else {
                        a_gate.park();
                        rt.get(*q_cell.get().unwrap())
                    }
                })
            };
            let x = rt.create_query(move |rt| if rt.get(s) == 0 { rt.get(a) } else { rt.get(k) });
            let b = rt.create_query(move |rt| rt.get(x) + 1);
            let r = rt.create_query(move |rt| rt.get(b) + 1);
            let q = rt.create_query(move |rt| if rt.get(u) == 0 { rt.get(v) } else { rt.get(r) });
            q_cell.set(q).unwrap();
            assert_eq!(rt.get(q), 3);
            assert_eq!(rt.get(r), 3);

            rt.set(s, 1);
            rt.set(t, 1);
            rt.set(u, 1);
            a_gate.arm();
            let thread_a = {
                let rt = Arc::clone(&rt);
                thread::spawn(move || rt.get(a))
            };
            a_gate.wait_parked();

            let wait = Gate::new();
            park_at(
                &rt,
                vec![(ParkPoint::BeforeComputeOne(id(a)), wait.clone())],
            );
            let thread_b = {
                let rt = Arc::clone(&rt);
                thread::spawn(move || rt.get(q))
            };
            wait.wait_parked();
            wait.release();
            a_gate.release();
            thread_a.join().unwrap();
            thread_b.join().unwrap();
        },
    )
    .assert_killed();
}
