//! The `ensure_clean` walk: its cost on wide DAGs and its recovery after a
//! cycle panic, each case run under `Local` and `Shared`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, OnceLock};
use std::thread;

use super::{Runtime, EXACT_MODE_AT};
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
/// 0, its successor. The first read nests n levels before the frame-stack
/// check panics, and every member lands Failed with its ring edge. Setting
/// `g` breaks the cycle and flips the ring Dirty, so the next walk
/// re-expands the ring until the exact-mode switch, which bounds the rest.
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
        visits <= u64::from(EXACT_MODE_AT) + edges,
        "recovery walk made {visits} first visits for {edges} edges"
    );
    assert!(
        visits >= u64::from(EXACT_MODE_AT),
        "the ring must re-expand until the switch, got {visits} first visits"
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
