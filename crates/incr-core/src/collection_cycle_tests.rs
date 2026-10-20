//! T8's operator rules: a cursor-driven operator that runs as a cycle
//! member, or that read a provisional value, lands the output a fresh
//! operator built after the episode would have, and a `group_by` router
//! refuses a provisional value with the cycle and routes nothing.

use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
use std::sync::{Arc, OnceLock};

use super::*;
use crate::cells::{Local, Shared};
use crate::cycle::{CycleStrategy, Fixpoint, QueryOpts};

/// A runtime a user function captures to read nodes, which the operator
/// API's `Fn(&T)` signatures otherwise can't reach.
struct RtRef<C: Cells>(*const Runtime<C>);

// SAFETY: each test keeps its runtime boxed and alive past every operator
// run, and a `Local` runtime's operators only run on the test's thread.
unsafe impl<C: Cells> Send for RtRef<C> {}
// SAFETY: as above.
unsafe impl<C: Cells> Sync for RtRef<C> {}

impl<C: Cells> RtRef<C> {
    fn rt(&self) -> &Runtime<C> {
        // SAFETY: as above.
        unsafe { &*self.0 }
    }
}

fn fixpoint(limit: u32) -> QueryOpts<u64> {
    QueryOpts {
        cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_u64).max_iterations(limit)),
        ..QueryOpts::default()
    }
}

fn sorted(mut v: Vec<u64>) -> Vec<u64> {
    v.sort_unstable();
    v
}

/// M = min(count(F) + 1, 5) and F = S.filter(x <= M), over S = 0..10, so
/// M climbs 0, 2, 4, 5 and F lands {0..=5}.
fn filter_member<C: Cells>() {
    let rt = Box::new(Runtime::<C>::new());
    let s = rt.create_collection::<u64>();
    let m_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let (r, m) = (RtRef(&*rt), m_cell.clone());
    let f = s.filter(&rt, move |&x| x <= r.rt().get(*m.get().unwrap()));
    let count = f.count(&rt);
    let m = rt.create_query_with(move |rt| (rt.get(count) + 1).min(5), fixpoint(20));
    m_cell.set(m).unwrap();
    for x in 0..10 {
        s.insert(&rt, x);
    }

    assert_eq!(rt.get(m), 5);
    let landed = rt.get(m);
    let fresh = s.filter(&rt, move |&x| x <= landed);
    assert_eq!(sorted(f.values(&rt)), sorted(fresh.values(&rt)));
    assert_eq!(rt.get(count), 6);
    rt.assert_invariant_i();
}

/// As [`filter_member`], with an aggregate counting the members at or
/// below M in place of the filter and its count.
fn aggregate_member<C: Cells>() {
    let rt = Box::new(Runtime::<C>::new());
    let s = rt.create_collection::<u64>();
    let m_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let (r, m) = (RtRef(&*rt), m_cell.clone());
    let below = s.aggregate(
        &rt,
        0_u64,
        move |&x| u64::from(x <= r.rt().get(*m.get().unwrap())),
        |a, b| a + b,
    );
    let m = rt.create_query_with(move |rt| (rt.get(below) + 1).min(5), fixpoint(20));
    m_cell.set(m).unwrap();
    for x in 0..10 {
        s.insert(&rt, x);
    }

    assert_eq!(rt.get(m), 5);
    let landed = rt.get(m);
    let fresh = s.aggregate(&rt, 0_u64, move |&x| u64::from(x <= landed), |a, b| a + b);
    assert_eq!(rt.get(below), rt.get(fresh));
    assert_eq!(rt.get(below), 6);
    rt.assert_invariant_i();
}

/// A router whose `key_fn` reads Fixpoint member M while M reads the
/// router: the router's run gets `Err(Cycle)` at M's `try_get` and creates
/// no group. Once a `set` unlinks M from `key_fn`, it routes normally.
fn group_by_refuses<C: Cells>() {
    let rt = Box::new(Runtime::<C>::new());
    let s = rt.create_collection::<u64>();
    let link = rt.create_input(1_u64);
    let m_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let (r, m) = (RtRef(&*rt), m_cell.clone());
    let grouped = s.group_by(&rt, move |&x| {
        let rt = r.rt();
        if rt.get(link) == 1 {
            (x + rt.get(*m.get().unwrap())) % 2
        } else {
            x % 2
        }
    });
    let router = grouped.version_node();
    let refused = Arc::new(AtomicBool::new(false));
    let seen = refused.clone();
    let m = rt.create_query_with(
        move |rt| match rt.try_get(router) {
            Ok(v) => v,
            Err(_) => {
                seen.store(true, SeqCst);
                7
            }
        },
        fixpoint(20),
    );
    m_cell.set(m).unwrap();
    for x in 1..=3 {
        s.insert(&rt, x);
    }

    let _ = rt.get(m);
    assert!(refused.load(SeqCst), "the router read a provisional M");
    assert_eq!(grouped.group_count(), 0, "a refused run routes nothing");

    // The refusal left the router marked for a rebuild, whose key_fn pass
    // over the snapshot meets M again. The read's walk runs the router as
    // M's dep, before M's closure, so the cycle reaches this `try_get`.
    s.insert(&rt, 4);
    assert!(rt.try_get(m).is_err(), "the rebuild read a provisional M");
    assert_eq!(grouped.group_count(), 0);

    rt.set(link, 0);
    let _ = rt.get(m);
    let mut keys = grouped.keys();
    keys.sort_unstable();
    assert_eq!(keys, vec![0, 1]);
    assert_eq!(
        sorted(grouped.get_group(&1).unwrap().values(&rt)),
        vec![1, 3]
    );
    rt.assert_invariant_i();
}

/// The provisional value reaches the router through Q, a query that is no
/// member: Q reads M on the stack, defers, and hands its frame `LOW`.
fn group_by_refuses_through_a_deferral<C: Cells>() {
    let rt = Box::new(Runtime::<C>::new());
    let s = rt.create_collection::<u64>();
    let m_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let m = m_cell.clone();
    let q = rt.create_query(move |rt| rt.get(*m.get().unwrap()) % 2);
    let r = RtRef(&*rt);
    let grouped = s.group_by(&rt, move |&x| (x + r.rt().get(q)) % 2);
    let router = grouped.version_node();
    let m = rt.create_query_with(move |rt| rt.try_get(router).map_or(7, |v| v), fixpoint(20));
    m_cell.set(m).unwrap();
    for x in 1..=3 {
        s.insert(&rt, x);
    }
    assert_eq!(rt.get(m), 7, "the router refused");
    assert_eq!(grouped.group_count(), 0);
    rt.assert_invariant_i();
}

/// The router's upstream is a filter whose `pred` reads M, and the episode
/// is entered at M, so the filter is a member off the stack when the
/// router reads its version.
fn group_by_refuses_a_member_upstream<C: Cells>() {
    let rt = Box::new(Runtime::<C>::new());
    let s = rt.create_collection::<u64>();
    let m_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let (r, m) = (RtRef(&*rt), m_cell.clone());
    let f = s.filter(&rt, move |&x| x <= r.rt().get(*m.get().unwrap()));
    let grouped = f.group_by(&rt, |&x| x % 2);
    let router = grouped.version_node();
    let m = rt.create_query_with(move |rt| rt.try_get(router).map_or(7, |v| v), fixpoint(20));
    m_cell.set(m).unwrap();
    for x in 1..=3 {
        s.insert(&rt, x);
    }
    assert_eq!(rt.get(m), 7, "the router refused");
    assert_eq!(grouped.group_count(), 0);
    rt.assert_invariant_i();
}

#[test]
fn local_group_by_refuses_through_a_deferral() {
    group_by_refuses_through_a_deferral::<Local>();
}

#[test]
fn shared_group_by_refuses_through_a_deferral() {
    group_by_refuses_through_a_deferral::<Shared>();
}

#[test]
fn local_group_by_refuses_a_member_upstream() {
    group_by_refuses_a_member_upstream::<Local>();
}

#[test]
fn shared_group_by_refuses_a_member_upstream() {
    group_by_refuses_a_member_upstream::<Shared>();
}

#[test]
fn local_filter_member_lands_a_fresh_filters_output() {
    filter_member::<Local>();
}

#[test]
fn shared_filter_member_lands_a_fresh_filters_output() {
    filter_member::<Shared>();
}

#[test]
fn local_aggregate_member_lands_a_fresh_aggregates_output() {
    aggregate_member::<Local>();
}

#[test]
fn shared_aggregate_member_lands_a_fresh_aggregates_output() {
    aggregate_member::<Shared>();
}

#[test]
fn local_group_by_refuses_a_provisional_key() {
    group_by_refuses::<Local>();
}

#[test]
fn shared_group_by_refuses_a_provisional_key() {
    group_by_refuses::<Shared>();
}
