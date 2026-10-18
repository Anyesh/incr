//! `delete_with_dependents`: each case is a generic body run once under
//! `Local` and once under `Shared`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::handle::Incr;
use crate::locks::Lock;

fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
    match err.downcast::<String>() {
        Ok(s) => *s,
        Err(err) => err.downcast::<&str>().map(|s| s.to_string()).unwrap(),
    }
}

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

fn chain<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(1_i64);
    let a = rt.create_query(move |rt| rt.get(x) + 1);
    let b = rt.create_query(move |rt| rt.get(a) * 10);
    assert_eq!(rt.get(b), 20);

    assert_eq!(rt.delete_with_dependents(x), 3);
    assert!([x, a, b].iter().all(|&h| !rt.is_alive(h)));
    assert_eq!(rt.node_count(), 0);
    rt.assert_invariant_i();
}
local_and_shared!(chain, local_cascade_chain, shared_cascade_chain);

fn diamond<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(1_i64);
    let l = rt.create_query(move |rt| rt.get(x) + 1);
    let r = rt.create_query(move |rt| rt.get(x) + 2);
    let j = rt.create_query(move |rt| rt.get(l) + rt.get(r));
    assert_eq!(rt.get(j), 5);

    assert_eq!(rt.delete_with_dependents(x), 4, "j counted once");
    assert_eq!(rt.node_count(), 0);
    rt.assert_invariant_i();
}
local_and_shared!(diamond, local_cascade_diamond, shared_cascade_diamond);

fn mixed_types_recycle<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(4_i64);
    let s = rt.create_query(move |rt| "s".repeat(rt.get(x) as usize));
    let v = rt.create_query(move |rt| rt.get(s).into_bytes());
    let f = rt.create_query(move |rt| rt.get(v).len() as f64);
    assert_eq!(rt.get(f), 4.0);
    let arena_lens = || {
        (
            rt.arena::<i64>().len(),
            rt.arena::<String>().len(),
            rt.arena::<Vec<u8>>().len(),
            rt.arena::<f64>().len(),
        )
    };
    let before = arena_lens();

    assert_eq!(rt.delete_with_dependents(x), 4);
    assert_eq!(rt.node_count(), 0);

    let f2 = rt.create_input(1.5_f64);
    let v2 = rt.create_input(vec![9_u8]);
    let s2 = rt.create_input("again".to_string());
    let x2 = rt.create_input(-1_i64);
    assert_eq!(arena_lens(), before, "every arena reused its released slot");
    assert_eq!(rt.get(f2), 1.5);
    assert_eq!(rt.get(v2), vec![9]);
    assert_eq!(rt.get(s2), "again");
    assert_eq!(rt.get(x2), -1);
    rt.assert_invariant_i();
}
local_and_shared!(
    mixed_types_recycle,
    local_cascade_mixed_types_recycle_slots,
    shared_cascade_mixed_types_recycle_slots
);

fn drops_observers<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(1_i64);
    let q = rt.create_query(move |rt| rt.get(x) * 2);
    let fired = Arc::new(AtomicU32::new(0));
    for h in [x, q] {
        let fired = Arc::clone(&fired);
        rt.observe(h, move |_| {
            fired.fetch_add(1, Ordering::SeqCst);
        });
    }
    rt.stabilize();
    let baseline = fired.load(Ordering::SeqCst);

    assert_eq!(rt.delete_with_dependents(x), 2);
    assert!(rt.inner.read().observers.is_empty());
    rt.stabilize();
    assert_eq!(fired.load(Ordering::SeqCst), baseline);
}
local_and_shared!(
    drops_observers,
    local_cascade_drops_observers,
    shared_cascade_drops_observers
);

fn stale_handles<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(1_i64);
    let q = rt.create_query(move |rt| rt.get(x) + 1);
    assert_eq!(rt.get(q), 2);
    assert_eq!(rt.delete_with_dependents(x), 2);

    assert_eq!(rt.delete_with_dependents(x), 0, "a stale root is done");
    let recycled = rt.create_input(7_i64);
    assert_eq!(recycled.slot(), q.slot().min(x.slot()));
    assert_eq!(rt.delete_with_dependents(q), 0);
    assert_eq!(rt.delete_with_dependents(x), 0);
    assert!(
        rt.is_alive(recycled),
        "a stale root reached the recycled slot"
    );

    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(q))).unwrap_err());
    assert!(msg.contains("stale"), "got: {msg}");
}
local_and_shared!(
    stale_handles,
    local_cascade_stale_handles,
    shared_cascade_stale_handles
);

fn outside_inputs_keep_edges<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(1_i64);
    let y = rt.create_input(10_i64);
    let both = rt.create_query(move |rt| rt.get(x) + rt.get(y));
    let only_y = rt.create_query(move |rt| rt.get(y) * 2);
    assert_eq!(rt.get(both), 11);
    assert_eq!(rt.get(only_y), 20);

    assert_eq!(rt.delete_with_dependents(x), 2);
    assert!(rt.is_alive(y) && rt.is_alive(only_y));
    assert!(rt.has_reverse_edge(y, only_y));
    assert_eq!(rt.inner.read().dependents[y.slot() as usize].len(), 1);
    rt.set(y, 21);
    assert_eq!(rt.get(only_y), 42);
    rt.assert_invariant_i();
}
local_and_shared!(
    outside_inputs_keep_edges,
    local_cascade_leaves_outside_inputs_their_edges,
    shared_cascade_leaves_outside_inputs_their_edges
);

fn group_by_pipeline<C: Cells>(rt: &Runtime<C>) {
    let survivor = rt.create_input(0_i64);
    let c = rt.create_collection::<i64>();
    c.insert(rt, 3);
    c.insert(rt, 4);
    let groups = c.group_by(rt, |x| x % 3);
    let _ = rt.get(groups.version_node());
    c.insert(rt, 5);
    let _ = rt.get(groups.version_node());
    assert_eq!(groups.group_count(), 3);
    let counts: Vec<Incr<u64>> = (0..3)
        .map(|k| groups.get_group(&k).unwrap().count(rt))
        .collect();
    assert!(counts.iter().all(|&n| rt.get(n) == 1));
    let before = rt.node_count();

    let removed = rt.delete_with_dependents(c.version_node());
    assert_eq!(removed, 1 + 1 + 3 + 3, "source, router, groups, counts");
    assert_eq!(rt.node_count(), before - removed);
    assert!(rt.is_alive(survivor));
    assert!(!rt.is_alive(groups.version_node()));
    assert!(counts.iter().all(|&n| !rt.is_alive(n)));
    rt.assert_invariant_i();
}
local_and_shared!(
    group_by_pipeline,
    local_cascade_group_by_pipeline_from_source,
    shared_cascade_group_by_pipeline_from_source
);

fn owner_flag_follows_its_children<C: Cells>(rt: &Runtime<C>) {
    let owner = rt.create_input(0_u8);
    let a = rt.create_input(1_u8);
    let b = rt.create_input(2_u8);
    rt.adopt(owner.erase(), a.erase());
    rt.adopt(owner.erase(), b.erase());
    rt.assert_invariant_i();
    rt.delete_node(a);
    rt.assert_invariant_i();
    rt.delete_node(b);
    rt.assert_invariant_i();
    rt.delete_node(owner);
    assert!(!rt.is_alive(owner));
    let recycled = rt.create_input(3_u8);
    assert_eq!(recycled.slot(), owner.slot());
    rt.assert_invariant_i();
}
local_and_shared!(
    owner_flag_follows_its_children,
    local_owner_flag_follows_its_children,
    shared_owner_flag_follows_its_children
);

fn never_run_query_survives<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(1_i64);
    let q = rt.create_query(move |rt| rt.get(x) + 1);

    assert_eq!(rt.delete_with_dependents(x), 1);
    assert!(rt.is_alive(q), "a never-run query has no edge to follow");
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(q))).unwrap_err());
    assert!(msg.contains("stale"), "got: {msg}");
}
local_and_shared!(
    never_run_query_survives,
    local_cascade_misses_a_never_run_query,
    shared_cascade_misses_a_never_run_query
);

fn from_inside_compute<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(1_i64);
    let q = rt.create_query(move |rt| rt.delete_with_dependents(x) as i64);
    let _ = rt.get(q);
}

#[test]
#[should_panic(expected = "during compute")]
fn local_cascade_from_inside_compute_panics() {
    from_inside_compute(&Runtime::<Local>::new());
}

#[test]
#[should_panic(expected = "during compute")]
fn shared_cascade_from_inside_compute_panics() {
    from_inside_compute(&Runtime::<Shared>::new());
}

#[test]
#[should_panic(expected = "foreign runtime")]
fn cascade_with_foreign_handle_panics() {
    let rt1: Runtime<Shared> = Runtime::new();
    let rt2: Runtime<Shared> = Runtime::new();
    let a = rt1.create_input(1_i64);
    let _ = rt2.create_input(1_i64);
    rt2.delete_with_dependents(a);
}
