//! Eager group routing: a write that reaches a `group_by` router routes it
//! before the outermost call returns, so groups, their totals and their
//! watches are current without a read of the router.

use std::any::Any;
use std::cell::RefCell;
use std::hash::Hasher;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use super::*;
use crate::cells::{Local, Shared};
use crate::dep_stack::DepStack;
use crate::runtime::{owed_to, queued_routers, RoutingFailed};

const SETTLE: Duration = Duration::from_millis(150);

macro_rules! both {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Local>();
                }
            )*
        }
        mod shared {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Shared>();
                }
            )*
        }
    };
}

both!(
    an_insert_routes_a_fresh_group_by,
    a_watched_group_total_is_reported_and_an_untouched_one_is_not,
    a_router_is_reported_only_while_the_user_watches_it,
    a_nested_group_by_over_a_group_is_routed,
    a_router_dirtied_inside_a_compute_is_routed_before_the_read_returns,
    a_write_in_an_observer_callback_is_routed_before_stabilize_returns,
    later_observers_read_groups_a_callback_moved,
    a_second_runtime_used_inside_a_compute_routes_at_its_own_exit,
    a_routing_hold_routes_once_when_it_drops,
    a_suppressed_hold_leaves_routing_to_route_pending,
    a_key_fn_that_reads_the_source_routes_after_a_hold,
    a_pending_list_stays_bounded_under_writes_with_no_read,
    a_failing_key_fn_is_stored_and_fails_again_while_its_row_remains,
    a_router_recovers_once_the_row_its_key_fn_refused_is_gone,
    a_stored_failure_is_not_taken_inside_a_compute,
    a_runtime_owes_its_queued_routers_and_stored_failures_until_cleared,
    routing_runs_after_a_mutator_disposes,
    drain_dirtied_watched_reports_a_total_routing_moved,
    a_drain_inside_another_runtimes_compute_leaves_its_frame_untainted,
    a_panic_unwinding_through_a_write_does_not_route,
    a_deleted_router_on_the_pending_list_is_skipped,
    apply_and_extend_route_once,
);

thread_local! {
    static STASH: RefCell<Vec<Rc<dyn Any>>> = const { RefCell::new(Vec::new()) };
}

/// Keeps `value` for closures on this thread that cannot capture it (a
/// `Local` collection is not `Send`).
fn stash<T: 'static>(value: T) -> usize {
    STASH.with(|s| {
        let mut s = s.borrow_mut();
        s.push(Rc::new(value));
        s.len() - 1
    })
}

fn stashed<T: 'static>(at: usize) -> Rc<T> {
    let any = STASH.with(|s| Rc::clone(&s.borrow()[at]));
    any.downcast::<T>()
        .unwrap_or_else(|_| panic!("stash entry {at} has another type"))
}

fn count_of<K, C>(rt: &Runtime<C>, g: &GroupedCollection<K, i64, C>, key: K) -> Option<u64>
where
    K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
    C: Cells,
{
    g.get_group(&key)
        .map(|group| group.snapshot_len() as u64)
        .inspect(|&n| {
            assert_eq!(
                group_values(rt, g, key.clone()).len() as u64,
                n,
                "a group's values disagree with its length"
            );
        })
}

fn group_values<K, C>(rt: &Runtime<C>, g: &GroupedCollection<K, i64, C>, key: K) -> Vec<i64>
where
    K: Clone + PartialEq + Eq + Hash + Send + Sync + 'static,
    C: Cells,
{
    let mut out = g
        .get_group(&key)
        .map_or_else(Vec::new, |group| group.values(rt));
    out.sort_unstable();
    out
}

fn message(failed: &RoutingFailed) -> String {
    failed.message().unwrap_or_default().to_string()
}

fn an_insert_routes_a_fresh_group_by<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    c.insert(&rt, 1);
    let g = c.group_by(&rt, |x| x % 2);
    let odd = g.get_group(&1).expect("a bootstrap group").count(&rt);
    assert_eq!(rt.get(odd), 1);

    c.insert(&rt, 3);
    assert_eq!(rt.get(odd), 2, "a group's count after an insert");
    c.insert(&rt, 4);
    let even = g.get_group(&0).expect("a new key's group after an insert");
    assert_eq!(rt.get(even.count(&rt)), 1);
    assert_eq!(group_values(&rt, &g, 1), vec![1, 3]);
}

fn a_watched_group_total_is_reported_and_an_untouched_one_is_not<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    c.extend(&rt, [1, 2]);
    let g = c.group_by(&rt, |x| x % 2);
    let odd = g.get_group(&1).unwrap().count(&rt);
    let even = g.get_group(&0).unwrap().count(&rt);
    assert_eq!((rt.get(odd), rt.get(even)), (1, 1));
    rt.watch(odd);
    rt.watch(even);
    assert!(rt.drain_dirtied_watched().is_empty());

    c.insert(&rt, 5);
    assert_eq!(rt.drain_dirtied_watched(), vec![odd.erase()]);
    assert_eq!(rt.get(odd), 2);
}

fn a_router_is_reported_only_while_the_user_watches_it<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    let g = c.group_by(&rt, |x| x % 2);
    let router = g.version_node();

    c.insert(&rt, 1);
    assert!(
        rt.drain_dirtied_watched().is_empty(),
        "an unwatched router was reported"
    );

    rt.watch(router);
    let held = rt.hold_routing_suppressed();
    c.insert(&rt, 3);
    drop(held);
    assert_eq!(rt.drain_dirtied_watched(), vec![router.erase()]);

    rt.unwatch(router);
    c.insert(&rt, 5);
    assert!(rt.drain_dirtied_watched().is_empty());
    assert_eq!(
        count_of(&rt, &g, 1),
        Some(3),
        "a router stopped routing once unwatched"
    );
}

fn a_nested_group_by_over_a_group_is_routed<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    c.insert(&rt, 1);
    let g = c.group_by(&rt, |x| x % 2);
    let odd = g.get_group(&1).unwrap();
    let nested = odd.group_by(&rt, |x| x % 3);

    c.insert(&rt, 3);
    c.insert(&rt, 7);
    assert_eq!(count_of(&rt, &nested, 0), Some(1));
    assert_eq!(count_of(&rt, &nested, 1), Some(2));
}

fn a_router_dirtied_inside_a_compute_is_routed_before_the_read_returns<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    c.insert(&rt, 1);
    let g = c.group_by(&rt, |x| x % 2);
    let odd = g.get_group(&1).unwrap().count(&rt);
    assert_eq!(rt.get(odd), 1);
    let at = stash(c);
    let writer = rt.create_query(move |rt| {
        stashed::<IncrCollection<i64, C>>(at).insert(rt, 3);
        1_u64
    });

    assert_eq!(rt.get(writer), 1);
    assert_eq!(rt.get(odd), 2);
}

fn a_write_in_an_observer_callback_is_routed_before_stabilize_returns<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    c.insert(&rt, 1);
    let g = c.group_by(&rt, |x| x % 2);
    let odd = g.get_group(&1).unwrap().count(&rt);
    assert_eq!(rt.get(odd), 1);
    let trigger = rt.create_input(0_u64);
    let at = stash(c);
    rt.observe(trigger, move |_| {
        let c = stashed::<IncrCollection<i64, C>>(at);
        let rt = stashed::<Runtime<C>>(at + 1);
        c.insert(&rt, 3);
    });
    let rt = stashed::<Runtime<C>>(stash(rt));
    rt.set(trigger, 1);
    rt.stabilize();
    assert_eq!(rt.get(odd), 2);
}

fn later_observers_read_groups_a_callback_moved<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    c.insert(&rt, 1);
    let g = c.group_by(&rt, |x| x % 2);
    let odd = g.get_group(&1).unwrap().count(&rt);
    assert_eq!(rt.get(odd), 1);
    let trigger = rt.create_input(0_u64);
    let at = stash(c);
    let rt = stashed::<Runtime<C>>(stash(rt));
    rt.observe(trigger, move |_| {
        let c = stashed::<IncrCollection<i64, C>>(at);
        let rt = stashed::<Runtime<C>>(at + 1);
        c.insert(&rt, 5);
    });
    let seen = Arc::new(AtomicUsize::new(0));
    let seen_by = Arc::clone(&seen);
    rt.observe(odd, move |&n| seen_by.store(n as usize, Ordering::Relaxed));

    rt.set(trigger, 1);
    stashed::<IncrCollection<i64, C>>(at).insert(&rt, 3);
    rt.stabilize();
    assert_eq!(
        seen.load(Ordering::Relaxed),
        3,
        "a later observer read a group a callback had moved before routing"
    );
}

fn a_second_runtime_used_inside_a_compute_routes_at_its_own_exit<C: Cells>() {
    let a = Runtime::<C>::new();
    let b = Runtime::<C>::new();
    let c = b.create_collection::<i64>();
    c.insert(&b, 1);
    let g = c.group_by(&b, |x| x % 2);
    let odd = g.get_group(&1).unwrap().count(&b);
    assert_eq!(b.get(odd), 1);
    let at = stash((b, c));
    let q = a.create_query(move |_| {
        let world = stashed::<(Runtime<C>, IncrCollection<i64, C>)>(at);
        let (b, c) = &*world;
        c.insert(b, 3);
        b.get(odd)
    });
    assert_eq!(a.get(q), 2, "the other runtime's group inside the compute");
}

fn a_routing_hold_routes_once_when_it_drops<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let g = c.group_by(&rt, move |x| {
        counted.fetch_add(1, Ordering::Relaxed);
        x % 2
    });
    let runs = rt.get(g.version_node());

    let hold = rt.hold_routing();
    c.insert(&rt, 1);
    c.insert(&rt, 3);
    c.delete(&rt, &1);
    c.insert(&rt, 4);
    assert_eq!(calls.load(Ordering::Relaxed), 0, "routing ran under a hold");
    assert_eq!(g.group_count(), 0);
    let inner = rt.hold_routing();
    drop(inner);
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "an inner hold's drop routed"
    );
    drop(hold);

    assert_eq!(calls.load(Ordering::Relaxed), 3);
    assert_eq!(
        rt.get(g.version_node()),
        runs + 1,
        "the router ran more than once"
    );
    assert_eq!(
        (count_of(&rt, &g, 1), count_of(&rt, &g, 0)),
        (Some(1), Some(1))
    );
}

fn a_suppressed_hold_leaves_routing_to_route_pending<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    let g = c.group_by(&rt, |x| x % 2);
    let hold = rt.hold_routing_suppressed();
    c.insert(&rt, 1);
    drop(hold);
    assert_eq!(
        g.group_count(),
        0,
        "a suppressed hold routed when it dropped"
    );
    assert_eq!(queued_routers(&rt), 1);

    assert!(rt.route_pending().is_none());
    assert_eq!(count_of(&rt, &g, 1), Some(1));
    assert_eq!(queued_routers(&rt), 0);
}

fn a_key_fn_that_reads_the_source_routes_after_a_hold<C: Cells>() {
    let at = stash(Runtime::<C>::new());
    let rt = stashed::<Runtime<C>>(at);
    let c = rt.create_collection::<i64>();
    let size = c.count(&rt);
    let g = c.group_by(&rt, move |x| {
        let rt = stashed::<Runtime<C>>(at);
        x % 2 + 10 * i64::from(rt.get(size) > 100)
    });

    let hold = rt.hold_routing();
    c.insert(&rt, 1);
    c.insert(&rt, 2);
    drop(hold);
    assert_eq!(
        (count_of(&rt, &g, 1), count_of(&rt, &g, 0)),
        (Some(1), Some(1))
    );
    assert_eq!(rt.get(size), 2);
}

fn a_pending_list_stays_bounded_under_writes_with_no_read<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    let g = c.group_by(&rt, |x| x % 2);
    let hold = rt.hold_routing_suppressed();
    for x in 0..1000 {
        c.insert(&rt, x);
    }
    assert_eq!(queued_routers(&rt), 1);
    drop(hold);
    assert!(rt.route_pending().is_none());
    assert_eq!(queued_routers(&rt), 0);
    assert_eq!(count_of(&rt, &g, 0), Some(500));
}

fn bad_key(x: &i64) -> i64 {
    assert!(*x != 13, "key_fn refused 13");
    x % 2
}

fn a_failing_key_fn_is_stored_and_fails_again_while_its_row_remains<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    let unrelated = rt.create_input(0_u64);
    let g = c.group_by(&rt, bad_key);
    c.insert(&rt, 1);
    assert!(rt.take_routing_failure().is_none());

    c.insert(&rt, 13);
    let failed = rt
        .take_routing_failure()
        .expect("a failed routing after a write is stored");
    assert_eq!(failed.router(), g.version_node().erase());
    assert!(message(&failed).contains("key_fn refused 13"), "{failed:?}");
    assert!(rt.take_routing_failure().is_none(), "a failure taken twice");

    rt.set(unrelated, 1);
    assert!(rt.take_routing_failure().is_none());
    c.insert(&rt, 3);
    assert!(
        rt.take_routing_failure().is_some(),
        "a write reaching a router with a bad row routes and fails again"
    );
    assert_eq!(
        group_values(&rt, &g, 1),
        vec![1],
        "a failed routing moved a group"
    );
}

fn a_router_recovers_once_the_row_its_key_fn_refused_is_gone<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    let g = c.group_by(&rt, bad_key);
    c.insert(&rt, 1);
    c.insert(&rt, 13);
    assert!(rt.take_routing_failure().is_some());

    c.delete(&rt, &13);
    assert!(
        rt.take_routing_failure().is_none(),
        "routing failed after the refused row went"
    );
    c.insert(&rt, 3);
    assert!(rt.take_routing_failure().is_none());
    assert_eq!(group_values(&rt, &g, 1), vec![1, 3]);
    let _ = rt.get(g.version_node());
}

fn a_runtime_owes_its_queued_routers_and_stored_failures_until_cleared<C: Cells>() {
    let rt = Runtime::<C>::new();
    let other = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    let _g = c.group_by(&rt, bad_key);
    {
        let _hold = rt.hold_routing_suppressed();
        c.insert(&rt, 1);
        assert_eq!(owed_to(&rt), 1, "a queued router is owed");
    }
    assert_eq!(
        owed_to(&other),
        0,
        "another runtime owes what this one queued"
    );
    assert!(rt.route_pending().is_none());
    assert_eq!(owed_to(&rt), 0);

    c.insert(&rt, 13);
    assert_eq!(owed_to(&rt), 1, "a stored failure is owed");
    assert!(other.take_routing_failure().is_none());
    assert!(rt.take_routing_failure().is_some());
    assert_eq!(owed_to(&rt), 0);
}

fn a_stored_failure_is_not_taken_inside_a_compute<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    let _g = c.group_by(&rt, bad_key);
    c.insert(&rt, 13);
    let at = stash(rt);
    let rt = stashed::<Runtime<C>>(at);
    let q = rt.create_query(move |_| {
        let rt = stashed::<Runtime<C>>(at);
        u64::from(rt.take_routing_failure().is_some())
    });
    assert_eq!(rt.get(q), 0, "a compute took a stored failure");
    assert!(rt.take_routing_failure().is_some());
}

thread_local! {
    static EVENTS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

fn event(e: String) {
    EVENTS.with(|events| events.borrow_mut().push(e));
}

/// An element whose `Drop` reports only the instance a caller made, not
/// the engine's copies.
#[derive(Debug)]
struct Tracked {
    v: i64,
    original: bool,
}

impl Clone for Tracked {
    fn clone(&self) -> Self {
        Tracked {
            v: self.v,
            original: false,
        }
    }
}

impl PartialEq for Tracked {
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

impl Eq for Tracked {}

impl Hash for Tracked {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.v.hash(state);
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        if self.original {
            event(format!("drop {}", self.v));
        }
    }
}

fn routing_runs_after_a_mutator_disposes<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<Tracked>();
    let _g = c.group_by(&rt, |t: &Tracked| {
        event(format!("key {}", t.v));
        t.v % 2
    });
    c.insert(
        &rt,
        Tracked {
            v: 1,
            original: true,
        },
    );
    EVENTS.with(|events| events.borrow_mut().clear());

    c.replace(
        &rt,
        Some(&Tracked {
            v: 1,
            original: false,
        }),
        Tracked {
            v: 2,
            original: true,
        },
    );
    let events = EVENTS.with(|events| events.borrow().clone());
    assert_eq!(events, vec!["drop 1".to_string(), "key 2".to_string()]);
}

fn drain_dirtied_watched_reports_a_total_routing_moved<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    c.insert(&rt, 1);
    let g = c.group_by(&rt, |x| x % 2);
    let odd = g.get_group(&1).unwrap().count(&rt);
    assert_eq!(rt.get(odd), 1);
    rt.watch(odd);
    let hold = rt.hold_routing_suppressed();
    c.insert(&rt, 3);
    drop(hold);
    assert_eq!(queued_routers(&rt), 1);
    assert_eq!(rt.drain_dirtied_watched(), vec![odd.erase()]);
    assert_eq!(rt.get(odd), 2);
}

fn a_drain_inside_another_runtimes_compute_leaves_its_frame_untainted<C: Cells>() {
    let a = Runtime::<C>::new();
    let b = Runtime::<C>::new();
    let c = b.create_collection::<i64>();
    let g = c.group_by(&b, |x| x % 2);
    let at = stash((b, c, g));
    let a_at = stash(a);
    let a = stashed::<Runtime<C>>(a_at);
    let q = a.create_query(move |_| {
        let world = stashed::<(
            Runtime<C>,
            IncrCollection<i64, C>,
            GroupedCollection<i64, i64, C>,
        )>(at);
        let (b, c, g) = &*world;
        c.insert(b, 3);
        assert_eq!(g.group_count(), 1, "the other runtime routed at its exit");
        let a = stashed::<Runtime<C>>(a_at);
        let (_, _, ctl) = a.dep_stack.top_for_lookup().expect("the compute's frame");
        // SAFETY: the frame is this compute's, so its control block is live.
        let ctl = unsafe { &*ctl };
        u64::from(ctl.has(crate::cycle::UNREGENERABLE) || ctl.has(crate::cycle::FOREIGN_HOLD))
    });
    assert_eq!(
        a.get(q),
        0,
        "routing another runtime tainted the reading frame"
    );
}

struct WriteOnDrop<'a, C: Cells> {
    rt: &'a Runtime<C>,
    c: &'a IncrCollection<i64, C>,
}

impl<C: Cells> Drop for WriteOnDrop<'_, C> {
    fn drop(&mut self) {
        self.c.insert(self.rt, 3);
    }
}

fn a_panic_unwinding_through_a_write_does_not_route<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    let g = c.group_by(&rt, |x| x % 2);
    let unwound = catch_unwind(AssertUnwindSafe(|| {
        let _write = WriteOnDrop { rt: &rt, c: &c };
        panic!("unwinding");
    }));
    assert!(unwound.is_err());
    assert_eq!(g.group_count(), 0, "a write routed during an unwind");
    assert_eq!(queued_routers(&rt), 1);
    rt.stabilize();
    assert_eq!(count_of(&rt, &g, 1), Some(1), "stabilize routes on entry");
}

fn a_deleted_router_on_the_pending_list_is_skipped<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    let g = c.group_by(&rt, |x| x % 2);
    let hold = rt.hold_routing_suppressed();
    c.insert(&rt, 1);
    assert_eq!(queued_routers(&rt), 1);
    rt.delete_with_dependents(g.version_node());
    drop(hold);
    assert!(rt.route_pending().is_none());
    assert_eq!(queued_routers(&rt), 0);
    c.insert(&rt, 2);
    assert!(rt.take_routing_failure().is_none());
}

fn apply_and_extend_route_once<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<i64>();
    let g = c.group_by(&rt, |x| x % 2);
    let runs = rt.get(g.version_node());
    c.extend(&rt, [1, 2, 3]);
    assert_eq!(rt.get(g.version_node()), runs + 1);
    c.apply(&rt, [Change::Insert(5), Change::Delete(2)]);
    assert_eq!(rt.get(g.version_node()), runs + 2);
    assert_eq!(
        (count_of(&rt, &g, 1), count_of(&rt, &g, 0)),
        (Some(3), Some(0))
    );
}

#[cfg(debug_assertions)]
mod lock_order_at_a_drain {
    use super::*;
    use crate::lock_order::{acquire, LockKind};

    fn a_binding_lock_held_across_a_write_without_a_hold_panics<C: Cells>() {
        let rt = Runtime::<C>::new();
        let c = rt.create_collection::<i64>();
        let _g = c.group_by(&rt, |x| x % 2);
        let caught = catch_unwind(AssertUnwindSafe(|| {
            let _index = acquire(LockKind::Binding);
            c.insert(&rt, 1);
        }));
        let payload = caught.expect_err("routing ran under a Binding lock");
        let text = payload
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default();
        assert!(
            text.contains("routing runs while holding [Binding]"),
            "{text}"
        );
    }

    fn a_hold_taken_before_the_lock_routes_after_it<C: Cells>() {
        let rt = Runtime::<C>::new();
        let c = rt.create_collection::<i64>();
        let g = c.group_by(&rt, |x| x % 2);
        {
            let _hold = rt.hold_routing();
            let _index = acquire(LockKind::Binding);
            c.insert(&rt, 1);
        }
        assert_eq!(count_of(&rt, &g, 1), Some(1));
    }

    both!(
        a_binding_lock_held_across_a_write_without_a_hold_panics,
        a_hold_taken_before_the_lock_routes_after_it,
    );
}

/// A second thread whose write reaches a router another thread is routing
/// waits on it, then routes the nested router that run dirtied, even when
/// the first thread runs both routings.
#[test]
fn shared_a_waiter_routes_the_nested_router_the_others_run_dirtied() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let c = rt.create_collection::<i64>();
    c.insert(&rt, 1);
    let (entered1, first_entered) = mpsc::channel::<()>();
    let (go1, first_go) = mpsc::channel::<()>();
    let first = std::sync::Mutex::new(Some((entered1, first_go)));
    let g = c.group_by(&rt, move |x| {
        if *x == 1001 {
            if let Some((entered, go)) = first.lock().unwrap().take() {
                entered.send(()).unwrap();
                go.recv().unwrap();
            }
        }
        x % 2
    });
    let (entered2, second_entered) = mpsc::channel::<()>();
    let (go2, second_go) = mpsc::channel::<()>();
    let second = std::sync::Mutex::new(Some((entered2, second_go)));
    let odd = g.get_group(&1).unwrap();
    let nested = odd.group_by(&rt, move |x| {
        if *x == 1001 {
            if let Some((entered, go)) = second.lock().unwrap().take() {
                entered.send(()).unwrap();
                go.recv().unwrap();
            }
        }
        x % 3
    });

    let (release, released) = mpsc::channel::<()>();
    let (queued, was_queued) = mpsc::channel::<()>();
    let waiter = {
        let rt = Arc::clone(&rt);
        let c = c.clone();
        let nested = nested.clone();
        std::thread::spawn(move || {
            let hold = rt.hold_routing();
            c.insert(&rt, 3);
            queued.send(()).unwrap();
            released.recv().unwrap();
            drop(hold);
            (count_of(&rt, &nested, 0), count_of(&rt, &nested, 2))
        })
    };
    was_queued.recv().unwrap();
    let writer = {
        let rt = Arc::clone(&rt);
        let c = c.clone();
        std::thread::spawn(move || c.insert(&rt, 1001))
    };
    first_entered.recv().unwrap();
    release.send(()).unwrap();
    std::thread::sleep(SETTLE);
    go1.send(()).unwrap();
    second_entered.recv().unwrap();
    std::thread::sleep(SETTLE);
    go2.send(()).unwrap();
    writer.join().unwrap();
    assert_eq!(waiter.join().unwrap(), (Some(1), Some(1)));
}

/// Two threads insert into one source, and each reads its own row's
/// group straight after its insert.
#[test]
fn shared_each_writer_reads_its_own_group_after_its_insert() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let c = rt.create_collection::<i64>();
    let g = c.group_by(&rt, |x| x % 7);
    let threads: Vec<_> = (0..2_i64)
        .map(|t| {
            let rt = Arc::clone(&rt);
            let c = c.clone();
            let g = g.clone();
            std::thread::spawn(move || {
                for i in 0..300_i64 {
                    let x = i * 2 + t;
                    c.insert(&rt, x);
                    let group = g.get_group(&(x % 7)).expect("the row's group");
                    assert!(
                        group.values(&rt).contains(&x),
                        "thread {t} did not see {x} in its group"
                    );
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    assert_eq!(g.keys().len(), 7);
}

#[test]
fn local_the_router_has_its_upstream_edge_from_creation() {
    let rt = Runtime::<Local>::new();
    let c = rt.create_collection::<i64>();
    let g = c.group_by(&rt, |x| x % 2);
    assert_eq!(
        rt.state_of(g.version_node()),
        crate::state::NodeState::Clean
    );
}
