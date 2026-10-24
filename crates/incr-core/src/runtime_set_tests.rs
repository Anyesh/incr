//! `set` claims its input (bug g): the claim protocol against deletes,
//! recycles, equal in-flight sets and owner deletes. Park tests run under
//! `Shared`; the tests that need no second thread run under both.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use super::race_tests::{grouped, id, panic_message, park_at, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::node::NodeData;
use crate::state::NodeState;

type Rt = Runtime<Shared>;

const SETTLE: Duration = Duration::from_millis(50);

/// Bug (g): x is deleted and its slot recycled by another input while
/// `set(x)` is parked between its handle check and its claim. The set
/// must panic as stale and leave the new occupant's value alone.
#[test]
fn set_vs_delete_recycle() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(7_u64);
    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetBeforeClaim(id(x)), gate.clone())]);

    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.set(x, 8))))
    };
    gate.wait_parked();
    rt.delete_node(x);
    let y = rt.create_input(100_u64);
    assert_eq!(
        y.slot(),
        x.slot(),
        "the delete should free x's slot for reuse"
    );
    gate.release();

    let outcome = setter.join().unwrap();
    let msg = panic_message(outcome.expect_err("a set of a deleted input returned"));
    assert!(msg.contains("stale"), "got: {msg}");
    assert_eq!(rt.state_of(y), NodeState::Clean);
    assert_eq!(rt.get(y), 100, "the set wrote into the recycled slot");
    rt.assert_invariant_i();
}

/// `set(x)` resumes after a query recycled x's slot. Its generation check
/// and its claim share one read hold, so it panics as stale without
/// claiming the query, which a set's claim on its `verified_at` would
/// corrupt.
#[test]
fn set_never_claims_the_query_that_recycled_its_slot() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(7_u64);
    let y = rt.create_input(1_u64);
    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetBeforeClaim(id(x)), gate.clone())]);

    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.set(x, 8))))
    };
    gate.wait_parked();
    rt.delete_node(x);
    let q = rt.create_query(move |rt: &Rt| rt.get(y) * 10);
    assert_eq!(
        q.slot(),
        x.slot(),
        "the delete should free x's slot for reuse"
    );
    assert_eq!(rt.get(q), 10);
    gate.release();

    let outcome = setter.join().unwrap();
    let msg = panic_message(outcome.expect_err("a set of a deleted input returned"));
    assert!(msg.contains("stale"), "got: {msg}");
    assert!(!rt.nodes.get(q.slot()).set_claimed());
    assert_eq!(rt.state_of(q), NodeState::Clean);
    rt.set(y, 2);
    assert_eq!(rt.get(q), 20);
    rt.assert_invariant_i();
}

/// Records every park point a runtime reaches.
fn record_parks<C: Cells>(rt: &Runtime<C>) -> Arc<Mutex<Vec<ParkPoint>>> {
    let reached = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&reached);
    rt.park_hook
        .set(Box::new(move |p| log.lock().unwrap().push(p)))
        .ok()
        .expect("park hook installed twice");
    reached
}

/// A no-op set compares under its claim and releases it without a bump
/// or a walk; a changing set bumps and walks, and both leave the input
/// unclaimed and Clean.
fn noop_set_bumps_and_walks_nothing<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(3_u64);
    let d = rt.create_query(move |rt: &Runtime<C>| rt.get(x) + 1);
    assert_eq!(rt.get(d), 4);
    let reached = record_parks(rt);
    let revision = rt.current_revision();
    let claimed = || rt.nodes.get(x.slot()).set_claimed();

    rt.set(x, 3);

    assert_eq!(rt.current_revision(), revision);
    assert_eq!(rt.state_of(d), NodeState::Clean);
    assert!(!claimed());
    assert_eq!(
        reached.lock().unwrap().as_slice(),
        [
            ParkPoint::SetBeforeClaim(id(x)),
            ParkPoint::SetAfterClaim(id(x)),
        ]
    );
    reached.lock().unwrap().clear();
    rt.set(x, 4);
    assert_eq!(
        reached.lock().unwrap().as_slice(),
        [
            ParkPoint::SetBeforeClaim(id(x)),
            ParkPoint::SetAfterClaim(id(x)),
            ParkPoint::SetAfterBump,
            ParkPoint::SetAfterWalk,
        ]
    );
    assert!(!claimed());
    assert_eq!(rt.state_of(x), NodeState::Clean);
    assert_eq!(rt.nodes.get(x.slot()).verified_at(), rt.current_revision());
    assert_eq!(rt.get(d), 5);
}

#[test]
fn local_noop_set_bumps_and_walks_nothing() {
    noop_set_bumps_and_walks_nothing(&Runtime::<Local>::new());
}

#[test]
fn shared_noop_set_bumps_and_walks_nothing() {
    noop_set_bumps_and_walks_nothing(&Rt::new());
}

/// Revisions step by 2 from 2. An input carries the odd stamp below the
/// revision it was created, recycled or set at, and a compute carries the
/// even revision it ran at.
fn inputs_carry_odd_stamps<C: Cells>(rt: &Runtime<C>) {
    let changed = |slot: u32| rt.nodes.get(slot).changed_at();
    let walked = || C::walked(&rt.walk, &rt.revision);
    assert_eq!((rt.current_revision(), walked()), (2, 2));
    let x = rt.create_input(1_u64);
    assert_eq!(changed(x.slot()), 1);
    assert_eq!(rt.nodes.get(x.slot()).verified_at(), 2);
    let d = rt.create_query(move |rt: &Runtime<C>| rt.get(x) + 1);
    assert_eq!(rt.get(d), 2);
    assert_eq!(changed(d.slot()), 2);

    rt.set(x, 2);
    assert_eq!((rt.current_revision(), walked()), (4, 4));
    assert_eq!(changed(x.slot()), 3);
    assert_eq!(rt.nodes.get(x.slot()).verified_at(), 4);
    assert_eq!(rt.get(d), 3);
    assert_eq!(changed(d.slot()), 4);

    rt.delete_node(d);
    let y = rt.create_input(5_u64);
    assert_eq!(y.slot(), d.slot());
    assert_eq!(changed(y.slot()), 3);
}

#[test]
fn local_inputs_carry_odd_stamps() {
    inputs_carry_odd_stamps(&Runtime::<Local>::new());
}

#[test]
fn shared_inputs_carry_odd_stamps() {
    inputs_carry_odd_stamps(&Rt::new());
}

/// A set equal to a value another set has written but not yet walked
/// returns only once that walk is recorded, so its caller can rely on
/// dependents having been dirtied.
#[test]
fn equal_set_returns_after_the_in_flight_change_has_walked() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let d = rt.create_query(move |rt: &Rt| rt.get(x) + 1);
    assert_eq!(rt.get(d), 2);
    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterWalk, gate.clone())]);

    let first = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(x, 5))
    };
    gate.wait_parked();
    let second = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || {
            rt.set(x, 5);
            rt.state_of(d)
        })
    };
    thread::sleep(SETTLE);
    assert!(
        !second.is_finished(),
        "the equal set returned before the first set's walk was recorded"
    );

    gate.release();
    first.join().unwrap();
    assert_eq!(second.join().unwrap(), NodeState::Dirty);
    rt.assert_walks_complete();
    assert_eq!(rt.get(d), 6);
    rt.assert_invariant_i();
}

/// A set equal to the value of a set still under its claim waits for that
/// set to land and walk, then returns without writing.
#[test]
fn equal_set_waits_on_the_claim_of_the_set_that_wrote_it() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let d = rt.create_query(move |rt: &Rt| rt.get(x) + 1);
    assert_eq!(rt.get(d), 2);
    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterBump, gate.clone())]);

    let first = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(x, 5))
    };
    gate.wait_parked();
    let second = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || {
            rt.set(x, 5);
            rt.state_of(d)
        })
    };
    thread::sleep(SETTLE);
    assert!(
        !second.is_finished(),
        "the equal set returned under the first set's claim"
    );

    gate.release();
    first.join().unwrap();
    assert_eq!(second.join().unwrap(), NodeState::Dirty);
    assert_eq!(rt.get(d), 6);
    rt.assert_invariant_i();
}

/// A set's claim is invisible to readers: a read of the input and of a
/// dependent goes through under it with the old value, and the dependent
/// sees the new value once the set has walked.
#[test]
fn reader_of_an_input_under_a_set_claim_does_not_wait() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let d = rt.create_query(move |rt: &Rt| rt.get(x) + 1);
    assert_eq!(rt.get(d), 2);
    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterClaim(id(x)), gate.clone())]);

    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(x, 5))
    };
    gate.wait_parked();
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || (rt.get(x), rt.get(d)))
    };
    assert_eq!(reader.join().unwrap(), (1, 2));
    assert_eq!(rt.state_of(x), NodeState::Clean);

    gate.release();
    setter.join().unwrap();
    assert_eq!(rt.get(d), 6);
    rt.assert_invariant_i();
}

/// A delete of an input waits for the set that holds it to release its
/// claim before it retires the node, and a set that comes after the
/// delete marked it panics as stale instead of claiming it.
#[test]
fn delete_of_an_input_waits_for_its_set_and_refuses_later_sets() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterClaim(id(x)), gate.clone())]);

    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(x, 5))
    };
    gate.wait_parked();
    let deleter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.delete_with_dependents(x))
    };
    let node = rt.nodes.get(x.slot());
    while Shared::u64_load_acquire(&node.verified_at) & NodeData::<Shared>::DELETED == 0 {
        thread::yield_now();
    }
    let late = catch_unwind(AssertUnwindSafe(|| rt.set(x, 6)));
    let msg = panic_message(late.expect_err("a set of a marked input returned"));
    assert!(msg.contains("stale"), "got: {msg}");
    thread::sleep(SETTLE);
    assert!(
        !deleter.is_finished(),
        "the delete finished under the set's claim"
    );
    assert_eq!(node.generation(), x.generation());

    gate.release();
    setter.join().unwrap();
    assert_eq!(deleter.join().unwrap(), 1);
    assert!(!rt.is_alive(x));
    rt.assert_invariant_i();
}

/// A strict delete of a router finds one of its group version nodes
/// under a parked `set`'s claim. It must wait for the set to release its
/// claim before it retires the node.
#[test]
fn delete_of_an_owner_waits_for_a_set_of_an_owned_input() {
    let rt = Arc::new(Rt::new());
    let (router, versions) = grouped(&rt);
    let group = versions[0];
    let gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::SetAfterClaim(id(group)), gate.clone())],
    );

    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(group, 99))
    };
    gate.wait_parked();
    let deleter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.try_delete_node(router))
    };
    thread::sleep(SETTLE);
    assert!(
        !deleter.is_finished(),
        "the router delete finished under the group's set claim"
    );
    assert_eq!(
        rt.nodes.get(group.slot()).generation(),
        group.generation(),
        "the group was retired under its set's claim"
    );

    gate.release();
    setter.join().unwrap();
    assert_eq!(deleter.join().unwrap(), Ok(()));
    assert!(!rt.is_alive(router));
    assert!(versions.iter().all(|&v| !rt.is_alive(v)));
    rt.assert_invariant_i();
}

/// A mapped write records the stamp its own write settled (T10 mapped
/// stamp). A second `set` of the input can claim and settle as soon as the
/// first releases its claim, before the first returns, so a stamp loaded
/// after the return would be the second write's.
#[test]
fn set_returning_displaced_returns_the_stamp_it_settled() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(0_u64);
    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterWalk, gate.clone())]);
    let mapped = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set_returning_displaced(x, 1).1)
    };
    gate.wait_parked();
    let settled = rt.changed_at(x);
    rt.set(x, 2);
    let bypass = rt.changed_at(x);
    assert_ne!(bypass, settled);
    gate.release();
    let stamp = mapped.join().unwrap();
    assert_eq!(stamp, settled, "the stamp is the mapped write's own");
    assert_ne!(stamp, rt.changed_at(x), "a later set changes changed_at");
}

#[test]
fn set_returning_displaced_returns_the_current_stamp_on_an_equal_value() {
    let rt = Rt::new();
    let x = rt.create_input(4_u64);
    rt.set(x, 5);
    let before = rt.changed_at(x);
    let (compared, stamp) = rt.set_returning_displaced(x, 5);
    assert!(matches!(compared, crate::value_slot::Compared::Equal(5)));
    assert_eq!(stamp, before);
}

/// One input read by 50k queries (5k under `INCR_TEST_REDUCED`), half of
/// which then drop it, and a quarter of the rest are deleted. Each unhook
/// from the input's dependent list must not scan or shift the whole list.
fn wide_input_rewires_in_sublinear_time<C: Cells>(rt: &Runtime<C>) {
    let n = crate::test_size::test_size(50_000);
    let x = rt.create_input(1_i64);
    let drop_x = rt.create_input(false);
    let queries: Vec<_> = (0..n)
        .map(|i| {
            rt.create_query(move |rt| {
                if i % 2 == 0 && rt.get(drop_x) {
                    0
                } else {
                    rt.get(x)
                }
            })
        })
        .collect();
    for &q in &queries {
        assert_eq!(rt.get(q), 1);
    }
    let start = std::time::Instant::now();
    rt.set(drop_x, true);
    for &q in queries.iter().step_by(2) {
        assert_eq!(rt.get(q), 0);
    }
    for &q in queries.iter().skip(1).step_by(4) {
        rt.delete_node(q);
    }
    let took = start.elapsed();
    assert_eq!(
        rt.inner.read().dependents[x.slot() as usize].iter().count(),
        n / 4
    );
    assert!(
        took < Duration::from_secs(5),
        "unhooking {} dependents took {took:?}",
        n / 2 + n / 8
    );
}

#[cfg(not(miri))]
#[test]
fn local_wide_input_rewires_in_sublinear_time() {
    wide_input_rewires_in_sublinear_time(&Runtime::<Local>::new());
}

#[cfg(not(miri))]
#[test]
fn shared_wide_input_rewires_in_sublinear_time() {
    wide_input_rewires_in_sublinear_time(&Runtime::<Shared>::new());
}
