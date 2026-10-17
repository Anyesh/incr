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

/// `set(x)`'s claim lands on the Clean query that recycled x's slot. With
/// a walk flagging the query while the claim is held it lands Dirty,
/// without one Clean; either way it is not left Computing and the set
/// panics as stale.
fn set_claim_lands_on_a_recycled_query(flagged_by_a_walk: bool) {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(7_u64);
    let y = rt.create_input(1_u64);
    let before = Gate::new();
    let after = Gate::new();
    park_at(
        &rt,
        vec![
            (ParkPoint::SetBeforeClaim(id(x)), before.clone()),
            (ParkPoint::SetAfterClaim(id(x)), after.clone()),
        ],
    );

    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.set(x, 8))))
    };
    before.wait_parked();
    rt.delete_node(x);
    let q = rt.create_query(move |rt: &Rt| rt.get(y) * 10);
    assert_eq!(
        q.slot(),
        x.slot(),
        "the delete should free x's slot for reuse"
    );
    assert_eq!(rt.get(q), 10);
    before.release();

    after.wait_parked();
    assert_eq!(rt.state_of(q), NodeState::Computing);
    if flagged_by_a_walk {
        rt.set(y, 2);
        assert_eq!(rt.state_of(q), NodeState::ComputingDirty);
    }
    after.release();

    let outcome = setter.join().unwrap();
    let msg = panic_message(outcome.expect_err("a set of a deleted input returned"));
    assert!(msg.contains("stale"), "got: {msg}");
    let (state, value) = if flagged_by_a_walk {
        (NodeState::Dirty, 20)
    } else {
        (NodeState::Clean, 10)
    };
    assert_eq!(rt.state_of(q), state);
    assert_eq!(rt.get(q), value);
    rt.assert_invariant_i();
}

#[test]
fn set_claim_on_a_recycled_query_lands_dirty_when_a_walk_flagged_it() {
    set_claim_lands_on_a_recycled_query(true);
}

#[test]
fn set_claim_on_a_recycled_query_lands_clean_when_nothing_flagged_it() {
    set_claim_lands_on_a_recycled_query(false);
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

fn noop_set_takes_no_claim<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(3_u64);
    let d = rt.create_query(move |rt: &Runtime<C>| rt.get(x) + 1);
    assert_eq!(rt.get(d), 4);
    let reached = record_parks(rt);
    let revision = rt.current_revision();

    rt.set(x, 3);

    assert_eq!(rt.current_revision(), revision);
    assert_eq!(rt.state_of(d), NodeState::Clean);
    assert!(
        reached.lock().unwrap().is_empty(),
        "a no-op set went past its no-op check: {:?}",
        reached.lock().unwrap()
    );
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
    assert_eq!(rt.get(d), 5);
}

#[test]
fn local_noop_set_takes_no_claim() {
    noop_set_takes_no_claim(&Runtime::<Local>::new());
}

#[test]
fn shared_noop_set_takes_no_claim() {
    noop_set_takes_no_claim(&Rt::new());
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

/// A reader of an input blocks while a value-changing set holds its
/// claim and sees the new value once the set lands.
#[test]
fn reader_of_an_input_under_a_set_claim_waits_for_the_landing() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let gate = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterClaim(id(x)), gate.clone())]);

    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(x, 5))
    };
    gate.wait_parked();
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(x))
    };
    thread::sleep(SETTLE);
    assert!(
        !reader.is_finished(),
        "a read went through under the set's claim"
    );

    gate.release();
    setter.join().unwrap();
    assert_eq!(reader.join().unwrap(), 5);
    rt.assert_invariant_i();
}

/// A strict delete of a router finds one of its group version nodes
/// under a parked `set`'s claim. It must wait for the set to land instead
/// of stealing the claim.
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
    assert!(rt.is_alive(group), "the group's set claim was stolen");

    gate.release();
    setter.join().unwrap();
    assert_eq!(deleter.join().unwrap(), Ok(()));
    assert!(!rt.is_alive(router));
    assert!(versions.iter().all(|&v| !rt.is_alive(v)));
    rt.assert_invariant_i();
}
