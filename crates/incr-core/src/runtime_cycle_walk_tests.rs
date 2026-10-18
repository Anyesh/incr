//! Walks over former cycle members. After a cycle breaks, its members
//! are New with installed edges on each other, and a walk must expand
//! each of them at most once. The edges are installed by hand here, as
//! an episode leaves them; each case runs in a child process, so a walk
//! that spins is killed rather than hanging the suite.

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::child_test::run_in_child;
use crate::state::NodeState;

/// P and U are a 2-member cycle read by X, with P also reading j.
fn two_member_set_returns<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let j = rt.create_input(1_i64);
    let p = rt.create_query(move |rt| rt.get(j));
    let u = rt.create_query(move |rt| rt.get(p) + 1);
    let x = rt.create_query(move |rt| rt.get(p) * 10);
    assert_eq!(rt.get(u), 2);
    assert_eq!(rt.get(x), 10);
    rt.add_edges_for_test(p, &[u]);
    rt.force_state_for_test(p, NodeState::Cyclic);
    rt.force_state_for_test(u, NodeState::Cyclic);

    rt.set(j, 2);
    assert_eq!(rt.state_of(p), NodeState::New);
    assert_eq!(rt.state_of(u), NodeState::New);
    assert_eq!(rt.state_of(x), NodeState::Dirty);
    rt.set(j, 3);
    assert_eq!(rt.state_of(p), NodeState::New);
    assert_eq!(rt.state_of(x), NodeState::Dirty);

    let before = rt.expansions();
    assert_eq!(rt.get(u), 4);
    assert!(
        rt.expansions() - before <= 5,
        "{} expansions for 3 edges and 2 members",
        rt.expansions() - before
    );
    assert_eq!(rt.get(x), 30);
    rt.assert_invariant_i();
}

/// P reads itself, as a one-member cycle does.
fn self_reading_set_returns<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let j = rt.create_input(1_i64);
    let p = rt.create_query(move |rt| rt.get(j));
    let x = rt.create_query(move |rt| rt.get(p) * 10);
    assert_eq!(rt.get(x), 10);
    rt.add_edges_for_test(p, &[p]);
    rt.force_state_for_test(p, NodeState::Cyclic);

    rt.set(j, 2);
    assert_eq!(rt.state_of(p), NodeState::New);
    assert_eq!(rt.state_of(x), NodeState::Dirty);
    rt.set(j, 3);

    let before = rt.expansions();
    assert_eq!(rt.get(p), 3);
    assert!(
        rt.expansions() - before <= 3,
        "{} expansions for 2 edges and 1 member",
        rt.expansions() - before
    );
    assert_eq!(rt.get(x), 30);
    rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_set_under_new_two_member_cycle_returns() {
    run_in_child(
        "runtime::cycle_walk_tests::local_set_under_new_two_member_cycle_returns",
        two_member_set_returns::<Local>,
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_set_under_new_two_member_cycle_returns() {
    run_in_child(
        "runtime::cycle_walk_tests::shared_set_under_new_two_member_cycle_returns",
        two_member_set_returns::<Shared>,
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_set_under_new_self_reading_member_returns() {
    run_in_child(
        "runtime::cycle_walk_tests::local_set_under_new_self_reading_member_returns",
        self_reading_set_returns::<Local>,
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_set_under_new_self_reading_member_returns() {
    run_in_child(
        "runtime::cycle_walk_tests::shared_set_under_new_self_reading_member_returns",
        self_reading_set_returns::<Shared>,
    )
    .assert_clean();
}
