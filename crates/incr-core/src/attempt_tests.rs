use super::*;

#[test]
fn a_take_over_wounds_a_pinned_attempt_in_flight_and_keeps_the_pin() {
    begin();
    let a = mine();
    assert!(a.pin());
    assert!(!a.wound_in_flight(), "no discard in flight");
    enter_flight();
    assert!(a.wound_in_flight());
    assert_eq!(a.flags(), IN_FLIGHT | PINNED | WOUNDED);
    assert!(leave_flight());
    assert_eq!(a.flags(), PINNED | WOUNDED);
}

#[test]
fn a_wounded_attempt_takes_no_pin() {
    begin();
    let a = mine();
    enter_flight();
    assert!(a.wound_in_flight());
    assert!(!a.pin());
    assert!(leave_flight());
    assert_eq!(a.flags(), WOUNDED);
}

#[test]
fn a_flight_that_ends_first_leaves_the_attempt_unwounded() {
    begin();
    let a = mine();
    enter_flight();
    assert!(!leave_flight());
    assert!(!a.wound_in_flight());
    assert!(!a.wounded());
}

#[test]
fn an_old_attempts_word_acts_on_no_later_attempt() {
    begin();
    let old = mine();
    begin();
    enter_flight();
    assert!(!old.wound_in_flight());
    assert!(!old.pin());
    assert!(!old.wounded());
    assert_eq!(mine().flags(), IN_FLIGHT);
    assert!(!leave_flight());
}
