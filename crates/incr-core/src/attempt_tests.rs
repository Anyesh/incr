use super::*;

#[test]
fn a_take_over_wounds_a_pinned_attempt_in_flight_and_keeps_the_pin() {
    begin();
    let a = mine();
    assert!(a.pin());
    assert!(!a.wound_in_flight(claim(1)), "no discard in flight");
    enter_flight();
    assert!(a.wound_in_flight(claim(1)));
    assert_eq!(a.flags(), IN_FLIGHT | PINNED | WOUNDED);
    assert!(leave_flight());
    assert_eq!(a.flags(), PINNED | WOUNDED);
}

#[test]
fn a_wounded_attempt_takes_no_pin() {
    begin();
    let a = mine();
    enter_flight();
    assert!(a.wound_in_flight(claim(1)));
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
    assert!(!a.wound_in_flight(claim(1)));
    assert!(!a.wounded());
}

#[test]
fn an_old_attempts_word_acts_on_no_later_attempt() {
    begin();
    let old = mine();
    begin();
    enter_flight();
    assert!(!old.wound_in_flight(claim(1)));
    assert!(!old.pin());
    assert!(!old.wounded());
    assert_eq!(mine().flags(), IN_FLIGHT);
    assert!(!leave_flight());
}

#[test]
// Miri's clock advances with executed code, so the STUCK_AFTER loop alone
// outlasts STUCK_FOR.
#[cfg_attr(miri, ignore)]
fn a_waiter_takes_over_only_a_flight_that_stays_open() {
    begin();
    let a = mine();
    let mut patience = Patience::default();
    enter_flight();
    for _ in 0..STUCK_AFTER {
        assert!(!a.wound_if_stuck(claim(1), &mut patience));
    }
    assert!(!leave_flight());
    enter_flight();
    for _ in 0..STUCK_AFTER {
        assert!(
            !a.wound_if_stuck(claim(1), &mut patience),
            "a new flight starts the count over"
        );
    }
    assert!(
        !a.wound_if_stuck(claim(1), &mut patience),
        "open for under STUCK_FOR"
    );
    std::thread::sleep(STUCK_FOR);
    assert!(a.wound_if_stuck(claim(1), &mut patience));
    assert!(leave_flight());
}

fn claim(slot: u32) -> crate::waits::Claim {
    (crate::handle::RuntimeId::for_test(1), slot)
}

#[test]
fn a_take_over_tells_its_owner_the_claim_it_met() {
    begin();
    let a = mine();
    let met = claim(7);
    enter_flight();
    assert!(a.wound_in_flight(met));
    assert!(
        a.wound_in_flight(claim(8)),
        "a later waiter finds it wounded and lists nothing"
    );
    assert!(leave_flight());
    assert_eq!(take_met(), vec![met]);
    assert!(take_met().is_empty());
}

#[test]
fn a_claim_met_on_an_old_attempt_reaches_no_later_attempt() {
    begin();
    let old = mine();
    begin();
    enter_flight();
    assert!(!old.wound_in_flight(claim(7)));
    let met = claim(8);
    assert!(mine().wound_in_flight(met));
    assert!(leave_flight());
    assert_eq!(take_met(), vec![met]);
}

#[test]
fn a_discard_the_engine_raises_again_is_back_in_flight() {
    use crate::cycle_state::UnwindKind;
    use crate::cycle_state::{reopen_discard, set_unwind_kind, set_wounded, settle_discard};
    begin();
    set_unwind_kind(Some(UnwindKind::Discard));
    assert!(!settle_discard());
    assert!(!in_flight(), "the engine holds it");
    reopen_discard();
    assert!(in_flight(), "raised again past the engine");
    assert!(mine().wound_in_flight(claim(1)));
    assert!(settle_discard());
    set_unwind_kind(None);
    set_wounded(false);
    crate::reserve::clear_taken();
}

/// A waiter's take-over attempts on `a`'s attempt through `4 * STUCK_FOR`
/// on a thread of its own, after `first` registers what it waits on and
/// holds: whether it took the attempt over, and how often the wound order
/// refused it.
fn take_over_from(
    a: &Attempt,
    rt: crate::handle::RuntimeId,
    first: fn(crate::handle::RuntimeId) -> Option<crate::waits::Registration>,
) -> (bool, u32) {
    let a = a.clone();
    std::thread::spawn(move || {
        let _waiting = first(rt);
        let mut patience = Patience::default();
        let until = std::time::Instant::now() + 4 * STUCK_FOR;
        let took = loop {
            if a.wound_if_stuck((rt, 1), &mut patience) {
                break true;
            }
            if std::time::Instant::now() >= until {
                break false;
            }
            std::thread::yield_now();
        };
        (took, crate::waits::take_refusals())
    })
    .join()
    .unwrap()
}

/// A take-over is for an owner the registry cannot see. An older owner
/// whose wait closes a waits-for cycle through registered holders, here
/// the waiter's own claim, gives way only by the wound order.
#[test]
#[cfg_attr(miri, ignore)]
fn a_waiter_takes_over_no_older_owner_waiting_in_the_registry() {
    let rt = crate::handle::RuntimeId::for_test(u64::MAX - 11);
    begin();
    let a = enter_hold();
    let waiting = crate::waits::Registration::new(|_| 1, (rt, 2), vec![(rt, 1)]);
    let (took, refused) = take_over_from(&a, rt, |rt| {
        Some(crate::waits::Registration::new(
            |_| 2,
            (rt, 1),
            vec![(rt, 2)],
        ))
    });
    assert!(
        !took,
        "a younger waiter wounded an older owner that waits on it in the registry"
    );
    assert!(refused > 0, "the wound order never decided a take-over");
    assert!(!a.wounded());
    drop(waiting);
    assert!(
        take_over_from(&a, rt, |_| None).0,
        "an owner outside the registry is taken over"
    );
    assert!(leave_hold());
}

/// An older owner that waits on a claim no registered waiter holds may
/// wait on a thread in user code that waits on the waiter: the take-over
/// stays, or the three would wait on one another for good.
#[test]
#[cfg_attr(miri, ignore)]
fn a_waiter_takes_over_an_older_owner_whose_wait_leaves_the_registry() {
    let rt = crate::handle::RuntimeId::for_test(u64::MAX - 12);
    begin();
    let a = enter_hold();
    let waiting = crate::waits::Registration::new(|_| 1, (rt, 2), vec![(rt, 1)]);
    assert!(
        take_over_from(&a, rt, |_| None).0,
        "the owner's wait on an unregistered holder kept its landing from the waiter"
    );
    drop(waiting);
    assert!(leave_hold());
}
