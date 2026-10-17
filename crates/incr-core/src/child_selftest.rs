use crate::child_test::{run_in_child, run_in_child_within};
use std::time::Duration;

#[test]
#[cfg_attr(miri, ignore)]
fn child_that_returns_exits_clean() {
    run_in_child("child_selftest::child_that_returns_exits_clean", || {}).assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn child_that_panics_reports_its_message() {
    run_in_child(
        "child_selftest::child_that_panics_reports_its_message",
        || panic!("stale handle marker 7731"),
    )
    .assert_panicked_with("stale handle marker 7731");
}

#[test]
#[cfg_attr(miri, ignore)]
fn child_that_spins_is_killed_at_the_timeout() {
    run_in_child_within(
        "child_selftest::child_that_spins_is_killed_at_the_timeout",
        Duration::from_millis(300),
        || loop {
            std::thread::sleep(Duration::from_millis(5));
        },
    )
    .assert_killed();
}
