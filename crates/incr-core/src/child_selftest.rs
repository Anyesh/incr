use crate::child_test::{run_in_child, run_in_child_within, HANG_TIMEOUT};
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

#[test]
#[cfg_attr(miri, ignore)]
#[should_panic(expected = "never ran a test named child_selftest::no_such_test")]
fn child_that_names_no_test_fails() {
    run_in_child("child_selftest::no_such_test", || {});
}

/// A sanitized build (`INCR_TEST_REDUCED`) gives a child longer than the
/// plain build's timeout.
#[test]
#[cfg_attr(miri, ignore)]
fn a_reduced_build_child_outlives_the_plain_timeout() {
    if std::env::var_os("INCR_TEST_REDUCED").is_none() {
        return;
    }
    run_in_child(
        "child_selftest::a_reduced_build_child_outlives_the_plain_timeout",
        || std::thread::sleep(HANG_TIMEOUT + Duration::from_secs(1)),
    )
    .assert_clean();
}
