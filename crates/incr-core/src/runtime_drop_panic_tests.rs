//! Panics that follow a compute's value write, and the retire sites that
//! can run a user `Drop`. The `Shared` cases run in a child process: a
//! panicking `Drop` inside a haphazard reclaim leaks the rest of that
//! batch, and the child keeps the test binary's global domain clean.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicI64, Ordering::SeqCst};
use std::sync::Arc;
use std::thread;

use super::race_tests::{park_at, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::child_test::run_in_child;
use crate::reclaim_probe::{self, RetireSite};
use crate::state::NodeState;

/// A value whose `Drop` panics once, for the value `ARMED` names.
#[derive(Clone, PartialEq, Debug)]
struct Boom(i64);

static ARMED: AtomicI64 = AtomicI64::new(i64::MIN);

fn arm(value: i64) {
    ARMED.store(value, SeqCst);
}

impl Drop for Boom {
    fn drop(&mut self) {
        if ARMED
            .compare_exchange(self.0, i64::MIN, SeqCst, SeqCst)
            .is_ok()
        {
            panic!("drop of Boom({}) panicked on purpose", self.0);
        }
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(s) => *s,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(s) => (*s).to_string(),
            Err(_) => "<non-string panic payload>".to_string(),
        },
    }
}

/// The global domain reclaims on the first retire of the process and
/// then every two seconds; one throwaway retire here starts that clock,
/// so a value retired next stays pending until a site is forced.
fn prime_reclaim_timer<C: Cells>(rt: &Runtime<C>) {
    let w = rt.create_input(String::new());
    rt.set(w, "primed".to_string());
}

/// Q recomputes from v1 to v2 and the drop of v1 panics after the slot
/// holds v2. Q must land with its `changed_at` stamped: its next
/// recompute yields v2 again, cuts off, and D (which verified against
/// v1) must still recompute instead of verifying Clean over v2.
fn drop_panic_after_write_stamps_changed_at<C: Cells>(rt: &Runtime<C>) {
    let x = rt.create_input(1_i64);
    let trigger = rt.create_input(0_i64);
    let q = rt.create_query(move |rt| {
        let _ = rt.get(trigger);
        Boom(rt.get(x))
    });
    let d = rt.create_query(move |rt| rt.get(q).0 * 10);
    assert_eq!(rt.get(d), 10);

    rt.set(x, 2);
    arm(1);
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(q))).unwrap_err();
    assert!(
        panic_message(err).contains("drop of Boom(1)"),
        "the drop panic must leave the read"
    );
    assert_eq!(rt.state_of(q), NodeState::Failed);

    rt.set(trigger, 1);
    assert_eq!(rt.get(d), 20);
    assert_eq!(rt.state_of(q), NodeState::Clean);
}

#[test]
fn local_drop_panic_after_write_stamps_changed_at() {
    drop_panic_after_write_stamps_changed_at(&Runtime::<Local>::new());
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_drop_panic_after_write_stamps_changed_at() {
    run_in_child(
        "runtime::drop_panic_tests::shared_drop_panic_after_write_stamps_changed_at",
        || {
            let rt = Runtime::<Shared>::new();
            prime_reclaim_timer(&rt);
            reclaim_probe::force(RetireSite::ComputeWrite);
            drop_panic_after_write_stamps_changed_at(&rt);
        },
    )
    .assert_clean();
}

/// The Superseded variant: Q's failing recompute also reads a new dep
/// whose `set` is still in flight, so the `Err` path publishes
/// Superseded and lands Dirty. Without the stamp the next recompute
/// would cut off over v2 and leave D verified Clean against v1.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_drop_panic_after_write_lands_dirty_when_superseded() {
    run_in_child(
        "runtime::drop_panic_tests::shared_drop_panic_after_write_lands_dirty_when_superseded",
        || {
            let rt = Arc::new(Runtime::<Shared>::new());
            prime_reclaim_timer(&rt);
            reclaim_probe::force(RetireSite::ComputeWrite);
            let x = rt.create_input(1_i64);
            let late = rt.create_input(5_i64);
            let trigger = rt.create_input(0_i64);
            let q = rt.create_query(move |rt| {
                if rt.get(trigger) != 0 {
                    let _ = rt.get(late);
                }
                Boom(rt.get(x))
            });
            let d = rt.create_query(move |rt| rt.get(q).0 * 10);
            assert_eq!(rt.get(d), 10);

            let set_gate = Gate::new();
            park_at(&rt, vec![(ParkPoint::SetAfterBump, set_gate.clone())]);
            let setter = {
                let rt = Arc::clone(&rt);
                thread::spawn(move || rt.set(late, 6))
            };
            set_gate.wait_parked();

            rt.set(trigger, 1);
            rt.set(x, 2);
            arm(1);
            let err = catch_unwind(AssertUnwindSafe(|| rt.get(q))).unwrap_err();
            assert!(panic_message(err).contains("drop of Boom(1)"));
            assert_eq!(rt.state_of(q), NodeState::Dirty);
            set_gate.release();
            setter.join().unwrap();

            assert_eq!(rt.get(d), 20);
            assert_eq!(rt.state_of(q), NodeState::Clean);
            rt.assert_invariant_i();
        },
    )
    .assert_clean();
}
