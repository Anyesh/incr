//! Panics that follow a compute's value write, and the retire sites that
//! can run a user `Drop`. The `Shared` cases run in a child process: a
//! panicking `Drop` inside a haphazard reclaim leaks the rest of that
//! batch, and the child keeps the test binary's global domain clean.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering::SeqCst};
use std::sync::{Arc, OnceLock};
use std::thread;

use super::race_tests::{park_at, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::child_test::run_in_child;
use crate::handle::Incr;
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
/// A value whose `Drop` runs `action` once, when armed. Retired values
/// are dropped by whichever retire reclaims them, so `action` runs at
/// the retire site a test forces.
#[derive(Clone)]
struct OnDrop {
    id: u8,
    armed: Arc<AtomicBool>,
    action: Arc<dyn Fn() + Send + Sync>,
}

impl OnDrop {
    fn armed(id: u8, action: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            id,
            armed: Arc::new(AtomicBool::new(true)),
            action: Arc::new(action),
        }
    }

    fn inert(id: u8) -> Self {
        Self {
            id,
            armed: Arc::new(AtomicBool::new(false)),
            action: Arc::new(|| {}),
        }
    }
}

impl PartialEq for OnDrop {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Drop for OnDrop {
    fn drop(&mut self) {
        if self.armed.swap(false, SeqCst) {
            (self.action)();
        }
    }
}

/// A `Boom(1)` sits retired in the global domain, armed, so the next
/// forced reclaim runs its panicking `Drop`.
fn retire_armed_boom<C: Cells>(rt: &Runtime<C>) {
    let pending = rt.create_input(Boom(1));
    rt.set(pending, Boom(2));
    arm(1);
}

type Rt = Runtime<Shared>;

/// A query over `inputs[0..8]` when `switch` is 0 and `inputs[1..9]`
/// otherwise, so a switch flip displaces one overflow dep list for
/// another at its publish.
fn overflow_query(rt: &Rt) -> (Incr<u8>, Vec<Incr<i64>>, Incr<i64>) {
    let switch = rt.create_input(0_u8);
    let inputs: Vec<_> = (0..9_i64).map(|v| rt.create_input(v)).collect();
    let captured = inputs.clone();
    let q = rt.create_query(move |rt| {
        let start = rt.get(switch) as usize;
        captured[start..start + 8].iter().map(|&i| rt.get(i)).sum()
    });
    (switch, inputs, q)
}

/// At a dep-list retire after an overflow install, the node has landed
/// and its reverse edges are complete before the panic leaves `get`.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_dep_list_retire_runs_after_the_landing() {
    run_in_child(
        "runtime::drop_panic_tests::shared_dep_list_retire_runs_after_the_landing",
        || {
            let rt = Rt::new();
            prime_reclaim_timer(&rt);
            reclaim_probe::force(RetireSite::DepListRetire);
            let (switch, inputs, q) = overflow_query(&rt);
            assert_eq!(rt.get(q), 28);

            retire_armed_boom(&rt);
            rt.set(switch, 1);
            let err = catch_unwind(AssertUnwindSafe(|| rt.get(q))).unwrap_err();
            assert!(panic_message(err).contains("drop of Boom(1)"));

            assert_eq!(rt.state_of(q), NodeState::Clean);
            assert!(rt.has_reverse_edge(inputs[8], q));
            assert!(!rt.has_reverse_edge(inputs[0], q));
            rt.assert_edges_complete();
            rt.set(inputs[8], 100);
            assert_eq!(rt.state_of(q), NodeState::Dirty);
            assert_eq!(rt.get(q), 36 - 8 + 100);
        },
    )
    .assert_clean();
}

/// On the `Err` path the displaced list is retired in carried mode: the
/// drop panic inside the retire is swallowed and the closure's own
/// panic arrives, with the union of deps published.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_err_path_retire_keeps_the_original_panic() {
    run_in_child(
        "runtime::drop_panic_tests::shared_err_path_retire_keeps_the_original_panic",
        || {
            let rt = Rt::new();
            prime_reclaim_timer(&rt);
            reclaim_probe::force(RetireSite::DepListRetire);
            let switch = rt.create_input(0_u8);
            let inputs: Vec<_> = (0..9_i64).map(|v| rt.create_input(v)).collect();
            let captured = inputs.clone();
            let q = rt.create_query(move |rt| {
                let flipped = rt.get(switch) != 0;
                let sum: i64 = captured[..8].iter().map(|&i| rt.get(i)).sum();
                if flipped {
                    let _ = rt.get(captured[8]);
                    panic!("user panic 4242");
                }
                sum
            });
            assert_eq!(rt.get(q), 28);

            retire_armed_boom(&rt);
            rt.set(switch, 1);
            let msg = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(q))).unwrap_err());
            assert!(msg.contains("user panic 4242"), "got: {msg}");
            assert!(!msg.contains("Boom"), "got: {msg}");
            assert_eq!(ARMED.load(SeqCst), i64::MIN, "the retire did not run");

            assert_eq!(rt.state_of(q), NodeState::Failed);
            assert!(rt.has_reverse_edge(inputs[8], q));
            rt.assert_edges_complete();
        },
    )
    .assert_clean();
}

/// A node that lands Superseded poisons its reader before its drain
/// retires the displaced list, so a reader that swallows the drop panic
/// still lands Dirty.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_swallowed_drop_panic_over_a_superseded_node_lands_dirty() {
    run_in_child(
        "runtime::drop_panic_tests::shared_swallowed_drop_panic_over_a_superseded_node_lands_dirty",
        || {
            let rt = Arc::new(Rt::new());
            prime_reclaim_timer(&rt);
            reclaim_probe::force(RetireSite::DepListRetire);
            let switch = rt.create_input(0_u8);
            let late = rt.create_input(5_i64);
            let inputs: Vec<_> = (0..9_i64).map(|v| rt.create_input(v)).collect();
            let captured = inputs.clone();
            let q = rt.create_query(move |rt| {
                let start = rt.get(switch) as usize;
                let sum: i64 = captured[start..start + 8].iter().map(|&i| rt.get(i)).sum();
                if start != 0 {
                    sum + rt.get(late)
                } else {
                    sum
                }
            });
            // P reaches Q only through its own nested get, so Q's drain
            // panics inside P's closure rather than in the walk.
            let reads_q = rt.create_input(0_u8);
            let p = rt.create_query(move |rt| {
                if rt.get(reads_q) == 0 {
                    return 0;
                }
                catch_unwind(AssertUnwindSafe(|| rt.get(q))).unwrap_or(-1)
            });
            assert_eq!(rt.get(q), 28);
            assert_eq!(rt.get(p), 0);

            let set_gate = Gate::new();
            park_at(&rt, vec![(ParkPoint::SetAfterBump, set_gate.clone())]);
            let setter = {
                let rt = Arc::clone(&rt);
                thread::spawn(move || rt.set(late, 6))
            };
            set_gate.wait_parked();

            retire_armed_boom(&rt);
            rt.set(switch, 1);
            rt.set(reads_q, 1);
            assert_eq!(rt.get(p), -1);
            assert_eq!(ARMED.load(SeqCst), i64::MIN, "the retire did not run");
            assert_eq!(rt.state_of(q), NodeState::Dirty);
            assert_eq!(rt.state_of(p), NodeState::Dirty);

            set_gate.release();
            setter.join().unwrap();
            assert_eq!(rt.get(p), 36 + 6);
            rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

/// A delete retires its displaced dep list after the generation bump, so
/// a pending `Drop` that deletes the node again panics as stale instead
/// of waiting for a bump this thread has yet to do.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_delete_retire_runs_after_the_generation_bump() {
    run_in_child(
        "runtime::drop_panic_tests::shared_delete_retire_runs_after_the_generation_bump",
        || {
            let rt = Arc::new(Rt::new());
            prime_reclaim_timer(&rt);
            reclaim_probe::force(RetireSite::DeleteRetire);
            let (_, _, q) = overflow_query(&rt);
            assert_eq!(rt.get(q), 28);

            let again = {
                let rt = Arc::clone(&rt);
                OnDrop::armed(1, move || rt.delete_node(q))
            };
            let pending = rt.create_input(again);
            rt.set(pending, OnDrop::inert(2));

            let msg =
                panic_message(catch_unwind(AssertUnwindSafe(|| rt.delete_node(q))).unwrap_err());
            assert!(msg.contains("stale Incr"), "got: {msg}");
            assert!(!rt.is_alive(q));
            rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

/// A pending `Drop` at the delete's retire creates a node, which recycles
/// the freed slot; the recycled node's state and deps are intact.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_delete_retire_drop_can_recycle_the_freed_slot() {
    run_in_child(
        "runtime::drop_panic_tests::shared_delete_retire_drop_can_recycle_the_freed_slot",
        || {
            let rt = Arc::new(Rt::new());
            prime_reclaim_timer(&rt);
            reclaim_probe::force(RetireSite::DeleteRetire);
            let (_, _, q) = overflow_query(&rt);
            assert_eq!(rt.get(q), 28);

            let recycled: Arc<OnceLock<Incr<i64>>> = Arc::default();
            let creator = {
                let rt = Arc::clone(&rt);
                let recycled = Arc::clone(&recycled);
                OnDrop::armed(1, move || {
                    recycled.set(rt.create_input(7_i64)).unwrap();
                })
            };
            let pending = rt.create_input(creator);
            rt.set(pending, OnDrop::inert(2));

            rt.delete_node(q);
            let n = *recycled.get().expect("the retire did not run");
            assert_eq!(
                n.slot(),
                q.slot(),
                "the new node must recycle the freed slot"
            );
            assert!(rt.is_alive(n));
            assert_eq!(rt.state_of(n), NodeState::Clean);
            assert_eq!(rt.get(n), 7);
            let reader = rt.create_query(move |rt| rt.get(n) * 2);
            assert_eq!(rt.get(reader), 14);
            rt.set(n, 8);
            assert_eq!(rt.get(reader), 16);
            rt.assert_invariant_i();
        },
    )
    .assert_clean();
}
