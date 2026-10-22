//! Panics that follow a compute's value write, and the retire sites that
//! can run a user `Drop`. The `Shared` cases run in a child process: a
//! panicking `Drop` inside a haphazard reclaim leaks the rest of that
//! batch, and the child keeps the test binary's global domain clean.

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, Ordering::SeqCst};
use std::sync::{Arc, OnceLock};
use std::thread;

use super::race_tests::{id, park_at, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::child_test::run_in_child;
use crate::deferred::{ErasedDrop, SPILL_ALLOCS};
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
pub(super) fn prime_reclaim_timer<C: Cells>(rt: &Runtime<C>) {
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
/// whose `set` has landed but not yet recorded its walk, so the `Err`
/// path publishes Superseded and lands Dirty. Without the stamp the next recompute
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

            // Parked after the landing: a read of `late` under the set's
            // claim would wait, and the walk is still unrecorded here.
            let set_gate = Gate::new();
            park_at(&rt, vec![(ParkPoint::SetAfterWalk, set_gate.clone())]);
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
pub(super) fn retire_armed_boom<C: Cells>(rt: &Runtime<C>) {
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

            // Parked after the landing: a read of `late` under the set's
            // claim would wait, and the walk is still unrecorded here.
            let set_gate = Gate::new();
            park_at(&rt, vec![(ParkPoint::SetAfterWalk, set_gate.clone())]);
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
/// Counts its drops, as a bag item a test hook plants.
struct CountOnDrop(Arc<AtomicU32>);

impl Drop for CountOnDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, SeqCst);
    }
}

/// Q's first attempt reads a dep that is deleted under it, lands New and
/// retries; the retry's publish displaces Q's overflow list and the
/// drain's retire panics. P reaches Q through its own nested get and
/// swallows the panic, and must land Dirty, not Clean.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_stale_path_drain_panic_under_a_swallowing_parent_lands_dirty() {
    run_in_child(
        "runtime::drop_panic_tests::shared_stale_path_drain_panic_under_a_swallowing_parent_lands_dirty",
        || {
            let rt = Arc::new(Rt::new());
            prime_reclaim_timer(&rt);
            reclaim_probe::force(RetireSite::DepListRetire);
            let switch = rt.create_input(0_u8);
            let z = rt.create_input(100_i64);
            let inputs: Vec<_> = (0..8_i64).map(|v| rt.create_input(v)).collect();
            let z_gate = Gate::new();
            let attempts = Arc::new(AtomicU32::new(0));
            let q = {
                let captured = inputs.clone();
                let z_gate = z_gate.clone();
                let attempts = Arc::clone(&attempts);
                rt.create_query(move |rt| {
                    if rt.get(switch) == 0 {
                        return captured.iter().map(|&i| rt.get(i)).sum::<i64>();
                    }
                    if attempts.fetch_add(1, SeqCst) == 0 {
                        let v = rt.get(z);
                        z_gate.park();
                        v
                    } else {
                        captured[..6].iter().map(|&i| rt.get(i)).sum::<i64>()
                    }
                })
            };
            let reads_q = rt.create_input(0_u8);
            let p = rt.create_query(move |rt| {
                if rt.get(reads_q) == 0 {
                    return 0;
                }
                catch_unwind(AssertUnwindSafe(|| rt.get(q))).unwrap_or(-1)
            });
            assert_eq!(rt.get(q), 28);
            assert_eq!(rt.get(p), 0);

            retire_armed_boom(&rt);
            rt.set(switch, 1);
            rt.set(reads_q, 1);
            z_gate.arm();
            let deleter = {
                let rt = Arc::clone(&rt);
                let z_gate = z_gate.clone();
                thread::spawn(move || {
                    z_gate.wait_parked();
                    rt.delete_node(z);
                    z_gate.release();
                })
            };
            assert_eq!(rt.get(p), -1);
            deleter.join().unwrap();
            assert_eq!(attempts.load(SeqCst), 2, "the Stale attempt must retry");
            assert_eq!(ARMED.load(SeqCst), i64::MIN, "the retire did not run");
            assert_eq!(rt.state_of(q), NodeState::Clean);
            assert_eq!(rt.state_of(p), NodeState::Dirty);
            // P's -1 came from the swallowed panic, not from Q's value,
            // so the next read must recompute P over Q rather than
            // verify it Clean against a Q whose stamps did not move.
            assert_eq!(rt.get(p), 15);
            assert_eq!(rt.state_of(p), NodeState::Clean);
            rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

/// Q lands Clean and its normal-path drain retires the displaced dep
/// list, whose reclaim panics. P swallows that panic and lands Dirty;
/// its next read must recompute from Q's real value, not re-verify Clean
/// over the swallowed result.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_drain_panic_under_a_swallowing_parent_recomputes_on_the_next_read() {
    run_in_child(
        "runtime::drop_panic_tests::shared_drain_panic_under_a_swallowing_parent_recomputes_on_the_next_read",
        || {
            let rt = Rt::new();
            prime_reclaim_timer(&rt);
            reclaim_probe::force(RetireSite::DepListRetire);
            let (switch, _inputs, q) = overflow_query(&rt);
            let reads_q = rt.create_input(0_u8);
            let p = rt.create_query(move |rt| {
                if rt.get(reads_q) == 0 {
                    return 0;
                }
                catch_unwind(AssertUnwindSafe(|| rt.get(q))).unwrap_or(-1)
            });
            assert_eq!(rt.get(q), 28);
            assert_eq!(rt.get(p), 0);

            retire_armed_boom(&rt);
            rt.set(switch, 1);
            rt.set(reads_q, 1);
            assert_eq!(rt.get(p), -1);
            assert_eq!(ARMED.load(SeqCst), i64::MIN, "the retire did not run");
            assert_eq!(rt.state_of(q), NodeState::Clean);
            assert_eq!(rt.state_of(p), NodeState::Dirty);

            assert_eq!(rt.get(p), 36);
            assert_eq!(rt.state_of(p), NodeState::Clean);
            rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

/// A Stale retry carries its bag; when the retry's claim lands on the
/// slot's next occupant, the `Gone` exit drains it exactly once.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_gone_exit_after_a_recycle_drops_its_bag_item_once() {
    run_in_child(
        "runtime::drop_panic_tests::shared_gone_exit_after_a_recycle_drops_its_bag_item_once",
        || {
            let rt = Arc::new(Rt::new());
            let z = rt.create_input(1_i64);
            let z_gate = Gate::new();
            let claim_gate = Gate::new();
            let q = {
                let (z_gate, claim_gate) = (z_gate.clone(), claim_gate.clone());
                rt.create_query(move |rt| {
                    let v = rt.get(z);
                    claim_gate.arm();
                    z_gate.park();
                    v
                })
            };
            let hook_gate = claim_gate.clone();
            rt.park_hook
                .set(Box::new(move |p| {
                    if p == ParkPoint::BeforeClaim(id(q)) {
                        hook_gate.park();
                    }
                }))
                .ok()
                .expect("park hook installed twice");
            let dropped = Arc::new(AtomicU32::new(0));
            let planted = Arc::clone(&dropped);
            rt.bag_hook
                .set(Box::new(move |p| {
                    (p == ParkPoint::StaleRetry(id(q)))
                        .then(|| Box::new(CountOnDrop(Arc::clone(&planted))) as ErasedDrop)
                }))
                .ok()
                .expect("bag hook installed twice");
            z_gate.arm();
            let driver = {
                let rt = Arc::clone(&rt);
                let (z_gate, claim_gate) = (z_gate.clone(), claim_gate.clone());
                thread::spawn(move || {
                    z_gate.wait_parked();
                    rt.delete_node(z);
                    z_gate.release();
                    claim_gate.wait_parked();
                    rt.delete_node(q);
                    let occupant = rt.create_query(|_| 0_i64);
                    assert_eq!(
                        occupant.slot(),
                        q.slot(),
                        "the occupant must recycle q's slot"
                    );
                    claim_gate.release();
                    occupant
                })
            };
            let msg = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(q))).unwrap_err());
            assert!(msg.contains("stale Incr"), "got: {msg}");
            let occupant = driver.join().unwrap();
            assert_eq!(dropped.load(SeqCst), 1);
            assert_eq!(rt.state_of(occupant), NodeState::New);
            assert_eq!(rt.get(occupant), 0);
            rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

/// A single-node delete uses the bag's typed slots and never touches the
/// spill; an owned-tree delete of 100 groups is one batch, which grows
/// the spill at most once.
#[test]
fn local_delete_spill_grows_at_most_once_per_batch() {
    use crate::collection::IncrCollection;
    let rt = Runtime::<Local>::new();
    let col: IncrCollection<i64, Local> = rt.create_collection();
    for i in 0..100_i64 {
        col.insert(&rt, i);
    }
    let router = col.group_by(&rt, |x| *x).version_node();
    let _ = rt.get(router);
    let lone = rt.create_input(0_i64);

    SPILL_ALLOCS.with(|c| c.set(0));
    rt.delete_node(lone);
    assert_eq!(SPILL_ALLOCS.with(Cell::get), 0);

    rt.delete_node(router);
    let growths = SPILL_ALLOCS.with(Cell::get);
    assert!((1..=2).contains(&growths), "spill grew {growths} times");
    assert!(!rt.is_alive(router));
}

/// A value holding a probe `Arc`, so a test can count the values still
/// alive in their arena slots.
#[derive(Clone)]
struct Held(Arc<()>);

impl PartialEq for Held {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// An owned-tree delete is a batch: the first member's drops take the
/// bag's typed slot and the rest spill. Every member's value leaves its
/// arena slot at the drain, and the slots are reused.
#[test]
fn local_owned_tree_delete_releases_every_member_value() {
    let rt = Runtime::<Local>::new();
    let probe = Arc::new(());
    let owner = rt.create_input(Held(Arc::clone(&probe)));
    for _ in 0..3 {
        let child = rt.create_input(Held(Arc::clone(&probe)));
        rt.adopt(owner.erase(), child.erase());
    }
    assert_eq!(Arc::strong_count(&probe), 5);
    let tag = rt.nodes.get(owner.slot()).type_tag();
    let arena_len = || rt.arenas.tagged_arena::<Held>(tag).unwrap().len();
    let slots = arena_len();
    rt.delete_node(owner);
    assert_eq!(
        Arc::strong_count(&probe),
        1,
        "a member's value was never released"
    );
    for _ in 0..4 {
        rt.create_input(Held(Arc::clone(&probe)));
    }
    assert_eq!(
        arena_len(),
        slots,
        "a released member's arena slot was not reused"
    );
}

struct RtPtr(*const Runtime<Local>);
// SAFETY: test-only. The runtime outlives the observers holding its
// pointer, and a Local runtime's pointer never leaves its thread.
unsafe impl Send for RtPtr {}
unsafe impl Sync for RtPtr {}

impl RtPtr {
    fn rt(&self) -> &Runtime<Local> {
        // SAFETY: see the impls above.
        unsafe { &*self.0 }
    }
}

/// A router over three groups, all Clean, whose version nodes it owns.
fn local_grouped(rt: &Runtime<Local>) -> (Incr<u64>, Vec<Incr<u64>>) {
    use crate::collection::IncrCollection;
    let c: IncrCollection<i64, Local> = rt.create_collection();
    for v in [3, 4, 5] {
        c.insert(rt, v);
    }
    let groups = c.group_by(rt, |x| x % 3);
    let _ = rt.get(groups.version_node());
    let versions = groups
        .keys()
        .iter()
        .map(|k| groups.get_group(k).unwrap().version_node())
        .collect();
    (groups.version_node(), versions)
}

/// A member's `Drop` deletes the root and a sibling of the owned tree it
/// is deleted with. Every member's generation is bumped before any
/// `Drop` runs, so both deletes panic as stale instead of waiting on a
/// claim this thread holds, which under Local would never end.
#[test]
#[cfg_attr(miri, ignore)]
fn local_owned_tree_drop_that_deletes_the_root_panics_stale() {
    run_in_child(
        "runtime::drop_panic_tests::local_owned_tree_drop_that_deletes_the_root_panics_stale",
        || {
            let rt = Runtime::<Local>::new();
            let (router, versions) = local_grouped(&rt);
            let rt_ptr = RtPtr(&rt);
            let sibling = versions[1];
            let deleter = OnDrop::armed(1, move || {
                let rt = rt_ptr.rt();
                let root = catch_unwind(AssertUnwindSafe(|| rt.delete_node(router))).unwrap_err();
                assert!(panic_message(root).contains("stale Incr"));
                rt.delete_node(sibling);
            });
            rt.observe(versions[0], move |_| {
                let _keep = &deleter;
            });
            rt.delete_node(router);
        },
    )
    .assert_panicked_with("stale Incr");
}

/// A strict delete of an owned tree whose first version-node observer
/// has a panicking `Drop`: the panic arrives after the whole tree is
/// deleted, and the other observers still drop.
#[test]
#[cfg_attr(miri, ignore)]
fn local_owned_tree_observer_drop_panic_arrives_after_the_delete() {
    run_in_child(
        "runtime::drop_panic_tests::local_owned_tree_observer_drop_panic_arrives_after_the_delete",
        || {
            let rt = Runtime::<Local>::new();
            let (router, versions) = local_grouped(&rt);
            let rt_ptr = RtPtr(&rt);
            let members = versions.clone();
            let boom = OnDrop::armed(1, move || {
                let rt = rt_ptr.rt();
                assert!(!rt.is_alive(router));
                assert!(members.iter().all(|&v| !rt.is_alive(v)));
                panic!("observer drop panicked on purpose");
            });
            rt.observe(versions[0], move |_| {
                let _keep = &boom;
            });
            let dropped = Arc::new(AtomicU32::new(0));
            for &v in &versions[1..] {
                let counter = CountOnDrop(Arc::clone(&dropped));
                rt.observe(v, move |_| {
                    let _keep = &counter;
                });
            }

            let err = catch_unwind(AssertUnwindSafe(|| rt.delete_node(router))).unwrap_err();
            assert!(panic_message(err).contains("observer drop panicked on purpose"));
            assert_eq!(dropped.load(SeqCst), versions.len() as u32 - 1);
            assert_eq!(rt.node_count(), 1, "only the source is left");
            rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

/// Under Local no other thread can finish a claim, so a delete that finds
/// a node Dead at its own generation panics instead of spinning for the
/// bump it is itself waiting to do.
#[test]
#[cfg_attr(miri, ignore)]
fn local_delete_of_a_node_claimed_dead_panics_instead_of_spinning() {
    run_in_child(
        "runtime::drop_panic_tests::local_delete_of_a_node_claimed_dead_panics_instead_of_spinning",
        || {
            let rt = Runtime::<Local>::new();
            let x = rt.create_input(1_i64);
            rt.nodes.get(x.slot()).set_state(NodeState::Dead.as_u8());
            rt.delete_node(x);
        },
    )
    .assert_panicked_with("claimed by a delete on this thread");
}

/// A contained payload whose `Drop` panics, released by a set's dirty walk,
/// must not stop that walk before it flags the input's other dependents.
fn ticket_drop_panic_keeps_the_set_walk<C: Cells>(rt: &Runtime<C>) {
    let i = rt.create_input(0_i64);
    let sel = rt.create_input(0_i64);
    let e = rt.create_query(move |rt| {
        let v = rt.get(i);
        if v == 1 {
            std::panic::panic_any(Boom(7));
        }
        v
    });
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(e) } else { 100 });
    let k = rt.create_query(move |rt| rt.get(i) * 10);
    assert_eq!(rt.get(k), 0);
    assert_eq!(rt.get(x), 0);
    rt.set(i, 1);
    rt.set(sel, 1);
    assert_eq!(rt.get(x), 100);
    assert_eq!(rt.contained_payloads(), 1);
    assert_eq!(rt.get(k), 10);
    arm(7);
    let err = catch_unwind(AssertUnwindSafe(|| rt.set(i, 2))).unwrap_err();
    arm(i64::MIN);
    assert!(panic_message(err).contains("drop of Boom(7)"));
    assert_eq!(rt.get(k), 20);
    rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_ticket_drop_panic_keeps_the_set_walk() {
    run_in_child(
        "runtime::drop_panic_tests::local_ticket_drop_panic_keeps_the_set_walk",
        || ticket_drop_panic_keeps_the_set_walk(&Runtime::<Local>::new()),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_ticket_drop_panic_keeps_the_set_walk() {
    run_in_child(
        "runtime::drop_panic_tests::shared_ticket_drop_panic_keeps_the_set_walk",
        || ticket_drop_panic_keeps_the_set_walk(&Runtime::<Shared>::new()),
    )
    .assert_clean();
}

/// A deleted node's contained payload whose `Drop` panics must not stop
/// the delete releasing the node's own value.
fn ticket_drop_panic_keeps_the_delete<C: Cells>(rt: &Runtime<C>) {
    let probe = Arc::new(());
    let i = rt.create_input(0_i64);
    let sel = rt.create_input(0_i64);
    let p = probe.clone();
    let e = rt.create_query(move |rt| {
        if rt.get(i) == 1 {
            std::panic::panic_any(Boom(8));
        }
        Held(p.clone())
    });
    let x = rt.create_query(move |rt| {
        if rt.get(sel) == 0 {
            let _ = rt.get(e);
            0
        } else {
            100
        }
    });
    assert_eq!(rt.get(x), 0);
    rt.set(i, 1);
    rt.set(sel, 1);
    assert_eq!(rt.get(x), 100);
    assert_eq!(rt.contained_payloads(), 1);
    arm(8);
    let err = catch_unwind(AssertUnwindSafe(|| rt.delete_node(e))).unwrap_err();
    arm(i64::MIN);
    assert!(panic_message(err).contains("drop of Boom(8)"));
    assert_eq!(
        Arc::strong_count(&probe),
        1,
        "the deleted node's value stayed"
    );
    assert_eq!(rt.contained_payloads(), 0);
    assert_eq!(rt.get(x), 100);
    rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_ticket_drop_panic_keeps_the_delete() {
    run_in_child(
        "runtime::drop_panic_tests::local_ticket_drop_panic_keeps_the_delete",
        || ticket_drop_panic_keeps_the_delete(&Runtime::<Local>::new()),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_ticket_drop_panic_keeps_the_delete() {
    run_in_child(
        "runtime::drop_panic_tests::shared_ticket_drop_panic_keeps_the_delete",
        || {
            let rt = Runtime::<Shared>::new();
            prime_reclaim_timer(&rt);
            reclaim_probe::force(RetireSite::DeleteRetire);
            ticket_drop_panic_keeps_the_delete(&rt);
        },
    )
    .assert_clean();
}

/// A compute the walk contains, flagged by its own `set` so it lands Dirty
/// past the walk, releases the ticket it just took: that payload's
/// panicking `Drop` leaves the read once the landing is done.
fn ticket_drop_panic_keeps_a_flagged_landing<C: Cells>(rt: &Runtime<C>) {
    let i = rt.create_input(0_i64);
    let j = rt.create_input(0_i64);
    let sel = rt.create_input(0_i64);
    let e = rt.create_query(move |rt| {
        let a = rt.get(j);
        if rt.get(i) == 1 && a == 0 {
            rt.set(j, 1);
            std::panic::panic_any(Boom(9));
        }
        a
    });
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(e) } else { 100 });
    assert_eq!(rt.get(x), 0);
    rt.set(i, 1);
    rt.set(sel, 1);
    arm(9);
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_err();
    arm(i64::MIN);
    assert!(panic_message(err).contains("drop of Boom(9)"));
    rt.assert_invariant_i();
    assert_eq!(rt.contained_payloads(), 0);
    assert_eq!(rt.get(e), 1);
    rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_ticket_drop_panic_keeps_a_flagged_landing() {
    run_in_child(
        "runtime::drop_panic_tests::local_ticket_drop_panic_keeps_a_flagged_landing",
        || ticket_drop_panic_keeps_a_flagged_landing(&Runtime::<Local>::new()),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_ticket_drop_panic_keeps_a_flagged_landing() {
    run_in_child(
        "runtime::drop_panic_tests::shared_ticket_drop_panic_keeps_a_flagged_landing",
        || ticket_drop_panic_keeps_a_flagged_landing(&Runtime::<Shared>::new()),
    )
    .assert_clean();
}

/// Two contained payloads released by one set's walk: the first one's
/// panicking `Drop` neither stops the walk nor keeps the second.
fn ticket_drop_panic_keeps_a_walk_holding_two<C: Cells>(rt: &Runtime<C>) {
    let i = rt.create_input(0_i64);
    let sel = rt.create_input(0_i64);
    let e1 = rt.create_query(move |rt| {
        let v = rt.get(i);
        if v == 1 {
            std::panic::panic_any(Boom(10));
        }
        v
    });
    let e2 = rt.create_query(move |rt| {
        let v = rt.get(i);
        if v == 1 {
            std::panic::panic_any(Boom(11));
        }
        v
    });
    let x = rt.create_query(move |rt| {
        if rt.get(sel) == 0 {
            rt.get(e1) + rt.get(e2)
        } else {
            100
        }
    });
    let k = rt.create_query(move |rt| rt.get(i) * 10);
    assert_eq!(rt.get(k), 0);
    assert_eq!(rt.get(x), 0);
    rt.set(i, 1);
    rt.set(sel, 1);
    assert_eq!(rt.get(x), 100);
    assert_eq!(rt.contained_payloads(), 2);
    assert_eq!(rt.get(k), 10);
    arm(10);
    let err = catch_unwind(AssertUnwindSafe(|| rt.set(i, 2))).unwrap_err();
    arm(i64::MIN);
    assert!(panic_message(err).contains("drop of Boom(10)"));
    assert_eq!(rt.contained_payloads(), 0);
    assert_eq!(rt.get(k), 20);
    assert_eq!(rt.get(e1) + rt.get(e2), 4);
    rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_ticket_drop_panic_keeps_a_walk_holding_two() {
    run_in_child(
        "runtime::drop_panic_tests::local_ticket_drop_panic_keeps_a_walk_holding_two",
        || ticket_drop_panic_keeps_a_walk_holding_two(&Runtime::<Local>::new()),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_ticket_drop_panic_keeps_a_walk_holding_two() {
    run_in_child(
        "runtime::drop_panic_tests::shared_ticket_drop_panic_keeps_a_walk_holding_two",
        || ticket_drop_panic_keeps_a_walk_holding_two(&Runtime::<Shared>::new()),
    )
    .assert_clean();
}

/// A delete whose own value's `Drop` panics still drops the tickets it
/// buried, so no later, unrelated delete raises their payloads.
fn delete_drop_panic_drops_its_buried_tickets<C: Cells>(rt: &Runtime<C>) {
    let i = rt.create_input(0_i64);
    let sel = rt.create_input(0_i64);
    let e = rt.create_query(move |rt| {
        if rt.get(i) == 1 {
            std::panic::panic_any(Boom(22));
        }
        Boom(21)
    });
    let x = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(e).0 } else { 100 });
    assert_eq!(rt.get(x), 21);
    rt.set(i, 1);
    rt.set(sel, 1);
    assert_eq!(rt.get(x), 100);
    assert_eq!(rt.contained_payloads(), 1);
    arm(21);
    let err = catch_unwind(AssertUnwindSafe(|| rt.delete_node(e))).unwrap_err();
    arm(i64::MIN);
    assert!(panic_message(err).contains("drop of Boom(21)"));
    assert_eq!(rt.contained_payloads(), 0);
    rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_delete_drop_panic_drops_its_buried_tickets() {
    run_in_child(
        "runtime::drop_panic_tests::local_delete_drop_panic_drops_its_buried_tickets",
        || delete_drop_panic_drops_its_buried_tickets(&Runtime::<Local>::new()),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_delete_drop_panic_drops_its_buried_tickets() {
    run_in_child(
        "runtime::drop_panic_tests::shared_delete_drop_panic_drops_its_buried_tickets",
        || {
            let rt = Runtime::<Shared>::new();
            prime_reclaim_timer(&rt);
            reclaim_probe::force(RetireSite::DeleteRetire);
            delete_drop_panic_drops_its_buried_tickets(&rt);
        },
    )
    .assert_clean();
}
