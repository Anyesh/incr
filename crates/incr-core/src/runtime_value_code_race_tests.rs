//! Value code reading nodes that another thread is setting, computing or
//! deleting: the `PartialEq` of a value type reads through `get`, which
//! takes no lock on a Clean node and never waits or computes for value
//! code. Each `Shared` test parks one thread at a named point and reads
//! from another while it is parked.

use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use super::race_tests::{clean_node_that_can_turn_cyclic, id, panic_message, park_at, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::handle::Incr;
use crate::value_code::{self, CONTRACT_MESSAGE};

type Rt = Runtime<Shared>;

const DELETED_MESSAGE: &str = "was deleted while a read was reaching it";

thread_local! {
    /// What [`Probe`]'s `PartialEq` runs first on this thread, if anything.
    static ON_EQ: RefCell<Option<Box<dyn Fn()>>> = const { RefCell::new(None) };
}

/// A value whose `PartialEq` runs this thread's [`ON_EQ`] first.
#[derive(Clone, Debug)]
struct Probe(u64);

impl PartialEq for Probe {
    fn eq(&self, other: &Self) -> bool {
        ON_EQ.with(|hook| {
            if let Some(hook) = hook.borrow().as_ref() {
                hook();
            }
        });
        self.0 == other.0
    }
}

/// Runs `f` with `hook` as this thread's [`ON_EQ`], which is cleared on
/// every exit.
fn with_eq_hook<R>(hook: impl Fn() + 'static, f: impl FnOnce() -> R) -> R {
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            ON_EQ.with(|h| h.borrow_mut().take());
        }
    }
    ON_EQ.with(|h| *h.borrow_mut() = Some(Box::new(hook)));
    let _clear = Clear;
    f()
}

/// On a new thread, `set(x, Probe(1))` with a `PartialEq` that reads `y`.
/// Returns what the read saw, or the panic message of the set.
fn set_reading<T: crate::Value>(
    rt: &Arc<Rt>,
    x: Incr<Probe>,
    y: Incr<T>,
) -> thread::JoinHandle<Result<Vec<T>, String>> {
    let rt = Arc::clone(rt);
    thread::spawn(move || {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let hook = {
            let rt = Arc::clone(&rt);
            let seen = Arc::clone(&seen);
            move || {
                let v = rt.get(y);
                seen.lock().unwrap().push(v);
            }
        };
        let outcome = with_eq_hook(hook, || {
            catch_unwind(AssertUnwindSafe(|| rt.set(x, Probe(1))))
        });
        assert_eq!(value_code::thread_word(), 0, "the value-code count leaked");
        match outcome {
            Ok(()) => Ok(std::mem::take(&mut *seen.lock().unwrap())),
            Err(payload) => Err(panic_message(payload)),
        }
    })
}

/// A set's claim leaves its input Clean, so value code on another thread
/// reads it with no wait: here after the write, while the set is parked
/// before its landing.
#[test]
fn partial_eq_reads_an_input_another_thread_is_setting() {
    let rt = Arc::new(Rt::new());
    let y = rt.create_input(1_u64);
    let x = rt.create_input(Probe(0));
    let parked = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterBump, parked.clone())]);
    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(y, 2))
    };
    parked.wait_parked();

    let seen = set_reading(&rt, x, y).join().unwrap().unwrap();
    assert_eq!(seen, vec![2], "the write precedes the bump");
    assert_eq!(rt.get(x).0, 1);

    parked.release();
    setter.join().unwrap();
    assert_eq!(rt.get(y), 2);
    rt.assert_invariant_i();
}

/// A query another thread is computing is not Clean, and value code may
/// not wait for it, so the read panics with the contract message and the
/// set that ran the `PartialEq` changes nothing.
#[test]
fn partial_eq_reading_a_query_another_thread_computes_panics() {
    let rt = Arc::new(Rt::new());
    let computing = Gate::new();
    let q = {
        let computing = computing.clone();
        rt.create_query(move |_: &Rt| {
            computing.park();
            7_u64
        })
    };
    let x = rt.create_input(Probe(0));
    computing.arm();
    let computer = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(q))
    };
    computing.wait_parked();

    let message = set_reading(&rt, x, q).join().unwrap().unwrap_err();
    assert!(message.contains(CONTRACT_MESSAGE), "{message}");
    assert_eq!(rt.get(x).0, 0);

    computing.release();
    assert_eq!(computer.join().unwrap(), 7);
    rt.set(x, Probe(2));
    assert_eq!(rt.get(x).0, 2);
    rt.assert_invariant_i();
}

/// The fast path missed on a query another thread was computing, which
/// lands before the value-code arm loads the state again: the arm finds
/// it Clean and reads the new value.
#[test]
fn value_code_read_that_misses_the_fast_path_reads_a_node_now_clean() {
    let rt = Arc::new(Rt::new());
    let src = rt.create_input(3_u64);
    let computing = Gate::new();
    let q = {
        let computing = computing.clone();
        rt.create_query(move |rt: &Rt| {
            let v = rt.get(src);
            computing.park();
            v * 2
        })
    };
    let x = rt.create_input(Probe(0));
    computing.arm();
    let computer = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(q))
    };
    computing.wait_parked();

    let reload = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::ValueCodeBeforeReload(id(q)), reload.clone())],
    );
    let reader = set_reading(&rt, x, q);
    reload.wait_parked();
    computing.release();
    assert_eq!(computer.join().unwrap(), 6);
    reload.release();
    assert_eq!(reader.join().unwrap().unwrap(), vec![6]);
    assert_eq!(rt.get(x).0, 1);
    rt.assert_invariant_i();
}

/// A delete parked after its `Dead` claim, before the generation bump:
/// the read gets the panic any reader of a node being deleted gets, not
/// the contract panic or the stale one, and the set it ran under keeps
/// its old value. `strict` parks `delete_node`'s path instead of
/// `delete_with_dependents`'s.
fn value_code_read_of_a_node_being_deleted(strict: bool) {
    let rt = Arc::new(Rt::new());
    let y = rt.create_input(1_u64);
    let x = rt.create_input(Probe(0));
    let claimed = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::DeleteAfterCheck(id(y)), claimed.clone())],
    );
    let deleter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || {
            if strict {
                rt.delete_node(y);
            } else {
                rt.delete_with_dependents(y);
            }
        })
    };
    claimed.wait_parked();

    let message = set_reading(&rt, x, y).join().unwrap().unwrap_err();
    assert!(message.contains(DELETED_MESSAGE), "{message}");
    assert!(!message.contains(CONTRACT_MESSAGE), "{message}");
    assert!(!message.contains("stale"), "{message}");
    assert_eq!(rt.get(x).0, 0);

    claimed.release();
    deleter.join().unwrap();
    assert!(!rt.is_alive(y));
    rt.set(x, Probe(2));
    assert_eq!(rt.get(x).0, 2);
    rt.assert_invariant_i();
}

#[test]
fn value_code_read_of_a_node_being_deleted_panics_deleted() {
    value_code_read_of_a_node_being_deleted(false);
}

#[test]
fn value_code_read_of_a_node_a_strict_delete_holds_panics_deleted() {
    value_code_read_of_a_node_being_deleted(true);
}

thread_local! {
    static LOCAL_RT: RefCell<Option<Rc<Runtime<Local>>>> = const { RefCell::new(None) };
}

/// Under Local only this thread can be mid-delete, so value code that
/// reads the node is value code called from inside the delete: the read
/// panics with the contract message. The park hook reaches the runtime
/// through a thread-local, since the hook must be `Send + Sync`, and
/// catches the panic there, so the delete then completes.
#[test]
fn local_value_code_read_of_a_node_this_thread_deletes_panics_contract() {
    for strict in [false, true] {
        let rt = Rc::new(Runtime::<Local>::new());
        let y = rt.create_input(1_u64);
        let x = rt.create_input(Probe(0));
        LOCAL_RT.with(|r| *r.borrow_mut() = Some(Rc::clone(&rt)));
        let message: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let recorded = Arc::clone(&message);
        let installed = rt.park_hook.set(Box::new(move |point| {
            if point != ParkPoint::DeleteAfterCheck(id(y)) {
                return;
            }
            let rt = LOCAL_RT.with(|r| r.borrow().clone()).unwrap();
            let reader = Rc::clone(&rt);
            let err = with_eq_hook(
                move || {
                    reader.get(y);
                },
                || catch_unwind(AssertUnwindSafe(|| rt.set(x, Probe(1)))),
            )
            .expect_err("the read of a node this thread is deleting was allowed");
            *recorded.lock().unwrap() = Some(panic_message(err));
        }));
        assert!(installed.is_ok());

        if strict {
            rt.delete_node(y);
        } else {
            rt.delete_with_dependents(y);
        }
        let message = message.lock().unwrap().take().expect("the hook never ran");
        assert!(message.contains(CONTRACT_MESSAGE), "{message}");
        assert!(!rt.is_alive(y));
        assert_eq!(rt.get(x).0, 0);
        rt.set(x, Probe(2));
        assert_eq!(rt.get(x).0, 2);
        LOCAL_RT.with(|r| r.borrow_mut().take());
    }
}

/// A `PartialEq` may read the input its own `set` holds: the input stays
/// Clean under the claim, so the read returns the value being replaced.
fn partial_eq_reads_its_own_input_under_its_set<C: Cells>(rt: Rc<Runtime<C>>) {
    let x = rt.create_input(Probe(1));
    let seen = Rc::new(RefCell::new(Vec::new()));
    let hook = {
        let rt = Rc::clone(&rt);
        let seen = Rc::clone(&seen);
        move || seen.borrow_mut().push(rt.get(x).0)
    };
    with_eq_hook(hook, || rt.set(x, Probe(2)));
    assert_eq!(*seen.borrow(), vec![1]);
    assert_eq!(rt.get(x).0, 2);
}

#[test]
fn local_partial_eq_reads_its_own_input_under_its_set() {
    partial_eq_reads_its_own_input_under_its_set(Rc::new(Runtime::<Local>::new()));
}

#[test]
fn shared_partial_eq_reads_its_own_input_under_its_set() {
    partial_eq_reads_its_own_input_under_its_set(Rc::new(Runtime::<Shared>::new()));
}

/// `Q = Probe(x)`, counting its runs.
fn probe_query(rt: &Rt, x: Incr<u64>) -> (Incr<Probe>, Arc<AtomicUsize>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let q = {
        let runs = Arc::clone(&runs);
        rt.create_query(move |rt: &Rt| {
            runs.fetch_add(1, Ordering::SeqCst);
            Probe(rt.get(x))
        })
    };
    (q, runs)
}

/// A `PartialEq` that runs inside Q's recompute reads y while a `set(y)`
/// is parked after its write and bump. Q's previous run was its first
/// compute, which wrote without comparing, so Q has no edge to y yet:
/// `publish_deps` finds y's change unsettled and Q lands Dirty, so the
/// next read of Q runs it again against y's new value.
#[test]
fn compute_frame_read_of_a_new_edge_under_a_parked_set_lands_dirty() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let y = rt.create_input(10_u64);
    let (q, runs) = probe_query(&rt, x);
    assert_eq!(rt.get(q).0, 1);
    rt.set(x, 2);

    let parked = Gate::new();
    park_at(&rt, vec![(ParkPoint::SetAfterBump, parked.clone())]);
    let setter = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.set(y, 20))
    };
    parked.wait_parked();

    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let hook = {
                let rt = Arc::clone(&rt);
                let seen = Arc::clone(&seen);
                move || seen.lock().unwrap().push(rt.get(y))
            };
            let value = with_eq_hook(hook, || rt.get(q).0);
            let seen = std::mem::take(&mut *seen.lock().unwrap());
            (value, seen)
        })
    };
    let (value, seen) = reader.join().unwrap();
    assert_eq!(value, 2);
    assert_eq!(seen, vec![20]);
    assert_eq!(runs.load(Ordering::SeqCst), 2);
    assert_ne!(
        rt.state_of(q),
        crate::state::NodeState::Clean,
        "Q landed Clean over a dep whose change was unsettled"
    );

    parked.release();
    setter.join().unwrap();
    assert_eq!(rt.get(q).0, 2);
    assert_eq!(
        runs.load(Ordering::SeqCst),
        3,
        "the next read did not rerun Q"
    );
    rt.assert_invariant_i();
}

/// An earlier recompute installed Q's edge to y through the same
/// `PartialEq` read. Q's next recompute reads y and then holds inside the
/// `PartialEq`, before its landing, while `set(y)` lands and walks: the
/// walk flags Q `ComputingDirty`, so Q lands Dirty. With `land_first`, Q
/// lands Clean before the set, and the walk moves it to Dirty. Either way
/// the next read of Q runs it again.
fn compute_frame_read_over_an_installed_edge(land_first: bool) {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let y = rt.create_input(10_u64);
    let (q, runs) = probe_query(&rt, x);
    // The hook stays on for the setup, so every recompute's `PartialEq`
    // reads y and keeps the edge.
    let reader = Arc::clone(&rt);
    with_eq_hook(
        move || {
            reader.get(y);
        },
        || {
            assert_eq!(rt.get(q).0, 1);
            rt.set(x, 2);
            assert_eq!(rt.get(q).0, 2);
            assert_eq!(runs.load(Ordering::SeqCst), 2);
            rt.set(y, 11);
            assert_eq!(rt.get(q).0, 2);
            assert_eq!(
                runs.load(Ordering::SeqCst),
                3,
                "the edge to y was not installed"
            );
            rt.set(x, 3);
        },
    );

    let holding = Gate::new();
    // Armed before the reader starts, so its `PartialEq` cannot pass the
    // gate first.
    if !land_first {
        holding.arm();
    }
    let reader = {
        let rt = Arc::clone(&rt);
        let holding = holding.clone();
        thread::spawn(move || {
            let hook = {
                let rt = Arc::clone(&rt);
                move || {
                    rt.get(y);
                    holding.park();
                }
            };
            with_eq_hook(hook, || rt.get(q).0)
        })
    };
    if land_first {
        assert_eq!(reader.join().unwrap(), 3);
        rt.set(y, 12);
    } else {
        holding.wait_parked();
        rt.set(y, 12);
        holding.release();
        assert_eq!(reader.join().unwrap(), 3);
    }
    assert_eq!(runs.load(Ordering::SeqCst), 4);
    assert_ne!(rt.state_of(q), crate::state::NodeState::Clean);
    assert_eq!(rt.get(q).0, 3);
    assert_eq!(
        runs.load(Ordering::SeqCst),
        5,
        "the next read did not rerun Q"
    );
    rt.assert_invariant_i();
}

#[test]
fn compute_frame_read_over_an_installed_edge_flagged_mid_compute() {
    compute_frame_read_over_an_installed_edge(false);
}

#[test]
fn compute_frame_read_over_an_installed_edge_dirtied_after_landing() {
    compute_frame_read_over_an_installed_edge(true);
}

static GIL: Mutex<()> = Mutex::new(());
static IN_CLONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A value whose `Clone` takes [`GIL`], as a Python value's clone takes
/// the GIL in incr-concurrent-python.
#[derive(Debug, PartialEq)]
struct NeedsGil(u64);

impl Clone for NeedsGil {
    fn clone(&self) -> Self {
        IN_CLONE.store(true, Ordering::SeqCst);
        let _gil = GIL.lock().unwrap();
        NeedsGil(self.0)
    }
}

/// Bug (f): a reader's `Clone` waits for a lock that another thread holds
/// while it creates a node of the same type. The Clean read holds no incr
/// lock across the clone, so the create, which takes the arena's lock,
/// completes, and the reader finishes once the lock is released.
#[test]
fn clone_waiting_on_a_lock_a_creator_holds_does_not_deadlock() {
    let rt = Arc::new(Rt::new());
    let a = rt.create_input(NeedsGil(1));
    let gil = GIL.lock().unwrap();
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(a).0)
    };
    while !IN_CLONE.load(Ordering::SeqCst) {
        thread::yield_now();
    }
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let creator = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || {
            let b = rt.create_input(NeedsGil(2));
            done_tx.send(()).unwrap();
            b
        })
    };
    done_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("the create waited on the reader's clone");
    drop(gil);
    assert_eq!(reader.join().unwrap(), 1);
    let b = creator.join().unwrap();
    assert_eq!(rt.get(b).0, 2);
}

/// The value-code arm reloads `a` Clean and parks before its slot read;
/// another thread's set and episode land `a` Cyclic and clear its slot.
/// The arm goes on from the state it reloads, which is not Clean, so the
/// read is the contract panic, where it used to panic that a Clean node
/// had no value.
#[test]
fn value_code_read_of_a_node_now_clean_vs_cyclic_landing() {
    let rt = Arc::new(Rt::new());
    let (g, a, b) = clean_node_that_can_turn_cyclic(&rt);
    rt.set(g, 2);
    let x = rt.create_input(Probe(0));
    let reload = Gate::new();
    let before_read = Gate::new();
    park_at(
        &rt,
        vec![
            (ParkPoint::ValueCodeBeforeReload(id(a)), reload.clone()),
            (ParkPoint::GetBeforeRead(id(a)), before_read.clone()),
        ],
    );
    let reader = set_reading(&rt, x, a);
    reload.wait_parked();

    let computer = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(a))
    };
    before_read.wait_parked();
    before_read.release();
    assert_eq!(computer.join().unwrap(), 7);
    before_read.arm();
    reload.release();
    before_read.wait_parked();

    rt.set(g, 0);
    assert!(rt.try_get(b).is_err());
    assert_eq!(rt.state_of(a), crate::state::NodeState::Cyclic);
    before_read.release();

    let msg = reader.join().unwrap().expect_err("a has no value");
    assert!(msg.contains(CONTRACT_MESSAGE), "got: {msg}");
    assert_eq!(rt.get(x).0, 0);
    rt.assert_invariant_i();
}
