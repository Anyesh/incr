//! Wound-wait under `Shared`: claim waits that close a waits-for cycle
//! across threads wound the younger thread, which releases its claims and
//! retries. Each test runs in a child process, because a regression hangs
//! and because the wound counter is process-wide.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use super::race_tests::{id, park_at, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::Shared;
use crate::handle::Incr;
use crate::waits;

/// Asserts the cycle was resolved by wounds. Usually one: a retry that
/// takes its released claims back before the older thread does, which
/// `await_winner`'s head start makes rare but cannot rule out under load,
/// is wounded again over the same cycle, which costs a retry and never
/// changes a value.
pub(crate) fn assert_wounded() {
    assert!(waits::WOUNDS.load(Ordering::Relaxed) >= 1);
}

pub(crate) fn wait_registered(n: usize) {
    while waits::registered() < n {
        thread::sleep(Duration::from_millis(1));
    }
}

fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
    match err.downcast::<String>() {
        Ok(s) => *s,
        Err(err) => err.downcast::<&str>().map(|s| s.to_string()).unwrap(),
    }
}

/// How `q` reads `r` in [`StaleEdge`].
#[derive(Copy, Clone, PartialEq)]
enum QReads {
    Plain,
    /// Through a `catch_unwind` that returns this value instead.
    Swallowing(i64),
}

/// Thread A computes `a`, which now reads `q`; thread B computes `q`,
/// whose closure reads `r`, and the walk for `r` reaches `a` through
/// `x`'s stale edge, so B waits on A's claim while A waits on B's.
/// `q` reads `u` and then `r` (`v` before `u` was set), and `d` reads
/// `q`; unless `q_new`, both are Clean on `q` = 3 before the sets, and
/// otherwise B's compute of `q` is its first. Every node reads 4 (`d`
/// 40) once the graph settles.
struct StaleEdge {
    rt: Arc<Runtime<Shared>>,
    a: Incr<i64>,
    q: Incr<i64>,
    d: Incr<i64>,
    k: Incr<i64>,
    a_gate: Gate,
    b_gate: Gate,
}

impl StaleEdge {
    fn new(reads: QReads, q_new: bool) -> Self {
        let rt = Arc::new(Runtime::<Shared>::new());
        let s = rt.create_input(0_i64);
        let t = rt.create_input(0_i64);
        let u = rt.create_input(0_i64);
        let i = rt.create_input(1_i64);
        let k = rt.create_input(2_i64);
        let v = rt.create_input(3_i64);
        let q_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let a_gate = Gate::new();
        let a = {
            let q_cell = q_cell.clone();
            let a_gate = a_gate.clone();
            rt.create_query(move |rt| {
                if rt.get(t) == 0 {
                    rt.get(i)
                } else {
                    a_gate.park();
                    rt.get(*q_cell.get().unwrap())
                }
            })
        };
        let x = rt.create_query(move |rt| if rt.get(s) == 0 { rt.get(a) } else { rt.get(k) });
        let b = rt.create_query(move |rt| rt.get(x) + 1);
        let r = rt.create_query(move |rt| rt.get(b) + 1);
        let make_q = |rt: &Runtime<Shared>| {
            rt.create_query(move |rt| {
                if rt.get(u) == 0 {
                    return rt.get(v);
                }
                match reads {
                    QReads::Plain => rt.get(r),
                    QReads::Swallowing(instead) => {
                        catch_unwind(AssertUnwindSafe(|| rt.get(r))).unwrap_or(instead)
                    }
                }
            })
        };
        assert_eq!(rt.get(r), 3);
        let q = make_q(&rt);
        let d = rt.create_query(move |rt| rt.get(q) * 10);
        if !q_new {
            assert_eq!(rt.get(d), 30);
        }
        q_cell.set(q).unwrap();
        rt.set(s, 1);
        rt.set(t, 1);
        rt.set(u, 1);
        Self {
            rt,
            a,
            q,
            d,
            k,
            a_gate,
            b_gate: Gate::new(),
        }
    }

    /// Parks A inside `a`'s closure and B at its wait on `a`, then lets
    /// them wait on each other, the thread named by `a_first` first, so
    /// that it holds the older ticket. Returns A's thread and B's, which
    /// runs `b`.
    fn run<R: Send + 'static>(
        &self,
        a_first: bool,
        b: impl FnOnce(&Runtime<Shared>) -> R + Send + 'static,
    ) -> (
        thread::JoinHandle<i64>,
        thread::JoinHandle<thread::Result<R>>,
    ) {
        self.a_gate.arm();
        let thread_a = {
            let rt = Arc::clone(&self.rt);
            let a = self.a;
            thread::spawn(move || rt.get(a))
        };
        self.a_gate.wait_parked();
        park_at(
            &self.rt,
            vec![(ParkPoint::BeforeComputeOne(id(self.a)), self.b_gate.clone())],
        );
        let thread_b = {
            let rt = Arc::clone(&self.rt);
            thread::spawn(move || catch_unwind(AssertUnwindSafe(|| b(&rt))))
        };
        self.b_gate.wait_parked();
        let (first, second) = if a_first {
            (&self.a_gate, &self.b_gate)
        } else {
            (&self.b_gate, &self.a_gate)
        };
        first.release();
        wait_registered(1);
        second.release();
        (thread_a, thread_b)
    }
}

/// The second thread to register closes the waits-for cycle and is
/// wounded: it releases its claim and retries, and the older thread
/// resolves the graph by the single-thread rules, so the outcome is the
/// one a single thread gets reading the older thread's node first.
fn stale_edge_waits_for_cycle(a_first: bool) {
    let g = StaleEdge::new(QReads::Plain, false);
    let q = g.q;
    let (thread_a, thread_b) = g.run(a_first, move |rt| rt.get(q));
    assert_eq!(thread_a.join().unwrap(), 4);
    let b_result = thread_b.join().unwrap();
    if a_first {
        assert_eq!(b_result.unwrap(), 4);
    } else {
        // Read first on one thread, `q` reaches `a` through the stale
        // edge while `q` is on the stack, which is a cycle.
        let message = panic_message(b_result.unwrap_err());
        assert!(message.contains("dependency cycle"), "{message}");
        assert_eq!(g.rt.get(q), 4);
    }
    assert_wounded();
    g.rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_stale_edge_waits_for_cycle_with_a_older() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_stale_edge_waits_for_cycle_with_a_older",
        || stale_edge_waits_for_cycle(true),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_stale_edge_waits_for_cycle_with_b_older() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_stale_edge_waits_for_cycle_with_b_older",
        || stale_edge_waits_for_cycle(false),
    )
    .assert_clean();
}

/// B is wounded inside `stabilize`, which retries; its observer fires
/// once, with the settled value, and nothing reaches the caller.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_wound_under_stabilize_is_retried() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_wound_under_stabilize_is_retried",
        || {
            let g = StaleEdge::new(QReads::Plain, false);
            let fired = Arc::new(Mutex::new(Vec::new()));
            {
                let fired = fired.clone();
                g.rt.observe(g.q, move |v| fired.lock().unwrap().push(*v));
            }
            let (thread_a, thread_b) = g.run(true, |rt| rt.stabilize());
            assert_eq!(thread_a.join().unwrap(), 4);
            thread_b.join().unwrap().unwrap();
            assert_eq!(*fired.lock().unwrap(), vec![4]);
            assert_wounded();
        },
    )
    .assert_clean();
}

/// `q` swallows its wound and returns the value the retry computes, 4,
/// after the slot held 3. Its dependent `d`, Clean on 3, recomputes.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_swallowed_wound_after_the_write() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_swallowed_wound_after_the_write",
        || {
            let g = StaleEdge::new(QReads::Swallowing(4), false);
            let d = g.d;
            let (thread_a, thread_b) = g.run(true, move |rt| rt.get(d));
            assert_eq!(thread_a.join().unwrap(), 4);
            assert_eq!(thread_b.join().unwrap().unwrap(), 40);
            assert_eq!(g.rt.get(d), 40);
            assert_wounded();
            g.rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

/// The same on `q`'s first compute, swallowing to a value the retry does
/// not compute: `q` ends with the retry's value and its reverse edges, so
/// a later `set` reaches it.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_swallowed_wound_on_a_first_compute() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_swallowed_wound_on_a_first_compute",
        || {
            let g = StaleEdge::new(QReads::Swallowing(99), true);
            let q = g.q;
            let (thread_a, thread_b) = g.run(true, move |rt| rt.get(q));
            assert_eq!(thread_a.join().unwrap(), 4);
            assert_eq!(thread_b.join().unwrap().unwrap(), 4);
            assert_wounded();
            g.rt.set(g.k, 10);
            assert_eq!(g.rt.get(q), 12);
            assert_eq!(g.rt.get(g.a), 12);
            g.rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

/// `q`'s verify fails only because `x` is Computing on thread O. B's
/// recompute of `q` is wounded inside it, swallows the wound and returns
/// a value other than its stored one, so the write happens, and `x` then
/// lands with a cutoff. The retry recomputes `q` instead of verifying it
/// Clean over the swallowed value. O reaches `q` through `y`'s stale edge,
/// and the walk that waited on B's claim visits `q` again once B
/// releases it, so it skips `x`, its own claim, and taints `q` instead of
/// computing it.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_swallowed_wound_in_a_recompute_whose_dep_cuts_off() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_swallowed_wound_in_a_recompute_whose_dep_cuts_off",
        || {
            let rt = Arc::new(Runtime::<Shared>::new());
            let sw = rt.create_input(0_i64);
            let xin = rt.create_input(0_i64);
            let x_gate = Gate::new();
            let q_gate = Gate::new();
            let q_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
            let y = {
                let q_cell = q_cell.clone();
                rt.create_query(move |rt| {
                    if rt.get(sw) == 0 {
                        rt.get(*q_cell.get().unwrap())
                    } else {
                        7
                    }
                })
            };
            let x = {
                let x_gate = x_gate.clone();
                rt.create_query(move |rt| {
                    if rt.get(xin) >= 2 {
                        x_gate.park();
                        rt.get(y);
                    }
                    5
                })
            };
            let q_runs = Arc::new(AtomicUsize::new(0));
            let q = {
                let q_gate = q_gate.clone();
                let q_runs = q_runs.clone();
                rt.create_query(move |rt| {
                    q_runs.fetch_add(1, Ordering::Relaxed);
                    q_gate.park();
                    catch_unwind(AssertUnwindSafe(|| rt.get(x) + 1)).unwrap_or(-1)
                })
            };
            q_cell.set(q).unwrap();
            assert_eq!(rt.get(y), 6);
            rt.set(sw, 1);
            rt.set(xin, 1);
            assert_eq!(rt.get(x), 5);

            let claim = Gate::new();
            park_at(&rt, vec![(ParkPoint::BeforeClaim(id(q)), claim.clone())]);
            let thread_b = {
                let rt = Arc::clone(&rt);
                thread::spawn(move || rt.get(q))
            };
            claim.wait_parked();
            rt.set(xin, 2);
            x_gate.arm();
            let thread_o = {
                let rt = Arc::clone(&rt);
                thread::spawn(move || rt.get(x))
            };
            x_gate.wait_parked();
            q_gate.arm();
            claim.release();
            q_gate.wait_parked();
            x_gate.release();
            wait_registered(1);
            q_gate.release();

            assert_eq!(thread_o.join().unwrap(), 5);
            assert_eq!(thread_b.join().unwrap(), 6);
            assert_eq!(rt.get(q), 6);
            // The setup's run, B's wounded run and its retry: O never
            // runs `q`, whose read of `x` would be a cycle on O's stack.
            assert_eq!(q_runs.load(Ordering::Relaxed), 3);
            assert_wounded();
            rt.assert_invariant_i();
        },
    )
    .assert_clean();
}
