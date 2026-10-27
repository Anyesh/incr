//! Wound-wait under `Shared`: claim waits that close a waits-for cycle
//! across threads wound the younger thread, which releases its claims and
//! retries. Each test runs in a child process, because a regression hangs
//! and because `waits::registered` counts the waiters of every runtime.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Barrier, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use super::observe_impl::observed_is_clear;
use super::race_tests::{id, park_at, release_parked_on_unwind, Gate};
use super::{GroupInfo, NodeId, ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{CycleStrategy, Fixpoint, QueryOpts};
use crate::handle::{Incr, RuntimeId};
use crate::state::NodeState;
use crate::waits;

/// A poll for another thread's progress gives up within the 10 s a gated
/// thread parks and a child test runs, so a failed poll reports itself
/// instead of being killed.
const POLL_DEADLINE: Duration = Duration::from_secs(4);

/// Asserts a cycle through `rt`'s claims was resolved by wounds. Usually
/// one: a retry that takes its released claims back before the older
/// thread does, which `await_winner`'s head start makes rare but cannot
/// rule out under load, is wounded again over the same cycle, which costs
/// a retry and never changes a value.
pub(crate) fn assert_wounded(rt: RuntimeId) {
    assert!(waits::WOUNDS.on(rt) >= 1);
}

pub(crate) fn wait_registered(rt: RuntimeId, n: usize) {
    while waits::registered_on(rt) < n {
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
    /// Through a `catch_unwind` after which the closure calls the engine
    /// once more, recording in [`AFTER_CATCH`] whether that call unwound.
    SwallowingThen(After),
}

/// The engine call `q` makes after catching an unwind of its read of `r`:
/// a read of the Dirty `w`, or a set of `w`'s input.
#[derive(Copy, Clone, PartialEq)]
enum After {
    Read,
    Set,
}

/// For each call after a catch in [`QReads::SwallowingThen`], whether it
/// unwound and how many times `w`'s closure ran inside it. Each test runs
/// in its own child process.
static AFTER_CATCH: Mutex<Vec<(bool, usize)>> = Mutex::new(Vec::new());

static W_RUNS: AtomicUsize = AtomicUsize::new(0);

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
        Self::with_replay(reads, q_new, false)
    }

    /// [`Self::new`] with every query replayable if `replayable`, so a frame
    /// budget replays its reads.
    fn with_replay(reads: QReads, q_new: bool, replayable: bool) -> Self {
        let mut rt = Runtime::<Shared>::new();
        rt.all_replayable = replayable;
        let rt = Arc::new(rt);
        if replayable {
            rt.set_placeholder(0_i64);
        }
        let s = rt.create_input(0_i64);
        let t = rt.create_input(0_i64);
        let u = rt.create_input(0_i64);
        let i = rt.create_input(1_i64);
        let k = rt.create_input(2_i64);
        let v = rt.create_input(3_i64);
        let w_in = rt.create_input(0_i64);
        let w = rt.create_query(move |rt| {
            W_RUNS.fetch_add(1, Ordering::Relaxed);
            rt.get(w_in) + 1
        });
        assert_eq!(rt.get(w), 1);
        rt.set(w_in, 1);
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
                    QReads::SwallowingThen(after) => catch_unwind(AssertUnwindSafe(|| rt.get(r)))
                        .unwrap_or_else(|_| {
                            let runs = W_RUNS.load(Ordering::Relaxed);
                            let call = catch_unwind(AssertUnwindSafe(|| match after {
                                After::Read => drop(rt.get(w)),
                                After::Set => rt.set(w_in, 2),
                            }));
                            let ran = W_RUNS.load(Ordering::Relaxed) - runs;
                            AFTER_CATCH.lock().unwrap().push((call.is_err(), ran));
                            if let Err(payload) = call {
                                std::panic::resume_unwind(payload);
                            }
                            -1
                        }),
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
        self.run_parking_b(a_first, None, b)
    }

    /// [`Self::run`], also parking B at `b_exit` when its outermost exit
    /// checks its cycle verdicts.
    fn run_parking_b<R: Send + 'static>(
        &self,
        a_first: bool,
        b_exit: Option<Gate>,
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
        let b_thread = Arc::new(OnceLock::new());
        match b_exit {
            None => park_at(
                &self.rt,
                vec![(ParkPoint::BeforeComputeOne(id(self.a)), self.b_gate.clone())],
            ),
            Some(exit) => {
                let (b_gate, b_thread, a) = (self.b_gate.clone(), b_thread.clone(), id(self.a));
                b_gate.arm();
                exit.arm();
                let installed = self.rt.park_hook.set(Box::new(move |p| {
                    if p == ParkPoint::BeforeComputeOne(a) {
                        b_gate.park();
                    } else if p == ParkPoint::VerdictCheck
                        && b_thread.get() == Some(&thread::current().id())
                    {
                        exit.park();
                    }
                }));
                assert!(installed.is_ok(), "park hook installed twice");
            }
        }
        let thread_b = {
            let rt = Arc::clone(&self.rt);
            thread::spawn(move || {
                b_thread.set(thread::current().id()).unwrap();
                catch_unwind(AssertUnwindSafe(|| b(&rt)))
            })
        };
        self.b_gate.wait_parked();
        let (first, second) = if a_first {
            (&self.a_gate, &self.b_gate)
        } else {
            (&self.b_gate, &self.a_gate)
        };
        first.release();
        wait_registered(self.rt.runtime_id, 1);
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
    assert_wounded(g.rt.runtime_id);
    g.rt.assert_invariant_i();
}

/// B, the older thread, closes the cycle and its unwind releases the
/// members before its outermost exit, so A's retry can land them first.
/// B read nothing after the cycle, so its outcome stays the cycle, the
/// order with B's read first, at every frame budget.
fn older_threads_uncaught_cycle_stands(budget: Option<u32>) {
    let g = StaleEdge::with_replay(QReads::Plain, false, budget.is_some());
    if let Some(n) = budget {
        g.rt.set_max_frames(n);
    }
    let q = g.q;
    let exit = Gate::new();
    let (thread_a, thread_b) = g.run_parking_b(false, Some(exit.clone()), move |rt| rt.get(q));
    exit.wait_parked();
    let deadline = Instant::now() + POLL_DEADLINE;
    while !thread_a.is_finished() {
        assert!(
            Instant::now() < deadline,
            "A never finished while B was parked"
        );
        thread::sleep(Duration::from_millis(1));
    }
    exit.release();
    assert_eq!(thread_a.join().unwrap(), 4);
    let message = panic_message(thread_b.join().unwrap().unwrap_err());
    assert!(message.contains("dependency cycle"), "{message}");
    assert_eq!(g.rt.verdict_reruns.load(Ordering::Relaxed), 0);
    if budget.is_some() {
        let replayed = (
            g.rt.dooms.lock().unwrap().len(),
            g.rt.entries.lock().unwrap().len(),
            g.rt.deliveries.lock().unwrap().len(),
        );
        assert_ne!(
            replayed,
            (0, 0, 0),
            "the budget replayed no read (dooms, entries, deliveries)"
        );
    }
    assert_eq!(g.rt.get(q), 4);
    assert_wounded();
    g.rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_older_threads_uncaught_cycle_stands_over_a_later_landing() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_older_threads_uncaught_cycle_stands_over_a_later_landing",
        || older_threads_uncaught_cycle_stands(None),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_older_threads_uncaught_cycle_stands_at_frame_budget_1() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_older_threads_uncaught_cycle_stands_at_frame_budget_1",
        || older_threads_uncaught_cycle_stands(Some(1)),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_older_threads_uncaught_cycle_stands_at_frame_budget_2() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_older_threads_uncaught_cycle_stands_at_frame_budget_2",
        || older_threads_uncaught_cycle_stands(Some(2)),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_older_threads_uncaught_cycle_stands_at_frame_budget_3() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_older_threads_uncaught_cycle_stands_at_frame_budget_3",
        || older_threads_uncaught_cycle_stands(Some(3)),
    )
    .assert_clean();
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
            assert_wounded(g.rt.runtime_id);
        },
    )
    .assert_clean();
}

/// Three queued observed nodes, B wounded while `stabilize` processes
/// the second, `q`: the retry takes `q` and the third back from the
/// queue (`q`'s wound release may queue it first), every observer fires
/// once, and the observer's thread-local is
/// clear afterwards.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_wound_on_the_second_of_three_observers() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_wound_on_the_second_of_three_observers",
        || {
            let g = StaleEdge::new(QReads::Plain, false);
            let w = g.rt.create_input(7_i64);
            let first = g.rt.create_query(move |rt| rt.get(w) + 1);
            let third = g.rt.create_query(move |rt| rt.get(w) + 2);
            let fired = Arc::new(Mutex::new(Vec::new()));
            for (tag, h) in [(0, first), (1, g.q), (2, third)] {
                let fired = fired.clone();
                g.rt.observe(h, move |v| fired.lock().unwrap().push((tag, *v)));
            }
            let (thread_a, thread_b) = g.run(true, |rt| {
                rt.stabilize();
                observed_is_clear()
            });
            assert_eq!(thread_a.join().unwrap(), 4);
            assert!(thread_b.join().unwrap().unwrap());
            let mut fired = fired.lock().unwrap().clone();
            fired.sort_unstable();
            assert_eq!(fired, vec![(0, 8), (1, 4), (2, 9)]);
            assert_wounded(g.rt.runtime_id);
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
            assert_wounded(g.rt.runtime_id);
            g.rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

/// A closure that swallowed its wound reads on into a cycle on its own
/// stack, which a `try_get` below catches: the wound still unwinds to the
/// outermost entry, which retries, so the thread leaves the read
/// unwounded. Left wounded, its later reads would treat what their walks
/// contain as a wounded attempt's.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_swallowed_wound_outlives_a_caught_cycle() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_swallowed_wound_outlives_a_caught_cycle",
        || {
            let rt = Runtime::<Shared>::new();
            let wound = Arc::new(AtomicBool::new(true));
            let p_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
            let m = {
                let wound = wound.clone();
                let p_cell = p_cell.clone();
                rt.create_query(move |rt| {
                    if wound.swap(false, Ordering::Relaxed) {
                        let _ = catch_unwind(AssertUnwindSafe(|| rt.raise_wound()));
                    }
                    rt.get(*p_cell.get().unwrap())
                })
            };
            let p = rt.create_query(move |rt| rt.get(m));
            p_cell.set(p).unwrap();
            let q = rt.create_query(move |rt| rt.try_get(p).unwrap_or(99));
            assert_eq!(rt.get(q), 99);
            assert!(!wound.load(Ordering::Relaxed));
            assert!(!crate::cycle_state::wounded());
            rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

/// A wound unwinding past a guard whose drop reads into a cycle on the
/// stack: the drop's `try_get` gets the cycle, since a wound raised there
/// would abort the process, and the wound still reaches the outermost
/// entry, which retries.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_wound_unwinding_past_a_drop_that_reads_a_cycle_retries() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_wound_unwinding_past_a_drop_that_reads_a_cycle_retries",
        || {
            struct ReadsOnDrop<'a> {
                rt: &'a Runtime<Shared>,
                p: Incr<i64>,
                got: &'a AtomicBool,
            }
            impl Drop for ReadsOnDrop<'_> {
                fn drop(&mut self) {
                    if self.rt.try_get(self.p).is_err() {
                        self.got.store(true, Ordering::Relaxed);
                    }
                }
            }
            let rt = Runtime::<Shared>::new();
            let wound = Arc::new(AtomicBool::new(true));
            let got = Arc::new(AtomicBool::new(false));
            let q_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
            let p = {
                let q_cell = q_cell.clone();
                rt.create_query(move |rt| rt.get(*q_cell.get().unwrap()))
            };
            let q = {
                let wound = wound.clone();
                let got = got.clone();
                rt.create_query(move |rt| {
                    if wound.swap(false, Ordering::Relaxed) {
                        let _guard = ReadsOnDrop { rt, p, got: &got };
                        rt.raise_wound();
                    }
                    7_i64
                })
            };
            q_cell.set(q).unwrap();
            assert_eq!(rt.get(q), 7);
            assert!(got.load(Ordering::Relaxed));
            assert!(!crate::cycle_state::wounded());
            rt.assert_invariant_i();
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
            assert_wounded(g.rt.runtime_id);
            g.rt.set(g.k, 10);
            assert_eq!(g.rt.get(q), 12);
            assert_eq!(g.rt.get(g.a), 12);
            g.rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

/// `q` catches the wound of its read of `r` and then reads the Dirty `w`
/// (or sets `w`'s input): that call raises the wound again before any
/// engine work, as one after a caught discard does. Computing `w` on the
/// wounded thread leaves its pending wound to whatever that work does: in
/// seed 0x2e05 a speculative drop inside such a recompute overwrote it,
/// and the catching closure's value landed. The retry computes 4.
fn call_after_a_caught_wound_raises_it_again(after: After) {
    let g = StaleEdge::new(QReads::SwallowingThen(after), false);
    let q = g.q;
    let (thread_a, thread_b) = g.run(true, move |rt| rt.get(q));
    assert_eq!(thread_a.join().unwrap(), 4);
    assert_eq!(thread_b.join().unwrap().unwrap(), 4);
    assert_wounded(g.rt.runtime_id);
    let raised = AFTER_CATCH.lock().unwrap().clone();
    assert!(
        !raised.is_empty() && raised.iter().all(|&r| r == (true, 0)),
        "each call after a caught wound: (whether it raised, w's runs in it): {raised:?}"
    );
    g.rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_read_after_a_caught_wound_raises_it_again() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_read_after_a_caught_wound_raises_it_again",
        || call_after_a_caught_wound_raises_it_again(After::Read),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_set_after_a_caught_wound_raises_it_again() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_set_after_a_caught_wound_raises_it_again",
        || call_after_a_caught_wound_raises_it_again(After::Set),
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
    // A forced revisit of `q` makes B wait on O's claim of `x`, which the
    // gates hold until B runs `q`.
    if super::forced_revisit() {
        return;
    }
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
            wait_registered(rt.runtime_id, 1);
            q_gate.release();

            assert_eq!(thread_o.join().unwrap(), 5);
            assert_eq!(thread_b.join().unwrap(), 6);
            assert_eq!(rt.get(q), 6);
            // The setup's run, B's wounded run and its retry: O never
            // runs `q`, whose read of `x` would be a cycle on O's stack.
            assert_eq!(q_runs.load(Ordering::Relaxed), 3);
            assert_wounded(rt.runtime_id);
            rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

/// `below` (unseeded) and `m` (Fixpoint) form one cycle that is warm
/// from a `get(m)`. After each `set` two threads read one member each,
/// and their claim waits close a waits-for cycle. Every single-thread
/// order gives `below = 6` and `m = 5`, so the wound must not change
/// either result.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_concurrent_reads_of_a_warm_cycle_match_the_single_thread_result() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_concurrent_reads_of_a_warm_cycle_match_the_single_thread_result",
        || {
            let rt = Runtime::<Shared>::new();
            let n = rt.create_input(10_i64);
            let m_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
            let below = {
                let m_cell = m_cell.clone();
                rt.create_query(move |rt| rt.get(n).min(rt.get(*m_cell.get().unwrap()) + 1))
            };
            let m = rt.create_query_with(
                move |rt| (rt.get(below) + 1).min(5),
                QueryOpts {
                    cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64).max_iterations(20)),
                    ..QueryOpts::default()
                },
            );
            m_cell.set(m).unwrap();
            assert_eq!((rt.get(m), rt.get(below)), (5, 6));
            for round in 0..200_i64 {
                rt.set(n, 11 + round);
                let start = Barrier::new(2);
                let read = |node: Incr<i64>| {
                    start.wait();
                    catch_unwind(AssertUnwindSafe(|| rt.get(node))).map_err(panic_message)
                };
                let (b, m) = thread::scope(|s| {
                    let b = s.spawn(|| read(below));
                    let m = s.spawn(|| read(m));
                    (b.join().unwrap(), m.join().unwrap())
                });
                assert_eq!((b, m), (Ok(6), Ok(5)), "round {round}");
                rt.assert_invariant_i();
            }
        },
    )
    .assert_clean();
}

/// How [`Duel`] puts the unseeded `x` and the Fallback `m` (100) in a
/// group whose cycle runs through `w` (Fallback 500) once `k` is 1.
#[derive(Copy, Clone, Debug)]
enum GroupShape {
    /// The group is warm from a `get(m)`, so B's `get(m)` verifies it and
    /// runs its episode, and B is wounded in a member run.
    Episode,
    /// Only `x` keeps the record, as a single-node landing of `m` leaves
    /// it, so B computes `m` alone and its group verify of `x` finds `m`
    /// on B's stack and computes `x` alone too.
    MemberAlone,
}

/// Thread A reads `w`, which reads `x`; thread B reads `m`, which reads
/// `x` and then `w`. One of them parks before its claim wait until the
/// other has registered its own, so the other holds the older ticket.
struct Duel {
    rt: Arc<Runtime<Shared>>,
    x: Incr<i64>,
    m: Incr<i64>,
    w: Incr<i64>,
    a_gate: Gate,
    b_gate: Gate,
}

impl Duel {
    fn new(shape: GroupShape) -> Self {
        let rt = Arc::new(Runtime::<Shared>::new());
        let k = rt.create_input(0_i64);
        let m_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let w_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let x = {
            let m_cell = m_cell.clone();
            rt.create_query(move |rt| match (shape, rt.get(k)) {
                (GroupShape::MemberAlone, 0) => 1,
                _ => rt.get(*m_cell.get().unwrap()) + 1,
            })
        };
        let (a_gate, b_gate) = (Gate::new(), Gate::new());
        let w = {
            let a_gate = a_gate.clone();
            rt.create_query_with_fallback(
                move |rt| {
                    a_gate.park();
                    match rt.get(k) {
                        0 => 0,
                        _ => rt.get(x) + 1000,
                    }
                },
                |_c: &crate::cycle::Cycle| 500,
            )
        };
        let m = {
            let (w_cell, b_gate) = (w_cell.clone(), b_gate.clone());
            rt.create_query_with_fallback(
                move |rt| match (shape, rt.get(k)) {
                    (GroupShape::MemberAlone, 0) => 0,
                    (_, 0) => rt.get(x),
                    _ => {
                        let x = rt.get(x);
                        b_gate.park();
                        x + rt.get(*w_cell.get().unwrap())
                    }
                },
                |_c: &crate::cycle::Cycle| 100,
            )
        };
        m_cell.set(m).unwrap();
        w_cell.set(w).unwrap();
        match shape {
            GroupShape::Episode => assert_eq!((rt.get(m), rt.get(x), rt.get(w)), (100, 101, 0)),
            GroupShape::MemberAlone => {
                assert_eq!((rt.get(m), rt.get(x), rt.get(w)), (0, 1, 0));
                rt.force_group_for_test(&[x.erase(), m.erase()]);
                let (node, flags) = rt.nodes.get_with_flags(m.slot());
                rt.leave_group(id(m), node, flags, NodeState::Clean);
            }
        }
        rt.set(k, 1);
        Duel {
            rt,
            x,
            m,
            w,
            a_gate,
            b_gate,
        }
    }

    fn read(&self, node: Incr<i64>) -> Result<i64, String> {
        catch_unwind(AssertUnwindSafe(|| self.rt.get(node))).map_err(panic_message)
    }

    /// A's read of `w`, B's of `m`, then a read of every node.
    fn outcome(&self, a: Result<i64, String>, b: Result<i64, String>) -> Vec<Result<i64, String>> {
        vec![
            a,
            b,
            self.read(self.w),
            self.read(self.m),
            self.read(self.x),
        ]
    }

    fn single_thread(shape: GroupShape, a_first: bool) -> Vec<Result<i64, String>> {
        let g = Duel::new(shape);
        if a_first {
            let a = g.read(g.w);
            g.outcome(a, g.read(g.m))
        } else {
            let b = g.read(g.m);
            g.outcome(g.read(g.w), b)
        }
    }

    fn concurrent(shape: GroupShape, a_older: bool) -> Vec<Result<i64, String>> {
        let g = Duel::new(shape);
        let wounds = waits::WOUNDS.on(g.rt.runtime_id);
        let first = if a_older { &g.b_gate } else { &g.a_gate };
        first.arm();
        let (a, b) = thread::scope(|s| {
            let _released = release_parked_on_unwind();
            let g = &g;
            let parked = |n: Incr<i64>| {
                let read = s.spawn(move || g.read(n));
                first.wait_parked();
                read
            };
            let (a, b) = if a_older {
                let b = parked(g.m);
                (s.spawn(|| g.read(g.w)), b)
            } else {
                let a = parked(g.w);
                (a, s.spawn(|| g.read(g.m)))
            };
            wait_registered(g.rt.runtime_id, 1);
            first.release();
            (a.join().unwrap(), b.join().unwrap())
        });
        assert!(
            waits::WOUNDS.on(g.rt.runtime_id) > wounds,
            "{shape:?} a_older {a_older}: no wound"
        );
        let outcome = g.outcome(a, b);
        g.rt.assert_invariant_i();
        outcome
    }
}

/// A wound's discard landing of a group member keeps it in its group, so
/// the older thread verifies the group as a single thread would. Out of
/// it, `x` is computed alone at A's root, and in the `MemberAlone` shape
/// that enters the cycle at the unseeded `x`: a cycle error that no
/// single-thread order raises.
fn wounded_member_keeps_its_group(shape: GroupShape) {
    let a_first = Duel::single_thread(shape, true);
    let b_first = Duel::single_thread(shape, false);
    for round in 0..20 {
        for a_older in [true, false] {
            let got = Duel::concurrent(shape, a_older);
            assert!(
                got == a_first || got == b_first,
                "{shape:?} round {round} a_older {a_older}: {got:?}, \
                 single-thread {a_first:?} or {b_first:?}"
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_wound_in_a_group_episode_keeps_the_group() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_wound_in_a_group_episode_keeps_the_group",
        || wounded_member_keeps_its_group(GroupShape::Episode),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_wound_of_a_member_computed_alone_keeps_its_group() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_wound_of_a_member_computed_alone_keeps_its_group",
        || wounded_member_keeps_its_group(GroupShape::MemberAlone),
    )
    .assert_clean();
}

/// Which release of group verify's claims [`Released`] exercises.
#[derive(Copy, Clone, Debug)]
enum Release {
    /// `r` and `x` keep a record of the group {r, m, x}, which `m` has
    /// left, as a single-node landing of `m` leaves it. B's group verify
    /// of `x`, in `m`'s run, claims `r` and fails on `m`, B's own claim,
    /// so step 1 releases `r`.
    Step1,
    /// `r` and `x` keep a record of the group {r, x}, and `n` reads `x`,
    /// which read `m` before `k` was set. B's walk for `n`, in `m`'s run,
    /// reaches `m`, B's own claim, through `x`, so it verifies `x` only:
    /// its group verify claims `r`, finds `k` changed and releases both.
    Tainted,
}

/// Once `k` is 1, `r` reads `m` (Fallback 100), `m` reads `n` (Tainted)
/// or `x`, `n` reads `x`, and `x` reads `r`; `r` and `x` are unseeded.
/// B's read of `m` computes `m` alone, its group verify of `x` releases
/// `r` as `release` says, and `x` computes alone and parks before
/// reading `r`. A reads `r` and waits on `m`, and B's read of `r` closes
/// the waits-for cycle, so B is wounded. If the release took `r` out of
/// its group, A computes `r` alone, and the cycle enters at the unseeded
/// `x`: a cycle error that no single-thread order raises.
struct Released {
    rt: Arc<Runtime<Shared>>,
    r: Incr<i64>,
    m: Incr<i64>,
    n: Incr<i64>,
    x: Incr<i64>,
    gate: Gate,
    /// A's thread, and the nodes A's reads visit or wait on.
    a_thread: Arc<OnceLock<thread::ThreadId>>,
    visits: Mutex<mpsc::Receiver<NodeId>>,
}

impl Released {
    fn new(release: Release) -> Self {
        let rt = Arc::new(Runtime::<Shared>::new());
        let k = rt.create_input(0_i64);
        let m_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let x_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let r = {
            let m_cell = m_cell.clone();
            rt.create_query(move |rt| match rt.get(k) {
                0 => 1,
                _ => rt.get(*m_cell.get().unwrap()) + 1,
            })
        };
        let n = {
            let x_cell = x_cell.clone();
            rt.create_query(move |rt| rt.get(*x_cell.get().unwrap()) + 100 * rt.get(k))
        };
        let m = {
            let x_cell = x_cell.clone();
            rt.create_query_with_fallback(
                move |rt| match (release, rt.get(k)) {
                    (_, 0) => 0,
                    (Release::Step1, _) => rt.get(*x_cell.get().unwrap()) + 10,
                    (Release::Tainted, _) => rt.get(n) + 10,
                },
                |_c: &crate::cycle::Cycle| 100,
            )
        };
        let gate = Gate::new();
        let x = {
            let gate = gate.clone();
            rt.create_query(move |rt| match (release, rt.get(k)) {
                (Release::Step1, 0) => 0,
                (Release::Tainted, 0) => rt.get(m),
                _ => {
                    gate.park();
                    rt.get(r) + 1000
                }
            })
        };
        m_cell.set(m).unwrap();
        x_cell.set(x).unwrap();
        assert!(r.slot() < m.slot() && m.slot() < x.slot());
        assert_eq!((rt.get(r), rt.get(m), rt.get(x), rt.get(n)), (1, 0, 0, 0));
        match release {
            Release::Step1 => {
                rt.force_group_for_test(&[r.erase(), m.erase(), x.erase()]);
                let (node, flags) = rt.nodes.get_with_flags(m.slot());
                rt.leave_group(id(m), node, flags, NodeState::Clean);
            }
            Release::Tainted => rt.force_group_for_test(&[r.erase(), x.erase()]),
        }
        rt.set(k, 1);
        let a_thread: Arc<OnceLock<thread::ThreadId>> = Arc::default();
        let (tx, visits) = mpsc::channel();
        let (a, tx) = (a_thread.clone(), Mutex::new(tx));
        let hooked = rt.park_hook.set(Box::new(move |p| {
            if let ParkPoint::BeforeComputeOne(id) | ParkPoint::ComputeWaits(id) = p {
                if a.get() == Some(&thread::current().id()) {
                    let _ = tx.lock().unwrap().send(id);
                }
            }
        }));
        assert!(hooked.is_ok());
        Released {
            rt,
            r,
            m,
            n,
            x,
            gate,
            a_thread,
            visits: Mutex::new(visits),
        }
    }

    fn read(&self, node: Incr<i64>) -> Result<i64, String> {
        catch_unwind(AssertUnwindSafe(|| self.rt.get(node))).map_err(panic_message)
    }

    /// A's read of `r`, B's of `m`, then a read of every node.
    fn outcome(&self, a: Result<i64, String>, b: Result<i64, String>) -> Vec<Result<i64, String>> {
        vec![
            a,
            b,
            self.read(self.r),
            self.read(self.m),
            self.read(self.x),
        ]
    }

    fn single_thread(release: Release, a_first: bool) -> Vec<Result<i64, String>> {
        let g = Released::new(release);
        if a_first {
            let a = g.read(g.r);
            g.outcome(a, g.read(g.m))
        } else {
            let b = g.read(g.m);
            g.outcome(g.read(g.r), b)
        }
    }

    /// B parks in `x` holding its claims; A's read starts once it has
    /// and B resumes once A waits on one of them, whether A registers
    /// that wait or, holding no claim, waits unregistered.
    fn concurrent(release: Release) -> Vec<Result<i64, String>> {
        let g = Released::new(release);
        g.gate.arm();
        let (a, b) = thread::scope(|s| {
            let _released = release_parked_on_unwind();
            let g = &g;
            let b = s.spawn(move || g.read(g.m));
            g.gate.wait_parked();
            let held: Vec<NodeId> = [g.r, g.m, g.n, g.x]
                .into_iter()
                .filter(|h| matches!(g.rt.state_of(*h), NodeState::Computing))
                .map(id)
                .collect();
            let a = s.spawn(move || {
                g.a_thread.set(thread::current().id()).unwrap();
                g.read(g.r)
            });
            let deadline = Instant::now() + POLL_DEADLINE;
            while waits::registered_on(g.rt.runtime_id) == 0 {
                match g
                    .visits
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_millis(1))
                {
                    Ok(at) if held.contains(&at) => break,
                    _ => assert!(
                        Instant::now() < deadline,
                        "{release:?}: A never waited on B"
                    ),
                }
            }
            g.gate.release();
            (a.join().unwrap(), b.join().unwrap())
        });
        let outcome = g.outcome(a, b);
        g.rt.assert_invariant_i();
        outcome
    }
}

/// Group verify's releases keep the claims in their group, so a read
/// after the releasing thread's wound gets a single-thread result.
fn released_member_keeps_its_group(release: Release) {
    let a_first = Released::single_thread(release, true);
    let b_first = Released::single_thread(release, false);
    for round in 0..20 {
        let got = Released::concurrent(release);
        assert!(
            got == a_first || got == b_first,
            "{release:?} round {round}: {got:?}, single-thread {a_first:?} or {b_first:?}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_group_verify_step_1_release_keeps_the_group() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_group_verify_step_1_release_keeps_the_group",
        || released_member_keeps_its_group(Release::Step1),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_tainted_group_verify_release_keeps_the_group() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_tainted_group_verify_release_keeps_the_group",
        || released_member_keeps_its_group(Release::Tainted),
    )
    .assert_clean();
}

/// Once `k` is 1, `r` (Fixpoint) reads `z`, and `y` when `z` is not 0;
/// `z` (Fixpoint) reads `w`, and `y` when `w` is 0; `y` (unseeded) reads
/// `w`, then `z` when `w` is 0 and `r` otherwise; `w` (Fixpoint) is 7.
/// All four keep a record of one group, and `b` reads `y`. In an
/// episode's first iteration the seeds are 0, so `r` reaches no other
/// member: A's episode at `r` dissolves and releases `z`, `y` and `w`
/// from the group, as under Local, but they stay A's claims until its
/// read ends. B's read of `b` waits on
/// A's claim of `y` holding `b`, so B is older; A parks in `z` after the
/// dissolve until B waits, and its one read of `b` closes the waits-for
/// cycle: from `z` before A lands a released member (`z_reads_b`), or
/// from `r` once `z` and `w` have landed (`r_reads_b`).
struct DissolveOn<C: Cells> {
    rt: Arc<Runtime<C>>,
    r: Incr<i64>,
    z: Incr<i64>,
    y: Incr<i64>,
    w: Incr<i64>,
    /// B's entry: a claim B holds while it waits on A's `y`.
    b: Incr<i64>,
    /// Reads `r`, then `b` once `top_reads_b` is set.
    top: Incr<i64>,
    /// Parks `z`'s run in A's episode, with every member claimed.
    early: Gate,
    /// Parks `z`'s run after the dissolve.
    late: Gate,
    z_reads_b: Arc<AtomicBool>,
    r_reads_b: Arc<AtomicBool>,
    top_reads_b: Arc<AtomicBool>,
    /// `r`'s next run reads only `k`, so its landing rests on no held one.
    r_alone: Arc<AtomicBool>,
    /// `r`'s state as other threads see it when `top` reads `b`.
    r_at_top: Arc<Mutex<Option<NodeState>>>,
}

type Dissolve = DissolveOn<Shared>;

impl<C: Cells> DissolveOn<C> {
    fn new() -> Self {
        let rt = Arc::new(Runtime::<C>::new());
        let k = rt.create_input(0_i64);
        let z_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let y_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let w_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let b_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let (z_reads_b, r_reads_b) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        );
        let fixpoint = || QueryOpts {
            cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64).max_iterations(20)),
            ..QueryOpts::default()
        };
        let r_alone = Arc::new(AtomicBool::new(false));
        let r = {
            let (z_cell, y_cell, b_cell) = (z_cell.clone(), y_cell.clone(), b_cell.clone());
            let (r_reads_b, r_alone) = (r_reads_b.clone(), r_alone.clone());
            rt.create_query_with(
                move |rt| match rt.get(k) {
                    0 => 1,
                    _ if r_alone.swap(false, Ordering::SeqCst) => 1,
                    _ if rt.get(*z_cell.get().unwrap()) == 0 => 1,
                    _ => {
                        // B holds `b`, so this read wounds A and never returns.
                        if r_reads_b.swap(false, Ordering::SeqCst) {
                            rt.get(*b_cell.get().unwrap());
                        }
                        (rt.get(*y_cell.get().unwrap()) + 1).min(50)
                    }
                },
                fixpoint(),
            )
        };
        let (early, late) = (Gate::new(), Gate::new());
        let z = {
            let (y_cell, w_cell, b_cell) = (y_cell.clone(), w_cell.clone(), b_cell.clone());
            let (early, late) = (early.clone(), late.clone());
            let z_reads_b = z_reads_b.clone();
            rt.create_query_with(
                move |rt| {
                    if rt.get(k) == 0 {
                        return 0;
                    }
                    late.park();
                    // As in `r`: a read that wounds A and never returns.
                    if z_reads_b.swap(false, Ordering::SeqCst) {
                        rt.get(*b_cell.get().unwrap());
                    }
                    early.park();
                    match rt.get(*w_cell.get().unwrap()) {
                        0 => rt.get(*y_cell.get().unwrap()),
                        _ => 3,
                    }
                },
                fixpoint(),
            )
        };
        let y = {
            let w_cell = w_cell.clone();
            rt.create_query(move |rt| match rt.get(k) {
                0 => 0,
                _ if rt.get(*w_cell.get().unwrap()) == 0 => rt.get(z) + 10,
                _ => rt.get(r) + 10,
            })
        };
        let w = rt.create_query_with(move |rt| 7 * rt.get(k), fixpoint());
        let b = rt.create_query(move |rt| rt.get(y));
        let top_reads_b = Arc::new(AtomicBool::new(false));
        let r_at_top = Arc::new(Mutex::new(None));
        let top = {
            let (top_reads_b, r_at_top) = (top_reads_b.clone(), r_at_top.clone());
            rt.create_query(move |rt| {
                let v = rt.get(r);
                // As in `r`: a read that wounds A and never returns.
                if top_reads_b.swap(false, Ordering::SeqCst) {
                    *r_at_top.lock().unwrap() = Some(rt.nodes.get(r.slot()).state());
                    rt.get(b);
                }
                v
            })
        };
        z_cell.set(z).unwrap();
        y_cell.set(y).unwrap();
        w_cell.set(w).unwrap();
        b_cell.set(b).unwrap();
        assert!(r.slot() < z.slot() && z.slot() < y.slot() && y.slot() < w.slot());
        assert_eq!((rt.get(r), rt.get(z), rt.get(y), rt.get(w)), (1, 0, 0, 0));
        rt.force_group_for_test(&[r.erase(), z.erase(), y.erase(), w.erase()]);
        rt.set(k, 1);
        DissolveOn {
            rt,
            r,
            z,
            y,
            w,
            b,
            top,
            early,
            late,
            z_reads_b,
            r_reads_b,
            top_reads_b,
            r_alone,
            r_at_top,
        }
    }

    fn read(&self, node: Incr<i64>) -> Result<i64, String> {
        catch_unwind(AssertUnwindSafe(|| self.rt.get(node))).map_err(panic_message)
    }

    /// A's read of `r`, B's read, then a read of every node.
    fn outcome(&self, a: Result<i64, String>, b: Result<i64, String>) -> Vec<Result<i64, String>> {
        vec![
            a,
            b,
            self.read(self.r),
            self.read(self.z),
            self.read(self.y),
            self.read(self.w),
        ]
    }
}

impl Dissolve {
    fn single_thread(a_first: bool) -> Vec<Result<i64, String>> {
        let g = Dissolve::new();
        if a_first {
            let a = g.read(g.r);
            g.outcome(a, g.read(g.b))
        } else {
            let b = g.read(g.b);
            g.outcome(g.read(g.r), b)
        }
    }

    /// The outcome, the runtime's `unwound_landings`, and `r`'s slot.
    fn concurrent(z_reads_b: bool) -> Concurrent {
        let g = Dissolve::new();
        let wounds = waits::WOUNDS.on(g.rt.runtime_id);
        g.early.arm();
        let (a, b) = thread::scope(|s| {
            let _released = release_parked_on_unwind();
            let g = &g;
            let a = s.spawn(move || g.read(g.r));
            g.early.wait_parked();
            let b = s.spawn(move || g.read(g.b));
            g.late.arm();
            g.early.release();
            g.late.wait_parked();
            let deadline = Instant::now() + POLL_DEADLINE;
            while !waits::waited_on((g.rt.runtime_id, g.y.slot())) {
                assert!(Instant::now() < deadline, "B never waited on y");
                thread::sleep(Duration::from_millis(1));
            }
            let flag = if z_reads_b {
                &g.z_reads_b
            } else {
                &g.r_reads_b
            };
            flag.store(true, Ordering::SeqCst);
            g.late.release();
            (a.join().unwrap(), b.join().unwrap())
        });
        assert!(waits::WOUNDS.on(g.rt.runtime_id) > wounds, "no wound");
        let landings = g.rt.unwound_landings.lock().unwrap().clone();
        let outcome = g.outcome(a, b);
        g.rt.assert_invariant_i();
        (outcome, landings, g.r.slot())
    }
}

type Concurrent = (Vec<Result<i64, String>>, Vec<(u32, bool, bool, bool)>, u32);

fn dissolve_keeps_the_released_members_grouped(z_reads_b: bool) {
    let a_first = Dissolve::single_thread(true);
    let b_first = Dissolve::single_thread(false);
    for round in 0..20 {
        let (got, ..) = Dissolve::concurrent(z_reads_b);
        assert!(
            got == a_first || got == b_first,
            "round {round}: {got:?}, single-thread {a_first:?} or {b_first:?}"
        );
    }
}

/// The members a dissolve released before A's wound keep their group, so
/// B verifies it as a single thread would. Released alone, B computes
/// `y` outside it, and `r`'s read of `y` meets the unseeded `y` on B's
/// stack: a cycle error that no single-thread order raises.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_wound_after_an_episode_dissolve_keeps_the_released_members_grouped() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_wound_after_an_episode_dissolve_keeps_the_released_members_grouped",
        || dissolve_keeps_the_released_members_grouped(true),
    )
    .assert_clean();
}

/// A's wound after its episode at `r` dissolved lands `r` as before its
/// claim, back in its old group with `z`, `y` and `w`, which the cut
/// released and the retraction puts back: a cut is an exposure, so the
/// whole group retracts. Back in that group alone, `r` would be
/// group-verified by B, who cannot claim a member on its own stack and
/// computes `r` alone (seed 0x366148bf).
#[test]
#[cfg_attr(miri, ignore)]
fn shared_wound_of_a_dissolved_root_lands_no_record_naming_a_member_that_left() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_wound_of_a_dissolved_root_lands_no_record_naming_a_member_that_left",
        || {
            for z_reads_b in [true, false] {
                for round in 0..5 {
                    let (_, landings, r) = Dissolve::concurrent(z_reads_b);
                    assert!(
                        landings.iter().any(|&(slot, dissolved, ..)| slot == r && dissolved),
                        "round {round}: no wound landed r after its dissolve: {landings:?}"
                    );
                    assert!(
                        landings
                            .iter()
                            .all(|&(slot, _, _, grouped)| slot != r || grouped),
                        "round {round} z_reads_b {z_reads_b}: r landed out of its group: {landings:?}"
                    );
                    assert!(
                        landings.iter().all(|&(_, _, stale, _)| !stale),
                        "round {round} z_reads_b {z_reads_b}: {landings:?}"
                    );
                }
            }
        },
    )
    .assert_clean();
}

/// As above, with A's wound after it landed `w` and `z`: their landings
/// retract with the read, so B verifies the group as a single thread
/// would.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_wound_after_clean_landings_of_released_members_keeps_their_group() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_wound_after_clean_landings_of_released_members_keeps_their_group",
        || dissolve_keeps_the_released_members_grouped(false),
    )
    .assert_clean();
}

/// [`stale_edge_waits_for_cycle`] with B's read of `q` made inside a
/// `Local` runtime's query: the wound or cycle raised in the `Shared`
/// read reaches the `Local` entry around it, which is the thread's
/// outermost, and B gets the outcome it gets without the `Local` entry,
/// holding no reservation afterwards. With A older, B is wounded under the
/// `Local` entry and checks in its head start whether A took the claim,
/// which is the `Shared` runtime's.
fn stale_edge_waits_for_cycle_under_local(a_first: bool) {
    let g = StaleEdge::new(QReads::Plain, false);
    let checks = waits::OWN_CLAIM_CHECKS.on(g.rt.runtime_id);
    let q = g.q;
    let shared = Arc::clone(&g.rt);
    let (thread_a, thread_b) = g.run(a_first, move |_| {
        let local = Runtime::<crate::cells::Local>::new();
        let n = local.create_query(move |_| shared.get(q));
        let v = catch_unwind(AssertUnwindSafe(|| local.get(n)));
        assert!(crate::reserve::none_held(), "B kept a reservation");
        v.unwrap_or_else(|p| std::panic::resume_unwind(p))
    });
    assert_eq!(thread_a.join().unwrap(), 4);
    let b_result = thread_b.join().unwrap();
    if a_first {
        assert_eq!(b_result.unwrap(), 4);
    } else {
        let message = panic_message(b_result.unwrap_err());
        assert!(message.contains("dependency cycle"), "{message}");
        assert_eq!(g.rt.get(q), 4);
    }
    assert_wounded(g.rt.runtime_id);
    assert!(
        !a_first || waits::OWN_CLAIM_CHECKS.on(g.rt.runtime_id) > checks,
        "the wounded thread's head start never checked the claim its winner waited on"
    );
    g.rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_stale_edge_under_local_with_a_older() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_stale_edge_under_local_with_a_older",
        || stale_edge_waits_for_cycle_under_local(true),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_stale_edge_under_local_with_b_older() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_stale_edge_under_local_with_b_older",
        || stale_edge_waits_for_cycle_under_local(false),
    )
    .assert_clean();
}

/// A's read of `r`, then B's of `b`, on one thread.
fn released_member_reads<C: Cells>() -> Vec<Result<i64, String>> {
    let g = DissolveOn::<C>::new();
    let a = g.read(g.r);
    g.outcome(a, g.read(g.b))
}

/// **A cut's released members stay the owner's until its read returns.**
/// A's read of `r` on [`Dissolve`] dissolves its episode, releases `z`,
/// `y` and `w`, and parks. They are still A's, as in the unbounded run,
/// where its episode holds them until it ends: B's read of `b` waits on
/// A's `y` instead of computing `y` from a release A's attempt may still
/// retract, and both reads match A's first, as a single thread or Local
/// gives them.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_released_member_waits_for_the_cut_owners_read() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_released_member_waits_for_the_cut_owners_read",
        || {
            let want = released_member_reads::<Local>();
            assert_eq!(released_member_reads::<Shared>(), want);
            let g = Dissolve::new();
            let released = Gate::new();
            park_at(
                &g.rt,
                vec![(ParkPoint::MembersReleased(id(g.r)), released.clone())],
            );
            released.arm();
            let (states, b_waited, b_done_early, a, b) = thread::scope(|s| {
                let _released = release_parked_on_unwind();
                let g = &g;
                let a = s.spawn(move || g.read(g.r));
                released.wait_parked();
                let states: Vec<NodeState> = [g.z, g.y, g.w]
                    .map(|m| g.rt.nodes.get(m.slot()).state())
                    .into();
                let b = s.spawn(move || g.read(g.b));
                let deadline = Instant::now() + POLL_DEADLINE;
                let y = (g.rt.runtime_id, g.y.slot());
                while !waits::waited_on(y) && !b.is_finished() && Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(1));
                }
                let (b_waited, b_done_early) = (waits::waited_on(y), b.is_finished());
                released.release();
                (
                    states,
                    b_waited,
                    b_done_early,
                    a.join().unwrap(),
                    b.join().unwrap(),
                )
            });
            assert_eq!(
                states,
                [NodeState::Computing; 3],
                "a released member left A's claim before A's read returned"
            );
            assert!(!b_done_early, "B read w while A's attempt was unpublished");
            assert!(b_waited, "B never waited on y");
            assert_eq!(g.outcome(a, b), want);
            g.rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

fn group_record<C: Cells>(rt: &Runtime<C>, node: Incr<i64>) -> Option<Arc<GroupInfo>> {
    rt.inner.read().member_recs.get(&node.slot()).cloned()
}

/// A's read of `r` alone on [`DissolveOn`]: the records it leaves, whether
/// `z`, `y` or `w` is still in the cut group, and the values.
fn published_cut<C: Cells>() -> (Vec<String>, bool, Vec<Result<i64, String>>) {
    let g = DissolveOn::<C>::new();
    let cut = group_record(&g.rt, g.r).expect("r is forced into a group");
    let a = g.read(g.r);
    let in_cut = [g.z, g.y, g.w]
        .into_iter()
        .any(|m| group_record(&g.rt, m).is_some_and(|rec| Arc::ptr_eq(&rec, &cut)));
    (
        g.rt.stale_group_records(),
        in_cut,
        g.outcome(a, g.read(g.w)),
    )
}

/// Where A's attempt at [`Dissolve`]'s `top` is wounded after its
/// episode at `r` dissolved.
#[derive(Clone, Copy, Debug)]
enum CutWound {
    /// In `z`'s run, before A lands a released member.
    InZ,
    /// In `r`'s run, after `z` and `w` landed.
    InR,
    /// In `top`, after `r`, rerun reading only `k`, landed.
    AfterR,
}

/// **A cut's group exit publishes and retracts with the attempt.** On
/// [`Dissolve`] A's episode at `r` dissolves and releases `z`, `y` and
/// `w`, then B's read wounds A, at each [`CutWound`]. Right after the
/// retraction all four are back in their old group, whole, as before A's
/// read, so B's `group_verify` claims the group a single thread
/// verifies: a cut is an exposure, so the claim it keeps, `r`, lands held
/// even when its run read nothing held, and retracts with them. A read
/// that publishes takes the released set out of the cut group with no
/// record naming a node that left, as under Local.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_cut_group_exit_publishes_and_retracts_with_the_read() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_cut_group_exit_publishes_and_retracts_with_the_read",
        || {
            let local = published_cut::<Local>();
            assert_eq!(local.0, Vec::<String>::new(), "Local leaves a stale record");
            assert!(!local.1, "Local keeps a released member in the cut group");
            assert_eq!(published_cut::<Shared>(), local);
            for wound in [CutWound::InZ, CutWound::InR, CutWound::AfterR] {
                cut_exit_retracts(wound);
            }
        },
    )
    .assert_clean();
}

fn cut_exit_retracts(wound: CutWound) {
    let orders = [true, false].map(|a_first| {
        let g = DissolveOn::<Local>::new();
        if a_first {
            let a = g.read(g.top);
            g.outcome(a, g.read(g.b))
        } else {
            let b = g.read(g.b);
            g.outcome(g.read(g.top), b)
        }
    });
    let g = Dissolve::new();
    let old = group_record(&g.rt, g.r).expect("r is forced into a group");
    let (a_thread, b_thread) = (Arc::new(OnceLock::new()), Arc::new(OnceLock::new()));
    let at_retraction: Arc<OnceLock<(Vec<String>, Vec<bool>)>> = Arc::new(OnceLock::new());
    let rt_id = g.rt.runtime_id;
    let wounds = waits::WOUNDS.on(rt_id);
    {
        let rt = Arc::downgrade(&g.rt);
        let members = [g.r, g.z, g.y, g.w];
        let (a_thread, b_thread) = (a_thread.clone(), b_thread.clone());
        let (at_retraction, old) = (at_retraction.clone(), old.clone());
        let installed = g.rt.park_hook.set(Box::new(move |p| {
            let me = Some(thread::current().id());
            if p == ParkPoint::WoundRetracted && me == a_thread.get().copied() {
                let rt = rt.upgrade().expect("the runtime outlives its reads");
                let in_old = members
                    .map(|m| group_record(&rt, m).is_some_and(|r| Arc::ptr_eq(&r, &old)))
                    .into();
                let _ = at_retraction.set((rt.stale_group_records(), in_old));
            } else if me == b_thread.get().copied() && waits::WOUNDS.on(rt_id) > wounds {
                // B holds still from A's wound until A's retraction is seen,
                // so nothing B lands can hide or fake the retracted records.
                let deadline = Instant::now() + POLL_DEADLINE;
                while at_retraction.get().is_none() && Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }));
        assert!(installed.is_ok(), "park hook installed twice");
    }
    let b_waits = |g: &Dissolve| {
        let deadline = Instant::now() + POLL_DEADLINE;
        while !waits::waited_on((g.rt.runtime_id, g.y.slot())) {
            assert!(Instant::now() < deadline, "{wound:?}: B never waited on y");
            thread::sleep(Duration::from_millis(1));
        }
    };
    g.early.arm();
    let (a, b) = thread::scope(|s| {
        let _released = release_parked_on_unwind();
        let g = &g;
        let a = s.spawn(|| {
            a_thread.set(thread::current().id()).unwrap();
            g.read(g.top)
        });
        g.early.wait_parked();
        let b = s.spawn(|| {
            b_thread.set(thread::current().id()).unwrap();
            g.read(g.b)
        });
        if let CutWound::AfterR = wound {
            b_waits(g);
            g.r_alone.store(true, Ordering::SeqCst);
            g.top_reads_b.store(true, Ordering::SeqCst);
            g.early.release();
        } else {
            g.late.arm();
            g.early.release();
            g.late.wait_parked();
            b_waits(g);
            let flag = match wound {
                CutWound::InZ => &g.z_reads_b,
                _ => &g.r_reads_b,
            };
            flag.store(true, Ordering::SeqCst);
            g.late.release();
        }
        (a.join().unwrap(), b.join().unwrap())
    });
    assert!(waits::WOUNDS.on(rt_id) > wounds, "{wound:?}: no wound");
    if let CutWound::AfterR = wound {
        assert!(!g.r_alone.load(Ordering::SeqCst), "r never ran alone");
        assert_eq!(
            *g.r_at_top.lock().unwrap(),
            Some(NodeState::Computing),
            "r's landing after the cut was not held"
        );
    }
    let (stale, in_old) = at_retraction
        .get()
        .cloned()
        .unwrap_or_else(|| panic!("{wound:?}: A never retracted"));
    assert_eq!(
        stale,
        Vec::<String>::new(),
        "{wound:?}: a stale record after the retraction"
    );
    assert_eq!(
        in_old, [true; 4],
        "{wound:?}: r, z, y and w not back in their old group"
    );
    let outcome = g.outcome(a, b);
    assert!(
        orders.contains(&outcome),
        "{wound:?}: {outcome:?}, Local's orders {orders:?}"
    );
    g.rt.assert_invariant_i();
}

/// **A frame's kept value stays with its node.** After `k` is set, `x`
/// and `y` form a cycle, as do `u` and `v`, all four Fallback; `q`
/// reads `y`, `q2` reads `v`, and `r` reads `x`, `p` and `u` and then
/// `q2`. A's read of `r` holds the episode {x, y}; `p` reads `y` through
/// the view, so its frame holds its write, then claims `e`, so its own
/// landing consults nothing, and catches the wound its read of B's `q`
/// raises: its write keeps what it displaced for an entry its rewritten
/// pop never opens. A's retry holds the episode {u, v} and B2's `q2`
/// wounds it again: right after that retraction `u`'s slot holds its own
/// value from before the attempt, 5, not a value another write kept, and
/// nothing panics.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_kept_value_of_a_swallowed_wound_stays_with_its_node() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_kept_value_of_a_swallowed_wound_stays_with_its_node",
        || {
            fn build<C: Cells>(rt: &Runtime<C>, gates: [Gate; 5]) -> [Incr<i64>; 9] {
                let [gq, gq2, gp, gr, gh] = gates;
                let k = rt.create_input(0_i64);
                let y_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
                let v_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
                let x = {
                    let y_cell = y_cell.clone();
                    rt.create_query_with_fallback(
                        move |rt| match rt.get(k) {
                            0 => 1,
                            _ => rt.get(*y_cell.get().unwrap()) + 1,
                        },
                        |_c: &crate::cycle::Cycle| 10,
                    )
                };
                let y = rt.create_query_with_fallback(
                    move |rt| match rt.get(k) {
                        0 => 2,
                        _ => rt.get(x) + 100,
                    },
                    |_c: &crate::cycle::Cycle| 20,
                );
                let u = {
                    let v_cell = v_cell.clone();
                    rt.create_query_with_fallback(
                        move |rt| match rt.get(k) {
                            0 => 5,
                            _ => rt.get(*v_cell.get().unwrap()) + 1,
                        },
                        |_c: &crate::cycle::Cycle| 50,
                    )
                };
                let v = rt.create_query_with_fallback(
                    move |rt| match rt.get(k) {
                        0 => 6,
                        _ => rt.get(u) + 1,
                    },
                    |_c: &crate::cycle::Cycle| 60,
                );
                let e = rt.create_query(move |rt| rt.get(k) * 3);
                let h = rt.create_query(move |rt| {
                    gh.park();
                    rt.get(k)
                });
                let q = rt.create_query(move |rt| match rt.get(k) {
                    0 => 0,
                    _ => {
                        gq.park();
                        rt.get(y) + 1
                    }
                });
                let q2 = rt.create_query(move |rt| match rt.get(k) {
                    0 => 0,
                    _ => {
                        rt.get(h);
                        gq2.park();
                        rt.get(v) + 1
                    }
                });
                let p = rt.create_query(move |rt| match rt.get(k) {
                    0 => 777,
                    _ => {
                        rt.get(y);
                        rt.get(e);
                        gp.park();
                        let _ = catch_unwind(AssertUnwindSafe(|| rt.get(q)));
                        778
                    }
                });
                let r = rt.create_query(move |rt| match rt.get(k) {
                    0 => 0,
                    _ => {
                        let sum = rt.get(x) + rt.get(p) + rt.get(u);
                        // A's retry, after the first wound, parks once here.
                        if waits::WOUNDS.on(rt.runtime_id) > 0 {
                            gr.park();
                        }
                        sum + rt.get(q2)
                    }
                });
                y_cell.set(y).unwrap();
                v_cell.set(v).unwrap();
                assert_eq!([rt.get(r), rt.get(q), rt.get(q2), rt.get(u)], [0, 0, 0, 5]);
                rt.set(k, 1);
                [r, q, q2, x, y, u, v, p, h]
            }

            let local = Runtime::<Local>::new();
            let [r, q, q2, ..] = build(&local, [(); 5].map(|_| Gate::new()));
            let (lq, lq2) = (local.get(q), local.get(q2));
            let want = (local.get(r), lq, lq2);

            let rt = Arc::new(Runtime::<Shared>::new());
            let [gq, gq2, gp, gr, gh] = [(); 5].map(|_| Gate::new());
            let [r, q, q2, _, y, u, v, _, h] =
                build(&rt, [&gq, &gq2, &gp, &gr, &gh].map(Gate::clone));
            let a_thread = Arc::new(OnceLock::new());
            let b2_thread = Arc::new(OnceLock::new());
            let r_released = Arc::new(AtomicBool::new(false));
            let u_retracted: Arc<OnceLock<Option<i64>>> = Arc::new(OnceLock::new());
            let wounds = waits::WOUNDS.on(rt.runtime_id);
            {
                let weak = Arc::downgrade(&rt);
                let (a_thread, b2_thread) = (a_thread.clone(), b2_thread.clone());
                let (r_released, seen) = (r_released.clone(), u_retracted.clone());
                let installed = rt.park_hook.set(Box::new(move |p| {
                    let me = Some(thread::current().id());
                    if !r_released.load(Ordering::SeqCst) || seen.get().is_some() {
                        return;
                    }
                    if p == ParkPoint::WoundRetracted && me == a_thread.get().copied() {
                        let rt = weak.upgrade().expect("the runtime outlives its reads");
                        let _ = seen.set(rt.slot_value_for_test(u));
                        // A's retry would take u and v back before B2 reads v and
                        // be wounded again, so A waits for B2's read to land.
                        let deadline = Instant::now() + Duration::from_secs(4);
                        while rt.nodes.get(q2.slot()).state() != NodeState::Clean
                            && Instant::now() < deadline
                        {
                            thread::sleep(Duration::from_millis(1));
                        }
                    } else if me == b2_thread.get().copied() {
                        // B2 holds still until A's retraction is seen.
                        let deadline = Instant::now() + Duration::from_secs(4);
                        while seen.get().is_none() && Instant::now() < deadline {
                            thread::sleep(Duration::from_millis(1));
                        }
                    }
                }));
                assert!(installed.is_ok(), "park hook installed twice");
            }
            let spawn = |id: Option<Arc<OnceLock<thread::ThreadId>>>, node: Incr<i64>| {
                let rt = rt.clone();
                thread::spawn(move || {
                    if let Some(id) = id {
                        id.set(thread::current().id()).unwrap();
                    }
                    catch_unwind(AssertUnwindSafe(|| rt.get(node))).map_err(panic_message)
                })
            };
            let wait_on = |node: Incr<i64>, what: &str| {
                let deadline = Instant::now() + Duration::from_secs(4);
                while !waits::waited_on((rt.runtime_id, node.slot())) {
                    assert!(Instant::now() < deadline, "{what}");
                    thread::sleep(Duration::from_millis(1));
                }
            };
            let join = |h: thread::JoinHandle<Result<i64, String>>, who: &str| {
                let deadline = Instant::now() + Duration::from_secs(8);
                while !h.is_finished() {
                    assert!(Instant::now() < deadline, "{who}'s read hung");
                    thread::sleep(Duration::from_millis(1));
                }
                h.join().unwrap()
            };

            gq.arm();
            gq2.arm();
            gp.arm();
            gr.arm();
            gh.arm();
            // B2's wait on H's claim of h takes its ticket first, so A, whose
            // ticket comes from its first wait, is the younger at both wounds.
            let h_thread = spawn(None, h);
            gh.wait_parked();
            let b2 = spawn(Some(b2_thread.clone()), q2);
            wait_on(h, "B2 never waited on h");
            gh.release();
            gq2.wait_parked();
            let b = spawn(None, q);
            gq.wait_parked();
            let a = spawn(Some(a_thread.clone()), r);
            gp.wait_parked();
            gq.release();
            wait_on(y, "B never waited on y");
            gp.release();
            gr.wait_parked();
            assert!(waits::WOUNDS.on(rt.runtime_id) > wounds, "no first wound");
            gq2.release();
            wait_on(v, "B2 never waited on v");
            r_released.store(true, Ordering::SeqCst);
            gr.release();
            let (a, b, b2) = (join(a, "A"), join(b, "B"), join(b2, "B2"));
            join(h_thread, "H").unwrap();

            assert_eq!(
                u_retracted.get().copied(),
                Some(Some(5)),
                "u's slot after the second retraction"
            );
            assert_eq!((a, b, b2), (Ok(want.0), Ok(want.1), Ok(want.2)));
            rt.assert_invariant_i();
        },
    )
    .assert_clean();
}

/// **A retraction restores the value a re-run overwrote.** Once `i` is
/// 1, `c` and `b` (Fixpoint) read each other, `a` reads `c`, and `b`
/// reads `a` only while it sees `c`'s seed. A's read of `o` runs the
/// episode at `c`: its first round reaches `a`, its last does not, so the
/// cut to {c, b} releases `a` unrun and `a`'s entry records no value. `o`
/// then reads `a`, and A claims it again and computes it alone. B's `qb`,
/// which reads `b`, then wounds A: right after the retraction `a`'s slot
/// holds its value from before A's read, 30, not the 105 the wounded
/// attempt wrote.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_retraction_restores_a_released_members_rerun() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_retraction_restores_a_released_members_rerun",
        || {
            fn build<C: Cells>(rt: &Runtime<C>, gates: [Gate; 3]) -> [Incr<i64>; 5] {
                let [gh, gb, go] = gates;
                let i = rt.create_input(0_i64);
                let fixpoint = || QueryOpts {
                    cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64).max_iterations(20)),
                    ..QueryOpts::default()
                };
                let a_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
                let b_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
                let c = {
                    let b_cell = b_cell.clone();
                    rt.create_query_with(
                        move |rt| match rt.get(i) {
                            0 => 7,
                            _ => rt.get(*b_cell.get().unwrap()).min(5),
                        },
                        fixpoint(),
                    )
                };
                let a = rt.create_query(move |rt| match rt.get(i) {
                    0 => 30,
                    _ => rt.get(c) + 100,
                });
                let b = {
                    let a_cell = a_cell.clone();
                    rt.create_query_with(
                        move |rt| match rt.get(i) {
                            0 => 8,
                            _ => match rt.get(c) {
                                0 => rt.get(*a_cell.get().unwrap()),
                                seen => seen + 1,
                            },
                        },
                        fixpoint(),
                    )
                };
                let h = rt.create_query(move |rt| {
                    gh.park();
                    rt.get(i)
                });
                let qb = rt.create_query(move |rt| match rt.get(i) {
                    0 => 0,
                    _ => {
                        rt.get(h);
                        gb.park();
                        rt.get(b) + 1000
                    }
                });
                let o = rt.create_query(move |rt| match rt.get(i) {
                    // c read last, so a walk of o enters the cycle at c.
                    0 => rt.get(a) + rt.get(c),
                    _ => {
                        let read = rt.get(c) + rt.get(a);
                        go.park();
                        read + rt.get(qb)
                    }
                });
                a_cell.set(a).unwrap();
                b_cell.set(b).unwrap();
                assert_eq!([rt.get(o), rt.get(a), rt.get(qb)], [37, 30, 0]);
                rt.set(i, 1);
                [o, qb, a, h, b]
            }

            let local = Runtime::<Local>::new();
            let [o, qb, ..] = build(&local, [(); 3].map(|_| Gate::new()));
            let lqb = local.get(qb);
            let want = (local.get(o), lqb);

            let rt = Arc::new(Runtime::<Shared>::new());
            let [gh, gb, go] = [(); 3].map(|_| Gate::new());
            let [o, qb, a, h, b_node] = build(&rt, [&gh, &gb, &go].map(Gate::clone));
            let (a_thread, b_thread) = (Arc::new(OnceLock::new()), Arc::new(OnceLock::new()));
            let o_released = Arc::new(AtomicBool::new(false));
            let a_retracted: Arc<OnceLock<Option<i64>>> = Arc::new(OnceLock::new());
            {
                let weak = Arc::downgrade(&rt);
                let (a_thread, b_thread) = (a_thread.clone(), b_thread.clone());
                let (o_released, seen) = (o_released.clone(), a_retracted.clone());
                let installed = rt.park_hook.set(Box::new(move |p| {
                    let me = Some(thread::current().id());
                    if !o_released.load(Ordering::SeqCst) || seen.get().is_some() {
                        return;
                    }
                    if p == ParkPoint::WoundRetracted && me == a_thread.get().copied() {
                        let rt = weak.upgrade().expect("the runtime outlives its reads");
                        let _ = seen.set(rt.slot_value_for_test(a));
                        // A's retry would take the cut back before B reads b.
                        let deadline = Instant::now() + Duration::from_secs(4);
                        while rt.nodes.get(qb.slot()).state() != NodeState::Clean
                            && Instant::now() < deadline
                        {
                            thread::sleep(Duration::from_millis(1));
                        }
                    } else if me == b_thread.get().copied() {
                        // B holds still until A's retraction is seen.
                        let deadline = Instant::now() + Duration::from_secs(4);
                        while seen.get().is_none() && Instant::now() < deadline {
                            thread::sleep(Duration::from_millis(1));
                        }
                    }
                }));
                assert!(installed.is_ok(), "park hook installed twice");
            }
            let spawn = |id: Option<Arc<OnceLock<thread::ThreadId>>>, node: Incr<i64>| {
                let rt = rt.clone();
                thread::spawn(move || {
                    if let Some(id) = id {
                        id.set(thread::current().id()).unwrap();
                    }
                    catch_unwind(AssertUnwindSafe(|| rt.get(node))).map_err(panic_message)
                })
            };
            let wait_on = |slot: u32, what: &str| {
                let deadline = Instant::now() + Duration::from_secs(4);
                while !waits::waited_on((rt.runtime_id, slot)) {
                    assert!(Instant::now() < deadline, "{what}");
                    thread::sleep(Duration::from_millis(1));
                }
            };
            let join = |h: thread::JoinHandle<Result<i64, String>>, who: &str| {
                let deadline = Instant::now() + Duration::from_secs(8);
                while !h.is_finished() {
                    assert!(Instant::now() < deadline, "{who}'s read hung");
                    thread::sleep(Duration::from_millis(1));
                }
                h.join().unwrap()
            };

            gh.arm();
            gb.arm();
            go.arm();
            // B's wait on H's claim of h takes its ticket first, so A is the
            // younger at the wound.
            let h_thread = spawn(None, h);
            gh.wait_parked();
            let b = spawn(Some(b_thread.clone()), qb);
            wait_on(h.slot(), "B never waited on h");
            gh.release();
            gb.wait_parked();
            let a_read = spawn(Some(a_thread.clone()), o);
            go.wait_parked();
            gb.release();
            wait_on(b_node.slot(), "B never waited on b");
            o_released.store(true, Ordering::SeqCst);
            go.release();
            let (got_o, got_qb) = (join(a_read, "A"), join(b, "B"));
            join(h_thread, "H").unwrap();

            assert_eq!(
                a_retracted.get().copied(),
                Some(Some(30)),
                "a's slot after the retraction"
            );
            assert_eq!((got_o, got_qb), (Ok(want.0), Ok(want.1)));
            rt.assert_invariant_i();
            rt.assert_released_entries_match_deps();
        },
    )
    .assert_clean();
}

/// **A retraction restores a released claim's entry.** `x` reads `y`,
/// which reads `x` once `j` is 0, while `m` is not 1; both are Fixpoint
/// heads, so a walk enters the cycle at a head whichever it reaches
/// first. Each set of `m` changes a dep of `x`, so the reads after it
/// recompute `x` rather than settle it from `y`. With `k` at 1, `x`'s
/// run panics inside the episode, and `y`'s claim is released New over
/// `[j]` and the `x` its run read, with an entry keeping `[j]`. With `k`
/// back at 0 and `m` at 2, B's `c` reads `x`: the episode reruns `y` from
/// New and lands it held Clean, which drops the entry, and `c` then raises
/// a wound. The retraction lands `y` New over the same two deps, so its
/// entry is back and a walk still expands only `j`.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_retraction_restores_a_released_claims_entry() {
    let other = Arc::new(Runtime::<Shared>::new());
    let j = other.create_input(1_i64);
    let k = other.create_input(0_i64);
    let m = other.create_input(1_i64);
    let y_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let x = {
        let y_cell = y_cell.clone();
        other.create_query_with(
            move |rt| {
                if rt.get(m) == 1 {
                    return 5;
                }
                let v = rt.get(*y_cell.get().unwrap());
                assert!(rt.get(k) == 0, "x boom");
                (v + 1).min(3)
            },
            QueryOpts {
                cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64)),
                ..QueryOpts::default()
            },
        )
    };
    let y = other.create_query_with(
        move |rt| if rt.get(j) == 1 { 1 } else { rt.get(x) },
        QueryOpts {
            cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64)),
            ..QueryOpts::default()
        },
    );
    y_cell.set(y).unwrap();
    assert_eq!((other.get(x), other.get(y)), (5, 1));

    other.set(j, 0);
    other.set(k, 1);
    other.set(m, 0);
    let err = catch_unwind(AssertUnwindSafe(|| other.get(x))).unwrap_err();
    assert_eq!(err.downcast_ref::<&str>(), Some(&"x boom"));
    let kept =
        move |rt: &Runtime<Shared>| rt.released_kept(NodeId(y.slot()), rt.nodes.get(y.slot()));
    assert_eq!(other.state_of(y), NodeState::New);
    assert_eq!(kept(&other), Some(1), "y's entry after the release");
    other.assert_released_entries_match_deps();

    other.set(k, 0);
    other.set(m, 2);
    let wounded = Arc::new(AtomicBool::new(false));
    let c = {
        let wounded = wounded.clone();
        other.create_query(move |rt| {
            let v = rt.get(x);
            if !wounded.swap(true, Ordering::SeqCst) {
                rt.raise_wound();
            }
            v
        })
    };
    let rt = Runtime::<Shared>::new();
    let retracted: Arc<OnceLock<(NodeState, Option<u32>)>> = Arc::new(OnceLock::new());
    let root = {
        let (other, retracted) = (other.clone(), retracted.clone());
        rt.create_query(move |_| {
            let first = catch_unwind(AssertUnwindSafe(|| other.get(c)));
            if first.is_err() {
                let _ = retracted.set((other.state_of(y), kept(&other)));
            }
            first.unwrap_or(0)
        })
    };
    assert_eq!(rt.get(root), 3);
    assert!(wounded.load(Ordering::SeqCst));
    assert_eq!(
        retracted.get().copied(),
        Some((NodeState::New, Some(1))),
        "y's state and entry right after the retraction"
    );
    assert_eq!(other.get(y), 3);
    other.assert_released_entries_match_deps();
    other.assert_invariant_i();
    assert_eq!(crate::hold::held_count(), 0);
}
