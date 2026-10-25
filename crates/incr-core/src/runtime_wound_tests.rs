//! Wound-wait under `Shared`: claim waits that close a waits-for cycle
//! across threads wound the younger thread, which releases its claims and
//! retries. Each test runs in a child process, because a regression hangs
//! and because the wound counter is process-wide.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Barrier, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use super::observe_impl::observed_is_clear;
use super::race_tests::{id, park_at, Gate};
use super::{NodeId, ParkPoint, Runtime};
use crate::cells::Shared;
use crate::cycle::{CycleStrategy, Fixpoint, QueryOpts};
use crate::handle::Incr;
use crate::state::NodeState;
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
        let wounds = waits::WOUNDS.load(Ordering::Relaxed);
        let first = if a_older { &g.b_gate } else { &g.a_gate };
        first.arm();
        let (a, b) = thread::scope(|s| {
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
            wait_registered(1);
            first.release();
            (a.join().unwrap(), b.join().unwrap())
        });
        assert!(
            waits::WOUNDS.load(Ordering::Relaxed) > wounds,
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
            let deadline = Instant::now() + Duration::from_secs(20);
            while waits::registered() == 0 {
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
/// All four keep a record of one group. In an episode's first iteration
/// the seeds are 0, so `r` reaches no other member: A's episode at `r`
/// dissolves and releases `z`, `y` and `w` ungrouped, while B's at `y`
/// keeps {y, z} and lets `r` join. B waits on A's claim of `y` first, so
/// B is older; A parks in `z` after the dissolve until B, which took `y`,
/// waits on `r`, and A's read of `y` closes the waits-for cycle. With
/// `first_reads_y`, `z`'s run after the dissolve reads `y` first, so A is
/// wounded before it lands a released member.
struct Dissolve {
    rt: Arc<Runtime<Shared>>,
    r: Incr<i64>,
    z: Incr<i64>,
    y: Incr<i64>,
    w: Incr<i64>,
    /// Parks `z`'s run in A's episode, with every member claimed.
    early: Gate,
    /// Parks `z`'s run after the dissolve, A holding only `r` and `z`.
    late: Gate,
    first_reads_y: Arc<AtomicBool>,
}

impl Dissolve {
    fn new() -> Self {
        let rt = Arc::new(Runtime::<Shared>::new());
        let k = rt.create_input(0_i64);
        let z_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let y_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let w_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let fixpoint = || QueryOpts {
            cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64).max_iterations(20)),
            ..QueryOpts::default()
        };
        let r = {
            let (z_cell, y_cell) = (z_cell.clone(), y_cell.clone());
            rt.create_query_with(
                move |rt| match rt.get(k) {
                    0 => 1,
                    _ if rt.get(*z_cell.get().unwrap()) == 0 => 1,
                    _ => (rt.get(*y_cell.get().unwrap()) + 1).min(50),
                },
                fixpoint(),
            )
        };
        let (early, late) = (Gate::new(), Gate::new());
        let first_reads_y = Arc::new(AtomicBool::new(false));
        let z = {
            let (y_cell, w_cell) = (y_cell.clone(), w_cell.clone());
            let (early, late) = (early.clone(), late.clone());
            let first_reads_y = first_reads_y.clone();
            rt.create_query_with(
                move |rt| {
                    if rt.get(k) == 0 {
                        return 0;
                    }
                    late.park();
                    // B holds `y`, so this read wounds A and never returns.
                    if first_reads_y.load(Ordering::SeqCst) {
                        rt.get(*y_cell.get().unwrap());
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
        z_cell.set(z).unwrap();
        y_cell.set(y).unwrap();
        w_cell.set(w).unwrap();
        assert!(r.slot() < z.slot() && z.slot() < y.slot() && y.slot() < w.slot());
        assert_eq!((rt.get(r), rt.get(z), rt.get(y), rt.get(w)), (1, 0, 0, 0));
        rt.force_group_for_test(&[r.erase(), z.erase(), y.erase(), w.erase()]);
        rt.set(k, 1);
        Dissolve {
            rt,
            r,
            z,
            y,
            w,
            early,
            late,
            first_reads_y,
        }
    }

    fn read(&self, node: Incr<i64>) -> Result<i64, String> {
        catch_unwind(AssertUnwindSafe(|| self.rt.get(node))).map_err(panic_message)
    }

    /// A's read of `r`, B's of `y`, then a read of every node.
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

    fn single_thread(a_first: bool) -> Vec<Result<i64, String>> {
        let g = Dissolve::new();
        if a_first {
            let a = g.read(g.r);
            g.outcome(a, g.read(g.y))
        } else {
            let b = g.read(g.y);
            g.outcome(g.read(g.r), b)
        }
    }

    fn concurrent(first_reads_y: bool) -> Vec<Result<i64, String>> {
        let g = Dissolve::new();
        let wounds = waits::WOUNDS.load(Ordering::Relaxed);
        g.early.arm();
        let (a, b) = thread::scope(|s| {
            let g = &g;
            let a = s.spawn(move || g.read(g.r));
            g.early.wait_parked();
            let b = s.spawn(move || g.read(g.y));
            g.late.arm();
            g.early.release();
            g.late.wait_parked();
            let deadline = Instant::now() + Duration::from_secs(20);
            while !waits::waited_on((g.rt.runtime_id, g.r.slot())) {
                assert!(Instant::now() < deadline, "B never waited on r");
                thread::sleep(Duration::from_millis(1));
            }
            g.first_reads_y.store(first_reads_y, Ordering::SeqCst);
            g.late.release();
            (a.join().unwrap(), b.join().unwrap())
        });
        assert!(waits::WOUNDS.load(Ordering::Relaxed) > wounds, "no wound");
        let outcome = g.outcome(a, b);
        g.rt.assert_invariant_i();
        outcome
    }
}

fn dissolve_keeps_the_released_members_grouped(first_reads_y: bool) {
    let a_first = Dissolve::single_thread(true);
    let b_first = Dissolve::single_thread(false);
    for round in 0..20 {
        let got = Dissolve::concurrent(first_reads_y);
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
#[ignore = "T8-44e3-clean-landing-variant"]
fn shared_wound_after_an_episode_dissolve_keeps_the_released_members_grouped() {
    crate::child_test::run_in_child(
        "runtime::wound_tests::shared_wound_after_an_episode_dissolve_keeps_the_released_members_grouped",
        || dissolve_keeps_the_released_members_grouped(true),
    )
    .assert_clean();
}

/// As above, with A landing `w` and `z` Clean outside the group before its
/// wound: B's group verify fails on the Clean `z` and computes `y` alone.
#[test]
#[ignore = "T8-44e3-clean-landing-variant: a wounded read's Clean landings of released members"]
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
    let checks = waits::OWN_CLAIM_CHECKS.load(Ordering::Relaxed);
    let g = StaleEdge::new(QReads::Plain, false);
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
    assert_wounded();
    assert!(
        !a_first || waits::OWN_CLAIM_CHECKS.load(Ordering::Relaxed) > checks,
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
