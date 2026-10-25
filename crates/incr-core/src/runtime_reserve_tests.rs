//! Doom reservations end with their owner: at a wound, when a cycle
//! unwind drops the owner's context before its retry, and when the
//! owner's retry panics. Each test parks the dooming thread holding a
//! reservation, has another thread wait on it, and checks that both
//! threads finish with a serial order's outcomes.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::SeqCst};
use std::sync::{mpsc, Arc, OnceLock};
use std::thread;
use std::time::Duration;

use super::race_tests::{await_reservation_wait, id, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::Shared;
use crate::child_test::run_in_child;
use crate::cycle::{QueryOpts, Replay};
use crate::handle::Incr;

type Rt = Runtime<Shared>;
type Outcome = Result<u64, String>;

const JOIN_TIMEOUT: Duration = Duration::from_secs(20);

fn replayable<T>() -> QueryOpts<T> {
    QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    }
}

fn outcome(f: impl FnOnce() -> u64) -> Outcome {
    catch_unwind(AssertUnwindSafe(f)).map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_default()
    })
}

/// A named thread whose [`Bounded::join`] fails the test if it has not
/// returned by [`JOIN_TIMEOUT`]: a reservation that outlives its owner
/// leaves its waiter spinning forever.
struct Bounded(mpsc::Receiver<Outcome>);

fn spawn(name: &str, f: impl FnOnce() -> u64 + Send + 'static) -> Bounded {
    let (tx, rx) = mpsc::channel();
    thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let _ = tx.send(outcome(f));
        })
        .unwrap();
    Bounded(rx)
}

impl Bounded {
    fn join(self) -> Outcome {
        self.0
            .recv_timeout(JOIN_TIMEOUT)
            .expect("a thread waiting on a reservation never returned")
    }
}

/// Parks thread `name` at `point` once `gate` is armed.
fn hook(rt: &Rt, name: &'static str, point: ParkPoint, gate: Gate) {
    let installed = rt.park_hook.set(Box::new(move |p| {
        if p == point && thread::current().name() == Some(name) {
            gate.park();
        }
    }));
    assert!(installed.is_ok(), "park hook installed twice");
}

/// T reads X with `try_get`, X reads Y, Y reads W with `try_get`, and W
/// parks on its gate once armed and then reads X with `try_get`: a cycle
/// through X, Y and W.
struct Cross {
    rt: Arc<Rt>,
    t: Incr<u64>,
    x: Incr<u64>,
    y: Incr<u64>,
    w: Incr<u64>,
    w_gate: Gate,
}

fn cross(rt: Rt) -> Cross {
    let rt = Arc::new(rt);
    rt.set_placeholder(0_u64);
    let xc: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let w_gate = Gate::new();
    let w = {
        let (xc, gate) = (xc.clone(), w_gate.clone());
        rt.create_query_with(
            move |rt: &Rt| {
                gate.park();
                rt.try_get(*xc.get().unwrap()).map_or(5, |v| v + 100)
            },
            replayable(),
        )
    };
    let y = rt.create_query_with(
        move |rt: &Rt| rt.try_get(w).map_or(50, |v| v + 10),
        replayable(),
    );
    let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 1, replayable());
    xc.set(x).unwrap();
    let t = rt.create_query_with(
        move |rt: &Rt| rt.try_get(x).map_or(1000, |v| v + 1),
        replayable(),
    );
    Cross {
        rt,
        t,
        x,
        y,
        w,
        w_gate,
    }
}

/// **A wound ends the owner's reservations**, at budget 2 on [`Cross`]:
/// O reads W first, so its ticket is older, and parks in W's closure
/// holding W. R reads T, dooms on Y below X, and parks before Y's entry,
/// reserving T, X and Y. O's W then reads X and waits on its reservation;
/// R's entry Y reads W and waits on O. The cycle passes through R's
/// reservation, so R, the younger, is wounded: its contexts drop with
/// their reservations, O goes on, and the reads are the unbounded run's
/// with W's first.
#[test]
fn shared_reservation_ends_when_its_owner_is_wounded() {
    let s = cross(Rt::new());
    s.rt.set_max_frames(2);
    let entry = Gate::new();
    hook(&s.rt, "r", ParkPoint::BeforeEntry(id(s.y)), entry.clone());
    s.w_gate.arm();
    let o = {
        let (rt, w) = (s.rt.clone(), s.w);
        spawn("o", move || rt.get(w))
    };
    s.w_gate.wait_parked();
    entry.arm();
    let r = {
        let (rt, t) = (s.rt.clone(), s.t);
        spawn("r", move || rt.get(t))
    };
    entry.wait_parked();
    let wounds = crate::waits::WOUNDS.load(SeqCst);
    let waits = s.rt.reservation_waits.load(SeqCst);
    s.w_gate.release();
    await_reservation_wait(&s.rt, waits);
    entry.release();
    let (w, t) = (o.join(), r.join());

    let u = cross(Rt::new());
    u.rt.set_max_frames(u32::MAX);
    let want = (outcome(|| u.rt.get(u.w)), outcome(|| u.rt.get(u.t)));
    assert_eq!((w, t), want);
    assert!(
        crate::waits::WOUNDS.load(SeqCst) > wounds,
        "no wound passed through the reservation"
    );
    s.rt.assert_invariant_i();
}

/// **A cycle unwind drops the owner's context before its retry**, at
/// budget 2 on [`Cross`] read from T: R dooms on Y below X and parks
/// before Y's entry, reserving X; another thread's read of X waits. Y's
/// entry reads W, whose read of X is a virtual hit that closes the
/// unseeded cycle, and the context ends with that reservation still
/// held. The waiting read then runs X itself, and both reads are the
/// unbounded run's with T's first.
#[test]
fn shared_reservation_ends_when_its_context_drops() {
    let s = cross(Rt::new());
    s.rt.set_max_frames(2);
    let entry = Gate::new();
    hook(&s.rt, "r", ParkPoint::BeforeEntry(id(s.y)), entry.clone());
    entry.arm();
    let ended = Arc::new(AtomicU32::new(0));
    let r = {
        let (rt, t, ended) = (s.rt.clone(), s.t, ended.clone());
        spawn("r", move || {
            let v = rt.get(t);
            ended.store(crate::reserve::context_ends(), SeqCst);
            v
        })
    };
    entry.wait_parked();
    let waits = s.rt.reservation_waits.load(SeqCst);
    let other = {
        let (rt, x) = (s.rt.clone(), s.x);
        spawn("other", move || rt.get(x))
    };
    await_reservation_wait(&s.rt, waits);
    entry.release();
    let (t, x_read) = (r.join(), other.join());

    let u = cross(Rt::new());
    u.rt.set_max_frames(u32::MAX);
    let want = (outcome(|| u.rt.get(u.t)), outcome(|| u.rt.get(u.x)));
    assert_eq!((t, x_read), want);
    assert!(
        ended.load(SeqCst) > 0,
        "no reservation ended with its context"
    );
    s.rt.assert_invariant_i();
}

/// **The owner's retry panics**, at budget 1: T reads X, X reads Y, and Y
/// panics once switched. R dooms on X and then on Y, parks before Y's
/// entry with X reserved, and Y is switched; another thread's reads of X
/// wait. Y's entry panics, X's retry lands Failed on it and T's panics out
/// of R's read, and the waiting reads get the unbounded run's outcomes
/// with T's read first.
#[test]
fn shared_reservation_ends_when_its_owners_retry_panics() {
    type Shape = (Arc<Rt>, Incr<u64>, Incr<u64>, Incr<u64>, Arc<AtomicBool>);
    fn shape(rt: Rt) -> Shape {
        let rt = Arc::new(rt);
        rt.set_placeholder(0_u64);
        let boom = Arc::new(AtomicBool::new(false));
        let y = {
            let boom = boom.clone();
            rt.create_query_with(
                move |_: &Rt| {
                    if boom.load(SeqCst) {
                        panic!("y boom");
                    }
                    7
                },
                replayable(),
            )
        };
        let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 1, replayable());
        let t = rt.create_query_with(move |rt: &Rt| rt.get(x) + 1, replayable());
        (rt, t, x, y, boom)
    }
    let (rt, t, x, y, boom) = shape(Rt::new());
    rt.set_max_frames(1);
    let entry = Gate::new();
    hook(&rt, "r", ParkPoint::BeforeEntry(id(y)), entry.clone());
    entry.arm();
    let r = {
        let rt = rt.clone();
        spawn("r", move || rt.get(t))
    };
    entry.wait_parked();
    let waits = rt.reservation_waits.load(SeqCst);
    boom.store(true, SeqCst);
    let other = {
        let rt = rt.clone();
        spawn("other", move || rt.get(x))
    };
    await_reservation_wait(&rt, waits);
    entry.release();
    let (t_read, x_read) = (r.join(), other.join());

    let (u, ut, ux, _, uboom) = shape(Rt::new());
    u.set_max_frames(u32::MAX);
    uboom.store(true, SeqCst);
    let want = (outcome(|| u.get(ut)), outcome(|| u.get(ux)));
    assert_eq!((t_read, x_read), want);
    rt.assert_invariant_i();
}

fn reserved_elsewhere(claim: crate::waits::Claim) -> bool {
    thread::spawn(move || crate::reserve::elsewhere(claim, 1))
        .join()
        .unwrap()
}

/// A read's reservation of its own node and a context's reservation of
/// the same node each keep it until both have ended, whichever ends first.
#[test]
fn a_read_and_a_context_both_reserving_a_node_keep_it_until_both_end() {
    use crate::reserve::{end_context, end_contexts, note, note_root, Root, Scope};
    let rt = Rt::new();
    let claim = (rt.runtime_id, 7);

    {
        let _scope = Scope::new(Some(5));
        note(claim, 1);
    }
    let root = Root::arm(true, claim);
    note_root(claim, 1);
    end_contexts();
    assert!(
        reserved_elsewhere(claim),
        "the context's end ended the read's"
    );
    drop(root);
    assert!(!reserved_elsewhere(claim));

    let root = Root::arm(true, claim);
    note_root(claim, 1);
    {
        let _scope = Scope::new(Some(6));
        note(claim, 1);
    }
    drop(root);
    assert!(
        reserved_elsewhere(claim),
        "the read's end ended the context's"
    );
    end_context(6);
    assert!(!reserved_elsewhere(claim));
}

/// A read nested in a read of the same node, whose walk lands the node
/// Dirty, leaves the reservation to the enclosing read, which takes the
/// value after it.
#[test]
fn a_nested_read_of_the_same_node_leaves_its_reservation_to_the_enclosing_read() {
    use crate::reserve::{note_root, Root};
    let rt = Rt::new();
    let claim = (rt.runtime_id, 7);
    let outer = Root::arm(true, claim);
    let inner = Root::arm(true, claim);
    note_root(claim, 1);
    drop(inner);
    assert!(reserved_elsewhere(claim), "the nested read ended it");
    drop(outer);
    assert!(!reserved_elsewhere(claim));
}

/// A Shared read nested in a Local runtime's outermost entry dooms at
/// budget 1 and reserves the claims its doom released. Once the Local
/// read has returned no entry is open on the thread, so none of those
/// reservations may be left: after a set, another thread recomputes the
/// chain without waiting on them.
#[test]
fn shared_reservations_end_with_the_local_entry_around_them() {
    let shared = Arc::new(Rt::new());
    shared.set_placeholder(0_u64);
    shared.set_max_frames(1);
    let x = shared.create_input(1_u64);
    let mut top = x;
    for _ in 0..8 {
        let below = top;
        top = shared.create_query_with(move |rt: &Rt| rt.get(below) + 1, replayable());
    }
    let local = Runtime::<crate::cells::Local>::new();
    let read = {
        let shared = shared.clone();
        local.create_query(move |_| shared.get(top))
    };
    assert_eq!(local.get(read), 9);
    assert!(!shared.dooms.lock().unwrap().is_empty(), "no doom ran");
    assert!(
        crate::reserve::none_held(),
        "a reservation outlived every entry on the thread"
    );
    shared.set(x, 2);
    let other = {
        let shared = shared.clone();
        spawn("other", move || shared.get(top))
    };
    assert_eq!(other.join(), Ok(10));
}

/// The same nesting where the chain's leaf panics: the doomed read's
/// retry panics out of the Shared read and the Local entry around it.
/// No entry is open once it has, so no reservation may be left: after
/// the leaf is cured, another thread recomputes the chain without
/// waiting on one.
#[test]
fn shared_reservations_end_when_a_panic_leaves_the_local_entry_around_them() {
    let shared = Arc::new(Rt::new());
    shared.set_placeholder(0_u64);
    shared.set_max_frames(1);
    let boom = Arc::new(AtomicBool::new(true));
    let s = shared.create_input(1_u64);
    let leaf = {
        let boom = boom.clone();
        shared.create_query_with(
            move |rt: &Rt| {
                let v = rt.get(s);
                if boom.load(SeqCst) {
                    panic!("leaf boom");
                }
                v
            },
            replayable(),
        )
    };
    let mut top = leaf;
    for _ in 0..4 {
        let below = top;
        top = shared.create_query_with(move |rt: &Rt| rt.get(below) + 1, replayable());
    }
    let local = Runtime::<crate::cells::Local>::new();
    let read = {
        let shared = shared.clone();
        local.create_query(move |_| shared.get(top))
    };
    assert!(outcome(|| local.get(read)).is_err());
    assert!(!shared.dooms.lock().unwrap().is_empty(), "no doom ran");
    assert!(
        crate::reserve::none_held(),
        "a reservation outlived every entry on the thread"
    );
    boom.store(false, SeqCst);
    shared.set(s, 2);
    let other = {
        let shared = shared.clone();
        spawn("other", move || shared.get(top))
    };
    assert_eq!(other.join(), Ok(6));
}

/// The Shared read under a Local entry is its runtime kind's outermost
/// entry and ends what the thread still reserves when it exits, as a
/// Shared outermost entry does: a reservation its compute left PENDING
/// (here noted directly, standing in for a doom whose mark unwound) is
/// gone before the Local closure goes on.
#[test]
fn shared_entry_under_a_local_entry_ends_what_it_left_reserved() {
    let shared = Arc::new(Rt::new());
    let leftover = (crate::handle::RuntimeId::for_test(u64::MAX - 11), 3);
    let leaves = shared.create_query(move |_: &Rt| {
        let _scope = crate::reserve::Scope::new(Some(crate::reserve::PENDING));
        crate::reserve::note(leftover, 0);
        1_u64
    });
    let local = Runtime::<crate::cells::Local>::new();
    let read = {
        let shared = shared.clone();
        local.create_query(move |_| {
            let v = shared.get(leaves);
            (v, crate::reserve::none_held())
        })
    };
    assert_eq!(local.get(read), (1, true));
}

/// A Local closure that reads two Shared chains sharing their lower half,
/// each doomed at budget 1: the second Shared entry starts where the
/// first did with no context of it left, and both reads, and what the
/// thread holds afterwards, are the unbounded run's.
#[test]
fn second_shared_entry_under_a_local_entry_starts_with_no_context() {
    let run = |budget: u32| {
        let shared = Arc::new(Rt::new());
        shared.set_placeholder(0_u64);
        shared.set_max_frames(budget);
        let x = shared.create_input(1_u64);
        let mut low = x;
        for _ in 0..4 {
            let below = low;
            low = shared.create_query_with(move |rt: &Rt| rt.get(below) + 1, replayable());
        }
        let (mut a, mut b) = (low, low);
        for _ in 0..4 {
            let (below_a, below_b) = (a, b);
            a = shared.create_query_with(move |rt: &Rt| rt.get(below_a) + 1, replayable());
            b = shared.create_query_with(move |rt: &Rt| rt.get(below_b) * 2, replayable());
        }
        let local = Runtime::<crate::cells::Local>::new();
        let read = {
            let shared = shared.clone();
            local.create_query(move |_| {
                let first = shared.get(a);
                let left = crate::replay::with(|c| c.len());
                (first, left, shared.get(b))
            })
        };
        let got = local.get(read);
        let no_doom = shared.dooms.lock().unwrap().is_empty();
        (got, no_doom)
    };
    let (unbounded, _) = run(u32::MAX);
    let (bounded, no_doom) = run(1);
    assert!(!no_doom, "no doom ran");
    assert_eq!(bounded, unbounded);
    assert!(crate::reserve::none_held());
}

/// `get(t)` on a helper thread, or `None` if it has not returned in 10 s.
fn within_10s(rt: &Arc<Rt>, t: Incr<u64>) -> Option<Outcome> {
    let (tx, rx) = mpsc::channel();
    let rt = rt.clone();
    thread::spawn(move || {
        let _ = tx.send(outcome(|| rt.get(t)));
    });
    rx.recv_timeout(Duration::from_secs(10)).ok()
}

/// T reads X, which reads Y, then joins a scoped thread that reads X;
/// U reads X, then joins, rayon-style, a read of the input on its own
/// thread with a scoped thread's read of Z, which reads X. Over budget,
/// each doomed run goes on past its placeholder for X into the join,
/// while X waits for the retry under the doom's reservation: the child
/// must compute what it reads, as it would on a runtime without
/// reservations, rather than wait on the thread that joins it. The
/// unbounded runs give main's values.
#[test]
fn a_doomed_runs_joined_child_reads_a_node_the_doom_released() {
    for frames in [u32::MAX, 1, 2] {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        let i = rt.create_input(3_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 10, replayable());
        let z = rt.create_query_with(move |rt: &Rt| rt.get(x) + 1000, replayable());
        let t = rt.create_query_with(
            move |rt: &Rt| {
                let v = rt.get(x);
                let child = thread::scope(|s| s.spawn(|| rt.get(x)).join().unwrap());
                v + child
            },
            replayable(),
        );
        let u = rt.create_query_with(
            move |rt: &Rt| {
                let v = rt.get(x);
                let (a, b) = thread::scope(|s| {
                    let b = s.spawn(|| rt.get(z));
                    (rt.get(i), b.join().unwrap())
                });
                v + a + b
            },
            replayable(),
        );
        rt.set_max_frames(frames);
        assert_eq!(within_10s(&rt, t), Some(Ok(28)), "budget {frames}");
        rt.set(i, 4);
        assert_eq!(
            within_10s(&rt, u),
            Some(Ok(15 + 4 + 1015)),
            "budget {frames}"
        );
        rt.assert_invariant_i();
    }
}

/// R reads X, whose read dooms past the budget, then S, which dooms too.
/// X joins a thread that reads S, but only on a run whose read of X1 is no
/// placeholder, so only its retry as an entry does. A doomed R that ran on
/// past X's placeholder would have released S under a reservation before
/// that entry ran; the child must not wait on it, as on main, where nothing
/// holds S when X runs.
#[test]
fn an_entrys_joined_child_reads_a_sibling_its_doom_never_reached() {
    for frames in [u32::MAX, 1, 2] {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        let i = rt.create_input(3_u64);
        let s1 = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        let s = rt.create_query_with(move |rt: &Rt| rt.get(s1) + 1, replayable());
        let x1 = rt.create_query_with(move |rt: &Rt| rt.get(i) + 10, replayable());
        let x = rt.create_query_with(
            move |rt: &Rt| {
                let v = rt.get(x1);
                if v == 0 {
                    return 0;
                }
                v + thread::scope(|sc| sc.spawn(|| rt.get(s)).join().unwrap())
            },
            replayable(),
        );
        let r = rt.create_query_with(move |rt: &Rt| rt.get(x) + rt.get(s), replayable());
        rt.set_max_frames(frames);
        assert_eq!(within_10s(&rt, r), Some(Ok(13 + 5 + 5)), "budget {frames}");
        rt.set(i, 4);
        assert_eq!(within_10s(&rt, r), Some(Ok(14 + 6 + 6)), "budget {frames}");
        rt.assert_invariant_i();
    }
}

/// As above with S of a type that has no placeholder, so a doomed R that
/// ran on past X's placeholder would run S natively, and S, reading S1
/// past the budget, would doom and release itself under a reservation
/// before X's entry joins the child that reads it.
#[test]
fn an_entrys_joined_child_reads_a_sibling_a_doomed_run_would_read_natively() {
    for frames in [u32::MAX, 1, 2] {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        let i = rt.create_input(3_u64);
        let s1 = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        let s = rt.create_query_with(move |rt: &Rt| rt.get(s1) as u32 + 1, replayable());
        let x1 = rt.create_query_with(move |rt: &Rt| rt.get(i) + 10, replayable());
        let x = rt.create_query_with(
            move |rt: &Rt| {
                let v = rt.get(x1);
                if v == 0 {
                    return 0;
                }
                v + u64::from(thread::scope(|sc| sc.spawn(|| rt.get(s)).join().unwrap()))
            },
            replayable(),
        );
        let r = rt.create_query_with(
            move |rt: &Rt| rt.get(x) + u64::from(rt.get(s)),
            replayable(),
        );
        rt.set_max_frames(frames);
        assert_eq!(within_10s(&rt, r), Some(Ok(13 + 5 + 5)), "budget {frames}");
        rt.set(i, 4);
        assert_eq!(within_10s(&rt, r), Some(Ok(14 + 6 + 6)), "budget {frames}");
        rt.assert_invariant_i();
    }
}

/// T catches its read of X, as code handling a failed dep does, then joins
/// a thread that reads X. Over budget the read dooms in X's subtree, and
/// T's catch swallows the doom's unwind, so the join runs while X is
/// released under T's reservation; on main X is Clean by then.
#[test]
fn a_closure_that_catches_a_doom_joins_a_reader_of_the_doomed_node() {
    for frames in [u32::MAX, 1, 2] {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        let i = rt.create_input(3_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 10, replayable());
        let t = rt.create_query_with(
            move |rt: &Rt| {
                let v = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_or(0);
                v + thread::scope(|s| s.spawn(|| rt.get(x)).join().unwrap())
            },
            replayable(),
        );
        rt.set_max_frames(frames);
        assert_eq!(within_10s(&rt, t), Some(Ok(28)), "budget {frames}");
        rt.assert_invariant_i();
    }
}

/// As above with X in a cycle with W, each reading the other with
/// `try_get`, and T's child reading W with `try_get`, as a member of a
/// cycle with no strategy has no value for `get`: over budget the doom falls inside the
/// episode, whose claims the doomed frames may keep held while T's catch
/// runs the join.
#[test]
fn a_closure_that_catches_a_doom_in_a_cycle_joins_a_reader_of_a_member() {
    let run = |frames: u32| {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        let i = rt.create_input(3_u64);
        let xc: Arc<OnceLock<Incr<u64>>> = Arc::default();
        let w = {
            let xc = xc.clone();
            rt.create_query_with(
                move |rt: &Rt| rt.try_get(*xc.get().unwrap()).map_or(5, |v| v + 1),
                replayable(),
            )
        };
        let x = rt.create_query_with(
            move |rt: &Rt| rt.try_get(w).map_or(rt.get(i) + 100, |v| v + 1000),
            replayable(),
        );
        xc.set(x).unwrap();
        let t = rt.create_query_with(
            move |rt: &Rt| {
                let v = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_or(0);
                v + thread::scope(|s| s.spawn(|| rt.try_get(w).unwrap_or(9)).join().unwrap())
            },
            replayable(),
        );
        rt.set_max_frames(frames);
        let got = within_10s(&rt, t);
        if got.is_some() {
            rt.assert_invariant_i();
        }
        got
    };
    let unbounded = run(u32::MAX);
    assert!(matches!(unbounded, Some(Ok(_))), "{unbounded:?}");
    for frames in [1, 2] {
        assert_eq!(run(frames), unbounded, "budget {frames}");
    }
}

/// X catches a cycle and so lands Dirty whenever it is read, and T
/// catches X's doom and joins a child reading X. The restart a take-over
/// gives T reads X Dirty again, over budget: if X dooms again, T and its
/// next child trade take-overs for good. Each take-over must add a claim
/// the read never reserves again, so T's read sees at most one per node
/// it reaches (T, X, C).
#[test]
fn a_take_over_of_a_node_that_lands_dirty_restarts_its_read_boundedly() {
    for frames in [u32::MAX, 1, 2] {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        let cc: Arc<OnceLock<Incr<u64>>> = Arc::default();
        let c = {
            let cc = cc.clone();
            rt.create_query_with(move |rt: &Rt| rt.get(*cc.get().unwrap()), replayable())
        };
        cc.set(c).unwrap();
        let x = rt.create_query_with(
            move |rt: &Rt| rt.try_get(c).map_or(7, |v| v + 1),
            replayable(),
        );
        let takes = Arc::new(AtomicU32::new(0));
        let t = {
            let takes = takes.clone();
            rt.create_query_with(
                move |rt: &Rt| {
                    takes.fetch_max(crate::reserve::takes(), SeqCst);
                    let v = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_or(0);
                    v + thread::scope(|s| s.spawn(|| rt.get(x)).join().unwrap())
                },
                replayable(),
            )
        };
        rt.set_max_frames(frames);
        assert_eq!(within_10s(&rt, t), Some(Ok(14)), "budget {frames}");
        let n = takes.load(SeqCst);
        assert!(n <= 3, "budget {frames}: {n} take-overs");
        rt.assert_invariant_i();
    }
}

struct ReadOnDrop<'a>(&'a Rt, Incr<u64>);

impl Drop for ReadOnDrop<'_> {
    fn drop(&mut self) {
        let _ = self.0.get(self.1);
    }
}

/// T holds a guard that reads X when dropped, and T's read of Y is over
/// budget, so the guard reads X inside T's doom unwind. A read there can't
/// raise again without aborting the process, so it takes the placeholder.
#[test]
#[cfg_attr(miri, ignore)]
fn a_doomed_closures_destructor_reads_inside_its_unwind() {
    run_in_child(
        "runtime::reserve_tests::a_doomed_closures_destructor_reads_inside_its_unwind",
        || {
            for frames in [u32::MAX, 1, 2] {
                let rt = Arc::new(Rt::new());
                rt.set_placeholder(0_u64);
                let i = rt.create_input(3_u64);
                let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
                let x = rt.create_query_with(move |rt: &Rt| rt.get(i) + 2, replayable());
                let t = rt.create_query_with(
                    move |rt: &Rt| {
                        let _guard = ReadOnDrop(rt, x);
                        rt.get(y) + 10
                    },
                    replayable(),
                );
                rt.set_max_frames(frames);
                assert_eq!(within_10s(&rt, t), Some(Ok(14)), "budget {frames}");
            }
        },
    )
    .assert_clean();
}

/// A closure that catches an ordinary panic of its dep finds its attempt
/// out of flight at any budget: only a discard's unwind is in flight.
#[test]
fn a_caught_user_panic_never_puts_the_attempt_in_flight() {
    for frames in [u32::MAX, 1, 2] {
        let rt = Rt::new();
        rt.set_placeholder(0_u64);
        let i = rt.create_input(3_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        let f = rt.create_query_with(
            move |rt: &Rt| {
                let v = rt.get(y);
                assert!(v == 0, "f boom {v}");
                v
            },
            replayable(),
        );
        let seen: Arc<std::sync::Mutex<Vec<bool>>> = Arc::default();
        let t = {
            let seen = seen.clone();
            rt.create_query_with(
                move |rt: &Rt| match catch_unwind(AssertUnwindSafe(|| rt.get(f))) {
                    Ok(v) => v,
                    Err(p)
                        if p.is::<crate::cycle::DoomUnwind>()
                            || p.is::<crate::cycle::DiscardUnwind>() =>
                    {
                        std::panic::resume_unwind(p)
                    }
                    Err(_) => {
                        seen.lock().unwrap().push(crate::attempt::in_flight());
                        7
                    }
                },
                replayable(),
            )
        };
        rt.set_max_frames(frames);
        assert_eq!(outcome(|| rt.get(t)), Ok(7), "budget {frames}");
        let seen = seen.lock().unwrap();
        assert!(
            !seen.is_empty() && seen.iter().all(|&f| !f),
            "budget {frames}: {seen:?}"
        );
    }
}

static JOIN_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static JOIN_TARGET: OnceLock<(Arc<Rt>, Incr<JoinV>)> = OnceLock::new();
static JOIN_CHILD: std::sync::Mutex<Option<Option<u64>>> = std::sync::Mutex::new(None);

/// A value whose clone, once armed, joins a scoped thread that reads the
/// node [`JOIN_TARGET`] names.
#[derive(PartialEq, Debug)]
struct JoinV(u64);

impl Clone for JoinV {
    fn clone(&self) -> Self {
        if JOIN_ARMED.swap(false, SeqCst) {
            let (rt, r) = JOIN_TARGET.get().unwrap();
            let got = thread::scope(|s| {
                s.spawn(|| catch_unwind(AssertUnwindSafe(|| rt.get(*r).0)).ok())
                    .join()
                    .unwrap()
            });
            *JOIN_CHILD.lock().unwrap() = Some(got);
        }
        JoinV(self.0)
    }
}

/// R, over a dependency cycle it catches, lands Dirty with its run count,
/// and the read's clone of R's value joins a thread that reads R. The
/// child must rerun R, as a second read does, rather than wait on the
/// read's reservation of its own node, and the read keeps its own value.
#[test]
fn a_reads_clone_joins_a_child_reading_the_node_it_landed_dirty() {
    let rt = Arc::new(Rt::new());
    let cc: Arc<OnceLock<Incr<JoinV>>> = Arc::default();
    let c = {
        let cc = cc.clone();
        rt.create_query(move |rt: &Rt| rt.get(*cc.get().unwrap()))
    };
    cc.set(c).unwrap();
    let runs = Arc::new(AtomicU32::new(0));
    let r = {
        let runs = runs.clone();
        rt.create_query(move |rt: &Rt| {
            let n = u64::from(runs.fetch_add(1, SeqCst)) + 1;
            JoinV(rt.try_get(c).map_or(n * 100, |v| v.0 + n))
        })
    };
    assert!(JOIN_TARGET.set((rt.clone(), r)).is_ok());
    JOIN_ARMED.store(true, SeqCst);
    let (tx, rx) = mpsc::channel();
    let reader = rt.clone();
    thread::spawn(move || {
        let _ = tx.send(outcome(|| reader.get(r).0));
    });
    assert_eq!(rx.recv_timeout(Duration::from_secs(10)).ok(), Some(Ok(100)));
    assert_eq!(*JOIN_CHILD.lock().unwrap(), Some(Some(200)));
}
