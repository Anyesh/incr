//! Doom reservations end with their owner: at a wound, when a cycle
//! unwind drops the owner's context before its retry, when the owner's
//! retry panics, and at the owner's final landing of the node. Each test
//! parks the dooming thread, has another thread read a node it reserved
//! or reserved and landed, and checks that both threads finish with a
//! serial order's outcomes.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::SeqCst};
use std::sync::{mpsc, Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use super::race_tests::{await_reservation_wait, id, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::child_test::{run_in_child, run_in_child_within, Outcome as ChildOutcome};
use crate::cycle::{QueryOpts, Replay};
use crate::handle::Incr;
use crate::state::NodeState;

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

type BegunOp = (mpsc::Sender<()>, thread::JoinHandle<bool>);

/// Another thread's op, begun before this returns, that checks the
/// reservation of `claim` once [`checked`] lets it.
fn begun_op(claim: crate::waits::Claim) -> BegunOp {
    let (ready_tx, ready) = mpsc::channel();
    let (go, go_rx) = mpsc::channel::<()>();
    let op = thread::spawn(move || {
        crate::reserve::op_begin();
        ready_tx.send(()).unwrap();
        go_rx.recv().unwrap();
        crate::reserve::elsewhere(claim, 1)
    });
    ready.recv().unwrap();
    (go, op)
}

fn checked((go, op): BegunOp) -> bool {
    go.send(()).unwrap();
    op.join().unwrap()
}

/// A reservation at rest holds the ops begun before its rest, and an
/// op begun after passes; a second rest keeps the first one's stamp.
#[test]
fn a_rested_reservation_holds_only_ops_begun_before_its_first_rest() {
    use crate::reserve::{end_contexts, note, rest, Scope};
    let rt = Rt::new();
    let claim = (rt.runtime_id, 7);
    {
        let _scope = Scope::new(Some(5));
        note(claim, 1);
    }
    let before = begun_op(claim);
    rest(claim, 1);
    let between = begun_op(claim);
    rest(claim, 1);
    let after = begun_op(claim);
    assert!(checked(before), "an op begun before the rest passed");
    assert!(!checked(between), "the second rest restamped it");
    assert!(!checked(after), "an op begun after the rest waited");
    end_contexts();
}

/// Reserving a rested node again wakes its reservation for every op.
#[test]
fn reserving_a_rested_node_again_holds_ops_begun_after_the_rest() {
    use crate::reserve::{end_contexts, note, rest, Scope};
    let rt = Rt::new();
    let claim = (rt.runtime_id, 7);
    let _scope = Scope::new(Some(5));
    note(claim, 1);
    rest(claim, 1);
    note(claim, 1);
    assert!(
        checked(begun_op(claim)),
        "the second reservation stayed at rest"
    );
    end_contexts();
}

/// A context's frame leaving the node does not rest a read's reservation
/// of its own node: the read takes the value after any op begins.
#[test]
fn a_rest_leaves_a_read_reserving_its_own_node_holding_every_op() {
    use crate::reserve::{end_contexts, note, note_root, rest, Root, Scope};
    let rt = Rt::new();
    let claim = (rt.runtime_id, 7);
    let root = Root::arm(true, claim);
    note_root(claim, 1);
    {
        let _scope = Scope::new(Some(5));
        note(claim, 1);
    }
    rest(claim, 1);
    assert!(
        checked(begun_op(claim)),
        "the rest let a later op past the read"
    );
    drop(root);
    end_contexts();
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
    let run = |frames: u32| {
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
        let got = within_10s(&rt, t);
        if got.is_some() {
            rt.assert_invariant_i();
        }
        (got, takes.load(SeqCst))
    };
    let (unbounded, _) = run(u32::MAX);
    assert!(unbounded.is_some(), "the unbounded read hung");
    for frames in [1, 2] {
        let (got, n) = run(frames);
        assert_eq!(got, unbounded, "budget {frames}");
        assert!(n <= 3, "budget {frames}: {n} take-overs");
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

struct OnDrop<F: FnMut()>(F);

impl<F: FnMut()> Drop for OnDrop<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}

fn message(got: std::thread::Result<u64>) -> String {
    match got {
        Ok(v) => format!("Ok({v})"),
        Err(e) if e.is::<crate::cycle::WoundUnwind>() => "a wound".to_string(),
        Err(e) => e
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| e.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "an engine unwind".to_string()),
    }
}

/// Silences P's panic, which a guard's read raises inside an unwind, and
/// leaves every other panic to the default hook. The default hook prints a
/// full backtrace for a panic inside an unwind, and under TSan symbolizing
/// it takes 14 to 20 s, past the child's timeout.
fn quiet_p_booms() {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if info.payload().downcast_ref::<&str>() != Some(&"p boom") {
                default(info);
            }
        }));
    });
}

/// T holds a guard whose destructor reads P, which panics, and `try_get`s
/// C, which closes a cycle of its own, catching both; T's read of A unwinds
/// a cycle through T. The destructor's reads must leave the cycle's unwind
/// as they found it, and keep their own: `try_get(T)` reports what it does
/// without the guard, the guard catches P's panic and C's cycle as `Err`,
/// and C stays a cycle, not a failure.
fn a_destructors_read_keeps_a_cycles_unwind<C: Cells>() {
    quiet_p_booms();
    let mut reports = Vec::new();
    for guarded in [false, true] {
        let rt = Runtime::<C>::new();
        let p = rt.create_query(|_: &Runtime<C>| -> u64 { panic!("p boom") });
        let bb: Arc<OnceLock<Incr<u64>>> = Arc::default();
        let a = {
            let bb = bb.clone();
            rt.create_query(move |rt: &Runtime<C>| rt.get(*bb.get().unwrap()))
        };
        bb.set(rt.create_query(move |rt: &Runtime<C>| rt.get(a)))
            .unwrap();
        let dd: Arc<OnceLock<Incr<u64>>> = Arc::default();
        let c = {
            let dd = dd.clone();
            rt.create_query(move |rt: &Runtime<C>| rt.get(*dd.get().unwrap()))
        };
        dd.set(rt.create_query(move |rt: &Runtime<C>| rt.get(c)))
            .unwrap();
        let seen: Arc<std::sync::Mutex<Vec<String>>> = Arc::default();
        let t = {
            let seen = seen.clone();
            rt.create_query(move |rt: &Runtime<C>| {
                let _guard = guarded.then(|| {
                    OnDrop(|| {
                        let got = catch_unwind(AssertUnwindSafe(|| rt.get(p)));
                        seen.lock().unwrap().push(message(got));
                        let got = catch_unwind(AssertUnwindSafe(|| rt.try_get(c).is_err()));
                        seen.lock()
                            .unwrap()
                            .push(format!("{:?}", got.map_err(|e| message(Err(e)))));
                    })
                });
                rt.get(a)
            })
        };
        let got = catch_unwind(AssertUnwindSafe(|| rt.try_get(t).map_err(|_| ())));
        reports.push(format!("{:?}", got.map_err(|e| message(Err(e)))));
        if guarded {
            assert_eq!(*seen.lock().unwrap(), ["p boom", "Ok(true)"]);
            let again = catch_unwind(AssertUnwindSafe(|| rt.try_get(c).is_err()));
            assert!(
                matches!(again, Ok(true)),
                "the cycle C closes landed as a failure"
            );
        }
    }
    assert_eq!(reports[0], reports[1]);
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_destructors_read_keeps_a_cycles_unwind_local() {
    run_in_child(
        "runtime::reserve_tests::a_destructors_read_keeps_a_cycles_unwind_local",
        a_destructors_read_keeps_a_cycles_unwind::<Local>,
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_destructors_read_keeps_a_cycles_unwind_shared() {
    run_in_child(
        "runtime::reserve_tests::a_destructors_read_keeps_a_cycles_unwind_shared",
        a_destructors_read_keeps_a_cycles_unwind::<Shared>,
    )
    .assert_clean();
}

/// T holds a guard whose destructor reads P on a second runtime, which
/// panics, and catches it; over budget, T's read of Y raises T's doom, so
/// the guard reads inside the doom's unwind. With `wound`, a waiter that met
/// T's claim wounds T's attempt once, while that discard is in flight, just
/// before the read. The read must leave the unwind's kind and flight to T's
/// own catch: the guard never catches a wound, and T returns its value.
/// Inside the doomed run the read is speculative, so P's panic may reach
/// the guard as a discard.
fn destructor_read_on_another_runtime(wound: bool) {
    for frames in [u32::MAX, 1, 2] {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        let other = Arc::new(Rt::new());
        let p = other.create_query(|_: &Rt| -> u64 { panic!("p boom") });
        let i = rt.create_input(3_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        let tt: Arc<OnceLock<Incr<u64>>> = Arc::default();
        let seen: Arc<std::sync::Mutex<Vec<String>>> = Arc::default();
        let wounds = Arc::new(AtomicU32::new(0));
        let t = {
            let (other, seen, wounds, tt) =
                (other.clone(), seen.clone(), wounds.clone(), tt.clone());
            rt.create_query_with(
                move |rt: &Rt| {
                    let _guard = OnDrop(|| {
                        if wound && crate::attempt::in_flight() && wounds.load(SeqCst) == 0 {
                            let claim = (rt.runtime_id, tt.get().unwrap().slot());
                            let attempt = crate::attempt::mine();
                            thread::scope(|s| {
                                s.spawn(|| assert!(attempt.wound_in_flight(claim)));
                            });
                            wounds.fetch_add(1, SeqCst);
                        }
                        let got = catch_unwind(AssertUnwindSafe(|| other.get(p)));
                        seen.lock().unwrap().push(message(got));
                    });
                    rt.get(y) + 10
                },
                replayable(),
            )
        };
        tt.set(t).unwrap();
        rt.set_max_frames(frames);
        assert_eq!(within_10s(&rt, t), Some(Ok(14)), "budget {frames}");
        let seen = seen.lock().unwrap();
        assert!(!seen.is_empty(), "budget {frames}");
        assert!(
            seen.iter().all(|m| m != "a wound"),
            "budget {frames}: {seen:?}"
        );
        if wound && frames == 1 {
            assert!(wounds.load(SeqCst) > 0, "no wound at budget 1");
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_destructors_read_on_another_runtime_keeps_a_doom() {
    run_in_child(
        "runtime::reserve_tests::a_destructors_read_on_another_runtime_keeps_a_doom",
        || destructor_read_on_another_runtime(false),
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_destructors_read_on_another_runtime_keeps_a_wounded_doom() {
    run_in_child(
        "runtime::reserve_tests::a_destructors_read_on_another_runtime_keeps_a_wounded_doom",
        || destructor_read_on_another_runtime(true),
    )
    .assert_clean();
}

/// A Local query L reads T on a Shared runtime, where T catches X's doom
/// and joins a child that takes it over, so the wound reaches L's entry and
/// L reads T again in a second Shared entry. Neither Shared entry is the
/// thread's outermost, yet each is an attempt of its own: once L returns,
/// the thread keeps no wound and no taken claim from them.
#[test]
fn a_shared_entry_under_a_local_one_leaves_no_take_over_behind() {
    let shared = Arc::new(Rt::new());
    shared.set_placeholder(0_u64);
    let i = shared.create_input(3_u64);
    let y = shared.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
    let x = shared.create_query_with(move |rt: &Rt| rt.get(y) + 10, replayable());
    let t = shared.create_query_with(
        move |rt: &Rt| {
            let v = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_or(0);
            v + thread::scope(|s| s.spawn(|| rt.get(x)).join().unwrap())
        },
        replayable(),
    );
    shared.set_max_frames(1);
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let local = Runtime::<Local>::new();
        let l = local.create_query(move |_: &Runtime<Local>| shared.get(t));
        let got = local.get(l);
        let _ = tx.send((
            got,
            crate::attempt::mine().wounded(),
            crate::reserve::takes(),
        ));
    });
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(10)).ok(),
        Some((28, false, 0))
    );
}

/// As above with S, between catching X's doom and joining its child,
/// reading a node of a second Shared runtime. That entry is nested in S's,
/// which is the thread's outermost Shared entry, so it starts no attempt:
/// one that did would move the epoch on under the child's take-over of
/// S's attempt, and S would wait on its child for good.
#[test]
fn a_shared_entry_inside_a_shared_one_under_a_local_one_starts_no_attempt() {
    let shared = Arc::new(Rt::new());
    let other = Arc::new(Rt::new());
    shared.set_placeholder(0_u64);
    let j = other.create_input(5_u64);
    let o = other.create_query_with(move |rt: &Rt| rt.get(j) + 1, replayable());
    let i = shared.create_input(3_u64);
    let y = shared.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
    let x = shared.create_query_with(move |rt: &Rt| rt.get(y) + 10, replayable());
    let s = {
        let other = other.clone();
        shared.create_query_with(
            move |rt: &Rt| {
                let v = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_or(0);
                let w = other.get(o);
                v + w + thread::scope(|sc| sc.spawn(|| rt.get(x)).join().unwrap())
            },
            replayable(),
        )
    };
    shared.set_max_frames(1);
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let local = Runtime::<Local>::new();
        let l = local.create_query(move |_: &Runtime<Local>| shared.get(s));
        let got = local.get(l);
        let _ = tx.send((
            got,
            crate::attempt::mine().wounded(),
            crate::reserve::takes(),
        ));
    });
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(10)).ok(),
        Some((34, false, 0))
    );
}

/// The Dirty-landing take-over under a Local entry: each wound retries
/// the Local entry, whose next Shared entry is a new attempt. The claims
/// earlier take-overs met must stay taken across those retries, or every
/// retry dooms X again and is taken over again.
#[test]
fn a_take_over_under_a_local_entry_restarts_its_read_boundedly() {
    let run = |frames: u32| {
        let shared = Arc::new(Rt::new());
        shared.set_placeholder(0_u64);
        let cc: Arc<OnceLock<Incr<u64>>> = Arc::default();
        let c = {
            let cc = cc.clone();
            shared.create_query_with(move |rt: &Rt| rt.get(*cc.get().unwrap()), replayable())
        };
        cc.set(c).unwrap();
        let x = shared.create_query_with(
            move |rt: &Rt| rt.try_get(c).map_or(7, |v| v + 1),
            replayable(),
        );
        let takes = Arc::new(AtomicU32::new(0));
        let t = {
            let takes = takes.clone();
            shared.create_query_with(
                move |rt: &Rt| {
                    let v = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_or(0);
                    let w = thread::scope(|s| s.spawn(|| rt.get(x)).join().unwrap());
                    takes.fetch_max(crate::reserve::takes(), SeqCst);
                    v + w
                },
                replayable(),
            )
        };
        shared.set_max_frames(frames);
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let local = Runtime::<Local>::new();
            let l = local.create_query(move |_: &Runtime<Local>| shared.get(t));
            let _ = tx.send(outcome(|| local.get(l)));
        });
        (
            rx.recv_timeout(Duration::from_secs(10)).ok(),
            takes.load(SeqCst),
        )
    };
    let (unbounded, _) = run(u32::MAX);
    assert!(unbounded.is_some(), "the unbounded read hung");
    for frames in [1, 2] {
        let (got, takes) = run(frames);
        assert_eq!(got, unbounded, "budget {frames}");
        // Only a contained cycle leaves X a value for the child to wait on;
        // when C's cycle escapes, the read fails with no take-over.
        if frames == 1 && matches!(unbounded, Some(Ok(_))) {
            assert!(takes > 0, "no take-over at budget 1");
        }
    }
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
/// Whether a rerun contains C's cycle or lets it out is the containment
/// rule's call; the child and the second read must agree on it.
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
    let child = JOIN_CHILD.lock().unwrap().take();
    let child = child.expect("the clone joined no child");
    let again = thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.get(r).0)).ok())
        .join()
        .unwrap();
    assert_eq!(child.is_some(), again.is_some(), "{child:?}, {again:?}");
    if let Some(v) = child {
        assert_eq!((v, again), (200, Some(300)));
    }
}

/// **A reservation outlives the owner's final landing**, at budget 2: R
/// reads T, T reads X, X reads Y and Y reads input I, so R dooms on Y and
/// reserves T and X. On T's final run, after it has read X, R parks and a
/// set of I dirties X again; T's closure then has another thread read X
/// and joins it. The unbounded run lets that read rerun X, so T's join
/// must not wait on R's reservation of X.
#[test]
fn shared_reservation_ends_at_its_owners_final_landing() {
    type Shape = (Arc<Rt>, Incr<u64>, Incr<u64>, Gate, Arc<AtomicBool>);
    fn shape(rt: Rt) -> Shape {
        let rt = Arc::new(rt);
        rt.set_placeholder(0_u64);
        let i = rt.create_input(1_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 10, replayable());
        let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 1, replayable());
        let gate = Gate::new();
        let stalled = Arc::new(AtomicBool::new(false));
        let t = {
            let (gate, stalled) = (gate.clone(), stalled.clone());
            let shared = Arc::downgrade(&rt);
            rt.create_query_with(
                move |rt: &Rt| {
                    let v = rt.get(x);
                    if v == 0 {
                        return 0;
                    }
                    gate.park();
                    let waits = rt.reservation_waits.load(SeqCst);
                    let (tx, rx) = mpsc::channel();
                    let rt2 = shared.upgrade().unwrap();
                    thread::spawn(move || {
                        let _ = tx.send(outcome(|| rt2.get(x)));
                    });
                    // A wait on R's reservation, held until this read ends,
                    // hangs a plain join.
                    let deadline = Instant::now() + JOIN_TIMEOUT;
                    loop {
                        if let Ok(read) = rx.try_recv() {
                            return v + read.unwrap_or(1000);
                        }
                        if rt.reservation_waits.load(SeqCst) != waits {
                            stalled.store(true, SeqCst);
                            return v;
                        }
                        assert!(
                            Instant::now() < deadline,
                            "the joined read neither returned nor waited"
                        );
                        thread::yield_now();
                    }
                },
                replayable(),
            )
        };
        (rt, t, i, gate, stalled)
    }
    fn run(frames: u32) -> (Outcome, bool) {
        let (rt, t, i, gate, stalled) = shape(Rt::new());
        rt.set_max_frames(frames);
        gate.arm();
        let r = {
            let rt = rt.clone();
            spawn("r", move || rt.get(t))
        };
        gate.wait_parked();
        rt.set(i, 2);
        gate.release();
        let read = r.join();
        rt.assert_invariant_i();
        (read, stalled.load(SeqCst))
    }
    let want = run(u32::MAX);
    assert!(!want.1, "the unbounded run's join stalled");
    assert_eq!(run(2), want);
}

/// Reads `x` on another thread while its doomed owner is parked, after
/// the owner's landing of it: the read must not wait on a reservation.
fn read_unreserved(rt: &Arc<Rt>, x: Incr<u64>) -> Outcome {
    let (claim, generation) = ((rt.runtime_id, id(x).0), x.generation());
    let reserved = thread::spawn(move || crate::reserve::elsewhere(claim, generation));
    assert!(
        !reserved.join().unwrap(),
        "the owner's landing left X reserved"
    );
    let waits = rt.reservation_waits.load(SeqCst);
    let read = {
        let rt = rt.clone();
        spawn("other", move || rt.get(x))
    }
    .join();
    assert_eq!(
        rt.reservation_waits.load(SeqCst),
        waits,
        "the read of X waited"
    );
    read
}

/// **A foreign rerun after the owner's Clean landing**, at budget 1: R
/// reads F, F reads X and X reads input I, so R dooms on X; X's entry
/// lands it Clean, which ends R's reservation of X, and R parks before
/// F's entry. A set of I dirties X and another thread reads X. F's retry
/// reads X as it then is, so once both reads are done F holds X's new
/// value.
#[test]
fn a_foreign_rerun_after_the_owners_landing_leaves_no_stale_delivery() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let i = rt.create_input(1_u64);
    let x = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
    let f = rt.create_query_with(move |rt: &Rt| rt.get(x) * 10, replayable());
    let entry = Gate::new();
    hook(&rt, "r", ParkPoint::BeforeEntry(id(f)), entry.clone());
    entry.arm();
    let r = {
        let rt = rt.clone();
        spawn("r", move || rt.get(f))
    };
    entry.wait_parked();
    rt.set(i, 2);
    let x_read = read_unreserved(&rt, x);
    entry.release();
    let f_read = r.join();
    assert_eq!(x_read, Ok(3));
    assert!(matches!(f_read, Ok(20) | Ok(30)), "{f_read:?}");
    assert_eq!(rt.get(f), 30, "F kept the value X had before the set");
    rt.assert_invariant_i();
}

/// **A Failed landing's delivery stays the read it stands for**, at
/// budget 1: R reads F, F reads X and X reads input I, panicking while I
/// is 1. R dooms on X, X's entry lands it Failed and its panic is
/// delivered to F's retry; R parks before F's entry. A set of I to 2 lets
/// X compute, and another thread reads X. F's retry takes the panic, as
/// the unbounded run's F read it when X landed, but once both reads are
/// done F must not stay Failed over X's new value.
#[test]
fn a_foreign_rerun_after_a_failed_landing_leaves_no_stale_delivery() {
    let rt = Arc::new(Rt::new());
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let i = rt.create_input(1_u64);
    let x = rt.create_query_with(
        move |rt: &Rt| {
            let v = rt.get(i);
            assert!(v != 1, "x boom");
            v + 1
        },
        replayable(),
    );
    let f = rt.create_query_with(move |rt: &Rt| rt.get(x) * 10, replayable());
    let entry = Gate::new();
    hook(&rt, "r", ParkPoint::BeforeEntry(id(f)), entry.clone());
    entry.arm();
    let r = {
        let rt = rt.clone();
        spawn("r", move || rt.get(f))
    };
    entry.wait_parked();
    rt.set(i, 2);
    let x_read = read_unreserved(&rt, x);
    entry.release();
    let f_read = r.join();
    assert_eq!(f_read, Err("x boom".to_string()));
    assert_eq!(x_read, Ok(3));
    assert_eq!(
        outcome(|| rt.get(f)),
        Ok(30),
        "F stayed Failed over X's new value"
    );
    rt.assert_invariant_i();
}

/// **A Dirty landing's reservation does not outlive the read**, at budget
/// 2: as in [`shared_reservation_ends_at_its_owners_final_landing`], but
/// X's final run parks after reading Y, a set of I flags it, and it lands
/// Dirty. T's closure then has another thread read X and joins it; the
/// unbounded run lets that read rerun X. After R's read no reservation of
/// R's is left.
#[test]
fn shared_reservation_of_a_dirty_landing_ends_with_the_read() {
    type Shape = (Arc<Rt>, Incr<u64>, Incr<u64>, Gate, Arc<AtomicBool>);
    fn shape(rt: Rt) -> Shape {
        let rt = Arc::new(rt);
        rt.set_placeholder(0_u64);
        let i = rt.create_input(1_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 10, replayable());
        let gate = Gate::new();
        let x = {
            let gate = gate.clone();
            rt.create_query_with(
                move |rt: &Rt| {
                    let v = rt.get(y);
                    // Only a run that read Y's real value: the doomed one reads
                    // the placeholder.
                    if v != 0 {
                        gate.park();
                    }
                    v + 1
                },
                replayable(),
            )
        };
        let stalled = Arc::new(AtomicBool::new(false));
        let t = {
            let stalled = stalled.clone();
            let shared = Arc::downgrade(&rt);
            rt.create_query_with(
                move |rt: &Rt| {
                    let v = rt.get(x);
                    if v == 0 {
                        return 0;
                    }
                    let waits = rt.reservation_waits.load(SeqCst);
                    let (tx, rx) = mpsc::channel();
                    let rt2 = shared.upgrade().unwrap();
                    thread::spawn(move || {
                        let _ = tx.send(outcome(|| rt2.get(x)));
                    });
                    let deadline = Instant::now() + JOIN_TIMEOUT;
                    loop {
                        if let Ok(read) = rx.try_recv() {
                            return v + read.unwrap_or(1000);
                        }
                        if rt.reservation_waits.load(SeqCst) != waits {
                            stalled.store(true, SeqCst);
                            return v;
                        }
                        assert!(
                            Instant::now() < deadline,
                            "the joined read neither returned nor waited"
                        );
                        thread::yield_now();
                    }
                },
                replayable(),
            )
        };
        (rt, t, i, gate, stalled)
    }
    fn run(frames: u32) -> (Outcome, bool) {
        let (rt, t, i, gate, stalled) = shape(Rt::new());
        rt.set_max_frames(frames);
        gate.arm();
        let r = {
            let rt = rt.clone();
            spawn("r", move || {
                let v = rt.get(t);
                assert!(crate::reserve::none_held(), "R's read left a reservation");
                v
            })
        };
        gate.wait_parked();
        rt.set(i, 2);
        gate.release();
        let read = r.join();
        rt.assert_invariant_i();
        (read, stalled.load(SeqCst))
    }
    let want = run(u32::MAX);
    assert!(!want.1, "the unbounded run's join stalled");
    assert_eq!(run(2), want);
}

/// **A Dirty landing's delivery survives a claim given back unrun**, at
/// budget 1: A reads T, T reads X, X reads Y and Y reads input I, so A
/// dooms on X and reserves T and X. X's entry parks after reading Y,
/// while B waits to read X; a set of I flags X and it lands Dirty. B
/// claims X before A loads X's state for the landing's delivery, finds
/// A's reservation and gives the claim back unrun. T's retry must take
/// the delivery of X's run, as the unbounded T reads that run's value,
/// rather than run X again. X reads no held landing, so it lands unheld
/// even while this thread holds landings.
#[test]
fn a_dirty_landings_delivery_survives_a_claim_given_back_unrun() {
    use std::sync::atomic::AtomicU8;
    fn run(frames: u32) -> (Outcome, Outcome, u8) {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(frames);
        let i = rt.create_input(1_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 10, replayable());
        let gate = Gate::new();
        let x = {
            let gate = gate.clone();
            rt.create_query_with(
                move |rt: &Rt| {
                    let v = rt.get(y);
                    if v != 0 {
                        gate.park();
                    }
                    v + 1
                },
                replayable(),
            )
        };
        let t = rt.create_query_with(move |rt: &Rt| rt.get(x) * 100, replayable());
        let stage = Arc::new(AtomicU8::new(0));
        let pinned = frames != u32::MAX;
        if pinned {
            let stage = stage.clone();
            let shared = Arc::downgrade(&rt);
            let wait = move |stage: &AtomicU8, want: u8, ms: u64| {
                let deadline = Instant::now() + Duration::from_millis(ms);
                while stage.load(SeqCst) < want && Instant::now() < deadline {
                    thread::yield_now();
                }
            };
            let installed = rt.park_hook.set(Box::new(move |p| {
                let who = thread::current().name().map(str::to_owned);
                match (who.as_deref(), p) {
                    (Some("b"), ParkPoint::BeforeClaim(n)) if n.0 == x.slot() => {
                        wait(&stage, 1, 4000);
                    }
                    (Some("a"), ParkPoint::HoldLanding(n)) if n.0 == x.slot() => {
                        let rt = shared.upgrade().unwrap();
                        let cell = rt.nodes.get(x.slot()).state_cell();
                        let seen = rt.seen_quietly(cell, crate::state::load::<Shared>(cell));
                        if seen == NodeState::Dirty
                            && stage.compare_exchange(0, 1, SeqCst, SeqCst).is_ok()
                        {
                            wait(&stage, 2, 4000);
                        }
                    }
                    (Some("b"), ParkPoint::BeforeReservationCheck(n))
                        if n.0 == x.slot()
                            && stage.compare_exchange(1, 2, SeqCst, SeqCst).is_ok() =>
                    {
                        // Holds the claim while A loads X's state, for at
                        // most a moment if A waits for its give-back.
                        wait(&stage, 3, 200);
                    }
                    (Some("a"), ParkPoint::BeforeEntry(n)) if n.0 == t.slot() => {
                        let _ = stage.compare_exchange(2, 3, SeqCst, SeqCst);
                    }
                    _ => {}
                }
            }));
            assert!(installed.is_ok());
        }
        gate.arm();
        let a = {
            let rt = rt.clone();
            spawn("a", move || rt.get(t))
        };
        gate.wait_parked();
        let b = {
            let rt = rt.clone();
            spawn("b", move || rt.get(x))
        };
        rt.set(i, 2);
        gate.release();
        let (a, b) = (a.join(), b.join());
        rt.assert_invariant_i();
        (a, b, stage.load(SeqCst))
    }
    let (a, b, _) = run(u32::MAX);
    let (ga, gb, stage) = run(1);
    assert_eq!(stage, 3, "the interleaving was not pinned");
    assert_eq!((ga, gb), (a, b));
}

/// **A reservation does not outlive the entry that reran its node**: R
/// reads V, V joins another thread's read of X, parks, and then reads X.
/// X reads input I and then Y1 while I is 1, else Y2, a node no run has
/// computed. A set of I while V is parked leaves X Dirty, so V's read
/// runs X, X runs Y2 below it, and at budget 2 R dooms on Y2 and
/// reserves V and X. X's entry parks after reading Y, a second set of I
/// flags it, and it lands Dirty. V's entry runs V again from its start,
/// so it joins a read of X again: X's run is over, as in the unbounded
/// run, so that read must rerun X rather than wait on the reservation.
#[test]
fn shared_reservation_ends_with_the_entry_that_reran_its_node() {
    entry_rerun_join(First::None);
}

/// As [`shared_reservation_ends_with_the_entry_that_reran_its_node`], with
/// X first catching the panic of A, a node that panics when I is not 1, so
/// the doom holds A's unwind for X's frame while V's retry joins.
#[test]
fn shared_reservation_ends_with_the_entry_that_reran_its_node_over_an_outcome() {
    entry_rerun_join(First::Caught);
}

/// X's first read, before it reads Y1 or Y2.
#[derive(Clone, Copy, PartialEq, Debug)]
enum First {
    None,
    /// A, which panics when I is not 1, caught as 7.
    Caught,
}

fn entry_rerun_join(first: First) {
    fn run(frames: u32, first: First) -> (Outcome, u32, bool, bool, bool) {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(frames);
        let i = rt.create_input(1_u64);
        let y1 = rt.create_query_with(move |rt: &Rt| rt.get(i) + 10, replayable());
        let y2 = rt.create_query_with(move |rt: &Rt| rt.get(i) + 10, replayable());
        let a = rt.create_query_with(
            move |rt: &Rt| {
                let i = rt.get(i);
                if i != 1 && first == First::Caught {
                    panic!("a boom");
                }
                i + 100
            },
            replayable(),
        );
        let (gx, gv) = (Gate::new(), Gate::new());
        let x = {
            let gx = gx.clone();
            rt.create_query_with(
                move |rt: &Rt| {
                    let a = match first {
                        First::None => 0,
                        First::Caught => catch_unwind(AssertUnwindSafe(|| rt.get(a))).unwrap_or(7),
                    };
                    let v = if rt.get(i) == 1 {
                        rt.get(y1)
                    } else {
                        rt.get(y2)
                    };
                    if v != 0 {
                        gx.park();
                    }
                    v + 1 + a
                },
                replayable(),
            )
        };
        let joins = Arc::new(AtomicU32::new(0));
        let stalled = Arc::new(AtomicBool::new(false));
        let held = Arc::new(AtomicBool::new(false));
        let v = {
            let (gv, joins, stalled) = (gv.clone(), joins.clone(), stalled.clone());
            let held = held.clone();
            let shared = Arc::downgrade(&rt);
            rt.create_query_with(
                move |rt: &Rt| {
                    joins.fetch_add(1, SeqCst);
                    let holds_x = crate::replay::with(|c| {
                        c.iter().flat_map(|ctx| ctx.pending.iter()).any(|d| {
                            !d.taken()
                                && d.key.node == x.slot()
                                && matches!(d.what, crate::replay::Delivered::Unwind { .. })
                        })
                    });
                    if holds_x {
                        held.store(true, SeqCst);
                    }
                    let waits = rt.reservation_waits.load(SeqCst);
                    let (tx, rx) = mpsc::channel();
                    let rt2 = shared.upgrade().unwrap();
                    thread::spawn(move || {
                        let _ = tx.send(outcome(|| rt2.get(x)));
                    });
                    let deadline = Instant::now() + JOIN_TIMEOUT;
                    let joined = loop {
                        if let Ok(read) = rx.try_recv() {
                            break read.unwrap_or(1000);
                        }
                        if rt.reservation_waits.load(SeqCst) != waits {
                            stalled.store(true, SeqCst);
                            break 2000;
                        }
                        assert!(
                            Instant::now() < deadline,
                            "the joined read neither returned nor waited"
                        );
                        thread::yield_now();
                    };
                    gv.park();
                    joined + rt.get(x)
                },
                replayable(),
            )
        };
        gv.arm();
        let r = {
            let rt = rt.clone();
            spawn("r", move || {
                let got = rt.get(v);
                assert!(crate::reserve::none_held(), "R's read left a reservation");
                got
            })
        };
        gv.wait_parked();
        rt.set(i, 2);
        gx.arm();
        gv.release();
        gx.wait_parked();
        rt.set(i, 3);
        gx.release();
        let read = r.join();
        rt.assert_invariant_i();
        let doomed = !rt.dooms.lock().unwrap().is_empty();
        (
            read,
            joins.load(SeqCst),
            stalled.load(SeqCst),
            doomed,
            held.load(SeqCst),
        )
    }
    let (_, joins, stalled, doomed, _) = run(u32::MAX, first);
    assert_eq!(
        (joins, stalled, doomed),
        (1, false, false),
        "the unbounded run, {first:?}"
    );
    let (read, joins, stalled, doomed, held) = run(2, first);
    assert!(doomed, "no doom ran, {first:?}");
    if first == First::Caught {
        assert!(held, "no outcome was held for X when V's retry joined");
    }
    // A run that joins after the retry's may meet the doom again.
    assert!(joins >= 2, "V's retry did not join, {first:?}");
    assert!(
        matches!(read, Ok(v) if v < 1000),
        "R's read or a joined read failed, {first:?}: {read:?}"
    );
    assert!(
        !stalled,
        "the joined read waited on R's reservation, {first:?}: {read:?}"
    );
}

/// T reads X, which dooms at budget 1, and runs on past the placeholder:
/// it hits a table, asks whether the hit is alive, reads it and reads
/// another runtime's Clean node. What those fast reads mark on T's frame
/// goes with its discarded run, so its retry gives the unbounded value.
#[test]
fn a_doomed_run_on_past_its_placeholder_leaves_its_retry_the_unbounded_value() {
    for frames in [u32::MAX, 1] {
        let rt = Arc::new(Rt::new());
        let other = Arc::new(Rt::new());
        let c = other.create_input(5_u64);
        let oc = other.create_query(move |rt: &Rt| rt.get(c) * 2);
        assert_eq!(other.get(oc), 10);
        rt.set_placeholder(0_u64);
        let table = Arc::new(rt.create_table::<u64, u64>());
        let h = table.get_or_insert_input(&rt, 1, || 7);
        let i = rt.create_input(3_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 10, replayable());
        let runs = Arc::new(AtomicU32::new(0));
        let t = {
            let (table, other, runs) = (table.clone(), other.clone(), runs.clone());
            rt.create_query_with(
                move |rt: &Rt| {
                    runs.fetch_add(1, SeqCst);
                    let v = rt.get(x);
                    let hit = table.get_or_insert_input(rt, 1, || 99);
                    v + rt.get(hit) + u64::from(rt.is_alive(hit)) + other.get(oc)
                },
                replayable(),
            )
        };
        rt.set_max_frames(frames);
        assert_eq!(rt.get(t), 14 + 7 + 1 + 10, "budget {frames}");
        assert_eq!(table.get(&rt, &1), Some(h), "budget {frames}");
        let want = if frames == 1 { 2 } else { 1 };
        assert_eq!(runs.load(SeqCst), want, "budget {frames}");
        rt.assert_invariant_i();
        other.assert_invariant_i();
        assert!(
            crate::reserve::none_held(),
            "budget {frames}: a reservation outlived the read"
        );
    }
}

/// Reads Y, then joins a thread that reads X, when dropped by a panic.
struct JoinOnPanic<'a> {
    rt: &'a Rt,
    x: Incr<u64>,
    y: Incr<u64>,
}

impl Drop for JoinOnPanic<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            let (rt, x) = (self.rt, self.x);
            let _ = rt.get(self.y);
            thread::scope(|s| s.spawn(|| rt.get(x)).join().unwrap());
        }
    }
}

/// T reads X, which dooms at budget 1, and panics on the placeholder; a
/// local's drop then reads Y, which takes the placeholder, and joins a
/// thread that reads X, which the doom reserved: the child takes X over
/// from the panicking doom instead of waiting on the joining thread.
#[test]
#[cfg_attr(miri, ignore)]
fn a_panic_past_a_placeholder_keeps_the_doom_in_flight_for_its_drops() {
    run_in_child(
        "runtime::reserve_tests::a_panic_past_a_placeholder_keeps_the_doom_in_flight_for_its_drops",
        || {
            for frames in [u32::MAX, 1] {
                let rt = Arc::new(Rt::new());
                rt.set_placeholder(0_u64);
                let i = rt.create_input(3_u64);
                let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
                let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 10, replayable());
                let t = rt.create_query_with(
                    move |rt: &Rt| {
                        let v = rt.get(x);
                        let _guard = JoinOnPanic { rt, x, y };
                        assert_ne!(v, 0, "placeholder");
                        v + 14
                    },
                    replayable(),
                );
                rt.set_max_frames(frames);
                assert_eq!(within_10s(&rt, t), Some(Ok(28)), "budget {frames}");
                rt.assert_invariant_i();
                assert!(
                    !crate::reserve::any(),
                    "budget {frames}: a reservation outlived the read"
                );
            }
        },
    )
    .assert_clean();
}

/// T joins a thread that reads its parent P, but only on a placeholder
/// for X, so the unbounded run never joins. Over budget the doomed T runs
/// on into the join, and the reader waits on P, which this thread computes:
/// a hang, as on main b9f0d0bf, where a doomed run goes on in user code
/// too. Known limitation: the bounded run may hang, killed by the child's
/// timeout, but never gives another value.
#[test]
#[cfg_attr(miri, ignore)]
fn a_placeholder_join_of_a_reader_of_the_parent_hangs_as_on_main() {
    fn shape(frames: u32) -> Option<Outcome> {
        let rt = Arc::new(Rt::new());
        rt.set_placeholder(0_u64);
        let i = rt.create_input(3_u64);
        let y = rt.create_query_with(move |rt: &Rt| rt.get(i) + 1, replayable());
        let x = rt.create_query_with(move |rt: &Rt| rt.get(y) + 10, replayable());
        let parent = Arc::new(OnceLock::<Incr<u64>>::new());
        let t = {
            let parent = parent.clone();
            rt.create_query_with(
                move |rt: &Rt| {
                    let v = rt.get(x);
                    if v == 0 {
                        let p = *parent.get().unwrap();
                        thread::scope(|s| s.spawn(|| rt.get(p)).join().unwrap());
                    }
                    v
                },
                replayable(),
            )
        };
        let p = rt.create_query_with(move |rt: &Rt| rt.get(t) + 1, replayable());
        parent.set(p).unwrap();
        rt.set_max_frames(frames);
        within_10s(&rt, p)
    }
    assert_eq!(shape(u32::MAX), Some(Ok(15)));
    // A hung budget leaves its doom's reservations in the process, so the
    // budgets after it may hang on them too; each still gets its 10 s.
    let child = run_in_child_within(
        "runtime::reserve_tests::a_placeholder_join_of_a_reader_of_the_parent_hangs_as_on_main",
        Duration::from_secs(40),
        || {
            for frames in [1, 2, 3] {
                let got = shape(frames);
                assert!(
                    matches!(got, None | Some(Ok(15))),
                    "budget {frames}: {got:?}"
                );
            }
        },
    );
    assert!(
        matches!(child, ChildOutcome::Clean | ChildOutcome::Killed),
        "{child:?}"
    );
}
