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
use crate::cells::Shared;
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
#[ignore = "T10-reservation-dirty-landing-join: stalls on 34412289 too"]
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
