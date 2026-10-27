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

/// A claim no walk listing in these tests names as a peer.
fn outside(rt: &Rt) -> crate::waits::Claim {
    (rt.runtime_id, 9)
}

/// A walk listing holds every op until its thread next enters user code
/// outside its peers, as the unbounded run holds the node while it runs
/// only engine code; there it ends, so a reader that code joins passes
/// whenever it began.
#[test]
fn a_walk_listing_ends_at_its_threads_next_entry_into_user_code() {
    use crate::reserve::{note_walk, user_code, Peers, Root};
    let rt = Rt::new();
    let (root, n) = ((rt.runtime_id, 3), (rt.runtime_id, 7));
    let read = Root::arm(true, root);
    note_walk(n, 1, &Peers::new([]));
    assert!(
        checked(begun_op(n)),
        "an op begun after the listing passed before any user code"
    );
    let before = begun_op(n);
    user_code(outside(&rt));
    let after = begun_op(n);
    assert!(!checked(before), "an op begun before user code waited");
    assert!(!checked(after), "an op begun in user code waited");
    assert!(!reserved_elsewhere(n));
    drop(read);
}

/// A walk listing stays awake through the user code of a peer the unwind
/// released with it, whose frame the unbounded run still holds the node
/// under, and ends at the next entry of other user code.
#[test]
fn a_walk_listing_stays_awake_through_its_peers_user_code() {
    use crate::reserve::{note_walk, user_code, Peers, Root};
    let rt = Rt::new();
    let (root, n) = ((rt.runtime_id, 3), (rt.runtime_id, 7));
    let read = Root::arm(true, root);
    note_walk(n, 1, &Peers::new([5, 7]));
    user_code((rt.runtime_id, 5));
    assert!(checked(begun_op(n)), "a peer's user code ended the listing");
    user_code(outside(&rt));
    assert!(!checked(begun_op(n)), "other user code left the listing");
    drop(read);
}

/// A walk listing ends where its thread enters a cycle callback, a peer's
/// or not: the callback may join a thread that reads the node, which the
/// unbounded run, holding no frame of it there, lets pass.
#[test]
fn a_walk_listing_ends_at_its_threads_next_cycle_callback() {
    use crate::reserve::{note_walk, Peers, Root};
    let rt = Rt::new();
    let (root, n) = ((rt.runtime_id, 3), (rt.runtime_id, 7));
    let read = Root::arm(true, root);
    note_walk(n, 1, &Peers::new([7]));
    let before = begun_op(n);
    rt.callback(|| ());
    assert!(!checked(before), "the callback left the listing");
    assert!(!reserved_elsewhere(n));
    drop(read);
}

/// A walk listing ends where its node lands settled, so an op begun before
/// waits no longer than the unbounded run's claim.
#[test]
fn a_walk_listing_ends_at_its_nodes_settled_landing() {
    use crate::reserve::{end_walk_landed, note_walk, Peers, Root};
    let rt = Rt::new();
    let (root, n) = ((rt.runtime_id, 3), (rt.runtime_id, 7));
    let read = Root::arm(true, root);
    note_walk(n, 1, &Peers::new([]));
    let before = begun_op(n);
    end_walk_landed(n, 1);
    assert!(!checked(before), "the settled landing left the listing");
    assert!(!reserved_elsewhere(n));
    drop(read);
}

/// A context that also reserves the node and ends before the walk's
/// thread runs user code leaves the walk listing awake.
#[test]
fn a_context_ending_before_user_code_leaves_its_walk_listing_awake() {
    use crate::reserve::{end_contexts, note, note_walk, user_code, Peers, Root, Scope};
    let rt = Rt::new();
    let (root, n) = ((rt.runtime_id, 3), (rt.runtime_id, 7));
    let read = Root::arm(true, root);
    {
        let _scope = Scope::new(Some(5));
        note(n, 1);
    }
    note_walk(n, 1, &Peers::new([]));
    end_contexts();
    assert!(
        checked(begun_op(n)),
        "the context's end rested the walk listing"
    );
    user_code(outside(&rt));
    assert!(
        !checked(begun_op(n)),
        "the listing stayed awake past user code"
    );
    drop(read);
    assert!(!reserved_elsewhere(n));
}

/// A walk listing that wakes this thread's rested reservation of the node
/// holds every op until the thread next enters user code, then leaves the
/// reservation at its earlier stamp: an op begun after that rest, which
/// the thread may join, passes again, and one begun before still waits.
#[test]
fn a_walk_listing_over_a_rested_reservation_leaves_its_stamp() {
    use crate::reserve::{end_contexts, note, note_walk, rest, user_code, Peers, Root, Scope};
    let rt = Rt::new();
    let (root, n) = ((rt.runtime_id, 3), (rt.runtime_id, 7));
    let read = Root::arm(true, root);
    {
        let _scope = Scope::new(Some(5));
        note(n, 1);
    }
    let before = begun_op(n);
    rest(n, 1);
    let between = begun_op(n);
    note_walk(n, 1, &Peers::new([]));
    assert!(
        checked(begun_op(n)),
        "an op begun after the listing passed before any user code"
    );
    user_code(outside(&rt));
    assert!(!checked(between), "the listing's end restamped the node");
    assert!(
        checked(before),
        "the listing's end dropped the context's hold"
    );
    end_contexts();
    drop(read);
    assert!(!reserved_elsewhere(n));
}

/// A context's rest of a node an awake walk listing also holds takes
/// effect where the listing ends, not before.
#[test]
fn a_rest_under_an_awake_walk_listing_waits_for_its_end() {
    use crate::reserve::{end_contexts, note, note_walk, rest, user_code, Peers, Root, Scope};
    let rt = Rt::new();
    let (root, n) = ((rt.runtime_id, 3), (rt.runtime_id, 7));
    let read = Root::arm(true, root);
    let scope = Scope::new(Some(5));
    note(n, 1);
    note_walk(n, 1, &Peers::new([]));
    let before = begun_op(n);
    rest(n, 1);
    assert!(
        checked(begun_op(n)),
        "the rest ended the awake listing's hold"
    );
    user_code(outside(&rt));
    assert!(checked(before), "an op begun before the rest passed");
    assert!(!checked(begun_op(n)), "an op begun after the rest waited");
    drop(scope);
    end_contexts();
    drop(read);
    assert!(!reserved_elsewhere(n));
}

/// One release of `k` nodes, each listed with all `k` as peers, kept
/// awake through an entry of each and ended at an entry outside: the work
/// stays linear in `k`.
#[test]
fn the_walk_listings_of_one_release_stay_linear() {
    use crate::reserve::{note_walk, take_walk_work, user_code, Peers, Root};
    for k in [1000, 2000, 4000] {
        let rt = Rt::new();
        let read = Root::arm(true, (rt.runtime_id, u32::MAX));
        let peers = Peers::new(0..k);
        for slot in 0..k {
            note_walk((rt.runtime_id, slot), 1, &peers);
        }
        take_walk_work();
        for slot in 0..k {
            user_code((rt.runtime_id, slot));
        }
        user_code((rt.runtime_id, k));
        let work = take_walk_work();
        assert!(work <= 3 * u64::from(k), "{work} units for {k} listings");
        assert!(!reserved_elsewhere((rt.runtime_id, 0)));
        drop(read);
    }
}

/// A nested read's walk listing of a node the enclosing read's walk also
/// lists ends with the nested read's peers and its end, and leaves the
/// enclosing listing awake until that one's own end.
#[test]
fn a_nested_reads_walk_listing_leaves_the_enclosing_one_awake() {
    use crate::reserve::{note_walk, user_code, Peers, Root};
    let rt = Rt::new();
    let n = (rt.runtime_id, 7);
    let outer = Root::arm(true, (rt.runtime_id, 3));
    note_walk(n, 1, &Peers::new([5]));
    let inner = Root::arm(true, (rt.runtime_id, 4));
    note_walk(n, 1, &Peers::new([6]));
    user_code((rt.runtime_id, 5));
    assert!(
        checked(begun_op(n)),
        "the nested listing's end ended the enclosing one"
    );
    drop(inner);
    assert!(
        checked(begun_op(n)),
        "the nested read's end ended the enclosing listing"
    );
    user_code(outside(&rt));
    assert!(!checked(begun_op(n)), "the enclosing listing stayed awake");
    drop(outer);
    assert!(!reserved_elsewhere(n));
}

/// Ending every reservation of the thread, as a wound does, ends its walk
/// listings too: a context's later rest of the node takes effect at once.
#[test]
fn ending_every_reservation_ends_the_walk_listings() {
    use crate::reserve::{end_all, end_contexts, note, note_walk, rest, Peers, Root, Scope};
    let rt = Rt::new();
    let (root, n) = ((rt.runtime_id, 3), (rt.runtime_id, 7));
    let read = Root::arm(true, root);
    note_walk(n, 1, &Peers::new([5]));
    end_all();
    let scope = Scope::new(Some(5));
    note(n, 1);
    let before = begun_op(n);
    rest(n, 1);
    assert!(checked(before), "an op begun before the rest passed");
    assert!(
        !checked(begun_op(n)),
        "an ended listing kept the rested node awake"
    );
    drop(scope);
    end_contexts();
    drop(read);
    assert!(!reserved_elsewhere(n));
}

/// A walk listing over a context's rested reservation of the node, whose
/// context ends while it is awake, leaves a later context's reservation of
/// the node awake where it ends: the earlier stamp was the ended context's.
#[test]
fn a_walk_listings_end_leaves_a_later_contexts_reservation_awake() {
    use crate::reserve::{
        end_context, end_contexts, note, note_walk, rest, user_code, Peers, Root, Scope,
    };
    let rt = Rt::new();
    let (root, n) = ((rt.runtime_id, 3), (rt.runtime_id, 7));
    let read = Root::arm(true, root);
    {
        let _scope = Scope::new(Some(5));
        note(n, 1);
    }
    rest(n, 1);
    note_walk(n, 1, &Peers::new([]));
    end_context(5);
    {
        let _scope = Scope::new(Some(6));
        note(n, 1);
    }
    let after = begun_op(n);
    user_code(outside(&rt));
    assert!(
        checked(after),
        "the listing's end rested the later context's reservation"
    );
    end_contexts();
    drop(read);
    assert!(!reserved_elsewhere(n));
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
