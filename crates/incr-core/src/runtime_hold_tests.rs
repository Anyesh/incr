//! What an attempt holds under `Shared` (Variant V), and that a lone
//! thread still reads what `Local` reads.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use super::race_tests::{park_at, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{Cycle, CycleStrategy, Fixpoint, QueryOpts};
use crate::deferred::Deferred;
use crate::handle::Incr;
use crate::node::NodeId;
use crate::segmented_nodes::IN_GROUP;
use crate::state::NodeState;

fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = err.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        String::from("<non-string panic>")
    }
}

/// a (Fallback) and b read each other.
fn two_cycle<C: Cells>(rt: &Runtime<C>) -> (Incr<i64>, Incr<i64>) {
    let bc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let bc = bc.clone();
        rt.create_query_with_fallback(move |rt| rt.get(*bc.get().unwrap()) + 1, |_c: &Cycle| 100)
    };
    let b = rt.create_query(move |rt| rt.get(a) * 2);
    bc.set(b).unwrap();
    (a, b)
}

/// A read meets a two-node cycle first and then a wide acyclic tail. The
/// cycle's episode is held, and so is what read it, but the tail read
/// nothing held and lands at once, so other threads never wait on it for
/// the rest of the read.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_hold_stays_on_the_cycle_and_its_readers() {
    let rt = Runtime::<Shared>::new();
    let (a, b) = two_cycle(&rt);
    let tail: Vec<Incr<i64>> = (0..1000)
        .map(|i| {
            let input = rt.create_input(i);
            rt.create_query(move |rt| rt.get(input) + 1)
        })
        .collect();
    let held = Arc::new(AtomicUsize::new(usize::MAX));
    let root = {
        let held = held.clone();
        rt.create_query(move |rt| {
            let mut sum = rt.get(a);
            for &t in &tail {
                sum += rt.get(t);
            }
            held.store(crate::hold::held_count(), Ordering::SeqCst);
            sum
        })
    };
    assert_eq!(rt.get(root), 100 + (1..=1000).sum::<i64>());
    assert_eq!(
        held.load(Ordering::SeqCst),
        2,
        "held at the end of root's run: a and b"
    );
    assert_eq!(
        crate::hold::held_count(),
        0,
        "published at the outermost return"
    );
    assert_eq!(rt.get(b), 200);
    rt.assert_invariant_i();
}

/// A read deletes a node it read after that node read a cycle member, so
/// the node's landing is held when the delete runs. The delete and the
/// read end as under `Local`.
fn delete_of_a_node_landed_earlier_in_the_read<C: Cells>() -> (Result<i64, String>, Option<String>)
{
    let rt = Runtime::<C>::new();
    let (a, _b) = two_cycle(&rt);
    let c = rt.create_query(move |rt| rt.get(a) + 1);
    let deleted: Arc<OnceLock<Option<String>>> = Arc::new(OnceLock::new());
    let root = {
        let deleted = deleted.clone();
        rt.create_query(move |rt| {
            let v = rt.get(c);
            let r = catch_unwind(AssertUnwindSafe(|| rt.delete_node(c)));
            let _ = deleted.set(r.err().map(panic_message));
            v
        })
    };
    let got = catch_unwind(AssertUnwindSafe(|| rt.get(root))).map_err(panic_message);
    (got, deleted.get().cloned().flatten())
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_delete_of_a_node_landed_earlier_in_the_read() {
    assert_eq!(
        delete_of_a_node_landed_earlier_in_the_read::<Shared>(),
        delete_of_a_node_landed_earlier_in_the_read::<Local>()
    );
}

/// The delete of a node whose landing the read holds moves the reshape
/// count, as every delete does, so no stash taken over the node stands.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_delete_of_a_held_landing_counts_a_reshape() {
    let rt = Runtime::<Shared>::new();
    let (a, _b) = two_cycle(&rt);
    let c = rt.create_query(move |rt| rt.get(a) + 1);
    let counted: Arc<OnceLock<(bool, u64)>> = Arc::new(OnceLock::new());
    let root = {
        let counted = counted.clone();
        rt.create_query(move |rt: &Runtime<Shared>| {
            let v = rt.get(c);
            let before = Shared::u64_load_relaxed(&rt.reshapes);
            let deleted = catch_unwind(AssertUnwindSafe(|| rt.delete_node(c))).is_ok();
            let _ = counted.set((deleted, Shared::u64_load_relaxed(&rt.reshapes) - before));
            v
        })
    };
    let _ = catch_unwind(AssertUnwindSafe(|| rt.get(root)));
    assert_eq!(counted.get(), Some(&(true, 1)));
}

/// p reads a held cycle and then d, its first read of d, so p's edge on d
/// waits for the read's exit. Another thread deletes d, which has no
/// dependent yet, and recycles its slot before the exit publishes. As for
/// an unheld compute over a recycled dep, the read's value is not
/// returned, p's retry fails on d's stale handle, and the slot's new node
/// gets no edge.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_held_new_dep_deleted_and_recycled_before_the_exit() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let (a, _) = two_cycle(&*rt);
    let d = rt.create_input(7_i64);
    let p = rt.create_query(move |rt| rt.get(a) + rt.get(d));
    let recycled: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let hook: super::ParkHook = {
        let (weak, recycled) = (Arc::downgrade(&rt), recycled.clone());
        let owner = std::thread::current().id();
        Box::new(move |point| {
            if point != super::ParkPoint::BeforeHeldPublish
                || std::thread::current().id() != owner
                || recycled.get().is_some()
            {
                return;
            }
            let rt = weak.upgrade().unwrap();
            let node = std::thread::spawn(move || {
                rt.delete_node(d);
                rt.create_input(99_i64)
            })
            .join()
            .unwrap();
            recycled.set(node).unwrap();
        })
    };
    assert!(rt.park_hook.set(hook).is_ok());
    let read = catch_unwind(AssertUnwindSafe(|| rt.get(p)));
    assert!(
        read.is_err(),
        "the read returned a value over a deleted dep"
    );
    let recycled = *recycled.get().expect("the read published held landings");
    assert_eq!(recycled.slot(), d.slot(), "the delete freed d's slot");
    assert!(
        !rt.has_reverse_edge(recycled, p),
        "phantom edge on the recycled slot"
    );
    assert_eq!(rt.state_of(p), NodeState::Failed);
    assert!(rt.try_delete_node(recycled).is_ok());
    rt.assert_invariant_i();
}

/// q reads the held cycle and then d, and panics on d's first value, so its
/// Failed landing is held with its edge to d unpublished. A set of d
/// before the exit walks no edge to q: the set must still reach q, or q
/// stays Failed over a value it never read.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_held_failed_landing_sees_a_set_on_its_new_dep_before_the_exit() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let (a, _) = two_cycle(&*rt);
    let d = rt.create_input(7_i64);
    let q = rt.create_query(move |rt| {
        let cycle = rt.get(a);
        let v = rt.get(d);
        assert!(v != 7, "boom");
        cycle * 1000 + v
    });
    let set = Arc::new(AtomicBool::new(false));
    let hook: super::ParkHook = {
        let (weak, set) = (Arc::downgrade(&rt), set.clone());
        let owner = std::thread::current().id();
        Box::new(move |point| {
            if point != super::ParkPoint::BeforeHeldPublish
                || std::thread::current().id() != owner
                || set.load(Ordering::SeqCst)
            {
                return;
            }
            let rt = weak.upgrade().unwrap();
            std::thread::spawn(move || rt.set(d, 8)).join().unwrap();
            set.store(true, Ordering::SeqCst);
        })
    };
    assert!(rt.park_hook.set(hook).is_ok());
    let first = catch_unwind(AssertUnwindSafe(|| rt.get(q)));
    assert!(first.is_err(), "q panics on d's first value");
    assert!(
        set.load(Ordering::SeqCst),
        "the read published held landings"
    );
    let cycle = rt.get(a);
    let again = catch_unwind(AssertUnwindSafe(|| rt.get(q))).map_err(panic_message);
    assert_eq!(
        again,
        Ok(cycle * 1000 + 8),
        "q stayed Failed over the set of d"
    );
    rt.assert_invariant_i();
}

/// d's Failed landing is published by an earlier read. q then reads d and
/// the held cycle, so its own Failed landing is held with its edge to d
/// unpublished. Before the exit a set flips d and another read reruns it
/// to a new Failed landing with no write: the flip must reach q as its
/// edge would have, so that q runs again rather than stay Failed.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_held_failed_landing_sees_its_failed_dep_rerun_before_the_exit() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let (a, _) = two_cycle(&*rt);
    let i = rt.create_input(1_i64);
    let d = rt.create_query(move |rt| -> i64 { panic!("boom {}", rt.get(i)) });
    let q = rt.create_query(move |rt| -> i64 {
        let saw = catch_unwind(AssertUnwindSafe(|| rt.get(d))).map_err(panic_message);
        let cycle = rt.get(a);
        panic!("q saw {saw:?} over {cycle}")
    });
    let armed = Arc::new(AtomicBool::new(false));
    let rerun = Arc::new(AtomicBool::new(false));
    let hook: super::ParkHook = {
        let (weak, armed, rerun) = (Arc::downgrade(&rt), armed.clone(), rerun.clone());
        let owner = std::thread::current().id();
        Box::new(move |point| {
            if point != super::ParkPoint::BeforeHeldPublish
                || std::thread::current().id() != owner
                || !armed.load(Ordering::SeqCst)
                || rerun.load(Ordering::SeqCst)
            {
                return;
            }
            let rt = weak.upgrade().unwrap();
            std::thread::spawn(move || {
                rt.set(i, 2);
                let again = catch_unwind(AssertUnwindSafe(|| rt.get(d))).map_err(panic_message);
                assert_eq!(again, Err(String::from("boom 2")));
            })
            .join()
            .unwrap();
            rerun.store(true, Ordering::SeqCst);
        })
    };
    assert!(rt.park_hook.set(hook).is_ok());
    let early = catch_unwind(AssertUnwindSafe(|| rt.get(d))).map_err(panic_message);
    assert_eq!(early, Err(String::from("boom 1")));
    armed.store(true, Ordering::SeqCst);
    let first = catch_unwind(AssertUnwindSafe(|| rt.get(q))).map_err(panic_message);
    assert!(
        first.as_ref().is_err_and(|m| m.starts_with("q saw")),
        "q ran and panicked: {first:?}"
    );
    assert!(
        rerun.load(Ordering::SeqCst),
        "the read published held landings"
    );
    let again = catch_unwind(AssertUnwindSafe(|| rt.get(q))).map_err(panic_message);
    assert!(
        again.as_ref().is_err_and(|m| m.starts_with("q saw")),
        "q stayed Failed over the rerun of d: {again:?}"
    );
    rt.assert_invariant_i();
}

/// Reads `p`, whose landing is held, with its publication panicking at the
/// first other landing it publishes, where a child sweep's caught panic
/// resumes, and `also` sees every other park point. Returns that landing's
/// slot.
fn panic_publishing_before(
    rt: &Runtime<Shared>,
    p: Incr<i64>,
    also: impl Fn(super::ParkPoint) + Send + Sync + 'static,
) -> u32 {
    let armed = Arc::new(AtomicBool::new(true));
    let p_first = Arc::new(AtomicBool::new(false));
    let at = Arc::new(AtomicUsize::new(usize::MAX));
    let hook: super::ParkHook = {
        let (armed, p_first, at) = (armed.clone(), p_first.clone(), at.clone());
        Box::new(move |point| {
            let super::ParkPoint::HeldPublished(id) = point else {
                also(point);
                return;
            };
            if !armed.load(Ordering::SeqCst) {
                return;
            }
            if id == super::NodeId(p.slot()) {
                p_first.store(true, Ordering::SeqCst);
                return;
            }
            armed.store(false, Ordering::SeqCst);
            at.store(id.0 as usize, Ordering::SeqCst);
            panic!("a held publication panicked on purpose");
        })
    };
    assert!(rt.park_hook.set(hook).is_ok());
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(p))).unwrap_err();
    assert!(panic_message(err).contains("on purpose"));
    assert!(
        !p_first.load(Ordering::SeqCst),
        "p published before the panic"
    );
    at.load(Ordering::SeqCst) as u32
}

/// A held publication panics after its first landing, as a child sweep's
/// caught panic resumes there. p, still to land, had dropped o for n, and
/// publication had already moved its reverse edges: giving p back moves
/// them back, so a later set of o reaches p and n stays deletable, and a
/// landing published over one given back is Dirty, as (I) needs.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_held_publication_panic_gives_back_the_moved_edges() {
    let rt = Runtime::<Shared>::new();
    let t = rt.create_input(1_i64);
    let bc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let bc = bc.clone();
        rt.create_query_with_fallback(
            move |rt| rt.get(*bc.get().unwrap()) + rt.get(t),
            |_c: &Cycle| 100,
        )
    };
    let b = rt.create_query(move |rt| rt.get(a) * 2);
    bc.set(b).unwrap();
    let (s, o, n) = (
        rt.create_input(false),
        rt.create_input(10_i64),
        rt.create_input(20_i64),
    );
    let p = rt.create_query(move |rt| rt.get(a) + if rt.get(s) { rt.get(n) } else { rt.get(o) });
    rt.get(p);
    rt.set(s, true);
    rt.set(t, 2);
    panic_publishing_before(&rt, p, |_| {});
    assert!(rt.has_reverse_edge(o, p), "o lost its edge to p");
    assert!(!rt.has_reverse_edge(n, p), "phantom edge on n");
    rt.assert_invariant_i();
    rt.set(s, false);
    assert_eq!(rt.get(p), rt.get(a) + 10);
    rt.set(o, 15);
    assert_eq!(rt.get(p), rt.get(a) + 15, "a set of o missed p");
    assert!(rt.try_delete_node(n).is_ok());
    rt.assert_invariant_i();
}

/// a and b iterate to a fixpoint from 0, each reading the other; the one
/// `t_in_a` names ends at t, the other at its value.
fn fixpoint_pair<C: Cells>(rt: &Runtime<C>, t: Incr<i64>, t_in_a: bool) -> (Incr<i64>, Incr<i64>) {
    let opts = || QueryOpts {
        cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64)),
        ..QueryOpts::default()
    };
    let bc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let bc = bc.clone();
        rt.create_query_with(
            move |rt| {
                let b = rt.get(*bc.get().unwrap());
                if t_in_a {
                    rt.get(t)
                } else {
                    b
                }
            },
            opts(),
        )
    };
    let b = rt.create_query_with(
        move |rt| {
            let a = rt.get(a);
            if t_in_a {
                a
            } else {
                rt.get(t)
            }
        },
        opts(),
    );
    bc.set(b).unwrap();
    (a, b)
}

/// A held publication panics after one member of a cycle published over
/// the other, which it gives back. Once t goes back to its old value, a
/// read of either member first returns what a fresh runtime does, not the
/// published member's value from the given-back member's held one. Each
/// member reads t in turn, so in one shape the member that published read
/// t only through the one given back.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_held_publication_panic_rereads_over_a_given_back_landing() {
    let mut over_given = false;
    for (t_in_a, a_first) in [(true, true), (true, false), (false, true), (false, false)] {
        let local = Runtime::<Local>::new();
        let t = local.create_input(1_i64);
        let (a, b) = fixpoint_pair(&local, t, t_in_a);
        let want = (local.get(a), local.get(b));
        let rt = Runtime::<Shared>::new();
        let t = rt.create_input(1_i64);
        let (a, b) = fixpoint_pair(&rt, t, t_in_a);
        let p = rt.create_query(move |rt| rt.get(a) + rt.get(b));
        rt.get(p);
        rt.set(t, 2);
        let published = panic_publishing_before(&rt, p, |_| {});
        let t_reader = if t_in_a { a.slot() } else { b.slot() };
        over_given |= published != t_reader;
        rt.assert_invariant_i();
        rt.set(t, 1);
        let got = if a_first {
            (rt.get(a), rt.get(b))
        } else {
            let b = rt.get(b);
            (rt.get(a), b)
        };
        assert_eq!(got, want, "t in a: {t_in_a}, a first: {a_first}");
        assert_eq!(rt.get(p), want.0 + want.1);
        rt.assert_invariant_i();
    }
    assert!(
        over_given,
        "no shape published the member that reads t through the other"
    );
}

/// g (Fallback) reads x while t is 1 and p otherwise; p reads g. Once t is
/// 2 a read of g forms a cycle that lands p, created first, before g, and
/// g's dep list before it read p was t and x.
struct Formed {
    t: Incr<i64>,
    p: Incr<i64>,
    g: Incr<i64>,
}

fn formed_cycle<C: Cells>(rt: &Runtime<C>) -> Formed {
    let t = rt.create_input(1_i64);
    let x = rt.create_input(5_i64);
    let gc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let p = {
        let gc = gc.clone();
        rt.create_query(move |rt| rt.get(*gc.get().unwrap()))
    };
    let g = rt.create_query_with_fallback(
        move |rt| if rt.get(t) == 1 { rt.get(x) } else { rt.get(p) },
        |_c: &Cycle| 100,
    );
    gc.set(g).unwrap();
    Formed { t, p, g }
}

/// Reads `f.g` at t = 2, its publication panicking right after p lands, so
/// g goes back to x's value with its held `changed_at`.
fn panic_after_p(
    rt: &Runtime<Shared>,
    f: &Formed,
    also: impl Fn(super::ParkPoint) + Send + Sync + 'static,
) {
    assert_eq!(rt.get(f.p), 5);
    rt.set(f.t, 2);
    let published = panic_publishing_before(rt, f.g, also);
    assert_eq!(published, f.p.slot(), "g published before p");
}

/// A cycle formed in the hold publishes p over g, then panics and gives g
/// back. With t back at 1, g recomputes to the value it went back to, so
/// its `changed_at` stays the held one: p must recompute rather than
/// verify against it.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_held_publication_panic_reader_of_a_formed_cycle_recomputes() {
    let rt = Runtime::<Shared>::new();
    let f = formed_cycle(&rt);
    panic_after_p(&rt, &f, |_| {});
    rt.set(f.t, 1);
    assert_eq!(rt.get(f.g), 5);
    assert_eq!(rt.get(f.p), 5, "p verified over the given-back g");
    rt.assert_invariant_i();
}

/// As above, with another thread reverting t and reading p once g's
/// give-back has landed.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_held_publication_panic_reader_read_during_the_give_back() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let f = formed_cycle(&*rt);
    let during: Arc<OnceLock<i64>> = Arc::new(OnceLock::new());
    let also = {
        let (weak, during, (t, p)) = (Arc::downgrade(&rt), during.clone(), (f.t, f.p));
        move |point| {
            if point != super::ParkPoint::GaveBack || during.get().is_some() {
                return;
            }
            let rt = weak.upgrade().unwrap();
            let read = std::thread::spawn(move || {
                rt.set(t, 1);
                rt.get(p)
            })
            .join()
            .unwrap();
            during.set(read).unwrap();
        }
    };
    panic_after_p(&rt, &f, also);
    assert_eq!(
        during.get(),
        Some(&5),
        "a read during the give-back saw p over g's held value"
    );
    assert_eq!(rt.get(f.p), 5);
    rt.assert_invariant_i();
}

/// A held publication panics in the first landing it publishes, before
/// that node lands: no node is left Computing without an owner, and every
/// read afterwards returns what Local returns.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_held_publication_panic_before_a_landing_leaves_no_node_computing() {
    let local = Runtime::<Local>::new();
    let (la, lb) = two_cycle(&local);
    let lp = local.create_query(move |rt| rt.get(la) + rt.get(lb));
    let want = (local.get(lp), local.get(la), local.get(lb));
    let rt = Runtime::<Shared>::new();
    let (a, b) = two_cycle(&rt);
    let p = rt.create_query(move |rt| rt.get(a) + rt.get(b));
    let landing = Arc::new(AtomicUsize::new(usize::MAX));
    let hook: super::ParkHook = {
        let landing = landing.clone();
        Box::new(move |point| {
            if let super::ParkPoint::HeldLanding(id) = point {
                if landing.swap(id.0 as usize, Ordering::SeqCst) == usize::MAX {
                    panic!("a held landing panicked on purpose");
                }
            }
        })
    };
    assert!(rt.park_hook.set(hook).is_ok());
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(p))).unwrap_err();
    assert!(panic_message(err).contains("on purpose"));
    assert_ne!(landing.load(Ordering::SeqCst), usize::MAX);
    for (name, state) in [
        ("a", rt.state_of(a)),
        ("b", rt.state_of(b)),
        ("p", rt.state_of(p)),
    ] {
        assert!(
            !matches!(state, NodeState::Computing | NodeState::ComputingDirty),
            "{name} left {state:?}"
        );
    }
    assert_eq!((rt.get(p), rt.get(a), rt.get(b)), want);
    rt.assert_invariant_i();
}

/// A held publication's last landing, p, is published, a set of t dirties
/// it and another thread claims it, and only then does the publication
/// panic, as a child sweep's caught panic resumes after the landing. The
/// panic leaves p alone: the other thread's claim lands its own value.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_held_publication_panic_after_its_last_landing_leaves_a_new_claim_alone() {
    fn graph<C: Cells>(rt: &Runtime<C>, gate: Arc<[AtomicBool; 2]>) -> (Incr<i64>, Incr<i64>) {
        let (a, b) = two_cycle(rt);
        let t = rt.create_input(0_i64);
        let p = rt.create_query(move |rt| {
            if gate[0].load(Ordering::SeqCst) {
                gate[1].store(true, Ordering::SeqCst);
                while gate[0].load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
            }
            rt.get(a) + rt.get(b) + rt.get(t)
        });
        (t, p)
    }
    let local = Runtime::<Local>::new();
    let (lt, lp) = graph(&local, Arc::default());
    local.get(lp);
    local.set(lt, 1);
    let want = local.get(lp);

    let rt = Arc::new(Runtime::<Shared>::new());
    let gate: Arc<[AtomicBool; 2]> = Arc::default();
    let (t, p) = graph(&*rt, gate.clone());
    let claimer: Arc<std::sync::Mutex<Option<std::thread::JoinHandle<i64>>>> = Arc::default();
    let hook: super::ParkHook = {
        let (weak, gate, claimer) = (Arc::downgrade(&rt), gate.clone(), claimer.clone());
        let armed = AtomicBool::new(true);
        Box::new(move |point| {
            if point != super::ParkPoint::HeldPublished(super::NodeId(p.slot()))
                || !armed.swap(false, Ordering::SeqCst)
            {
                return;
            }
            let rt = weak.upgrade().unwrap();
            gate[0].store(true, Ordering::SeqCst);
            *claimer.lock().unwrap() = Some(std::thread::spawn(move || {
                rt.set(t, 1);
                rt.get(p)
            }));
            let start = std::time::Instant::now();
            while !gate[1].load(Ordering::SeqCst) {
                assert!(
                    start.elapsed() < std::time::Duration::from_secs(20),
                    "the other thread never claimed p"
                );
                std::thread::yield_now();
            }
            panic!("a held publication panicked on purpose");
        })
    };
    assert!(rt.park_hook.set(hook).is_ok());
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(p))).unwrap_err();
    assert!(panic_message(err).contains("on purpose"));
    gate[0].store(false, Ordering::SeqCst);
    let claimed = claimer
        .lock()
        .unwrap()
        .take()
        .expect("p was never published");
    let got = claimed.join().map_err(panic_message);
    assert_eq!(got, Ok(want), "the claim on p lost its landing");
    assert_eq!(rt.state_of(p), NodeState::Clean);
    assert_eq!(rt.get(p), want);
    rt.assert_invariant_i();
}

/// a (Fallback) and b read each other, counting their runs.
fn counted_cycle<C: Cells>(rt: &Runtime<C>, runs: &Arc<AtomicUsize>) -> (Incr<i64>, Incr<i64>) {
    let bc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let (bc, runs) = (bc.clone(), runs.clone());
        rt.create_query_with_fallback(
            move |rt| {
                runs.fetch_add(1, Ordering::SeqCst);
                rt.get(*bc.get().unwrap()) + 1
            },
            |_c: &Cycle| 100,
        )
    };
    let b = {
        let runs = runs.clone();
        rt.create_query(move |rt| {
            runs.fetch_add(1, Ordering::SeqCst);
            rt.get(a) * 2
        })
    };
    bc.set(b).unwrap();
    (a, b)
}

fn local_cycle_value_and_runs() -> (i64, usize) {
    let runs = Arc::new(AtomicUsize::new(0));
    let local = Runtime::<Local>::new();
    let (a, b) = counted_cycle(&local, &runs);
    (local.get(a) + local.get(b), runs.load(Ordering::SeqCst))
}

/// A read of runtime A whose compute reads runtime B's cycle: B's held
/// episode publishes at the exit of B's entry, before A's run goes on,
/// and a wound of A's attempt after that leaves B published.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_cross_read_publishes_at_the_other_runtimes_exit() {
    let other = Arc::new(Runtime::<Shared>::new());
    let runs = Arc::new(AtomicUsize::new(0));
    let (a, b) = counted_cycle(&other, &runs);
    let rt = Runtime::<Shared>::new();
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let wounded = Arc::new(AtomicBool::new(false));
    let root = {
        let (other, seen, wounded) = (other.clone(), seen.clone(), wounded.clone());
        rt.create_query(move |rt| {
            let v = other.get(a) + other.get(b);
            seen.lock()
                .unwrap()
                .extend([other.state_of(a), other.state_of(b)]);
            if !wounded.swap(true, Ordering::SeqCst) {
                rt.raise_wound();
            }
            v
        })
    };
    let (want, local_runs) = local_cycle_value_and_runs();
    assert_eq!(rt.get(root), want);
    assert!(seen.lock().unwrap().iter().all(|s| *s == NodeState::Clean));
    assert_eq!(runs.load(Ordering::SeqCst), local_runs);
    assert_eq!(crate::hold::held_count(), 0);
}

/// A runtime the closures of the other read, which outlives every
/// read: `Runtime<Local>` is not `Sync`, so no `Arc` of it can go
/// into a closure.
struct Rt<C: Cells>(*const Runtime<C>);
// SAFETY: only dereferenced on the reading thread, while the runtime
// is alive.
unsafe impl<C: Cells> Send for Rt<C> {}
// SAFETY: as above.
unsafe impl<C: Cells> Sync for Rt<C> {}
impl<C: Cells> Rt<C> {
    fn get(&self) -> &Runtime<C> {
        // SAFETY: see the type.
        unsafe { &*self.0 }
    }
}

/// A read of A runs B's q, which reads back into A's cycle while A's
/// episode is held: q's landing rests on this attempt's unpublished
/// state, so it retracts at the exit of B's entry, and q's next read runs
/// it once more than `Local` does.
fn read_back_into_a_held_cycle<C: Cells>() -> (i64, i64, NodeState, usize) {
    let rt = Runtime::<C>::new();
    let other = Runtime::<C>::new();
    let (x, y) = two_cycle(&rt);
    let q_runs = Arc::new(AtomicUsize::new(0));
    let q = {
        let (back, q_runs) = (Rt(&raw const rt), q_runs.clone());
        other.create_query(move |_| {
            q_runs.fetch_add(1, Ordering::SeqCst);
            back.get().get(y) + 1
        })
    };
    let root = {
        let other = Rt(&raw const other);
        rt.create_query(move |rt| rt.get(x) + other.get().get(q))
    };
    let got = rt.get(root);
    let after = other.state_of(q);
    let again = other.get(q);
    (got, again, after, q_runs.load(Ordering::SeqCst))
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_read_back_into_a_held_cycle_retracts_the_other_runtime() {
    let (got, again, after, runs) = read_back_into_a_held_cycle::<Shared>();
    let (want, want_again, _, local_runs) = read_back_into_a_held_cycle::<Local>();
    assert_eq!((got, again), (want, want_again));
    assert!(
        matches!(after, NodeState::Dirty | NodeState::New),
        "{after:?}"
    );
    assert_eq!(runs, local_runs + 1);
}

/// A read of A's `x` runs B's `q`, which reads `x` back while A's frame
/// holds its claim: the read meets a claim, not a held landing, and gets
/// the cycle `Local` gets, so `q` lands as under `Local`.
fn read_back_into_a_claimed_frame<C: Cells>() -> (i64, i64, NodeState, usize) {
    let rt = Runtime::<C>::new();
    let other = Runtime::<C>::new();
    let x_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let q_runs = Arc::new(AtomicUsize::new(0));
    let q = {
        let (back, x_cell, q_runs) = (Rt(&raw const rt), x_cell.clone(), q_runs.clone());
        other.create_query(move |_| {
            q_runs.fetch_add(1, Ordering::SeqCst);
            back.get()
                .try_get(*x_cell.get().unwrap())
                .map_or(-1, |v| v + 1)
        })
    };
    let x = {
        let other = Rt(&raw const other);
        rt.create_query(move |_| other.get().get(q) + 10)
    };
    x_cell.set(x).unwrap();
    let got = rt.get(x);
    let after = other.state_of(q);
    let again = other.get(q);
    (got, again, after, q_runs.load(Ordering::SeqCst))
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_read_back_into_a_claimed_frame_lands_as_local() {
    let shared = read_back_into_a_claimed_frame::<Shared>();
    let local = read_back_into_a_claimed_frame::<Local>();
    assert_eq!(shared, local);
    assert_eq!(crate::hold::held_count(), 0);
}

/// A's closure catches the wound B's `c` raised and reads B's cycle
/// again: the thread is still wounded, so that second entry of B unwinds
/// with the wound before it lands anything, and the retry reads the cycle
/// once more. A first entry's exit under a wound with nothing unwinding
/// is thus not reached by a read; its retraction is the fallback.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_entry_of_a_wounded_thread_unwinds_before_landing() {
    let other = Arc::new(Runtime::<Shared>::new());
    let runs = Arc::new(AtomicUsize::new(0));
    let (a, b) = counted_cycle(&other, &runs);
    let wounded = Arc::new(AtomicBool::new(false));
    let c = {
        let wounded = wounded.clone();
        other.create_query(move |rt| {
            let v = rt.get(a) + rt.get(b);
            if !wounded.swap(true, Ordering::SeqCst) {
                rt.raise_wound();
            }
            v
        })
    };
    let rt = Runtime::<Shared>::new();
    let second_entry = Arc::new(std::sync::Mutex::new(Vec::new()));
    let root = {
        let (other, second_entry) = (other.clone(), second_entry.clone());
        rt.create_query(move |_| {
            let first = catch_unwind(AssertUnwindSafe(|| other.get(c)));
            if first.is_err() {
                let v = other.get(a) + other.get(b);
                second_entry.lock().unwrap().push(v);
            }
            first.unwrap_or(0)
        })
    };
    let (want, _) = local_cycle_value_and_runs();
    assert_eq!(rt.get(root), want);
    assert!(second_entry.lock().unwrap().is_empty());
    assert_eq!(other.state_of(a), NodeState::Clean);
    assert_eq!(crate::hold::held_count(), 0);
}

/// A wound raised inside B's entry unwinds through it: B's held episode
/// retracts there, and the retry reads B again from scratch.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_wound_through_the_other_runtimes_entry_retracts_it() {
    let other = Arc::new(Runtime::<Shared>::new());
    let runs = Arc::new(AtomicUsize::new(0));
    let (a, b) = counted_cycle(&other, &runs);
    let wounded = Arc::new(AtomicBool::new(false));
    let c = {
        let wounded = wounded.clone();
        other.create_query(move |rt| {
            let v = rt.get(a) + rt.get(b);
            if !wounded.swap(true, Ordering::SeqCst) {
                rt.raise_wound();
            }
            v
        })
    };
    let rt = Runtime::<Shared>::new();
    let root = {
        let other = other.clone();
        rt.create_query(move |_| other.get(c))
    };
    let (want, local_runs) = local_cycle_value_and_runs();
    assert_eq!(rt.get(root), want);
    assert_eq!(other.state_of(a), NodeState::Clean);
    assert_eq!(other.state_of(c), NodeState::Clean);
    assert!(wounded.load(Ordering::SeqCst));
    assert_eq!(runs.load(Ordering::SeqCst), 2 * local_runs);
    assert_eq!(crate::hold::held_count(), 0);
}

/// Another thread drops the last handle of B while a read of A that read
/// B is still running: the drop returns at once, since nothing the read
/// keeps points at B past B's entry.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_drop_of_a_runtime_read_from_another_returns_at_once() {
    use std::sync::mpsc;
    use std::time::Duration;
    let other = Arc::new(Runtime::<Shared>::new());
    let (a, b) = two_cycle(&other);
    let last = other.clone();
    let rt = Runtime::<Shared>::new();
    let (tx_read, rx_read) = mpsc::channel::<()>();
    let (tx_dropped, rx_dropped) = mpsc::channel::<()>();
    let gate = Arc::new(std::sync::Mutex::new(Some((other, tx_read, rx_dropped))));
    let root = {
        let gate = gate.clone();
        rt.create_query(move |_| {
            let Some((other, tx_read, rx_dropped)) = gate.lock().unwrap().take() else {
                return 0;
            };
            let v = other.get(a) + other.get(b);
            drop(other);
            tx_read.send(()).unwrap();
            rx_dropped
                .recv_timeout(Duration::from_secs(10))
                .expect("the drop returned while the read was running");
            v
        })
    };
    let dropper = std::thread::spawn(move || {
        rx_read.recv().unwrap();
        drop(last);
        tx_dropped.send(()).unwrap();
    });
    let local = Runtime::<Local>::new();
    let (la, lb) = two_cycle(&local);
    assert_eq!(rt.get(root), local.get(la) + local.get(lb));
    dropper.join().unwrap();
}

/// After a cross-read, dropping both runtimes frees both: no state of
/// one is kept by the other or by the thread.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_cross_read_frees_both_runtimes() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let other = Arc::new(Runtime::<Shared>::new());
    let (a, b) = two_cycle(&other);
    let root = {
        let other = other.clone();
        rt.create_query(move |_| other.get(a) + other.get(b))
    };
    rt.get(root);
    let (weak_rt, weak_other) = (Arc::downgrade(&rt), Arc::downgrade(&other));
    drop(other);
    drop(rt);
    assert!(weak_rt.upgrade().is_none());
    assert!(weak_other.upgrade().is_none());
}

/// A stabilize of B made inside an entry of A, whose observer computes
/// B's cycle without a read: that entry is B's first on the thread, so
/// B's held landings end at its exit, and another thread reads the cycle
/// at once.
fn stabilize_of_another_runtime_ends_its_holds(enter_a: impl FnOnce(Arc<Runtime<Shared>>)) {
    let (want, _) = local_cycle_value_and_runs();
    let other = Arc::new(Runtime::<Shared>::new());
    let runs = Arc::new(AtomicUsize::new(0));
    let (a, b) = counted_cycle(&other, &runs);
    let fired = Arc::new(AtomicUsize::new(0));
    {
        let fired = fired.clone();
        other.observe(a, move |_: &i64| {
            fired.fetch_add(1, Ordering::SeqCst);
        });
    }
    enter_a(other.clone());
    assert_eq!(fired.load(Ordering::SeqCst), 1);
    assert_eq!(crate::hold::held_count(), 0);
    assert_eq!(other.state_of(a), NodeState::Clean);
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = {
        let other = other.clone();
        std::thread::spawn(move || tx.send(other.get(a) + other.get(b)).unwrap())
    };
    let got = rx.recv_timeout(std::time::Duration::from_secs(5));
    assert_eq!(got, Ok(want), "another thread's read of the cycle");
    reader.join().unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_stabilize_of_another_runtime_ends_its_holds_at_the_exit() {
    stabilize_of_another_runtime_ends_its_holds(|other| {
        let rt = Runtime::<Shared>::new();
        let root = rt.create_query(move |_rt| {
            other.stabilize();
            1_i64
        });
        assert_eq!(rt.get(root), 1);
    });
}

/// As above from A's observer callback, which runs under a barrier frame
/// with no compute of A on top.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_stabilize_of_another_runtime_from_a_callback_ends_its_holds() {
    stabilize_of_another_runtime_ends_its_holds(|other| {
        let rt = Runtime::<Shared>::new();
        let root = rt.create_query(|_rt| 1_i64);
        let called = Arc::new(AtomicBool::new(false));
        {
            let called = called.clone();
            rt.observe(root, move |_: &i64| {
                other.stabilize();
                called.store(true, Ordering::SeqCst);
            });
        }
        rt.stabilize();
        assert!(called.load(Ordering::SeqCst));
    });
}

/// An equal write hands back the rejected new value, not what the slot
/// held, so a held node's undo never takes it: a retraction would restore
/// a value of the wrong shape (`Displaced`) and panic partway.
#[test]
fn shared_held_undo_takes_a_displaced_value_but_not_a_rejected_one() {
    let rt = Runtime::<Shared>::new();
    let k = rt.create_input(1_i64);
    let x = rt.create_query(move |rt| rt.get(k) * 2);
    assert_eq!(rt.get(x), 2);
    let undo_value = |wrote: bool| {
        let node = rt.nodes.get(x.slot());
        let mut bag = crate::deferred::Deferred::new();
        rt.open_hold(super::NodeId(x.slot()), node, false, None, &mut bag);
        rt.displaced(node, Some(Box::new(2_i64)), wrote, &mut bag);
        let kept = crate::hold::take(rt.runtime_id)
            .entries
            .into_iter()
            .map(|e| e.undo.value.is_some())
            .collect::<Vec<_>>();
        bag.drain(&rt);
        kept
    };
    assert_eq!(
        undo_value(false),
        vec![false],
        "a rejected value became the undo"
    );
    assert_eq!(
        undo_value(true),
        vec![true],
        "a displaced value was not kept"
    );
    assert!(!crate::hold::active());
}

/// An entry of `other` that `enter_a` makes inside another runtime's
/// entry ends `other`'s holds at its exit, whatever kind of entry it is
/// and wherever it is made from: nothing stays held, the cycle's nodes
/// are not left Computing, and another thread reads the cycle.
fn entry_of_another_runtime_ends_its_holds(
    enter_a: impl FnOnce(Arc<Runtime<Shared>>, Incr<i64>, Incr<i64>),
) {
    let (want, _) = local_cycle_value_and_runs();
    let other = Arc::new(Runtime::<Shared>::new());
    let runs = Arc::new(AtomicUsize::new(0));
    let (a, b) = counted_cycle(&other, &runs);
    enter_a(other.clone(), a, b);
    assert_eq!(crate::hold::held_count(), 0);
    for n in [a, b] {
        let state = other.state_of(n);
        assert!(
            !matches!(state, NodeState::Computing | NodeState::ComputingDirty),
            "{state:?}"
        );
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = {
        let other = other.clone();
        std::thread::spawn(move || tx.send(other.get(a) + other.get(b)).unwrap())
    };
    let got = rx.recv_timeout(std::time::Duration::from_secs(5));
    assert_eq!(got, Ok(want), "another thread's read of the cycle");
    reader.join().unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_get_of_another_runtime_from_a_callback_ends_its_holds() {
    entry_of_another_runtime_ends_its_holds(|other, a, b| {
        let rt = Runtime::<Shared>::new();
        let root = rt.create_query(|_rt| 1_i64);
        rt.observe(root, move |_: &i64| {
            let _ = other.get(a) + other.get(b);
        });
        rt.stabilize();
    });
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_try_get_of_another_runtime_from_a_callback_ends_its_holds() {
    entry_of_another_runtime_ends_its_holds(|other, a, _b| {
        let rt = Runtime::<Shared>::new();
        let root = rt.create_query(|_rt| 1_i64);
        rt.observe(root, move |_: &i64| {
            let _ = other.try_get(a);
        });
        rt.stabilize();
    });
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_get_of_another_runtime_from_a_local_compute_ends_its_holds() {
    entry_of_another_runtime_ends_its_holds(|other, a, b| {
        let rt = Runtime::<Local>::new();
        let root = rt.create_query(move |_rt| other.get(a) + other.get(b));
        let _ = rt.get(root);
    });
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_stabilize_of_another_runtime_from_a_local_compute_ends_its_holds() {
    entry_of_another_runtime_ends_its_holds(|other, a, _b| {
        other.observe(a, |_: &i64| {});
        let rt = Runtime::<Local>::new();
        let root = rt.create_query(move |_rt| {
            other.stabilize();
            1_i64
        });
        let _ = rt.get(root);
    });
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_get_of_another_runtime_from_a_local_callback_ends_its_holds() {
    entry_of_another_runtime_ends_its_holds(|other, a, b| {
        let rt = Runtime::<Local>::new();
        let root = rt.create_query(|_rt| 1_i64);
        rt.observe(root, move |_: &i64| {
            let _ = other.get(a) + other.get(b);
        });
        rt.stabilize();
    });
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_get_of_another_runtime_from_a_shared_compute_ends_its_holds() {
    entry_of_another_runtime_ends_its_holds(|other, a, b| {
        let rt = Runtime::<Shared>::new();
        let root = rt.create_query(move |_rt| other.get(a) + other.get(b));
        let _ = rt.get(root);
    });
}

/// A walk of B's `x` cleans `f`, which reads B's cycle, held in the
/// attempt, and panics: x's entry contains the panic and holds its
/// payload for f's Failed landing, also held. A wound then retracts the
/// attempt, so f lands Dirty unrun, and the retry's run of f returns. No
/// ticket may outlive the retracted landing it was held for.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_retracted_failed_landing_releases_its_ticket() {
    let other = Arc::new(Runtime::<Shared>::new());
    let k = other.create_input(1_i64);
    let sel = other.create_input(0_i64);
    let j = other.create_input(1_i64);
    let bc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let bc = bc.clone();
        other.create_query_with_fallback(
            move |rt| rt.get(*bc.get().unwrap()) + rt.get(k),
            |_c: &Cycle| 100,
        )
    };
    let b = other.create_query(move |rt| rt.get(a) * 2);
    bc.set(b).unwrap();
    let f_runs = Arc::new(AtomicUsize::new(0));
    let f = {
        let f_runs = f_runs.clone();
        other.create_query(move |rt| {
            let run = f_runs.fetch_add(1, Ordering::SeqCst) + 1;
            let v = rt.get(j) + rt.get(a);
            assert!(rt.get(j) != 0 || run != 2, "f boom {run}");
            v
        })
    };
    let wounded = Arc::new(AtomicBool::new(false));
    // At the wound: the payloads held and the landings this thread holds.
    let at_wound = Arc::new(std::sync::Mutex::new((0, 0)));
    let x = {
        let (wounded, at_wound) = (wounded.clone(), at_wound.clone());
        other.create_query(move |rt| {
            if rt.get(sel) == 0 {
                return rt.get(f);
            }
            if !wounded.swap(true, Ordering::SeqCst) {
                *at_wound.lock().unwrap() = (rt.contained_payloads(), crate::hold::held_count());
                rt.raise_wound();
            }
            100
        })
    };
    // A runtime does not track another's nodes, so each read is a new
    // query of A's.
    let rt = Runtime::<Shared>::new();
    let read_x = |rt: &Runtime<Shared>| {
        let other = other.clone();
        let root = rt.create_query(move |_| other.get(x));
        rt.get(root)
    };
    read_x(&rt);
    other.set(k, 2);
    other.set(j, 0);
    other.set(sel, 1);
    assert_eq!(read_x(&rt), 100);
    assert!(wounded.load(Ordering::SeqCst));
    let (payloads, held) = *at_wound.lock().unwrap();
    assert_eq!(payloads, 1, "f's contained panic was held at the wound");
    assert!(held > 0, "the attempt held landings at the wound");
    assert_eq!(crate::hold::held_count(), 0);
    assert_eq!(f_runs.load(Ordering::SeqCst), 3);
    assert_eq!(other.state_of(f), NodeState::Clean);
    assert_eq!(other.contained_payloads(), 0);
    assert_eq!(other.tickets_kept(), (0, false));
}

/// The states of `p`, `m` and `w`.
type Chain = (NodeState, NodeState, NodeState);

type Handles = (Incr<i64>, Incr<i64>, Incr<i64>);

/// `r` reads `w`, which reads `m`, which reads `p`; a set makes `p` panic,
/// and the panic unwinds through `m` and `w` to `r`, which catches it. With
/// `wound`, `r`'s first run then wounds its attempt, and records the
/// states the unwind left for other threads at that point.
fn unwind_then_wound<C: Cells>(wound: bool) -> (i64, Chain, Option<Chain>) {
    let rt = Runtime::<C>::new();
    let i = rt.create_input(1_i64);
    let p = rt.create_query(move |rt| {
        let v = rt.get(i);
        assert!(v != 2, "p boom");
        v
    });
    let m = rt.create_query(move |rt| rt.get(p) + 1);
    let w = rt.create_query(move |rt| rt.get(m) + 1);
    let chain: Arc<OnceLock<Handles>> = Arc::new(OnceLock::new());
    let at_wound = Arc::new(std::sync::Mutex::new(None));
    let wounded = Arc::new(AtomicBool::new(!wound));
    let r = {
        let (chain, at_wound, wounded) = (chain.clone(), at_wound.clone(), wounded.clone());
        rt.create_query(move |rt| {
            let v = catch_unwind(AssertUnwindSafe(|| rt.get(w))).unwrap_or(0);
            if !wounded.swap(true, Ordering::SeqCst) {
                let &(p, m, w) = chain.get().unwrap();
                *at_wound.lock().unwrap() = Some((rt.state_of(p), rt.state_of(m), rt.state_of(w)));
                rt.raise_wound();
            }
            v
        })
    };
    chain.set((p, m, w)).unwrap();
    assert_eq!(rt.get(r), 3);
    wounded.store(!wound, Ordering::SeqCst);
    rt.set(i, 2);
    let v = rt.get(r);
    let after = (rt.state_of(p), rt.state_of(m), rt.state_of(w));
    let at = *at_wound.lock().unwrap();
    (v, after, at)
}

/// A user panic's unwind lands every frame it passes. Under `Shared`
/// those landings are held to the attempt's end, so a wound later in the
/// attempt takes them back before any other thread plans from them; the
/// retry then lands what `Local` lands.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_unwind_landings_are_held_until_a_wound_retracts_them() {
    let (want, local_after, _) = unwind_then_wound::<Local>(false);
    let (v, after, at_wound) = unwind_then_wound::<Shared>(true);
    let held = (
        NodeState::Computing,
        NodeState::Computing,
        NodeState::Computing,
    );
    assert_eq!(
        at_wound,
        Some(held),
        "the unwind's landings were published before the wound"
    );
    assert_eq!((v, after), (want, local_after));
    assert_eq!(crate::hold::held_count(), 0);
}

/// `d` catches `f`'s panic, so its attempt holds `f`'s Failed landing, and
/// then joins a thread that reads `f`. The reader waits on the held landing
/// while `d`'s thread waits on the reader in user code, where no claim wait
/// shows: the reader takes the landing over, wounding `d`'s attempt, and
/// both finish with what a single thread reads.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_holder_joining_a_reader_of_its_held_panic_landing_finishes() {
    let rt = Runtime::<Shared>::new();
    let x = rt.create_input(0_i64);
    let f = rt.create_query(move |rt| {
        let v = rt.get(x);
        assert!(v % 2 == 1, "even");
        v
    });
    let joined = Arc::new(std::sync::Mutex::new(Vec::new()));
    let d = {
        let joined = joined.clone();
        rt.create_query(move |rt: &Runtime<Shared>| {
            let v = catch_unwind(AssertUnwindSafe(|| rt.get(f))).map_or(7, |v| v * 10);
            let failed = std::thread::scope(|s| {
                s.spawn(|| catch_unwind(AssertUnwindSafe(|| rt.get(f))).is_err())
                    .join()
                    .unwrap()
            });
            joined.lock().unwrap().push(failed);
            v
        })
    };
    assert_eq!(rt.get(d), 7);
    let joined = std::mem::take(&mut *joined.lock().unwrap());
    assert!(!joined.is_empty() && joined.iter().all(|&failed| failed));
    assert_eq!(rt.state_of(f), NodeState::Failed);
    assert_eq!(crate::hold::held_count(), 0);
    assert_eq!(crate::waits::registered_on(rt.runtime_id), 0);
    rt.set(x, 1);
    assert_eq!(rt.get(d), 10);
}

/// `d` catches `f`'s panic, so its attempt holds `f`'s Failed landing, and
/// then reads `g`, which another thread claimed first and whose run reads
/// `f`: each thread waits on the other, through a claim and a held
/// landing. Both finish with what a single thread reads, and neither is
/// left holding or registered.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_holder_waiting_on_a_claim_of_a_reader_of_its_held_panic_landing_finishes() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let (caught, claimed) = (
        crate::runtime::race_tests::Gate::new(),
        crate::runtime::race_tests::Gate::new(),
    );
    let x = rt.create_input(0_i64);
    let f = rt.create_query(move |rt| {
        let v = rt.get(x);
        assert!(v % 2 == 1, "even");
        v
    });
    let g = {
        let claimed = claimed.clone();
        rt.create_query(move |rt: &Runtime<Shared>| {
            claimed.park();
            catch_unwind(AssertUnwindSafe(|| rt.get(f))).unwrap_or(1)
        })
    };
    let d = {
        let caught = caught.clone();
        rt.create_query(move |rt: &Runtime<Shared>| {
            let v = catch_unwind(AssertUnwindSafe(|| rt.get(f))).map_or(7, |v| v * 10);
            caught.park();
            v + rt.get(g)
        })
    };
    let read = |rt: &Arc<Runtime<Shared>>, n: Incr<i64>| {
        let rt = Arc::clone(rt);
        std::thread::spawn(move || {
            let v = rt.get(n);
            (v, crate::hold::held_count())
        })
    };
    caught.arm();
    let holder = read(&rt, d);
    caught.wait_parked();
    claimed.arm();
    let reader = read(&rt, g);
    claimed.wait_parked();
    claimed.release();
    caught.release();
    assert_eq!(holder.join().unwrap(), (8, 0));
    assert_eq!(reader.join().unwrap(), (1, 0));
    assert_eq!(crate::waits::registered_on(rt.runtime_id), 0);
    assert_eq!(rt.state_of(f), NodeState::Failed);
    rt.assert_invariant_i();
    rt.set(x, 1);
    assert_eq!(rt.get(d), 11);
}

/// Once `on` is set, `r`'s read walks `d` as an entry, and `d` first runs
/// `f` beneath it, catches `f`'s panic and parks while its attempt holds
/// `f`'s Failed landing. A set of `x`, `f`'s dep, then finds no published
/// edge to walk, so `f` lands Dirty at the attempt's end rather than
/// Failed, and so do `d` and `r` over it: the next read runs them again.
fn held_failed_landing_over_a_moved_dep(
    stop: bool,
) -> (Arc<Runtime<Shared>>, Incr<i64>, Incr<i64>) {
    let rt = Arc::new(Runtime::<Shared>::new());
    rt.set_max_frames(16);
    let gate = crate::runtime::race_tests::Gate::new();
    let x = rt.create_input(0_i64);
    let on = rt.create_input(0_i64);
    let f = rt.create_query_with(
        move |rt| {
            let v = rt.get(x);
            assert!(v % 2 == 1, "even");
            v
        },
        replayable(),
    );
    let d = {
        let gate = gate.clone();
        rt.create_query(move |rt: &Runtime<Shared>| {
            if rt.get(on) == 0 {
                return 0;
            }
            let v = catch_unwind(AssertUnwindSafe(|| rt.get(f))).map_or(7, |v| v * 10);
            gate.park();
            v
        })
    };
    let r = rt.create_query(move |rt: &Runtime<Shared>| rt.get(d));
    assert_eq!(rt.get(r), 0);
    rt.set(on, 1);
    gate.arm();
    let reader = {
        let rt = Arc::clone(&rt);
        std::thread::spawn(move || rt.get(r))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(f), NodeState::Computing);
    rt.set(x, 1);
    if stop {
        let f_id = NodeId(f.slot());
        let once = AtomicBool::new(true);
        let hook: super::ParkHook = Box::new(move |point| {
            if point == ParkPoint::HeldLanding(f_id) && once.swap(false, Ordering::SeqCst) {
                panic!("a held landing panicked on purpose");
            }
        });
        assert!(rt.park_hook.set(hook).is_ok());
    }
    gate.release();
    match reader.join() {
        Ok(v) => assert!(!stop && v == 7, "{v}"),
        Err(_) => assert!(stop),
    }
    (rt, f, r)
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_held_failed_landing_over_a_dep_a_set_moved_lands_dirty() {
    let (rt, f, r) = held_failed_landing_over_a_moved_dep(false);
    assert_eq!(rt.state_of(f), NodeState::Dirty);
    rt.assert_invariant_i();
    assert_eq!(rt.get(r), 10);
}

/// An unheld dep left Dirty with no set since the read did not move: in
/// 0x366149ed a budget's unrun releases left one so, and lowering its
/// Failed reader published six more Failed landings Dirty, one of which
/// a read then verified back to its old value.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_a_dep_left_dirty_with_no_set_since_the_read_did_not_move() {
    let rt = Runtime::<Shared>::new();
    let x = rt.create_input(0_i64);
    let q = rt.create_query(move |rt: &Runtime<Shared>| rt.get(x));
    assert_eq!(rt.get(q), 0);
    rt.set(x, 1);
    assert_eq!(rt.state_of(q), NodeState::Dirty);
    let rev = rt.current_revision();
    assert!(!rt.moved_since_read(NodeId(q.slot()), &Default::default(), rev));
    rt.set(x, 2);
    assert!(rt.moved_since_read(NodeId(q.slot()), &Default::default(), rev));
}

/// The landing lowered to Dirty drops its run's failed-run record and the
/// ticket of the panic d contained, as a held Failed landing a set flagged
/// does: the node is Failed no longer. With `stop`, a panic at the held
/// landing stops the publication before f lands.
fn lowered_held_failed_landing_drops_its_record(stop: bool) {
    let (rt, f, _) = held_failed_landing_over_a_moved_dep(stop);
    assert_eq!(rt.state_of(f), NodeState::Dirty);
    assert_eq!(rt.failed_run_token(f.slot(), f.generation()), None);
    assert_eq!(rt.contained_payloads(), 0);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_held_failed_landing_lowered_over_a_moved_dep_drops_its_record() {
    lowered_held_failed_landing_drops_its_record(false);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_stopped_held_failed_landing_lowered_over_a_moved_dep_drops_its_record() {
    lowered_held_failed_landing_drops_its_record(true);
}

/// As above with g, which lets f's panic out, between d and f: a set
/// that moved f's dep would have flipped g with f, so g lands Dirty too
/// and its next read runs it.
#[test]
fn shared_held_failed_reader_of_a_landing_a_set_moved_lands_dirty() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let gate = crate::runtime::race_tests::Gate::new();
    let x = rt.create_input(0_i64);
    let on = rt.create_input(0_i64);
    let f = rt.create_query(move |rt| -> i64 { panic!("f boom {}", rt.get(x)) });
    let g = rt.create_query(move |rt| rt.get(f) + 1);
    let d = {
        let gate = gate.clone();
        rt.create_query(move |rt: &Runtime<Shared>| {
            if rt.get(on) == 0 {
                return 0;
            }
            let v = catch_unwind(AssertUnwindSafe(|| rt.get(g))).map_or(7, |v| v * 10);
            gate.park();
            v
        })
    };
    let r = rt.create_query(move |rt: &Runtime<Shared>| rt.get(d));
    assert_eq!(rt.get(r), 0);
    rt.set(on, 1);
    gate.arm();
    let reader = {
        let rt = Arc::clone(&rt);
        std::thread::spawn(move || rt.get(r))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(g), NodeState::Computing);
    rt.set(x, 1);
    gate.release();
    assert_eq!(reader.join().unwrap(), 7);
    assert_eq!(
        (rt.state_of(f), rt.state_of(g)),
        (NodeState::Dirty, NodeState::Dirty)
    );
    rt.assert_invariant_i();
    let again = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(g))).unwrap_err());
    assert_eq!(again, "f boom 1");
}

/// The thread's state after a caught wound: whether it held, and what
/// the read after it raised.
type AfterWound = Arc<std::sync::Mutex<Option<(bool, Option<String>)>>>;

/// B's `x` reads B's held cycle, then D's `z`, which reads that cycle
/// back: D retracts at its exit and taints x, which carries `HOLD`. A
/// wound out of x passes no `HOLD` to `Local` A's reader, and the exit of
/// B, the thread's first `Shared` entry, retracts B. The reader catches
/// the wound and reads C, its first entry of C, with nothing held.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_under_local_reader_that_caught_a_wound_out_of_a_retracted_entry_holds_nothing() {
    let b_rt = Arc::new(Runtime::<Shared>::new());
    let (a, b) = two_cycle(&*b_rt);
    let d_rt = Arc::new(Runtime::<Shared>::new());
    let z = {
        let b_rt = b_rt.clone();
        d_rt.create_query(move |_| b_rt.get(a) + 1)
    };
    let wounded = Arc::new(AtomicBool::new(false));
    let x = {
        let (d_rt, wounded) = (d_rt.clone(), wounded.clone());
        b_rt.create_query(move |rt| {
            let v = rt.get(a) + rt.get(b) + d_rt.get(z);
            if !wounded.swap(true, Ordering::SeqCst) {
                rt.raise_wound();
            }
            v
        })
    };
    let c_rt = Arc::new(Runtime::<Shared>::new());
    let q = c_rt.create_query(|_| 7_i64);
    let after: AfterWound = Arc::default();
    let rt = Runtime::<Local>::new();
    let root = {
        let (b_rt, c_rt, after) = (b_rt.clone(), c_rt.clone(), after.clone());
        rt.create_query(move |_| {
            let v = match catch_unwind(AssertUnwindSafe(|| b_rt.get(x))) {
                Ok(v) => v,
                Err(_) => {
                    let held = crate::hold::active();
                    let raised = catch_unwind(AssertUnwindSafe(|| c_rt.get(q)))
                        .err()
                        .filter(|p| p.is::<String>() || p.is::<&str>())
                        .map(panic_message);
                    *after.lock().unwrap() = Some((held, raised));
                    0
                }
            };
            v + c_rt.get(q)
        })
    };
    let got = rt.get(root);
    assert!(wounded.load(Ordering::SeqCst));
    let (held, raised) = after
        .lock()
        .unwrap()
        .take()
        .expect("the reader caught the wound");
    assert_eq!(
        raised, None,
        "the read of C after the caught wound panicked"
    );
    assert!(!held, "the thread holds after B's retraction");
    assert_eq!(got, b_rt.get(x) + 7);
    assert_eq!(crate::hold::held_count(), 0);
}

/// B's `x` reads B's held cycle, then D's `z`, which reads that cycle
/// back: D retracts at its exit and taints x, so B's exit, its first
/// entry under `Local` A's reader, retracts B and taints the reader. A
/// read of C follows. Nothing may stay held past A's outermost exit.
#[test]
#[cfg_attr(miri, ignore)]
fn local_reader_of_a_retracted_shared_entry_holds_nothing_at_its_exit() {
    let b_rt = Arc::new(Runtime::<Shared>::new());
    let (a, b) = two_cycle(&*b_rt);
    let d_rt = Arc::new(Runtime::<Shared>::new());
    let z = {
        let b_rt = b_rt.clone();
        d_rt.create_query(move |_| b_rt.get(a) + 1)
    };
    let x = {
        let d_rt = d_rt.clone();
        b_rt.create_query(move |rt| rt.get(a) + rt.get(b) + d_rt.get(z))
    };
    let c_rt = Arc::new(Runtime::<Shared>::new());
    let q = c_rt.create_query(|_| 7_i64);
    let rt = Runtime::<Local>::new();
    let root = {
        let (b_rt, c_rt) = (b_rt.clone(), c_rt.clone());
        rt.create_query(move |_| b_rt.get(x) + c_rt.get(q))
    };
    let got = rt.get(root);
    assert!(!crate::hold::active(), "the thread holds past its exit");
    assert_eq!(crate::hold::held_count(), 0);
    assert_eq!(got, b_rt.get(x) + 7);
}

thread_local! {
    /// Leaked, so no thread's exit drops it: a runtime dropped in TLS
    /// teardown can reach `BudgetEntry`'s thread-locals after their
    /// destruction while another test's runtime is unbudgeted.
    static LOCAL_RT: &'static Runtime<Local> = Box::leak(Box::new(Runtime::new()));
}

/// x, under an outermost `Shared` read, reads its held cycle and then a
/// `Local` node r that reads the cycle back: r's frame reads a held
/// landing, but no exit publishes a `Local` runtime's holds, so r must land
/// at once and pass the hold to x.
#[test]
#[cfg_attr(miri, ignore)]
fn local_reader_of_a_held_landing_holds_nothing_at_its_exit() {
    let b_rt = Arc::new(Runtime::<Shared>::new());
    let (a, b) = two_cycle(&*b_rt);
    let r = {
        let b_rt = b_rt.clone();
        LOCAL_RT.with(|l| l.create_query(move |_| b_rt.get(a) * 10))
    };
    let x = b_rt.create_query(move |rt| rt.get(a) + rt.get(b) + LOCAL_RT.with(|l| l.get(r)));
    let got = b_rt.get(x);
    assert!(!crate::hold::active(), "the thread holds past its exit");
    assert_eq!(crate::hold::held_count(), 0);
    assert_eq!(got, b_rt.get(a) + b_rt.get(b) + b_rt.get(a) * 10);
    assert_eq!(LOCAL_RT.with(|l| l.get(r)), b_rt.get(a) * 10);
}

/// An observed node reads a cycle and then panics: under `Shared` its
/// Failed landing is held to the end of `stabilize`'s entry, and the
/// observer walk must take it as the node's error, as under `Local`, and
/// go on to the next observer.
fn stabilize_records_a_failure_that_read_a_cycle<C: Cells>() {
    let rt = Runtime::<C>::new();
    let (a, _) = two_cycle(&rt);
    let failing = rt.create_query(move |rt| -> i64 {
        let v = rt.get(a);
        assert!(v < 0, "observed after the cycle");
        v
    });
    let input = rt.create_input(1_i64);
    let after = rt.create_query(move |rt| rt.get(input) + 1);
    let fired = Arc::new(AtomicUsize::new(0));
    rt.observe(failing, |_| {});
    {
        let fired = fired.clone();
        rt.observe(after, move |_| {
            fired.fetch_add(1, Ordering::SeqCst);
        });
    }
    let stabilized = catch_unwind(AssertUnwindSafe(|| rt.stabilize()));
    assert!(
        stabilized.is_ok(),
        "stabilize raised the observed node's panic"
    );
    assert_eq!(fired.load(Ordering::SeqCst), 1);
    assert_eq!(rt.state_of(failing), NodeState::Failed);
}

#[test]
fn local_stabilize_records_a_failure_that_read_a_cycle() {
    stabilize_records_a_failure_that_read_a_cycle::<Local>();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_stabilize_records_a_failure_that_read_a_cycle() {
    stabilize_records_a_failure_that_read_a_cycle::<Shared>();
}

/// A payload no copy can stand for, so a retry regenerates it.
struct HeldPayload(i64);

fn replayable<T>() -> crate::cycle::QueryOpts<T> {
    crate::cycle::QueryOpts {
        replay: crate::cycle::Replay::Allowed,
        ..Default::default()
    }
}

/// r reads a cycle, then catches b, which reads the cycle's held member
/// and panics with `HeldPayload`, then reads a chain past the budget. Under
/// `Shared` b's Failed landing is held to the end of the read, and the
/// retry's regeneration of b's payload must still give r what the
/// unbounded run's read did.
fn held_failed_regeneration_shape<C: Cells>(frames: u32, chain: u32) -> i64 {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_i64);
    rt.set_max_frames(frames);
    let (a, _) = two_cycle(&rt);
    let i = rt.create_input(1_i64);
    let b = rt.create_query_with(
        move |rt: &Runtime<C>| -> i64 { std::panic::panic_any(HeldPayload(rt.get(a))) },
        replayable(),
    );
    let mut deep = rt.create_query_with(move |rt| rt.get(i), replayable());
    for _ in 0..chain {
        let v = deep;
        deep = rt.create_query_with(move |rt| rt.get(v) + 1, replayable());
    }
    let r = rt.create_query_with(
        move |rt| {
            let cycle = rt.get(a);
            let saw = match catch_unwind(AssertUnwindSafe(|| rt.get(b))) {
                Err(p) => p.downcast_ref::<HeldPayload>().map_or(2, |h| h.0),
                Ok(_) => 3,
            };
            cycle * 1_000_000 + saw * 1000 + rt.get(deep)
        },
        replayable(),
    );
    rt.try_get(r).unwrap_or(7777)
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_held_failed_landing_regenerates_its_payload_as_unheld() {
    for chain in [0, 2, 6] {
        for local in [true, false] {
            let shape = |frames| {
                if local {
                    held_failed_regeneration_shape::<Local>(frames, chain)
                } else {
                    held_failed_regeneration_shape::<Shared>(frames, chain)
                }
            };
            let want = shape(u32::MAX);
            assert_eq!(
                want / 1_000_000,
                want % 1_000_000 / 1000,
                "b's payload carries a's value"
            );
            for frames in 1..9 {
                assert_eq!(
                    shape(frames),
                    want,
                    "local={local} chain={chain} frames={frames}"
                );
            }
        }
    }
}

/// a's `try_get(b)` catches the cycle that b's read of a closes.
fn caught_cycle<C: Cells>(rt: &Runtime<C>) -> (Incr<i64>, Incr<i64>) {
    let bc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let bc = bc.clone();
        rt.create_query(move |rt| rt.try_get(*bc.get().unwrap()).unwrap_or(7777))
    };
    let b = rt.create_query(move |rt| rt.get(a) + 1);
    bc.set(b).unwrap();
    (a, b)
}

/// c and d read each other with `get`, so a read of c raises their cycle.
fn raised_cycle(rt: &Runtime<Shared>) -> Incr<i64> {
    let dc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let c = {
        let dc = dc.clone();
        rt.create_query(move |rt| rt.get(*dc.get().unwrap()) + 1)
    };
    let d = rt.create_query(move |rt| rt.get(c) * 2);
    dc.set(d).unwrap();
    c
}

/// A caught cycle's value depends on the member the read entered it at, so
/// its catcher a is held, and so is x, which read a, until the outermost
/// exit publishes them.
#[test]
#[cfg_attr(miri, ignore)]
fn a_caught_cycle_holds_its_catcher_and_its_readers() {
    let rt = Runtime::<Shared>::new();
    let (a, _) = caught_cycle(&rt);
    let held = Arc::new(AtomicUsize::new(usize::MAX));
    let x = {
        let held = held.clone();
        rt.create_query(move |rt| {
            let v = rt.get(a);
            held.store(crate::hold::held_count(), Ordering::SeqCst);
            v + 1
        })
    };
    assert_eq!(rt.get(x), 7778);
    assert_eq!(
        held.load(Ordering::SeqCst),
        1,
        "a is held when x has read it"
    );
    assert!(!crate::hold::active(), "the thread holds past its exit");
    assert_eq!(crate::hold::held_count(), 0);
}

/// a catches the cycle and then reads b with `get`, so its run ends in the
/// cycle's unwind and opens no hold: the hold raised where the cycle closed
/// still ends with the read.
#[test]
#[cfg_attr(miri, ignore)]
fn a_catcher_unwinding_after_its_catch_holds_nothing_past_the_read() {
    let rt = Runtime::<Shared>::new();
    let bc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let bc = bc.clone();
        rt.create_query(move |rt| {
            let b = *bc.get().unwrap();
            let _ = rt.try_get(b);
            rt.get(b)
        })
    };
    let b = rt.create_query(move |rt| rt.get(a) + 1);
    bc.set(b).unwrap();
    let got = catch_unwind(AssertUnwindSafe(|| rt.get(a)));
    assert!(got.is_err(), "a's run ends in the cycle");
    assert!(!crate::hold::active(), "the thread holds past its exit");
    assert_eq!(crate::hold::held_count(), 0);
}

/// x catches a cycle of another `Shared` runtime, which closes and unwinds
/// inside that runtime's first entry: the entry's exit publishes, but x's
/// value still depends on where the cycle was entered, so x stays held,
/// and y, which read x, with it.
#[test]
#[cfg_attr(miri, ignore)]
fn a_cycle_caught_across_runtimes_holds_its_catcher() {
    let rt = Runtime::<Shared>::new();
    let other = Arc::new(Runtime::<Shared>::new());
    let c = raised_cycle(&other);
    let x = {
        let other = other.clone();
        rt.create_query(move |_| other.try_get(c).unwrap_or(7777))
    };
    let held = Arc::new(AtomicUsize::new(usize::MAX));
    let y = {
        let held = held.clone();
        rt.create_query(move |rt| {
            let v = rt.get(x);
            held.store(crate::hold::held_count(), Ordering::SeqCst);
            v + 1
        })
    };
    assert_eq!(rt.get(y), 7778);
    assert_eq!(
        held.load(Ordering::SeqCst),
        1,
        "x is held when y has read it"
    );
    assert!(!crate::hold::active(), "the thread holds past its exit");
    assert_eq!(crate::hold::held_count(), 0);
}

/// a reads e of another `Shared` runtime, and e catches its read of a: the
/// cycle closes under e's frame, which is not a's runtime's, so the read
/// has no active frame there. e's entry exit publishes, and a, whose value
/// depends on where the cycle was entered, stays held while y reads it.
#[test]
#[cfg_attr(miri, ignore)]
fn a_cycle_closed_under_another_runtimes_frame_holds_its_reader() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let other = Arc::new(Runtime::<Shared>::new());
    let ac: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let e = {
        let (rt, ac) = (rt.clone(), ac.clone());
        other.create_query(move |_| rt.try_get(*ac.get().unwrap()).unwrap_or(7777))
    };
    let a = {
        let other = other.clone();
        rt.create_query(move |_| other.get(e) + 1)
    };
    ac.set(a).unwrap();
    let held = Arc::new(AtomicUsize::new(usize::MAX));
    let y = {
        let held = held.clone();
        rt.create_query(move |rt| {
            let v = rt.get(a);
            held.store(crate::hold::held_count(), Ordering::SeqCst);
            v + 1
        })
    };
    assert_eq!(rt.get(y), 7779);
    assert_eq!(
        held.load(Ordering::SeqCst),
        1,
        "a is held when y has read it"
    );
    assert!(!crate::hold::active(), "the thread holds past its exit");
    assert_eq!(crate::hold::held_count(), 0);
}

/// As above with a `Local` frame r between: r catches the cycle and lands
/// at once, as no `Local` landing is held, and the hold goes to x, the
/// nearest `Shared` frame below it.
#[test]
#[cfg_attr(miri, ignore)]
fn a_cycle_caught_under_a_local_frame_holds_the_shared_reader() {
    let rt = Runtime::<Shared>::new();
    let other = Arc::new(Runtime::<Shared>::new());
    let c = raised_cycle(&other);
    let r = LOCAL_RT.with(|l| l.create_query(move |_| other.try_get(c).unwrap_or(7777)));
    let x = rt.create_query(move |_| LOCAL_RT.with(|l| l.get(r)) + 1);
    let held = Arc::new(AtomicUsize::new(usize::MAX));
    let y = {
        let held = held.clone();
        rt.create_query(move |rt| {
            let v = rt.get(x);
            held.store(crate::hold::held_count(), Ordering::SeqCst);
            v + 1
        })
    };
    assert_eq!(rt.get(y), 7779);
    assert_eq!(
        held.load(Ordering::SeqCst),
        1,
        "x is held when y has read it"
    );
    assert!(!crate::hold::active(), "the thread holds past its exit");
    assert_eq!(crate::hold::held_count(), 0);
}

/// With no `Shared` frame below the `Local` catcher, nothing is held, and
/// the inner entry's exit ends the hold its cycle raised.
#[test]
#[cfg_attr(miri, ignore)]
fn a_cycle_caught_under_a_local_root_holds_nothing_past_the_read() {
    let other = Arc::new(Runtime::<Shared>::new());
    let c = raised_cycle(&other);
    let r = LOCAL_RT.with(|l| l.create_query(move |_| other.try_get(c).unwrap_or(7777)));
    assert_eq!(LOCAL_RT.with(|l| l.get(r)), 7777);
    assert!(!crate::hold::active(), "the thread holds past its exit");
    assert_eq!(crate::hold::held_count(), 0);
}

/// Waits for `flag`, or gives up after a few seconds so that a run the
/// test did not plan for still ends.
fn wait_for(flag: &AtomicBool) {
    wait_until(|| flag.load(Ordering::SeqCst));
}

fn wait_until(done: impl Fn() -> bool) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !done() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// Thread 1's read of A holds A's cycle episode and then waits on B's b1,
/// which thread 2 is running; b1 then reads the held member of A. Each
/// thread waits on the other across two runtimes, so the wait registry
/// must see thread 1's held landing of A while it waits in B.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_wait_across_runtimes_on_a_held_episode_ends() {
    crate::child_test::run_in_child(
        "runtime::hold_tests::shared_wait_across_runtimes_on_a_held_episode_ends",
        || {
            let ra = Arc::new(Runtime::<Shared>::new());
            let rb = Arc::new(Runtime::<Shared>::new());
            let (c1, _) = two_cycle(&ra);
            let (claimed, held) = (
                Arc::new(AtomicBool::new(false)),
                Arc::new(AtomicBool::new(false)),
            );
            let b1 = {
                let (ra, claimed, held) = (ra.clone(), claimed.clone(), held.clone());
                rb.create_query(move |_| {
                    claimed.store(true, Ordering::SeqCst);
                    wait_for(&held);
                    ra.get(c1) + 1
                })
            };
            let q = {
                let (rb, held) = (rb.clone(), held.clone());
                ra.create_query(move |rt| {
                    let v = rt.get(c1);
                    held.store(true, Ordering::SeqCst);
                    v + rb.get(b1)
                })
            };
            let t2 = {
                let rb = rb.clone();
                std::thread::spawn(move || rb.get(b1))
            };
            wait_for(&claimed);
            assert_eq!(ra.get(q), 100 + 101);
            assert_eq!(t2.join().unwrap(), 101);
        },
    )
    .assert_clean();
}

/// Thread 2's walk of A's c, inside B's b1, loads c New, and thread 1's
/// read of q claims c and lands A's c-d cycle held. Thread 2's read never
/// restarts after a release, so only the walk's own check keeps it off the
/// held run's deps. In round 1 thread 2 reads c's deps [d] while thread 1
/// holds them, and thread 1, waiting on b1, is wounded. In round 2 thread
/// 2 reads [d] again and stops before checking them; thread 3, older than
/// both from its wait on g, waits on c inside B's x, and thread 1 waits on
/// x and is wounded, retracting c and d to New before thread 2's check,
/// which a check of the open mark alone misses. A walk that kept either
/// expansion computes d before c and enters the cycle at d, which has no
/// strategy; a read from c gets c's fallback.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_walk_skips_deps_another_threads_retracted_held_run_installed() {
    crate::child_test::run_in_child(
        "runtime::hold_tests::shared_walk_skips_deps_another_threads_retracted_held_run_installed",
        || {
            let ra = Arc::new(Runtime::<Shared>::new());
            let rb = Arc::new(Runtime::<Shared>::new());
            let (c, _) = two_cycle(&ra);
            let b1 = {
                let ra = ra.clone();
                rb.create_query(move |_| ra.get(c) + 1)
            };
            let (g_started, g_go, reads_c) = (
                Arc::new(AtomicBool::new(false)),
                Arc::new(AtomicBool::new(false)),
                Arc::new(AtomicBool::new(false)),
            );
            let g = {
                let (g_started, g_go) = (g_started.clone(), g_go.clone());
                rb.create_query(move |_| {
                    g_started.store(true, Ordering::SeqCst);
                    wait_for(&g_go);
                    0
                })
            };
            let x = {
                let (ra, reads_c) = (ra.clone(), reads_c.clone());
                rb.create_query(move |rt| {
                    let gv = rt.get(g);
                    wait_for(&reads_c);
                    ra.get(c) + 1 + gv
                })
            };
            let (held, waiting) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
            let q = {
                let (rb, held, waiting) = (rb.clone(), held.clone(), waiting.clone());
                ra.create_query(move |rt| {
                    let v = rt.get(c);
                    let round = held.fetch_add(1, Ordering::SeqCst) + 1;
                    wait_until(|| waiting.load(Ordering::SeqCst) >= round);
                    v + rb.get(if round == 2 { x } else { b1 })
                })
            };
            let c_id = NodeId(c.erase().slot());
            let (before, after, retracted) = (Gate::new(), Gate::new(), Gate::new());
            park_at(
                &ra,
                vec![
                    (ParkPoint::WalkBeforeDeps(c_id), before.clone()),
                    (ParkPoint::WalkAfterDeps(c_id), after.clone()),
                    (ParkPoint::WoundRetracted, retracted.clone()),
                ],
            );
            let t4 = {
                let rb = rb.clone();
                std::thread::spawn(move || rb.get(g))
            };
            wait_for(&g_started);
            let t3 = {
                let rb = rb.clone();
                std::thread::spawn(move || rb.get(x))
            };
            wait_until(|| crate::waits::registered() >= 1);
            g_go.store(true, Ordering::SeqCst);
            assert_eq!(t4.join().unwrap(), 0);
            wait_until(|| crate::waits::registered() == 0);

            let t2 = {
                let rb = rb.clone();
                std::thread::spawn(move || {
                    super::NO_RESTART_HERE.with(|f| f.set(true));
                    catch_unwind(AssertUnwindSafe(|| rb.get(b1)))
                })
            };
            before.wait_parked();
            let t1 = {
                let ra = ra.clone();
                std::thread::spawn(move || ra.get(q))
            };
            after.wait_parked();
            after.release();
            wait_until(|| held.load(Ordering::SeqCst) >= 1);
            before.arm();
            before.release();
            wait_until(|| crate::waits::registered() >= 1);
            waiting.store(1, Ordering::SeqCst);
            retracted.wait_parked();
            wait_until(|| before.reached() || t2.is_finished());
            if t2.is_finished() {
                let b = t2.join().unwrap().map_err(panic_message);
                panic!("thread 2's read ended in round 1: {b:?}");
            }
            before.wait_parked();

            retracted.arm();
            retracted.release();
            wait_until(|| held.load(Ordering::SeqCst) >= 2);
            after.arm();
            before.arm();
            before.release();
            after.wait_parked();
            reads_c.store(true, Ordering::SeqCst);
            wait_until(|| crate::waits::registered() >= 1);
            waiting.store(2, Ordering::SeqCst);
            retracted.wait_parked();
            before.wait_parked();
            after.release();
            let b = t2.join().unwrap().map_err(panic_message);
            assert_eq!(b, Ok(101));

            waiting.store(3, Ordering::SeqCst);
            before.release();
            retracted.release();
            assert_eq!(t3.join().unwrap(), 101);
            assert_eq!(t1.join().unwrap(), 100 + 101);
        },
    )
    .assert_clean();
}

/// A read of A holds `k` landings of A's two-node cycles, then reads `n`
/// nodes of B, each a first entry of B inside A's entry. Prints the time
/// of the read for the sizes given.
#[test]
#[ignore = "a timing probe"]
fn shared_first_entries_under_held_landings_probe() {
    for size in [1000usize, 2000, 4000, 8000] {
        let ra = Runtime::<Shared>::new();
        let rb = Arc::new(Runtime::<Shared>::new());
        let cycles: Vec<Incr<i64>> = (0..size / 2).map(|_| two_cycle(&ra).0).collect();
        let bs: Vec<Incr<i64>> = (0..size)
            .map(|i| {
                let input = rb.create_input(i as i64);
                rb.create_query(move |rt| rt.get(input))
            })
            .collect();
        let held = Arc::new(AtomicUsize::new(0));
        let root = {
            let (rb, held) = (rb.clone(), held.clone());
            ra.create_query(move |rt| {
                let mut sum = 0;
                for &c in &cycles {
                    sum += rt.get(c);
                }
                held.store(crate::hold::held_count(), Ordering::SeqCst);
                for &b in &bs {
                    sum += rb.get(b);
                }
                sum
            })
        };
        let t = std::time::Instant::now();
        ra.get(root);
        println!(
            "HELDPROBE size {size} held {} read {:.1} ms",
            held.load(Ordering::SeqCst),
            t.elapsed().as_secs_f64() * 1e3
        );
    }
}

/// Thread 2's walk of A's c, inside B's b1, loads c New and stops just
/// before reading c's deps. Thread 1's read of q then lands A's cycles c-d,
/// e-f and k-l held twice, and is wounded each time by an older thread
/// waiting on e (round 1) or k (round 2), retracting them all to New.
/// Thread 2 reads c's deps [d] during round 2's hold and checks them only
/// after round 2's retraction, so a check that counts held episodes in a
/// few bits sees the count it started with. Thread 2's read never restarts
/// after a release. A walk that kept the expansion computes d before c and
/// enters the cycle at d, which has no strategy.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_walk_skips_deps_of_two_held_runs_retracted_during_one_read() {
    crate::child_test::run_in_child(
        "runtime::hold_tests::shared_walk_skips_deps_of_two_held_runs_retracted_during_one_read",
        || {
            let ra = Arc::new(Runtime::<Shared>::new());
            let rb = Arc::new(Runtime::<Shared>::new());
            let (c, _) = two_cycle(&ra);
            let (e, _) = two_cycle(&ra);
            let (k, _) = two_cycle(&ra);
            let b1 = {
                let ra = ra.clone();
                rb.create_query(move |_| ra.get(c) + 1)
            };
            let flag = || Arc::new(AtomicBool::new(false));
            let (z_started, z_go, reads_e, reads_k) = (flag(), flag(), flag(), flag());
            let z = {
                let (z_started, z_go) = (z_started.clone(), z_go.clone());
                rb.create_query(move |_| {
                    z_started.store(true, Ordering::SeqCst);
                    wait_for(&z_go);
                    0
                })
            };
            let waits_on = |member: Incr<i64>, go: &Arc<AtomicBool>| {
                let (ra, go) = (ra.clone(), go.clone());
                rb.create_query(move |rt| {
                    let zv = rt.get(z);
                    wait_for(&go);
                    ra.get(member) + 1 + zv
                })
            };
            let (x1, x2) = (waits_on(e, &reads_e), waits_on(k, &reads_k));
            let (held, waiting) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
            let q = {
                let (rb, held, waiting) = (rb.clone(), held.clone(), waiting.clone());
                ra.create_query(move |rt| {
                    let v = rt.get(c) + rt.get(e) + rt.get(k);
                    let round = held.fetch_add(1, Ordering::SeqCst) + 1;
                    wait_until(|| waiting.load(Ordering::SeqCst) >= round);
                    v + rb.get(match round {
                        1 => x1,
                        2 => x2,
                        _ => b1,
                    })
                })
            };
            let c_id = NodeId(c.erase().slot());
            let (before, after, retracted) = (Gate::new(), Gate::new(), Gate::new());
            park_at(
                &ra,
                vec![
                    (ParkPoint::WalkBeforeDeps(c_id), before.clone()),
                    (ParkPoint::WalkAfterDeps(c_id), after.clone()),
                    (ParkPoint::WoundRetracted, retracted.clone()),
                ],
            );
            let spawn_b = |n: Incr<i64>| {
                let rb = rb.clone();
                std::thread::spawn(move || rb.get(n))
            };
            let t4 = spawn_b(z);
            wait_for(&z_started);
            let (t3, t5) = (spawn_b(x1), spawn_b(x2));
            wait_until(|| crate::waits::registered() >= 2);
            z_go.store(true, Ordering::SeqCst);
            assert_eq!(t4.join().unwrap(), 0);
            wait_until(|| crate::waits::registered() == 0);

            let t2 = {
                let rb = rb.clone();
                std::thread::spawn(move || {
                    super::NO_RESTART_HERE.with(|f| f.set(true));
                    catch_unwind(AssertUnwindSafe(|| rb.get(b1)))
                })
            };
            before.wait_parked();
            let t1 = {
                let ra = ra.clone();
                std::thread::spawn(move || ra.get(q))
            };
            after.wait_parked();
            after.release();
            wait_until(|| held.load(Ordering::SeqCst) >= 1);
            reads_e.store(true, Ordering::SeqCst);
            wait_until(|| crate::waits::registered() >= 1);
            waiting.store(1, Ordering::SeqCst);
            retracted.wait_parked();
            assert_eq!(t3.join().unwrap(), 101);

            retracted.arm();
            retracted.release();
            wait_until(|| held.load(Ordering::SeqCst) >= 2);
            after.arm();
            before.release();
            after.wait_parked();
            reads_k.store(true, Ordering::SeqCst);
            wait_until(|| crate::waits::registered() >= 1);
            waiting.store(2, Ordering::SeqCst);
            retracted.wait_parked();
            assert_eq!(t5.join().unwrap(), 101);
            after.release();
            let b = t2.join().unwrap().map_err(panic_message);
            assert_eq!(b, Ok(101));

            waiting.store(3, Ordering::SeqCst);
            retracted.release();
            assert_eq!(t1.join().unwrap(), 3 * 100 + 101);
        },
    )
    .assert_clean();
}

/// Runs `f` on a fresh thread and returns its result, or `None` if it has
/// not returned within 10 s (the thread is left running).
fn within_10s<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(10)).ok()
}

/// x and w read each other through `try_get`, x a Fallback head if
/// `fallback`; T reads x under a catch and then joins a thread that reads
/// w with `try_get`. T's value, or `None` if T or its child hung, and the
/// take-overs T's read restarted for.
fn join_reader_of_held_member(fallback: bool) -> (Option<u64>, u32) {
    let rt = Arc::new(Runtime::<Shared>::new());
    let i = rt.create_input(3_u64);
    let xc: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let w = {
        let xc = xc.clone();
        rt.create_query(move |rt: &Runtime<Shared>| {
            rt.try_get(*xc.get().unwrap()).map_or(5, |v| v + 1)
        })
    };
    let read_w = move |rt: &Runtime<Shared>| rt.try_get(w).map_or(rt.get(i) + 100, |v| v + 1000);
    let x = if fallback {
        rt.create_query_with(
            read_w,
            crate::cycle::QueryOpts {
                cycle: crate::cycle::CycleStrategy::fallback(|_| 50_u64),
                ..crate::cycle::QueryOpts::default()
            },
        )
    } else {
        rt.create_query(read_w)
    };
    assert!(xc.set(x).is_ok());
    let takes = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let t = {
        let takes = takes.clone();
        rt.create_query(move |rt: &Runtime<Shared>| {
            takes.fetch_max(crate::reserve::takes(), Ordering::SeqCst);
            let v = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_or(0);
            v + std::thread::scope(|s| s.spawn(|| rt.try_get(w).unwrap_or(9)).join().unwrap())
        })
    };
    let got = within_10s(move || rt.get(t));
    (got, takes.load(Ordering::SeqCst))
}

/// **A closure that holds a cycle's landings can join a reader of one.**
/// x and w land Cyclic, held by T's read, which caught the cycle; the
/// child reading w must not wait for T's attempt to end, since T waits for
/// it. The child takes w over and T's read restarts, at most once per held
/// node, with what every order gives.
#[test]
#[cfg_attr(miri, ignore)]
fn a_join_reads_a_member_its_reader_holds_cyclic() {
    let (got, takes) = join_reader_of_held_member(false);
    assert_eq!(got, Some(9));
    assert!(takes <= 2, "{takes} take-overs");
}

/// As above with x a Fallback head, so the episode lands values held.
#[test]
#[cfg_attr(miri, ignore)]
fn a_join_reads_a_member_its_reader_holds_with_a_value() {
    let (got, takes) = join_reader_of_held_member(true);
    assert_eq!(got, Some(101));
    assert!(takes <= 2, "{takes} take-overs");
}

/// **A cycle landed after a take-over sees a later set.** T holds the
/// cycle x, w and joins a child that takes w over, reads it and sets i, so
/// T's retry runs the cycle again with w taken, and w now reads k too: the
/// episode lands unheld, and w's new dep must get its reverse edge as any
/// landing's does, or a later set of k never reaches w, which then reads
/// stale against a runtime that saw i's and k's last values from the start.
#[test]
#[cfg_attr(miri, ignore)]
fn a_cycle_landed_after_a_take_over_sees_a_later_set() {
    let cycle = |input: u64, extra: u64| {
        let rt = Arc::new(Runtime::<Shared>::new());
        let i = rt.create_input(input);
        let k = rt.create_input(extra);
        let xc: Arc<OnceLock<Incr<u64>>> = Arc::default();
        let w = {
            let xc = xc.clone();
            rt.create_query(move |rt: &Runtime<Shared>| {
                let more = if rt.get(i) >= 4 { rt.get(k) } else { 0 };
                rt.try_get(*xc.get().unwrap()).map_or(5, |v| v + 1) + more
            })
        };
        let x = rt.create_query_with(
            move |rt: &Runtime<Shared>| rt.try_get(w).map_or(100, |v| v + 1000),
            crate::cycle::QueryOpts {
                cycle: crate::cycle::CycleStrategy::fallback(|_| 50_u64),
                ..crate::cycle::QueryOpts::default()
            },
        );
        assert!(xc.set(x).is_ok());
        (rt, i, k, x, w)
    };
    let (rt, i, k, x, w) = cycle(3, 20);
    let takes = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let set_once = Arc::new(AtomicBool::new(false));
    let t = {
        let (takes, set_once) = (takes.clone(), set_once.clone());
        rt.create_query(move |rt: &Runtime<Shared>| {
            takes.fetch_max(crate::reserve::takes(), Ordering::SeqCst);
            let v = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_or(0);
            v + std::thread::scope(|s| {
                s.spawn(|| {
                    let r = rt.try_get(w).unwrap_or(9);
                    if !set_once.swap(true, Ordering::SeqCst) {
                        rt.set(i, 4);
                    }
                    r
                })
                .join()
                .unwrap()
            })
        })
    };
    let reader = rt.clone();
    assert!(
        within_10s(move || reader.get(t)).is_some(),
        "T or its child hung"
    );
    assert!(
        takes.load(Ordering::SeqCst) >= 1,
        "T's read was never taken over"
    );
    rt.set(k, 30);
    let after = (rt.try_get(x).ok(), rt.try_get(w).ok());
    let (fresh, _, _, fx, fw) = cycle(4, 30);
    assert_eq!(after, (fresh.try_get(fx).ok(), fresh.try_get(fw).ok()));
}

/// **A reader that dirties the cycle on every run restarts its holder a
/// bounded number of times.** As above, but every child sets i after its
/// read, so each retry of T runs the cycle again. A taken claim is never
/// held again in T's outermost entry, so after the first take-over the
/// next child reads w published and T ends; were w held again, each child
/// would take it over and T would restart for good.
#[test]
#[cfg_attr(miri, ignore)]
fn a_reader_that_dirties_the_cycle_each_run_restarts_its_holder_boundedly() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let i = rt.create_input(0_u64);
    let xc: Arc<OnceLock<Incr<u64>>> = Arc::default();
    let w = {
        let xc = xc.clone();
        rt.create_query(move |rt: &Runtime<Shared>| {
            rt.try_get(*xc.get().unwrap()).map_or(5, |v| v + 1) + rt.get(i)
        })
    };
    let x = rt.create_query_with(
        move |rt: &Runtime<Shared>| rt.try_get(w).map_or(100, |v| v + 1000),
        crate::cycle::QueryOpts {
            cycle: crate::cycle::CycleStrategy::fallback(|_| 50_u64),
            ..crate::cycle::QueryOpts::default()
        },
    );
    assert!(xc.set(x).is_ok());
    let takes = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let runs = Arc::new(AtomicUsize::new(0));
    let t = {
        let (takes, runs) = (takes.clone(), runs.clone());
        rt.create_query(move |rt: &Runtime<Shared>| {
            takes.fetch_max(crate::reserve::takes(), Ordering::SeqCst);
            runs.fetch_add(1, Ordering::SeqCst);
            let v = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_or(0);
            v + std::thread::scope(|s| {
                s.spawn(|| {
                    let r = rt.try_get(w).unwrap_or(9);
                    rt.set(i, rt.get(i) + 1);
                    r
                })
                .join()
                .unwrap()
            })
        })
    };
    let reader = rt.clone();
    assert!(
        within_10s(move || reader.get(t)).is_some(),
        "T restarted for good"
    );
    assert!(
        takes.load(Ordering::SeqCst) >= 1,
        "T's read was never taken over"
    );
    let runs = runs.load(Ordering::SeqCst);
    assert!(runs <= 3, "T ran {runs} times");
}

/// **Two holders that join readers of each other's landings both end.**
/// Each of two runtimes has a `try_get` cycle that its read T catches and
/// holds; once both hold, each T joins a child reading the other's member.
/// Each child takes its member over, so both reads restart, and a taken
/// claim is never held again, so neither restarts for good.
#[test]
#[cfg_attr(miri, ignore)]
fn two_holders_that_join_readers_of_each_others_landings_both_end() {
    struct Side {
        rt: Arc<Runtime<Shared>>,
        x: Incr<u64>,
        w: Incr<u64>,
    }
    fn side() -> Side {
        let rt = Arc::new(Runtime::<Shared>::new());
        let xc: Arc<OnceLock<Incr<u64>>> = Arc::default();
        let w = {
            let xc = xc.clone();
            rt.create_query(move |rt: &Runtime<Shared>| {
                rt.try_get(*xc.get().unwrap()).map_or(5, |v| v + 1)
            })
        };
        let x = rt.create_query(move |rt: &Runtime<Shared>| rt.try_get(w).map_or(7, |v| v + 1));
        assert!(xc.set(x).is_ok());
        Side { rt, x, w }
    }
    let (a, b) = (side(), side());
    let arrived = Arc::new(AtomicUsize::new(0));
    let reader = |me: &Side, other: &Side| {
        let (x, w, other_rt) = (me.x, other.w, other.rt.clone());
        let arrived = arrived.clone();
        let takes = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let t = {
            let takes = takes.clone();
            me.rt.create_query(move |rt: &Runtime<Shared>| {
                takes.fetch_max(crate::reserve::takes(), Ordering::SeqCst);
                let v = catch_unwind(AssertUnwindSafe(|| rt.get(x))).unwrap_or(0);
                // Both hold their cycle before either joins, on the first
                // attempt; a restart finds both arrived.
                arrived.fetch_add(1, Ordering::SeqCst);
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while arrived.load(Ordering::SeqCst) < 2 && std::time::Instant::now() < deadline {
                    std::thread::yield_now();
                }
                let other_rt = other_rt.clone();
                v + std::thread::scope(|s| {
                    s.spawn(move || other_rt.try_get(w).unwrap_or(9))
                        .join()
                        .unwrap()
                })
            })
        };
        let rt = me.rt.clone();
        (
            std::thread::spawn(move || within_10s(move || rt.get(t))),
            takes,
        )
    };
    let (ta, takes_a) = reader(&a, &b);
    let (tb, takes_b) = reader(&b, &a);
    let (got_a, got_b) = (ta.join().unwrap(), tb.join().unwrap());
    assert_eq!((got_a, got_b), (Some(9), Some(9)));
    for takes in [takes_a, takes_b] {
        let n = takes.load(Ordering::SeqCst);
        assert!(n <= 2, "{n} take-overs");
    }
}

/// r reads a cycle k moved and catches x, which reads the cycle's held
/// member and panics, so x's Failed landing is held. At publication x lands
/// otherwise: stopped by a panic before its landing (`stop`), or flagged by
/// a set of i. Another thread's walk of y then runs x, which lands Failed
/// with its own record and a ticket for the panic y's entry contained,
/// before the publication drops what x's held run left: it must leave both.
fn held_failed_target_keeps_the_next_runs_record(stop: bool) {
    let rt = Arc::new(Runtime::<Shared>::new());
    rt.set_placeholder(0_i64);
    rt.set_max_frames(16);
    let k = rt.create_input(1_i64);
    let bc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let bc = bc.clone();
        rt.create_query_with_fallback(
            move |rt| rt.get(*bc.get().unwrap()) + rt.get(k),
            |_c: &Cycle| 100,
        )
    };
    let b = rt.create_query(move |rt| rt.get(a) * 2);
    bc.set(b).unwrap();
    let (i, j, sel) = (
        rt.create_input(0_i64),
        rt.create_input(1_i64),
        rt.create_input(0_i64),
    );
    let x = rt.create_query_with(
        move |rt: &Runtime<Shared>| -> i64 {
            let v = rt.get(a) + rt.get(i);
            if rt.get(j) == 0 {
                std::panic::panic_any(HeldPayload(v));
            }
            v
        },
        replayable(),
    );
    let y = rt.create_query(move |rt| if rt.get(sel) == 0 { rt.get(x) } else { 100 });
    let r = rt.create_query_with(
        move |rt| rt.get(a) + i64::from(catch_unwind(AssertUnwindSafe(|| rt.get(x))).is_err()),
        replayable(),
    );
    let _ = rt.get(y);
    rt.set(k, 2);
    rt.set(j, 0);
    rt.set(sel, 1);
    let x_id = super::NodeId(x.slot());
    let next_run: Arc<OnceLock<(Option<u64>, usize)>> = Arc::default();
    let hook: super::ParkHook = {
        let (weak, next_run) = (Arc::downgrade(&rt), next_run.clone());
        let (landing, otherwise) = (AtomicBool::new(true), AtomicBool::new(true));
        Box::new(move |point| {
            let rt = weak.upgrade().unwrap();
            if point == super::ParkPoint::HeldLanding(x_id) && landing.swap(false, Ordering::SeqCst)
            {
                if stop {
                    panic!("a held landing panicked on purpose");
                }
                let set = Arc::clone(&rt);
                std::thread::spawn(move || set.set(i, 1)).join().unwrap();
            }
            if point == super::ParkPoint::FailedLandedOtherwise(x_id)
                && otherwise.swap(false, Ordering::SeqCst)
            {
                let run = Arc::clone(&rt);
                let walk = std::thread::spawn(move || run.get(y));
                let start = std::time::Instant::now();
                while !walk.is_finished() {
                    assert!(
                        start.elapsed() < std::time::Duration::from_secs(20),
                        "the other thread's walk of y never finished"
                    );
                    std::thread::yield_now();
                }
                assert_eq!(walk.join().unwrap(), 100);
                let record = rt.failed_run_token(x_id.0, x.generation());
                next_run.set((record, rt.contained_payloads())).unwrap();
            }
        })
    };
    assert!(rt.park_hook.set(hook).is_ok());
    let read = catch_unwind(AssertUnwindSafe(|| rt.get(r)));
    assert_eq!(read.is_err(), stop);
    let (b_record, b_tickets) = *next_run.get().expect("x never landed otherwise");
    assert!(b_record.is_some(), "x's run left no record");
    assert_eq!(b_tickets, 1, "y's entry held no payload for x's landing");
    assert_eq!(
        rt.failed_run_token(x_id.0, x.generation()),
        b_record,
        "the publication forgot the record of x's next run"
    );
    assert_eq!(
        rt.contained_payloads(),
        1,
        "the publication released the ticket of x's next run"
    );
    rt.assert_invariant_i();
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_stopped_held_failed_target_keeps_the_next_runs_record() {
    held_failed_target_keeps_the_next_runs_record(true);
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_flagged_held_failed_target_keeps_the_next_runs_record() {
    held_failed_target_keeps_the_next_runs_record(false);
}

/// A group verify's claims that leave their group inside an attempt land
/// held: the attempt's retraction puts each back Dirty, in its group, with
/// its verification.
#[test]
fn shared_group_leave_in_an_attempt_retracts_with_its_record() {
    let rt = Runtime::<Shared>::new();
    let k = rt.create_input(1_i64);
    let m = rt.create_query(move |rt| rt.get(k) + 1);
    let held = rt.create_query(move |rt| rt.get(k) + 2);
    assert_eq!(rt.get(m), 2);
    assert_eq!(rt.get(held), 3);
    let (node, flags) = rt.nodes.get_with_flags(m.slot());
    let group = Arc::new(super::GroupInfo {
        members: vec![(m.slot(), node.generation())],
        cycle: Cycle::new(vec![m.erase()]),
    });
    rt.inner
        .write()
        .set_member_rec(rt.runtime_id, m.slot(), Some(group.clone()));
    Shared::u8_store_relaxed(flags, Shared::u8_load_relaxed(flags) | IN_GROUP);
    let verified = node.verified_at();
    assert_ne!(verified, 0);

    let mut bag = Deferred::new();
    rt.force_state_for_test(held, NodeState::Computing);
    rt.open_hold(
        NodeId(held.slot()),
        rt.nodes.get(held.slot()),
        false,
        None,
        &mut bag,
    );
    rt.force_state_for_test(m, NodeState::Computing);
    rt.release_group_claims(&[m.slot()], true, &mut bag);
    rt.retract_held();
    bag.drain(&rt);

    assert_eq!(rt.state_of(m), NodeState::Dirty);
    let rec = rt.inner.read().member_recs.get(&m.slot()).cloned();
    assert!(
        rec.is_some_and(|r| Arc::ptr_eq(&r, &group)),
        "the retraction left the member out of its group"
    );
    assert_ne!(
        Shared::u8_load_relaxed(flags) & IN_GROUP,
        0,
        "IN_GROUP stayed clear"
    );
    assert_eq!(node.verified_at(), verified);
}

/// A leave inside an attempt, where a waiter took one of the claims,
/// could not be retracted, so every claim stays in its group.
#[test]
fn shared_group_leave_with_a_taken_claim_keeps_the_group() {
    let rt = Runtime::<Shared>::new();
    let k = rt.create_input(1_i64);
    let u = rt.create_query(move |rt| rt.get(k) + 1);
    let m = rt.create_query(move |rt| rt.get(k) + 2);
    let held = rt.create_query(move |rt| rt.get(k) + 3);
    for q in [u, m, held] {
        rt.get(q);
    }
    let mut members = vec![
        (u.slot(), rt.nodes.get(u.slot()).generation()),
        (m.slot(), rt.nodes.get(m.slot()).generation()),
    ];
    members.sort_unstable();
    let group = Arc::new(super::GroupInfo {
        members,
        cycle: Cycle::new(vec![u.erase(), m.erase()]),
    });
    for q in [u, m] {
        rt.inner
            .write()
            .set_member_rec(rt.runtime_id, q.slot(), Some(group.clone()));
        let flags = rt.nodes.get_with_flags(q.slot()).1;
        Shared::u8_store_relaxed(flags, Shared::u8_load_relaxed(flags) | IN_GROUP);
    }

    let mut bag = Deferred::new();
    rt.force_state_for_test(held, NodeState::Computing);
    rt.open_hold(
        NodeId(held.slot()),
        rt.nodes.get(held.slot()),
        false,
        None,
        &mut bag,
    );
    rt.force_state_for_test(u, NodeState::Computing);
    rt.force_state_for_test(m, NodeState::Computing);
    crate::reserve::mark_claims_taken([(rt.runtime_id, m.slot())]);
    rt.release_group_claims(&[u.slot(), m.slot()], true, &mut bag);
    let kept: Vec<(bool, bool)> = [u, m]
        .iter()
        .map(|q| {
            let rec = rt.inner.read().member_recs.get(&q.slot()).cloned();
            let flags = rt.nodes.get_with_flags(q.slot()).1;
            (
                rec.is_some_and(|r| Arc::ptr_eq(&r, &group)),
                Shared::u8_load_relaxed(flags) & IN_GROUP != 0,
            )
        })
        .collect();
    rt.retract_held();
    crate::reserve::clear_taken();
    bag.drain(&rt);

    assert_eq!(rt.state_of(u), NodeState::Dirty);
    assert_eq!(rt.state_of(m), NodeState::Dirty);
    assert_eq!(
        kept,
        vec![(true, true), (true, true)],
        "a claim left its group (record, IN_GROUP) for the untaken and the taken claim"
    );
}

/// x reads a, whose catch of its cycle with b is held until x's exit, then
/// joins a child that reads a. The child must not wait on x's hold, which
/// is raised where the cycle closes at every budget here. Whichever reader
/// of a comes second gets the cycle, as on main, so x returns a value or
/// panics; either way it returns, and each budget gets the unbounded run's
/// outcome.
#[test]
#[cfg_attr(miri, ignore)]
fn a_closure_that_reads_a_caught_cycle_joins_a_reader_of_its_catcher() {
    let run = |frames: u32| {
        let rt = Arc::new(Runtime::<Shared>::new());
        let bc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let a = {
            let bc = bc.clone();
            rt.create_query_with(
                move |rt: &Runtime<Shared>| rt.try_get(*bc.get().unwrap()).unwrap_or(7777),
                replayable(),
            )
        };
        let b = rt.create_query_with(move |rt: &Runtime<Shared>| rt.get(a) + 1, replayable());
        bc.set(b).unwrap();
        let x = rt.create_query_with(
            move |rt: &Runtime<Shared>| {
                let v = rt.get(a);
                v + std::thread::scope(|s| s.spawn(|| rt.try_get(a).unwrap_or(-1)).join().unwrap())
            },
            replayable(),
        );
        rt.set_max_frames(frames);
        let (tx, rx) = std::sync::mpsc::channel();
        let reader = rt.clone();
        std::thread::spawn(move || {
            let _ = tx.send(catch_unwind(AssertUnwindSafe(|| reader.get(x))).ok());
        });
        rx.recv_timeout(std::time::Duration::from_secs(10)).ok()
    };
    let unbounded = run(u32::MAX);
    assert!(unbounded.is_some(), "x does not return");
    for frames in [1, 2] {
        assert_eq!(run(frames), unbounded, "budget {frames}");
    }
}

/// A value without a placeholder, so reads of it nest natively.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Nat(u64);

#[derive(Clone, Copy)]
enum DeliveryNode {
    U(Incr<u64>),
    N(Incr<Nat>),
}

/// The 0x8b19 budget-1 delivery: 6 reads 9 and then 4, whose Fixpoint
/// reads 1; 1 `try_get`s 2 and 2 reads 1. At one frame 2's doomed run
/// closes the cycle after a speculative discard, and the retry of 1 has it
/// delivered (`take_delivery`), so 1 lands its catch, 7777, held. `at_held`
/// runs in 4's closure right after its read of 1, with 1's handle. Returns
/// the `try_get` of 6.
fn caught_delivery_shape(
    frames: u32,
    at_held: impl Fn(&Runtime<Shared>, Incr<u64>) + Send + Sync + 'static,
) -> u64 {
    use crate::cycle::{CycleStrategy, Fixpoint, QueryOpts, Replay};
    let rt = Runtime::<Shared>::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(frames);
    let h: Arc<[OnceLock<DeliveryNode>; 14]> = Arc::new(std::array::from_fn(|_| OnceLock::new()));
    let rd = {
        let h = h.clone();
        move |rt: &Runtime<Shared>, i: usize, try_: bool| match (*h[i].get().unwrap(), try_) {
            (DeliveryNode::U(x), false) => rt.get(x),
            (DeliveryNode::U(x), true) => rt.try_get(x).unwrap_or(7777),
            (DeliveryNode::N(x), false) => rt.get(x).0,
            (DeliveryNode::N(x), true) => rt.try_get(x).map_or(7777, |n| n.0),
        }
    };
    type Compute = Box<dyn Fn(&Runtime<Shared>) -> u64 + Send + Sync>;
    let even = |v: u64| v.is_multiple_of(2);
    let caught = |f: &dyn Fn() -> u64| catch_unwind(AssertUnwindSafe(f)).unwrap_or(9999);
    let q = |f: Compute, cycle: CycleStrategy<u64>, replay: Replay| {
        DeliveryNode::U(rt.create_query_with(move |rt| f(rt), QueryOpts { cycle, replay }))
    };
    let fixpoint = |limit| CycleStrategy::Fixpoint(Fixpoint::new(|| 0_u64).max_iterations(limit));
    let r = rd.clone();
    let n0 = q(
        Box::new(move |rt| r(rt, 5, false)),
        CycleStrategy::None,
        Replay::Allowed,
    );
    let r = rd.clone();
    let n1 = q(
        Box::new(move |rt| r(rt, 2, true)),
        CycleStrategy::None,
        Replay::Allowed,
    );
    let r = rd.clone();
    let n2 = q(
        Box::new(move |rt| {
            let v = r(rt, 1, false);
            if v.is_multiple_of(7) {
                std::panic::panic_any(String::from("boom 2"));
            }
            let c = if even(v) { 0 } else { r(rt, 13, true) };
            if even(c) {
                r(rt, 8, false)
            } else {
                r(rt, 0, false)
            }
        }),
        CycleStrategy::None,
        Replay::Allowed,
    );
    let r = rd.clone();
    let n3 = q(
        Box::new(move |rt| {
            caught(&|| {
                if even(r(rt, 11, false)) {
                    r(rt, 4, false)
                } else {
                    r(rt, 9, false)
                }
            })
        }),
        CycleStrategy::None,
        Replay::Allowed,
    );
    let n4 = {
        let (r, h) = (rd.clone(), h.clone());
        DeliveryNode::N(rt.create_query_with(
            move |rt| {
                Nat(caught(&|| {
                    let one = r(rt, 1, false);
                    let DeliveryNode::U(n1) = *h[1].get().unwrap() else {
                        unreachable!()
                    };
                    at_held(rt, n1);
                    let c = if even(one) {
                        r(rt, 9, false)
                    } else {
                        r(rt, 5, true)
                    };
                    if even(c) {
                        r(rt, 5, true).wrapping_add(r(rt, 12, true))
                    } else {
                        0
                    }
                }))
            },
            QueryOpts {
                cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| Nat(0)).max_iterations(2)),
                replay: Replay::Allowed,
            },
        ))
    };
    let n5 = q(Box::new(|_| 1), fixpoint(3), Replay::Allowed);
    let r = rd.clone();
    let n6 = q(
        Box::new(move |rt| {
            let c = if even(r(rt, 9, false)) {
                r(rt, 4, false)
            } else {
                r(rt, 7, false)
            };
            if c % 7 == 3 {
                std::panic::panic_any(String::from("boom 6"));
            }
            if even(c) {
                let d = if even(r(rt, 8, true)) {
                    r(rt, 1, true)
                } else {
                    r(rt, 7, true)
                };
                4_u64.wrapping_add(d)
            } else {
                r(rt, 2, false)
            }
        }),
        CycleStrategy::None,
        Replay::Allowed,
    );
    let n7 = q(Box::new(|_| 1), CycleStrategy::None, Replay::Allowed);
    let n8 = q(
        Box::new(move |_| caught(&|| 2)),
        CycleStrategy::None,
        Replay::Native,
    );
    let r = rd.clone();
    let n9 = q(
        Box::new(move |rt| {
            if even(caught(&|| r(rt, 10, false))) {
                0
            } else if even(r(rt, 11, false)) {
                if even(r(rt, 13, false)) {
                    r(rt, 5, false)
                } else {
                    r(rt, 1, false)
                }
            } else {
                r(rt, 13, false)
            }
        }),
        CycleStrategy::None,
        Replay::Allowed,
    );
    let inputs = [2_u64, 2, 0, 3].map(|v| rt.create_input(v));
    for (i, n) in [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9]
        .into_iter()
        .enumerate()
    {
        let _ = h[i].set(n);
    }
    for (i, x) in inputs.iter().enumerate() {
        let _ = h[10 + i].set(DeliveryNode::U(*x));
    }
    let DeliveryNode::U(top) = n0 else {
        unreachable!()
    };
    let _ = catch_unwind(AssertUnwindSafe(|| rt.get(top)));
    rt.set(inputs[0], 4);
    rt.set(inputs[3], 1);
    rt.set(inputs[0], 2);
    let DeliveryNode::U(six) = n6 else {
        unreachable!()
    };
    rt.try_get(six).unwrap_or(7777)
}

/// The schedule reaches the delivery: by 4's read of 1 at one frame, a
/// cycle was delivered to 1's frame and 1's catch of it is held; the
/// unbounded run catches the cycle natively.
#[test]
#[cfg_attr(miri, ignore)]
fn the_8b19_schedule_reaches_a_held_delivery() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let run = |frames: u32| {
        seen.lock().unwrap().clear();
        let log = seen.clone();
        let v = caught_delivery_shape(frames, move |rt, one| {
            let delivered = rt
                .deliveries
                .lock()
                .unwrap()
                .iter()
                .any(|d| d.cycle && d.frame == one.slot());
            log.lock()
                .unwrap()
                .push((delivered, crate::hold::held_count()));
        });
        (v, seen.lock().unwrap().clone())
    };
    let (unbounded, at_max) = run(u32::MAX);
    assert!(
        !at_max.is_empty() && at_max.iter().all(|&(d, _)| !d),
        "{at_max:?}"
    );
    let (bounded, at_one) = run(1);
    assert!(at_one.iter().any(|&(d, held)| d && held > 0), "{at_one:?}");
    assert_eq!(bounded, unbounded);
}

/// d catches f's panic, sets f's input and reads f again: the set flips
/// f's landing as it would a published one, so the second read reruns f.
/// Returns both messages.
fn set_after_a_caught_panic<C: Cells>() -> (String, String) {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_i64);
    let k = rt.create_input(0_i64);
    let f = rt.create_query(move |rt: &Runtime<C>| -> i64 { panic!("f boom {}", rt.get(k)) });
    let d = rt.create_query(move |rt: &Runtime<C>| {
        let first = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(f))).unwrap_err());
        rt.set(k, 1);
        let second = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(f))).unwrap_err());
        i64::from(first == "f boom 0") + 2 * i64::from(second == "f boom 1")
    });
    let v = rt.get(d);
    let after = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(f))).unwrap_err());
    (v.to_string(), after)
}

#[test]
fn shared_set_inside_the_attempt_flips_its_held_failed_landing() {
    let local = set_after_a_caught_panic::<Local>();
    assert_eq!(local.0, "3", "{local:?}");
    assert_eq!(set_after_a_caught_panic::<Shared>(), local);
}

/// g catches a's panic, sets a's input k and re-raises, so its own set
/// flags g mid-run and g's held Failed landing publishes Dirty. r reads g
/// twice: the second read must see g as the set left it and run it again.
/// Returns what r's second read raised: 1 for a's new panic, 2 for "last
/// compute panicked", 3 for a's first.
fn reread_of_a_landing_its_own_set_flagged<C: Cells>() -> i64 {
    let rt = Runtime::<C>::new();
    let j = rt.create_input(1_i64);
    let k = rt.create_input(0_i64);
    let a = rt.create_query(move |rt: &Runtime<C>| -> i64 {
        let (v, kk) = (rt.get(j), rt.get(k));
        if v == 0 {
            panic!("a boom {kk}")
        }
        v
    });
    let g = rt.create_query(move |rt: &Runtime<C>| -> i64 {
        match catch_unwind(AssertUnwindSafe(|| rt.get(a))) {
            Ok(v) => v,
            Err(p) => {
                rt.set(k, 1);
                std::panic::resume_unwind(p)
            }
        }
    });
    let r = rt.create_query(move |rt: &Runtime<C>| -> i64 {
        let first = catch_unwind(AssertUnwindSafe(|| rt.get(g)));
        if first.is_ok() {
            return 0;
        }
        let second = panic_message(catch_unwind(AssertUnwindSafe(|| rt.get(g))).unwrap_err());
        match second.as_str() {
            "a boom 1" => 1,
            "a boom 0" => 3,
            m if m.contains("last compute panicked") => 2,
            _ => 4,
        }
    });
    assert_eq!(rt.get(r), 0);
    rt.set(j, 0);
    rt.get(r)
}

#[test]
fn shared_reread_of_a_held_landing_its_own_set_flagged_runs_it_again() {
    let local = reread_of_a_landing_its_own_set_flagged::<Local>();
    assert_eq!(reread_of_a_landing_its_own_set_flagged::<Shared>(), local);
}

/// d reads b, whose unseeded cycle with a lands Cyclic and held, then sets
/// a's input g and reads b again: the set's walk meets a's held Cyclic
/// landing through the view and must take it to New, as it would a
/// published one, not spin on the node's word. Returns d's value.
fn set_over_a_held_cycle<C: Cells>(rt: &Runtime<C>) -> i64 {
    let g = rt.create_input(0_i64);
    let bc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let bc = bc.clone();
        rt.create_query(move |rt| {
            if rt.get(g) == 0 {
                rt.try_get(*bc.get().unwrap()).map_or(-1, |v| v + 1)
            } else {
                7
            }
        })
    };
    let b = rt.create_query(move |rt| rt.try_get(a).map_or(-1, |v| v * 2));
    bc.set(b).unwrap();
    let d = rt.create_query(move |rt: &Runtime<C>| {
        let before = rt.try_get(b).unwrap_or(-100);
        rt.set(g, 1);
        before * 1000 + rt.try_get(b).unwrap_or(-100)
    });
    rt.get(d)
}

#[test]
fn shared_set_inside_the_attempt_over_its_held_cycle_finishes() {
    let local = set_over_a_held_cycle(&Runtime::<Local>::new());
    let shared = within_10s(|| set_over_a_held_cycle(&Runtime::<Shared>::new()));
    assert_eq!(shared, Some(local));
}

/// Thread A's o catches f's panic, then reads r, which reads f's held
/// Failed landing and fails with it: r's landing is held because its run
/// read a held one. While A is parked before its exit, thread B reads r,
/// budgeted and replayable, so r's Failed landing has a regeneration
/// record once published. B must wait for A's publication rather than
/// rerun r over it, and fail either way.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_reader_of_a_held_failed_landing_is_not_rerun_by_another_thread() {
    let rt = Arc::new(Runtime::<Shared>::new());
    rt.set_placeholder(0_i64);
    rt.set_max_frames(16);
    let gate = Gate::new();
    let i = rt.create_input(1_i64);
    let f = rt.create_query_with(
        move |rt: &Runtime<Shared>| -> i64 { panic!("f boom {}", rt.get(i)) },
        replayable(),
    );
    let r = rt.create_query_with(move |rt: &Runtime<Shared>| rt.get(f) + 1, replayable());
    let o = {
        let gate = gate.clone();
        rt.create_query(move |rt: &Runtime<Shared>| {
            let caught = catch_unwind(AssertUnwindSafe(|| rt.get(f))).is_err();
            let read = catch_unwind(AssertUnwindSafe(|| rt.get(r))).is_err();
            gate.park();
            i64::from(caught) + i64::from(read)
        })
    };
    gate.arm();
    let a = {
        let rt = Arc::clone(&rt);
        std::thread::spawn(move || rt.get(o))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(r), NodeState::Computing);
    let b = {
        let rt = Arc::clone(&rt);
        std::thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.get(r))).is_err())
    };
    gate.release();
    assert_eq!(a.join().unwrap(), 2);
    let b = within_10s(move || b.join().unwrap()).expect("B's read of r never finished");
    assert!(b, "B's read of r returned a value");
    assert_eq!(rt.state_of(r), NodeState::Failed);
    assert_eq!(rt.state_of(f), NodeState::Failed);
    rt.assert_invariant_i();
}

/// f ran once at i = 1, so its edge to i is published. o catches f's
/// panic at i = 2, so f's Failed landing is held with a regeneration
/// record, and parks. B's set of i flags f's word through that edge; o's
/// own set of i, once released, then flips f's held landing from the
/// Dirty its view shows, as the set's walk would flip a published one.
/// f publishes Dirty, and the record of its Failed landing goes with it.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_flipped_held_failed_landing_drops_its_record() {
    let rt = Arc::new(Runtime::<Shared>::new());
    rt.set_placeholder(0_i64);
    rt.set_max_frames(16);
    let gate = Gate::new();
    let i = rt.create_input(1_i64);
    let f = rt.create_query_with(
        move |rt: &Runtime<Shared>| -> i64 {
            let v = rt.get(i);
            assert!(v == 1, "f boom {v}");
            v
        },
        replayable(),
    );
    let o = {
        let gate = gate.clone();
        rt.create_query(move |rt: &Runtime<Shared>| {
            let caught = catch_unwind(AssertUnwindSafe(|| rt.get(f))).is_err();
            gate.park();
            rt.set(i, 3);
            i64::from(caught)
        })
    };
    assert_eq!(rt.get(f), 1);
    rt.set(i, 2);
    gate.arm();
    let a = {
        let rt = Arc::clone(&rt);
        std::thread::spawn(move || rt.get(o))
    };
    gate.wait_parked();
    assert_eq!(rt.state_of(f), NodeState::Computing);
    {
        let rt = Arc::clone(&rt);
        std::thread::spawn(move || rt.set(i, 4)).join().unwrap();
    }
    gate.release();
    assert_eq!(a.join().unwrap(), 1);
    assert_ne!(rt.state_of(f), NodeState::Failed);
    assert_eq!(rt.failed_run_token(f.slot(), f.generation()), None);
    rt.assert_invariant_i();
}

/// `d` catches `f`'s panic, so its attempt holds `f`'s Failed landing, and
/// then walks a long chain that verifies without a run. Another thread's
/// read of `f` waits on the held landing the whole walk: the walk is the
/// owner's engine progress, so the reader waits it out instead of taking
/// the landing over and retrying `d`.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_reader_of_a_held_landing_waits_out_its_owners_walk() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let x = rt.create_input(0_i64);
    let f = rt.create_query(move |rt| {
        let v = rt.get(x);
        assert!(v % 2 == 1, "even");
        v
    });
    let y = rt.create_input(0_i64);
    let mut top = rt.create_query(move |rt| rt.get(y).min(0));
    // A release walk is that much faster per node.
    let len = if cfg!(debug_assertions) {
        100_000
    } else {
        600_000
    };
    for _ in 0..len {
        let below = top;
        top = rt.create_query(move |rt| rt.get(below));
        assert_eq!(rt.get(top), 0);
    }
    rt.set(y, 1);
    let (holding, runs) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicUsize::new(0)),
    );
    let walked = Arc::new(std::sync::Mutex::new(None));
    let d = {
        let (holding, runs, walked) = (holding.clone(), runs.clone(), walked.clone());
        rt.create_query(move |rt: &Runtime<Shared>| {
            runs.fetch_add(1, Ordering::SeqCst);
            let v = catch_unwind(AssertUnwindSafe(|| rt.get(f))).map_or(7, |v| v * 10);
            holding.store(true, Ordering::SeqCst);
            let start = std::time::Instant::now();
            let w = rt.get(top);
            *walked.lock().unwrap() = Some((start, std::time::Instant::now()));
            v + w
        })
    };
    let reader = {
        let (rt, holding) = (rt.clone(), holding.clone());
        std::thread::spawn(move || {
            while !holding.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            let started = std::time::Instant::now();
            let failed = catch_unwind(AssertUnwindSafe(|| rt.get(f))).is_err();
            (failed, started, std::time::Instant::now())
        })
    };
    assert_eq!(rt.get(d), 7);
    let (failed, started, done) = reader.join().unwrap();
    assert!(failed, "the reader of f got a value");
    let (walk_start, walk_end) = walked.lock().unwrap().expect("d walked its chain");
    // Waited through well past the take-over's 10 ms STUCK_FOR of the walk,
    // or the test shows nothing.
    assert!(
        started < walk_end
            && walk_end - started.max(walk_start) >= std::time::Duration::from_millis(40),
        "the reader waited only {:?} of the walk",
        walk_end.saturating_duration_since(started.max(walk_start))
    );
    assert!(done >= walk_end, "the reader read f before the walk ended");
    assert_eq!(
        runs.load(Ordering::SeqCst),
        1,
        "the reader took the held landing over during the owner's walk"
    );
    assert_eq!(crate::hold::held_count(), 0);
}

/// `d` catches `f`'s panic, so its attempt holds `f`'s Failed landing, and
/// then reads `g`, whose rerun replaces a value whose `Drop` joins a thread
/// that reads `f`. The drop is user code, so the reader takes the held
/// landing over instead of waiting on a holder that waits on it.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_holder_whose_replaced_value_drop_joins_a_reader_finishes() {
    crate::child_test::run_in_child_within(
        "runtime::hold_tests::shared_holder_whose_replaced_value_drop_joins_a_reader_finishes",
        std::time::Duration::from_secs(60),
        || {
            static ARMED: AtomicBool = AtomicBool::new(false);
            static JOINED: AtomicBool = AtomicBool::new(false);
            static READER: OnceLock<Box<dyn Fn() -> bool + Send + Sync>> = OnceLock::new();
            #[derive(Clone, Debug)]
            struct Joins(i64);
            impl PartialEq for Joins {
                fn eq(&self, other: &Self) -> bool {
                    self.0 == other.0
                }
            }
            impl Drop for Joins {
                fn drop(&mut self) {
                    if ARMED.swap(false, Ordering::SeqCst) {
                        let read = READER.get().expect("the reader is set before arming");
                        let failed = std::thread::scope(|s| s.spawn(read).join().unwrap());
                        JOINED.store(failed, Ordering::SeqCst);
                    }
                }
            }
            let rt = Arc::new(Runtime::<Shared>::new());
            let x = rt.create_input(0_i64);
            let f = rt.create_query(move |rt| {
                let v = rt.get(x);
                assert!(v % 2 == 1, "even");
                v
            });
            let y = rt.create_input(0_i64);
            let g = rt.create_query(move |rt| Joins(rt.get(y)));
            assert_eq!(rt.get(g), Joins(0));
            rt.set(y, 1);
            let d = rt.create_query(move |rt: &Runtime<Shared>| {
                let v = catch_unwind(AssertUnwindSafe(|| rt.get(f))).map_or(7, |v| v * 10);
                v + rt.get(g).0
            });
            let reader = rt.clone();
            assert!(READER
                .set(Box::new(move || catch_unwind(AssertUnwindSafe(
                    || reader.get(f)
                ))
                .is_err()))
                .is_ok());
            ARMED.store(true, Ordering::SeqCst);
            crate::reclaim_probe::force(crate::reclaim_probe::RetireSite::ComputeWrite);
            let (tx, rx) = std::sync::mpsc::channel();
            {
                let rt = rt.clone();
                std::thread::spawn(move || tx.send(rt.get(d)).unwrap());
            }
            let got = rx.recv_timeout(std::time::Duration::from_secs(30));
            assert_eq!(
                got,
                Ok(8),
                "the holder waited on the reader its drop joined"
            );
            assert!(
                !ARMED.load(Ordering::SeqCst),
                "the replaced value never dropped"
            );
            assert!(
                JOINED.load(Ordering::SeqCst),
                "the reader of f read its Failed landing"
            );
        },
    )
    .assert_clean();
}

/// A waiter on a held landing leaves its owner be while the landing's walk
/// listing is awake, however long the owner stays in a peer's user code,
/// and the landing can be taken over once other user code ends the
/// listing. The second wait calls `hold::hand_over`, since the runtime's
/// take-over would retract the entry onto an input that no claim ran.
#[test]
#[cfg_attr(miri, ignore)]
fn a_waiter_takes_no_held_landing_over_while_its_walk_listing_is_awake() {
    crate::child_test::run_in_child(
        "runtime::hold_tests::a_waiter_takes_no_held_landing_over_while_its_walk_listing_is_awake",
        waiter_takes_no_held_landing_over_while_its_walk_listing_is_awake,
    )
    .assert_clean();
}

fn waiter_takes_no_held_landing_over_while_its_walk_listing_is_awake() {
    use crate::attempt::{Outside, Patience};
    use crate::reserve::{note_walk, user_code, Peers, Root};
    use std::time::{Duration, Instant};
    let rt = Runtime::<Shared>::new();
    let slot = rt.create_input(1i64).slot();
    let node = rt.nodes.get(slot);
    let cell = crate::state::cell_key::<Shared>(node.state_cell());
    let (generation, claim) = (node.generation(), (rt.runtime_id, slot));
    crate::attempt::begin();
    let own = crate::hold::open(cell, rt.runtime_id, slot, generation, true, None, None);
    assert!(own.is_none());
    crate::hold::land(cell, NodeState::Clean, NodeState::Clean, false);
    let read = Root::arm(true, (rt.runtime_id, slot + 1));
    note_walk(claim, generation, &Peers::new([slot, slot + 2]));
    user_code((rt.runtime_id, slot + 2));
    let user = Outside::enter(true);
    let wait = |within: Duration, take: &(dyn Fn(&mut Patience) -> bool + Sync)| {
        std::thread::scope(|s| {
            s.spawn(|| {
                let mut patience = Patience::default();
                let until = Instant::now() + within;
                // Enough checks for a waiter's patience on a loaded runner.
                for calls in 0.. {
                    if calls >= 256 && Instant::now() >= until {
                        break;
                    }
                    if take(&mut patience) {
                        return true;
                    }
                    std::thread::yield_now();
                }
                false
            })
            .join()
            .unwrap()
        })
    };
    assert!(
        !wait(Duration::from_millis(100), &|p| rt
            .hand_over(NodeId(slot), p)),
        "a waiter took a held landing over in a peer's user code"
    );
    assert!(!crate::attempt::wounded());
    user_code((rt.runtime_id, slot + 3));
    assert!(
        crate::reserve::may_take_over(claim, generation, &mut Patience::default()),
        "other user code left the refusal standing"
    );
    assert!(
        wait(Duration::from_secs(5), &|p| crate::hold::hand_over(
            cell, claim, p
        )
        .is_some()),
        "the held landing was never takeable"
    );
    drop(user);
    drop(read);
    assert!(crate::hold::take(rt.runtime_id).wounded);
}

/// What `retracted_held_read_shape` saw.
#[derive(Debug)]
struct RetractedHeldRead {
    q: i64,
    r: i64,
    g: i64,
    d: i64,
    /// The thread of each run of d: 1 reads r, 2 reads g.
    d_runs_on: Vec<u8>,
    r_real_runs: usize,
    /// r's reads of g: begun, returned.
    g_reads: (usize, usize),
    /// q's reads of d during the read of r: d's raw state and the value.
    q_reads_before: Vec<(NodeState, i64)>,
    /// q's closure runs in the final read of q; 0 means a stash was adopted.
    q_runs_in_final: usize,
    stashes_before_final: u32,
    held_after: (usize, usize),
}

/// Thread 1 reads r, which reads d, which reads the held cycle a, so d's
/// landing (a plus its run count) is held; r then reads the replayable
/// chain p, which dooms at `frames`, and on its placeholder reads q, so
/// the driver runs q speculatively over d's held landing. r's first run
/// with p's real value parks while thread 2 claims g, whose run reads d
/// and registers a wait on d's held landing; r then reads g. Thread 1's
/// registration closes the waits-for cycle as the younger waiter, so it
/// is wounded and d's landing goes back; thread 2 runs d again, and
/// thread 1's retry reads d only after that run. q is then read in the
/// same revision, after both reads published.
fn retracted_held_read_shape(frames: u32) -> RetractedHeldRead {
    let mut rt = Runtime::<Shared>::new();
    rt.all_replayable = false;
    let rt = Arc::new(rt);
    rt.set_placeholder(0_i64);
    rt.set_max_frames(frames);
    let (a, _) = two_cycle(&rt);
    let d_runs_on = Arc::new(std::sync::Mutex::new(Vec::new()));
    let d = {
        let d_runs_on = d_runs_on.clone();
        rt.create_query(move |rt: &Runtime<Shared>| {
            let v = rt.get(a);
            let mut on = d_runs_on.lock().unwrap();
            on.push(std::thread::current().id());
            v + on.len() as i64
        })
    };
    let q_reads = Arc::new(std::sync::Mutex::new(Vec::new()));
    let q = {
        let q_reads = q_reads.clone();
        rt.create_query_with(
            move |rt: &Runtime<Shared>| {
                let v = rt.get(d);
                q_reads.lock().unwrap().push((rt.state_of(d), v));
                v * 10
            },
            replayable(),
        )
    };
    let x = rt.create_input(0_i64);
    let mut p = rt.create_query_with(move |rt: &Runtime<Shared>| rt.get(x) + 1, replayable());
    for _ in 0..3 {
        let below = p;
        p = rt.create_query_with(move |rt: &Runtime<Shared>| rt.get(below) + 1, replayable());
    }
    let g = rt.create_query(move |rt: &Runtime<Shared>| rt.get(d) + 1);
    let parked = Gate::new();
    let r_real_runs = Arc::new(AtomicUsize::new(0));
    let g_reads = Arc::new((AtomicUsize::new(0), AtomicUsize::new(0)));
    let r = {
        let (parked, r_real_runs, g_reads, d_runs_on) = (
            parked.clone(),
            r_real_runs.clone(),
            g_reads.clone(),
            d_runs_on.clone(),
        );
        rt.create_query_with(
            move |rt: &Runtime<Shared>| {
                // A retry that claimed d before thread 2 would close the
                // cycle again and wound itself a second time.
                if g_reads.0.load(Ordering::SeqCst) > g_reads.1.load(Ordering::SeqCst) {
                    wait_until(|| d_runs_on.lock().unwrap().len() >= 2);
                }
                let v = rt.get(d);
                let c = rt.get(p);
                if c == 0 {
                    let _ = rt.get(q);
                } else {
                    r_real_runs.fetch_add(1, Ordering::SeqCst);
                    parked.park();
                    g_reads.0.fetch_add(1, Ordering::SeqCst);
                    let _ = rt.get(g);
                    g_reads.1.fetch_add(1, Ordering::SeqCst);
                }
                v + c
            },
            replayable(),
        )
    };
    let d_claim = (rt.runtime_id, d.erase().slot());
    parked.arm();
    let _release = super::race_tests::ReleaseOnUnwind(vec![parked.clone()]);
    let t1 = {
        let rt = rt.clone();
        std::thread::spawn(move || (rt.get(r), crate::hold::held_count()))
    };
    parked.wait_parked();
    let t2 = {
        let rt = rt.clone();
        std::thread::spawn(move || (rt.get(g), crate::hold::held_count()))
    };
    // A parked holder is never taken over, so thread 2 stays registered
    // on d until the release.
    wait_until(|| crate::waits::waited_on(d_claim));
    assert!(
        crate::waits::waited_on(d_claim),
        "frames={frames}: thread 2's run of g waits on d's held landing"
    );
    parked.release();
    let threads = [t1.thread().id(), t2.thread().id()];
    let (r_value, t1_held) = t1.join().unwrap();
    let (g_value, t2_held) = t2.join().unwrap();
    assert_eq!(crate::waits::registered_on(rt.runtime_id), 0);
    let q_reads_before = std::mem::take(&mut *q_reads.lock().unwrap());
    let stashes_before_final = Shared::u32_load_relaxed(&rt.stash_count);
    let q_value = rt.get(q);
    let q_runs_in_final = q_reads.lock().unwrap().len();
    let d_runs_on = d_runs_on
        .lock()
        .unwrap()
        .iter()
        .map(|t| {
            threads
                .iter()
                .position(|x| x == t)
                .map_or(0, |i| i as u8 + 1)
        })
        .collect();
    RetractedHeldRead {
        q: q_value,
        r: r_value,
        g: g_value,
        d: rt.get(d),
        d_runs_on,
        r_real_runs: r_real_runs.load(Ordering::SeqCst),
        g_reads: (
            g_reads.0.load(Ordering::SeqCst),
            g_reads.1.load(Ordering::SeqCst),
        ),
        q_reads_before,
        q_runs_in_final,
        stashes_before_final,
        held_after: (t1_held, t2_held),
    }
}

/// d ran on thread 1 and again on thread 2, and thread 1's first read of
/// g unwound from its wait, so r's real run ran twice.
fn assert_wounded_schedule(side: &str, s: &RetractedHeldRead) {
    assert_eq!(s.d_runs_on, [1, 2], "{side}: d's runs: {s:?}");
    assert_eq!(
        (s.r_real_runs, s.g_reads),
        (2, (2, 1)),
        "{side}: thread 1's first read of g was wounded: {s:?}"
    );
    assert_eq!((s.g, s.d), (103, 102), "{side}: {s:?}");
    assert_eq!(s.held_after, (0, 0), "{side}: {s:?}");
}

/// Worker-13's case C: q's speculative run reads d's held landing as Clean
/// through the view and is stashed; a waits-for cycle with another thread
/// wounds the holder, which retracts d, and the other thread lands d again
/// with another value; a later read of q in the same revision must give
/// what the unbounded run gives, not adopt the stash over the retracted
/// landing.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_stash_over_a_retracted_held_landing_is_not_adopted() {
    let want = retracted_held_read_shape(u32::MAX);
    assert_wounded_schedule("unbounded", &want);
    assert_eq!(
        (want.q, want.r),
        (1020, 106),
        "unbounded: d lands 101 held, is retracted and lands 102; q reads 102: {want:?}"
    );
    assert!(want.q_reads_before.is_empty(), "{want:?}");
    let got = retracted_held_read_shape(2);
    assert_wounded_schedule("bounded", &got);
    assert_eq!(
        got.q_reads_before.first(),
        Some(&(NodeState::Computing, 101)),
        "q's speculative run read d's held landing first: {got:?}"
    );
    assert_eq!(
        (got.q, got.r),
        (want.q, want.r),
        "q read its stash over d's retracted landing (q ran {} times in the final read, {} stashes before it)",
        got.q_runs_in_final,
        got.stashes_before_final
    );
}
