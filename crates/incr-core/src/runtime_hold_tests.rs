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
