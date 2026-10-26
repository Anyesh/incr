//! What an attempt holds under `Shared` (Variant V), and that a lone
//! thread still reads what `Local` reads.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{Cycle, CycleStrategy, Fixpoint, QueryOpts};
use crate::handle::Incr;
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
/// resumes.
fn panic_publishing_before(rt: &Runtime<Shared>, p: Incr<i64>) {
    let armed = Arc::new(AtomicBool::new(true));
    let p_first = Arc::new(AtomicBool::new(false));
    let hook: super::ParkHook = {
        let (armed, p_first) = (armed.clone(), p_first.clone());
        Box::new(move |point| {
            let super::ParkPoint::HeldPublished(id) = point else {
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
    panic_publishing_before(&rt, p);
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

/// a and b iterate to a fixpoint from 0, a ending at t and b at a.
fn fixpoint_pair<C: Cells>(rt: &Runtime<C>, t: Incr<i64>) -> (Incr<i64>, Incr<i64>) {
    let opts = || QueryOpts {
        cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64)),
        ..QueryOpts::default()
    };
    let bc: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let a = {
        let bc = bc.clone();
        rt.create_query_with(
            move |rt| {
                let _ = rt.get(*bc.get().unwrap());
                rt.get(t)
            },
            opts(),
        )
    };
    let b = rt.create_query_with(move |rt| rt.get(a), opts());
    bc.set(b).unwrap();
    (a, b)
}

/// A held publication panics after one member of a cycle published over
/// the other, which it gives back. Once t goes back to its old value, a
/// read of either member first returns what a fresh runtime does, not the
/// published member's value from the given-back member's held one.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_held_publication_panic_rereads_over_a_given_back_landing() {
    let local = Runtime::<Local>::new();
    let t = local.create_input(1_i64);
    let (a, b) = fixpoint_pair(&local, t);
    let want = (local.get(a), local.get(b));
    for a_first in [true, false] {
        let rt = Runtime::<Shared>::new();
        let t = rt.create_input(1_i64);
        let (a, b) = fixpoint_pair(&rt, t);
        let p = rt.create_query(move |rt| rt.get(a) + rt.get(b));
        rt.get(p);
        rt.set(t, 2);
        panic_publishing_before(&rt, p);
        rt.assert_invariant_i();
        rt.set(t, 1);
        let got = if a_first {
            (rt.get(a), rt.get(b))
        } else {
            let b = rt.get(b);
            (rt.get(a), b)
        };
        assert_eq!(got, want, "a first: {a_first}");
        assert_eq!(rt.get(p), want.0 + want.1);
        rt.assert_invariant_i();
    }
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
