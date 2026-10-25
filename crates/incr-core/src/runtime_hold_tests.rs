//! What an attempt holds under `Shared` (Variant V), and that a lone
//! thread still reads what `Local` reads.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::cycle::Cycle;
use crate::handle::Incr;

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

/// A read on one runtime whose compute reads another runtime's cycle: the
/// other runtime's episode is held under the same attempt, and the first
/// runtime's outermost exit publishes it too.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_read_publishes_the_holds_of_another_runtime_it_read() {
    let other = Arc::new(Runtime::<Shared>::new());
    let (a, b) = two_cycle(&other);
    let rt = Runtime::<Shared>::new();
    let root = {
        let other = other.clone();
        rt.create_query(move |_| other.get(a) + other.get(b))
    };
    let local = Runtime::<Local>::new();
    let (la, lb) = two_cycle(&local);
    let want = local.get(la) + local.get(lb);
    assert_eq!(rt.get(root), want);
    assert_eq!(crate::hold::held_count(), 0);
    assert_eq!(other.state_of(a), crate::state::NodeState::Clean);
    assert_eq!(other.state_of(b), crate::state::NodeState::Clean);
}
