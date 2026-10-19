//! The plan's T8 tests at scale and across shapes: a 100k cycle, the T1
//! taint shape at 100k, short cycles at budget 1 for every strategy and
//! entry, the byte budget beating the frame budget, and dooms under
//! `stabilize`.

use std::sync::{Arc, Mutex};

use super::replay_tests::{chain, on_stack, replayable};
use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::handle::Incr;
use crate::state::NodeState;

const HEAVY: usize = 16 << 10;

/// A chain of `n` replayable queries over an input of 0, each keeping a
/// 16 KiB array live across its nested read, at a frame budget of 256 and
/// a byte budget of 64 KiB.
fn heavy_chain<C: Cells>(rt: &Runtime<C>, n: usize) -> Vec<Incr<u64>> {
    rt.set_placeholder(0_u64);
    rt.set_max_frames(256);
    rt.set_max_stack_bytes(64 << 10);
    let mut below = rt.create_input(0_u64);
    let mut nodes = Vec::with_capacity(n);
    for _ in 0..n {
        let dep = below;
        below = rt.create_query_with(
            move |rt| {
                let live = std::hint::black_box([1_u8; HEAVY]);
                let v = rt.get(dep) + 1;
                std::hint::black_box(&live);
                v
            },
            replayable(),
        );
        nodes.push(below);
    }
    nodes.reverse();
    nodes
}

fn first_doom_depth<C: Cells>(rt: &Runtime<C>) -> usize {
    rt.dooms.lock().unwrap().first().expect("a doom").depth
}

/// The byte budget dooms a 1000-chain of 16 KiB frames well below the
/// frame budget of 256, which alone would overflow a 2 MiB thread, under
/// `get` and under `stabilize` at the same depth, with the unbounded value.
fn byte_budget_first<C: Cells>() {
    let n = 1000;
    let rt = Runtime::<C>::new();
    let nodes = heavy_chain(&rt, n);
    assert_eq!(rt.get(nodes[0]), n as u64);
    let depth = first_doom_depth(&rt);
    assert!(depth <= 16, "the first doom came at depth {depth}");

    let rt = Runtime::<C>::new();
    let nodes = heavy_chain(&rt, n);
    let fired: Arc<Mutex<Vec<u64>>> = Arc::default();
    let seen = fired.clone();
    rt.observe(nodes[0], move |&v| seen.lock().unwrap().push(v));
    rt.stabilize();
    assert_eq!(*fired.lock().unwrap(), vec![n as u64]);
    assert_eq!(first_doom_depth(&rt), depth);
    assert_eq!(rt.state_of(nodes[0]), NodeState::Clean);
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_byte_budget_before_frame_budget() {
    on_stack(2 << 20, byte_budget_first::<Local>);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_byte_budget_before_frame_budget() {
    on_stack(2 << 20, byte_budget_first::<Shared>);
}

/// Two observed heads of 100-deep chains, first computed by one
/// `stabilize` at budget 1: each fires once with the unbounded value and
/// lands Clean, and the second's loop runs after the first's ends.
fn doom_under_stabilize<C: Cells>() {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let first = chain(&rt, 100);
    let second = chain(&rt, 100);
    let fired: Arc<Mutex<Vec<(usize, u64)>>> = Arc::default();
    for (i, c) in [&first, &second].into_iter().enumerate() {
        let fired = fired.clone();
        rt.observe(c.nodes[0], move |&v| fired.lock().unwrap().push((i, v)));
    }
    rt.stabilize();
    let mut fired = fired.lock().unwrap().clone();
    fired.sort_unstable();
    assert_eq!(fired, vec![(0, 100), (1, 100)]);
    for c in [&first, &second] {
        assert_eq!(rt.state_of(c.nodes[0]), NodeState::Clean);
        assert!(c.max_runs() <= 2, "a node ran {} times", c.max_runs());
    }
    let dooms = rt.dooms.lock().unwrap();
    let of = |c: &super::replay_tests::Chain| {
        let slots: Vec<u32> = c.nodes.iter().map(|n| n.slot()).collect();
        dooms
            .iter()
            .enumerate()
            .filter(|(_, d)| slots.contains(&d.node))
            .map(|(at, d)| (at, d.context))
            .collect::<Vec<_>>()
    };
    let (mut a, mut b) = (of(&first), of(&second));
    assert!(!a.is_empty() && !b.is_empty());
    if b[0].0 < a[0].0 {
        std::mem::swap(&mut a, &mut b);
    }
    assert_eq!(a.len() + b.len(), dooms.len());
    assert!(a.iter().all(|&(_, x)| x == a[0].1));
    assert!(b.iter().all(|&(_, x)| x == b[0].1));
    assert!(
        a.last().unwrap().0 < b[0].0,
        "the second head's loop began before the first's ended"
    );
    drop(dooms);
    assert!(
        crate::replay::with(|c| c.is_empty()),
        "a context is left open"
    );
    rt.assert_invariant_i();
}

#[test]
fn local_doom_under_stabilize() {
    doom_under_stabilize::<Local>();
}

#[test]
fn shared_doom_under_stabilize() {
    doom_under_stabilize::<Shared>();
}
