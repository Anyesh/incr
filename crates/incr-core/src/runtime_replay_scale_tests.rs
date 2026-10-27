//! The plan's T8 tests at scale and across shapes: a 100k cycle, the T1
//! taint shape at 100k, short cycles at budget 1 for every strategy and
//! entry, the byte budget beating the frame budget, and dooms under
//! `stabilize`.

use std::panic::{catch_unwind, AssertUnwindSafe};
#[cfg(not(debug_assertions))]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
use std::sync::{Arc, Mutex, OnceLock};

use super::replay_tests::{chain, on_stack, replayable};
use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{CycleStrategy, Fixpoint, QueryOpts};
use crate::dep_stack::DepStack;
use crate::handle::Incr;
use crate::state::NodeState;

fn counters(n: usize) -> Arc<Vec<AtomicU32>> {
    Arc::new((0..n).map(|_| AtomicU32::new(0)).collect())
}

fn max_of(runs: &[AtomicU32]) -> u32 {
    runs.iter().map(|r| r.load(SeqCst)).max().unwrap_or(0)
}

fn episodes<C: Cells>(rt: &Runtime<C>) -> u64 {
    rt.episodes.load(SeqCst)
}

#[derive(Copy, Clone, Debug)]
enum Strategy {
    None,
    Fallback,
    Fixpoint,
}

/// A ring of `n` replayable members, member k reading member k + 1 (mod
/// n) through `read`. Under `Fallback` member 0 has the fallback 3; under
/// `Fixpoint` every member iterates from 0, recording the highest
/// iteration in `iterations`.
fn ring<C: Cells>(
    rt: &Runtime<C>,
    n: usize,
    strategy: Strategy,
    runs: &Arc<Vec<AtomicU32>>,
    iterations: &Arc<AtomicU32>,
    read: fn(&Runtime<C>, Incr<i64>, usize) -> i64,
) -> Vec<Incr<i64>> {
    let members: Arc<OnceLock<Vec<Incr<i64>>>> = Arc::new(OnceLock::new());
    let built = (0..n)
        .map(|k| {
            let members = members.clone();
            let runs = runs.clone();
            let cycle = match strategy {
                Strategy::Fallback if k == 0 => CycleStrategy::fallback(|_| 3),
                Strategy::Fixpoint => {
                    let iterations = iterations.clone();
                    CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64).step(move |_, out, i| {
                        iterations.fetch_max(i, SeqCst);
                        out
                    }))
                }
                _ => CycleStrategy::None,
            };
            rt.create_query_with(
                move |rt| {
                    runs[k].fetch_add(1, SeqCst);
                    read(rt, members.get().unwrap()[(k + 1) % n], k)
                },
                QueryOpts {
                    cycle,
                    ..replayable()
                },
            )
        })
        .collect::<Vec<_>>();
    members.set(built.clone()).unwrap();
    built
}

/// A 100k-member ring at budget `frames`: Fallback lands member 0's
/// fallback and every other member over it, Fixpoint converges every
/// member to 5. Every member runs at most twice per pass, the discovery
/// pass counted with the iterations.
fn cycle_100k<C: Cells>(strategy: Strategy, frames: u32) {
    let n = 100_000;
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_i64);
    rt.set_max_frames(frames);
    let runs = counters(n);
    let iterations = Arc::new(AtomicU32::new(0));
    let members = ring(&rt, n, strategy, &runs, &iterations, |rt, next, _| {
        (rt.get(next) + 1).min(5)
    });
    let head = rt.get(members[0]);
    let passes = match strategy {
        Strategy::Fallback => {
            assert_eq!(head, 3);
            for (k, &m) in members.iter().enumerate().skip(1) {
                assert_eq!(rt.get(m), if k == n - 1 { 4 } else { 5 }, "member {k}");
            }
            2
        }
        Strategy::Fixpoint => {
            assert_eq!(head, 5);
            for &m in &members {
                assert_eq!(rt.get(m), 5);
            }
            iterations.load(SeqCst) + 1
        }
        Strategy::None => unreachable!(),
    };
    for &m in &members {
        assert_eq!(rt.state_of(m), NodeState::Clean);
    }
    assert!(rt
        .dooms
        .lock()
        .unwrap()
        .iter()
        .all(|d| d.depth <= frames as usize));
    assert!(
        max_of(&runs) <= 2 * passes,
        "a member ran {} times over {passes} passes",
        max_of(&runs)
    );
    rt.assert_invariant_i();
}

macro_rules! cycle_100k_tests {
    ($($name:ident: $cells:ident, $strategy:ident, $frames:expr;)*) => {
        $(
            #[test]
            #[cfg_attr(miri, ignore)]
            fn $name() {
                // Under a TSan debug build, budget 256's Shared ring
                // levels overflow an 8 MiB stack.
                on_stack(16 << 20, || cycle_100k::<$cells>(Strategy::$strategy, $frames));
            }
        )*
    };
}

cycle_100k_tests! {
    local_fallback_cycle_100k_at_budget_256: Local, Fallback, 256;
    local_fallback_cycle_100k_at_budget_1: Local, Fallback, 1;
    local_fixpoint_cycle_100k_at_budget_256: Local, Fixpoint, 256;
    local_fixpoint_cycle_100k_at_budget_1: Local, Fixpoint, 1;
    shared_fallback_cycle_100k_at_budget_256: Shared, Fallback, 256;
    shared_fallback_cycle_100k_at_budget_1: Shared, Fallback, 1;
    shared_fixpoint_cycle_100k_at_budget_256: Shared, Fixpoint, 256;
    shared_fixpoint_cycle_100k_at_budget_1: Shared, Fixpoint, 1;
}

/// A ring's retries redeliver each discarded walk root's direct read, one
/// per doom: the delivery index work that costs stays per doom, not per
/// delivery its maker made, so it does not grow with the budget.
#[test]
#[cfg_attr(miri, ignore)]
fn redelivered_root_reads_cost_the_same_at_any_budget() {
    let work = |frames: u32| {
        let units = Arc::new(Mutex::new(0));
        let out = units.clone();
        on_stack(1 << 30, move || {
            let n = 5_000;
            let rt = Runtime::<Local>::new();
            rt.set_placeholder(0_i64);
            rt.set_max_frames(frames);
            let runs = counters(n);
            let iterations = Arc::new(AtomicU32::new(0));
            let members = ring(
                &rt,
                n,
                Strategy::Fallback,
                &runs,
                &iterations,
                |rt, next, _| (rt.get(next) + 1).min(5),
            );
            crate::replay::take_note_work();
            assert_eq!(rt.get(members[0]), 3);
            *out.lock().unwrap() = crate::replay::take_note_work();
        });
        let units = *units.lock().unwrap();
        units
    };
    let (narrow, wide) = (work(256), work(4096));
    assert!(
        wide * 4 <= narrow * 5,
        "{narrow} units of note work at budget 256, {wide} at 4096"
    );
}

thread_local! {
    static DEEPEST: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// A ring's member runs (T5's R5) stay within the frame budget: an
/// over-budget read of an unbuffered member dooms its reader and pushes
/// the member as a held entry instead of running it nested.
fn ring_member_runs_within_budget<C: Cells>(strategy: Strategy) {
    let n = 2000;
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_i64);
    rt.set_max_frames(1);
    let runs = counters(n);
    let iterations = Arc::new(AtomicU32::new(0));
    DEEPEST.with(|d| d.set(0));
    let members = ring(&rt, n, strategy, &runs, &iterations, |rt, next, _| {
        DEEPEST.with(|d| d.set(d.get().max(rt.dep_stack.depth())));
        (rt.get(next) + 1).min(5)
    });
    let head = rt.get(members[0]);
    assert_eq!(
        head,
        if matches!(strategy, Strategy::Fallback) {
            3
        } else {
            5
        }
    );
    let deepest = DEEPEST.with(std::cell::Cell::get);
    assert!(
        deepest <= 4,
        "a member ran {deepest} frames deep at budget 1"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_ring_member_runs_within_budget() {
    ring_member_runs_within_budget::<Local>(Strategy::Fallback);
    ring_member_runs_within_budget::<Local>(Strategy::Fixpoint);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_ring_member_runs_within_budget() {
    ring_member_runs_within_budget::<Shared>(Strategy::Fallback);
    ring_member_runs_within_budget::<Shared>(Strategy::Fixpoint);
}

/// T1's taint shape: c_0 … c_{n-1}, c_i reading c_{i+1}, and X, which
/// reads s and, when s = 1, c_0; c_{n-1} reads t and then X when t = 0 or
/// j when t = 1. `runs[i]` counts c_i's runs and `runs[n]` X's.
#[cfg(not(debug_assertions))]
struct Taint {
    s: Incr<i64>,
    t: Incr<i64>,
    x: Incr<i64>,
    chain: Vec<Incr<i64>>,
    runs: Arc<Vec<AtomicU32>>,
}

#[cfg(not(debug_assertions))]
fn taint_shape<C: Cells>(rt: &Runtime<C>, n: usize) -> Taint {
    let s = rt.create_input(0_i64);
    let t = rt.create_input(0_i64);
    let j = rt.create_input(20_i64);
    let runs = counters(n + 1);
    let top: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let x = {
        let top = top.clone();
        let runs = runs.clone();
        rt.create_query_with(
            move |rt| {
                runs[n].fetch_add(1, SeqCst);
                if rt.get(s) == 1 {
                    rt.get(*top.get().unwrap())
                } else {
                    10
                }
            },
            replayable(),
        )
    };
    let mut chain = vec![x; n];
    chain[n - 1] = {
        let runs = runs.clone();
        rt.create_query_with(
            move |rt| {
                runs[n - 1].fetch_add(1, SeqCst);
                if rt.get(t) == 0 {
                    rt.get(x)
                } else {
                    rt.get(j)
                }
            },
            replayable(),
        )
    };
    for i in (0..n - 1).rev() {
        let below = chain[i + 1];
        let runs = runs.clone();
        chain[i] = rt.create_query_with(
            move |rt| {
                runs[i].fetch_add(1, SeqCst);
                rt.get(below) + 1
            },
            replayable(),
        );
    }
    top.set(chain[0]).unwrap();
    Taint {
        s,
        t,
        x,
        chain,
        runs,
    }
}

/// The T1 taint shape at 100k: after the chain's first compute through
/// replay and sets that leave c_{n-1}'s edge to X stale, `get(X)` returns
/// the unbounded value with no episode, at most two runs per node and at
/// most n + B + 1 expansions at budget 256, 2n + 3 at budget 1.
#[cfg(not(debug_assertions))]
fn taint_100k<C: Cells>(frames: u32) {
    let n = 100_000;
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_i64);
    rt.set_max_frames(frames);
    let shape = taint_shape(&rt, n);
    assert_eq!(rt.get(shape.chain[0]), 10 + n as i64 - 1);
    rt.set(shape.s, 1);
    rt.set(shape.t, 1);
    for r in shape.runs.iter() {
        r.store(0, SeqCst);
    }
    let before = rt.expansions();
    let episodes_before = episodes(&rt);
    assert_eq!(rt.get(shape.x), 20 + n as i64 - 1);
    let expansions = rt.expansions() - before;
    assert_eq!(episodes(&rt), episodes_before, "the read ran an episode");
    assert!(
        max_of(&shape.runs) <= 2,
        "a node ran {} times",
        max_of(&shape.runs)
    );
    let bound = if frames == 1 {
        2 * n as u64 + 3
    } else {
        (n + frames as usize + 1) as u64
    };
    assert!(
        expansions <= bound,
        "{expansions} expansions at budget {frames}, over {bound}"
    );
    rt.assert_invariant_i();
}

#[cfg(not(debug_assertions))]
#[test]
fn local_taint_100k_at_budget_256_on_a_1_mib_thread() {
    on_stack(1 << 20, || taint_100k::<Local>(256));
}

#[cfg(not(debug_assertions))]
#[test]
fn local_taint_100k_at_budget_1_on_a_1_mib_thread() {
    on_stack(1 << 20, || taint_100k::<Local>(1));
}

#[cfg(not(debug_assertions))]
#[test]
fn shared_taint_100k_at_budget_256_on_a_1_mib_thread() {
    on_stack(1 << 20, || taint_100k::<Shared>(256));
}

#[cfg(not(debug_assertions))]
#[test]
fn shared_taint_100k_at_budget_1_on_a_1_mib_thread() {
    on_stack(1 << 20, || taint_100k::<Shared>(1));
}

/// What a read of every ring member returns after `try_get` of the
/// `entry` member, and each member's state, at budget `frames`.
type RingOutcome = (Vec<Result<i64, Vec<u32>>>, Vec<NodeState>);

fn ring_outcome<C: Cells>(
    len: usize,
    strategy: Strategy,
    entry: usize,
    frames: u32,
) -> RingOutcome {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_i64);
    rt.set_max_frames(frames);
    let members = ring(
        &rt,
        len,
        strategy,
        &counters(len),
        &Arc::new(AtomicU32::new(0)),
        |rt, next, k| rt.try_get(next).map_or(-1, |v| (v + 1).min(10 + k as i64)),
    );
    let slots = |r: Result<i64, crate::cycle::Cycle>| {
        r.map_err(|c| c.participants().iter().map(|p| p.slot()).collect())
    };
    let mut values = vec![slots(rt.try_get(members[entry]))];
    values.extend(members.iter().map(|&m| slots(rt.try_get(m))));
    let states = members.iter().map(|&m| rt.state_of(m)).collect();
    if frames == 1 {
        assert!(
            !rt.dooms.lock().unwrap().is_empty(),
            "budget 1 never doomed"
        );
    }
    rt.assert_invariant_i();
    (values, states)
}

/// Rings of 3 to 10 members (3 and 4 under Miri) at budget 1 land as the
/// unbounded run does, for every strategy and every entry member.
fn short_cycles_at_budget_1<C: Cells>() {
    let longest = if cfg!(miri) { 4 } else { 10 };
    for len in 3..=longest {
        for strategy in [Strategy::None, Strategy::Fallback, Strategy::Fixpoint] {
            for entry in 0..len {
                assert_eq!(
                    ring_outcome::<C>(len, strategy, entry, 1),
                    ring_outcome::<C>(len, strategy, entry, u32::MAX),
                    "a {len}-ring under {strategy:?} entered at member {entry}"
                );
            }
        }
    }
}

#[test]
fn local_short_cycles_at_budget_1() {
    short_cycles_at_budget_1::<Local>();
}

#[test]
fn shared_short_cycles_at_budget_1() {
    short_cycles_at_budget_1::<Shared>();
}

/// F = IF(A, B, C) with C = F + 1 and A's placeholder false: at budget 1
/// the doomed F reads false from A, yet `get(F)` runs no episode and C at
/// most once.
fn if_abc_at_budget_1<C: Cells>() {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_i64);
    rt.set_placeholder(false);
    rt.set_max_frames(1);
    let a_in = rt.create_input(true);
    let b_in = rt.create_input(7_i64);
    let a = rt.create_query_with(move |rt| rt.get(a_in), replayable());
    let b = rt.create_query_with(move |rt| rt.get(b_in), replayable());
    let f_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
    let c_runs = Arc::new(AtomicU32::new(0));
    let c = {
        let f_cell = f_cell.clone();
        let c_runs = c_runs.clone();
        rt.create_query_with(
            move |rt| {
                c_runs.fetch_add(1, SeqCst);
                rt.get(*f_cell.get().unwrap()) + 1
            },
            replayable(),
        )
    };
    let f = rt.create_query_with(
        move |rt| if rt.get(a) { rt.get(b) } else { rt.get(c) },
        replayable(),
    );
    f_cell.set(f).unwrap();
    assert_eq!(rt.get(f), 7);
    assert!(
        !rt.dooms.lock().unwrap().is_empty(),
        "the read never doomed"
    );
    assert_eq!(episodes(&rt), 0, "the read ran an episode");
    assert!(
        c_runs.load(SeqCst) <= 1,
        "C ran {} times",
        c_runs.load(SeqCst)
    );
    rt.assert_invariant_i();
}

#[test]
fn local_if_abc_at_budget_1() {
    if_abc_at_budget_1::<Local>();
}

#[test]
fn shared_if_abc_at_budget_1() {
    if_abc_at_budget_1::<Shared>();
}

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
/// `get` and at most one level earlier under `stabilize`, with the
/// unbounded value.
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
    // The budget counts from the outermost entry, so stabilize's observer
    // frames come before the chain's; under TSan, Local's first level under
    // get ends less than 1 KiB inside the 64 KiB, so stabilize dooms one
    // level earlier.
    let stabilized = first_doom_depth(&rt);
    assert!(
        stabilized <= depth && depth - stabilized <= 1,
        "the doom came at depth {stabilized} under stabilize, {depth} under get"
    );
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

/// A retry diverted for the unwind of a read reads 100k Clean nodes first:
/// each diverted read finds no delivery for its node without counting the
/// frame's earlier reads, so the retry stays linear. The shape is
/// `caught_delivery_after_its_node_lands`, whose T reads the nodes first.
#[cfg(not(debug_assertions))]
fn wide_diverted_retry<C: Cells>() {
    let n = 100_000;
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let leaves: Arc<Vec<Incr<u64>>> =
        Arc::new((0..n as u64).map(|i| rt.create_query(move |_| i)).collect());
    for &leaf in leaves.iter() {
        rt.get(leaf);
    }
    let s_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let s_for_p = s_cell.clone();
    let p = rt.create_query_with(
        move |rt| rt.get(*s_for_p.get().unwrap()),
        QueryOpts {
            cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 3).max_iterations(1)),
            replay: crate::cycle::Replay::Allowed,
        },
    );
    let s = rt.create_query_with(move |rt| rt.get(p), replayable());
    s_cell.set(s).unwrap();
    let diverted = Arc::new(AtomicBool::new(false));
    let seen = diverted.clone();
    let t = rt.create_query_with(
        move |rt| {
            if rt.dep_stack.diverts_clean_reads(rt.runtime_id) {
                seen.store(true, SeqCst);
            }
            let sum = leaves.iter().map(|&leaf| rt.get(leaf)).sum::<u64>();
            let _ = catch_unwind(AssertUnwindSafe(|| rt.get(s)));
            let _ = rt.get(p);
            sum
        },
        replayable(),
    );
    let _ = catch_unwind(AssertUnwindSafe(|| rt.get(t)));
    assert!(diverted.load(SeqCst), "the retry was not diverted");
}

#[cfg(not(debug_assertions))]
#[test]
fn local_wide_diverted_retry() {
    wide_diverted_retry::<Local>();
}

#[cfg(not(debug_assertions))]
#[test]
fn shared_wide_diverted_retry() {
    wide_diverted_retry::<Shared>();
}

/// Sizes for the Failed-walk scaling tests. Main b9f0d0bf walks n + 1
/// entries for the chain and k + 3 for the fan-in; a walk that rewalks the
/// Failed node's deps per reader grows as n^2 / 2.
const FAILED_WALK_SIZES: &[usize] = if cfg!(miri) {
    &[4, 8]
} else {
    &[250, 500, 1000]
};

/// A chain of `n` queries over a Failed node, rerun after a set below the
/// chain: each rerun's read of the Failed node below it walks only that
/// node, as a walk entry Failed computes to nothing and a Failed root
/// raises, so the walks stay linear.
fn failed_entry_chain<C: Cells>(frames: Option<u32>) {
    for &n in FAILED_WALK_SIZES {
        failed_entry_chain_of::<C>(n, frames);
    }
}

fn failed_entry_chain_of<C: Cells>(n: usize, frames: Option<u32>) {
    let rt = Runtime::<C>::new();
    let opts = || match frames {
        Some(_) => replayable(),
        None => QueryOpts::default(),
    };
    if let Some(frames) = frames {
        rt.set_placeholder(0_u64);
        rt.set_max_frames(frames);
    }
    let failed = rt.create_query_with(|_| -> u64 { panic!("failed below the chain") }, opts());
    let input = rt.create_input(0_u64);
    let runs = counters(1);
    let mut below = {
        let runs = runs.clone();
        rt.create_query_with(
            move |rt| {
                runs[0].fetch_add(1, SeqCst);
                rt.get(input) + rt.get(failed)
            },
            opts(),
        )
    };
    for _ in 1..n {
        let (runs, dep) = (runs.clone(), below);
        below = rt.create_query_with(
            move |rt| {
                runs[0].fetch_add(1, SeqCst);
                rt.get(dep) + 1
            },
            opts(),
        );
    }
    let top = below;
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(top))).is_err());
    rt.set(input, 1);
    runs[0].store(0, SeqCst);
    let pops = rt.walk_pops.load(SeqCst);
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(top))).is_err());
    let pops = rt.walk_pops.load(SeqCst) - pops;
    assert_eq!(runs[0].load(SeqCst), n as u32);
    assert!(pops <= 2 * n as u64 + 2, "{pops} walk pops for a {n}-chain");
}

#[test]
fn local_failed_entry_chain() {
    on_stack(1 << 28, || failed_entry_chain::<Local>(None));
}

#[test]
fn shared_failed_entry_chain() {
    on_stack(1 << 28, || failed_entry_chain::<Shared>(None));
}

#[test]
fn local_failed_entry_chain_at_budget_1() {
    on_stack(1 << 28, || failed_entry_chain::<Local>(Some(1)));
}

#[test]
fn shared_failed_entry_chain_at_budget_1() {
    on_stack(1 << 28, || failed_entry_chain::<Shared>(Some(1)));
}

#[derive(Copy, Clone, Debug)]
enum Caught {
    /// Each query panics with the same text.
    Same,
    /// Each query panics with its own text, so no two payloads match.
    Distinct,
    /// The reader catches a query that reads the panicking one, so each
    /// panic lands at two positions.
    Nested,
    /// The reader also reads a query that catches a panic of its own and
    /// returns, whose pop retires that panic's note among the reader's.
    Mixed,
}

/// `n` replayable queries that panic, each read and caught by one
/// replayable reader at budget 1. With no placeholder the reads take the
/// native path. Each caught panic's landing stays noted while the reader
/// runs, and a payload reuses the last one's address, so under `Same` the
/// copy a read keeps matches every earlier note: the work on notes, their
/// copies and the kept reads stays linear only if copies share the notes.
fn many_caught_unwinds<C: Cells>(shape: Caught) {
    let n = 2000;
    let rt = Runtime::<C>::new();
    rt.set_max_frames(1);
    let failing = |rt: &Runtime<C>, i: usize| match shape {
        Caught::Distinct => {
            rt.create_query_with(move |_| -> u64 { panic!("boom {i}") }, replayable())
        }
        _ => rt.create_query_with(|_| -> u64 { panic!("boom") }, replayable()),
    };
    let read: Vec<(Incr<u64>, Option<Incr<u64>>)> = (0..n)
        .map(|i| {
            let x = failing(&rt, i);
            match shape {
                Caught::Nested => (
                    rt.create_query_with(move |rt| rt.get(x), replayable()),
                    None,
                ),
                Caught::Mixed => {
                    let inner = failing(&rt, i);
                    let ok = rt.create_query_with(
                        move |rt| {
                            u64::from(catch_unwind(AssertUnwindSafe(|| rt.get(inner))).is_err())
                        },
                        replayable(),
                    );
                    (x, Some(ok))
                }
                _ => (x, None),
            }
        })
        .collect();
    let reader = rt.create_query_with(
        move |rt| {
            read.iter()
                .map(|&(x, ok)| {
                    let caught = u64::from(catch_unwind(AssertUnwindSafe(|| rt.get(x))).is_err());
                    caught + ok.map_or(0, |ok| rt.get(ok))
                })
                .sum::<u64>()
        },
        replayable(),
    );
    crate::replay::take_note_work();
    let caught = match shape {
        Caught::Mixed => 2 * n,
        _ => n,
    };
    assert_eq!(rt.get(reader), caught as u64);
    let work = crate::replay::take_note_work();
    assert!(
        work <= 16 * caught as u64,
        "{shape:?}: {work} units of note work for {caught} caught unwinds"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_many_caught_unwinds_stay_linear() {
    for shape in [
        Caught::Same,
        Caught::Distinct,
        Caught::Nested,
        Caught::Mixed,
    ] {
        many_caught_unwinds::<Local>(shape);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_many_caught_unwinds_stay_linear() {
    for shape in [
        Caught::Same,
        Caught::Distinct,
        Caught::Nested,
        Caught::Mixed,
    ] {
        many_caught_unwinds::<Shared>(shape);
    }
}

/// One budgeted frame reading `n` nodes of another runtime, each twice,
/// all of them Clean when `clean`. Each read is kept for the frame's
/// retry, under its ordinal among the frame's reads of that node, and
/// finding that ordinal must not cost a pass over the frame's earlier
/// reads.
fn many_foreign_reads<C: Cells>(clean: bool) {
    let n = 2000;
    let foreign = Arc::new(Runtime::<Shared>::new());
    let reads: Vec<Incr<u64>> = (0..n).map(|i| foreign.create_query(move |_| i)).collect();
    if clean {
        for &x in &reads {
            foreign.get(x);
        }
    }
    let rt = Runtime::<C>::new();
    rt.set_max_frames(256);
    let reader = rt.create_query_with(
        move |_| {
            reads
                .iter()
                .chain(reads.iter())
                .map(|&x| foreign.get(x))
                .sum::<u64>()
        },
        replayable(),
    );
    crate::replay::take_note_work();
    crate::dep_stack::take_keep_settles();
    assert_eq!(rt.get(reader), n * (n - 1));
    let work = crate::replay::take_note_work();
    assert!(
        work <= 32 * n,
        "{work} units of note work for {} foreign reads",
        2 * n
    );
    // Each read after the first counts its ordinal among the kept ones.
    assert!(
        work >= 2 * n - 1,
        "{work} units of note work: not every foreign read was kept"
    );
    assert_eq!(
        crate::dep_stack::take_keep_settles(),
        1,
        "settles of one frame"
    );
}

/// An unbudgeted runtime's frame reading `n` Clean nodes of another
/// runtime settles once that it keeps none of them, and keeps none.
fn unbudgeted_foreign_clean_reads<C: Cells>() {
    let n = 1000;
    let foreign = Arc::new(Runtime::<Shared>::new());
    let reads: Vec<Incr<u64>> = (0..n).map(|i| foreign.create_query(move |_| i)).collect();
    for &x in &reads {
        foreign.get(x);
    }
    let rt = Runtime::<C>::new();
    rt.set_max_frames(u32::MAX);
    let reader = rt.create_query_with(
        move |_| reads.iter().map(|&x| foreign.get(x)).sum::<u64>(),
        replayable(),
    );
    crate::replay::take_note_work();
    crate::dep_stack::take_keep_settles();
    assert_eq!(rt.get(reader), n * (n - 1) / 2);
    assert_eq!(
        crate::dep_stack::take_keep_settles(),
        1,
        "settles of one frame"
    );
    assert_eq!(crate::replay::take_note_work(), 0, "kept-read work");
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_unbudgeted_foreign_clean_reads_settle_once() {
    unbudgeted_foreign_clean_reads::<Local>();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_unbudgeted_foreign_clean_reads_settle_once() {
    unbudgeted_foreign_clean_reads::<Shared>();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_many_foreign_reads_stay_linear() {
    many_foreign_reads::<Local>(false);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_many_foreign_reads_stay_linear() {
    many_foreign_reads::<Shared>(false);
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_many_foreign_clean_reads_stay_linear() {
    many_foreign_reads::<Local>(true);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_many_foreign_clean_reads_stay_linear() {
    many_foreign_reads::<Shared>(true);
}

/// The note work of the first read of a chain of `n` queries over a
/// panicking node at budget `frames`, and of the second read after a set
/// below the chain. One payload goes up the chain, and its ticket gains a
/// landing per step: each reader takes the ticket and the next landing
/// holds it again.
fn contained_chain<C: Cells>(frames: Option<u32>, n: usize) -> (u64, u64) {
    let rt = Runtime::<C>::new();
    let opts = || match frames {
        Some(_) => replayable(),
        None => QueryOpts::default(),
    };
    if let Some(frames) = frames {
        rt.set_placeholder(0_u64);
        rt.set_max_frames(frames);
    }
    let failed = rt.create_query_with(|_| -> u64 { panic!("failed below the chain") }, opts());
    let input = rt.create_input(0_u64);
    let mut below = rt.create_query_with(move |rt| rt.get(input) + rt.get(failed), opts());
    for _ in 1..n {
        let dep = below;
        below = rt.create_query_with(move |rt| rt.get(dep) + 1, opts());
    }
    let top = below;
    crate::replay::take_note_work();
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(top))).is_err());
    let first = crate::replay::take_note_work();
    rt.set(input, 1);
    crate::replay::take_note_work();
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(top))).is_err());
    (first, crate::replay::take_note_work())
}

fn contained_chain_stays_linear<C: Cells>(frames: Option<u32>) {
    let sizes: &[usize] = if cfg!(miri) {
        &[8]
    } else {
        &[1000, 2000, 4000]
    };
    for &n in sizes {
        let (first, work) = contained_chain::<C>(frames, n);
        assert!(
            first <= 24 * n as u64,
            "{frames:?}: {first} units of note work for a {n}-chain's first read"
        );
        assert!(
            work <= 16 * n as u64,
            "{frames:?}: {work} units of note work for a {n}-chain"
        );
    }
}

#[test]
fn local_contained_chain_stays_linear() {
    on_stack(1 << 28, || contained_chain_stays_linear::<Local>(None));
}

#[test]
fn shared_contained_chain_stays_linear() {
    on_stack(1 << 28, || contained_chain_stays_linear::<Shared>(None));
}

#[test]
fn local_contained_chain_stays_linear_at_budget_1() {
    on_stack(1 << 28, || contained_chain_stays_linear::<Local>(Some(1)));
}

#[test]
fn shared_contained_chain_stays_linear_at_budget_1() {
    on_stack(1 << 28, || contained_chain_stays_linear::<Shared>(Some(1)));
}

#[test]
fn local_contained_chain_stays_linear_at_budget_2() {
    on_stack(1 << 28, || contained_chain_stays_linear::<Local>(Some(2)));
}

#[test]
fn shared_contained_chain_stays_linear_at_budget_2() {
    on_stack(1 << 28, || contained_chain_stays_linear::<Shared>(Some(2)));
}

#[test]
fn local_contained_chain_stays_linear_at_budget_16() {
    on_stack(1 << 28, || contained_chain_stays_linear::<Local>(Some(16)));
}

#[test]
fn shared_contained_chain_stays_linear_at_budget_16() {
    on_stack(1 << 28, || contained_chain_stays_linear::<Shared>(Some(16)));
}

/// A panicking query under an `n`-deep chain, read and then read again
/// after an input change: each level's read takes the unwind and its frame
/// lands Failed, which keys what the read took by the frame's path. The
/// stack frames folded into paths on the first read and on the second, and
/// the entries the second read's walks pushed.
fn panicking_chain_path_steps<C: Cells>(frames: Option<u32>, n: usize) -> (u64, u64, u64) {
    let rt = Runtime::<C>::new();
    let opts = || match frames {
        Some(_) => replayable(),
        None => QueryOpts::default(),
    };
    if let Some(frames) = frames {
        rt.set_placeholder(0_u64);
        rt.set_max_frames(frames);
    }
    let failed = rt.create_query_with(|_| -> u64 { panic!("failed below the chain") }, opts());
    let input = rt.create_input(0_u64);
    let mut below = rt.create_query_with(move |rt| rt.get(input) + rt.get(failed), opts());
    for _ in 1..n {
        let dep = below;
        below = rt.create_query_with(move |rt| rt.get(dep) + 1, opts());
    }
    let top = below;
    crate::replay::take_path_steps();
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(top))).is_err());
    let first = crate::replay::take_path_steps();
    rt.set(input, 1);
    crate::runtime::take_walk_pushes();
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(top))).is_err());
    (
        first,
        crate::replay::take_path_steps(),
        crate::runtime::take_walk_pushes(),
    )
}

fn panicking_chain_paths_stay_linear<C: Cells>(frames: Option<u32>) {
    let n = if cfg!(miri) { 8 } else { 4000 };
    let (first, second, pushes) = panicking_chain_path_steps::<C>(frames, n);
    assert!(
        first <= 8 * n as u64 && second <= 8 * n as u64,
        "{frames:?}: {first} and {second} frames folded into paths for a {n}-chain"
    );
    assert!(
        pushes <= 8 * n as u64,
        "{frames:?}: the second read's walks pushed {pushes} entries for a {n}-chain"
    );
}

#[test]
fn local_panicking_chain_paths_stay_linear() {
    on_stack(1 << 28, || panicking_chain_paths_stay_linear::<Local>(None));
}

#[test]
fn shared_panicking_chain_paths_stay_linear() {
    on_stack(1 << 28, || {
        panicking_chain_paths_stay_linear::<Shared>(None)
    });
}

#[test]
fn local_panicking_chain_paths_stay_linear_at_budget_1() {
    on_stack(1 << 28, || {
        panicking_chain_paths_stay_linear::<Local>(Some(1))
    });
}

#[test]
fn shared_panicking_chain_paths_stay_linear_at_budget_1() {
    on_stack(1 << 28, || {
        panicking_chain_paths_stay_linear::<Shared>(Some(1))
    });
}

/// `k` Dirty readers over one Failed node with a Failed chain of `k` below
/// it, whose edges a Clean run recorded, all entries of one walk: the walk
/// passes the Failed node without walking the chain once per reader.
fn failed_entry_fan_in<C: Cells>() {
    for &k in FAILED_WALK_SIZES {
        failed_entry_fan_in_of::<C>(k);
    }
}

fn failed_entry_fan_in_of<C: Cells>(k: usize) {
    let rt = Runtime::<C>::new();
    let fail = rt.create_input(0_u64);
    let mut failed = rt.create_query(move |rt| {
        assert_eq!(rt.get(fail), 0, "failed below the fan");
        0
    });
    for _ in 0..k {
        let below = failed;
        failed = rt.create_query(move |rt| rt.get(below) + 1);
    }
    let input = rt.create_input(0_u64);
    let readers: Arc<Vec<Incr<u64>>> = Arc::new(
        (0..k)
            .map(|_| rt.create_query(move |rt| rt.get(input) + rt.get(failed)))
            .collect(),
    );
    let top = {
        let readers = readers.clone();
        rt.create_query(move |rt| {
            for &r in readers.iter() {
                let _ = catch_unwind(AssertUnwindSafe(|| rt.get(r)));
            }
            0
        })
    };
    rt.get(top);
    rt.set(fail, 1);
    let _ = catch_unwind(AssertUnwindSafe(|| rt.get(top)));
    assert_eq!(rt.state_of(failed), NodeState::Failed);
    rt.set(input, 1);
    let pops = rt.walk_pops.load(SeqCst);
    let _ = catch_unwind(AssertUnwindSafe(|| rt.get(top)));
    let pops = rt.walk_pops.load(SeqCst) - pops;
    assert!(pops <= 4 * k as u64 + 2, "{pops} walk pops for {k} readers");
}

#[test]
fn local_failed_entry_fan_in() {
    on_stack(1 << 28, failed_entry_fan_in::<Local>);
}

#[test]
fn shared_failed_entry_fan_in() {
    on_stack(1 << 28, failed_entry_fan_in::<Shared>);
}

#[cfg(not(debug_assertions))]
#[derive(Debug)]
struct ChainPayload(u64);

#[cfg(not(debug_assertions))]
thread_local! {
    /// The payloads the chain's readers caught, kept so that no payload's
    /// box is freed for the next to take its address: the engine keys a
    /// payload by its address and type, so every node's would share one
    /// key, a collision B7-reused-address-payload-content tracks.
    static CAUGHT: std::cell::RefCell<Vec<Box<dyn std::any::Any + Send>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Silences the panics of [`ChainPayload`] payloads, one per chain node,
/// and leaves every other panic to the default hook.
#[cfg(not(debug_assertions))]
fn quiet_chain_payload() {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !info.payload().is::<ChainPayload>() {
                default(info);
            }
        }));
    });
}

/// A 100k chain whose every level catches the custom panic payload of its
/// own Failed dep's ticket before it reads the level below, at budget
/// `frames`: each catch sees the payload, as in the unbounded run, a level
/// at the budget included, whose retries regenerate it, and the stack stays
/// within the thread's. The deps' panics are contained by a walk first, as
/// in `ticket_catching_chain`, because a chain whose levels compute their
/// panicking deps is quadratic under a budget whatever the payload
/// (T8-nested-caught-chain-quadratic).
#[cfg(not(debug_assertions))]
fn custom_payload_chain_100k<C: Cells>(frames: u32) {
    quiet_chain_payload();
    let n = crate::test_size::test_size(100_000) as u64;
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    let j = rt.create_input(1_u64);
    let sel = rt.create_input(0_u64);
    let ds: Vec<Incr<u64>> = (0..n)
        .map(|k| {
            rt.create_query_with(
                move |rt: &Runtime<C>| -> u64 {
                    if rt.get(j) == 0 {
                        std::panic::panic_any(ChainPayload(k));
                    }
                    0
                },
                replayable(),
            )
        })
        .collect();
    let all = ds.clone();
    let x = rt.create_query(move |rt: &Runtime<C>| {
        if rt.get(sel) == 0 {
            all.iter().map(|&d| rt.get(d)).sum()
        } else {
            0
        }
    });
    rt.get(x);
    rt.set(j, 0);
    rt.set(sel, 1);
    assert_eq!(rt.get(x), 0);
    assert_eq!(rt.contained_payloads(), n as usize);
    rt.set_max_frames(frames);
    let mut next = rt.create_query_with(move |rt| rt.get(sel) + 6, replayable());
    for (k, &d) in (0..n).zip(&ds) {
        let below = next;
        next = rt.create_query_with(
            move |rt| {
                let saw = match catch_unwind(AssertUnwindSafe(|| rt.get(d))) {
                    Err(p) => {
                        let saw = p.downcast_ref::<ChainPayload>().is_some_and(|c| c.0 == k);
                        CAUGHT.with(|c| c.borrow_mut().push(p));
                        u64::from(saw)
                    }
                    Ok(_) => 0,
                };
                saw + rt.get(below)
            },
            replayable(),
        );
    }
    assert_eq!(rt.get(next), n + 7);
}

#[cfg(not(debug_assertions))]
#[test]
fn local_custom_payload_chain_100k_at_budget_256_on_a_1_mib_thread() {
    on_stack(1 << 20, || custom_payload_chain_100k::<Local>(256));
}

#[cfg(not(debug_assertions))]
#[test]
fn local_custom_payload_chain_100k_at_budget_1_on_a_1_mib_thread() {
    on_stack(1 << 20, || custom_payload_chain_100k::<Local>(1));
}

#[cfg(not(debug_assertions))]
#[test]
fn shared_custom_payload_chain_100k_at_budget_256_on_a_1_mib_thread() {
    on_stack(1 << 20, || custom_payload_chain_100k::<Shared>(256));
}

#[cfg(not(debug_assertions))]
#[test]
fn shared_custom_payload_chain_100k_at_budget_1_on_a_1_mib_thread() {
    on_stack(1 << 20, || custom_payload_chain_100k::<Shared>(1));
}

/// A chain of `n` replayable queries at budget `frames`, where each level
/// reads and catches its own panicking query and then reads the level
/// below. Every caught panic's Failed landing is copied into a delivery,
/// and a later landing of a node no copy holds must not cost a pass over
/// the context's deliveries. `through` puts a query between each level and
/// its panicking one, so each copy holds a landing a level deeper.
fn nested_caught_chain<C: Cells>(frames: u32, through: bool) {
    // Each doomed level's retry catches its kept reads up on what up to
    // `frames` levels above it did, so the work per level grows with the
    // budget, never with the chain.
    let per_level = match (through, frames) {
        (false, _) => 48,
        (true, 1) => 72,
        (true, _) => 240,
    };
    let n = if cfg!(miri) { 8 } else { 4000 };
    let work = nested_caught_chain_work::<C>(frames, through, n);
    assert!(
        work <= per_level * n,
        "budget {frames}: {work} units of note work for a {n}-chain"
    );
    if through && !cfg!(miri) {
        let small = nested_caught_chain_work::<C>(frames, through, 1000);
        assert!(
            work <= 6 * small,
            "budget {frames}: {small} and {work} units of note work for 1000- and 4000-chains"
        );
    }
}

/// The note work of a chain of `n` levels as [`nested_caught_chain`] builds it.
fn nested_caught_chain_work<C: Cells>(frames: u32, through: bool, n: u64) -> u64 {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(frames);
    let input = rt.create_input(7_u64);
    let mut below = rt.create_query_with(move |rt| rt.get(input), replayable());
    for k in 0..n {
        let mut failing = rt.create_query_with(
            move |rt: &Runtime<C>| -> u64 { panic!("chain {}", k + rt.get(input)) },
            replayable(),
        );
        if through {
            let panicking = failing;
            failing = rt.create_query_with(move |rt| rt.get(panicking), replayable());
        }
        let dep = below;
        below = rt.create_query_with(
            move |rt| {
                let caught = catch_unwind(AssertUnwindSafe(|| rt.get(failing))).is_err();
                u64::from(caught) + rt.get(dep)
            },
            replayable(),
        );
    }
    crate::replay::take_note_work();
    assert_eq!(rt.get(below), n + 7);
    crate::replay::take_note_work()
}

#[test]
fn local_nested_caught_chain_stays_linear() {
    on_stack(1 << 26, || nested_caught_chain::<Local>(256, false));
}

#[test]
fn shared_nested_caught_chain_stays_linear() {
    on_stack(1 << 26, || nested_caught_chain::<Shared>(256, false));
}

#[test]
fn local_nested_caught_chain_through_a_query_stays_linear() {
    on_stack(1 << 26, || nested_caught_chain::<Local>(256, true));
}

#[test]
fn shared_nested_caught_chain_through_a_query_stays_linear() {
    on_stack(1 << 26, || nested_caught_chain::<Shared>(256, true));
}

#[test]
fn local_nested_caught_chain_stays_linear_at_budget_1() {
    on_stack(1 << 26, || nested_caught_chain::<Local>(1, false));
}

#[test]
fn shared_nested_caught_chain_stays_linear_at_budget_1() {
    on_stack(1 << 26, || nested_caught_chain::<Shared>(1, false));
}

#[test]
fn local_nested_caught_chain_through_a_query_stays_linear_at_budget_1() {
    on_stack(1 << 26, || nested_caught_chain::<Local>(1, true));
}

#[test]
fn shared_nested_caught_chain_through_a_query_stays_linear_at_budget_1() {
    on_stack(1 << 26, || nested_caught_chain::<Shared>(1, true));
}

/// One replayable reader catches `n / 2` panics and then reads a chain
/// `n / 2` deep, which the budget dooms: the caught reads kept for the
/// reader's retry must not each cost a pass over the chain's landings.
fn caught_fan_over_deep_chain<C: Cells>(frames: u32) {
    let n: u64 = if cfg!(miri) { 8 } else { 4000 };
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(frames);
    let input = rt.create_input(7_u64);
    let failing: Vec<Incr<u64>> = (0..n / 2)
        .map(|k| {
            rt.create_query_with(
                move |rt: &Runtime<C>| -> u64 { panic!("fan {}", k + rt.get(input)) },
                replayable(),
            )
        })
        .collect();
    let mut top = rt.create_query_with(move |rt| rt.get(input), replayable());
    for _ in 0..n / 2 {
        let below = top;
        top = rt.create_query_with(move |rt| rt.get(below) + 1, replayable());
    }
    let reader = rt.create_query_with(
        move |rt| {
            let caught = failing
                .iter()
                .filter(|&&f| catch_unwind(AssertUnwindSafe(|| rt.get(f))).is_err())
                .count() as u64;
            caught + rt.get(top)
        },
        replayable(),
    );
    crate::replay::take_note_work();
    assert_eq!(rt.get(reader), n + 7);
    let work = crate::replay::take_note_work();
    assert!(
        work <= 24 * n,
        "budget {frames}: {work} units of note work for a {n}-node caught fan over a chain"
    );
}

#[test]
fn local_caught_fan_over_deep_chain_stays_linear() {
    on_stack(1 << 26, || caught_fan_over_deep_chain::<Local>(256));
}

#[test]
fn shared_caught_fan_over_deep_chain_stays_linear() {
    on_stack(1 << 26, || caught_fan_over_deep_chain::<Shared>(256));
}

/// The note work of the first read of a replayed query 300 hops above a
/// parent that reads `n` New leaf queries, at the default budget: each
/// leaf's run under the open context takes its path from the parent's
/// step, which needs the parent's read count of that leaf. Note work counts
/// only a frame's read-count lookups past `SCAN_DEPS` deps, since one at or
/// under it costs at most that constant.
fn fan_out_under_replayed_chain<C: Cells>(n: u64) -> u64 {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    let leaves: Vec<Incr<u64>> = (0..n)
        .map(|i| rt.create_query_with(move |_| i, replayable()))
        .collect();
    let mut top = rt.create_query_with(
        move |rt| leaves.iter().map(|&l| rt.get(l)).sum::<u64>(),
        replayable(),
    );
    for _ in 0..300 {
        let below = top;
        top = rt.create_query_with(move |rt| rt.get(below), replayable());
    }
    crate::replay::take_note_work();
    assert_eq!(rt.get(top), n * (n - 1) / 2);
    crate::replay::take_note_work()
}

fn fan_out_under_replayed_chain_stays_linear<C: Cells>() {
    let n = crate::test_size::test_size(8_000) as u64;
    let half = fan_out_under_replayed_chain::<C>(n / 2);
    let full = fan_out_under_replayed_chain::<C>(n);
    assert!(
        full <= 2 * half + half / 4 && full <= 64 * n,
        "{half} units of note work at {} leaves, {full} at {n}",
        n / 2
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_fan_out_under_replayed_chain_stays_linear() {
    on_stack(1 << 26, fan_out_under_replayed_chain_stays_linear::<Local>);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_fan_out_under_replayed_chain_stays_linear() {
    on_stack(1 << 26, fan_out_under_replayed_chain_stays_linear::<Shared>);
}

/// One budgeted frame reading `n` nodes of another runtime and then a
/// 600-deep chain, which dooms it: the reads are kept as deliveries for
/// its retry, sharing the frame's, maker's and node's lists, and removing
/// them must not cost a pass over those lists each.
fn wide_frame_deliveries<C: Cells>() {
    let n = 4000;
    let foreign = Arc::new(Runtime::<Shared>::new());
    let reads: Vec<Incr<u64>> = (0..n).map(|i| foreign.create_query(move |_| i)).collect();
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(256);
    let top = chain(&rt, 600).nodes[0];
    let reader = rt.create_query_with(
        move |rt| reads.iter().map(|&x| foreign.get(x)).sum::<u64>() + rt.get(top),
        replayable(),
    );
    crate::replay::take_note_work();
    assert_eq!(rt.get(reader), n * (n - 1) / 2 + 600);
    assert!(!rt.dooms.lock().unwrap().is_empty(), "the chain doomed");
    let work = crate::replay::take_note_work();
    assert!(
        work <= 16 * n,
        "{work} units of note work for {n} delivered foreign reads"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_wide_frame_deliveries_stay_linear() {
    on_stack(1 << 28, wide_frame_deliveries::<Local>);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_wide_frame_deliveries_stay_linear() {
    on_stack(1 << 28, wide_frame_deliveries::<Shared>);
}

/// A read of one runtime holds `n` landings of its two-node cycles, then
/// reads `n` nodes of another runtime, each a first entry inside the
/// read: each entry's check and exit go over the other runtime's held
/// landings only.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_first_entries_under_held_landings_stay_linear() {
    let n: usize = 4000;
    let held = Runtime::<Shared>::new();
    let cycles: Vec<Incr<i64>> = (0..n / 2)
        .map(|_| {
            let other: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
            let a = {
                let other = other.clone();
                held.create_query_with_fallback(
                    move |rt| rt.get(*other.get().unwrap()) + 1,
                    |_: &crate::cycle::Cycle| 100,
                )
            };
            other
                .set(held.create_query(move |rt| rt.get(a) * 2))
                .unwrap();
            a
        })
        .collect();
    let read = Arc::new(Runtime::<Shared>::new());
    let reads: Vec<Incr<i64>> = (0..n as i64)
        .map(|i| read.create_query(move |_| i))
        .collect();
    let root = {
        let read = read.clone();
        held.create_query(move |rt| {
            let sum = cycles.iter().map(|&c| rt.get(c)).sum::<i64>();
            assert_eq!(crate::hold::held_count(), n);
            sum + reads.iter().map(|&r| read.get(r)).sum::<i64>()
        })
    };
    crate::hold::take_visits();
    assert_eq!(
        held.get(root),
        100 * (n / 2) as i64 + (n as i64) * (n as i64 - 1) / 2
    );
    let visits = crate::hold::take_visits();
    assert!(
        visits <= 8 * n,
        "{visits} held places visited for {n} first entries"
    );
}

/// As above with one text for every panic: each box takes an address an
/// earlier caught one freed, so the panics share a payload key and a note
/// log per address, and each kept read's copy holds every landing caught
/// on its log before it. The retry must not pay that prefix again per
/// read. `through` puts a query between the frame and each panicking
/// one, so every copy also holds landings a level deeper, which each
/// later caught panic's landings take from it.
fn same_text_fan_over_chain<C: Cells>(n: u64, through: bool, frames: u32, len: usize) -> u64 {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(frames);
    let failing: Vec<Incr<u64>> = (0..n)
        .map(|_| {
            let f = rt.create_query_with(|_| -> u64 { panic!("fan") }, replayable());
            if through {
                rt.create_query_with(move |rt| rt.get(f) + 1, replayable())
            } else {
                f
            }
        })
        .collect();
    let top = chain(&rt, len).nodes[0];
    let reader = rt.create_query_with(
        move |rt| {
            let caught = failing
                .iter()
                .filter(|&&f| catch_unwind(AssertUnwindSafe(|| rt.get(f))).is_err())
                .count() as u64;
            caught + rt.get(top)
        },
        replayable(),
    );
    crate::replay::take_note_work();
    assert_eq!(rt.get(reader), n + len as u64);
    assert!(!rt.dooms.lock().unwrap().is_empty(), "the chain doomed");
    crate::replay::take_note_work()
}

fn same_text_fan_over_chain_stays_linear<C: Cells>(through: bool) {
    // Below about 1000 panics the chain's own notes take a share of the
    // bound that only a shorter chain under a smaller budget leaves room for.
    let (frames, len, sizes): (u32, usize, &[u64]) = if cfg!(miri) {
        (8, 24, &[512])
    } else {
        (256, 600, &[1000, 4000, 16000])
    };
    for &n in sizes {
        let work = same_text_fan_over_chain::<C>(n, through, frames, len);
        let per_panic = if through { 40 } else { 32 };
        assert!(
            work <= per_panic * n,
            "{work} units of note work for {n} caught panics of one text, through={through}"
        );
    }
}

#[test]
fn local_same_text_fan_over_chain_stays_linear() {
    on_stack(1 << 26, || {
        same_text_fan_over_chain_stays_linear::<Local>(false)
    });
}

#[test]
fn shared_same_text_fan_over_chain_stays_linear() {
    on_stack(1 << 26, || {
        same_text_fan_over_chain_stays_linear::<Shared>(false);
    });
}

#[test]
fn local_same_text_fan_through_a_frame_over_chain_stays_linear() {
    on_stack(1 << 26, || {
        same_text_fan_over_chain_stays_linear::<Local>(true)
    });
}

#[test]
fn shared_same_text_fan_through_a_frame_over_chain_stays_linear() {
    on_stack(1 << 26, || {
        same_text_fan_over_chain_stays_linear::<Shared>(true)
    });
}
