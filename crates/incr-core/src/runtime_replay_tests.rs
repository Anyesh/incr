//! Deep chains by unwind and replay: dooms, discards, the driver loop and
//! the bounds on runs and walks the plan's T8 tests set.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{Arc, OnceLock};

use super::{Runtime, DEFAULT_MAX_FRAMES};
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{ComputeCtl, CycleStrategy, Fixpoint, QueryOpts, Replay, BUFFERED};
use crate::cycle_state::{with_held, Held};
use crate::dep_stack::DepStack;
use crate::handle::Incr;
use crate::node::NodeId;
use crate::state::NodeState;
use crate::test_size::test_size;

macro_rules! local_and_shared {
    ($body:ident, $local:ident, $shared:ident) => {
        #[test]
        fn $local() {
            $body(&Runtime::<Local>::new());
        }

        #[test]
        fn $shared() {
            $body(&Runtime::<Shared>::new());
        }
    };
}

pub(super) fn replayable<T>() -> QueryOpts<T> {
    QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    }
}

/// Runs `f` on a thread with a `stack`-byte stack.
pub(super) fn on_stack(stack: usize, f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(f)
        .expect("spawn")
        .join()
        .unwrap_or_else(|e| std::panic::resume_unwind(e));
}

/// A chain of `n` replayable queries over an input: `c[i] = c[i + 1] + 1`
/// and the last is the input plus one, each counting its runs.
pub(super) struct Chain {
    pub(super) nodes: Vec<Incr<u64>>,
    pub(super) input: Incr<u64>,
    pub(super) runs: Arc<Vec<AtomicU32>>,
}

pub(super) fn chain<C: Cells>(rt: &Runtime<C>, n: usize) -> Chain {
    let input = rt.create_input(0_u64);
    let runs: Arc<Vec<AtomicU32>> = Arc::new((0..n).map(|_| AtomicU32::new(0)).collect());
    let mut nodes = vec![input; n];
    let mut below = input;
    for i in (0..n).rev() {
        let runs = runs.clone();
        let node = rt.create_query_with(
            move |rt| {
                runs[i].fetch_add(1, SeqCst);
                rt.get(below) + 1
            },
            replayable(),
        );
        nodes[i] = node;
        below = node;
    }
    Chain { nodes, input, runs }
}

impl Chain {
    pub(super) fn max_runs(&self) -> u32 {
        self.runs.iter().map(|r| r.load(SeqCst)).max().unwrap_or(0)
    }

    pub(super) fn reset_runs(&self) {
        for r in self.runs.iter() {
            r.store(0, SeqCst);
        }
    }
}

fn chain_at_budget_1<C: Cells>(rt: &Runtime<C>) {
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let c = chain(rt, 50);
    assert_eq!(rt.get(c.nodes[0]), 50);
    assert!(c.max_runs() <= 2, "a node ran {} times", c.max_runs());
    assert_eq!(rt.dooms.lock().unwrap().len(), 49);
    for &n in &c.nodes {
        assert_eq!(rt.state_of(n), NodeState::Clean);
    }
    rt.assert_invariant_i();

    c.reset_runs();
    rt.set(c.input, 10);
    assert_eq!(rt.get(c.nodes[0]), 60);
    assert!(
        c.max_runs() <= 2,
        "a recompute ran a node {} times",
        c.max_runs()
    );
    c.reset_runs();
    assert_eq!(rt.get(c.nodes[0]), 60);
    assert_eq!(c.max_runs(), 0, "a Clean read runs nothing");
    rt.assert_invariant_i();
}
local_and_shared!(
    chain_at_budget_1,
    local_chain_at_budget_1,
    shared_chain_at_budget_1
);

/// A first compute of an `n`-level chain at budget `frames`, or at the
/// default budget when `None`: native depth stays within the budget, and
/// every node runs at most twice, every doom returning to one context.
fn deep_chain<C: Cells>(rt: &Runtime<C>, n: usize, frames: Option<u32>) {
    rt.set_placeholder(0_u64);
    if let Some(frames) = frames {
        rt.set_max_frames(frames);
    }
    let budget = rt.max_frames.load(Relaxed) as usize;
    let c = chain(rt, n);
    assert_eq!(rt.get(c.nodes[0]), n as u64);
    assert!(c.max_runs() <= 2, "a node ran {} times", c.max_runs());
    let dooms = rt.dooms.lock().unwrap();
    assert!(!dooms.is_empty());
    assert!(dooms.iter().all(|d| d.depth <= budget));
    assert!(
        dooms.iter().all(|d| d.context == dooms[0].context),
        "the chain's dooms opened more than one context"
    );
    for &n in &c.nodes {
        assert_eq!(rt.state_of(n), NodeState::Clean);
    }
}

#[cfg(not(debug_assertions))]
#[test]
fn local_chain_100k_on_a_1_mib_thread() {
    on_stack(1 << 20, || {
        deep_chain(&Runtime::<Local>::new(), 100_000, None)
    });
}

#[cfg(not(debug_assertions))]
#[test]
fn shared_chain_100k_on_a_1_mib_thread() {
    on_stack(1 << 20, || {
        deep_chain(&Runtime::<Shared>::new(), 100_000, None)
    });
}

#[cfg(not(debug_assertions))]
#[test]
fn local_chain_100k_at_budget_1_on_a_1_mib_thread() {
    on_stack(1 << 20, || {
        deep_chain(&Runtime::<Local>::new(), 100_000, Some(1))
    });
}

#[cfg(not(debug_assertions))]
#[test]
fn shared_chain_100k_at_budget_1_on_a_1_mib_thread() {
    on_stack(1 << 20, || {
        deep_chain(&Runtime::<Shared>::new(), 100_000, Some(1))
    });
}

// A debug build's frames are about 13 times a release build's.
#[test]
#[cfg_attr(miri, ignore)]
fn local_chain_10k_at_the_default_budget_on_a_2_mib_thread() {
    on_stack(2 << 20, || {
        deep_chain(&Runtime::<Local>::new(), test_size(10_000), None);
    });
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_chain_10k_at_the_default_budget_on_a_2_mib_thread() {
    on_stack(2 << 20, || {
        deep_chain(&Runtime::<Shared>::new(), test_size(10_000), None);
    });
}

/// A doomed first compute lands New, and the driver computes it again.
fn discarded_first_compute_lands_new<C: Cells>(rt: &Runtime<C>) {
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let c = chain(rt, 3);
    let seen: Arc<std::sync::Mutex<Vec<NodeState>>> = Arc::default();
    assert_eq!(rt.get(c.nodes[0]), 3);
    let dooms = rt.dooms.lock().unwrap().clone();
    assert_eq!(
        dooms.iter().map(|d| d.node).collect::<Vec<_>>(),
        vec![c.nodes[1].slot(), c.nodes[2].slot()]
    );
    for d in &dooms {
        assert_eq!(
            d.state,
            NodeState::New,
            "the doomed read's node is released New"
        );
        assert_eq!(d.verified_at, 0);
    }
    drop(seen);
    assert_eq!(c.runs[0].load(SeqCst), 2);
    assert_eq!(c.runs[2].load(SeqCst), 1);
}
local_and_shared!(
    discarded_first_compute_lands_new,
    local_discarded_first_compute_lands_new,
    shared_discarded_first_compute_lands_new
);

/// A wide node reads many New deps: it dooms on the first, pushes the rest
/// while doomed, and runs twice in all.
fn wide_node_runs_twice<C: Cells>(rt: &Runtime<C>) {
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let leaves: Vec<Incr<u64>> = (0..64_u64).map(|i| rt.create_query(move |_| i)).collect();
    let runs = Arc::new(AtomicU32::new(0));
    let r = runs.clone();
    let wide = rt.create_query_with(
        move |rt| {
            r.fetch_add(1, SeqCst);
            leaves.iter().map(|&l| rt.get(l)).sum::<u64>()
        },
        replayable(),
    );
    assert_eq!(rt.get(wide), (0..64).sum::<u64>());
    assert_eq!(runs.load(SeqCst), 2);
}
local_and_shared!(
    wide_node_runs_twice,
    local_wide_node_runs_twice,
    shared_wide_node_runs_twice
);

/// A read whose type has no placeholder recurses natively.
fn no_placeholder_recurses<C: Cells>(rt: &Runtime<C>) {
    let c = chain(rt, 1000);
    assert_eq!(rt.get(c.nodes[0]), 1000);
    assert!(rt.dooms.lock().unwrap().is_empty());
    assert_eq!(c.max_runs(), 1);
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Miri keeps a whole-stack backtrace per live heap allocation: O(depth^2) memory; passes with -Zmiri-disable-leak-backtraces"
)]
fn local_no_placeholder_recurses_natively() {
    on_stack(256 << 20, || {
        no_placeholder_recurses(&Runtime::<Local>::new())
    });
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Miri keeps a whole-stack backtrace per live heap allocation: O(depth^2) memory; passes with -Zmiri-disable-leak-backtraces"
)]
fn shared_no_placeholder_recurses_natively() {
    on_stack(256 << 20, || {
        no_placeholder_recurses(&Runtime::<Shared>::new())
    });
}

/// Every collection operator's query is `NOT_REPLAYABLE`, which wins
/// over `INCR_TEST_ALL_REPLAYABLE`; a plain query under the same switch
/// is replayable.
fn operators_are_not_replayable<C: Cells>(mut rt: Runtime<C>) {
    rt.all_replayable = true;
    let c = rt.create_collection::<i64>();
    let other = rt.create_collection::<i64>();
    let sorted = c.sort_by_key(&rt, |v| *v);
    let operators = [
        (
            "filter",
            c.filter(&rt, |v| v % 2 == 0).version_node().slot(),
        ),
        ("map", c.map(&rt, |v| v + 1).version_node().slot()),
        ("count", c.count(&rt).slot()),
        ("reduce", c.reduce(&rt, |vs| vs.iter().sum::<i64>()).slot()),
        (
            "aggregate",
            c.aggregate(&rt, 0, |v| *v, |a, b| a + b).slot(),
        ),
        (
            "join",
            c.join(&rt, &other, |v| *v, |v| *v).version_node().slot(),
        ),
        ("group_by", c.group_by(&rt, |v| v % 3).version_node().slot()),
        ("sort_by_key", sorted.version_node().slot()),
        ("pairwise", sorted.pairwise(&rt).version_node().slot()),
        ("window", sorted.window(&rt, 2).version_node().slot()),
    ];
    for (name, slot) in operators {
        assert!(!rt.replayable(slot), "{name} is replayable");
    }
    let plain = rt.create_query(|_| 0_i64);
    assert!(rt.replayable(plain.slot()));
}

#[test]
fn local_operators_are_not_replayable() {
    operators_are_not_replayable(Runtime::<Local>::new());
}

#[test]
fn shared_operators_are_not_replayable() {
    operators_are_not_replayable(Runtime::<Shared>::new());
}

/// No query in the operator sources is created any way but through
/// `create_operator_query`, so an operator added later can't skip the
/// `NOT_REPLAYABLE` mark.
#[test]
fn operator_sources_create_every_query_not_replayable() {
    for (file, src) in [
        ("collection.rs", include_str!("collection.rs")),
        ("sorted_collection.rs", include_str!("sorted_collection.rs")),
    ] {
        let code = src.split("\nmod tests {").next().unwrap();
        let mut classified = 0;
        for (i, line) in code.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            assert!(
                !line.contains("create_query"),
                "{file}:{} creates a query outside create_operator_query",
                i + 1
            );
            classified += line.matches("create_operator_query(").count();
        }
        assert!(classified > 0, "{file} creates no operator query");
    }
}

/// T reads S with `get` inside `catch_unwind`, then P, then itself with
/// `try_get`, then S again; P (Fixpoint) and S (unseeded) read each other.
/// Unbounded, S's read closes the cycle at S, T catches the unwind and is
/// poisoned, and its `get` panics with T left New. At budget 1 the unwind
/// is delivered to T's retry, whose read of P dooms it; P's episode then
/// lands S Clean, so the next retry reads S with a Clean `get`, which no
/// delivery can intercept (departure 7). The result is the unbounded
/// run's when P's episode has landed S before T runs (`warm`).
fn caught_delivery_after_its_node_lands<C: Cells>(
    frames: u32,
    warm: bool,
) -> (bool, [NodeState; 3]) {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(frames);
    let s_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let t_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let s_for_p = s_cell.clone();
    let p = rt.create_query_with(
        move |rt| rt.get(*s_for_p.get().unwrap()),
        QueryOpts {
            cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 3).max_iterations(1)),
            replay: Replay::Allowed,
        },
    );
    let s = rt.create_query_with(move |rt| rt.get(p), replayable());
    s_cell.set(s).unwrap();
    let t_for_t = t_cell.clone();
    let t = rt.create_query_with(
        move |rt| {
            let _ = catch_unwind(AssertUnwindSafe(|| rt.get(s)));
            let _ = rt.get(p);
            let _ = rt.try_get(*t_for_t.get().unwrap());
            rt.try_get(s).unwrap_or(7777)
        },
        replayable(),
    );
    t_cell.set(t).unwrap();
    if warm {
        let _ = rt.try_get(p);
    }
    let ok = catch_unwind(AssertUnwindSafe(|| rt.get(t))).is_ok();
    (ok, [rt.state_of(t), rt.state_of(p), rt.state_of(s)])
}

#[test]
fn local_caught_delivery_after_its_node_lands_reads_it_clean() {
    assert_eq!(
        caught_delivery_after_its_node_lands::<Local>(1, false),
        caught_delivery_after_its_node_lands::<Local>(u32::MAX, true)
    );
}

#[test]
fn shared_caught_delivery_after_its_node_lands_reads_it_clean() {
    assert_eq!(
        caught_delivery_after_its_node_lands::<Shared>(1, false),
        caught_delivery_after_its_node_lands::<Shared>(u32::MAX, true)
    );
}

/// `INCR_MAX_DEPTH` changes only the default frame budget: a byte budget
/// set explicitly runs at `DEFAULT_MAX_FRAMES`, and an explicit frame
/// budget stays.
fn explicit_budgets_win<C: Cells>() {
    let rt = Runtime::<C>::new();
    rt.set_max_stack_bytes(1 << 20);
    assert_eq!(rt.max_frames.load(Relaxed), DEFAULT_MAX_FRAMES);
    let rt = Runtime::<C>::new();
    rt.set_max_frames(3);
    rt.set_max_stack_bytes(1 << 20);
    assert_eq!(rt.max_frames.load(Relaxed), 3);
}

#[test]
fn local_explicit_budgets_win_over_incr_max_depth() {
    explicit_budgets_win::<Local>();
}

#[test]
fn shared_explicit_budgets_win_over_incr_max_depth() {
    explicit_budgets_win::<Shared>();
}

/// A frame's step is one rule: on a stack whose frames read a node more
/// than once, read another generation of it, and run a member of an
/// episode, the path `mark` chains up for each frame from the one below
/// is the one `path_at` gives it.
fn frame_paths_agree<C: Cells>(rt: &Runtime<C>) {
    let id = rt.runtime_id;
    let ctls: Vec<ComputeCtl> = (0..4).map(|_| ComputeCtl::new(0)).collect();
    ctls[1].raise(BUFFERED);
    let mut held = Held::new(0, 1, true, false);
    held.ep = 7;
    with_held(|h| h.insert((id, 11), held));
    let s = &rt.dep_stack;
    s.push_frame(id, NodeId(10), &ctls[0]);
    s.record_dep(id, NodeId(20), 0);
    s.record_read(id, NodeId(20), 0);
    s.push_frame(id, NodeId(11), &ctls[1]);
    s.record_read(id, NodeId(21), 1);
    s.push_frame(id, NodeId(12), &ctls[2]);
    s.record_dep(id, NodeId(21), 3);
    s.record_read(id, NodeId(21), 0);
    s.record_read(id, NodeId(21), 0);
    s.record_read(id, NodeId(21), 0);
    s.push_frame(id, NodeId(13), &ctls[3]);
    s.record_read(id, NodeId(23), 0);

    let steps = rt.assert_frame_paths_agree();

    for _ in 0..4 {
        s.pop_frame();
    }
    with_held(|h| h.remove(&(id, 11)));
    let seen: Vec<_> = steps
        .iter()
        .map(|st| (st.node, st.ordinal, st.episode))
        .collect();
    assert_eq!(
        seen,
        [
            (13, 1, None),
            (12, 3, None),
            (11, 1, Some(7)),
            (10, 2, None)
        ]
    );
}
local_and_shared!(
    frame_paths_agree,
    local_frame_paths_agree,
    shared_frame_paths_agree
);

/// Each round builds a runtime with a frame budget it never reaches, where
/// f reads p and p caught q's panic, so f's read of p is Dirty. Once the
/// round's runtime is dropped, no clone of p's value may survive it: the
/// probe's strong count is back to 1 after every round.
fn wave7_kept_dirty_read_outlives_runtime<C: Cells>() {
    std::thread::spawn(|| {
        let probe = Arc::new(());
        for round in 1..=3_usize {
            let rt = Runtime::<C>::new();
            rt.set_max_frames(64);
            let q = rt.create_query(|_rt| -> u64 { panic!("q boom") });
            let pr = probe.clone();
            let p = rt.create_query(move |rt| {
                let _ = catch_unwind(AssertUnwindSafe(|| rt.get(q)));
                pr.clone()
            });
            let f = rt.create_query(move |rt| {
                let _ = rt.get(p);
                0_u64
            });
            assert_eq!(rt.get(f), 0);
            drop(rt);
            haphazard::Domain::global().eager_reclaim();
            assert_eq!(
                Arc::strong_count(&probe),
                1,
                "round {round}: a kept read outlived its runtime"
            );
        }
    })
    .join()
    .unwrap();
}

#[test]
fn wave7_local_kept_dirty_read_is_released_with_its_runtime() {
    wave7_kept_dirty_read_outlives_runtime::<Local>();
}

#[test]
fn wave7_shared_kept_dirty_read_is_released_with_its_runtime() {
    wave7_kept_dirty_read_outlives_runtime::<Shared>();
}
