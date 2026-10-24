//! Deep chains by unwind and replay: dooms, discards, the driver loop and
//! the bounds on runs and walks the plan's T8 tests set.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{Arc, Mutex, OnceLock};

use super::{Runtime, DEFAULT_MAX_FRAMES};
use crate::cells::{Cells, Local, Shared};
use crate::child_test::run_alone;
use crate::cycle::{ComputeCtl, CycleStrategy, Fixpoint, QueryOpts, Replay, BUFFERED};
use crate::cycle_state::{set_unwind_kind, unwind_kind, with_held, Held, UnwindKind};
use crate::deferred::Deferred;
use crate::dep_stack::DepStack;
use crate::handle::Incr;
use crate::node::NodeId;
use crate::state::{self, NodeState};
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

/// A runtime whose `Replay::Native` queries stay native under
/// `INCR_TEST_ALL_REPLAYABLE`, for tests of what only native queries do.
fn native_runtime<C: Cells>() -> Runtime<C> {
    let mut rt = Runtime::new();
    rt.all_replayable = false;
    rt
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

/// Whether `get(t)` returned, with the states of T, P and S.
type CaughtOutcome = (bool, [NodeState; 3]);

/// T reads S with `get` inside `catch_unwind`, then P, then itself with
/// `try_get`, then S again; P (Fixpoint) and S (unseeded) read each other.
/// Unbounded, S's read closes the cycle at S, T catches the unwind and is
/// poisoned, and its `get` panics with T left New. At budget 1 the unwind
/// is delivered to T's retry, whose read of P dooms it; P's episode then
/// lands S Clean, and the next retry's Clean `get` of S still takes the
/// unwind its context holds, so the result is the unbounded run's.
fn caught_delivery_after_its_node_lands<C: Cells>(frames: u32) -> CaughtOutcome {
    caught_delivery_run::<C>(frames).0
}

/// [`caught_delivery_after_its_node_lands`], with whether each of T's runs
/// that reached its end began diverted and ended diverted.
fn caught_delivery_run<C: Cells>(frames: u32) -> (CaughtOutcome, Vec<(bool, bool)>) {
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
    let diverted = Arc::new(Mutex::new(Vec::new()));
    let seen = diverted.clone();
    let t = rt.create_query_with(
        move |rt| {
            let at_start = rt.dep_stack.diverts_clean_reads(rt.runtime_id);
            let _ = catch_unwind(AssertUnwindSafe(|| rt.get(s)));
            let _ = rt.get(p);
            let _ = rt.try_get(*t_for_t.get().unwrap());
            let out = rt.try_get(s).unwrap_or(7777);
            let at_end = rt.dep_stack.diverts_clean_reads(rt.runtime_id);
            seen.lock().unwrap().push((at_start, at_end));
            out
        },
        replayable(),
    );
    t_cell.set(t).unwrap();
    let ok = catch_unwind(AssertUnwindSafe(|| rt.get(t))).is_ok();
    let out = (ok, [rt.state_of(t), rt.state_of(p), rt.state_of(s)]);
    let diverted = diverted.lock().unwrap().clone();
    (out, diverted)
}

/// T's retry starts diverted, with deliveries to take, and its reads take
/// them all, so it ends reading fast.
fn a_retry_ends_its_divert_with_its_last_delivery<C: Cells>() {
    let runs = caught_delivery_run::<C>(1).1;
    assert!(runs.iter().any(|&(start, _)| start), "{runs:?}");
    assert!(runs.iter().all(|&(_, end)| !end), "{runs:?}");
}

#[test]
fn local_a_retry_ends_its_divert_with_its_last_delivery() {
    a_retry_ends_its_divert_with_its_last_delivery::<Local>();
}

#[test]
fn shared_a_retry_ends_its_divert_with_its_last_delivery() {
    a_retry_ends_its_divert_with_its_last_delivery::<Shared>();
}

#[test]
fn local_caught_delivery_after_its_node_lands_is_taken() {
    assert_eq!(
        caught_delivery_after_its_node_lands::<Local>(1),
        caught_delivery_after_its_node_lands::<Local>(u32::MAX)
    );
}

#[test]
fn shared_caught_delivery_after_its_node_lands_is_taken() {
    assert_eq!(
        caught_delivery_after_its_node_lands::<Shared>(1),
        caught_delivery_after_its_node_lands::<Shared>(u32::MAX)
    );
}

/// A runtime is listed as unbudgeted exactly while it has neither budget,
/// and leaves the list when it is dropped.
#[test]
fn the_unbudgeted_list_follows_the_budgets() {
    let rt = Runtime::<Local>::new();
    let id = rt.runtime_id;
    rt.set_max_frames(u32::MAX);
    assert!(super::replay_impl::unbudgeted(id));
    rt.set_max_stack_bytes(1 << 20);
    assert!(!super::replay_impl::unbudgeted(id));
    rt.set_max_stack_bytes(0);
    assert!(super::replay_impl::unbudgeted(id));
    rt.set_max_frames(8);
    assert!(!super::replay_impl::unbudgeted(id));
    rt.set_max_frames(u32::MAX);
    drop(rt);
    assert!(!super::replay_impl::unbudgeted(id));
}

/// Races `set_max_frames(u32::MAX)` with a byte budget: another thread
/// sets the byte budget after `set_max_frames` is called and before it
/// takes the list's lock.
pub(super) fn race_budget_setters(rt: &Runtime<Shared>, bytes: usize) {
    let at = std::ptr::from_ref(rt) as usize;
    super::replay_impl::NOTE_DECIDED.with(|h| {
        h.set(Some(Box::new(move || {
            // SAFETY: `rt` outlives `set_max_frames`, whose `note` runs this.
            let rt = unsafe { &*(at as *const Runtime<Shared>) };
            std::thread::scope(|s| {
                s.spawn(|| rt.set_max_stack_bytes(bytes));
            });
        })));
    });
    rt.set_max_frames(u32::MAX);
}

/// Two threads setting one runtime's budgets leave it listed exactly when
/// it has neither.
#[test]
fn racing_budget_setters_leave_a_byte_budget_unlisted() {
    let rt = Runtime::<Shared>::new();
    race_budget_setters(&rt, 1 << 20);
    assert_eq!(rt.max_stack_bytes.load(Relaxed), 1 << 20);
    assert!(
        !super::replay_impl::unbudgeted(rt.runtime_id),
        "listed unbudgeted while it has a byte budget"
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
    s.record_dep(id, NodeId(20), 0, 0);
    s.record_read(id, NodeId(20), 0);
    s.push_frame(id, NodeId(11), &ctls[1]);
    s.record_read(id, NodeId(21), 1);
    s.push_frame(id, NodeId(12), &ctls[2]);
    s.record_dep(id, NodeId(21), 3, 0);
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

/// A diverted frame's Clean read with no delivery to take leaves the
/// frame as the fast read does: the same value, one dep, the `reading` of
/// the frame's last non-Clean read, which a doom names, and the unwind
/// kind a closure that caught a cycle's unwind left set.
fn diverted_clean_read_is_a_fast_read<C: Cells>(rt: &Runtime<C>) {
    let id = rt.runtime_id;
    let c = chain(rt, 2);
    assert_eq!(rt.get(c.nodes[0]), 2);
    let s = &rt.dep_stack;
    let read = |divert: bool| {
        let ctl = ComputeCtl::new(0);
        s.push_frame(id, NodeId(c.nodes[0].slot()), &ctl);
        s.record_read(id, NodeId(c.input.slot()), c.input.generation());
        if divert {
            s.divert_clean_reads();
            crate::value_code::set_replay_bit(true);
            assert!(s.diverts_clean_reads(id));
        }
        set_unwind_kind(Some(UnwindKind::Cycle));
        let value = rt.get(c.nodes[1]);
        let kind = unwind_kind();
        set_unwind_kind(None);
        let still_diverted = s.diverts_clean_reads(id);
        crate::value_code::set_replay_bit(false);
        (value, kind, still_diverted, s.pop_frame_reading())
    };
    let fast = read(false);
    assert_eq!(read(true), fast);
    assert!(
        !fast.2,
        "with nothing left to take the frame reads fast again"
    );
    assert_eq!(
        fast.3 .0,
        vec![
            (NodeId(c.input.slot()), c.input.generation()),
            (NodeId(c.nodes[1].slot()), c.nodes[1].generation()),
        ]
    );
}
local_and_shared!(
    diverted_clean_read_is_a_fast_read,
    local_diverted_clean_read_is_a_fast_read,
    shared_diverted_clean_read_is_a_fast_read
);

/// Value code, such as a landed value's clone, can run inside a replay
/// context's borrow: its Clean read of a diverted frame's dep reads the
/// slot and leaves the contexts and the divert alone.
fn value_code_reads_past_a_diverted_frame<C: Cells>(rt: &Runtime<C>) {
    let id = rt.runtime_id;
    let c = chain(rt, 2);
    let want = rt.get(c.nodes[1]);
    let s = &rt.dep_stack;
    let ctl = ComputeCtl::new(0);
    s.push_frame(id, NodeId(c.nodes[0].slot()), &ctl);
    s.divert_clean_reads();
    crate::value_code::set_replay_bit(true);
    let got = crate::replay::with(|_| {
        let _code = crate::value_code::enter();
        rt.get(c.nodes[1])
    });
    crate::value_code::set_replay_bit(false);
    assert_eq!(got, want);
    assert!(s.diverts_clean_reads(id));
    s.pop_frame();
}
local_and_shared!(
    value_code_reads_past_a_diverted_frame,
    local_value_code_reads_past_a_diverted_frame,
    shared_value_code_reads_past_a_diverted_frame
);

/// A user `Drop` of a value a context held, such as a replaced landing's,
/// runs inside the contexts' borrow outside value code: its Clean read of
/// a diverted frame's dep reads the slot as value code's does.
fn a_drop_inside_a_replay_borrow_reads_past_a_diverted_frame<C: Cells>(rt: &Runtime<C>) {
    let id = rt.runtime_id;
    let c = chain(rt, 2);
    let want = rt.get(c.nodes[1]);
    let s = &rt.dep_stack;
    let ctl = ComputeCtl::new(0);
    s.push_frame(id, NodeId(c.nodes[0].slot()), &ctl);
    s.divert_clean_reads();
    crate::value_code::set_replay_bit(true);
    let got = crate::replay::with(|_| rt.get(c.nodes[1]));
    crate::value_code::set_replay_bit(false);
    assert_eq!(got, want);
    assert!(s.diverts_clean_reads(id));
    s.pop_frame();
}
local_and_shared!(
    a_drop_inside_a_replay_borrow_reads_past_a_diverted_frame,
    local_a_drop_inside_a_replay_borrow_reads_past_a_diverted_frame,
    shared_a_drop_inside_a_replay_borrow_reads_past_a_diverted_frame
);

/// Each round builds a runtime with a frame budget it never reaches, where
/// f reads p and p caught q's panic, so f's read of p is Dirty. Once the
/// round's runtime is dropped, no clone of p's value may survive it: the
/// probe's strong count is back to 1 after every round. With `member`, the
/// read is an episode member run's: f is in a cycle with a Fallback `a`.
fn wave7_kept_dirty_read_outlives_runtime<C: Cells>(member: bool) {
    std::thread::spawn(move || {
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
            let a_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
            let f = {
                let a_cell = a_cell.clone();
                rt.create_query(move |rt| {
                    let _ = rt.get(p);
                    a_cell.get().map_or(0, |&a| rt.get(a))
                })
            };
            if member {
                a_cell
                    .set(rt.create_query_with_fallback(move |rt| rt.get(f), |_| 0_u64))
                    .unwrap();
                assert_eq!(rt.get(*a_cell.get().unwrap()), 0);
            } else {
                assert_eq!(rt.get(f), 0);
            }
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
    wave7_kept_dirty_read_outlives_runtime::<Local>(false);
    wave7_kept_dirty_read_outlives_runtime::<Local>(true);
}

#[test]
fn wave7_shared_kept_dirty_read_is_released_with_its_runtime() {
    run_alone(
        "runtime::replay_tests::wave7_shared_kept_dirty_read_is_released_with_its_runtime",
        || {
            wave7_kept_dirty_read_outlives_runtime::<Shared>(false);
            wave7_kept_dirty_read_outlives_runtime::<Shared>(true);
        },
    );
}

/// A replayable `r` that reads `b` (5) at budget 1, so its doomed run reads
/// the placeholder 0 and takes the branch to `c`, which the unbounded run
/// never reads. `c` reads `input` (1) plus 10 and counts its runs, and is
/// replayable when `replay` is set.
struct SideRead {
    r: Incr<u64>,
    c: Incr<u64>,
    input: Incr<u64>,
    runs: Arc<AtomicU32>,
}

fn side_read<C: Cells>(rt: &Runtime<C>, replay: bool) -> SideRead {
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let input = rt.create_input(1_u64);
    let b = rt.create_query(|_| 5_u64);
    let runs = Arc::new(AtomicU32::new(0));
    let counted = runs.clone();
    let c = rt.create_query_with(
        move |rt| {
            counted.fetch_add(1, SeqCst);
            rt.get(input) + 10
        },
        QueryOpts {
            replay: if replay {
                Replay::Allowed
            } else {
                Replay::Native
            },
            ..Default::default()
        },
    );
    let r = rt.create_query_with(
        move |rt| {
            let x = rt.get(b);
            if x == 0 {
                rt.get(c)
            } else {
                x
            }
        },
        replayable(),
    );
    SideRead { r, c, input, runs }
}

/// The speculative run's compute of `c` lands nothing: `c` stays New with
/// no edge, so `input` deletes as in the unbounded run.
fn speculative_compute_leaves_new<C: Cells>(replay: bool) {
    let rt = Runtime::<C>::new();
    let s = side_read(&rt, replay);
    assert_eq!(rt.get(s.r), 5);
    assert_eq!(rt.state_of(s.c), NodeState::New);
    assert_eq!(rt.try_delete_node(s.input), Ok(()));
}

#[test]
fn local_speculative_compute_leaves_new() {
    speculative_compute_leaves_new::<Local>(false);
    speculative_compute_leaves_new::<Local>(true);
}

#[test]
fn shared_speculative_compute_leaves_new() {
    speculative_compute_leaves_new::<Shared>(false);
    speculative_compute_leaves_new::<Shared>(true);
}

/// A Native `c` that a speculative run computed runs at most once in the
/// revision: a later `get` adopts the stash, and after a set the stash is
/// gone and `c` computes for the new revision.
fn speculative_native_runs_once_per_revision<C: Cells>(rt: &Runtime<C>) {
    let s = side_read(rt, false);
    assert_eq!(rt.get(s.r), 5);
    assert_eq!(s.runs.load(SeqCst), 1);
    assert_eq!(rt.get(s.c), 11);
    assert_eq!(s.runs.load(SeqCst), 1, "c ran twice in one revision");
    rt.set(s.input, 2);
    assert_eq!(rt.get(s.c), 12);
    assert_eq!(s.runs.load(SeqCst), 2);
}

local_and_shared!(
    speculative_native_runs_once_per_revision,
    local_speculative_native_runs_once_per_revision,
    shared_speculative_native_runs_once_per_revision
);

/// `c` is Clean at 11 under `d`, and a set makes it Dirty. The speculative
/// recompute of `c` (12) leaves it Dirty with its slot at 11, so `d`'s
/// verify sees `c` change rather than cut off against the speculative 12.
fn speculative_recompute_leaves_dirty<C: Cells>(rt: &Runtime<C>) {
    let s = side_read(rt, true);
    let c = s.c;
    let d = rt.create_query(move |rt| rt.get(c) * 2);
    assert_eq!(rt.get(d), 22);
    rt.set(s.input, 2);
    assert_eq!(rt.get(s.r), 5);
    assert_eq!(rt.state_of(c), NodeState::Dirty);
    assert_eq!(rt.get(d), 24);
}

local_and_shared!(
    speculative_recompute_leaves_dirty,
    local_speculative_recompute_leaves_dirty,
    shared_speculative_recompute_leaves_dirty
);

/// Replayable links over Native leaves, each counting its runs: link 0
/// reads `fan` leaves and each later link the one before it and up to four
/// more. Every leaf reads `input`. At `budget` the links doom and retry,
/// and their doomed runs read leaves ahead of need, yet each leaf's closure
/// runs once per revision.
fn native_leaves_run_once_per_revision<C: Cells>(len: usize, fan: u64, budget: u32) {
    let rt = native_runtime::<C>();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(budget);
    let input = rt.create_input(0_u64);
    let width = |i: usize| if i == 0 { fan } else { fan.min(4) };
    let mut runs = Vec::new();
    let mut top: Option<Incr<u64>> = None;
    for i in 0..len {
        let leaves: Vec<Incr<u64>> = (0..width(i))
            .map(|j| {
                let count = Arc::new(AtomicU32::new(0));
                runs.push(count.clone());
                rt.create_query(move |rt| {
                    count.fetch_add(1, SeqCst);
                    rt.get(input) + j
                })
            })
            .collect();
        let below = top;
        top = Some(rt.create_query_with(
            move |rt| {
                below.map_or(0, |b| rt.get(b)) + leaves.iter().map(|&l| rt.get(l)).sum::<u64>()
            },
            replayable(),
        ));
    }
    let top = top.unwrap();
    let want = |v: u64| -> u64 {
        (0..len)
            .map(|i| width(i) * v + width(i) * (width(i) - 1) / 2)
            .sum()
    };
    for (revision, v) in [(1, 0), (2, 1)] {
        rt.set(input, v);
        assert_eq!(rt.get(top), want(v));
        let ran: Vec<u32> = runs.iter().map(|r| r.load(SeqCst)).collect();
        assert!(
            ran.iter().all(|&n| n == revision),
            "len {len} fan {fan} budget {budget}: leaf runs {:?} in revision {revision}",
            ran.iter().filter(|&&n| n != revision).collect::<Vec<_>>()
        );
    }
}

fn native_leaves_at_every_budget<C: Cells>() {
    on_stack(1 << 30, || {
        for (len, fan) in [(1, 10_000), (200, 10_000), (5000, 1)] {
            for budget in [1, 8, 32] {
                native_leaves_run_once_per_revision::<C>(len, fan, budget);
            }
        }
    });
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_native_leaves_run_once_per_revision() {
    native_leaves_at_every_budget::<Local>();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_native_leaves_run_once_per_revision() {
    native_leaves_at_every_budget::<Shared>();
}

/// The `side_read` shape with every query Native: nothing replays, so no
/// run is speculative at budget 1, `c` never runs and nothing is stashed.
fn native_graph_never_speculates<C: Cells>(rt: &Runtime<C>) {
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let runs = Arc::new(AtomicU32::new(0));
    let counted = runs.clone();
    let b = rt.create_query(|rt| {
        assert!(!rt.computing_speculative());
        5_u64
    });
    let c = rt.create_query(move |rt| {
        assert!(!rt.computing_speculative());
        counted.fetch_add(1, SeqCst);
        11_u64
    });
    let mut top = rt.create_query(move |rt| {
        assert!(!rt.computing_speculative());
        let x = rt.get(b);
        if x == 0 {
            rt.get(c)
        } else {
            x
        }
    });
    for _ in 0..50 {
        let below = top;
        top = rt.create_query(move |rt| {
            assert!(!rt.computing_speculative());
            rt.get(below) + 1
        });
    }
    assert_eq!(rt.get(top), 55);
    assert_eq!(runs.load(SeqCst), 0);
    assert_eq!(C::u32_load_relaxed(&rt.stash_count), 0);
    assert!(rt.dooms.lock().unwrap().is_empty());
}

#[test]
fn local_native_graph_never_speculates() {
    native_graph_never_speculates(&native_runtime::<Local>());
}

#[test]
fn shared_native_graph_never_speculates() {
    native_graph_never_speculates(&native_runtime::<Shared>());
}

/// `c` and `e` are a cycle with no strategy, reached only by the
/// speculative runs of `r` and of `r2`, its copy. The cycle stashes nothing
/// and lands neither node, and when `c` is Native `r2`'s run does not
/// compute it again in the revision; the later `get` of `c` computes it and
/// meets the cycle as the unbounded run does.
fn speculative_cycle_is_recomputed<C: Cells>(replay: bool) {
    let rt = native_runtime::<C>();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let opts = || QueryOpts {
        replay: if replay {
            Replay::Allowed
        } else {
            Replay::Native
        },
        ..Default::default()
    };
    let runs = Arc::new(AtomicU32::new(0));
    let counted = runs.clone();
    let e_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let e_of_c = e_cell.clone();
    let c = rt.create_query_with(
        move |rt| {
            counted.fetch_add(1, SeqCst);
            rt.get(*e_of_c.get().unwrap()) + 1
        },
        opts(),
    );
    e_cell
        .set(rt.create_query_with(move |rt| rt.get(c) + 1, opts()))
        .unwrap();
    let reader = || {
        let b = rt.create_query(|_| 5_u64);
        rt.create_query_with(
            move |rt| {
                let x = rt.get(b);
                if x == 0 {
                    rt.get(c)
                } else {
                    x
                }
            },
            replayable(),
        )
    };
    let (r, r2) = (reader(), reader());
    assert_eq!(rt.get(r), 5);
    assert_eq!(rt.state_of(c), NodeState::New);
    assert_eq!(rt.state_of(*e_cell.get().unwrap()), NodeState::New);
    let before = runs.load(SeqCst);
    assert!(before <= 1);
    assert_eq!(rt.get(r2), 5);
    if !replay {
        assert_eq!(runs.load(SeqCst), before, "replay {replay}: c ran twice");
    }
    let payload = catch_unwind(AssertUnwindSafe(|| rt.get(c))).expect_err("c cycles");
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("dependency cycle"), "{message}");
    assert!(runs.load(SeqCst) > before, "the real claim recomputes c");
}

#[test]
fn local_speculative_cycle_is_recomputed() {
    speculative_cycle_is_recomputed::<Local>(false);
    speculative_cycle_is_recomputed::<Local>(true);
}

#[test]
fn shared_speculative_cycle_is_recomputed() {
    speculative_cycle_is_recomputed::<Shared>(false);
    speculative_cycle_is_recomputed::<Shared>(true);
}

/// `r`'s speculative run computes the replayable `p` over the Native `n`,
/// which reads input `d`, and stashes both. `d` is deleted, so the real
/// claim of `p` takes `n`'s stash and fails it on `d`'s generation: `n`
/// keeps a tried mark for the revision, and only `p`'s goes.
fn failed_nested_adoption_keeps_the_tried_mark<C: Cells>(rt: &Runtime<C>) {
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let d = rt.create_input(1_u64);
    let n = rt.create_query(move |rt| rt.get(d) + 10);
    let p = rt.create_query_with(move |rt| rt.get(n) + 1, replayable());
    let b = rt.create_query(|_| 5_u64);
    let r = rt.create_query_with(
        move |rt| {
            let x = rt.get(b);
            if x == 0 {
                rt.get(p)
            } else {
                x
            }
        },
        replayable(),
    );
    assert_eq!(rt.get(r), 5);
    assert_eq!(C::u32_load_relaxed(&rt.stash_count), 2);
    assert_eq!(rt.try_delete_node(d), Ok(()));
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(p))).is_err());
    assert_eq!(C::u32_load_relaxed(&rt.stash_count), 1, "n lost its mark");
}

#[test]
fn local_failed_nested_adoption_keeps_the_tried_mark() {
    failed_nested_adoption_keeps_the_tried_mark(&native_runtime::<Local>());
}

#[test]
fn shared_failed_nested_adoption_keeps_the_tried_mark() {
    failed_nested_adoption_keeps_the_tried_mark(&native_runtime::<Shared>());
}

/// A speculative run's panic in `c` is stashed, not landed, and the later
/// real `get` raises that payload with no second run, landing `c` Failed.
fn speculative_panic_is_raised_once<C: Cells>(replay: bool) {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let b = rt.create_query(|_| 5_u64);
    let runs = Arc::new(AtomicU32::new(0));
    let counted = runs.clone();
    let c = rt.create_query_with(
        move |_| -> u64 {
            counted.fetch_add(1, SeqCst);
            panic!("c boom")
        },
        QueryOpts {
            replay: if replay {
                Replay::Allowed
            } else {
                Replay::Native
            },
            ..Default::default()
        },
    );
    let r = rt.create_query_with(
        move |rt| {
            let x = rt.get(b);
            if x == 0 {
                rt.get(c)
            } else {
                x
            }
        },
        replayable(),
    );
    assert_eq!(rt.get(r), 5);
    assert_eq!(runs.load(SeqCst), 1);
    assert_eq!(rt.state_of(c), NodeState::New);
    let payload = catch_unwind(AssertUnwindSafe(|| rt.get(c))).expect_err("c panics");
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"c boom"));
    assert_eq!(runs.load(SeqCst), 1, "c ran twice in one revision");
    assert_eq!(rt.state_of(c), NodeState::Failed);
}

#[test]
fn local_speculative_panic_is_raised_once() {
    speculative_panic_is_raised_once::<Local>(false);
    speculative_panic_is_raised_once::<Local>(true);
}

#[test]
fn shared_speculative_panic_is_raised_once() {
    speculative_panic_is_raised_once::<Shared>(false);
    speculative_panic_is_raised_once::<Shared>(true);
}

/// Counts its drops.
#[derive(Clone)]
struct Dropped(Arc<AtomicU32>);

impl PartialEq for Dropped {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.fetch_add(1, SeqCst);
    }
}

/// A value whose `PartialEq` panics once `boom` is set.
#[derive(Clone, Debug)]
struct EqBoom {
    v: u64,
    boom: Arc<AtomicBool>,
}

impl PartialEq for EqBoom {
    fn eq(&self, other: &Self) -> bool {
        assert!(!self.boom.load(SeqCst), "eq boom");
        self.v == other.v
    }
}

/// `x`'s recompute is stashed by a speculative run; when the real claim
/// adopts it, the write's compare panics. The unbounded run's write panics
/// after its one run too, so `x` lands Failed with that payload and its
/// closure does not run again.
fn adopted_write_panic_lands_failed<C: Cells>(replay: bool) {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let input = rt.create_input(1_u64);
    let b = rt.create_query(|_| 5_u64);
    let boom = Arc::new(AtomicBool::new(false));
    let runs = Arc::new(AtomicU32::new(0));
    let (counted, armed) = (runs.clone(), boom.clone());
    let x = rt.create_query_with(
        move |rt| {
            counted.fetch_add(1, SeqCst);
            EqBoom {
                v: rt.get(input),
                boom: armed.clone(),
            }
        },
        QueryOpts {
            replay: if replay {
                Replay::Allowed
            } else {
                Replay::Native
            },
            ..Default::default()
        },
    );
    let r = rt.create_query_with(
        move |rt| {
            let v = rt.get(b);
            if v == 0 {
                rt.get(x).v
            } else {
                v
            }
        },
        replayable(),
    );
    assert_eq!(rt.get(x).v, 1);
    rt.set(input, 2);
    assert_eq!(rt.get(r), 5);
    assert_eq!(runs.load(SeqCst), 2);
    boom.store(true, SeqCst);
    let payload = catch_unwind(AssertUnwindSafe(|| rt.get(x))).expect_err("x panics");
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"eq boom"));
    assert_eq!(
        runs.load(SeqCst),
        2,
        "x ran again after its adoption's write panicked"
    );
    assert_eq!(rt.state_of(x), NodeState::Failed);
}

#[test]
fn local_adopted_write_panic_lands_failed() {
    adopted_write_panic_lands_failed::<Local>(false);
    adopted_write_panic_lands_failed::<Local>(true);
}

#[test]
fn shared_adopted_write_panic_lands_failed() {
    adopted_write_panic_lands_failed::<Shared>(false);
    adopted_write_panic_lands_failed::<Shared>(true);
}

/// A Native creator `c` that only a speculative run of `r` reaches first:
/// its run, a value or a panic, is stashed with the child it made, so the
/// real `get(c)` adopts it and `c` runs once in the revision, with the
/// unbounded run's nodes. With `adopt` off, a set ends the revision before
/// any real read, and the stash's child goes with it. Returns `c`'s value
/// (`None` for a panic or no read), its runs and the node count.
fn native_creator_runs_once<C: Cells>(
    frames: u32,
    adopt: bool,
    panics: bool,
) -> (Option<u64>, u32, usize) {
    let rt = native_runtime::<C>();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(frames);
    let input = rt.create_input(1_u64);
    let b = rt.create_query(|_| 5_u64);
    let runs = Arc::new(AtomicU32::new(0));
    let counted = runs.clone();
    let c = rt.create_query(move |rt: &Runtime<C>| {
        counted.fetch_add(1, SeqCst);
        let k = rt.child_input(1_u32, || 7_u64);
        assert!(!panics || rt.get(k) != 7, "creator panic");
        rt.get(k) + rt.get(input)
    });
    let r = rt.create_query_with(
        move |rt| {
            let x = rt.get(b);
            if x == 0 {
                rt.get(c)
            } else {
                x
            }
        },
        replayable(),
    );
    assert_eq!(rt.get(r), 5);
    if adopt {
        let v = catch_unwind(AssertUnwindSafe(|| rt.get(c))).ok();
        return (v, runs.load(SeqCst), rt.node_count());
    }
    rt.set(input, 2);
    (None, runs.load(SeqCst), rt.node_count())
}

fn native_creator_runs_once_as_unbounded<C: Cells>() {
    for panics in [false, true] {
        let (v, runs, nodes) = native_creator_runs_once::<C>(1, true, panics);
        assert_eq!(
            (v, runs, nodes),
            native_creator_runs_once::<C>(u32::MAX, true, panics),
            "(c, c's runs, node count) at budget 1 vs unbounded, panics {panics}"
        );
        // The unbounded run never reaches `c`; one run ahead of need is
        // the bound.
        let (v, runs, nodes) = native_creator_runs_once::<C>(1, false, panics);
        let (_, unbounded_runs, unbounded_nodes) =
            native_creator_runs_once::<C>(u32::MAX, false, panics);
        assert_eq!(
            (v, runs, nodes),
            (None, unbounded_runs + 1, unbounded_nodes),
            "(c, c's runs, node count) at budget 1, no adoption, panics {panics}"
        );
    }
}

#[test]
fn local_native_creator_runs_once_as_unbounded() {
    native_creator_runs_once_as_unbounded::<Local>();
}

#[test]
fn shared_native_creator_runs_once_as_unbounded() {
    native_creator_runs_once_as_unbounded::<Shared>();
}

/// A creator's stash that its node's own claim drops unadopted, as a group
/// member's verify does, takes the child its run made with it.
fn dropped_creator_stash_discards_its_child<C: Cells>() {
    let rt = native_runtime::<C>();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let b = rt.create_query(|_| 5_u64);
    let c = rt.create_query(|rt: &Runtime<C>| rt.get(rt.child_input(1_u32, || 7_u64)));
    let r = rt.create_query_with(
        move |rt| {
            let x = rt.get(b);
            if x == 0 {
                rt.get(c)
            } else {
                x
            }
        },
        replayable(),
    );
    let before = rt.node_count();
    assert_eq!(rt.get(r), 5);
    assert_eq!(
        rt.node_count(),
        before + 1,
        "c's speculative run made its child"
    );
    let id = NodeId(c.slot());
    let node = rt.nodes.get(c.slot());
    let source = state::try_claim_compute::<C>(node.state_cell()).expect("c lands back unclaimed");
    let mut bag = Deferred::new();
    rt.drop_stash(id, c.generation(), &mut bag);
    rt.land_unrun(id, node, source);
    bag.drain(&rt);
    assert_eq!(rt.node_count(), before, "the dropped stash's child is left");
}

#[test]
fn local_dropped_creator_stash_discards_its_child() {
    dropped_creator_stash_discards_its_child::<Local>();
}

#[test]
fn shared_dropped_creator_stash_discards_its_child() {
    dropped_creator_stash_discards_its_child::<Shared>();
}

/// How the speculative run of `c` in [`native_rerun_exception`] reads
/// `p`, a query that panics.
#[derive(Copy, Clone, Debug, PartialEq)]
enum RerunShape {
    /// `c` catches the panic of `p`, which a real read left Failed: its run
    /// read a dep that did not land Clean.
    Poisoned,
    /// `c` catches the panic of `p`, whose own speculative run stashed it:
    /// the run unwinds with a discard where it would catch the panic.
    StashedPanic,
    /// `c` reads replayable `e` past the budget: its run dooms, and the
    /// run that retries it is stashed, so `c` runs as often as unbounded.
    Doomed,
    /// `c` reads Native `d`, which catches the panic of Failed `p`: `d`'s
    /// run keeps nothing, so the chain above it drops.
    DroppedChain,
}

/// A Native `c` reached first by a speculative run of `r` whose run keeps
/// nothing to adopt: the real `get(c)` runs its closure again, one run more
/// than the unbounded run, with the unbounded run's value. Returns `c`'s
/// value and runs.
fn native_rerun_exception<C: Cells>(frames: u32, shape: RerunShape) -> (u64, u32) {
    let rt = native_runtime::<C>();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(frames);
    let b = rt.create_query(|_| 5_u64);
    let p = rt.create_query(|_: &Runtime<C>| -> u64 { panic!("p fails") });
    let caught = move |rt: &Runtime<C>| catch_unwind(AssertUnwindSafe(|| rt.get(p))).unwrap_or(9);
    let d = rt.create_query(caught);
    let f = rt.create_query_with(|_| 3_u64, replayable());
    let e = rt.create_query_with(move |rt| rt.get(f) + 1, replayable());
    let runs = Arc::new(AtomicU32::new(0));
    let counted = runs.clone();
    let c = rt.create_query(move |rt: &Runtime<C>| {
        counted.fetch_add(1, SeqCst);
        match shape {
            RerunShape::Poisoned | RerunShape::StashedPanic => caught(rt) + 1,
            RerunShape::Doomed => rt.get(e) + 1,
            RerunShape::DroppedChain => rt.get(d) + 1,
        }
    });
    let r = rt.create_query_with(
        move |rt| {
            let x = rt.get(b);
            if x == 0 {
                rt.get(c)
            } else {
                x
            }
        },
        replayable(),
    );
    if shape != RerunShape::StashedPanic {
        assert!(catch_unwind(AssertUnwindSafe(|| rt.get(p))).is_err());
    }
    assert_eq!(rt.get(r), 5);
    let v = rt.get(c);
    (v, runs.load(SeqCst))
}

fn native_rerun_exceptions<C: Cells>() {
    for shape in [
        RerunShape::Poisoned,
        RerunShape::StashedPanic,
        RerunShape::Doomed,
        RerunShape::DroppedChain,
    ] {
        let (v, runs) = native_rerun_exception::<C>(u32::MAX, shape);
        let rerun = u32::from(shape != RerunShape::Doomed);
        assert_eq!(
            native_rerun_exception::<C>(1, shape),
            (v, runs + rerun),
            "{shape:?}: (c, c's runs) at budget 1 vs unbounded"
        );
    }
}

#[test]
fn local_native_rerun_exceptions() {
    native_rerun_exceptions::<Local>();
}

#[test]
fn shared_native_rerun_exceptions() {
    native_rerun_exceptions::<Shared>();
}

/// A stash no real claim adopts is freed when a set ends its revision, and
/// when the runtime drops.
fn unadopted_stash_is_freed<C: Cells>(set: bool) {
    let drops = Arc::new(AtomicU32::new(0));
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    rt.set_placeholder(Dropped(Arc::new(AtomicU32::new(0))));
    rt.set_max_frames(1);
    let input = rt.create_input(1_u64);
    let b = rt.create_query(|_| 5_u64);
    let made = drops.clone();
    let c = rt.create_query(move |rt| {
        rt.get(input);
        Dropped(made.clone())
    });
    let r = rt.create_query_with(
        move |rt| {
            let x = rt.get(b);
            if x == 0 {
                drop(rt.get(c));
            }
            x
        },
        replayable(),
    );
    assert_eq!(rt.get(r), 5);
    assert_eq!(rt.state_of(c), NodeState::New);
    assert_eq!(C::u32_load_relaxed(&rt.stash_count), 1);
    let before = drops.load(SeqCst);
    if set {
        rt.set(input, 2);
        assert_eq!(C::u32_load_relaxed(&rt.stash_count), 0);
    } else {
        drop(rt);
    }
    assert_eq!(
        drops.load(SeqCst),
        before + 1,
        "the stashed value was freed"
    );
}

#[test]
fn local_unadopted_stash_is_freed() {
    unadopted_stash_is_freed::<Local>(true);
    unadopted_stash_is_freed::<Local>(false);
}

#[test]
fn shared_unadopted_stash_is_freed() {
    unadopted_stash_is_freed::<Shared>(true);
    unadopted_stash_is_freed::<Shared>(false);
}

/// `r`'s read of `q` dooms at budget 1, so its doomed run reads `q`'s
/// placeholder 0 and goes on to `c`, a Native node that panics if it sees
/// `q` as 0. A frame that reads a placeholder is doomed and stashes
/// nothing, so `c`'s speculative run reads `q`'s real 7: it neither panics
/// nor runs again when a later `get` adopts it.
fn speculative_run_never_sees_a_placeholder<C: Cells>(rt: &Runtime<C>) {
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let q = rt.create_query_with(|_| 7_u64, replayable());
    let runs = Arc::new(AtomicU32::new(0));
    let counted = runs.clone();
    let c = rt.create_query(move |rt| {
        counted.fetch_add(1, SeqCst);
        let v = rt.get(q);
        assert_ne!(v, 0, "c read a placeholder");
        v + 1
    });
    let r = rt.create_query_with(
        move |rt| {
            if rt.get(q) == 0 {
                rt.get(c)
            } else {
                1
            }
        },
        replayable(),
    );
    assert_eq!(rt.get(r), 1);
    assert!(runs.load(SeqCst) <= 1);
    assert_eq!(rt.get(c), 8);
    assert_eq!(runs.load(SeqCst), 1, "c ran twice in one revision");
}

local_and_shared!(
    speculative_run_never_sees_a_placeholder,
    local_speculative_run_never_sees_a_placeholder,
    shared_speculative_run_never_sees_a_placeholder
);

/// `set_max_frames(u32::MAX)` racing `set_max_stack_bytes(0)` leaves the
/// runtime unbudgeted and listed whichever runs first, so no race may
/// leave it with `DEFAULT_MAX_FRAMES`.
#[test]
#[cfg_attr(miri, ignore)]
fn racing_frame_and_byte_setters_end_as_a_serial_order() {
    let rt = Runtime::<Shared>::new();
    let go = AtomicU32::new(0);
    let done = AtomicU32::new(0);
    let rounds = 20_000_u32;
    let mut bad = 0_u32;
    std::thread::scope(|s| {
        s.spawn(|| {
            for round in 1..=rounds {
                let mut spins = 0_u32;
                while go.load(SeqCst) != round {
                    spins += 1;
                    if spins.is_multiple_of(64) {
                        std::thread::yield_now();
                    }
                    std::hint::spin_loop();
                }
                rt.set_max_stack_bytes(0);
                done.fetch_add(1, SeqCst);
            }
        });
        for round in 1..=rounds {
            rt.set_budget(|| {
                rt.frames_set.store(false, SeqCst);
                (DEFAULT_MAX_FRAMES, 0)
            });
            go.store(round, SeqCst);
            rt.set_max_frames(u32::MAX);
            while done.load(SeqCst) != round {
                std::thread::yield_now();
            }
            if rt.max_frames.load(SeqCst) != u32::MAX
                || !super::replay_impl::unbudgeted(rt.runtime_id)
            {
                bad += 1;
            }
        }
    });
    assert_eq!(
        bad, 0,
        "{bad} of {rounds} races left a budget no serial order gives"
    );
}
