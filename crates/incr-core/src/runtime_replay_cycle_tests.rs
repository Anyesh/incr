//! Replay meeting cycles and placeholder-driven reads: reads of a virtual
//! frame's node, dropped speculative chains, walk-driven discards and
//! episodes suspended by a doomed member run. Each budgeted run is
//! compared with the same graph run unbounded.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
use std::sync::{Arc, OnceLock};

use super::cycle_impl::EpisodeSeen;
use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{Cycle, CycleStrategy, QueryOpts, Replay};
use crate::handle::Incr;
use crate::state::NodeState;

macro_rules! local_and_shared {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Local>();
                }
            )*
        }
        mod shared {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Shared>();
                }
            )*
        }
    };
}

fn replayable<T>() -> QueryOpts<T> {
    QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    }
}

fn replayable_fallback<T>(g: impl Fn(&Cycle) -> T + Send + Sync + 'static) -> QueryOpts<T> {
    QueryOpts {
        cycle: CycleStrategy::fallback(g),
        replay: Replay::Allowed,
    }
}

/// A runtime at frame budget `frames`, `None` for the unbounded run, with
/// placeholders for every value type the tests read.
fn runtime<C: Cells>(frames: Option<u32>) -> Runtime<C> {
    runtime_with::<C>(frames, false)
}

/// [`runtime`] with `bool`'s placeholder `ph`.
fn runtime_with<C: Cells>(frames: Option<u32>, ph: bool) -> Runtime<C> {
    let rt = Runtime::<C>::new();
    rt.set_max_frames(frames.unwrap_or(u32::MAX));
    rt.set_placeholder(0_u64);
    rt.set_placeholder(ph);
    rt
}

/// A handle cell, so a closure can read a query created after it.
fn cell<T>() -> Arc<OnceLock<Incr<T>>> {
    Arc::new(OnceLock::new())
}

/// Closure runs, one counter per named node.
#[derive(Clone)]
struct Runs(Arc<[AtomicU32; 8]>);

impl Runs {
    fn new() -> Self {
        Self(Arc::new(Default::default()))
    }

    fn hit(&self, i: usize) {
        self.0[i].fetch_add(1, SeqCst);
    }

    fn of(&self, i: usize) -> u32 {
        self.0[i].load(SeqCst)
    }
}

/// What a read returned: its value, a cycle's participants by slot, or a
/// panic's message.
#[derive(Debug, PartialEq, Eq)]
enum Outcome<T> {
    Value(T),
    Cycle(Vec<u32>),
    Panic(String),
}

fn message(err: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = err.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        String::from("<non-string panic>")
    }
}

fn slots(cycle: &Cycle) -> Vec<u32> {
    let mut s: Vec<u32> = cycle.participants().iter().map(|p| p.slot()).collect();
    s.sort_unstable();
    s
}

fn get<C: Cells, T: crate::Value>(rt: &Runtime<C>, h: Incr<T>) -> Outcome<T> {
    match catch_unwind(AssertUnwindSafe(|| rt.get(h))) {
        Ok(v) => Outcome::Value(v),
        Err(e) => Outcome::Panic(message(e)),
    }
}

fn try_get<C: Cells, T: crate::Value>(rt: &Runtime<C>, h: Incr<T>) -> Outcome<T> {
    match catch_unwind(AssertUnwindSafe(|| rt.try_get(h))) {
        Ok(Ok(v)) => Outcome::Value(v),
        Ok(Err(c)) => Outcome::Cycle(slots(&c)),
        Err(e) => Outcome::Panic(message(e)),
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Read {
    Get,
    TryGet,
}

impl Read {
    fn of<C: Cells, T: crate::Value>(self, rt: &Runtime<C>, h: Incr<T>) -> Outcome<T> {
        match self {
            Read::Get => get(rt, h),
            Read::TryGet => try_get(rt, h),
        }
    }
}

const F: usize = 0;
const A: usize = 1;
const B: usize = 2;
const CC: usize = 3;

/// Options of the F = IF(A, B, C), C = F + 1 graph.
#[derive(Copy, Clone, Default)]
struct IfOpts {
    a: bool,
    f_fallback: bool,
    c_fallback: bool,
    b_reads_f: bool,
    /// F reads input s first and returns B when s is 0.
    recompute: bool,
}

struct IfGraph {
    f: Incr<u64>,
    b: Incr<u64>,
    c: Incr<u64>,
    a_in: Incr<bool>,
    b_in: Incr<u64>,
    s_in: Incr<u64>,
    runs: Runs,
}

fn if_graph<C: Cells>(rt: &Runtime<C>, o: IfOpts) -> IfGraph {
    let runs = Runs::new();
    let a_in = rt.create_input(o.a);
    let b_in = rt.create_input(7_u64);
    let s_in = rt.create_input(0_u64);
    let f_cell = cell::<u64>();
    let c_cell = cell::<u64>();
    let a = {
        let runs = runs.clone();
        rt.create_query_with(
            move |rt| {
                runs.hit(A);
                rt.get(a_in)
            },
            replayable(),
        )
    };
    let b = {
        let (runs, f_cell) = (runs.clone(), f_cell.clone());
        rt.create_query_with(
            move |rt| {
                runs.hit(B);
                let v = rt.get(b_in);
                if o.b_reads_f {
                    v + rt.get(*f_cell.get().unwrap())
                } else {
                    v
                }
            },
            replayable(),
        )
    };
    let f = {
        let (runs, c_cell) = (runs.clone(), c_cell.clone());
        let body = move |rt: &Runtime<C>| {
            runs.hit(F);
            if o.recompute && rt.get(s_in) == 0 {
                return rt.get(b);
            }
            if rt.get(a) {
                rt.get(b)
            } else {
                rt.get(*c_cell.get().unwrap())
            }
        };
        if o.f_fallback {
            rt.create_query_with(body, replayable_fallback(|c| 1000 + c.len() as u64))
        } else {
            rt.create_query_with(body, replayable())
        }
    };
    f_cell.set(f).unwrap();
    let c = {
        let runs = runs.clone();
        let body = move |rt: &Runtime<C>| {
            runs.hit(CC);
            rt.get(f) + 1
        };
        if o.c_fallback {
            rt.create_query_with(body, replayable_fallback(|c| 2000 + c.len() as u64))
        } else {
            rt.create_query_with(body, replayable())
        }
    };
    c_cell.set(c).unwrap();
    IfGraph {
        f,
        b,
        c,
        a_in,
        b_in,
        s_in,
        runs,
    }
}

/// **A read of a VFrame's own node** at budget 1: F reads A over budget
/// and, on A's false placeholder, pushes C as suspect; C's read of F's
/// virtual frame drops the chain, so C lands New, F is never computed above
/// C, and F lands B's value as the unbounded run does, which never runs C.
fn read_of_a_vframes_own_node<C: Cells>() {
    let rt = runtime::<C>(Some(1));
    let g = if_graph(
        &rt,
        IfOpts {
            a: true,
            ..Default::default()
        },
    );
    assert_eq!(rt.get(g.f), 7);
    assert_eq!(rt.state_of(g.c), NodeState::New);
    let runs = |g: &IfGraph| [F, CC, A, B].map(|i| g.runs.of(i));
    assert_eq!(runs(&g), [3, 1, 1, 1], "closure runs of F, C, A and B");
    assert_eq!(rt.get(g.c), 8);
    assert_eq!(runs(&g), [3, 2, 1, 1], "C's second run reads F Clean");
    rt.assert_invariant_i();

    let un = runtime::<C>(None);
    let u = if_graph(
        &un,
        IfOpts {
            a: true,
            ..Default::default()
        },
    );
    assert_eq!(un.get(u.f), 7);
    assert_eq!(un.state_of(u.c), NodeState::New);
    assert_eq!(runs(&u), [1, 0, 1, 1]);
    assert_eq!(un.get(u.c), 8);
}

/// Runs `read` of F at budget 1 and unbounded on the IF graph with
/// options `o`, and compares the outcome, the states of F and C and a
/// later `try_get` of C.
fn if_matches_unbounded<C: Cells>(o: IfOpts, read: Read) -> Outcome<u64> {
    let observe = |frames: Option<u32>| {
        let rt = runtime::<C>(frames);
        let g = if_graph(&rt, o);
        let first = read.of(&rt, g.f);
        let states = (rt.state_of(g.f), rt.state_of(g.c));
        rt.assert_invariant_i();
        let later = try_get(&rt, g.c);
        rt.assert_invariant_i();
        (first, states, later)
    };
    let unbounded = observe(None);
    assert_eq!(
        observe(Some(1)),
        unbounded,
        "{read:?} at budget 1 against the unbounded run"
    );
    unbounded.0
}

/// The a = false IF form with F's Fallback: a real cycle {F, C} rooted at
/// F, as in the unbounded run.
fn if_false_fallback_on_f<C: Cells>() {
    let o = IfOpts {
        f_fallback: true,
        ..Default::default()
    };
    for read in [Read::Get, Read::TryGet] {
        assert_eq!(if_matches_unbounded::<C>(o, read), Outcome::Value(1002));
    }
}

/// The mirror: C has the Fallback and F is unseeded, so C's read of F is
/// an R3 hit and the read panics or returns `Err(Cycle)` over {F, C}.
fn if_false_fallback_on_c<C: Cells>() {
    let o = IfOpts {
        c_fallback: true,
        ..Default::default()
    };
    let got = if_matches_unbounded::<C>(o, Read::Get);
    assert!(
        matches!(&got, Outcome::Panic(m) if m.contains("dependency cycle")),
        "{got:?}"
    );
    let got = if_matches_unbounded::<C>(o, Read::TryGet);
    assert!(matches!(&got, Outcome::Cycle(p) if p.len() == 2), "{got:?}");
}

/// a = true with B reading F and a Fallback on F: B's entry makes a
/// virtual hit on F and the episode at F's pop lands {F, B}, with F never
/// on `vindex` twice and `vindex` empty once the loop ends.
fn if_b_reads_f<C: Cells>() {
    let o = IfOpts {
        a: true,
        f_fallback: true,
        b_reads_f: true,
        ..Default::default()
    };
    let observe = |frames: Option<u32>| {
        let rt = runtime::<C>(frames);
        let g = if_graph(&rt, o);
        crate::replay::take_vindex_repushes();
        let f = get(&rt, g.f);
        assert_eq!(crate::replay::take_vindex_repushes(), 0);
        assert!(crate::replay::vindexes_empty());
        let states = (rt.state_of(g.f), rt.state_of(g.b), rt.state_of(g.c));
        rt.assert_invariant_i();
        (f, try_get(&rt, g.b), states)
    };
    let unbounded = observe(None);
    assert_eq!(unbounded.0, Outcome::Value(1002), "{unbounded:?}");
    assert_eq!(observe(Some(1)), unbounded);
}

/// F reads s first: with s = 0, `get(C)` lands F and C Clean on B. Then
/// s = 1 and a new b make F's recompute read A over budget and push C as
/// suspect, whose read of F drops it Dirty; F lands the new b, C is never
/// Clean over a Dirty F, and a later `get(C)` reads it.
fn if_recompute<C: Cells>() {
    let o = IfOpts {
        a: true,
        recompute: true,
        ..Default::default()
    };
    let observe = |frames: Option<u32>| {
        let rt = runtime::<C>(frames);
        let g = if_graph(&rt, o);
        assert_eq!(rt.get(g.c), 8);
        assert_eq!(g.runs.of(A), 0);
        rt.set(g.s_in, 1);
        rt.set(g.b_in, 20);
        assert_eq!(rt.get(g.f), 20);
        assert_eq!(rt.state_of(g.c), NodeState::Dirty);
        rt.assert_invariant_i();
        assert_eq!(rt.get(g.c), 21);
        rt.assert_invariant_i();
        let _ = g.a_in;
    };
    observe(None);
    observe(Some(1));
}

/// T reads F; F and X, both with Fallbacks, read each other once s = 1,
/// and X then reads the head of a chain D whose last node has an installed
/// edge into F from a run with t = 0.
struct TaintGraph {
    t: Incr<u64>,
    d: Vec<Incr<u64>>,
    s_in: Incr<u64>,
    t_in: Incr<u64>,
}

fn taint_graph<C: Cells>(rt: &Runtime<C>, n: usize) -> TaintGraph {
    let s_in = rt.create_input(0_u64);
    let t_in = rt.create_input(0_u64);
    let f_cell = cell::<u64>();
    let d_cell = cell::<u64>();
    let mut d = Vec::with_capacity(n);
    let mut below = {
        let f_cell = f_cell.clone();
        rt.create_query_with(
            move |rt| {
                if rt.get(t_in) == 0 {
                    rt.get(*f_cell.get().unwrap()) + 1
                } else {
                    1
                }
            },
            replayable(),
        )
    };
    d.push(below);
    for _ in 1..n {
        let next = below;
        below = rt.create_query_with(move |rt| rt.get(next) + 1, replayable());
        d.push(below);
    }
    d.reverse();
    let x = {
        let (f_cell, d_cell) = (f_cell.clone(), d_cell.clone());
        rt.create_query_with(
            move |rt| {
                if rt.get(s_in) == 0 {
                    return 5;
                }
                rt.get(*f_cell.get().unwrap()) + rt.get(*d_cell.get().unwrap())
            },
            replayable_fallback(|c| 200 + c.len() as u64),
        )
    };
    let f = rt.create_query_with(
        move |rt| rt.get(x) + 1,
        replayable_fallback(|c| 100 + c.len() as u64),
    );
    f_cell.set(f).unwrap();
    d_cell.set(d[0]).unwrap();
    let t = rt.create_query_with(move |rt| rt.get(f) * 2, replayable());
    TaintGraph { t, d, s_in, t_in }
}

/// **Taint-memo entries owned by a virtual position** at budget 1: X's
/// read of F's virtual frame claims F, and the driver entry that computes
/// the Dirty chain D walks its stale edge into F, skips the claim anchored
/// at the virtual frame and memoizes the chain. The memo is empty with its
/// bit clear once `get` returns, and the values are the unbounded run's.
fn taint_memo_owned_by_a_virtual_position<C: Cells>() {
    let observe = |frames: Option<u32>| {
        let rt = runtime::<C>(frames);
        let g = taint_graph(&rt, 5);
        assert_eq!(rt.get(g.d[0]), 11);
        rt.set(g.s_in, 1);
        rt.set(g.t_in, 1);
        let t = get(&rt, g.t);
        assert!(crate::taint_memo::is_empty());
        assert_eq!(crate::value_code::thread_word(), 0);
        rt.assert_invariant_i();
        let skips = rt.virtual_anchor_skips.load(SeqCst);
        let d = g.d.iter().map(|&n| get(&rt, n)).collect::<Vec<_>>();
        rt.set(g.s_in, 0);
        let again = get(&rt, g.t);
        rt.assert_invariant_i();
        (t, d, again, skips)
    };
    let (t, d, again, skips) = observe(None);
    assert_eq!(skips, 0);
    let (bt, bd, bagain, bskips) = observe(Some(1));
    assert!(
        bskips > 0,
        "no walk skipped a claim anchored at a virtual frame"
    );
    assert_eq!(t, Outcome::Value(204));
    assert_eq!((bt, bd, bagain), (t, d, again));
}

/// B reads s and, when s = 1, also N, a New query over k; C reads B; G
/// reads C; H reads M, a New query, and then C.
struct WalkGraph {
    s_in: Incr<u64>,
    n: Incr<u64>,
    m: Incr<u64>,
    c: Incr<u64>,
    g: Incr<u64>,
    h: Incr<u64>,
}

fn walk_graph<C: Cells>(rt: &Runtime<C>) -> WalkGraph {
    let s_in = rt.create_input(0_u64);
    let k_in = rt.create_input(3_u64);
    let m_in = rt.create_input(40_u64);
    let n = rt.create_query_with(move |rt| rt.get(k_in), replayable());
    let b = rt.create_query_with(
        move |rt| {
            if rt.get(s_in) == 1 {
                10 + rt.get(n)
            } else {
                1
            }
        },
        replayable(),
    );
    let c = rt.create_query_with(move |rt| rt.get(b) * 2, replayable());
    let g = rt.create_query_with(move |rt| rt.get(c) + 100, replayable());
    let m = rt.create_query_with(move |rt| rt.get(m_in), replayable());
    let h = rt.create_query_with(move |rt| rt.get(m) + rt.get(c), replayable());
    WalkGraph {
        s_in,
        n,
        m,
        c,
        g,
        h,
    }
}

/// **A walk-driven discard** at budget 1: `get(C)`'s walk computes B, which
/// dooms on its read of N, so the walk runs the top-level context's loop
/// with B as its target and then computes C from its preserved `work`, with
/// the unbounded run's 3 expansions: a restarted walk would add C again.
impl WalkGraph {
    /// The nodes released by every doom so far, by name.
    fn dooms<C: Cells>(&self, rt: &Runtime<C>) -> String {
        let dooms = rt.dooms.lock().unwrap();
        dooms
            .iter()
            .map(|d| match d.node {
                s if s == self.n.slot() => 'N',
                s if s == self.m.slot() => 'M',
                _ => '?',
            })
            .collect()
    }
}

fn walk_driven_discard<C: Cells>() {
    let observe = |frames: Option<u32>| {
        let rt = runtime::<C>(frames);
        let w = walk_graph(&rt);
        assert_eq!(rt.get(w.c), 2);
        rt.set(w.s_in, 1);
        rt.dooms.lock().unwrap().clear();
        let before = rt.expansions();
        let c = get(&rt, w.c);
        rt.assert_invariant_i();
        (c, rt.expansions() - before, w.dooms(&rt))
    };
    let unbounded = observe(None);
    assert_eq!(unbounded, (Outcome::Value(26), 3, String::new()));
    // The plan counts 4, with B expanded again as the loop's target; the
    // driver instead runs that target as the walk's own compute of B, whose
    // deps the walk already cleaned, as the unbounded run does.
    assert_eq!(observe(Some(1)), (unbounded.0, unbounded.1, "N".into()));
}

/// G, New, reads C at budget 2, so B's doom marks G, whose read of C returns the
/// placeholder and abandons the walk; G's retry lands the unbounded value.
fn walk_driven_discard_marks_the_reader<C: Cells>() {
    let observe = |frames: Option<u32>| {
        let rt = runtime::<C>(frames);
        let w = walk_graph(&rt);
        assert_eq!(rt.get(w.c), 2);
        rt.set(w.s_in, 1);
        rt.dooms.lock().unwrap().clear();
        let g = get(&rt, w.g);
        rt.assert_invariant_i();
        (g, w.dooms(&rt))
    };
    assert_eq!(observe(None), (Outcome::Value(126), String::new()));
    assert_eq!(observe(Some(2)), (Outcome::Value(126), "N".into()));
}

/// H's doomed run pushes C as a driver entry, whose own walk meets B's
/// discard: the entry is kept, and its retry lands C.
fn walk_driven_discard_in_an_entry<C: Cells>() {
    let observe = |frames: Option<u32>| {
        let rt = runtime::<C>(frames);
        let w = walk_graph(&rt);
        assert_eq!(rt.get(w.c), 2);
        rt.set(w.s_in, 1);
        rt.dooms.lock().unwrap().clear();
        let h = get(&rt, w.h);
        rt.assert_invariant_i();
        (h, rt.state_of(w.c), w.dooms(&rt))
    };
    let (h, c, dooms) = observe(None);
    assert_eq!(
        (h, c, dooms),
        (Outcome::Value(66), NodeState::Clean, String::new())
    );
    assert_eq!(
        observe(Some(1)),
        (Outcome::Value(66), NodeState::Clean, "MN".into())
    );
}

/// The episodes started so far.
fn episode_starts<C: Cells>(rt: &Runtime<C>) -> usize {
    rt.episodes_seen
        .lock()
        .unwrap()
        .iter()
        .filter(|e| matches!(e, EpisodeSeen::Start(_)))
        .count()
}

/// **A virtual hit above an earlier one** at budget 1: F and X have
/// Fallbacks, F reads X, X reads Y, and Y, unseeded, reads X and then F.
/// Y's hit at X's virtual frame claims X, its hit at F's claims F only,
/// and one episode over {F, X, Y} lands the unbounded run's values.
fn virtual_hit_above_an_earlier_one<C: Cells>() {
    let observe = |frames: Option<u32>| {
        let rt = runtime::<C>(frames);
        let y_cell = cell::<u64>();
        let x = {
            let y_cell = y_cell.clone();
            rt.create_query_with(
                move |rt| rt.get(*y_cell.get().unwrap()) + 1,
                replayable_fallback(|c| 200 + c.len() as u64),
            )
        };
        let f = rt.create_query_with(
            move |rt| rt.get(x) + 1,
            replayable_fallback(|c| 100 + c.len() as u64),
        );
        let y = rt.create_query_with(move |rt| rt.get(x) + rt.get(f), replayable());
        y_cell.set(y).unwrap();
        let got = get(&rt, f);
        rt.assert_invariant_i();
        let states = [f, x, y].map(|n| rt.state_of(n));
        let dooms: Vec<u32> = rt.dooms.lock().unwrap().iter().map(|d| d.node).collect();
        assert!(
            frames.is_none() || dooms == vec![x.slot(), y.slot()],
            "{dooms:?}"
        );
        let values = [x, y].map(|n| try_get(&rt, n));
        (got, values, states, episode_starts(&rt))
    };
    let unbounded = observe(None);
    assert_eq!(unbounded.0, Outcome::Value(103), "{unbounded:?}");
    assert_eq!(unbounded.3, 1);
    assert_eq!(observe(Some(1)), unbounded);
}

/// **A drop through a frame that can't take a placeholder**: the a = false
/// IF form with C reading F through Q, a native query whose type has no
/// placeholder, so C's suspect read drops the chain by unwinding through
/// Q. The outcome is the unbounded run's, under both strategy placements,
/// and nothing lands Failed.
fn drop_through_a_native_frame<C: Cells>() {
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Bare(u64);

    for (f_fallback, c_fallback) in [(true, false), (false, true)] {
        for read in [Read::Get, Read::TryGet] {
            let observe = |frames: Option<u32>| {
                let rt = runtime::<C>(frames);
                let a_in = rt.create_input(false);
                let b_in = rt.create_input(7_u64);
                let a = rt.create_query_with(move |rt| rt.get(a_in), replayable());
                let b = rt.create_query_with(move |rt| rt.get(b_in), replayable());
                let c_cell = cell::<u64>();
                let body = {
                    let c_cell = c_cell.clone();
                    move |rt: &Runtime<C>| {
                        if rt.get(a) {
                            rt.get(b)
                        } else {
                            rt.get(*c_cell.get().unwrap())
                        }
                    }
                };
                let f = if f_fallback {
                    rt.create_query_with(body, replayable_fallback(|c| 1000 + c.len() as u64))
                } else {
                    rt.create_query_with(body, replayable())
                };
                let q = rt.create_query(move |rt| Bare(rt.get(f)));
                let body = move |rt: &Runtime<C>| rt.get(q).0 + 1;
                let c = if c_fallback {
                    rt.create_query_with(body, replayable_fallback(|c| 2000 + c.len() as u64))
                } else {
                    rt.create_query_with(body, replayable())
                };
                c_cell.set(c).unwrap();
                let got = read.of(&rt, f);
                let dooms = rt.dooms.lock().unwrap().first().map(|d| d.node);
                assert!(frames.is_none() || dooms == Some(a.slot()), "{dooms:?}");
                rt.assert_invariant_i();
                let states = [f, c].map(|n| rt.state_of(n));
                let q_state = rt.state_of(q);
                for s in states.iter().chain([&q_state]) {
                    assert_ne!(*s, NodeState::Failed);
                }
                (got, states, q_state, try_get(&rt, c))
            };
            let unbounded = observe(None);
            assert_eq!(
                observe(Some(1)),
                unbounded,
                "{read:?}, Fallback on F {f_fallback}, on C {c_fallback}"
            );
        }
    }
}

/// **A suspended episode** at budget 1: in a 3-member Fixpoint cycle M1 ->
/// M2 -> M3 -> M1, M2 reads a New query N from iteration 2 on, so its member
/// run dooms there. The episode suspends, N's entry drains, and the pop of
/// M2's virtual frame resumes the same episode in iteration 2 with the
/// same revisions and re-runs M2; no second episode starts, and the values
/// and iteration count are the unbounded run's.
fn suspended_episode<C: Cells>() {
    let observe = |frames: Option<u32>| {
        let rt = runtime::<C>(frames);
        let k_max = Arc::new(AtomicU32::new(0));
        let fixpoint = |k_max: &Arc<AtomicU32>| {
            let k_max = k_max.clone();
            QueryOpts {
                cycle: CycleStrategy::Fixpoint(crate::cycle::Fixpoint::new(|| 0_u64).step(
                    move |_, out, k| {
                        k_max.fetch_max(k, SeqCst);
                        out
                    },
                )),
                replay: Replay::Allowed,
            }
        };
        let one = rt.create_input(1_u64);
        let n = rt.create_query_with(move |rt| rt.get(one), replayable());
        let m1_cell = cell::<u64>();
        let m3 = {
            let m1_cell = m1_cell.clone();
            rt.create_query_with(
                move |rt| (rt.get(*m1_cell.get().unwrap()) + 1).min(3),
                fixpoint(&k_max),
            )
        };
        let m2 = rt.create_query_with(
            move |rt| {
                let v = rt.get(m3);
                if v >= 1 {
                    (v + rt.get(n)).min(3)
                } else {
                    v + 1
                }
            },
            fixpoint(&k_max),
        );
        let m1 = rt.create_query_with(move |rt| (rt.get(m2) + 1).min(3), fixpoint(&k_max));
        m1_cell.set(m1).unwrap();
        let got = get(&rt, m1);
        rt.assert_invariant_i();
        let values = [m2, m3].map(|m| try_get(&rt, m));
        let seen = rt.episodes_seen.lock().unwrap().clone();
        ((got, values, k_max.load(SeqCst)), seen, m2.slot())
    };
    let (unbounded, seen, _) = observe(None);
    assert_eq!(
        seen.iter()
            .filter(|e| matches!(e, EpisodeSeen::Start(_)))
            .count(),
        1
    );
    let (bounded, seen, m2) = observe(Some(1));
    assert_eq!(bounded, unbounded);
    let starts: Vec<u64> = seen
        .iter()
        .filter_map(|e| match e {
            EpisodeSeen::Start(id) => Some(*id),
            _ => None,
        })
        .collect();
    assert_eq!(starts.len(), 1, "{seen:?}");
    let id = starts[0];
    let suspend = seen
        .iter()
        .position(|e| matches!(e, EpisodeSeen::Suspend { .. }))
        .unwrap_or_else(|| panic!("no member run was suspended: {seen:?}"));
    let EpisodeSeen::Suspend { id: s_id, k, revs } = seen[suspend].clone() else {
        unreachable!()
    };
    assert_eq!((s_id, k), (id, 2), "{seen:?}");
    assert!(
        matches!(seen[suspend - 1], EpisodeSeen::Member { id: Some(e), slot } if e == id && slot == m2),
        "the suspended run is M2's: {seen:?}"
    );
    assert_eq!(
        seen[suspend + 1],
        EpisodeSeen::Resume {
            id,
            k: 2,
            revs,
            queued_above: 0
        },
        "{seen:?}"
    );
    assert_eq!(
        seen[suspend + 2],
        EpisodeSeen::Member {
            id: Some(id),
            slot: m2
        },
        "{seen:?}"
    );
}

/// **The marked-parent IF test** at budget 2: P = IF(F, X, Y), Y = P + 1,
/// F reading A, a New query over a true input with placeholder false. F's
/// read of A dooms, P is marked and takes F's placeholder, and its read of
/// Y is suspect, so Y's read of P drops the chain: P returns X's value and
/// Y stays New, as in the unbounded run.
#[test]
fn marked_parent_if_at_budget_2() {
    let observe = |frames: Option<u32>| {
        let rt = runtime::<Local>(frames);
        let a_in = rt.create_input(true);
        let x_in = rt.create_input(7_u64);
        let a = rt.create_query_with(move |rt| rt.get(a_in), replayable());
        let f = rt.create_query_with(move |rt| rt.get(a), replayable());
        let x = rt.create_query_with(move |rt| rt.get(x_in), replayable());
        let y_cell = cell::<u64>();
        let p = {
            let y_cell = y_cell.clone();
            rt.create_query_with(
                move |rt| {
                    if rt.get(f) {
                        rt.get(x)
                    } else {
                        rt.get(*y_cell.get().unwrap())
                    }
                },
                replayable(),
            )
        };
        let y = rt.create_query_with(move |rt| rt.get(p) + 1, replayable());
        y_cell.set(y).unwrap();
        let got = get(&rt, p);
        let dooms: Vec<u32> = rt.dooms.lock().unwrap().iter().map(|d| d.node).collect();
        assert!(frames.is_none() || dooms == vec![a.slot()], "{dooms:?}");
        rt.assert_invariant_i();
        (got, rt.state_of(y), get(&rt, y))
    };
    let unbounded = observe(None);
    assert_eq!(
        unbounded,
        (Outcome::Value(7), NodeState::New, Outcome::Value(8))
    );
    assert_eq!(observe(Some(2)), unbounded);
}

/// **A speculative root** at budget 1: F = IF(A, S, U) with A's placeholder
/// true and a = false, S a Fallback member and U unseeded, reading each
/// other. The speculative entry S never runs an episode; the real run
/// enters at U, so `get(F)` panics with the cycle message and `try_get(F)`
/// returns `Err(Cycle)` over {S, U}, with S and U New afterwards.
#[test]
fn speculative_root_never_runs_an_episode() {
    for read in [Read::Get, Read::TryGet] {
        let observe = |frames: Option<u32>| {
            let rt = runtime_with::<Local>(frames, true);
            let a_in = rt.create_input(false);
            let a = rt.create_query_with(move |rt| rt.get(a_in), replayable());
            let u_cell = cell::<u64>();
            let s = {
                let u_cell = u_cell.clone();
                rt.create_query_with(
                    move |rt| rt.get(*u_cell.get().unwrap()) + 1,
                    replayable_fallback(|c| 300 + c.len() as u64),
                )
            };
            let u = rt.create_query_with(move |rt| rt.get(s) + 1, replayable());
            u_cell.set(u).unwrap();
            let f = rt.create_query_with(
                move |rt| if rt.get(a) { rt.get(s) } else { rt.get(u) },
                replayable(),
            );
            let got = read.of(&rt, f);
            rt.assert_invariant_i();
            (got, rt.state_of(s), rt.state_of(u))
        };
        let unbounded = observe(None);
        match read {
            Read::Get => assert!(
                matches!(&unbounded.0, Outcome::Panic(m) if m.contains("dependency cycle")),
                "{unbounded:?}"
            ),
            Read::TryGet => {
                assert!(
                    matches!(&unbounded.0, Outcome::Cycle(p) if p.len() == 2),
                    "{unbounded:?}"
                )
            }
        }
        assert_eq!((unbounded.1, unbounded.2), (NodeState::New, NodeState::New));
        assert_eq!(observe(Some(1)), unbounded, "{read:?}");
    }
}

/// **A cooperative drop** at budget 1: F = IF(A, S, U) with A's
/// placeholder true and a = false, S and U Fallback members reading each
/// other, every query replayable and every read type with a placeholder.
/// The speculative chain from S drops with no engine unwind, which
/// `panic = abort` would turn into an abort, and F, S and U read as in the
/// unbounded run.
fn speculative_chain_drops_without_unwinding<C: Cells>() {
    let observe = |frames: Option<u32>| {
        let rt = runtime_with::<C>(frames, true);
        let a_in = rt.create_input(false);
        let a = rt.create_query_with(move |rt| rt.get(a_in), replayable());
        let u_cell = cell::<u64>();
        let s = {
            let u_cell = u_cell.clone();
            rt.create_query_with(
                move |rt| rt.get(*u_cell.get().unwrap()) + 1,
                replayable_fallback(|_| 100),
            )
        };
        let u = rt.create_query_with(move |rt| rt.get(s) + 10, replayable_fallback(|_| 200));
        u_cell.set(u).unwrap();
        let f = rt.create_query_with(
            move |rt| if rt.get(a) { rt.get(s) } else { rt.get(u) },
            replayable(),
        );
        let got = (rt.get(f), rt.get(s), rt.get(u));
        rt.assert_invariant_i();
        (
            got,
            rt.drop_unwinds.load(SeqCst),
            rt.cooperative_drops.load(SeqCst),
        )
    };
    let (unbounded, _, _) = observe(None);
    let (got, unwinds, drops) = observe(Some(1));
    assert_eq!(got, unbounded);
    assert_eq!(unwinds, 0, "a drop unwind was raised");
    assert!(drops > 0, "no chain was dropped");
}

local_and_shared!(
    speculative_chain_drops_without_unwinding,
    virtual_hit_above_an_earlier_one,
    drop_through_a_native_frame,
    suspended_episode,
    walk_driven_discard,
    walk_driven_discard_marks_the_reader,
    walk_driven_discard_in_an_entry,
    read_of_a_vframes_own_node,
    taint_memo_owned_by_a_virtual_position,
    if_false_fallback_on_f,
    if_false_fallback_on_c,
    if_b_reads_f,
    if_recompute,
);
