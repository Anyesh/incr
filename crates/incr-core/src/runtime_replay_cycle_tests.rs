//! Replay meeting cycles and placeholder-driven reads: reads of a virtual
//! frame's node, dropped speculative chains, walk-driven discards and
//! episodes suspended by a doomed member run. Each budgeted run is
//! compared with the same graph run unbounded.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
use std::sync::{Arc, OnceLock};

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
    let rt = Runtime::<C>::new();
    rt.set_max_frames(frames.unwrap_or(u32::MAX));
    rt.set_placeholder(0_u64);
    rt.set_placeholder(false);
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

local_and_shared!(
    read_of_a_vframes_own_node,
    if_false_fallback_on_f,
    if_false_fallback_on_c,
    if_b_reads_f,
    if_recompute,
);
