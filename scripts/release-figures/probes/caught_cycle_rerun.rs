//! The 0x1daa ladder of gh-cmd 54cebca1-c1eb3f: the generated replay
//! differential case 0x1daa, with `levels` queries between its cycle
//! members 10 and 5 that each read the next three times, and op 4 turned
//! into a `set` of the input right after the levels. It counts the closure
//! runs of op 6, the read whose walk enters the caught cycle below the
//! catching query, without a frame budget. The case is spelled out here
//! from the harness's `seed_1daa_with_levels`, so the probe uses the
//! public API only.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};

use incr_core::{Cells, CycleStrategy, Fixpoint, Incr, Local, QueryOpts, Replay, Runtime, Shared};

static RUNS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
enum Expr {
    Const(u64),
    Read(usize, bool),
    Add(Box<Expr>, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    PanicIf(Box<Expr>, u64),
    Catch(Box<Expr>),
}

#[derive(Clone, Copy, Debug)]
enum Strat {
    None,
    Fallback(u64),
    Fixpoint { initial: u64, limit: u32 },
}

#[derive(Clone, Debug)]
enum Spec {
    Input(u64),
    Query {
        expr: Expr,
        replay: bool,
        bare: bool,
        strategy: Strat,
    },
}

#[derive(Clone, Debug)]
enum Op {
    Set(usize, u64),
    Get(usize),
    TryGet(usize),
}

#[derive(Clone, PartialEq)]
struct Bare(u64);

#[derive(Clone, Copy)]
enum Handle {
    Plain(Incr<u64>),
    Bare(Incr<Bare>),
}

type Table = Arc<OnceLock<Vec<Handle>>>;

fn get<C: Cells>(rt: &Runtime<C>, h: Handle) -> u64 {
    match h {
        Handle::Plain(h) => rt.get(h),
        Handle::Bare(h) => rt.get(h).0,
    }
}

fn try_get<C: Cells>(rt: &Runtime<C>, h: Handle) -> Option<u64> {
    match h {
        Handle::Plain(h) => rt.try_get(h).ok(),
        Handle::Bare(h) => rt.try_get(h).ok().map(|b| b.0),
    }
}

fn eval<C: Cells>(rt: &Runtime<C>, table: &Table, at: usize, e: &Expr) -> u64 {
    let ev = |e: &Expr| eval(rt, table, at, e);
    match e {
        Expr::Const(v) => *v,
        Expr::Read(target, try_) => {
            let h = table.get().expect("built")[*target];
            if *try_ {
                try_get(rt, h).unwrap_or(7777)
            } else {
                get(rt, h)
            }
        }
        Expr::Add(a, b) => ev(a).wrapping_add(ev(b)),
        Expr::If(c, a, b) => {
            if ev(c) % 2 == 0 {
                ev(a)
            } else {
                ev(b)
            }
        }
        Expr::PanicIf(c, when) => {
            let v = ev(c);
            if v % 7 == *when {
                std::panic::panic_any(format!("boom 0:{at}"));
            }
            v
        }
        Expr::Catch(inner) => catch_unwind(AssertUnwindSafe(|| ev(inner))).unwrap_or(9999),
    }
}

fn strategy<T: Clone + PartialEq + Send + Sync + 'static>(
    s: Strat,
    wrap: fn(u64) -> T,
) -> CycleStrategy<T> {
    match s {
        Strat::None => CycleStrategy::None,
        Strat::Fallback(v) => CycleStrategy::fallback(move |_| wrap(v)),
        Strat::Fixpoint { initial, limit } => {
            CycleStrategy::Fixpoint(Fixpoint::new(move || wrap(initial)).max_iterations(limit))
        }
    }
}

fn make<C: Cells>(rt: &Runtime<C>, table: &Table, i: usize, spec: &Spec) -> Handle {
    match spec {
        Spec::Input(v) => Handle::Plain(rt.create_input(*v)),
        Spec::Query {
            expr,
            replay,
            bare,
            strategy: s,
        } => {
            let replay = if *replay {
                Replay::Allowed
            } else {
                Replay::Native
            };
            let (expr, table) = (expr.clone(), table.clone());
            let compute = move |rt: &Runtime<C>| {
                RUNS.fetch_add(1, Relaxed);
                eval(rt, &table, i, &expr)
            };
            if *bare {
                Handle::Bare(rt.create_query_with(
                    move |rt| Bare(compute(rt)),
                    QueryOpts {
                        cycle: strategy(*s, Bare),
                        replay,
                    },
                ))
            } else {
                Handle::Plain(rt.create_query_with(
                    compute,
                    QueryOpts {
                        cycle: strategy(*s, |v| v),
                        replay,
                    },
                ))
            }
        }
    }
}

fn apply<C: Cells>(rt: &Runtime<C>, table: &Table, op: &Op) {
    let h = |i: usize| table.get().expect("built")[i];
    let _ = catch_unwind(AssertUnwindSafe(|| match op {
        Op::Set(i, v) => match h(*i) {
            Handle::Plain(h) => rt.set(h, *v),
            Handle::Bare(_) => unreachable!("inputs are plain"),
        },
        Op::Get(i) => drop(get(rt, h(*i))),
        Op::TryGet(i) => drop(try_get(rt, h(*i))),
    }));
}

/// The closure runs of op `op` after the ops before it, nodes made in
/// reverse as the harness makes them.
fn op_runs<C: Cells>(specs: &[Spec], ops: &[Op], op: usize) -> u64 {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    let table: Table = Arc::new(OnceLock::new());
    let mut made: Vec<Option<Handle>> = vec![None; specs.len()];
    for (i, spec) in specs.iter().enumerate().rev() {
        made[i] = Some(make(&rt, &table, i, spec));
    }
    let _ = table.set(made.into_iter().map(|h| h.expect("made")).collect());
    for o in &ops[..op] {
        apply(&rt, &table, o);
    }
    let before = RUNS.load(Relaxed);
    apply(&rt, &table, &ops[op]);
    RUNS.load(Relaxed) - before
}

fn map_reads(e: &Expr, f: &impl Fn(usize) -> usize) -> Expr {
    let m = |x: &Expr| Box::new(map_reads(x, f));
    match e {
        Expr::Const(v) => Expr::Const(*v),
        Expr::Read(i, t) => Expr::Read(f(*i), *t),
        Expr::Add(a, b) => Expr::Add(m(a), m(b)),
        Expr::If(a, b, c) => Expr::If(m(a), m(b), m(c)),
        Expr::PanicIf(a, w) => Expr::PanicIf(m(a), *w),
        Expr::Catch(a) => Expr::Catch(m(a)),
    }
}

/// `seed_1daa_with_levels` of the harness, with op 4 made the set of node
/// `16 + levels` as in the ladder.
fn ladder_case(levels: usize) -> (Vec<Spec>, Vec<Op>) {
    let (base, ops, q) = case_1daa();
    let shift = |i: usize| if i >= q { i + levels } else { i };
    let mut specs: Vec<Spec> = base
        .iter()
        .map(|s| match s {
            Spec::Query {
                expr,
                replay,
                bare,
                strategy,
            } => Spec::Query {
                expr: map_reads(expr, &shift),
                replay: *replay,
                bare: *bare,
                strategy: *strategy,
            },
            other => other.clone(),
        })
        .collect();
    if levels > 0 {
        if let Spec::Query { expr, .. } = &mut specs[10] {
            *expr = map_reads(expr, &|i| if i == 5 { q } else { i });
        }
        let level = |j: usize| {
            let next = if j + 1 == levels { 5 } else { q + j + 1 };
            let read = || Box::new(Expr::Read(next, false));
            Spec::Query {
                expr: Expr::Add(Box::new(Expr::Add(read(), read())), read()),
                replay: true,
                bare: false,
                strategy: Strat::None,
            }
        };
        specs.splice(q..q, (0..levels).map(level));
    }
    let mut ops: Vec<Op> = ops
        .iter()
        .map(|op| match op {
            Op::Set(n, v) => Op::Set(shift(*n), *v),
            other => other.clone(),
        })
        .collect();
    ops[4] = Op::Set(16 + levels, 12345);
    (specs, ops)
}

#[test]
fn caught_cycle_rerun() {
    let tree = std::env::var("FIGURE_TREE").unwrap_or_default();
    std::panic::set_hook(Box::new(|_| {}));
    for levels in [0, 2, 4, 6] {
        let (specs, ops) = ladder_case(levels);
        let local = op_runs::<Local>(&specs, &ops, 6);
        let shared = op_runs::<Shared>(&specs, &ops, 6);
        println!("FIG {tree} caught_cycle_rerun levels={levels} op6_runs_local={local} op6_runs_shared={shared}");
    }
}

fn read(i: usize, try_: bool) -> Expr {
    Expr::Read(i, try_)
}

fn konst(v: u64) -> Expr {
    Expr::Const(v)
}

fn add(a: Expr, b: Expr) -> Expr {
    Expr::Add(Box::new(a), Box::new(b))
}

fn cond(c: Expr, a: Expr, b: Expr) -> Expr {
    Expr::If(Box::new(c), Box::new(a), Box::new(b))
}

fn panic_if(a: Expr, when: u64) -> Expr {
    Expr::PanicIf(Box::new(a), when)
}

fn catch(a: Expr) -> Expr {
    Expr::Catch(Box::new(a))
}

fn query(expr: Expr, replay: bool, bare: bool, strategy: Strat) -> Spec {
    Spec::Query {
        expr,
        replay,
        bare,
        strategy,
    }
}

/// `gen_case(0x1daa, false, false)` of the replay differential harness:
/// its specs, its ops and its query count.
fn case_1daa() -> (Vec<Spec>, Vec<Op>, usize) {
    let specs = vec![
        query(
            read(0, false),
            true,
            false,
            Strat::Fixpoint {
                initial: 1,
                limit: 2,
            },
        ),
        query(
            read(0, false),
            true,
            false,
            Strat::Fixpoint {
                initial: 4,
                limit: 1,
            },
        ),
        query(catch(read(3, false)), false, true, Strat::None),
        query(
            add(
                panic_if(read(4, false), 6),
                add(read(4, false), panic_if(read(18, true), 5)),
            ),
            true,
            false,
            Strat::None,
        ),
        query(konst(4), true, false, Strat::None),
        query(
            catch(cond(
                add(read(6, false), read(6, false)),
                add(read(6, false), read(7, false)),
                add(read(6, false), read(6, false)),
            )),
            true,
            false,
            Strat::Fallback(3),
        ),
        query(
            add(add(read(7, false), read(5, false)), read(7, true)),
            false,
            false,
            Strat::None,
        ),
        query(
            add(
                panic_if(cond(read(4, false), read(1, false), read(18, false)), 2),
                add(
                    add(read(8, false), read(8, false)),
                    cond(read(12, false), read(8, false), read(16, false)),
                ),
            ),
            true,
            false,
            Strat::None,
        ),
        query(
            cond(
                add(
                    cond(read(16, false), read(15, false), read(9, false)),
                    panic_if(read(9, false), 4),
                ),
                add(konst(2), add(read(9, false), read(14, false))),
                cond(
                    cond(read(9, false), read(9, false), read(9, false)),
                    add(read(12, true), read(9, false)),
                    read(4, true),
                ),
            ),
            false,
            false,
            Strat::None,
        ),
        query(
            cond(
                cond(
                    add(read(10, false), read(17, false)),
                    add(read(18, true), read(15, false)),
                    add(read(10, false), read(10, false)),
                ),
                read(10, false),
                cond(
                    add(read(10, false), read(15, true)),
                    panic_if(read(10, false), 3),
                    catch(read(11, false)),
                ),
            ),
            true,
            false,
            Strat::Fallback(4),
        ),
        query(
            add(
                konst(3),
                cond(
                    panic_if(read(11, false), 3),
                    read(18, false),
                    catch(read(5, false)),
                ),
            ),
            true,
            false,
            Strat::None,
        ),
        query(
            add(
                read(14, false),
                panic_if(add(read(12, false), read(18, false)), 3),
            ),
            false,
            false,
            Strat::Fixpoint {
                initial: 2,
                limit: 2,
            },
        ),
        query(
            cond(
                catch(read(18, false)),
                catch(read(13, true)),
                cond(
                    cond(read(13, true), read(18, false), read(13, false)),
                    read(13, false),
                    read(13, false),
                ),
            ),
            true,
            true,
            Strat::None,
        ),
        query(
            add(
                cond(
                    konst(0),
                    read(14, false),
                    add(read(0, false), read(1, false)),
                ),
                cond(
                    konst(0),
                    cond(read(14, false), read(14, false), read(14, false)),
                    add(read(16, true), read(15, false)),
                ),
            ),
            true,
            false,
            Strat::Fallback(0),
        ),
        query(konst(2), true, false, Strat::None),
        query(read(16, true), true, false, Strat::None),
        Spec::Input(4),
        Spec::Input(2),
        Spec::Input(1),
    ];
    let ops = vec![
        Op::Get(0),
        Op::Get(1),
        Op::TryGet(6),
        Op::Get(7),
        Op::Set(17, 0),
        Op::Get(2),
        Op::TryGet(9),
        Op::Set(17, 3),
        Op::Set(18, 0),
    ];
    (specs, ops, 16)
}
