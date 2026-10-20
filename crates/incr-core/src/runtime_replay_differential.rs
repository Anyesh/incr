//! `replay_differential`: random graphs of replayable queries, each run at
//! frame budgets 1, 2 and 3, with and without a byte budget, against the
//! same graph and operations run unbounded. Every `get` and `try_get`
//! result (the value, a cycle's participants, or the panic's class or
//! payload), every delete's result, and the sets of Failed and Cyclic
//! nodes after each operation must be the unbounded run's, while every
//! query's state agrees with the unbounded run's before the operation.
//!
//! A case may also delete nodes and recreate them into the freed slots,
//! keep a second runtime whose queries read and are read by the first's,
//! and include a root whose cycle dissolves at validation. Under Shared,
//! a second form steps two threads through barriers: one thread makes
//! each set, delete or recreate while the other waits, and reads overlap
//! only reads, so each result is still the unbounded run's at its
//! sequence point.
//!
//! A speculative entry can land a node the unbounded run leaves New, and
//! with panics the engine's outcomes depend on that history: a walk
//! computes a Dirty node's deps before its closure, so a dep's panic
//! leaves the walk where the closure would have caught it, and a Failed
//! dep panics at the walk where a New node's closure would catch its
//! read. So once the states differ, a case with panics stops comparing,
//! and a case without them, a third of the seeds, compares the results to
//! the end.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::{Arc, Barrier, Mutex};

use proptest::prelude::*;

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{CycleStrategy, Fixpoint, QueryOpts, Replay};
use crate::handle::{ErasedIncr, Incr};
use crate::state::NodeState;
use crate::Value;

/// A value type with no placeholder: reads of it always nest natively.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Bare(u64);

#[derive(Clone, Debug)]
enum Expr {
    Const(u64),
    /// A read of node `.0`, with `try_get` when `.1`.
    Read(usize, bool),
    /// A read of the other runtime's node `.0`, with `try_get` when `.1`.
    ReadOther(usize, bool),
    Add(Box<Expr>, Box<Expr>),
    /// The second operand when the first is even, else the third.
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    /// Panics with the node's payload when the operand is `.1` modulo 7.
    PanicIf(Box<Expr>, u64),
    /// The operand, or 9999 if it unwinds.
    Catch(Box<Expr>),
    /// The first operand on the closure's first run in its runtime, a
    /// doomed run included, else the second.
    FirstRun(Box<Expr>, Box<Expr>),
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

#[derive(Copy, Clone, Debug)]
enum Strat {
    None,
    Fallback(u64),
    Fixpoint { initial: u64, limit: u32 },
}

#[derive(Clone, Debug)]
enum Op {
    Set(usize, u64),
    Get(usize),
    TryGet(usize),
    /// `try_delete_node` of node `.0`'s current handle.
    Delete(usize),
    /// A node built from `.1` in place of node `.0`, if its current handle
    /// is deleted. The replaced handle is kept for `ReadOld`.
    Recreate(usize, Spec),
    /// A read of the `.0`-th replaced handle, modulo their count, with
    /// `try_get` when `.1`.
    ReadOld(usize, bool),
    /// A set of the second runtime's input `.0`.
    Set2(usize, u64),
    /// A read of the second runtime's query `.0`, with `try_get` when `.1`.
    Get2(usize, bool),
}

impl Op {
    fn is_read(&self) -> bool {
        matches!(
            self,
            Op::Get(_) | Op::TryGet(_) | Op::ReadOld(..) | Op::Get2(..)
        )
    }
}

/// splitmix64, so a case is its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

/// Queries are nodes `0..queries`, each reading only deeper queries or
/// inputs, and the inputs follow them. The second runtime's nodes are
/// laid out the same way in `specs2`; its first `lower` queries read
/// only their own runtime, and only they and its inputs are read by the
/// first runtime.
struct Case {
    specs: Vec<Spec>,
    panics: bool,
    cycles: bool,
    queries: usize,
    specs2: Vec<Spec>,
    queries2: usize,
    ops: Vec<Op>,
}

fn gen_expr(rng: &mut Rng, depth: u32, read: &impl Fn(&mut Rng) -> Expr) -> Expr {
    if depth == 0 {
        return read(rng);
    }
    match rng.below(100) {
        0..=7 => Expr::Const(rng.below(5)),
        8..=44 => read(rng),
        45..=64 => Expr::Add(
            Box::new(gen_expr(rng, depth - 1, read)),
            Box::new(gen_expr(rng, depth - 1, read)),
        ),
        65..=84 => Expr::If(
            Box::new(gen_expr(rng, depth - 1, read)),
            Box::new(gen_expr(rng, depth - 1, read)),
            Box::new(gen_expr(rng, depth - 1, read)),
        ),
        85..=91 => Expr::PanicIf(Box::new(gen_expr(rng, depth - 1, read)), rng.below(7)),
        _ => Expr::Catch(Box::new(gen_expr(rng, depth - 1, read))),
    }
}

/// Reads for query `at` of the first runtime.
fn reads(at: usize, queries: usize, nodes: usize, cycles: bool) -> impl Fn(&mut Rng) -> Expr {
    move |rng: &mut Rng| {
        let target = if cycles && rng.chance(12) {
            rng.below(at as u64 + 1) as usize
        } else if at + 1 < queries && rng.chance(55) {
            at + 1
        } else if at + 1 < queries && rng.chance(50) {
            at + 1 + rng.below((queries - at - 1) as u64) as usize
        } else {
            queries + rng.below((nodes - queries) as u64) as usize
        };
        Expr::Read(target, rng.chance(25))
    }
}

fn gen_strategy(rng: &mut Rng, cycles: bool) -> Strat {
    if !cycles {
        return Strat::None;
    }
    match rng.below(4) {
        0 | 1 => Strat::None,
        2 => Strat::Fallback(rng.below(5)),
        _ => Strat::Fixpoint {
            initial: rng.below(5),
            limit: 1 + rng.below(3) as u32,
        },
    }
}

/// The case the first version of this test generated for `seed`: one
/// runtime, sets and reads. The scans' seeds name these cases.
fn gen_base(seed: u64) -> Case {
    let mut rng = Rng(seed);
    let cycles = (seed / 3) % 2 == 1;
    let queries = 2 + rng.below(24) as usize;
    let inputs = 1 + rng.below(4) as usize;
    let nodes = queries + inputs;
    let mut specs = Vec::with_capacity(nodes);
    for at in 0..queries {
        let mut expr = gen_expr(&mut rng, 3, &reads(at, queries, nodes, cycles));
        if seed.is_multiple_of(3) {
            expr = without_panics(expr);
        }
        let replay = rng.chance(85);
        let bare = rng.chance(12);
        let strategy = gen_strategy(&mut rng, cycles);
        specs.push(Spec::Query {
            expr,
            replay,
            bare,
            strategy,
        });
    }
    for _ in 0..inputs {
        specs.push(Spec::Input(rng.below(5)));
    }
    let ops = (0..1 + rng.below(12))
        .map(|_| match rng.below(10) {
            0..=3 => Op::Set(queries + rng.below(inputs as u64) as usize, rng.below(5)),
            4..=7 => Op::Get(rng.below(queries as u64) as usize),
            _ => Op::TryGet(rng.below(queries as u64) as usize),
        })
        .collect();
    let mut ops: Vec<Op> = ops;
    ops.insert(0, Op::Get(0));
    Case {
        specs,
        panics: !seed.is_multiple_of(3),
        cycles,
        queries,
        specs2: Vec::new(),
        queries2: 0,
        ops,
    }
}

/// The base case with, each by its own draw, a dissolving root and
/// deletes with recreates, and with a second runtime if `cross`. If
/// `paired`, every delete is followed at once by its recreate, so no
/// closure reads a deleted node: two overlapping reads that reach the
/// same deleted node would race for which one gets the stale panic and
/// which gets "last compute panicked".
fn gen_case(seed: u64, cross: bool, paired: bool) -> Case {
    let mut case = gen_base(seed);
    let mut rng = Rng(seed ^ 0x6a09_e667_f3bc_c908);
    let template = rng.chance(30);
    if template {
        add_dissolving_root(&mut case, &mut rng);
    }
    if cross {
        add_second_runtime(&mut case, &mut rng, usize::from(template) * 4);
    }
    if rng.chance(45) {
        add_deletes(&mut case, &mut rng, usize::from(template) * 4, paired);
    }
    case
}

fn insert_op(case: &mut Case, rng: &mut Rng, op: Op) -> usize {
    let at = 1 + rng.below(case.ops.len() as u64) as usize;
    case.ops.insert(at, op);
    at
}

fn shift(e: Expr, by: usize) -> Expr {
    let s = |e: Box<Expr>| Box::new(shift(*e, by));
    match e {
        Expr::Read(t, try_) => Expr::Read(t + by, try_),
        Expr::Add(a, b) => Expr::Add(s(a), s(b)),
        Expr::If(c, a, b) => Expr::If(s(c), s(a), s(b)),
        Expr::PanicIf(a, when) => Expr::PanicIf(s(a), when),
        Expr::Catch(a) => Expr::Catch(s(a)),
        Expr::FirstRun(a, b) => Expr::FirstRun(s(a), s(b)),
        leaf => leaf,
    }
}

fn replayable(expr: Expr, strategy: Strat) -> Spec {
    Spec::Query {
        expr,
        replay: true,
        bare: false,
        strategy,
    }
}

/// `validation_dissolves` under a placeholder-dependent branch, as nodes
/// 0 to 3: R = if N is even then C else 3, C = A + 1 with a Fallback, A
/// reads C on its first run only and is 5 after, and N reads the first
/// input. At budget 1 R's read of N dooms, so the placeholder takes the
/// branch to C on a speculative entry, which claims a member of C's
/// episode; the episode dissolves at validation when A stops reading C,
/// and C is entered again in the same loop.
fn add_dissolving_root(case: &mut Case, rng: &mut Rng) {
    for spec in &mut case.specs {
        if let Spec::Query { expr, .. } = spec {
            *expr = shift(expr.clone(), 4);
        }
    }
    for op in &mut case.ops {
        match op {
            Op::Set(i, _) | Op::Get(i) | Op::TryGet(i) => *i += 4,
            _ => unreachable!("the base case only sets and reads"),
        }
    }
    let input = case.queries + 4;
    let read = |t| Box::new(Expr::Read(t, false));
    let template = [
        replayable(
            Expr::If(read(3), read(1), Box::new(Expr::Const(3))),
            Strat::None,
        ),
        replayable(
            Expr::Add(read(2), Box::new(Expr::Const(1))),
            Strat::Fallback(rng.below(5)),
        ),
        replayable(
            Expr::FirstRun(read(1), Box::new(Expr::Const(5))),
            Strat::None,
        ),
        replayable(Expr::Read(input, false), Strat::None),
    ];
    case.specs.splice(0..0, template);
    case.queries += 4;
    case.ops.insert(0, Op::Get(0));
    for _ in 0..1 + rng.below(3) {
        let op = match rng.below(3) {
            0 => Op::Set(input, rng.below(5)),
            1 => Op::Get(rng.below(2) as usize),
            _ => Op::TryGet(rng.below(2) as usize),
        };
        insert_op(case, rng, op);
    }
}

/// A second runtime: lower queries over its inputs, which the first
/// runtime's queries read, and upper queries that read the first
/// runtime's nodes, which the first runtime never reads, so the graph
/// across the two stays acyclic. The first `skip` queries of the first
/// runtime are a template and keep their shape.
fn add_second_runtime(case: &mut Case, rng: &mut Rng, skip: usize) {
    let lower = 1 + rng.below(4) as usize;
    let queries2 = lower + rng.below(3) as usize;
    let inputs2 = 1 + rng.below(3) as usize;
    let nodes1 = case.specs.len();
    for at in 0..queries2 {
        let read = move |rng: &mut Rng| {
            let try_ = rng.chance(25);
            if at >= lower && rng.chance(50) {
                // The template is entered at R or C only: A's first run,
                // a doomed one included, reads C, so a read of A's own
                // differs by budget.
                let target = match rng.below(nodes1 as u64) as usize {
                    2 if skip > 0 => 1,
                    target => target,
                };
                Expr::ReadOther(target, try_)
            } else if at + 1 < lower && rng.chance(50) {
                Expr::Read(at + 1 + rng.below((lower - at - 1) as u64) as usize, try_)
            } else if at >= lower && rng.chance(50) {
                Expr::Read(rng.below(lower as u64) as usize, try_)
            } else {
                Expr::Read(queries2 + rng.below(inputs2 as u64) as usize, try_)
            }
        };
        let mut expr = gen_expr(rng, 2, &read);
        if !case.panics {
            expr = without_panics(expr);
        }
        case.specs2.push(Spec::Query {
            expr,
            replay: rng.chance(85),
            bare: rng.chance(12),
            strategy: Strat::None,
        });
    }
    for _ in 0..inputs2 {
        case.specs2.push(Spec::Input(rng.below(5)));
    }
    case.queries2 = queries2;
    for spec in &mut case.specs[skip..case.queries] {
        if let Spec::Query { expr, .. } = spec {
            if rng.chance(30) {
                let target = if rng.chance(60) {
                    rng.below(lower as u64) as usize
                } else {
                    queries2 + rng.below(inputs2 as u64) as usize
                };
                *expr = Expr::Add(
                    Box::new(expr.clone()),
                    Box::new(Expr::ReadOther(target, rng.chance(25))),
                );
            }
        }
    }
    for _ in 0..1 + rng.below(5) {
        let op = if rng.chance(40) {
            Op::Set2(queries2 + rng.below(inputs2 as u64) as usize, rng.below(5))
        } else {
            Op::Get2(rng.below(queries2 as u64) as usize, rng.chance(30))
        };
        insert_op(case, rng, op);
    }
}

/// Deletes of the first runtime's nodes, most followed by a recreate into
/// the freed slot, reads of the deleted nodes, and reads of replaced
/// handles. The first `skip` nodes are a template and are never deleted:
/// its first-run counter counts doomed runs, so a recreated member would
/// first run the template's cycle at a point that differs by budget.
fn add_deletes(case: &mut Case, rng: &mut Rng, skip: usize, paired: bool) {
    let nodes = case.specs.len();
    for _ in 0..1 + rng.below(3) {
        let q = skip
            + if rng.chance(80) {
                rng.below((case.queries - skip).div_ceil(2) as u64) as usize
            } else {
                rng.below((nodes - skip) as u64) as usize
            };
        let at = insert_op(case, rng, Op::Delete(q));
        let mut next = at + 1;
        if !paired && q < case.queries && rng.chance(40) {
            let read = if rng.chance(50) {
                Op::Get(q)
            } else {
                Op::TryGet(q)
            };
            case.ops.insert(next, read);
            next += 1;
        }
        if paired || rng.chance(70) {
            let spec = if q < case.queries {
                let mut expr = gen_expr(rng, 3, &reads(q, case.queries, nodes, case.cycles));
                if !case.panics {
                    expr = without_panics(expr);
                }
                Spec::Query {
                    expr,
                    replay: rng.chance(85),
                    bare: rng.chance(12),
                    strategy: gen_strategy(rng, case.cycles),
                }
            } else {
                Spec::Input(rng.below(5))
            };
            case.ops.insert(next, Op::Recreate(q, spec));
        }
    }
    for _ in 0..rng.below(3) {
        let op = Op::ReadOld(rng.below(4) as usize, rng.chance(50));
        insert_op(case, rng, op);
    }
}

impl Strat {
    fn build<T: Value>(self, wrap: fn(u64) -> T) -> CycleStrategy<T> {
        match self {
            Strat::None => CycleStrategy::None,
            Strat::Fallback(v) => CycleStrategy::fallback(move |_| wrap(v)),
            Strat::Fixpoint { initial, limit } => {
                CycleStrategy::Fixpoint(Fixpoint::new(move || wrap(initial)).max_iterations(limit))
            }
        }
    }
}

fn without_panics(e: Expr) -> Expr {
    let w = |e: Box<Expr>| Box::new(without_panics(*e));
    match e {
        Expr::PanicIf(inner, _) => without_panics(*inner),
        Expr::Add(a, b) => Expr::Add(w(a), w(b)),
        Expr::If(c, a, b) => Expr::If(w(c), w(a), w(b)),
        Expr::Catch(inner) => Expr::Catch(w(inner)),
        Expr::FirstRun(a, b) => Expr::FirstRun(w(a), w(b)),
        leaf => leaf,
    }
}

#[derive(Copy, Clone, Debug)]
enum Handle {
    Plain(Incr<u64>),
    Bare(Incr<Bare>),
}

impl Handle {
    fn erased(self) -> ErasedIncr {
        match self {
            Handle::Plain(h) => h.into(),
            Handle::Bare(h) => h.into(),
        }
    }
}

struct RtPtr<C: Cells>(*const Runtime<C>);
// SAFETY: test-only. Both runtimes of a run outlive every call into their
// closures, and a Local runtime's pointer never leaves its thread.
unsafe impl<C: Cells> Send for RtPtr<C> {}
unsafe impl<C: Cells> Sync for RtPtr<C> {}

/// The two runtimes of a run and each one's current handle per node,
/// which closures read through, so a recreated node is read in place of
/// the one it replaced.
struct World<C: Cells> {
    rts: [RtPtr<C>; 2],
    tables: [Vec<Mutex<Option<Handle>>>; 2],
}

impl<C: Cells> World<C> {
    fn rt(&self, side: usize) -> &Runtime<C> {
        // SAFETY: see RtPtr.
        unsafe { &*self.rts[side].0 }
    }

    fn handle(&self, side: usize, i: usize) -> Handle {
        self.tables[side][i].lock().unwrap().expect("built")
    }

    fn replace(&self, side: usize, i: usize, h: Handle) -> Option<Handle> {
        self.tables[side][i].lock().unwrap().replace(h)
    }
}

fn read_value<C: Cells>(rt: &Runtime<C>, h: Handle, try_: bool) -> u64 {
    match (h, try_) {
        (Handle::Plain(h), false) => rt.get(h),
        (Handle::Plain(h), true) => rt.try_get(h).unwrap_or(7777),
        (Handle::Bare(h), false) => rt.get(h).0,
        (Handle::Bare(h), true) => rt.try_get(h).map_or(7777, |b| b.0),
    }
}

struct Node<'a, C: Cells> {
    rt: &'a Runtime<C>,
    world: &'a World<C>,
    side: usize,
    at: usize,
    first: bool,
}

fn eval<C: Cells>(n: &Node<'_, C>, e: &Expr) -> u64 {
    match e {
        Expr::Const(v) => *v,
        Expr::Read(target, try_) => read_value(n.rt, n.world.handle(n.side, *target), *try_),
        Expr::ReadOther(target, try_) => {
            let other = 1 - n.side;
            read_value(n.world.rt(other), n.world.handle(other, *target), *try_)
        }
        Expr::Add(a, b) => eval(n, a).wrapping_add(eval(n, b)),
        Expr::If(c, a, b) => {
            if eval(n, c) % 2 == 0 {
                eval(n, a)
            } else {
                eval(n, b)
            }
        }
        Expr::PanicIf(c, when) => {
            let v = eval(n, c);
            if v % 7 == *when {
                std::panic::panic_any(format!("boom {}:{}", n.side, n.at));
            }
            v
        }
        Expr::Catch(inner) => catch_unwind(AssertUnwindSafe(|| eval(n, inner))).unwrap_or(9999),
        Expr::FirstRun(a, b) => {
            if n.first {
                eval(n, a)
            } else {
                eval(n, b)
            }
        }
    }
}

fn make<C: Cells>(world: &Arc<World<C>>, side: usize, i: usize, spec: &Spec) -> Handle {
    let rt = world.rt(side);
    match spec {
        Spec::Input(v) => Handle::Plain(rt.create_input(*v)),
        Spec::Query {
            expr,
            replay,
            bare,
            strategy,
        } => {
            let replay = if *replay {
                Replay::Allowed
            } else {
                Replay::Native
            };
            let (expr, world, runs) = (expr.clone(), world.clone(), AtomicU32::new(0));
            let compute = move |rt: &Runtime<C>| {
                let node = Node {
                    rt,
                    world: &world,
                    side,
                    at: i,
                    first: runs.fetch_add(1, Relaxed) == 0,
                };
                eval(&node, &expr)
            };
            if *bare {
                Handle::Bare(rt.create_query_with(
                    move |rt| Bare(compute(rt)),
                    QueryOpts {
                        cycle: strategy.build(Bare),
                        replay,
                    },
                ))
            } else {
                Handle::Plain(rt.create_query_with(
                    compute,
                    QueryOpts {
                        cycle: strategy.build(|v| v),
                        replay,
                    },
                ))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Out {
    Value(u64),
    Cycle(Vec<u32>),
    Panic(String),
    /// A delete's result, with the error's name.
    Deleted(Result<(), String>),
    /// A recreate, and whether it took its node's slot.
    Recreated(bool),
    Done,
}

fn class(payload: Box<dyn std::any::Any + Send>) -> String {
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default();
    for (needle, class) in [
        ("dependency cycle", "cycle"),
        ("last compute panicked", "failed"),
        ("stale Incr", "stale"),
        ("own compute", "own-compute"),
    ] {
        if message.contains(needle) {
            return class.to_owned();
        }
    }
    message
}

fn read_out<C: Cells>(rt: &Runtime<C>, h: Handle, try_: bool) -> Out {
    let read = catch_unwind(AssertUnwindSafe(|| match (h, try_) {
        (Handle::Plain(h), false) => Ok(rt.get(h)),
        (Handle::Plain(h), true) => rt.try_get(h),
        (Handle::Bare(h), false) => Ok(rt.get(h).0),
        (Handle::Bare(h), true) => rt.try_get(h).map(|b| b.0),
    }));
    match read {
        Ok(Ok(v)) => Out::Value(v),
        Ok(Err(c)) => Out::Cycle(c.participants().iter().map(|p| p.slot()).collect()),
        Err(p) => Out::Panic(class(p)),
    }
}

fn set_out<C: Cells>(rt: &Runtime<C>, h: Handle, v: u64) -> Out {
    let Handle::Plain(h) = h else {
        unreachable!("inputs are plain")
    };
    match catch_unwind(AssertUnwindSafe(|| rt.set(h, v))) {
        Ok(()) => Out::Done,
        Err(p) => Out::Panic(class(p)),
    }
}

/// A run's two runtimes, its node tables, and the handles recreates
/// replaced.
struct Live<C: Cells> {
    world: Arc<World<C>>,
    retired: Mutex<Vec<Handle>>,
    queries: [usize; 2],
}

/// After an operation: every query's state, every node's dependencies in
/// the order its last landing installed them, and whether a read returned
/// a node's Dirty value, or had a delivery, that a later landing made
/// Clean. Both runtimes' entries, the first's first.
type Snapshot = (Vec<NodeState>, Vec<Vec<u32>>, bool);

impl<C: Cells> Live<C> {
    fn new(rts: &[Box<Runtime<C>>; 2], case: &Case) -> Self {
        let table = |n: usize| (0..n).map(|_| Mutex::new(None)).collect();
        let world = Arc::new(World {
            rts: [RtPtr(&*rts[0]), RtPtr(&*rts[1])],
            tables: [table(case.specs.len()), table(case.specs2.len())],
        });
        for (side, specs) in [&case.specs, &case.specs2].into_iter().enumerate() {
            for (i, spec) in specs.iter().enumerate().rev() {
                let h = make(&world, side, i, spec);
                world.replace(side, i, h);
            }
        }
        Live {
            world,
            retired: Mutex::new(Vec::new()),
            queries: [case.queries, case.queries2],
        }
    }

    fn apply(&self, op: &Op) -> Out {
        let w = &*self.world;
        match op {
            Op::Set(i, v) => set_out(w.rt(0), w.handle(0, *i), *v),
            Op::Get(i) => read_out(w.rt(0), w.handle(0, *i), false),
            Op::TryGet(i) => read_out(w.rt(0), w.handle(0, *i), true),
            Op::Delete(i) => Out::Deleted(
                w.rt(0)
                    .try_delete_node(w.handle(0, *i).erased())
                    .map_err(|e| format!("{e:?}")),
            ),
            Op::Recreate(i, spec) => {
                let old = w.handle(0, *i);
                if w.rt(0).is_alive(old.erased()) {
                    return Out::Done;
                }
                let new = make(&self.world, 0, *i, spec);
                w.replace(0, *i, new);
                self.retired.lock().unwrap().push(old);
                Out::Recreated(new.erased().slot() == old.erased().slot())
            }
            Op::ReadOld(k, try_) => {
                let old = {
                    let retired = self.retired.lock().unwrap();
                    (!retired.is_empty()).then(|| retired[k % retired.len()])
                };
                old.map_or(Out::Done, |h| read_out(w.rt(0), h, *try_))
            }
            Op::Set2(i, v) => set_out(w.rt(1), w.handle(1, *i), *v),
            Op::Get2(i, try_) => read_out(w.rt(1), w.handle(1, *i), *try_),
        }
    }

    /// This thread's checks after its operation: no engine panic outside
    /// the documented ones, and no virtual frame left or pushed twice.
    fn check_thread(&self) -> Result<(), String> {
        if let Some(internal) = INTERNAL.with(|i| i.borrow_mut().take()) {
            return Err(format!("an engine panic: {internal}"));
        }
        let repushes = crate::replay::take_vindex_repushes();
        if repushes != 0 {
            return Err(format!("{repushes} virtual frames pushed again"));
        }
        if !crate::replay::vindexes_empty() {
            return Err("a virtual frame is left indexed".to_owned());
        }
        Ok(())
    }

    fn assert_invariants(&self) {
        self.world.rt(0).assert_invariant_i();
        self.world.rt(1).assert_invariant_i();
    }

    fn snapshot(&self) -> Snapshot {
        let w = &*self.world;
        let mut states = Vec::new();
        let mut deps = Vec::new();
        let mut replaced = false;
        for side in 0..2 {
            let rt = w.rt(side);
            for q in 0..self.queries[side] {
                let h = w.handle(side, q);
                states.push(if !rt.is_alive(h.erased()) {
                    NodeState::Dead
                } else {
                    match h {
                        Handle::Plain(h) => rt.state_of(h),
                        Handle::Bare(h) => rt.state_of(h),
                    }
                });
            }
            for x in 0..rt.nodes.len() {
                let mut d = Vec::new();
                rt.nodes.get(x).for_each_dep(|n| d.push(n.0));
                deps.push(d);
            }
            replaced |= std::mem::take(&mut *rt.dirty_reads.lock().unwrap())
                .into_iter()
                .any(|n| rt.nodes.get(n).state() == NodeState::Clean);
        }
        (states, deps, replaced)
    }
}

/// The Failed and Cyclic nodes, by index.
fn settled(states: &[NodeState]) -> Vec<(usize, NodeState)> {
    states
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s, NodeState::Failed | NodeState::Cyclic))
        .map(|(i, &s)| (i, s))
        .collect()
}

/// A run's outcomes with the snapshot after each, operation by operation.
type Trace = Vec<(Out, Snapshot)>;

fn runtimes<C: Cells>(frames: Option<u32>, bytes: usize) -> [Box<Runtime<C>>; 2] {
    [(); 2].map(|()| {
        let rt = Box::new(Runtime::<C>::new());
        rt.set_placeholder(0_u64);
        rt.set_max_frames(frames.unwrap_or(u32::MAX));
        rt.set_max_stack_bytes(bytes);
        rt
    })
}

fn reset_thread() {
    hook();
    INTERNAL.with(|i| i.borrow_mut().take());
    crate::replay::take_vindex_repushes();
}

fn run<C: Cells>(case: &Case, frames: Option<u32>, bytes: usize) -> Trace {
    let rts = runtimes::<C>(frames, bytes);
    let live = Live::new(&rts, case);
    reset_thread();
    let mut trace = Vec::new();
    for op in &case.ops {
        let out = live.apply(op);
        live.assert_invariants();
        if let Err(m) = live.check_thread() {
            panic!("operation {op:?}: {m}");
        }
        trace.push((out, live.snapshot()));
    }
    trace
}

/// Who runs what, between barriers: a set, delete or recreate by one
/// thread while the other waits, or a read by each thread, by index into
/// the case's operations.
#[derive(Debug)]
enum Phase {
    Mutate(usize, usize),
    Read([Option<usize>; 2]),
}

impl Phase {
    fn ops(&self) -> Vec<usize> {
        match self {
            Phase::Mutate(i, _) => vec![*i],
            Phase::Read(slots) => {
                let mut ops: Vec<usize> = slots.iter().flatten().copied().collect();
                ops.sort_unstable();
                ops
            }
        }
    }

    fn of(&self, thread: usize) -> Option<usize> {
        match *self {
            Phase::Mutate(i, t) => (t == thread).then_some(i),
            Phase::Read(slots) => slots[thread],
        }
    }
}

/// Pairs consecutive reads into a read phase, in the case's order, so the
/// unbounded single-thread run of the case gives each read's expected
/// result at its sequence point.
fn phases(case: &Case, seed: u64) -> Vec<Phase> {
    let mut rng = Rng(seed ^ 0xbb67_ae85_84ca_a73b);
    let mut phases = Vec::new();
    let mut i = 0;
    while i < case.ops.len() {
        let t = rng.below(2) as usize;
        if !case.ops[i].is_read() {
            phases.push(Phase::Mutate(i, t));
            i += 1;
        } else if i + 1 < case.ops.len() && case.ops[i + 1].is_read() {
            let mut slots = [None; 2];
            slots[t] = Some(i);
            slots[1 - t] = Some(i + 1);
            phases.push(Phase::Read(slots));
            i += 2;
        } else {
            let mut slots = [None; 2];
            slots[t] = Some(i);
            phases.push(Phase::Read(slots));
            i += 1;
        }
    }
    phases
}

/// The case's operations on two threads by `phases`, with each phase's
/// snapshot on its operations, and which operations end a phase.
fn run_two(case: &Case, phases: &[Phase], frames: Option<u32>, bytes: usize) -> (Trace, Vec<bool>) {
    let rts = runtimes::<Shared>(frames, bytes);
    let live = Live::new(&rts, case);
    let n = case.ops.len();
    let outs: Mutex<Vec<Option<Out>>> = Mutex::new(vec![None; n]);
    let snaps: Mutex<Vec<Option<Snapshot>>> = Mutex::new(vec![None; phases.len()]);
    let failures: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let barrier = Barrier::new(2);
    std::thread::scope(|s| {
        for t in 0..2 {
            let (live, outs, snaps, failures, barrier) =
                (&live, &outs, &snaps, &failures, &barrier);
            s.spawn(move || {
                reset_thread();
                for (p, phase) in phases.iter().enumerate() {
                    barrier.wait();
                    if let Some(i) = phase.of(t) {
                        let op = &case.ops[i];
                        match catch_unwind(AssertUnwindSafe(|| live.apply(op))) {
                            Ok(out) => outs.lock().unwrap()[i] = Some(out),
                            Err(p) => failures.lock().unwrap().push(format!(
                                "thread {t}, operation {i} ({op:?}) panicked: {}",
                                class(p)
                            )),
                        }
                        if let Err(m) = live.check_thread() {
                            failures
                                .lock()
                                .unwrap()
                                .push(format!("thread {t}, operation {i} ({op:?}): {m}"));
                        }
                    }
                    barrier.wait();
                    if t == 0 {
                        if let Err(e) = catch_unwind(AssertUnwindSafe(|| live.assert_invariants()))
                        {
                            failures
                                .lock()
                                .unwrap()
                                .push(format!("after phase {p}: {}", class(e)));
                        }
                        snaps.lock().unwrap()[p] = Some(live.snapshot());
                    }
                }
            });
        }
    });
    let failures = failures.into_inner().unwrap();
    assert!(failures.is_empty(), "{failures:#?}");
    let (outs, snaps) = (outs.into_inner().unwrap(), snaps.into_inner().unwrap());
    let mut trace: Vec<Option<(Out, Snapshot)>> = vec![None; n];
    let mut settles = vec![false; n];
    for (phase, snap) in phases.iter().zip(snaps) {
        let snap = snap.expect("every phase snapshots");
        let ops = phase.ops();
        settles[*ops.last().unwrap()] = true;
        for i in ops {
            trace[i] = Some((outs[i].clone().expect("every operation ran"), snap.clone()));
        }
    }
    (trace.into_iter().map(Option::unwrap).collect(), settles)
}

/// Compares `got` with the unbounded run's `want`, the states only after
/// the operations `settles` marks.
fn compare(case: &Case, want: &Trace, got: &Trace, settles: Option<&[bool]>, label: &str) {
    // A closure's read of a deleted node panics as stale, so a case with
    // deletes has panics.
    let panics = case.panics || case.ops.iter().any(|op| matches!(op, Op::Delete(_)));
    let mut agree = true;
    for (i, ((want_out, want), (got_out, got))) in want.iter().zip(got).enumerate() {
        // Once a speculative landing or a delivery has left the states
        // apart, a panic or a cycle may come out otherwise: a walk of a
        // Dirty node computes its deps before its closure, so it enters a
        // cycle, and so picks its root and strategy, where a compute of
        // the node New would not.
        if !agree && (panics || case.cycles) {
            break;
        }
        // A read of the other runtime records no edge (T5 Scoping), so a
        // set of the second runtime leaves the first's readers of it on
        // what they read when they last ran. Once the states differ, the
        // runs ran those readers at different times.
        if !agree && matches!(case.ops[i], Op::Set2(..)) {
            break;
        }
        // A read that lands its target Dirty returns a value computed
        // against a read that a later landing in the same revision
        // changed. A replayed frame reads after that landing when the read
        // is a Clean `get`, which no delivery can intercept, so the
        // bounded run may return the later value.
        let target = match case.ops[i] {
            Op::Get(q) | Op::TryGet(q) => Some(q),
            Op::Get2(q, _) => Some(case.queries + q),
            _ => None,
        };
        if target.is_some_and(|q| want.0[q] == NodeState::Dirty) {
            agree = false;
            continue;
        }
        // So may a frame that read a node's Dirty value, or took its
        // unwind or a delivery for its read, which a later landing made
        // Clean: its next retry reads it with a Clean `get`.
        if want.2 || got.2 {
            agree = false;
            continue;
        }
        // A delete's dependents are history too: a speculative entry can
        // land a reader of the node Clean, with its edge, where the
        // unbounded run left the reader New. Past a delete that came out
        // otherwise the two runs hold different graphs.
        if !agree && matches!(case.ops[i], Op::Delete(_)) && want_out != got_out {
            break;
        }
        let settles = settles.is_none_or(|s| s[i]);
        let compared = |out: &Out, snap: &Snapshot| {
            (out.clone(), (agree && settles).then(|| settled(&snap.0)))
        };
        assert_eq!(
            compared(want_out, want),
            compared(got_out, got),
            "{label}: operation {i} ({:?}) differs from the unbounded run\nops: {:?}\n\
             case: {:#?}\nsecond runtime: {:#?}",
            case.ops[i],
            case.ops,
            case.specs,
            case.specs2,
        );
        // A dep list is history too: under a panic a read can stop the
        // closure short of the reads after it.
        if settles {
            agree &= want.0 == got.0 && want.1 == got.1;
        }
    }
}

/// Runs `f`, prefixing the message of any panic out of it with `label`.
fn labelled<T>(label: &str, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| {
        let message = p
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| p.downcast_ref::<&str>().copied())
            .unwrap_or("a non-string panic");
        panic!("{label}: {message}")
    })
}

fn differential<C: Cells>(case: &Case, seed: u64) {
    let unbounded = labelled(&format!("seed {seed:#x}, unbounded"), || {
        run::<C>(case, None, 0)
    });
    for frames in [1, 2, 3] {
        for bytes in [0, 16 * 1024] {
            let label = format!("seed {seed:#x}, budget {frames} frames, {bytes} bytes");
            let bounded = labelled(&label, || run::<C>(case, Some(frames), bytes));
            compare(case, &unbounded, &bounded, None, &label);
        }
    }
}

/// The two-thread form: every phase schedule at every budget, unbounded
/// included, against the unbounded single-thread run.
fn differential_two(case: &Case, seed: u64) {
    let phases = phases(case, seed);
    let unbounded = run::<Shared>(case, None, 0);
    for frames in [None, Some(1), Some(2), Some(3)] {
        for bytes in [0, 16 * 1024] {
            let label = format!(
                "seed {seed:#x}, two threads {phases:?}, budget {frames:?} frames, {bytes} bytes"
            );
            let (got, settles) = labelled(&label, || run_two(case, &phases, frames, bytes));
            compare(case, &unbounded, &got, Some(&settles), &label);
        }
    }
}
/// Seeds that once failed, each with what it caught.
#[test]
fn regressions() {
    // A closure caught the panic of its read of a Failed member in a walk
    // the Failed node's read ran, and landed Clean over it.
    let case = gen_base(0x112);
    quiet(|| {
        run::<Local>(&case, None, 0);
        run::<Shared>(&case, None, 0);
    });
    // 0xbc2: a closure caught the cycle unwind of a dep it had read before,
    // and landed Clean over that dep through the same-deps publish. 0x74f
    // and 0x597: a read whose compute ran an episode that landed the node
    // Dirty or New returned as if it landed Clean.
    // 0x4d1b and 0x00e8_d4a5_307b: a cycle episode that aborted left its
    // root's reader unpoisoned, so a `try_get` that caught the unwind
    // landed Clean over a dep the abort left Dirty.
    for seed in [0xbc2, 0x74f, 0x597, 0x4d1b, 0x00e8_d4a5_307b] {
        let case = gen_base(seed);
        quiet(|| {
            run::<Local>(&case, None, 0);
            run::<Shared>(&case, None, 0);
        });
    }
}
/// Cases with deletes whose readers a speculative entry landed Clean, so
/// the bounded run's delete is refused where the unbounded one's is not.
#[test]
#[cfg_attr(miri, ignore)]
fn delete_seeds() {
    for seed in [0x5a, 0x32a] {
        let case = gen_case(seed, false, false);
        quiet(|| {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        });
    }
}

/// A case with cycles on two threads in which a thread is wounded out of
/// an entry above virtual frames that own deferred claims.
#[test]
#[cfg_attr(miri, ignore)]
fn two_threads_hang() {
    let case = gen_case(0x159, false, false);
    quiet(|| differential_two(&case, 0x159));
}

/// Cases with a second runtime that once failed: 0x135's second runtime
/// read the template's A directly, whose first run is a doomed one at
/// budget 1; in 0xfe, 0x308, 0x38b and 0x3d0 a doomed frame's read of
/// the other runtime computed a node that panics and landed it Failed, so
/// the real read panicked "last compute panicked" (Shared only: a Local
/// stack can't see another runtime's frames); in 0xa02 a speculative
/// landing read the second runtime's input before a set of it, and a
/// later walk verified that reader Clean on the old value.
#[test]
#[cfg_attr(miri, ignore)]
fn cross_seeds() {
    for (seed, local) in [
        (0x135, true),
        (0xfe, false),
        (0x308, false),
        (0x38b, false),
        (0x3d0, false),
        (0xa02, false),
    ] {
        let case = gen_case(seed, true, false);
        quiet(|| {
            if local {
                differential::<Local>(&case, seed);
            }
            differential::<Shared>(&case, seed);
        });
    }
}

/// Seeds the scans found failing at a frame budget, each a replay fault
/// since fixed.
#[test]
#[cfg_attr(miri, ignore)]
fn replay_seeds() {
    for seed in [
        0x4c,
        0x112,
        0x21b,
        0x363,
        0x3d0,
        0x425,
        0x4ea,
        0x597,
        0x603,
        0x640,
        0x730,
        0x74f,
        0x7e4,
        0x7f5,
        0x832,
        0x837,
        0x849,
        0x850,
        0x934,
        0x96b,
        0x9ca,
        0x9dc,
        0xa55,
        0xa5b,
        0xafd,
        0xb0e,
        0xbb6,
        0xbbc,
        0xbc2,
        0xbe7,
        0xcb7,
        0xd60,
        0xdd3,
        0xe3f,
        0xe7b,
        0xf41,
        0xfee,
        0x1359,
        0x135a,
        0x114b,
        0x13b9,
        0x1b3e,
        0x1c12,
        0x1ce4,
        0x1d02,
        0x1e89,
        0x21b9,
        0x2224,
        0x22ef,
        0x271d,
        0x27d7,
        0x2997,
        0x2bd9,
        0x2cce,
        0x2d76,
        0x4654,
        0x518e,
        0x69d7,
        0x7043,
        0x7fe4,
        0x8074,
        0x8b41,
        0x9e14,
        0xa7c1,
        0xbc86,
        0xbef0,
        0xc118,
        0xd4f2,
        0x0001_2b74,
        0x0001_2d3c,
        0x0001_2fa6,
        0x0001_55f8,
        0x0001_624c,
        0x0001_698b,
        0x0001_94e6,
        0x0001_a9c1,
        0x00e8_d4a5_2919,
        0x00e8_d4a5_2db3,
        0x00e8_d4a5_33fa,
        0x0876_ebf0_5e41_2095,
        0x551d_e2c5_404a_84b1,
        0x84a1_d45b_e6f9_b165,
        0xa649_6004_aa21_dd3b,
        0xa8cc_85b9_bbfd_657d,
        0xe386_b8af_df64_ca89,
        0xf9e8_e4b4_cddf_4529,
    ] {
        let case = gen_base(seed);
        quiet(|| {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        });
    }
}

fn cases() -> u32 {
    if std::env::var_os("INCR_TEST_REDUCED").is_some() {
        32
    } else {
        256
    }
}

thread_local! {
    static INTERNAL: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Installs the hook that keeps the cases' own panics quiet and records an
/// engine panic that is not one of its documented ones, which a closure
/// that catches it would otherwise hide.
fn hook() {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let payload = info.payload();
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("");
            if message.starts_with("incr-core:")
                && !message.starts_with("incr-core: dependency cycle: ")
                && !message.contains("read but its last compute panicked")
                && !message.starts_with("incr-core: stale Incr<")
                && !message.contains("called from inside node")
            {
                INTERNAL.with(|i| {
                    i.borrow_mut()
                        .get_or_insert_with(|| format!("{message} at {:?}", info.location()));
                });
            }
            if !message.starts_with("boom") {
                default(info);
            }
        }));
    });
}

/// Runs `f` with panic messages from closures silenced: the generated
/// graphs panic on purpose.
fn quiet(f: impl FnOnce()) {
    hook();
    f();
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases()))]

    #[test]
    #[cfg_attr(miri, ignore)]
    fn replay_differential_local(seed in any::<u64>()) {
        quiet(|| differential::<Local>(&gen_case(seed, false, false), seed));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn replay_differential_shared(seed in any::<u64>()) {
        quiet(|| differential::<Shared>(&gen_case(seed, false, false), seed));
    }

    // Only a case without panics or cycles: which of two overlapping reads
    // computes a node first decides which one gets a closure's panic
    // payload and which gets "last compute panicked", and where a cycle is
    // entered decides its root and strategy, at every budget, unbounded
    // included.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn replay_differential_shared_two_threads(seed in any::<u64>()) {
        let seed = seed / 6 * 6;
        quiet(|| differential_two(&gen_case(seed, false, true), seed));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[ignore = "cross-runtime reads record no edge, so a speculative entry's early read keeps a value a later set changed"]
    fn replay_differential_cross_runtime_local(seed in any::<u64>()) {
        quiet(|| differential::<Local>(&gen_case(seed, true, false), seed));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[ignore = "cross-runtime reads record no edge, so a speculative entry's early read keeps a value a later set changed"]
    fn replay_differential_cross_runtime_shared(seed in any::<u64>()) {
        quiet(|| differential::<Shared>(&gen_case(seed, true, false), seed));
    }
}

fn shared() -> bool {
    std::env::var_os("REPLAY_SHARED").is_some()
}

fn two_threads() -> bool {
    std::env::var_os("REPLAY_TWO").is_some()
}

/// Whether a two-thread scan takes seeds with panics or cycles too, which
/// the proptest leaves out.
fn all_seeds() -> bool {
    std::env::var_os("REPLAY_ALL_SEEDS").is_some()
}

/// The case for `seed`, the base one if `REPLAY_BASE` is set, with a
/// second runtime if `REPLAY_CROSS` is.
fn case_of(seed: u64) -> Case {
    if std::env::var_os("REPLAY_BASE").is_some() {
        gen_base(seed)
    } else {
        gen_case(
            seed,
            std::env::var_os("REPLAY_CROSS").is_some(),
            two_threads(),
        )
    }
}

/// The differential the environment picks for `seed`.
fn check(seed: u64) {
    let case = case_of(seed);
    if two_threads() {
        differential_two(&case, seed);
    } else if shared() {
        differential::<Shared>(&case, seed);
    } else {
        differential::<Local>(&case, seed);
    }
}

/// One case by its seed (`REPLAY_SEED`, hex), printing both runs' traces,
/// under Shared if `REPLAY_SHARED` is set, and with `REPLAY_TWO` the
/// two-thread run's phases and trace.
#[test]
#[ignore = "a diagnostic for one seed"]
fn replay_one() {
    if shared() {
        one::<Shared>();
    } else {
        one::<Local>();
    }
}

fn one<C: Cells>() {
    let seed = u64::from_str_radix(
        std::env::var("REPLAY_SEED")
            .expect("REPLAY_SEED")
            .trim_start_matches("0x"),
        16,
    )
    .expect("a hex seed");
    let frames: u32 = std::env::var("REPLAY_FRAMES").map_or(1, |v| v.parse().expect("a budget"));
    let case = case_of(seed);
    for (i, s) in case.specs.iter().enumerate() {
        eprintln!("{i}: {s:?}");
    }
    for (i, s) in case.specs2.iter().enumerate() {
        eprintln!("second runtime {i}: {s:?}");
    }
    eprintln!("ops: {:?}", case.ops);
    quiet(|| {
        let unbounded = run::<C>(&case, None, 0);
        eprintln!("unbounded: {unbounded:?}");
        let bytes: usize = std::env::var("REPLAY_BYTES").map_or(0, |v| v.parse().expect("bytes"));
        let bounded = catch_unwind(AssertUnwindSafe(|| run::<C>(&case, Some(frames), bytes)));
        eprintln!("budget {frames}: {bounded:?}");
        if two_threads() {
            let phases = phases(&case, seed);
            eprintln!("phases: {phases:?}");
            let two = catch_unwind(AssertUnwindSafe(|| {
                run_two(&case, &phases, Some(frames), bytes)
            }));
            eprintln!("two threads, budget {frames}: {two:?}");
        }
        if std::env::var_os("REPLAY_FULL").is_some() {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
            differential_two(&case, seed);
        }
    });
}

fn size(e: &Expr) -> usize {
    match e {
        Expr::Const(_) | Expr::Read(..) | Expr::ReadOther(..) => 1,
        Expr::Add(a, b) | Expr::FirstRun(a, b) => 1 + size(a) + size(b),
        Expr::If(a, b, c) => 1 + size(a) + size(b) + size(c),
        Expr::PanicIf(a, _) | Expr::Catch(a) => 1 + size(a),
    }
}

/// The smallest failing case among seeds `0..REPLAY_SCAN`, by total
/// expression size and operation count, under Shared if `REPLAY_SHARED`
/// is set and on two threads if `REPLAY_TWO` is.
#[test]
#[ignore = "a diagnostic that scans seeds"]
fn replay_scan() {
    let n: u64 = std::env::var("REPLAY_SCAN").map_or(2000, |v| v.parse().expect("a count"));
    let mut best: Option<(usize, u64)> = None;
    let mut failing: Vec<u64> = Vec::new();
    if let Ok(seed) = std::env::var("REPLAY_CHECK") {
        let seed = u64::from_str_radix(&seed, 16).expect("a hex seed");
        check(seed);
        return;
    }
    quiet(|| {
        let from: u64 = std::env::var("REPLAY_FROM").map_or(0, |v| v.parse().expect("a seed"));
        for seed in from..n {
            if two_threads() && !seed.is_multiple_of(6) && !all_seeds() {
                continue;
            }
            let case = case_of(seed);
            let weight: usize = case
                .specs
                .iter()
                .chain(&case.specs2)
                .map(|s| match s {
                    Spec::Query { expr, .. } => size(expr),
                    Spec::Input(_) => 0,
                })
                .sum::<usize>()
                + case.ops.len();
            let all = std::env::var_os("REPLAY_ALL").is_some();
            if !all && best.is_some_and(|(w, _)| w <= weight) {
                continue;
            }
            if std::env::var_os("REPLAY_TRACE").is_some() {
                eprintln!("seed {seed}");
            }
            let failed = catch_unwind(AssertUnwindSafe(|| check(seed))).err();
            if let Some(payload) = failed {
                if all {
                    let message = class(payload);
                    let first = message.lines().next().unwrap_or_default();
                    eprintln!("FAIL {seed:x}: {first}");
                    failing.push(seed);
                }
                if best.is_none_or(|(w, _)| weight < w) {
                    best = Some((weight, seed));
                }
            }
        }
    });
    eprintln!("smallest failing: {best:?}");
    if !failing.is_empty() {
        eprintln!("failing: {} {failing:x?}", failing.len());
    }
}
