//! `replay_differential`: random graphs of replayable queries, each run at
//! frame budgets 1, 2 and 3, with and without a byte budget, against the
//! same graph and operations run unbounded. Every `get` and `try_get`
//! result (the value, a cycle's participants, or the panic's class or
//! payload), every delete's result, and the sets of Failed and Cyclic
//! nodes after each operation must be the unbounded run's, while every
//! query's state agrees with the unbounded run's before the operation
//! (with one exception on two threads, below).
//!
//! A case may also delete nodes and recreate them into the freed slots,
//! keep a second runtime whose queries read and are read by the first's,
//! and include a root whose cycle dissolves at validation. Under Shared,
//! a second form steps two threads through barriers: one thread makes
//! each set, delete or recreate while the other waits, and reads overlap
//! only reads, so each result is still the unbounded run's at its
//! sequence point, with either of two overlapping reads first, except
//! that a panic is one outcome kind whatever its text: which of two
//! racing readers of a panicked node gets the closure's payload and which
//! gets "last compute panicked" is not part of the result (the wiki
//! decision "Shared panic payload contract"). A get whose thread's
//! `try_get` got a cycle that the other read's get then landed is ordered
//! per read: a cycle's result depends on the member a get enters at, so
//! its run may mix both orders' outcomes, but each of its node's reads
//! must be an order's outcome for that read and its value the closure
//! over them. The Failed and Cyclic sets
//! after two overlapping reads are history, as the states are: a walk
//! computes from its plan, so it can land a node Failed from a dep the
//! other thread landed Failed after the walk expanded it. Where no order
//! gives those sets, each node the reads left settled must have a Failed
//! or Cyclic dep or a closure that panics or cycles when re-run, each
//! node an order settles that they left Dirty or New must still read so,
//! and the run ends there.
//!
//! A speculative entry can land a node the unbounded run leaves New, and
//! with panics the engine's outcomes depend on that history: a walk
//! computes a Dirty node's deps before its closure, so a dep's panic
//! leaves the walk where the closure would have caught it, and a Failed
//! dep panics at the walk where a New node's closure would catch its
//! read. So once the states differ, a case with panics stops comparing,
//! and a case without them, a third of the seeds, compares the results to
//! the end.

use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{
    AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst,
};
use std::sync::{Arc, Barrier, Mutex};

use proptest::prelude::*;

use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{CycleStrategy, Fixpoint, QueryOpts, Replay};
use crate::handle::{ErasedIncr, Incr};
use crate::node::NodeId;
use crate::state::NodeState;
use crate::Value;

/// A value type with no placeholder unless [`BARE_PLACEHOLDER`] is set,
/// so reads of it nest natively. `.1` records whether a speculative
/// entry's run built it, and equality ignores it.
#[derive(Clone, Debug)]
struct Bare(u64, bool);

impl PartialEq for Bare {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Bare {}

/// What [`Bare`]'s construction and `Drop` check, set only in a child
/// process: the values the unbounded runs build, then a `Drop` that panics
/// on a speculative run's value outside them, which only a bounded run can
/// make.
enum BareDrops {
    Off,
    Recording(HashSet<u64>),
    Panicking(HashSet<u64>),
}

static BARE_DROPS: Mutex<BareDrops> = Mutex::new(BareDrops::Off);

/// When set, [`Bare`] has a placeholder, so its reads replay as `u64`'s do.
static BARE_PLACEHOLDER: AtomicBool = AtomicBool::new(false);

impl Bare {
    fn made(v: u64) -> Self {
        let speculative = crate::replay::speculative();
        if let BareDrops::Recording(built) = &mut *BARE_DROPS.lock().unwrap() {
            built.insert(v);
        }
        Bare(v, speculative)
    }
}

impl Drop for Bare {
    fn drop(&mut self) {
        let panics = self.1
            && match &*BARE_DROPS.lock().unwrap() {
                BareDrops::Panicking(built) => !built.contains(&self.0),
                _ => false,
            };
        if panics {
            std::panic::panic_any(String::from("speculative drop"));
        }
    }
}

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
#[derive(Clone)]
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

/// `e` with its reads of the first `skip` of `nodes` nodes moved to the
/// nodes after them.
fn past_template(e: Expr, skip: usize, nodes: usize) -> Expr {
    let p = |e: Box<Expr>| Box::new(past_template(*e, skip, nodes));
    match e {
        Expr::Read(t, try_) if t < skip => Expr::Read(skip + t % (nodes - skip), try_),
        Expr::Add(a, b) => Expr::Add(p(a), p(b)),
        Expr::If(c, a, b) => Expr::If(p(c), p(a), p(b)),
        Expr::PanicIf(a, when) => Expr::PanicIf(p(a), when),
        Expr::Catch(a) => Expr::Catch(p(a)),
        Expr::FirstRun(a, b) => Expr::FirstRun(p(a), p(b)),
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
/// handles. The first `skip` nodes are a template, never deleted and never
/// read by a recreated node: its first-run counter counts doomed runs, so
/// a recreated member, or a recreated reader of a member that has not run
/// yet, would first run the template's cycle at a point that differs by
/// budget.
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
                let expr = gen_expr(rng, 3, &reads(q, case.queries, nodes, case.cycles));
                let mut expr = past_template(expr, skip, nodes);
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

    fn state<C: Cells>(self, rt: &Runtime<C>) -> NodeState {
        match self {
            Handle::Plain(h) => rt.state_of(h),
            Handle::Bare(h) => rt.state_of(h),
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
        (Handle::Bare(h), false) => rt.get(h).0,
        (h, true) => try_value(rt, h).unwrap_or(7777),
    }
}

fn try_value<C: Cells>(rt: &Runtime<C>, h: Handle) -> Result<u64, crate::cycle::Cycle> {
    match h {
        Handle::Plain(h) => rt.try_get(h),
        Handle::Bare(h) => rt.try_get(h).map(|b| b.0),
    }
}

/// A closure's read as the recorder keeps it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ReadOut {
    Value(u64),
    /// A `try_get`'s `Err`, with the cycle's participants by slot.
    Cycle(Vec<u32>),
    /// The read unwound into the closure's code.
    Unwound,
}

/// A closure's read: the target's side and index, whether by `try_get`,
/// and its outcome.
type Read = (usize, usize, bool, ReadOut);

/// One run of a query's closure: its node's side and index, whether it
/// was the node's first, its reads in order, its value if it returned,
/// and when it ended, on a clock both threads share.
#[derive(Clone, Debug)]
struct Eval {
    node: (usize, usize),
    first: bool,
    reads: Vec<Read>,
    value: Option<u64>,
    end: u64,
    /// When each read that got a cycle returned, by its index in
    /// `reads`, on the same clock.
    caught: Vec<(usize, u64)>,
    /// When each read that got a value returned, likewise.
    returned: Vec<(usize, u64)>,
}

thread_local! {
    /// While set, the runs of closures that end on this thread.
    static EVALS: std::cell::RefCell<Option<Vec<Eval>>> = const { std::cell::RefCell::new(None) };
    /// The closures running on this thread while `EVALS` is set,
    /// innermost last.
    static OPEN: std::cell::RefCell<Vec<Eval>> = const { std::cell::RefCell::new(Vec::new()) };
}

static EVAL_CLOCK: AtomicU64 = AtomicU64::new(0);

/// `f`'s result with the closure runs that ended on this thread inside it.
fn record<R>(f: impl FnOnce() -> R) -> (R, Vec<Eval>) {
    EVALS.with(|e| *e.borrow_mut() = Some(Vec::new()));
    let r = f();
    let log = EVALS.with(|e| e.borrow_mut().take()).unwrap_or_default();
    (r, log)
}

/// A closure run, open while it runs if the thread records.
struct Running(bool);

impl Running {
    fn start(node: (usize, usize), first: bool) -> Self {
        let on = EVALS.with(|e| e.borrow().is_some());
        if on {
            OPEN.with(|o| {
                o.borrow_mut().push(Eval {
                    node,
                    first,
                    reads: Vec::new(),
                    value: None,
                    end: 0,
                    caught: Vec::new(),
                    returned: Vec::new(),
                })
            });
        }
        Running(on)
    }

    fn returned(&self, v: u64) {
        if self.0 {
            OPEN.with(|o| o.borrow_mut().last_mut().expect("the run is open").value = Some(v));
        }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        if !self.0 {
            return;
        }
        let mut e = OPEN
            .with(|o| o.borrow_mut().pop())
            .expect("the run is open");
        e.end = EVAL_CLOCK.fetch_add(1, SeqCst);
        EVALS.with(|l| {
            if let Some(l) = l.borrow_mut().as_mut() {
                l.push(e);
            }
        });
    }
}

/// [`read_value`], kept on the innermost open run.
fn read_logged<C: Cells>(rt: &Runtime<C>, h: Handle, target: (usize, usize), try_: bool) -> u64 {
    let at = OPEN.with(|o| {
        let mut o = o.borrow_mut();
        let depth = o.len().checked_sub(1)?;
        let reads = &mut o[depth].reads;
        reads.push((target.0, target.1, try_, ReadOut::Unwound));
        Some((depth, reads.len() - 1))
    });
    let Some((depth, k)) = at else {
        return read_value(rt, h, try_);
    };
    let (v, out) = if try_ {
        match try_value(rt, h) {
            Ok(v) => (v, ReadOut::Value(v)),
            Err(c) => {
                let at = EVAL_CLOCK.fetch_add(1, SeqCst);
                OPEN.with(|o| o.borrow_mut()[depth].caught.push((k, at)));
                (
                    7777,
                    ReadOut::Cycle(c.participants().iter().map(|p| p.slot()).collect()),
                )
            }
        }
    } else {
        let v = read_value(rt, h, false);
        (v, ReadOut::Value(v))
    };
    let returned = matches!(out, ReadOut::Value(_)).then(|| EVAL_CLOCK.fetch_add(1, SeqCst));
    OPEN.with(|o| {
        let mut o = o.borrow_mut();
        o[depth].reads[k].3 = out;
        o[depth].returned.extend(returned.map(|at| (k, at)));
    });
    v
}

/// `e` as [`eval`] computes it, with each read taking the next of
/// `reads`: `Ok` with the value, `Err(None)` if it unwinds, or
/// `Err(Some(why))` if a read is not the one `reads` holds next.
fn formula(
    e: &Expr,
    (side, at, first): (usize, usize, bool),
    reads: &mut std::slice::Iter<'_, Read>,
) -> Result<u64, Option<String>> {
    let mut read = |s: usize, target: usize, try_: bool| match reads.next() {
        Some((rs, rt, rtry, out)) if (*rs, *rt, *rtry) == (s, target, try_) => match out {
            ReadOut::Value(v) => Ok(*v),
            ReadOut::Cycle(_) => Ok(7777),
            ReadOut::Unwound => Err(None),
        },
        next => Err(Some(format!(
            "node {at} reads {target} on side {s}, but its run's next read is {next:?}"
        ))),
    };
    let n = (side, at, first);
    match e {
        Expr::Const(v) => Ok(*v),
        Expr::Read(target, try_) => read(side, *target, *try_),
        Expr::ReadOther(target, try_) => read(1 - side, *target, *try_),
        Expr::Add(a, b) => Ok(formula(a, n, reads)?.wrapping_add(formula(b, n, reads)?)),
        Expr::If(c, a, b) => {
            if formula(c, n, reads)? % 2 == 0 {
                formula(a, n, reads)
            } else {
                formula(b, n, reads)
            }
        }
        Expr::PanicIf(c, when) => {
            let v = formula(c, n, reads)?;
            if v % 7 == *when {
                Err(None)
            } else {
                Ok(v)
            }
        }
        Expr::Catch(inner) => match formula(inner, n, reads) {
            Err(None) => Ok(9999),
            other => other,
        },
        Expr::FirstRun(a, b) => formula(if first { a } else { b }, n, reads),
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
        Expr::Read(target, try_) => read_logged(
            n.rt,
            n.world.handle(n.side, *target),
            (n.side, *target),
            *try_,
        ),
        Expr::ReadOther(target, try_) => {
            let other = 1 - n.side;
            read_logged(
                n.world.rt(other),
                n.world.handle(other, *target),
                (other, *target),
                *try_,
            )
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

/// How a node's closure ran before the node is made.
enum Ran {
    Never,
    /// Its last run was not its first, so any run of it is a rerun.
    Rerun,
    /// Its last run was its first and landed Clean over these reads. An
    /// order verifies them and keeps that run's value, so the made
    /// closure's first run takes `FirstRun`'s first branch again unless one
    /// of them now reads differently.
    First(Vec<Read>),
}

/// Whether each of `reads` gets the same kind of outcome, and the same
/// value, read again in order.
fn reads_alike<C: Cells>(world: &World<C>, reads: &[Read]) -> bool {
    reads.iter().all(|&(side, d, try_, ref out)| {
        let (rt, h) = (world.rt(side), world.handle(side, d));
        let again = catch_unwind(AssertUnwindSafe(|| {
            if try_ {
                try_value(rt, h).ok()
            } else {
                Some(read_value(rt, h, false))
            }
        }));
        match (out, again) {
            (ReadOut::Value(v), Ok(Some(w))) => *v == w,
            (ReadOut::Cycle(_), Ok(None)) | (ReadOut::Unwound, Err(_)) => true,
            _ => false,
        }
    })
}

/// Node `i`'s query or input on `side`, its closure having run as `ran`.
fn make<C: Cells>(world: &Arc<World<C>>, side: usize, i: usize, spec: &Spec, ran: Ran) -> Handle {
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
            // A `FirstRun` closure reads differently once it has run, so
            // its Native query stays Native under `INCR_TEST_ALL_REPLAYABLE`,
            // as a collection operator's does: a doomed first run would make
            // the retry take the other branch.
            let pinned = replay == Replay::Native && has_first_run(expr);
            let (expr, world) = (expr.clone(), world.clone());
            let runs = AtomicU32::new(u32::from(!matches!(ran, Ran::Never)));
            let kept = match ran {
                Ran::First(reads) => Some(reads),
                Ran::Never | Ran::Rerun => None,
            };
            let compute = move |rt: &Runtime<C>| {
                let first = match runs.fetch_add(1, Relaxed) {
                    0 => true,
                    1 => kept.as_ref().is_some_and(|r| reads_alike(&world, r)),
                    _ => false,
                };
                let node = Node {
                    rt,
                    world: &world,
                    side,
                    at: i,
                    first,
                };
                let running = Running::start((side, i), node.first);
                let v = eval(&node, &expr);
                running.returned(v);
                v
            };
            if *bare {
                Handle::Bare(rt.create_query_erased::<true, _, _>(
                    move |rt| Bare::made(compute(rt)),
                    erased(
                        pinned,
                        QueryOpts {
                            cycle: strategy.build(|v| Bare(v, false)),
                            replay,
                        },
                    ),
                ))
            } else {
                Handle::Plain(rt.create_query_erased::<true, _, _>(
                    compute,
                    erased(
                        pinned,
                        QueryOpts {
                            cycle: strategy.build(|v| v),
                            replay,
                        },
                    ),
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

/// The caches of one phase's history checks: reads taken by op and node,
/// and runs taken by end and the value they were taken for.
struct History<'a> {
    label: &'a str,
    prefix: &'a Trace,
    p: usize,
    live: &'a Live<Shared>,
    runs: Vec<(usize, Eval)>,
    allowed: HashMap<(usize, usize), HashSet<Read>>,
    taken: HashMap<(u64, Option<u64>), bool>,
}

/// A run's two runtimes, its node tables, and the handles recreates
/// replaced.
struct Live<C: Cells> {
    world: Arc<World<C>>,
    retired: Mutex<Vec<Handle>>,
    queries: [usize; 2],
    /// The closure runs each operation's thread ended, by operation, in
    /// a two-thread run.
    evals: Mutex<HashMap<usize, Vec<Eval>>>,
}

/// After an operation: every query's state, every node's dependencies in
/// the order its last landing installed them, every node's state, indexed
/// as the deps are, and every node's cycle-group record as its members'
/// (slot, generation) pairs, sorted, indexed the same way. Both runtimes'
/// entries, the first's first.
type Snapshot = (
    Vec<NodeState>,
    Vec<Vec<u32>>,
    Vec<NodeState>,
    Vec<Vec<(u32, u32)>>,
);

impl<C: Cells> Live<C> {
    fn new(rts: &[Box<Runtime<C>>; 2], case: &Case) -> Self {
        let table = |n: usize| (0..n).map(|_| Mutex::new(None)).collect();
        let world = Arc::new(World {
            rts: [RtPtr(&*rts[0]), RtPtr(&*rts[1])],
            tables: [table(case.specs.len()), table(case.specs2.len())],
        });
        for (side, specs) in [&case.specs, &case.specs2].into_iter().enumerate() {
            for (i, spec) in specs.iter().enumerate().rev() {
                let h = make(&world, side, i, spec, Ran::Never);
                world.replace(side, i, h);
            }
        }
        Live {
            world,
            retired: Mutex::new(Vec::new()),
            queries: [case.queries, case.queries2],
            evals: Mutex::new(HashMap::new()),
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
                let new = make(&self.world, 0, *i, spec, Ran::Never);
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
    /// the documented ones, no virtual frame left or pushed twice, and no
    /// unbuilt value dropped inside the replay contexts' borrow.
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
        let held = crate::replay::take_held_drops();
        if held != 0 {
            return Err(format!(
                "{held} held values dropped inside the replay contexts' borrow"
            ));
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
        let mut nodes = Vec::new();
        let mut groups = Vec::new();
        for side in 0..2 {
            let rt = w.rt(side);
            for q in 0..self.queries[side] {
                let h = w.handle(side, q);
                let state = if rt.is_alive(h.erased()) {
                    h.state(rt)
                } else {
                    NodeState::Dead
                };
                states.push(state);
            }
            for x in 0..rt.nodes.len() {
                let mut d = Vec::new();
                rt.nodes.get(x).for_each_dep(|n| d.push(n.0));
                deps.push(d);
                nodes.push(rt.nodes.get(x).state());
            }
            let inner = rt.inner.read();
            for x in 0..rt.nodes.len() {
                let mut members =
                    (inner.member_recs.get(&x)).map_or(Vec::new(), |g| g.members.clone());
                members.sort_unstable();
                groups.push(members);
            }
        }
        (states, deps, nodes, groups)
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
        if BARE_PLACEHOLDER.load(SeqCst) {
            rt.set_placeholder(Bare(0, false));
        }
        rt.set_max_frames(frames.unwrap_or(u32::MAX));
        rt.set_max_stack_bytes(bytes);
        rt
    })
}

fn reset_thread() {
    hook();
    INTERNAL.with(|i| i.borrow_mut().take());
    crate::replay::take_vindex_repushes();
    crate::replay::take_held_drops();
}

fn run<C: Cells>(case: &Case, frames: Option<u32>, bytes: usize) -> Trace {
    run_logged::<C>(case, frames, bytes, false).0
}

/// [`run`], with each operation's closure runs if `log`.
fn run_logged<C: Cells>(
    case: &Case,
    frames: Option<u32>,
    bytes: usize,
    log: bool,
) -> (Trace, Vec<Vec<Eval>>) {
    let rts = runtimes::<C>(frames, bytes);
    let live = Live::new(&rts, case);
    reset_thread();
    let (mut trace, mut evals) = (Vec::new(), Vec::new());
    for op in &case.ops {
        let (out, ran) = if log {
            record(|| live.apply(op))
        } else {
            (live.apply(op), Vec::new())
        };
        live.assert_invariants();
        if let Err(m) = live.check_thread() {
            panic!("operation {op:?}: {m}");
        }
        trace.push((out, live.snapshot()));
        evals.push(ran);
    }
    (trace, evals)
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
/// unbounded single-thread run of the case, with either read of each pair
/// first, gives each read's expected result at its sequence point.
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
/// snapshot on its operations.
fn run_two(case: &Case, phases: &[Phase], frames: Option<u32>, bytes: usize) -> Trace {
    run_two_until(case, phases, frames, bytes, &|_| {}, &|_, _, _| false)
}

thread_local! {
    /// Which of a two-thread run's threads this is.
    static TWO_THREAD: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// The phase a two-thread run's thread is in.
    static TWO_PHASE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Thread 0's call at the end of each read phase with two reads, by the
/// phase's index, with the trace up to it, while thread 1 waits: whether
/// the run stops there.
type AtOverlap<'a> = dyn Fn(usize, &Trace, &Live<Shared>) -> bool + Sync + 'a;

/// The extras of a set that their history in the phase grounds.
type Grounded<'a> = dyn Fn(&[(usize, NodeState)]) -> HashSet<usize> + 'a;

/// [`run_two`], with `rig` called on the built world before the threads
/// start, and stopping at the first phase `at_overlap` stops at, the
/// trace ending with it.
fn run_two_until(
    case: &Case,
    phases: &[Phase],
    frames: Option<u32>,
    bytes: usize,
    rig: &dyn Fn(&Live<Shared>),
    at_overlap: &AtOverlap<'_>,
) -> Trace {
    let rts = runtimes::<Shared>(frames, bytes);
    let live = Live::new(&rts, case);
    rig(&live);
    let n = case.ops.len();
    let outs: Mutex<Vec<Option<Out>>> = Mutex::new(vec![None; n]);
    let snaps: Mutex<Vec<Option<Snapshot>>> = Mutex::new(vec![None; phases.len()]);
    let failures: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let stop = AtomicBool::new(false);
    let barrier = Barrier::new(2);
    std::thread::scope(|s| {
        for t in 0..2 {
            let (live, outs, snaps, failures, stop, barrier) =
                (&live, &outs, &snaps, &failures, &stop, &barrier);
            s.spawn(move || {
                reset_thread();
                TWO_THREAD.with(|c| c.set(Some(t)));
                for (p, phase) in phases.iter().enumerate() {
                    barrier.wait();
                    // Thread 0 sets it before this barrier.
                    if stop.load(Relaxed) {
                        break;
                    }
                    if let Some(i) = phase.of(t) {
                        TWO_PHASE.with(|c| c.set(p));
                        let op = &case.ops[i];
                        match catch_unwind(AssertUnwindSafe(|| record(|| live.apply(op)))) {
                            Ok((out, ran)) => {
                                outs.lock().unwrap()[i] = Some(out);
                                live.evals.lock().unwrap().insert(i, ran);
                            }
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
                    if t != 0 {
                        continue;
                    }
                    let invariants = || {
                        if let Err(e) = catch_unwind(AssertUnwindSafe(|| live.assert_invariants()))
                        {
                            failures
                                .lock()
                                .unwrap()
                                .push(format!("after phase {p}: {}", class(e)));
                        }
                    };
                    invariants();
                    snaps.lock().unwrap()[p] = Some(live.snapshot());
                    if phase.ops().len() < 2 || !failures.lock().unwrap().is_empty() {
                        continue;
                    }
                    let prefix = {
                        let (outs, snaps) = (outs.lock().unwrap(), snaps.lock().unwrap());
                        assemble(&phases[..=p], &outs, &snaps)
                    };
                    match catch_unwind(AssertUnwindSafe(|| at_overlap(p, &prefix, live))) {
                        Ok(false) => {}
                        Ok(true) => {
                            stop.store(true, Relaxed);
                            invariants();
                            if let Err(m) = live.check_thread() {
                                failures
                                    .lock()
                                    .unwrap()
                                    .push(format!("after phase {p}: {m}"));
                            }
                        }
                        Err(e) => {
                            stop.store(true, Relaxed);
                            failures
                                .lock()
                                .unwrap()
                                .push(format!("at the end of phase {p}: {}", class(e)));
                        }
                    }
                }
                TWO_THREAD.with(|c| c.set(None));
            });
        }
    });
    let failures = failures.into_inner().unwrap();
    assert!(failures.is_empty(), "{failures:#?}");
    let (outs, snaps) = (outs.into_inner().unwrap(), snaps.into_inner().unwrap());
    let ran = snaps.iter().take_while(|s| s.is_some()).count();
    assemble(&phases[..ran], &outs, &snaps)
}

/// The trace of `phases`, the first of the case's, each operation with its
/// phase's snapshot.
fn assemble(phases: &[Phase], outs: &[Option<Out>], snaps: &[Option<Snapshot>]) -> Trace {
    let mut trace = Vec::new();
    for (phase, snap) in phases.iter().zip(snaps) {
        let snap = snap.as_ref().expect("every phase snapshots");
        for i in phase.ops() {
            debug_assert_eq!(i, trace.len(), "phases run the case in order");
            trace.push((outs[i].clone().expect("every operation ran"), snap.clone()));
        }
    }
    trace
}

/// The case's operations in their phases' order, with the two reads of
/// each read phase whose bit `swaps` sets (bit k for the k-th phase with
/// two) in the other order.
struct Order {
    case: Case,
    /// Each operation's index in the original case.
    of: Vec<usize>,
    /// Each operation's phase's last operation, by index here.
    ends: Vec<usize>,
    /// How many phases with two reads run up to and including each
    /// operation's.
    pairs: Vec<u32>,
}

impl Order {
    fn new(case: &Case, phases: &[Phase], swaps: u64) -> Self {
        let (mut of, mut ends, mut pairs) = (Vec::new(), Vec::new(), Vec::new());
        let mut pair = 0;
        for phase in phases {
            let mut ops = phase.ops();
            if ops.len() == 2 {
                if swaps >> pair & 1 == 1 {
                    ops.swap(0, 1);
                }
                pair += 1;
            }
            let end = of.len() + ops.len() - 1;
            for i in ops {
                of.push(i);
                ends.push(end);
                pairs.push(pair);
            }
        }
        assert!(pair < 64, "a swap mask holds 63 read phases");
        let ops = of.iter().map(|&i| case.ops[i].clone()).collect();
        Order {
            case: Case {
                ops,
                ..case.clone()
            },
            of,
            ends,
            pairs,
        }
    }

    /// A two-thread trace of the original case's first operations, whole
    /// phases, in this order.
    fn arrange(&self, got: &Trace) -> Trace {
        self.of[..got.len()]
            .iter()
            .map(|&i| got[i].clone())
            .collect()
    }

    fn settles(&self) -> Vec<bool> {
        self.ends.iter().enumerate().map(|(j, &e)| j == e).collect()
    }
}

/// The orders of each read phase's two reads, as swap masks, whose
/// unbounded single-thread runs, each kept in `wants`, `got` equals up to
/// its length (whole phases), the first only unless `all`, and the
/// closest that differs. The Failed and Cyclic sets after `got`'s last
/// phase count only if `settle_last`. Under Shared the unbounded run is
/// the one with the landings this thread observes in the order it
/// observes them (T8, the read case table), so either read of a pair may
/// land a node both read first.
#[allow(clippy::too_many_arguments)]
fn orders(
    case: &Case,
    phases: &[Phase],
    wants: &mut HashMap<u64, Trace>,
    got: &Trace,
    label: &str,
    settle_last: bool,
    all: bool,
    per_read: &PerRead,
    cycles: &CycleReads,
) -> (Vec<u64>, Option<(usize, u64, String)>) {
    let mut found = Vec::new();
    let mut swaps = 0;
    let mut closest: Option<(usize, u64, String)> = None;
    loop {
        let order = Order::new(case, phases, swaps);
        let want = wants.entry(swaps).or_insert_with(|| {
            labelled(&format!("{label}, unbounded, swaps {swaps:#b}"), || {
                run::<Shared>(&order.case, None, 0)
            })
        });
        let mut settles = order.settles();
        settles.truncate(got.len());
        if !settle_last {
            settles[got.len() - 1] = false;
        }
        // Every order that runs the same read first in each phase up to
        // operation j's differs at j too, or matches there: search
        // depth-first, the k-th phase with two reads at depth k.
        let arranged = order.arrange(got);
        let per: PerRead = (order.of.iter().enumerate())
            .filter_map(|(j, i)| Some((j, per_read.get(i)?.clone())))
            .collect();
        let cyc: CycleReads = (order.of.iter().enumerate())
            .filter_map(|(j, i)| Some((j, cycles.get(i)?.clone())))
            .collect();
        let diff = first_diff(
            &order.case,
            want,
            &arranged,
            Some(&settles),
            PanicText::Ignored,
            &per,
            &cyc,
        );
        let j = match diff {
            Some((j, what)) => {
                if closest.as_ref().is_none_or(|c| j > c.0) {
                    closest = Some((j, swaps, what));
                }
                j
            }
            None => {
                found.push(swaps);
                if !all {
                    break;
                }
                got.len() - 1
            }
        };
        let Some(k) = (0..order.pairs[j]).rev().find(|&k| swaps >> k & 1 == 0) else {
            break;
        };
        swaps = swaps & ((1 << k) - 1) | 1 << k;
    }
    (found, closest)
}

/// Panics unless the two-thread trace `got` equals the unbounded
/// single-thread run of some order of each read phase's two reads, keeping
/// each order's run in `wants`.
fn assert_serial(
    case: &Case,
    phases: &[Phase],
    wants: &mut HashMap<u64, Trace>,
    got: &Trace,
    label: &str,
) {
    assert_serial_with(
        case,
        phases,
        wants,
        got,
        label,
        &PerRead::new(),
        &CycleReads::new(),
    );
}

/// [`assert_serial`], with the gets `per_read` orders per read and the
/// cycles `cycles` verified.
fn assert_serial_with(
    case: &Case,
    phases: &[Phase],
    wants: &mut HashMap<u64, Trace>,
    got: &Trace,
    label: &str,
    per_read: &PerRead,
    cycles: &CycleReads,
) {
    let (found, closest) = orders(
        case, phases, wants, got, label, true, false, per_read, cycles,
    );
    if !found.is_empty() {
        return;
    }
    let (j, swaps, what) = closest.expect("an order was tried");
    let i = Order::new(case, phases, swaps).of[j];
    panic!(
        "{label}: operation {i} ({:?}) differs from the unbounded run of every order of \
         the overlapping reads, the closest swapping {swaps:#b}\n{what}\n{}",
        case.ops[i],
        describe(case)
    );
}

/// The two-thread oracle of one case, keeping each order's unbounded run
/// across its runs.
struct Serial<'a> {
    case: &'a Case,
    phases: &'a [Phase],
    wants: Mutex<HashMap<u64, Trace>>,
    /// Each order's closure runs, by operation, for [`Self::per_read`].
    logs: Mutex<HashMap<u64, Vec<Vec<Eval>>>>,
    per_read: Mutex<PerRead>,
    cycles: Mutex<CycleReads>,
}

impl<'a> Serial<'a> {
    fn new(case: &'a Case, phases: &'a [Phase]) -> Self {
        Serial {
            case,
            phases,
            wants: Mutex::new(HashMap::new()),
            logs: Mutex::new(HashMap::new()),
            per_read: Mutex::new(PerRead::new()),
            cycles: Mutex::new(CycleReads::new()),
        }
    }

    /// At the end of a read phase with two reads: `None` while an order's
    /// unbounded run gives `prefix`, its Failed and Cyclic sets included,
    /// or while none gives its outcomes, which [`Self::check`] reports.
    /// Otherwise the phase's Failed and Cyclic nodes are history, as its
    /// states are (T8, the read case table: a walk computes from its plan,
    /// so a node can land Failed from a dep another thread landed Failed
    /// after the walk expanded it), and each must instead be true of its
    /// inputs, by [`consistent`], which changes the world.
    fn at_overlap(
        &self,
        label: &str,
        prefix: &Trace,
        live: &Live<Shared>,
    ) -> Option<Result<(), String>> {
        let (case, phases) = (self.case, self.phases);
        self.per_read(label, prefix, live);
        self.cycle_reads(prefix, live);
        let per_read = self.per_read.lock().unwrap().clone();
        let cycles = self.cycles.lock().unwrap().clone();
        let mut wants = self.wants.lock().unwrap();
        if !orders(
            case, phases, &mut wants, prefix, label, true, false, &per_read, &cycles,
        )
        .0
        .is_empty()
        {
            return None;
        }
        let (found, _) = orders(
            case, phases, &mut wants, prefix, label, false, true, &per_read, &cycles,
        );
        if found.is_empty() {
            return None;
        }
        let end = prefix.len() - 1;
        let sets: Vec<_> = found.iter().map(|m| settled(&wants[m][end].1 .0)).collect();
        drop(wants);
        let history =
            |extra: &[(usize, NodeState)]| self.history_grounded(label, prefix, live, extra);
        let reads = (phases.iter())
            .find(|ph| ph.ops().last() == Some(&end))
            .map_or_else(Vec::new, Phase::ops);
        let start = (reads.first())
            .filter(|&&i| i > 0)
            .map(|&i| prefix[i - 1].1 .0.as_slice());
        Some(consistent(
            case, prefix, &sets, live, &reads, start, &history,
        ))
    }

    /// Orders per read each get of the read phase `prefix` ends with whose
    /// thread's `try_get` returned a cycle that the other read's get
    /// landed after it: every participant ended the phase Clean, and its
    /// last run that returned was on the other thread and ended after the
    /// `try_get` returned. Cycle results depend on the
    /// member a get enters at, so the get may mix outcomes of both orders.
    /// Its value must still be
    /// its node's closure over its last run's reads, each an outcome some
    /// order of the phase gives that read.
    fn per_read(&self, label: &str, prefix: &Trace, live: &Live<Shared>) {
        let end = prefix.len() - 1;
        let Some(p) = (self.phases.iter()).position(|ph| ph.ops().last() == Some(&end)) else {
            return;
        };
        let phase = &self.phases[p];
        let states = &prefix[end].1 .0;
        let w = &*live.world;
        let spec_of: HashMap<u32, usize> = (0..live.queries[0])
            .map(|x| (w.handle(0, x).erased().slot(), x))
            .collect();
        let evals = live.evals.lock().unwrap().clone();
        let none = Vec::new();
        for t in 0..2 {
            let (Some(i), Some(j)) = (phase.of(t), phase.of(1 - t)) else {
                continue;
            };
            let (Op::Get(q) | Op::TryGet(q)) = self.case.ops[i] else {
                continue;
            };
            if !matches!(self.case.ops[j], Op::Get(_) | Op::TryGet(_)) {
                continue;
            }
            let (mine, theirs) = (
                evals.get(&i).unwrap_or(&none),
                evals.get(&j).unwrap_or(&none),
            );
            let landed_there_after = |slot: &u32, at: u64| {
                spec_of.get(slot).is_some_and(|&x| {
                    states[x] == NodeState::Clean
                        && (mine.iter().chain(theirs))
                            .filter(|e| e.node == (0, x) && e.value.is_some())
                            .max_by_key(|e| e.end)
                            .is_some_and(|e| e.end > at && theirs.iter().any(|o| o.end == e.end))
                })
            };
            let caught = mine.iter().any(|e| {
                e.caught.iter().any(|&(k, at)| {
                    matches!(&e.reads[k], (0, _, true, ReadOut::Cycle(c))
                        if c.iter().all(|slot| landed_there_after(slot, at)))
                })
            });
            let last = (mine.iter().filter(|e| e.node == (0, q))).max_by_key(|e| e.end);
            if let (true, Some(run)) = (caught, last) {
                let verdict = self.closure_over_reads(label, prefix, p, i, q, run);
                self.per_read.lock().unwrap().insert(i, verdict);
            }
        }
    }

    /// Checks each read of the read phase `prefix` ends with that got
    /// `Err(Cycle)`, for [`CycleReads`]. The edges are runtime 0's: each
    /// read a closure run of that operation made, and the dep lists the
    /// phase left, which hold the walk's way through settled nodes.
    fn cycle_reads(&self, prefix: &Trace, live: &Live<Shared>) {
        let end = prefix.len() - 1;
        let Some(p) = (self.phases.iter()).position(|ph| ph.ops().last() == Some(&end)) else {
            return;
        };
        let w = &*live.world;
        let slot = |x: usize| w.handle(0, x).erased().slot();
        let evals = live.evals.lock().unwrap().clone();
        let deps = &prefix[end].1 .1;
        for i in self.phases[p].ops() {
            let (Out::Cycle(parts), Op::Get(q) | Op::TryGet(q)) = (&prefix[i].0, &self.case.ops[i])
            else {
                continue;
            };
            let mut edges: HashMap<u32, Vec<u32>> = HashMap::new();
            for e in evals
                .get(&i)
                .into_iter()
                .flatten()
                .filter(|e| e.node.0 == 0)
            {
                let reads = e.reads.iter().filter(|r| r.0 == 0).map(|r| slot(r.1));
                edges.entry(slot(e.node.1)).or_default().extend(reads);
            }
            for (n, d) in deps.iter().enumerate().take(w.rt(0).nodes.len() as usize) {
                edges.entry(n as u32).or_default().extend(d);
            }
            // The nodes one or more edges away.
            let reach = |from: u32| {
                let mut seen = HashSet::new();
                let mut todo = vec![from];
                while let Some(n) = todo.pop() {
                    for &m in edges.get(&n).into_iter().flatten() {
                        if seen.insert(m) {
                            todo.push(m);
                        }
                    }
                }
                seen
            };
            let target = slot(*q);
            let verdict = if parts.is_empty() {
                Err(format!("op {i}: a cycle with no participant"))
            } else if !parts.contains(&target) && !reach(target).iter().any(|n| parts.contains(n)) {
                Err(format!(
                    "op {i}: no participant of {parts:?} is node {q} or on its path"
                ))
            } else if let Some(a) = parts.iter().find(|&&a| {
                let r = reach(a);
                !parts.iter().all(|b| r.contains(b))
            }) {
                Err(format!(
                    "op {i}: participant {a} of {parts:?} does not reach every participant \
                     over the recorded reads and dep lists"
                ))
            } else {
                Ok(())
            };
            self.cycles.lock().unwrap().insert(i, verdict);
        }
    }

    /// The outcome node `q`'s closure gives over `run`'s reads, for op `i`
    /// of phase `p`, or why a read is no order's outcome for it, an
    /// order's outcome being that of a read in the order's last run of `q`
    /// in op `i`: the orders are those an order of the phases before gives
    /// `prefix` in, with either read of phase `p` first.
    fn closure_over_reads(
        &self,
        label: &str,
        prefix: &Trace,
        p: usize,
        i: usize,
        q: usize,
        run: &Eval,
    ) -> Result<Out, String> {
        let allowed = self.allowed_reads(label, prefix, p, i, q);
        if let Some(r) = run.reads.iter().find(|r| !allowed.contains(*r)) {
            return Err(format!(
                "op {i}: node {q} read {r:?}, which no order gives that read"
            ));
        }
        let Spec::Query { expr, .. } = spec_at(self.case, prefix, 0, q) else {
            return Err(format!("op {i}: node {q} is an input"));
        };
        let mut reads = run.reads.iter();
        let out = match formula(&expr, (0, q, run.first), &mut reads) {
            Ok(v) => Out::Value(v),
            Err(None) => Out::Panic(String::new()),
            Err(Some(why)) => return Err(format!("op {i}: {why}")),
        };
        match reads.next() {
            None => Ok(out),
            Some(r) => Err(format!(
                "op {i}: node {q}'s closure stops before its read {r:?}"
            )),
        }
    }

    /// The reads an order's last run of node `q` in op `i` of phase `p`
    /// makes, over the orders an order of the phases before gives `prefix`
    /// in, with either read of phase `p` first.
    fn allowed_reads(
        &self,
        label: &str,
        prefix: &Trace,
        p: usize,
        i: usize,
        q: usize,
    ) -> HashSet<Read> {
        let (case, phases) = (self.case, self.phases);
        let start = phases[p].ops()[0];
        let pair = phases[..p].iter().filter(|ph| ph.ops().len() == 2).count();
        let masks = if start == 0 {
            vec![0]
        } else {
            let before = prefix[..start].to_vec();
            let per_read = self.per_read.lock().unwrap().clone();
            let cycles = self.cycles.lock().unwrap().clone();
            let mut wants = self.wants.lock().unwrap();
            orders(
                case, phases, &mut wants, &before, label, true, true, &per_read, &cycles,
            )
            .0
        };
        let mut allowed: HashSet<Read> = HashSet::new();
        for m in masks.iter().flat_map(|m| [m & !(1 << pair), m | 1 << pair]) {
            let order = Order::new(case, phases, m);
            let at = order
                .of
                .iter()
                .position(|&x| x == i)
                .expect("every order runs it");
            let mut logs = self.logs.lock().unwrap();
            let log = logs.entry(m).or_insert_with(|| {
                labelled(&format!("{label}, unbounded, swaps {m:#b}, logged"), || {
                    run_logged::<Shared>(&order.case, None, 0, true).1
                })
            });
            let last = (log[at].iter().filter(|e| e.node == (0, q))).max_by_key(|e| e.end);
            allowed.extend(last.into_iter().flat_map(|e| e.reads.iter().cloned()));
        }
        allowed
    }

    /// The runs on runtime 0 of the read phase `prefix` ends with, by op,
    /// for [`Self::run_taken`].
    fn history<'h>(
        &self,
        label: &'h str,
        prefix: &'h Trace,
        live: &'h Live<Shared>,
    ) -> Option<History<'h>> {
        let end = prefix.len() - 1;
        let p = (self.phases.iter()).position(|ph| ph.ops().last() == Some(&end))?;
        let evals = live.evals.lock().unwrap().clone();
        let runs: Vec<(usize, Eval)> = self.phases[p]
            .ops()
            .into_iter()
            .flat_map(|i| {
                evals
                    .get(&i)
                    .into_iter()
                    .flatten()
                    .map(move |e| (i, e.clone()))
            })
            .filter(|(_, e)| e.node.0 == 0)
            .collect();
        Some(History {
            label,
            prefix,
            p,
            live,
            runs,
            allowed: HashMap::new(),
            taken: HashMap::new(),
        })
    }

    /// Node `x`'s last run in `cx`'s phase, with its op.
    fn last_run(cx: &History<'_>, x: usize) -> Option<(usize, Eval)> {
        (cx.runs.iter())
            .filter(|(_, e)| e.node.1 == x)
            .max_by_key(|(_, e)| e.end)
            .cloned()
    }

    /// Whether node `x`'s last run in the read phase `prefix` ends panicked
    /// in its own closure over reads the oracle takes: each one a read an
    /// order's last run of the node makes, the value an input holds, or the
    /// value of an earlier run of the read node in the phase whose closure
    /// gives it over reads the oracle takes in turn.
    fn history_cause(&self, label: &str, prefix: &Trace, live: &Live<Shared>, x: usize) -> bool {
        let Some(mut cx) = self.history(label, prefix, live) else {
            return false;
        };
        Self::last_run(&cx, x).is_some_and(|(i, last)| self.run_taken(&mut cx, i, &last, None))
    }

    /// The Failed nodes of `extra` the read phase `prefix` ends with left
    /// Failed through its own history: one whose last run in the phase
    /// panicked in its own closure over reads the oracle takes
    /// ([`Self::history_cause`]), or whose last run there unwound, over such
    /// reads before it, from a read of such a node or of a node the phase
    /// left Failed that is not in `extra`.
    fn history_grounded(
        &self,
        label: &str,
        prefix: &Trace,
        live: &Live<Shared>,
        extra: &[(usize, NodeState)],
    ) -> HashSet<usize> {
        let mut grounded = HashSet::new();
        let Some(mut cx) = self.history(label, prefix, live) else {
            return grounded;
        };
        let failed: Vec<(usize, Eval)> = (extra.iter())
            .filter(|&&(x, s)| s == NodeState::Failed && x < live.queries[0])
            .filter_map(|&(x, _)| Self::last_run(&cx, x))
            .collect();
        for (i, run) in &failed {
            if self.run_taken(&mut cx, *i, run, None) {
                grounded.insert(run.node.1);
            }
        }
        let states = &prefix.last().expect("a phase ran").1 .0;
        loop {
            let mut more = Vec::new();
            for (i, run) in &failed {
                let x = run.node.1;
                let Some(&(0, y, _, ReadOut::Unwound)) = run.reads.last() else {
                    continue;
                };
                let through = grounded.contains(&y)
                    || y < live.queries[0]
                        && states[y] == NodeState::Failed
                        && !extra.contains(&(y, NodeState::Failed));
                if !grounded.contains(&x) && through && self.unwound_taken(&mut cx, *i, run) {
                    more.push(x);
                }
            }
            if more.is_empty() {
                return grounded;
            }
            grounded.extend(more);
        }
    }

    /// Whether `run`, of op `i`, ends unwound by its last read, every read
    /// before it one the oracle takes.
    fn unwound_taken(&self, cx: &mut History<'_>, i: usize, run: &Eval) -> bool {
        let q = run.node.1;
        let Spec::Query { expr, .. } = spec_at(self.case, cx.prefix, 0, q) else {
            return false;
        };
        let mut reads = run.reads.iter();
        let unwinds = formula(&expr, (0, q, run.first), &mut reads) == Err(None);
        unwinds
            && reads.next().is_none()
            && (0..run.reads.len() - 1).all(|k| self.read_taken(cx, i, run, k))
    }

    /// Whether `run`, of op `i`, gives `value` (its own panic for `None`)
    /// over reads the oracle takes, as [`Self::history_cause`] takes them.
    fn run_taken(&self, cx: &mut History<'_>, i: usize, run: &Eval, value: Option<u64>) -> bool {
        if let Some(&t) = cx.taken.get(&(run.end, value)) {
            return t;
        }
        cx.taken.insert((run.end, value), false);
        let q = run.node.1;
        let gives = match spec_at(self.case, cx.prefix, 0, q) {
            Spec::Query { expr, .. } => {
                let mut reads = run.reads.iter();
                let out = formula(&expr, (0, q, run.first), &mut reads);
                reads.next().is_none()
                    && match (out, value) {
                        (Ok(v), Some(want)) => v == want,
                        (Err(None), None) => !run.reads.iter().any(|r| r.3 == ReadOut::Unwound),
                        _ => false,
                    }
            }
            Spec::Input(_) => false,
        };
        let taken = gives && (0..run.reads.len()).all(|k| self.read_taken(cx, i, run, k));
        cx.taken.insert((run.end, value), taken);
        taken
    }

    /// Whether the oracle takes read `k` of `run`, of op `i`, as
    /// [`Self::history_cause`] takes it: a value read's earlier run must
    /// have ended before the read returned.
    fn read_taken(&self, cx: &mut History<'_>, i: usize, run: &Eval, k: usize) -> bool {
        let (q, r) = (run.node.1, &run.reads[k]);
        let &(side, d, _, ref out) = r;
        if side != 0 {
            return false;
        }
        let allowed = match cx.allowed.get(&(i, q)) {
            Some(a) => a,
            None => {
                let a = self.allowed_reads(cx.label, cx.prefix, cx.p, i, q);
                cx.allowed.entry((i, q)).or_insert(a)
            }
        };
        if allowed.contains(r) {
            return true;
        }
        let ReadOut::Value(v) = *out else {
            return false;
        };
        if let Spec::Input(_) = spec_at(self.case, cx.prefix, 0, d) {
            let w = &*cx.live.world;
            return read_out(w.rt(0), w.handle(0, d), false) == Out::Value(v);
        }
        let Some(&(_, at)) = run.returned.iter().find(|&&(j, _)| j == k) else {
            return false;
        };
        let earlier: Vec<(usize, Eval)> = (cx.runs.iter())
            .filter(|(_, f)| f.node.1 == d && f.value == Some(v) && f.end < at)
            .cloned()
            .collect();
        earlier
            .iter()
            .any(|(j, f)| self.run_taken(cx, *j, f, Some(v)))
    }

    /// Panics unless the run's trace `got` is an order's unbounded run, or
    /// `at_overlap` stopped the run with a `verdict` that holds.
    fn check(&self, label: &str, got: &Trace, verdict: Option<Result<(), String>>) {
        match verdict {
            None => assert_serial_with(
                self.case,
                self.phases,
                &mut self.wants.lock().unwrap(),
                got,
                label,
                &self.per_read.lock().unwrap(),
                &self.cycles.lock().unwrap(),
            ),
            Some(Ok(())) => {}
            Some(Err(why)) => {
                let i = got.len() - 1;
                panic!(
                    "{label}: the Failed and Cyclic nodes after operation {i} ({:?}) are no \
                     order's and not true of their inputs\n{why}\n{}",
                    self.case.ops[i],
                    describe(self.case)
                )
            }
        }
    }

    /// One two-thread run, `rig` as [`run_two_until`] takes it, checked.
    fn differential(
        &self,
        label: &str,
        frames: Option<u32>,
        bytes: usize,
        rig: &dyn Fn(&Live<Shared>),
    ) {
        self.per_read.lock().unwrap().clear();
        self.cycles.lock().unwrap().clear();
        let verdict = Mutex::new(None);
        let got = labelled(label, || {
            run_two_until(
                self.case,
                self.phases,
                frames,
                bytes,
                rig,
                &|_, prefix, live| {
                    let v = self.at_overlap(label, prefix, live);
                    let stop = v.is_some();
                    *verdict.lock().unwrap() = v;
                    stop
                },
            )
        });
        self.check(label, &got, verdict.into_inner().unwrap());
    }
}

/// Whether the Failed and Cyclic nodes after `prefix`'s last phase are
/// true of their inputs for one of `sets`, each the Failed and Cyclic set
/// an order with the trace's outcomes leaves, `start` being the states
/// before the phase. Each node the phase left Failed or Cyclic where the
/// order does not must land so again when the phase's reads replay, in one
/// order for the whole set, over fresh queries in place of every query the
/// phase may have changed, unless the phase's own history left it Failed
/// (`history`, [`Serial::history_grounded`]), which it keeps. A cycle
/// whose members follow provisional values may close over other members in
/// the replay than in the phase. A node the order settles that the phase
/// left Dirty or New must still read so on a later `get`: a panic for
/// Failed, the cycle for Cyclic. The replays and gets run on this thread,
/// in that order.
fn consistent(
    case: &Case,
    prefix: &Trace,
    sets: &[Vec<(usize, NodeState)>],
    live: &Live<Shared>,
    phase_reads: &[usize],
    start: Option<&[NodeState]>,
    history: &Grounded<'_>,
) -> Result<(), String> {
    let w = &*live.world;
    let states = &prefix.last().expect("a phase ran").1 .0;
    let got = settled(states);
    let node = |x: usize| {
        let side = usize::from(x >= live.queries[0]);
        let q = x - side * live.queries[0];
        (side, q, w.handle(side, q))
    };
    let mut extra: Vec<(usize, NodeState)> = Vec::new();
    let mut missing: Vec<(usize, NodeState)> = Vec::new();
    for set in sets {
        extra.extend(got.iter().filter(|g| !set.contains(g)));
        missing.extend(set.iter().filter(|s| !got.contains(s)));
    }
    extra.sort_unstable_by_key(|&(x, _)| x);
    extra.dedup();
    missing.sort_unstable_by_key(|&(x, _)| x);
    missing.dedup();

    // Each replay reads only inputs, nodes Clean before and after the
    // phase, and kept nodes as the phase left them: no state under judgement.
    let kept = history(&extra);
    let fresh: Vec<usize> = (0..states.len())
        .filter(|&x| {
            let (side, q, _) = node(x);
            let clean = |s: NodeState| s == NodeState::Clean;
            states[x] != NodeState::Dead
                && !kept.contains(&x)
                && matches!(spec_at(case, prefix, side, q), Spec::Query { .. })
                && !(clean(states[x]) && start.is_some_and(|st| clean(st[x])))
        })
        .collect();
    // A fresh query's closure runs as it last ran before the phase, since
    // its last recreate (`Ran`). A first run is kept only if it landed
    // Clean: a Failed or poisoned landing leaves nothing an order verifies.
    let first_op = phase_reads.first().copied().unwrap_or(0);
    let evals = live.evals.lock().unwrap().clone();
    let ran = |x: usize| {
        let (side, q, _) = node(x);
        let made = (0..first_op).rev().find(|&i| {
            side == 0
                && matches!((&case.ops[i], &prefix[i].0), (Op::Recreate(r, _), Out::Recreated(_)) if *r == q)
        });
        let last = (made.map_or(0, |i| i + 1)..first_op)
            .filter_map(|i| evals.get(&i).map(|es| (i, es)))
            .flat_map(|(i, es)| es.iter().map(move |e| (i, e)))
            .filter(|(_, e)| e.node == (side, q))
            .max_by_key(|(_, e)| e.end);
        match last {
            None => Ran::Never,
            Some((i, l)) if l.first && prefix[i].1 .0[x] == NodeState::Clean => {
                Ran::First(l.reads.clone())
            }
            Some(_) => Ran::Rerun,
        }
    };
    let replay = |reads: &[usize]| {
        let reruns: Vec<(usize, usize, Handle, Handle)> = (fresh.iter())
            .map(|&x| {
                let (side, q, _) = node(x);
                let spec = spec_at(case, prefix, side, q);
                let rerun = make(&live.world, side, q, &spec, ran(x));
                (side, q, w.replace(side, q, rerun).expect("built"), rerun)
            })
            .collect();
        for &i in reads {
            let _ = catch_unwind(AssertUnwindSafe(|| live.apply(&case.ops[i])));
        }
        for &(side, q, h, _) in &reruns {
            w.replace(side, q, h);
        }
        let mut off: HashMap<usize, String> = HashMap::new();
        for (&x, &(side, _, _, rerun)) in fresh.iter().zip(&reruns) {
            let landed = w.rt(side).nodes.get(rerun.erased().slot()).state();
            if let Some(&(_, s)) = extra.iter().find(|&&(y, s)| y == x && s != landed) {
                off.insert(
                    x,
                    format!("node {x} {s:?}: a replay of the phase's reads landed it {landed:?}"),
                );
            }
        }
        off
    };
    let mut reads = phase_reads.to_vec();
    let mut replays = vec![replay(&reads)];
    reads.reverse();
    replays.push(replay(&reads));
    let mut after: HashMap<usize, Result<Out, String>> = HashMap::new();
    for &(x, _) in &missing {
        if after.contains_key(&x) {
            continue;
        }
        let left = states[x];
        let read = if matches!(left, NodeState::Dirty | NodeState::New) {
            let (side, _, h) = node(x);
            Ok(read_out(w.rt(side), h, false))
        } else {
            Err(format!("the phase left it {left:?}"))
        };
        after.insert(x, read);
    }

    let mut first = None;
    for set in sets {
        let mut wrong = (replays.iter())
            .map(|off| {
                (got.iter().filter(|g| !set.contains(g)))
                    .filter_map(|g| off.get(&g.0).cloned())
                    .collect::<Vec<_>>()
            })
            .min_by_key(Vec::len)
            .expect("two replays");
        for &(x, s) in set.iter().filter(|s| !got.contains(s)) {
            // A get of a Failed node panics and one of a Cyclic node gets
            // the cycle; a walk that meets a Failed dep panics before the
            // node's compute, which leaves it Dirty.
            match (&after[&x], s) {
                (Ok(Out::Panic(_)), NodeState::Failed) => {}
                (Ok(Out::Panic(c)), NodeState::Cyclic) if c == "cycle" => {}
                (Ok(out), _) => wrong.push(format!(
                    "node {x}: an order leaves it {s:?}, but a later get returned {out:?}"
                )),
                (Err(m), _) => wrong.push(format!("node {x}: an order leaves it {s:?}, but {m}")),
            }
        }
        if wrong.is_empty() {
            return Ok(());
        }
        first.get_or_insert(wrong.join("\n"));
    }
    Err(first.expect("an order has the trace's outcomes"))
}

/// Node `q`'s spec on `side` after `prefix`: the last recreate's that
/// took, or the case's.
fn spec_at(case: &Case, prefix: &Trace, side: usize, q: usize) -> Spec {
    if side == 1 {
        return case.specs2[q].clone();
    }
    prefix
        .iter()
        .zip(&case.ops[..prefix.len()])
        .rev()
        .find_map(|((out, _), op)| match (op, out) {
            (Op::Recreate(i, spec), Out::Recreated(_)) if *i == q => Some(spec.clone()),
            _ => None,
        })
        .unwrap_or_else(|| case.specs[q].clone())
}

/// What a get ordered per read must return, by operation: the closure
/// over its run's reads, or why those reads are no orders' outcomes.
type PerRead = HashMap<usize, Result<Out, String>>;

/// By operation, for a read that got `Err(Cycle)` on two threads: whether
/// its participants include the read's target or a node on its path and
/// each reaches every other over the reads its closure runs recorded and
/// the dep lists the phase left, or why not.
type CycleReads = HashMap<usize, Result<(), String>>;

/// How [`first_diff`] compares two panics.
#[derive(Copy, Clone, Debug)]
enum PanicText {
    Exact,
    /// Any two panics are equal: under Shared with racing threads, which
    /// reader of a panicked node gets the closure's payload and which gets
    /// "last compute panicked" is not part of the result.
    Ignored,
}

impl PanicText {
    fn same(self, a: &Out, b: &Out) -> bool {
        match (self, a, b) {
            (PanicText::Ignored, Out::Panic(_), Out::Panic(_)) => true,
            _ => a == b,
        }
    }
}

/// The first operation at which `got` differs from the unbounded run's
/// `want`, with both sides of it: in its outcome, or, after the operations
/// `settles` marks, in any node's state or recorded deps, and without
/// `settles` in any node's cycle-group record too. An exemption excuses one
/// operation's outcome and nothing after it.
fn first_diff(
    case: &Case,
    want: &Trace,
    got: &Trace,
    settles: Option<&[bool]>,
    text: PanicText,
    per_read: &PerRead,
    cycles: &CycleReads,
) -> Option<(usize, String)> {
    for (i, ((want_out, want), (got_out, got))) in want.iter().zip(got).enumerate() {
        if speculative_delete(&case.ops[i], want_out, got_out) {
            SPECULATIVE_DELETES.fetch_add(1, Relaxed);
            return None;
        }
        // Under two threads the cycle a read reports is built from this
        // thread's frames, so members another thread landed first are left
        // out of it: a verified part of an order's participants matches.
        let same = |want: &Out, got: &Out| {
            text.same(want, got)
                || matches!(
                    (text, want, got, cycles.get(&i)),
                    (PanicText::Ignored, Out::Cycle(w), Out::Cycle(g), Some(Ok(())))
                        if !g.is_empty() && g.iter().all(|p| w.contains(p))
                )
        };
        let wrong_out = if same(want_out, got_out) {
            None
        } else {
            // A get ordered per read: its outcome is its closure's over its
            // reads.
            match per_read.get(&i) {
                Some(Ok(out)) if same(out, got_out) => None,
                Some(Ok(out)) => Some(format!(
                    "\nordered per read: its closure over its reads gives {out:?}"
                )),
                Some(Err(why)) => Some(format!("\nordered per read: {why}")),
                None => Some(match cycles.get(&i) {
                    Some(Err(why)) => format!("\n{why}"),
                    _ => String::new(),
                }),
            }
        };
        let wrong_states = settles
            .is_none_or(|s| s[i])
            .then(|| states_diff(want, got, settles.is_none()))
            .flatten();
        if wrong_out.is_some() || wrong_states.is_some() {
            let (w, g) = ((want_out, settled(&want.0)), (got_out, settled(&got.0)));
            let states = wrong_states.map_or(String::new(), |d| format!("\n{d}"));
            let why = wrong_out.unwrap_or_default();
            return Some((i, format!("want {w:?}\n got {g:?}{states}{why}")));
        }
    }
    None
}

/// T8-speculative-landings, which removes this rule and the next: a
/// speculative run computes nodes the unbounded run never computes and
/// lands them Clean with their deps and group records, so a node New there
/// may be Clean, or Dirty after a later set, with deps and a record here.
fn speculative_landing(want: NodeState, got: NodeState) -> bool {
    want == NodeState::New && matches!(got, NodeState::Clean | NodeState::Dirty)
}

/// T8-speculative-landings: such a landed node's edge refuses a delete the
/// unbounded run makes. Up to it every difference was a speculative
/// landing, so the edge is one of theirs; past it the runs hold different
/// graphs, so the comparison of the case ends there, counted in
/// [`SPECULATIVE_DELETES`].
fn speculative_delete(op: &Op, want: &Out, got: &Out) -> bool {
    matches!(
        (op, want, got),
        (Op::Delete(_), Out::Deleted(Ok(())), Out::Deleted(Err(e))) if e == "HasDependents"
    )
}

static SPECULATIVE_DELETES: AtomicUsize = AtomicUsize::new(0);

/// The nodes whose state, recorded deps or, with `with_groups`, cycle-group
/// record differ between two snapshots, past the speculative landings.
/// The two-thread oracle leaves groups out, so that a record alone never
/// sends a phase's end to [`consistent`] in place of a matching order.
fn states_diff(want: &Snapshot, got: &Snapshot, with_groups: bool) -> Option<String> {
    let states: Vec<String> = (want.0.iter().zip(&got.0).enumerate())
        .filter(|(_, (w, g))| w != g && !speculative_landing(**w, **g))
        .map(|(x, (w, g))| format!("{x}: {w:?} for {g:?}"))
        .collect();
    let deps: Vec<String> = (want.1.iter().zip(&got.1).enumerate())
        .filter(|&(x, (w, g))| {
            w != g && !(w.is_empty() && speculative_landing(want.2[x], got.2[x]))
        })
        .map(|(x, (w, g))| format!("{x}: {w:?} for {g:?}"))
        .collect();
    let groups: Vec<String> = (want.3.iter().zip(&got.3).enumerate())
        .filter(|&(x, (w, g))| {
            with_groups && w != g && !(w.is_empty() && speculative_landing(want.2[x], got.2[x]))
        })
        .map(|(x, (w, g))| format!("{x}: {w:?} for {g:?}"))
        .collect();
    (!states.is_empty() || !deps.is_empty() || !groups.is_empty()).then(|| {
        format!(
            "states differ [{}], deps differ [{}], groups differ [{}]",
            states.join(", "),
            deps.join(", "),
            groups.join(", ")
        )
    })
}

#[test]
fn states_diff_reports_group_records_off_the_two_thread_oracle() {
    let snap = |state, group| -> Snapshot { (vec![], vec![vec![]], vec![state], vec![group]) };
    let (want, got) = (
        snap(NodeState::Clean, vec![(0, 1)]),
        snap(NodeState::Clean, vec![]),
    );
    assert_eq!(
        states_diff(&want, &got, true).as_deref(),
        Some("states differ [], deps differ [], groups differ [0: [(0, 1)] for []]")
    );
    assert_eq!(states_diff(&want, &got, false), None);
    let (want, got) = (
        snap(NodeState::Clean, vec![]),
        snap(NodeState::Clean, vec![(0, 1)]),
    );
    assert!(states_diff(&want, &got, true).is_some());
    let (want, got) = (
        snap(NodeState::New, vec![]),
        snap(NodeState::Clean, vec![(0, 1)]),
    );
    assert_eq!(states_diff(&want, &got, true), None);
}

/// Panics, naming the case, at the first operation at which `got` differs
/// from the unbounded run's `want`.
fn compare(case: &Case, want: &Trace, got: &Trace, label: &str) {
    if let Some((i, what)) = first_diff(
        case,
        want,
        got,
        None,
        PanicText::Exact,
        &PerRead::new(),
        &CycleReads::new(),
    ) {
        panic!(
            "{label}: operation {i} ({:?}) differs from the unbounded run\n{what}\n{}",
            case.ops[i],
            describe(case)
        );
    }
}

fn describe(case: &Case) -> String {
    format!(
        "ops: {:?}\ncase: {:#?}\nsecond runtime: {:#?}",
        case.ops, case.specs, case.specs2
    )
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

/// The budgets a single-thread differential compares with the unbounded
/// run: each frame budget with and without a byte budget, and the byte
/// budget alone.
const BUDGETS: [(Option<u32>, usize); 7] = [
    (Some(1), 0),
    (Some(1), 16 * 1024),
    (Some(2), 0),
    (Some(2), 16 * 1024),
    (Some(3), 0),
    (Some(3), 16 * 1024),
    (None, 16 * 1024),
];

fn budget_label(seed: u64, frames: Option<u32>, bytes: usize) -> String {
    match frames {
        Some(frames) => format!("seed {seed:#x}, budget {frames} frames, {bytes} bytes"),
        None => format!("seed {seed:#x}, budget None frames, {bytes} bytes"),
    }
}

fn differential<C: Cells>(case: &Case, seed: u64) {
    let unbounded = labelled(&format!("seed {seed:#x}, unbounded"), || {
        run::<C>(case, None, 0)
    });
    for (frames, bytes) in BUDGETS {
        let label = budget_label(seed, frames, bytes);
        let mut bounded = labelled(&label, || run::<C>(case, frames, bytes));
        // A forced revisit re-walks its entry, and a bounded run's retried
        // walks force it again at their own points, so only the outputs
        // are results.
        if super::forced_revisit() && !C::SINGLE_THREADED {
            for ((_, got), (_, want)) in bounded.iter_mut().zip(&unbounded) {
                *got = want.clone();
            }
        }
        compare(case, &unbounded, &bounded, &label);
    }
}

/// The two-thread form: every phase schedule at every budget, unbounded
/// included, against the unbounded single-thread run of some order of each
/// phase's two reads, by [`Serial`].
fn differential_two(case: &Case, seed: u64) {
    let phases = phases(case, seed);
    let serial = Serial::new(case, &phases);
    for frames in [None, Some(1), Some(2), Some(3)] {
        for bytes in [0, 16 * 1024] {
            let label = format!(
                "seed {seed:#x}, two threads {phases:?}, budget {frames:?} frames, {bytes} bytes"
            );
            serial.differential(&label, frames, bytes, &|_| {});
        }
    }
}

/// Seeds that once failed, each with what it caught.
#[test]
#[cfg_attr(miri, ignore)]
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

/// Wave 8a lead 0x56cd8c2, shrunk. 5 reads 6, which `try_get`s 7, a
/// bare Fixpoint member (initial 3, limit 1) that `try_get`s 8 and reads
/// it; 8 reads 10, which reads 5 and then 12 or `try_get`s 9, and 9
/// panics on 8's value 0. After `Get(4)` and a set of 12, `TryGet(10)`
/// panics "boom 0:9" in both runs, but at budget 1 the bounded run left
/// 5 to 8 out of the Failed set the unbounded run lands: 8's compute in
/// 7's walk of 8 was doomed, and 7's retry took the walk's rest but found
/// 8 held, so the walk never ran 9 and 10 failed outside 7's episode.
fn seed_56cd8c2_case() -> Case {
    let query = |expr, strategy, bare| Spec::Query {
        expr,
        replay: true,
        bare,
        strategy,
    };
    let read = |n, try_| Box::new(Expr::Read(n, try_));
    let zero = || query(Expr::Const(0), Strat::None, false);
    Case {
        specs: vec![
            zero(),
            zero(),
            zero(),
            zero(),
            query(
                Expr::Add(
                    read(7, true),
                    Box::new(Expr::Add(read(5, true), read(5, false))),
                ),
                Strat::None,
                false,
            ),
            query(Expr::Read(6, false), Strat::None, false),
            query(Expr::Read(7, true), Strat::None, false),
            query(
                Expr::Add(read(8, true), read(8, false)),
                Strat::Fixpoint {
                    initial: 3,
                    limit: 1,
                },
                true,
            ),
            query(Expr::Read(10, false), Strat::None, false),
            query(Expr::PanicIf(read(8, false), 0), Strat::None, false),
            query(
                Expr::If(read(5, false), read(12, false), read(9, true)),
                Strat::None,
                false,
            ),
            Spec::Input(4),
            Spec::Input(2),
        ],
        panics: true,
        cycles: true,
        queries: 11,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(4), Op::Set(12, 1), Op::TryGet(10)],
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn seed_56cd8c2_keeps_the_failed_set() {
    let case = seed_56cd8c2_case();
    quiet(|| {
        differential::<Local>(&case, 0x56c_d8c2);
        differential::<Shared>(&case, 0x56c_d8c2);
    });
}

/// 0xe126d92b8f, shrunk. 9 (Fallback) reads 4, which `try_get`s 5, and 5
/// reaches 9 again through 6. In `Get(9)` the walk of Fixpoint member 2's
/// run computes 9 twice at the same position and path, and at budget 3
/// each run's read of 4 dooms it. Both reads have one key, so the second
/// doom's delivery replaced the first, the member's restart gave it to
/// the first run, and the second run doomed 4 again, forever.
fn seed_e126d92b8f_case() -> Case {
    let q = |expr, replay, bare, strategy| Spec::Query {
        expr,
        replay,
        bare,
        strategy,
    };
    let r = |n, t| Box::new(Expr::Read(n, t));
    let c = |v| Box::new(Expr::Const(v));
    Case {
        specs: vec![
            q(Expr::Read(1, false), true, false, Strat::None),
            q(Expr::Read(2, false), true, false, Strat::None),
            q(
                Expr::If(r(8, false), r(1, true), c(0)),
                true,
                false,
                Strat::Fixpoint {
                    initial: 2,
                    limit: 3,
                },
            ),
            q(Expr::Const(3), false, false, Strat::None),
            q(Expr::Read(5, true), false, false, Strat::None),
            q(
                Expr::If(
                    Box::new(Expr::Add(r(10, false), c(1))),
                    Box::new(Expr::If(r(6, false), c(0), r(7, false))),
                    c(0),
                ),
                true,
                true,
                Strat::None,
            ),
            q(Expr::Read(9, false), true, false, Strat::None),
            q(
                Expr::If(r(2, false), r(8, true), c(0)),
                false,
                false,
                Strat::None,
            ),
            q(Expr::Read(9, true), true, false, Strat::None),
            q(
                Expr::Add(c(1), r(4, false)),
                true,
                false,
                Strat::Fallback(1),
            ),
            q(
                Expr::If(r(11, false), r(10, true), c(1)),
                true,
                false,
                Strat::None,
            ),
            q(Expr::Read(12, false), true, true, Strat::None),
            Spec::Input(1),
            Spec::Input(0),
            Spec::Input(3),
        ],
        panics: false,
        cycles: true,
        queries: 12,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::TryGet(1), Op::Set(12, 2), Op::Get(0), Op::Get(9)],
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn seed_e126d92b8f_repeated_runs_keep_their_deliveries() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let case = seed_e126d92b8f_case();
        let r = catch_unwind(AssertUnwindSafe(|| {
            quiet(|| {
                differential::<Local>(&case, 0xe1_26d9_2b8f);
                differential::<Shared>(&case, 0xe1_26d9_2b8f);
            });
        }));
        let _ = tx.send(r);
    });
    match rx.recv_timeout(std::time::Duration::from_secs(60)) {
        Ok(r) => r.unwrap_or_else(|p| std::panic::resume_unwind(p)),
        Err(_) => panic!("seed 0xe126d92b8f: a bounded run never returned"),
    }
}

/// 0x16d87, shrunk: a fresh `TryGet(18)`, which unbounded gets a cycle of
/// six participants. At budget 1, a speculative run's walk memoized 10
/// under the frame at position 1, so 11's later read of 10 computed it
/// directly where the unbounded run walks it into the episode rooted at
/// 10, and the cycle came out as three.
#[test]
#[cfg_attr(miri, ignore)]
fn seed_16d87_takes_no_speculative_memo() {
    use Expr::{Add, Catch, Const, If, PanicIf, Read};
    let b = Box::new;
    let query = |expr, replay, strategy| Spec::Query {
        expr,
        replay,
        bare: false,
        strategy,
    };
    let zero = |replay| query(Const(0), replay, Strat::None);
    let mut specs: Vec<Spec> = (0..25).map(|_| zero(true)).collect();
    specs[0] = query(Const(2), false, Strat::None);
    specs[2] = query(
        If(b(Read(17, false)), b(Const(0)), b(Const(0))),
        true,
        Strat::Fallback(0),
    );
    specs[4] = query(
        Add(b(Read(5, false)), b(Read(10, false))),
        true,
        Strat::None,
    );
    specs[5] = query(Add(b(Read(8, true)), b(Read(2, false))), true, Strat::None);
    specs[7] = query(
        If(b(Read(8, false)), b(Const(0)), b(Read(16, false))),
        true,
        Strat::None,
    );
    specs[8] = query(Catch(b(Read(9, true))), true, Strat::None);
    specs[9] = query(Read(12, false), true, Strat::None);
    specs[10] = query(
        Catch(b(If(b(Read(13, false)), b(Const(0)), b(Const(0))))),
        true,
        Strat::None,
    );
    specs[12] = query(Read(13, false), true, Strat::None);
    specs[13] = query(Read(14, false), true, Strat::None);
    specs[14] = query(
        Add(b(Read(7, false)), b(Read(15, false))),
        true,
        Strat::None,
    );
    specs[15] = query(
        If(
            b(Add(b(Read(22, false)), b(PanicIf(b(Read(16, false)), 3)))),
            b(Const(0)),
            b(Const(0)),
        ),
        true,
        Strat::None,
    );
    specs[16] = query(
        If(
            b(If(
                b(If(b(Read(17, false)), b(Const(0)), b(Const(0)))),
                b(Const(1)),
                b(Const(1)),
            )),
            b(Const(0)),
            b(Read(17, false)),
        ),
        true,
        Strat::Fallback(3),
    );
    specs[17] = query(Read(23, false), true, Strat::None);
    specs[18] = query(
        If(
            b(If(
                b(If(b(Read(2, true)), b(Read(23, false)), b(Const(0)))),
                b(Const(0)),
                b(Const(0)),
            )),
            b(Const(0)),
            b(Const(0)),
        ),
        true,
        Strat::None,
    );
    specs[21] = query(Const(2), true, Strat::None);
    specs[22] = query(Read(23, true), true, Strat::None);
    specs[23] = query(Read(24, false), false, Strat::None);
    specs[24] = query(
        If(b(Read(4, false)), b(Const(0)), b(Const(0))),
        true,
        Strat::Fixpoint {
            initial: 0,
            limit: 3,
        },
    );
    specs.extend([2, 4, 2].map(Spec::Input));
    let case = Case {
        specs,
        panics: true,
        cycles: true,
        queries: 25,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::TryGet(18)],
    };
    quiet(|| {
        differential::<Local>(&case, 0x16d87);
        differential::<Shared>(&case, 0x16d87);
    });
}

/// 0x56cd8c2's read of held 8 goes past `cycle_read` only to resume its
/// walk: each skip is a resume `clean_walk` then takes, so no read skips
/// twice, and the resumed entry 9 still reads held 8 as the unbounded run
/// does, a cycle unwind and then the in-cycle value.
#[test]
#[cfg_attr(miri, ignore)]
fn resuming_past_a_held_root_skips_no_cycle_read() {
    fn nine<C: Cells>(frames: Option<u32>) -> (Vec<Vec<Read>>, [u64; 2]) {
        let case = seed_56cd8c2_case();
        let rts = runtimes::<C>(frames, 0);
        let live = Live::new(&rts, &case);
        reset_thread();
        let mut reads = Vec::new();
        for op in &case.ops {
            let (_, ran) = record(|| live.apply(op));
            reads = ran
                .into_iter()
                .filter(|e| e.node == (0, 9))
                .map(|e| e.reads)
                .collect();
        }
        drop(live);
        let count = |i: usize| {
            rts.iter()
                .map(|rt| rt.resumes_past_held[i].load(SeqCst))
                .sum()
        };
        (reads, [count(0), count(1)])
    }
    fn check<C: Cells>() {
        let (want, none) = nine::<C>(None);
        assert_eq!(none, [0, 0]);
        let (got, [skipped, taken]) = nine::<C>(Some(1));
        assert_eq!(got, want);
        assert!(skipped > 0);
        assert_eq!(skipped, taken);
    }
    quiet(|| {
        check::<Local>();
        check::<Shared>();
    });
}

/// 0x2638, shrunk: the Clean form of 0x56cd8c2. After the set of 19
/// flips 17 even, `Get(2)` has 3 walk its old dep 4, whose old deps 7 and
/// 5 go first; 7 now reads 8 = 6 (`Fallback(3)`) and panics on 3, so the
/// unbounded run lands 7 and then 5 Failed, and 4 Clean without reading 5.
/// At a budget a doom stops 3's walk of 4 at 7's compute with 5 still
/// owed, and by 3's retry 4 has landed Clean. The retry's read of 4 must
/// still take the walk's rest, or 5 and 7 stay Dirty.
fn seed_2638_case() -> Case {
    let q = |expr, replay, strategy| Spec::Query {
        expr,
        replay,
        bare: false,
        strategy,
    };
    let r = |n| Box::new(Expr::Read(n, false));
    let zero = || q(Expr::Const(0), true, Strat::None);
    Case {
        specs: vec![
            zero(),
            q(Expr::Read(9, false), true, Strat::None),
            q(Expr::Read(3, false), true, Strat::None),
            q(Expr::Add(r(4), r(9)), true, Strat::None),
            q(
                Expr::If(r(17), Box::new(Expr::Const(0)), r(5)),
                false,
                Strat::None,
            ),
            q(Expr::Read(7, false), true, Strat::None),
            q(Expr::Read(15, false), true, Strat::Fallback(3)),
            q(
                Expr::If(
                    r(17),
                    Box::new(Expr::PanicIf(r(8), 3)),
                    Box::new(Expr::Const(0)),
                ),
                true,
                Strat::None,
            ),
            q(Expr::Read(6, false), true, Strat::None),
            q(Expr::Read(10, false), true, Strat::None),
            q(Expr::Read(11, false), true, Strat::None),
            q(Expr::Read(4, false), false, Strat::None),
            q(Expr::Const(0), false, Strat::None),
            zero(),
            zero(),
            q(Expr::Read(16, false), true, Strat::None),
            q(Expr::Read(3, true), true, Strat::None),
            q(Expr::Add(r(18), r(20)), true, Strat::None),
            q(Expr::Read(19, false), true, Strat::None),
            Spec::Input(2),
            Spec::Input(1),
        ],
        panics: true,
        cycles: true,
        queries: 19,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::TryGet(1), Op::Set(19, 1), Op::Get(2)],
    }
}

/// Every operation's outcome and every node's state at budgets 1 to 3
/// are the unbounded run's: the walk 0x2638's doom stops owes 5 and 7
/// their Failed landings whatever state its root is in by the retry.
#[test]
#[cfg_attr(miri, ignore)]
fn a_retry_resumes_its_walk_past_a_root_landed_clean() {
    fn check<C: Cells>() {
        let case = seed_2638_case();
        let want = run::<C>(&case, None, 0);
        for frames in [1, 2, 3] {
            let got = run::<C>(&case, Some(frames), 0);
            for (i, ((w, ws), (g, gs))) in want.iter().zip(&got).enumerate() {
                assert_eq!(
                    (w, &ws.0),
                    (g, &gs.0),
                    "budget {frames}, operation {i} ({:?})",
                    case.ops[i]
                );
            }
        }
    }
    quiet(|| {
        check::<Local>();
        check::<Shared>();
    });
}

/// 0x1dd5_ac14_6f19_4569 at budget 2: a doomed member run's speculative
/// R5 run of another member buffered a value read off the doom (a cycle's
/// `Err`), and the episode's retry read that buffer where the unbounded
/// run runs the member: `Get(6)` returned 7782 for 8.
#[test]
#[cfg_attr(miri, ignore)]
fn speculative_member_run_leaves_no_buffer() {
    let seed = 0x1dd5_ac14_6f19_4569;
    let case = gen_case(seed, false, false);
    quiet(|| {
        differential::<Local>(&case, seed);
        differential::<Shared>(&case, seed);
    });
}

/// 0x1dd5_ac14_6f19_4569 with every query bare and [`Bare`] given a
/// placeholder, whose `Drop` panics on a value only a speculative run
/// builds. At budget 1 a speculative run buffers 9999 on a member, and
/// the episode that lands outside speculation drops that buffer in its
/// bag, so `TryGet(10)` panics where the unbounded run gives 3. The
/// speculative rule must cover that drop: every budget's run matches the
/// unbounded one.
#[test]
#[cfg_attr(miri, ignore)]
fn wave6_speculative_buffer_drop_outside_speculation_is_not_raised_into_a_read() {
    crate::child_test::run_in_child(
        "runtime::replay_differential::wave6_speculative_buffer_drop_outside_speculation_is_not_raised_into_a_read",
        || {
            let seed = 0x1dd5_ac14_6f19_4569;
            let mut case = gen_case(seed, false, false);
            for spec in &mut case.specs {
                if let Spec::Query { bare, .. } = spec {
                    *bare = true;
                }
            }
            BARE_PLACEHOLDER.store(true, SeqCst);
            *BARE_DROPS.lock().unwrap() = BareDrops::Recording(HashSet::new());
            quiet(|| {
                run::<Local>(&case, None, 0);
                run::<Shared>(&case, None, 0);
            });
            let mut drops = BARE_DROPS.lock().unwrap();
            if let BareDrops::Recording(built) = std::mem::replace(&mut *drops, BareDrops::Off) {
                *drops = BareDrops::Panicking(built);
            }
            drop(drops);
            quiet(|| {
                differential::<Local>(&case, seed);
                differential::<Shared>(&case, seed);
            });
        },
    )
    .assert_clean();
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

/// Seed 0x51ea546ca5385809, shrunk. Query 0, a Fallback member of its own
/// cycle, reads 1, which reads 2 and then catches a second read of 2; 2
/// `try_get`s 3 and 3 reads 2. At budget 3, 3's entry unwinds into 2's
/// virtual frame, and the retry that reads 2's landing instead never
/// takes that delivery. 0's episode then runs 2 at the same position,
/// under 3 (its walk of 1 computes 3 first): that frame took the
/// delivery in place of its own cycle read, and 0 landed its fallback
/// where the unbounded run panics.
#[test]
#[cfg_attr(miri, ignore)]
fn member_run_frames_take_no_discovery_delivery() {
    let query = |expr, strategy| Spec::Query {
        expr,
        replay: true,
        bare: false,
        strategy,
    };
    let read = |n| Box::new(Expr::Read(n, false));
    let case = Case {
        specs: vec![
            query(Expr::Add(read(1), read(0)), Strat::Fallback(1)),
            query(
                Expr::Add(read(2), Box::new(Expr::Catch(read(2)))),
                Strat::None,
            ),
            query(Expr::Read(3, true), Strat::None),
            query(Expr::Read(2, false), Strat::None),
        ],
        panics: false,
        cycles: true,
        queries: 4,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0)],
    };
    quiet(|| {
        differential::<Local>(&case, 0x51ea_546c_a538_5809);
        differential::<Shared>(&case, 0x51ea_546c_a538_5809);
    });
}

/// Seed 0xfb9f33f99ad49897, shrunk. 0 reads 1, which reads 2 twice; 2
/// reads 3, a bare query that `try_get`s 4 and panics on its value; 4, a
/// Fallback(4) member, reads 5 twice, 5 `try_get`s 6 and 6 reads 3. 1's
/// second read walks 2 from its Dirty landing and computes 4 first; at
/// budget 1 a virtual hit makes 4 a stand-in, and the stand-in's run
/// dooms. Its new virtual frame dropped the walk's flag, so the panic
/// delivered there went to 1's read of 2 uncontained, and 2 never reran:
/// 0, 1 and 2 stayed Dirty where the unbounded run fails them. Seed
/// 0x7e17bf5981b9d32e, 0x40b5c24298f0a6ba and 0x3718100e34243fdb failed
/// the same way.
#[test]
#[cfg_attr(miri, ignore)]
fn a_stand_in_run_contains_a_panic_as_its_walk_does() {
    let query = |expr, bare, strategy| Spec::Query {
        expr,
        replay: true,
        bare,
        strategy,
    };
    let read = |n| Box::new(Expr::Read(n, false));
    let case = Case {
        specs: vec![
            query(Expr::Read(1, false), false, Strat::None),
            query(Expr::Add(read(2), read(2)), false, Strat::None),
            query(Expr::Read(3, false), false, Strat::None),
            query(
                Expr::PanicIf(Box::new(Expr::Read(4, true)), 4),
                true,
                Strat::None,
            ),
            query(Expr::Add(read(5), read(5)), false, Strat::Fallback(4)),
            query(Expr::Read(6, true), false, Strat::None),
            query(Expr::Read(3, false), false, Strat::None),
        ],
        panics: true,
        cycles: true,
        queries: 7,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0)],
    };
    quiet(|| {
        differential::<Local>(&case, 0xfb9f_33f9_9ad4_9897);
        differential::<Shared>(&case, 0xfb9f_33f9_9ad4_9897);
    });
    for seed in [
        0xfb9f_33f9_9ad4_9897,
        0x7e17_bf59_81b9_d32e,
        0x40b5_c242_98f0_a6ba,
        0x3718_100e_3424_3fdb,
    ] {
        let case = gen_case(seed, false, false);
        quiet(|| {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        });
    }
}

/// Seed 0x6b0e30c5cf9d0858 at budget 3: op 0's Get(0) raised "last
/// compute panicked" where the unbounded run raises "boom 0:9". As with
/// 0xeead60a378ca2b4, a retry's rerun of a caught read landed a node
/// Failed that the unbounded run leaves Dirty for a later read to compute
/// again, so the rerun must land it as its first take did.
#[test]
#[cfg_attr(miri, ignore)]
fn a_rerun_of_a_caught_read_lands_its_node_as_its_first_take_did() {
    for seed in [0x0eea_d60a_378c_a2b4, 0x6b0e_30c5_cf9d_0858] {
        let case = gen_case(seed, false, false);
        quiet(|| {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        });
    }
}

/// Seed 0xa18d458163c687fe at budget 1: op 0's Get(0) left node 1 Dirty
/// where the unbounded run lands it Failed. A `try_get` delivered a
/// closure's own panic after it swallowed a cycle as a cycle unwind, so
/// its reader landed as if it never ran.
#[test]
#[cfg_attr(miri, ignore)]
fn a_try_get_passes_on_a_swallowing_closures_panic() {
    let seed = 0xa18d_4581_63c6_87fe;
    let case = gen_case(seed, false, false);
    quiet(|| {
        differential::<Local>(&case, seed);
        differential::<Shared>(&case, seed);
    });
}

/// Base seeds 0x58b6599 at budget 3 and 0x58b6d24 at budget 1: a bounded
/// run landed a member Failed where the unbounded run leaves it New. Its
/// member run's `try_get` of another member of its episode took that
/// member's unwind, a closure's own panic after it swallowed the cycle, as
/// an ordinary panic, where the unbounded run's R5 read passes it on as
/// the cycle's.
#[test]
#[cfg_attr(miri, ignore)]
fn a_member_try_get_keeps_a_swallowing_closures_cycle() {
    for seed in [0x58b_6599, 0x58b_6d24] {
        let case = gen_base(seed);
        quiet(|| {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        });
    }
}

/// Member 5 (created first) try_gets 1, and then 4 on discovery's `Err`
/// but 3 once 1 has a buffer, so 3 joins the episode only in 5's member
/// run, where it computes natively. 3's `try_get` of 4, a held member
/// without a buffer, is R5 in the unbounded run: 4 swallows the cycle on 5
/// and panics, and that unwind stays the cycle's, so the episode's abort
/// leaves 3 New. A bounded run delivers 4's unwind to 3's retry, which is
/// not a member run, so `take_delivery` passes it on as an ordinary panic
/// and 3 lands Dirty.
#[test]
#[cfg_attr(miri, ignore)]
fn a_native_try_get_of_a_held_member_keeps_a_swallowing_closures_cycle() {
    let r = |n, t| Box::new(Expr::Read(n, t));
    let case = Case {
        specs: vec![
            replayable(Expr::Read(1, false), Strat::None),
            replayable(
                Expr::Add(r(2, false), Box::new(Expr::Const(1))),
                Strat::None,
            ),
            replayable(Expr::Read(5, true), Strat::None),
            replayable(Expr::Read(4, true), Strat::None),
            replayable(
                Expr::If(
                    Box::new(Expr::Catch(r(1, false))),
                    Box::new(Expr::PanicIf(Box::new(Expr::Catch(r(5, false))), 3)),
                    Box::new(Expr::Const(0)),
                ),
                Strat::None,
            ),
            replayable(Expr::If(r(1, true), r(3, true), r(4, true)), Strat::None),
            Spec::Input(1),
        ],
        panics: true,
        cycles: true,
        queries: 6,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0)],
    };
    fn outcomes<C: Cells>(case: &Case, frames: Option<u32>) -> Vec<(Out, Vec<NodeState>)> {
        run::<C>(case, frames, 0)
            .into_iter()
            .map(|(out, snap)| (out, snap.0))
            .collect()
    }
    quiet(|| {
        let unbounded = outcomes::<Local>(&case, None);
        assert_eq!(unbounded[0].1[3], NodeState::New);
        assert_eq!(outcomes::<Shared>(&case, None), unbounded);
        for frames in 1..=3 {
            assert_eq!(
                outcomes::<Local>(&case, Some(frames)),
                unbounded,
                "Local, budget {frames}"
            );
            assert_eq!(
                outcomes::<Shared>(&case, Some(frames)),
                unbounded,
                "Shared, budget {frames}"
            );
        }
    });
}

/// Seed 0xd3713dd8cf68c605, shrunk. 4 reads the ring 5, 6, 7, 8, 9, 10,
/// where 10, a Fallback(0) member, reads the input and 5, and 7 panics on
/// 8's value. At budget 2 the first pass delivered 9's cycle to 8's read
/// at its position, and the doomed entry's retry re-armed it. 7's panic
/// then released 8 and 9, which ends their runs as it does unbounded, but
/// 8's next run at that position took the old cycle where the unbounded
/// run reads 9 Failed: 8 stayed Dirty instead of Failed.
#[test]
#[cfg_attr(miri, ignore)]
fn a_released_claim_takes_no_delivery_of_its_ended_run() {
    let query = |expr, strategy| Spec::Query {
        expr,
        replay: true,
        bare: false,
        strategy,
    };
    let read = |n| Box::new(Expr::Read(n, false));
    let case = Case {
        specs: vec![
            query(Expr::Read(1, false), Strat::None),
            query(
                Expr::If(
                    read(2),
                    Box::new(Expr::Const(0)),
                    Box::new(Expr::Read(2, true)),
                ),
                Strat::None,
            ),
            query(Expr::Catch(read(3)), Strat::None),
            query(Expr::Read(10, false), Strat::None),
            query(Expr::Read(5, false), Strat::None),
            query(Expr::Read(6, false), Strat::None),
            query(Expr::Read(7, false), Strat::None),
            query(Expr::PanicIf(read(8), 0), Strat::None),
            query(Expr::Read(9, false), Strat::None),
            query(Expr::Read(10, false), Strat::None),
            query(Expr::Add(read(11), read(5)), Strat::Fallback(0)),
            Spec::Input(4),
        ],
        panics: true,
        cycles: true,
        queries: 11,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Set(11, 0), Op::Get(4)],
    };
    quiet(|| {
        exact_states::<Local>(&case);
        exact_states::<Shared>(&case);
    });
}

/// Seeds 0x6e3 and 0xdeb (and 0xd60, in [`strict_dirty_for_clean_seeds`]): a user panic in a ring releases members
/// whose doomed runs kept claim watermarks for their retries, and the
/// members' next runs at those positions run fresh, taking none of them.
#[test]
#[cfg_attr(miri, ignore)]
fn a_released_member_takes_no_record_of_its_ended_run() {
    for seed in [0x6e3, 0xdeb] {
        let case = gen_case(seed, false, false);
        quiet(|| {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        });
    }
}

/// Asserts every operation's outcome and node states at budgets 1 to 3
/// are the unbounded run's, for a case whose reads of Dirty values stop
/// `first_diff` comparing states although the runs agree.
fn exact_states<C: Cells>(case: &Case) {
    let states = |t: Trace| -> Vec<(Out, Vec<NodeState>)> {
        t.into_iter().map(|(out, snap)| (out, snap.0)).collect()
    };
    let unbounded = states(run::<C>(case, None, 0));
    for frames in [1, 2, 3] {
        assert_eq!(
            states(run::<C>(case, Some(frames), 0)),
            unbounded,
            "budget {frames}"
        );
    }
}

/// Seed 0xd3713dd8cf68c605 unshrunk, the case above's source.
#[test]
#[cfg_attr(miri, ignore)]
fn seed_d371_released_claim_takes_no_old_delivery() {
    let seed = 0xd371_3dd8_cf68_c605;
    let case = gen_case(seed, false, false);
    quiet(|| {
        differential::<Local>(&case, seed);
        differential::<Shared>(&case, seed);
    });
}

/// Seed 0x92b175856b: at budget 3, op 7, root 0's episode dissolves and
/// the budget dooms 0's run alone after it. Its discard keeps 0's record of
/// the dissolved group for other threads, and the retry group-verified 0 from
/// it and landed all seven nodes as members, where the unbounded run computes
/// 0 alone.
#[test]
#[cfg_attr(miri, ignore)]
fn seed_92b175856b_dissolved_root_retry_computes_alone() {
    let seed = 0x92_b175_856b;
    let case = gen_case(seed, false, false);
    quiet(|| {
        differential::<Local>(&case, seed);
        differential::<Shared>(&case, seed);
    });
}

/// Seed 0x59a582f: at budgets 2 and 3, op 3 `TryGet(1)` gave 0 for 26.
/// Root 2's episode dissolved at its virtual frame's pop, and the driver's
/// retry of 2 was sent straight to 10's compute, as 2's run before the
/// dissolve had read it, where the unbounded run's new run of 2 walks 10;
/// 2's episode then formed again. The case stops before op 15, whose
/// superseded landing of 3 is T8-strict-group-c-dirty-for-clean's.
#[test]
#[cfg_attr(miri, ignore)]
fn seed_59a582f_dissolved_root_reads_afresh() {
    let seed = 0x59a_582f;
    let mut case = gen_case(seed, false, false);
    case.ops.truncate(15);
    quiet(|| {
        differential::<Local>(&case, seed);
        differential::<Shared>(&case, seed);
    });
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
/// the real read panicked "last compute panicked"; in 0xa02 a speculative
/// landing read the second runtime's input before a set of it, and a
/// later walk verified that reader Clean on the old value. The rest failed
/// under Local only, while each Local runtime had its own frame stack: a
/// read from another runtime's closure nested in R's compute recorded into
/// R's frame and missed its doom, so in 0xb (rt2 5 -> rt1 4 -> rt2 2 at
/// budget 1) rt1 4 landed Clean with the placeholder. 0x2, 0x27c, 0x35c,
/// 0x1b5, 0x1e4 and 0x2e1 failed under a per-thread marker of the
/// innermost Local frame, which kept the stacks apart. In 0x432a a
/// frame's read of the other runtime returned a value its landing left
/// Dirty, and the frame's retry read that node again and walked into a
/// Failed dep; in 0x3f39 the retry's second read of a strategy-less ring
/// entered it elsewhere and closed a cycle the first read had not.
#[test]
#[cfg_attr(miri, ignore)]
fn cross_seeds() {
    for seed in [
        0x2, 0xb, 0xfa, 0xfb, 0xfe, 0x109, 0x135, 0x14a, 0x14d, 0x19a, 0x19e, 0x1b5, 0x1b6, 0x1e4,
        0x228, 0x27c, 0x27f, 0x2cd, 0x2e1, 0x308, 0x35c, 0x38b, 0x3d0, 0x3f7, 0x401, 0x441, 0x4f9,
        0x50d, 0x54f, 0x560, 0x573, 0x5a3, 0x5b6, 0x5c4, 0x5ec, 0x63e, 0x657, 0x734, 0x7a2, 0x3f39,
        0x432a,
    ] {
        let case = gen_case(seed, true, false);
        quiet(|| {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        });
    }
}

/// Two-thread seeds, with a second runtime when `.1`, whose reads pushed a
/// virtual frame onto a node that already had one, a group verify having
/// claimed a member with a virtual frame on its thread. The threads'
/// overlapping reads enter shared cycles, so only each run's own checks
/// apply, over several schedules.
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_repush_seeds() {
    let cross = gen_case(0xe43, true, false);
    quiet(|| {
        differential::<Local>(&cross, 0xe43);
        differential::<Shared>(&cross, 0xe43);
    });
    for (seed, cross) in [
        (0x527, false),
        (0x107d, false),
        (0x3f88, false),
        (0x4371, false),
        (0x59d, true),
        (0xe43, true),
    ] {
        run_two_repeatedly(seed, cross, &[Some(1), Some(2), Some(3)]);
    }
}

/// Two-thread cases with panics and cycles in which a read's walk landed
/// its node New as a cycle member and the other thread claimed the node
/// before the read loaded its state. Which thread computes a node first
/// decides these cases' results, so the runs check only that no engine
/// panic escapes and the invariants hold after each phase, a few times
/// over because the race is schedule-dependent.
#[test]
#[cfg_attr(miri, ignore)]
fn arena_missing_two_thread_seeds() {
    for (seed, cross) in [(0x33e9, false), (0xa55, false), (0x19f0, true)] {
        run_two_repeatedly(seed, cross, &[None, Some(1), Some(2), Some(3)]);
    }
}

/// The two-thread case of `seed`, with a second runtime if `cross`, run
/// four times at each of `budgets` with and without a byte budget, each
/// run held only to its own checks.
fn run_two_repeatedly(seed: u64, cross: bool, budgets: &[Option<u32>]) {
    let case = gen_case(seed, cross, true);
    let phases = phases(&case, seed);
    quiet(|| {
        for _ in 0..4 {
            for &frames in budgets {
                for bytes in [0, 16 * 1024] {
                    run_two(&case, &phases, frames, bytes);
                }
            }
        }
    });
}

/// Two-thread seeds whose fault needs a particular schedule, each run many
/// times at the budget it failed at. A case with panics or cycles may end
/// otherwise than the single-thread run under any schedule, so these
/// assert only what `run_two` checks: no engine panic, no virtual frame
/// left indexed after an operation, and the invariants after each phase.
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_schedule_seeds() {
    // 0x1e89: a wound in a doomed episode member run that a virtual
    // frame's pop ran was delivered to the frame below, which left the
    // doom's virtual frame naming the aborted episode.
    for (seed, frames, bytes) in [(0x1e89, 1, 16 * 1024)] {
        let case = gen_case(seed, false, true);
        let phases = phases(&case, seed);
        for _ in 0..300 {
            quiet(|| {
                run_two(&case, &phases, Some(frames), bytes);
            });
        }
    }
}

/// Two-thread seeds where, under a budget, the other thread computed a node
/// a doom released for its context's retry: each run's outcomes must be
/// some read order's unbounded outcomes.
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_doom_reservation_seeds() {
    for (seed, frames) in [(0x11d, 1), (0x3661_4919, 1)] {
        let case = gen_case(seed, false, true);
        let phases = phases(&case, seed);
        let pairs = phases.iter().filter(|p| p.ops().len() == 2).count();
        let outcomes = |t: &Trace, of: &dyn Fn(usize) -> usize| -> Vec<String> {
            (0..t.len()).map(|i| format!("{:?}", t[of(i)].0)).collect()
        };
        let mut serial = Vec::new();
        let mut got = Vec::new();
        quiet(|| {
            for swaps in 0..1u64 << pairs {
                let order = Order::new(&case, &phases, swaps);
                let t = run::<Shared>(&order.case, None, 0);
                let mut back = vec![0; t.len()];
                for (j, &i) in order.of.iter().enumerate() {
                    back[i] = j;
                }
                serial.push(outcomes(&t, &|i| back[i]));
            }
            for _ in 0..300 {
                got.push(outcomes(&run_two(&case, &phases, Some(frames), 0), &|i| i));
            }
        });
        let wrong = got.iter().filter(|g| !serial.contains(g)).count();
        assert_eq!(
            wrong, 0,
            "seed {seed:#x}: {wrong} of 300 runs match no read order's outcomes"
        );
    }
}

/// 0x25e4, over several schedules: a read that panics in a member of a
/// group lands that member Failed and releases the others Dirty, and a
/// deferral of their group exits showed the other thread them still
/// grouped beside the Failed member, so its read expanded the group into
/// a cycle no order of the two reads has.
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_group_exit_seeds() {
    let case = gen_case(0x25e4, false, true);
    quiet(|| {
        for _ in 0..8 {
            differential_two(&case, 0x25e4);
        }
    });
}

/// Two-thread seeds with a second runtime, each over several schedules.
/// 0x43c2 and 0x4c08, with no cycles, compare with the unbounded run: a
/// concurrent `Get2` panicked "dependency cycle" before a doomed frame's
/// read of the other runtime became speculative (ee874c3). In 0x4b15, at
/// budget 2, a read made a virtual hit while a context whose loop had not
/// started kept a virtual frame at the read's position, whose node the
/// other thread had landed, so the hit's claim of it failed on every
/// retry and the read spun.
#[test]
#[cfg_attr(miri, ignore)]
fn cross_two_thread_seeds() {
    for seed in [0x43c2, 0x4c08] {
        let case = gen_case(seed, true, true);
        quiet(|| {
            for _ in 0..4 {
                differential_two(&case, seed);
            }
        });
    }
    let case = gen_case(0x4b15, true, true);
    let phases = phases(&case, 0x4b15);
    quiet(|| {
        for _ in 0..16 {
            for bytes in [0, 16 * 1024] {
                run_two(&case, &phases, Some(2), bytes);
            }
        }
    });
}

/// The unbounded single-thread runs of 0x524 and 0x3c9 with their first
/// read phase's reads in the other order, laid out as two-thread traces,
/// differ from the case-order runs: 0x524's closure payload goes to the
/// other read, and 0x3c9's cycle is entered elsewhere. The two-thread
/// oracle takes each, and not with the read that differs given an outcome
/// of a kind no order gives.
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_oracle_takes_either_order_of_a_read_pair() {
    for seed in [0x524, 0x3c9] {
        let case = gen_case(seed, false, true);
        let phases = phases(&case, seed);
        let swapped = Order::new(&case, &phases, 1);
        quiet(|| {
            let serial = run::<Shared>(&swapped.case, None, 0);
            let mut got = vec![None; case.ops.len()];
            for (j, &i) in swapped.of.iter().enumerate() {
                got[i] = Some((serial[j].0.clone(), serial[swapped.ends[j]].1.clone()));
            }
            let got: Trace = got.into_iter().map(Option::unwrap).collect();
            let in_order = Order::new(&case, &phases, 0);
            let unbounded = run::<Shared>(&case, None, 0);
            let settles = in_order.settles();
            let (i, _) = first_diff(
                &case,
                &unbounded,
                &got,
                Some(&settles),
                PanicText::Exact,
                &PerRead::new(),
                &CycleReads::new(),
            )
            .unwrap_or_else(|| panic!("seed {seed:#x}: the swap changes no outcome"));
            assert_serial(&case, &phases, &mut HashMap::new(), &got, "swapped");

            let mut bad = got;
            bad[i].0 = match bad[i].0 {
                Out::Value(_) => Out::Panic("no order's outcome".to_owned()),
                _ => Out::Value(u64::MAX),
            };
            let caught = catch_unwind(AssertUnwindSafe(|| {
                assert_serial(&case, &phases, &mut HashMap::new(), &bad, "bad");
            }));
            assert!(caught.is_err(), "seed {seed:#x}: a wrong outcome passed");
        });
    }
}

/// The two-thread oracle takes a panic of any text where an order panics,
/// but not a value there, while the single-thread differential still
/// compares the text (the Shared panic payload contract).
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_oracle_takes_a_panic_of_any_text() {
    let case = gen_case(0x524, false, true);
    let phases = phases(&case, 0x524);
    quiet(|| {
        let unbounded = run::<Shared>(&case, None, 0);
        let i = unbounded
            .iter()
            .position(|(out, _)| matches!(out, Out::Panic(_)))
            .expect("0x524 panics");
        let mut other = unbounded.clone();
        other[i].0 = Out::Panic("another payload".to_owned());
        assert_serial(&case, &phases, &mut HashMap::new(), &other, "other text");
        let one = catch_unwind(AssertUnwindSafe(|| {
            compare(&case, &unbounded, &other, "one thread");
        }));
        assert!(
            one.is_err(),
            "the single-thread differential took other text"
        );

        let mut value = unbounded;
        value[i].0 = Out::Value(u64::MAX);
        let caught = catch_unwind(AssertUnwindSafe(|| {
            assert_serial(&case, &phases, &mut HashMap::new(), &value, "value");
        }));
        assert!(caught.is_err(), "a value passed where every order panics");
    });
}

/// Two-thread seeds with panics or cycles whose overlapping reads, at most
/// budgets, come out as the unbounded run of the other order: which read
/// lands a node both read first decides which gets a closure's payload,
/// or where a cycle is entered.
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_swapped_order_seeds() {
    for seed in [0x245, 0x3c9] {
        let case = gen_case(seed, false, true);
        quiet(|| {
            for _ in 0..4 {
                differential_two(&case, seed);
            }
        });
    }
}

/// The Failed set after op `i` of a trace, with the op's outcome.
fn failed_after(t: &Trace, i: usize) -> (Out, Vec<usize>) {
    let (out, snap) = &t[i];
    let set = (snap.0.iter().enumerate())
        .filter(|(_, s)| **s == NodeState::Failed)
        .map(|(x, _)| x)
        .collect();
    (out.clone(), set)
}

/// Seed 0xe486019a1c945cce's op 1, `Get(4)`: member 8 of the cycle
/// {4, 5, 7, 8} reads 9, whose walk computes 21, and 21's own code panics
/// in the episode of {9, 21, 22, 23}, which marks it failed; 9 then passes
/// that panic to 8. Unbounded, the mark falls inside 8's member run, so 8
/// lands New. At a budget of 1 the driver runs that episode between doomed
/// runs of 8, and the run finishing 8's frame counted marks from its own
/// start only, so 8 landed Failed as well.
#[test]
#[cfg_attr(miri, ignore)]
fn a_doomed_member_run_keeps_a_failure_marked_above_it() {
    let seed = 0xe486_019a_1c94_5cce;
    let case = gen_case(seed, false, false);
    quiet(|| {
        let mut fresh = case.clone();
        fresh.ops = vec![Op::Get(4)];
        let want = (Out::Panic("boom 0:21".to_owned()), vec![9, 21]);
        assert_eq!(failed_after(&run::<Local>(&fresh, None, 0), 0), want);
        assert!(matches!(case.ops[1], Op::Get(4)));
        assert_eq!(failed_after(&run::<Local>(&case, Some(1), 0), 1), want);
        differential::<Local>(&case, seed);
        differential::<Shared>(&case, seed);
        differential_two(&gen_case(seed, false, true), seed);
    });
}

/// Seed 0x5aab, two threads at a budget of 2: `Get(0)` and `Get(6)` both
/// enter the cycle 0, 1, 2, 3, 8, whose member 0 branches on a caught cycle
/// error. Serially whichever read runs first panics with the cycle and the
/// other returns 9999. A doom gives 0 the discard landing mid-run, the
/// other thread claims and computes it, and the two runs alternate inside
/// the cycle: both reads can panic, both return 9999, or `Get(0)` returns
/// 14, none of which an order gives.
#[test]
#[ignore = "T8-doomed-frame-keeps-claim: a doomed frame hands its node to another thread"]
#[cfg_attr(miri, ignore)]
fn a_doomed_frame_keeps_its_node_from_the_other_thread() {
    let case = gen_case(0x5aab, false, true);
    quiet(|| {
        for _ in 0..8 {
            differential_two(&case, 0x5aab);
        }
    });
}

/// Seed 0xe4aa8b2e, two threads at a budget of 1 or 2: both threads read
/// node 0, whose cycle branches on a caught panic, and `Get(0)` returns
/// 9999 where every order panics. Each thread's doom discard-lands a node
/// the other thread then claims, down the chain from node 26, as in 0x5aab.
/// It fails in about one debug `differential_two` run in five.
#[test]
#[ignore = "T8-doomed-frame-keeps-claim: a doomed frame hands its node to another thread"]
#[cfg_attr(miri, ignore)]
fn a_doomed_frame_keeps_its_node_from_the_other_thread_e4aa8b2e() {
    let case = gen_case(0xe4aa_8b2e, false, true);
    quiet(|| {
        for _ in 0..16 {
            differential_two(&case, 0xe4aa_8b2e);
        }
    });
}

/// Seed 0x57c2661 cut to `Get(12)` on a fresh runtime with 24 at 4: a
/// cycle reaches the read unbounded, aborting 13's episode, which is no
/// walk's. At budgets 1 to 3 a doom discards a member run of that episode
/// and the retry reaches its members as walk entries of 13's deps; their
/// episodes held those members, and contained the cycle every round, so
/// the read returned 15564.
#[test]
#[cfg_attr(miri, ignore)]
fn a_retry_walking_a_doomed_episodes_members_keeps_its_cycle() {
    let seed = 0x57c2661;
    let mut case = gen_case(seed, false, false);
    case.ops = vec![Op::Set(24, 4), Op::Get(12)];
    quiet(|| {
        assert!(matches!(run::<Local>(&case, None, 0)[1].0, Out::Panic(_)));
        differential::<Local>(&case, seed);
        differential::<Shared>(&case, seed);
    });
}

/// Seed 0x29bf2de20fa0e476's op 0, `Get(0)` on a fresh runtime: 1 branches
/// on a `try_get` into the cycle of 6, 7, 8, 13 and 17. With walked
/// episodes counting held members as released, the unbounded run panicked
/// in 1 and budget 3 returned 2; every budget must agree.
#[test]
#[cfg_attr(miri, ignore)]
fn a_walked_episode_in_a_try_get_cycle_answers_as_unbounded() {
    let seed = 0x29bf_2de2_0fa0_e476;
    let mut case = gen_case(seed, false, false);
    case.ops.truncate(1);
    assert!(matches!(case.ops[0], Op::Get(0)));
    quiet(|| {
        differential::<Local>(&case, seed);
        differential::<Shared>(&case, seed);
    });
}

/// Seed 0x52's op 9, `Get(12)` at budget 1: 12 reads 13 twice, and the
/// first read's walk of 13 is doomed with 7's rerun still in it. The
/// doomed run's second read of 13 pushed a speculative entry that landed
/// 13 before 12's retry, so the retry never resumed the walk and 7 stayed
/// Dirty where the unbounded run leaves it Clean.
#[test]
#[cfg_attr(miri, ignore)]
fn a_doomed_reread_of_the_blocked_node_leaves_the_walk_to_the_retry() {
    let seed = 0x52;
    let mut case = gen_case(seed, false, false);
    case.ops.truncate(10);
    assert!(matches!(case.ops[9], Op::Get(12)));
    quiet(|| {
        // A forced prune leaves the unread node 8 Dirty where the doomed
        // walk's retry computes it: walk scope, not a result.
        if *super::FORCE_REPLAN {
            outcomes_alike::<Local>(&case, seed);
            outcomes_alike::<Shared>(&case, seed);
        } else {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        }
    });
}

/// Seed 0x2897c4266833877, unbounded: op 3's `Get(18)` verifies 18's
/// group with a tainted walk entry, which releases the group's claims. A
/// released member that stays in its group is verified as one again and
/// gives 7793, where a fresh run raises the cycle. Leaving the group fixes
/// this but breaks [`seed_92b175856b_dissolved_root_retry_computes_alone`]
/// when the release happens inside an attempt a byte budget discards; a fix
/// must pass both.
#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "T10-ub-69a2639d: a released group member is verified with its stale group"]
fn a_tainted_group_release_still_raises_the_cycle() {
    let mut case = gen_case(0x0289_7c42_6683_3877, false, false);
    case.ops.truncate(4);
    assert!(matches!(case.ops[3], Op::Get(18)));
    quiet(|| {
        for trace in [run::<Local>(&case, None, 0), run::<Shared>(&case, None, 0)] {
            assert_eq!(trace[3].0, Out::Panic("cycle".to_owned()));
        }
    });
}

/// [`differential`]'s runs, compared by their operations' outcomes alone.
fn outcomes_alike<C: Cells>(case: &Case, seed: u64) {
    let outcomes = |t: Trace| t.into_iter().map(|(o, _)| o).collect::<Vec<_>>();
    let unbounded = outcomes(run::<C>(case, None, 0));
    for (frames, bytes) in BUDGETS {
        let bounded = outcomes(run::<C>(case, frames, bytes));
        assert_eq!(bounded, unbounded, "{}", budget_label(seed, frames, bytes));
    }
}

/// Seeds 0x833e (and 0x41eb, in [`strict_dirty_for_clean_seeds`]), one
/// thread: a dropped chain truncates the virtual frame of a doomed episode
/// member run whose failure count is still held, and releases the member
/// with the discard landing. The count goes with the frame, since the
/// member's next run there is a new one.
#[test]
#[cfg_attr(miri, ignore)]
fn a_dropped_chain_takes_its_member_runs_failure_counts() {
    quiet(|| {
        let case = gen_case(0x833e, false, false);
        differential::<Local>(&case, 0x833e);
        differential::<Shared>(&case, 0x833e);
    });
}

/// The same truncation under two threads, seed 0x2b13.
#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "T10-cmp-2b13-two-thread-boom: op 4 returns a panic no serial order gives"]
fn a_two_thread_dropped_chain_takes_its_member_runs_failure_counts() {
    quiet(|| {
        let case = gen_case(0x2b13, false, true);
        for _ in 0..4 {
            differential_two(&case, 0x2b13);
        }
    });
}

/// Waits for a forced schedule's `done`, failing the test if it stalls.
fn until(done: impl Fn() -> bool) {
    let start = std::time::Instant::now();
    while !done() {
        assert!(start.elapsed().as_secs() < 20, "the schedule stalled");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// 0x934's ops 7 (`Get(11)`) and 8 (`TryGet(13)`) overlap on a Dirty
/// cycle, and each order of them leaves {0, 3..13} Failed. Here op 7's
/// walk is held at its first compute, node 3, while op 8 lands 10 to 13
/// Failed; the walk then computes 3 from its plan, 3's read meets 12
/// Failed, and the walk contains that, so its readers rerun and land
/// Failed as in either order: the run is serial, with no history for
/// the oracle to judge. Swap bit 1, not bit 0 (ops 5 and 6), runs op 8
/// first, which gets the closure's payload.
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_oracle_takes_a_failed_set_of_an_overlap() {
    struct Hold {
        on: AtomicBool,
        first: AtomicBool,
        at3: AtomicBool,
    }
    // A forced revisit changes which of the overlapping reads raises the
    // closure's payload, the schedule this test pins.
    if super::forced_revisit() {
        return;
    }
    let case = gen_case(0x934, false, true);
    let phases = phases(&case, 0x934);
    let at = |i| phases.iter().position(|p| p.ops().contains(&i)).unwrap();
    let (before, overlap) = (at(6), at(7));
    assert_eq!(phases[before].ops(), [5, 6]);
    assert_eq!(phases[overlap].ops(), [7, 8]);
    assert_eq!(phases[overlap].of(0), Some(8));
    quiet(|| {
        let swapped = Order::new(&case, &phases, 0b10);
        let first = |i| swapped.of.iter().position(|&j| j == i).unwrap();
        assert!(first(8) < first(7));
        let serial = run::<Shared>(&swapped.case, None, 0);
        assert_eq!(serial[first(8)].0, Out::Panic("boom 0:11".to_owned()));
        let orders = [
            settled(&run::<Shared>(&case, None, 0)[8].1 .0),
            settled(&serial[first(7)].1 .0),
        ];

        let hold = Arc::new(Hold {
            on: AtomicBool::new(false),
            first: AtomicBool::new(false),
            at3: AtomicBool::new(false),
        });
        // Thread 0 runs op 8: it holds at its first park until op 7's walk
        // is at its compute of 3, which holds until op 8 has landed 13.
        let rig = |live: &Live<Shared>| {
            let (hold, world) = (hold.clone(), live.world.clone());
            let three = ParkPoint::BeforeComputeOne(NodeId(world.handle(0, 3).erased().slot()));
            let hook: crate::runtime::ParkHook = Box::new(move |p| {
                if !hold.on.load(SeqCst) {
                    return;
                }
                match TWO_THREAD.with(std::cell::Cell::get) {
                    Some(0) if !hold.first.swap(true, SeqCst) => until(|| hold.at3.load(SeqCst)),
                    Some(1) if p == three && !hold.at3.swap(true, SeqCst) => {
                        until(|| world.handle(0, 13).state(world.rt(0)) == NodeState::Failed)
                    }
                    _ => {}
                }
            });
            assert!(live.world.rt(0).park_hook.set(hook).is_ok());
        };
        let serial = Serial::new(&case, &phases);
        let (verdict, left) = (Mutex::new(None), Mutex::new(Vec::new()));
        let got = run_two_until(&case, &phases, None, 0, &rig, &|p, prefix, live| {
            if p == overlap {
                hold.on.store(false, SeqCst);
                *left.lock().unwrap() = settled(&prefix[8].1 .0);
            }
            let v = serial.at_overlap("0x934", prefix, live);
            if p == before {
                hold.on.store(true, SeqCst);
            }
            let stop = v.is_some();
            *verdict.lock().unwrap() = v;
            stop
        });
        let failed = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].map(|x| (x, NodeState::Failed));
        assert_eq!(orders, [failed, failed].map(Vec::from));
        assert_eq!(*left.lock().unwrap(), failed);
        let verdict = verdict.into_inner().unwrap();
        assert_eq!(verdict, None);
        serial.check("0x934", &got, verdict);
    });
}

/// 0x1006's ops 6 (`Get(12)`) and 7 (`Get(7)`) overlap on the cold cycle
/// 8 -> 9 -> 10 -> 11 -> 12 -> 8, whose strategy members are 11 and 12.
/// Op 7 enters it at 8, which has none, so 7's `try_get(4)` gets the
/// cycle (R3). Here op 7 then holds until op 6, entering at 12, has
/// landed 8 Clean, and 7 reads 8 = 2 twice: 7777 + 2 + 2, no order's
/// value. The oracle orders that get per read: each read is an order's
/// outcome for it, and the value is 7's closure over them.
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_oracle_orders_a_get_that_caught_a_cycle_per_read() {
    let case = gen_case(0x1006, false, true);
    let phases = phases(&case, 0x1006);
    let overlap = phases.iter().position(|p| p.ops() == [6, 7]).unwrap();
    assert_eq!(phases[overlap].of(1), Some(7));
    let (on, ready) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    // Thread 0 runs op 6: it holds at its first park until op 7 has got
    // the cycle and is at its claim of 8, which holds until 8 is Clean.
    let rig = |live: &Live<Shared>| {
        let (on, ready, world) = (on.clone(), ready.clone(), live.world.clone());
        let slot = |q| NodeId(world.handle(0, q).erased().slot());
        let (four, eight) = (slot(4), slot(8));
        let hook: crate::runtime::ParkHook = Box::new(move |p| {
            if !on.load(SeqCst) {
                return;
            }
            let state = |n: NodeId| world.rt(0).nodes.get(n.0).state();
            match TWO_THREAD.with(std::cell::Cell::get) {
                Some(0) => until(|| ready.load(SeqCst)),
                Some(1)
                    if p == ParkPoint::BeforeClaim(eight)
                        && state(four) == NodeState::New
                        && !ready.swap(true, SeqCst) =>
                {
                    until(|| state(eight) == NodeState::Clean)
                }
                _ => {}
            }
        });
        assert!(live.world.rt(0).park_hook.set(hook).is_ok());
    };
    quiet(|| {
        let serial = Serial::new(&case, &phases);
        let (verdict, seven) = (Mutex::new(None), Mutex::new(None));
        let got = run_two_until(&case, &phases, None, 0, &rig, &|p, prefix, live| {
            if p == overlap {
                on.store(false, SeqCst);
                let evals = live.evals.lock().unwrap();
                let last = evals[&7]
                    .iter()
                    .filter(|e| e.node == (0, 7))
                    .max_by_key(|e| e.end);
                *seven.lock().unwrap() = Some((prefix.clone(), last.cloned().unwrap()));
            }
            let v = serial.at_overlap("0x1006", prefix, live);
            if p + 1 == overlap {
                on.store(true, SeqCst);
            }
            let stop = v.is_some();
            *verdict.lock().unwrap() = v;
            stop
        });
        assert_eq!(got[7].0, Out::Value(7781));
        serial.check("0x1006", &got, verdict.into_inner().unwrap());
        let per_read = serial.per_read.lock().unwrap().clone();
        assert_eq!(
            per_read.keys().collect::<Vec<_>>(),
            [&7],
            "op 6 stays exact"
        );

        let (prefix, mut sevens) = seven.into_inner().unwrap().unwrap();
        let read = (sevens.reads.iter_mut())
            .find(|r| r.3 == ReadOut::Value(2))
            .unwrap();
        read.3 = ReadOut::Value(3);
        let why = serial
            .closure_over_reads("0x1006", &prefix, overlap, 7, 7, &sevens)
            .unwrap_err();
        assert!(why.contains("no order gives that read"), "{why}");
        let rejects = |bad: Trace, what: &str| {
            let fresh = Serial::new(&case, &phases);
            *fresh.per_read.lock().unwrap() = per_read.clone();
            let caught = catch_unwind(AssertUnwindSafe(|| fresh.check(what, &bad, None)));
            assert!(caught.is_err(), "{what} passed");
        };
        let mut off = got.clone();
        off[7].0 = Out::Value(7780);
        rejects(off, "a value off 7's closure over its reads");
        let mut partner = got.clone();
        partner[6].0 = Out::Value(u64::MAX);
        rejects(partner, "a wrong value for op 6");
        let mut failed = got.clone();
        for k in [6, 7] {
            failed[k].1 .0[5] = NodeState::Failed;
        }
        rejects(failed, "a Failed node no order leaves");
        // 7 read 4 through the cycle's unwind, so it lands Dirty, where 6
        // first leaves it Clean: the states differ past the phase and, in
        // a case with cycles, later results are history as after any such
        // phase.
        let in_order = run::<Shared>(&case, None, 0);
        assert_eq!(got[7].1 .0[7], NodeState::Dirty);
        assert_eq!(in_order[7].1 .0[7], NodeState::Clean);
    });
}

/// 0x1006's phase [6, 7] run serially: op 7 holds at its first park until
/// op 6 has landed 8..12 Clean, so no race is possible and op 7's outcome
/// must be the 6-first order's Value(8). With a stale-catch fault spliced
/// in (op 7 carries the racing run's reads, its `try_get(4)` = Cycle, and
/// returns 7781), the oracle must reject the trace: the cycle was caught
/// after the partner landed it, not before.
#[test]
#[cfg_attr(miri, ignore)]
fn wave6_per_read_rejects_a_cycle_caught_after_the_partner_landed_it() {
    let case = gen_case(0x1006, false, true);
    let phases = phases(&case, 0x1006);
    let overlap = phases.iter().position(|p| p.ops() == [6, 7]).unwrap();
    assert_eq!(phases[overlap].of(1), Some(7));
    let (on, ready) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let race_rig = |live: &Live<Shared>| {
        let (on, ready, world) = (on.clone(), ready.clone(), live.world.clone());
        let slot = |q| NodeId(world.handle(0, q).erased().slot());
        let (four, eight) = (slot(4), slot(8));
        let hook: crate::runtime::ParkHook = Box::new(move |p| {
            if !on.load(SeqCst) {
                return;
            }
            let state = |n: NodeId| world.rt(0).nodes.get(n.0).state();
            match TWO_THREAD.with(std::cell::Cell::get) {
                Some(0) => until(|| ready.load(SeqCst)),
                Some(1)
                    if p == ParkPoint::BeforeClaim(eight)
                        && state(four) == NodeState::New
                        && !ready.swap(true, SeqCst) =>
                {
                    until(|| state(eight) == NodeState::Clean)
                }
                _ => {}
            }
        });
        assert!(live.world.rt(0).park_hook.set(hook).is_ok());
    };
    let serial_rig = |live: &Live<Shared>| {
        let (on, world) = (on.clone(), live.world.clone());
        let hook: crate::runtime::ParkHook = Box::new(move |_| {
            if on.load(SeqCst) && TWO_THREAD.with(std::cell::Cell::get) == Some(1) {
                let clean = |q| {
                    world
                        .rt(0)
                        .nodes
                        .get(world.handle(0, q).erased().slot())
                        .state()
                        == NodeState::Clean
                };
                until(|| (8..=12).all(clean));
            }
        });
        assert!(live.world.rt(0).park_hook.set(hook).is_ok());
    };
    quiet(|| {
        let racing = Serial::new(&case, &phases);
        let race_evals = Mutex::new(None);
        let race_got = run_two_until(&case, &phases, None, 0, &race_rig, &|p, prefix, live| {
            if p == overlap {
                on.store(false, SeqCst);
                *race_evals.lock().unwrap() = Some(live.evals.lock().unwrap()[&7].clone());
            }
            let v = racing.at_overlap("0x1006 race", prefix, live);
            if p + 1 == overlap {
                on.store(true, SeqCst);
            }
            v.is_some()
        });
        assert_eq!(race_got[7].0, Out::Value(7781), "the race was driven");
        let race_evals = race_evals.into_inner().unwrap().unwrap();

        let serial = Serial::new(&case, &phases);
        let verdict = Mutex::new(None);
        let got = run_two_until(&case, &phases, None, 0, &serial_rig, &|p, prefix, live| {
            let mut prefix = prefix.clone();
            if p == overlap {
                on.store(false, SeqCst);
                assert_eq!(prefix[7].0, Out::Value(8), "the schedule was serial");
                // The fault catches the cycle after op 6 landed it, so its
                // runs take stamps after this run's.
                let after = EVAL_CLOCK.load(SeqCst);
                let mut stale = race_evals.clone();
                for e in &mut stale {
                    e.end += after;
                    e.caught.iter_mut().for_each(|c| c.1 += after);
                }
                live.evals.lock().unwrap().insert(7, stale);
                prefix[7].0 = Out::Value(7781);
            }
            let v = serial.at_overlap("0x1006 serial", &prefix, live);
            if p + 1 == overlap {
                on.store(true, SeqCst);
            }
            let stop = v.is_some();
            *verdict.lock().unwrap() = v;
            stop
        });
        let mut bad = got.clone();
        bad[7].0 = Out::Value(7781);
        let caught = catch_unwind(AssertUnwindSafe(|| {
            serial.check("0x1006 serial", &bad, verdict.into_inner().unwrap())
        }));
        assert!(
            caught.is_err(),
            "a cycle caught after the partner landed it was ordered per read: {:?}",
            serial.per_read.lock().unwrap().get(&7)
        );
    });
}

/// After 0x934's overlapping ops 7 and 8, a trace whose Failed set holds
/// node 1, which op 5 computed Clean, or lacks node 10, which every order
/// leaves Failed, is still rejected: no read of the phase reaches 1, so
/// their replay leaves it New, and 10 was left Clean, not Dirty or New.
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_oracle_rejects_a_failed_set_untrue_of_its_inputs() {
    let case = gen_case(0x934, false, true);
    let phases = phases(&case, 0x934);
    let overlap = phases.iter().position(|p| p.ops() == [7, 8]).unwrap();
    for (x, to, why) in [
        (
            1,
            NodeState::Failed,
            "node 1 Failed: a replay of the phase's reads landed it New",
        ),
        (
            10,
            NodeState::Clean,
            "node 10: an order leaves it Failed, but the phase left it Clean",
        ),
    ] {
        quiet(|| {
            let serial = Serial::new(&case, &phases);
            let verdict = Mutex::new(None);
            run_two_until(&case, &phases, None, 0, &|_| {}, &|p, prefix, live| {
                let mut prefix = prefix.clone();
                if p == overlap {
                    for (_, snap) in &mut prefix[7..] {
                        snap.0[x] = to;
                    }
                }
                let v = serial.at_overlap("0x934", &prefix, live);
                let stop = v.is_some();
                *verdict.lock().unwrap() = v;
                stop
            });
            match verdict.into_inner().unwrap() {
                Some(Err(m)) => assert!(m.contains(why), "{m}"),
                v => panic!("node {x} {to:?} passed: {v:?}"),
            }
        });
    }
}

/// 0x741b's first phase overlaps op 0 (`Get(0)`) with op 1 (`TryGet(2)`).
/// At a frame budget of 2, about one run in three, op 1's replayed
/// episode lands node 10 Failed before it reaches 9, and op 0's walk,
/// planned before that, meets 10 Failed through 9 and leaves 0, 9, 10
/// and others the interleaving picks Failed, a set neither order leaves:
/// op 0 first enters a cycle and op 1 leaves {10}, and op 1 first leaves
/// {0, 2, 3, 4, 5, 9, 10, 11, 12}. The oracle takes the set as history:
/// its nodes fail through 10, and the second order's others still panic
/// when read. Holding either thread at a park point
/// gives a serial run, so the runs are free and the test reruns until the
/// oracle takes one, each run checked.
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_oracle_takes_a_history_failed_set() {
    let case = gen_case(0x741b, false, true);
    let phases = phases(&case, 0x741b);
    assert_eq!(phases[0].ops(), [0, 1]);
    let orders = [Order::new(&case, &phases, 0), Order::new(&case, &phases, 1)]
        .map(|o| settled(&run::<Shared>(&o.case, None, 0)[1].1 .0));
    let failed = |xs: &[usize]| {
        xs.iter()
            .map(|&x| (x, NodeState::Failed))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        orders,
        [failed(&[10]), failed(&[0, 2, 3, 4, 5, 9, 10, 11, 12])]
    );
    quiet(|| {
        for _ in 0..2000 {
            let serial = Serial::new(&case, &phases);
            let verdict = Mutex::new(None);
            let got = run_two_until(&case, &phases, Some(2), 0, &|_| {}, &|_, prefix, live| {
                let v = serial.at_overlap("0x741b", prefix, live);
                let stop = v.is_some();
                *verdict.lock().unwrap() = v;
                stop
            });
            let verdict = verdict.into_inner().unwrap();
            let history = verdict.is_some();
            serial.check("0x741b", &got, verdict);
            if history {
                let set = settled(&got[1].1 .0);
                assert!(
                    failed(&[0, 9, 10]).iter().all(|f| set.contains(f)),
                    "{set:?}"
                );
                assert!(!orders.contains(&set), "{set:?}");
                return;
            }
        }
        panic!("no run of 0x741b left a history Failed set");
    });
}

/// Seed 0xc5001c2ab73c2d33, shrunk: `Get(0)` leaves the cycles
/// 6 -> 7 -> 8 -> 9 -> 6 and 4 -> 10 -> 11 -> 9 -> 6 -> 7 -> 8 -> 21 ->
/// 22 -> 23 -> 5 -> 3 -> 4 without values, so 6 can be deleted, and then
/// `Get(14)` reaches 9's read of the deleted 6 through 15, 16, 22, 23, 5,
/// 3, 4, 10 and 11. Every node on the way fails with the stale-handle
/// panic, which reaches `Get(14)` at every budget, as in the unbounded
/// run. At budget 3 a walk's entry contains 5's panic and a speculative
/// run reads 5 before the real run does, under 17's `Catch`, which drops
/// the payload it catches.
#[test]
#[cfg_attr(miri, ignore)]
fn stale_read_payload_reaches_the_root_through_replayed_frames() {
    let query = |expr| Spec::Query {
        expr,
        replay: true,
        bare: false,
        strategy: Strat::None,
    };
    let read = |n| Box::new(Expr::Read(n, false));
    let try_read = |n| Box::new(Expr::Read(n, true));
    let one = || Box::new(Expr::Const(1));
    let mut specs: Vec<Spec> = (0..25).map(|_| query(Expr::Const(1))).collect();
    specs[0] = query(Expr::If(
        Box::new(Expr::Add(try_read(1), one())),
        read(0),
        one(),
    ));
    for (at, to, try_) in [
        (1, 4, true),
        (3, 4, false),
        (4, 10, false),
        (5, 3, false),
        (6, 7, false),
        (7, 8, false),
        (10, 11, false),
        (11, 9, false),
        (15, 16, false),
        (16, 22, false),
        (21, 22, false),
        (22, 23, false),
        (23, 5, true),
    ] {
        specs[at] = query(Expr::Read(to, try_));
    }
    specs[8] = query(Expr::Add(read(21), read(9)));
    specs[9] = query(Expr::If(read(6), one(), one()));
    specs[14] = query(Expr::If(try_read(15), read(17), one()));
    specs[17] = query(Expr::Catch(read(16)));
    specs.push(Spec::Input(1));
    let case = Case {
        queries: 25,
        specs,
        panics: true,
        cycles: true,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Delete(6), Op::Get(14)],
    };
    quiet(|| {
        differential::<Local>(&case, 0xc500_1c2a_b73c_2d33);
        differential::<Shared>(&case, 0xc500_1c2a_b73c_2d33);
    });
}

/// Seed 0xeb379773cd1b3565, shrunk: 1 = `Catch(Read 2) + Read 2` over the
/// cycle 2 -> 3 -> 4 -> 5 -> 2, where 4 also reads the deleted 0. 1's
/// caught first read lands 2, 3 and 4 Dirty, and its second read lands them
/// Failed and panics with the stale-handle payload. At budget 3 that second
/// read dooms with 2 a virtual frame, and 1's retry must take its first
/// read's unwind again rather than run the chain a second time, which lands
/// it Failed under the `Catch` and leaves the second read "last compute
/// panicked".
#[test]
#[cfg_attr(miri, ignore)]
fn caught_read_of_a_virtual_frame_node_is_kept_for_the_retry() {
    let query = |expr, replay, strategy| Spec::Query {
        expr,
        replay,
        bare: false,
        strategy,
    };
    let read = |n| Box::new(Expr::Read(n, false));
    let case = Case {
        specs: vec![
            query(Expr::Const(1), false, Strat::None),
            query(Expr::Const(1), false, Strat::None),
            query(
                Expr::Read(3, false),
                true,
                Strat::Fixpoint {
                    initial: 0,
                    limit: 3,
                },
            ),
            query(Expr::Read(4, false), false, Strat::None),
            query(Expr::Add(read(5), read(0)), false, Strat::None),
            query(Expr::Read(2, false), true, Strat::None),
        ],
        panics: true,
        cycles: true,
        queries: 6,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![
            Op::Delete(0),
            Op::Delete(1),
            Op::Recreate(
                1,
                query(
                    Expr::Add(Box::new(Expr::Catch(read(2))), read(2)),
                    true,
                    Strat::None,
                ),
            ),
            Op::TryGet(1),
        ],
    };
    quiet(|| {
        differential::<Local>(&case, 0xeb37_9773_cd1b_3565);
        differential::<Shared>(&case, 0xeb37_9773_cd1b_3565);
    });
}

/// Seed 0x69fd501b4c8fbf5c, shrunk: `Get(7)` lands the cycle
/// 18 -> 19 -> 3 -> 17 -> 18, rooted at the Fallback 18. `Set(22, 1)` turns
/// 17's read of 18 into a read of 19, so the cycle left is {19, 3, 17},
/// which 17's Fixpoint converges without 18. `Get(6)`'s `try_get(7)` walks
/// 18 as an entry, whose group verify contains the cycle it finds. At
/// budget 1 that verify's episode is doomed, and the retry of 18 has to
/// stay a walk entry that contains it too, or the cycle reaches 7's read.
#[test]
#[cfg_attr(miri, ignore)]
fn try_get_of_a_former_cycle_member_reads_its_value() {
    let query = |expr, replay, strategy| Spec::Query {
        expr,
        replay,
        bare: false,
        strategy,
    };
    let read = |n| Box::new(Expr::Read(n, false));
    let mut specs: Vec<Spec> = (0..22)
        .map(|_| query(Expr::Const(1), false, Strat::None))
        .collect();
    specs[3] = query(Expr::Read(17, false), false, Strat::None);
    specs[6] = query(Expr::Read(7, true), false, Strat::None);
    specs[7] = query(Expr::Read(18, false), false, Strat::None);
    specs[17] = query(
        Expr::If(read(22), read(18), read(19)),
        false,
        Strat::Fixpoint {
            initial: 3,
            limit: 2,
        },
    );
    specs[18] = query(Expr::Read(19, false), true, Strat::Fallback(0));
    specs[19] = query(Expr::Read(3, false), false, Strat::None);
    specs.extend([4, 4, 3, 4].map(Spec::Input));
    let case = Case {
        specs,
        panics: false,
        cycles: true,
        queries: 22,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(7), Op::Set(22, 1), Op::Get(6)],
    };
    quiet(|| {
        differential::<Local>(&case, 0x69fd_501b_4c8f_bf5c);
        differential::<Shared>(&case, 0x69fd_501b_4c8f_bf5c);
    });
}

/// Seed 0x88f7, shrunk, with a second runtime: a user panic in a cycle
/// releases the other claims, which e9eea59d lands New, and the same read
/// then computes them again. After `Get(4)` and `Set(19, 0)`, `Get(9)`'s
/// walk runs 12, which panics; unbounded, the walk's entries 8 and 7
/// contain the panic and 9 reads as "last compute panicked". Every budget
/// must give the unbounded run's outcomes and states, on `Local` and
/// `Shared`.
#[test]
#[cfg_attr(miri, ignore)]
fn wave8_seed_88f7_released_claims_replay_as_unbounded() {
    use Expr::{Add, Catch, Const, If, PanicIf, Read};
    let b = Box::new;
    let query = |expr, replay, strategy| Spec::Query {
        expr,
        replay,
        bare: false,
        strategy,
    };
    let zero = |replay| query(Const(0), replay, Strat::None);

    let mut specs: Vec<Spec> = (0..19).map(|_| zero(true)).collect();
    specs[4] = query(
        If(
            b(Read(6, false)),
            b(Const(0)),
            b(Add(b(Read(11, true)), b(Read(5, false)))),
        ),
        true,
        Strat::Fallback(0),
    );
    specs[5] = query(Read(4, true), true, Strat::None);
    specs[6] = query(Catch(b(Read(7, false))), true, Strat::None);
    specs[7] = query(Read(8, false), true, Strat::None);
    specs[8] = query(Read(9, false), true, Strat::None);
    specs[9] = query(
        If(b(Read(10, false)), b(Const(0)), b(Const(1))),
        true,
        Strat::None,
    );
    specs[10] = query(Read(11, true), true, Strat::None);
    specs[11] = query(
        Add(
            b(If(b(Read(12, false)), b(Const(0)), b(Const(0)))),
            b(Const(1)),
        ),
        true,
        Strat::None,
    );
    specs[12] = query(
        PanicIf(
            b(If(
                b(Read(4, false)),
                b(If(
                    b(Read(19, false)),
                    b(Read(15, false)),
                    b(Read(13, false)),
                )),
                b(Const(0)),
            )),
            3,
        ),
        false,
        Strat::Fallback(1),
    );
    specs[13] = query(Const(3), true, Strat::None);
    specs[14] = zero(false);
    specs[15] = query(
        Add(
            b(If(b(Read(10, false)), b(Const(0)), b(Read(22, true)))),
            b(Const(1)),
        ),
        true,
        Strat::None,
    );
    specs[16] = zero(false);
    specs.extend([3, 0, 0, 2].map(Spec::Input));
    let walked = Case {
        specs,
        panics: true,
        cycles: true,
        queries: 19,
        specs2: vec![zero(true), zero(true), zero(false), Spec::Input(1)],
        queries2: 3,
        ops: vec![Op::Get(4), Op::Set(19, 0), Op::Get(9)],
    };

    quiet(|| {
        differential::<Local>(&walked, 0x88f7);
        differential::<Shared>(&walked, 0x88f7);
    });
}

/// A one-phase case's snapshots with nodes `xs` patched to `to`, judged
/// by `at_overlap`; `live_too` also sets the live nodes' states.
fn overlap_verdict(
    case: &Case,
    xs: &[usize],
    to: NodeState,
    live_too: bool,
) -> Option<Result<(), String>> {
    let patches: Vec<(usize, NodeState)> = xs.iter().map(|&x| (x, to)).collect();
    overlap_verdict_of(case, &patches, live_too)
}

/// [`overlap_verdict`] with each node patched to its own state.
fn overlap_verdict_of(
    case: &Case,
    patches: &[(usize, NodeState)],
    live_too: bool,
) -> Option<Result<(), String>> {
    let phases = phases(case, 0);
    let serial = Serial::new(case, &phases);
    let verdict = Mutex::new(None);
    quiet(|| {
        run_two_until(case, &phases, None, 0, &|_| {}, &|_, prefix, live| {
            let mut prefix = prefix.clone();
            for &(x, to) in patches {
                for (_, snap) in &mut prefix {
                    snap.0[x] = to;
                }
                if live_too {
                    let (rt, h) = (live.world.rt(0), live.world.handle(0, x));
                    rt.nodes.get(h.erased().slot()).set_state(to as u8);
                }
            }
            let v = serial.at_overlap("patched", &prefix, live);
            let stop = v.is_some();
            *verdict.lock().unwrap() = v;
            stop
        });
    });
    verdict.into_inner().unwrap()
}

/// A node whose closure catches its Failed dep's panic lands Dirty,
/// poisoned, in every order, so no history leaves it Failed or Cyclic.
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_oracle_rejects_a_settled_reader_that_catches_its_dep() {
    let q = |expr| Spec::Query {
        expr,
        replay: false,
        bare: false,
        strategy: Strat::None,
    };
    let case = Case {
        specs: vec![
            q(Expr::Catch(Box::new(Expr::Read(1, false)))),
            q(Expr::PanicIf(Box::new(Expr::Read(2, false)), 0)),
            Spec::Input(7),
        ],
        panics: true,
        cycles: false,
        queries: 3,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(1), Op::Get(0)],
    };
    for to in [NodeState::Failed, NodeState::Cyclic] {
        match overlap_verdict(&case, &[0], to, false) {
            Some(Err(m)) => assert!(m.contains("node 0"), "{m}"),
            v => panic!("node 0 {to:?} passed: {v:?}"),
        }
    }
}

/// A cycle with no panic anywhere converges, so no history leaves its
/// members Failed, even though each one's recorded dep is the other.
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_oracle_rejects_a_failed_set_that_justifies_itself() {
    let q = |n| Spec::Query {
        expr: Expr::Read(n, false),
        replay: false,
        bare: false,
        strategy: Strat::Fixpoint {
            initial: 3,
            limit: 8,
        },
    };
    let case = Case {
        specs: vec![q(1), q(0)],
        panics: false,
        cycles: true,
        queries: 2,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Get(1)],
    };
    match overlap_verdict(&case, &[0, 1], NodeState::Failed, true) {
        Some(Err(_)) => {}
        v => panic!("a self-justified Failed set passed: {v:?}"),
    }
}

/// 0's first run is `Const(1)`, so no order computes 1 and none leaves
/// anything Failed. 0 patched Failed has no cause, though a later run of 0
/// would read 1, which reads 0: a check that read 0's patched state there
/// would make its own ground.
#[test]
#[cfg_attr(miri, ignore)]
fn wave6_two_thread_oracle_rejects_a_failed_node_grounded_by_its_own_rerun() {
    let q = |expr| Spec::Query {
        expr,
        replay: false,
        bare: false,
        strategy: Strat::None,
    };
    let case = Case {
        specs: vec![
            q(Expr::FirstRun(
                Box::new(Expr::Const(1)),
                Box::new(Expr::Read(1, false)),
            )),
            q(Expr::Read(0, false)),
        ],
        panics: false,
        cycles: false,
        queries: 2,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Get(0)],
    };
    match overlap_verdict(&case, &[0], NodeState::Failed, true) {
        Some(Err(m)) => assert!(m.contains("node 0"), "{m}"),
        v => panic!("node 0 Failed with no panic anywhere passed: {v:?}"),
    }
}

/// A Fixpoint cycle with no panic converges to 3 in every order, so no
/// history leaves its members Cyclic, even though each one's recorded dep
/// is the other, patched Cyclic: the oracle must reject the set.
#[test]
#[cfg_attr(miri, ignore)]
fn wave7_two_thread_oracle_rejects_a_cyclic_set_that_justifies_itself() {
    let q = |n| Spec::Query {
        expr: Expr::Read(n, false),
        replay: false,
        bare: false,
        strategy: Strat::Fixpoint {
            initial: 3,
            limit: 8,
        },
    };
    let case = Case {
        specs: vec![q(1), q(0)],
        panics: false,
        cycles: true,
        queries: 2,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Get(1)],
    };
    match overlap_verdict(&case, &[0, 1], NodeState::Cyclic, true) {
        Some(Err(_)) => {}
        v => panic!("a self-justified Cyclic set passed: {v:?}"),
    }
}

/// The converging pair of the test above with 0 patched Cyclic and 1
/// Failed: each stands only on the other's patched state, across the two
/// kinds.
#[test]
#[cfg_attr(miri, ignore)]
fn wave7_two_thread_oracle_rejects_a_mixed_set_that_justifies_itself() {
    let q = |n| Spec::Query {
        expr: Expr::Read(n, false),
        replay: false,
        bare: false,
        strategy: Strat::Fixpoint {
            initial: 3,
            limit: 8,
        },
    };
    let case = Case {
        specs: vec![q(1), q(0)],
        panics: false,
        cycles: true,
        queries: 2,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Get(1)],
    };
    let patches = [(0, NodeState::Cyclic), (1, NodeState::Failed)];
    match overlap_verdict_of(&case, &patches, true) {
        Some(Err(_)) => {}
        v => panic!("a self-justified mixed set passed: {v:?}"),
    }
}

/// A `get` entering the cycle at 0, which has no strategy, panics while the
/// cycle is discovered and leaves 0 and 1 New in every order: 0 patched
/// Cyclic is no history's, though a `get` entering at 0 panics as a cycle
/// would.
#[test]
#[cfg_attr(miri, ignore)]
fn wave7_two_thread_oracle_rejects_cyclic_on_a_discovery_panic() {
    let q = |n, strategy| Spec::Query {
        expr: Expr::Read(n, false),
        replay: false,
        bare: false,
        strategy,
    };
    let case = Case {
        specs: vec![
            q(1, Strat::None),
            q(
                0,
                Strat::Fixpoint {
                    initial: 3,
                    limit: 8,
                },
            ),
        ],
        panics: false,
        cycles: true,
        queries: 2,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Get(0)],
    };
    match overlap_verdict(&case, &[0], NodeState::Cyclic, true) {
        Some(Err(m)) => assert!(m.contains("node 0"), "{m}"),
        v => panic!("0 Cyclic on a discovery panic passed: {v:?}"),
    }
}

/// The plain cycle 0 -> 2 -> 3 -> 0, with 1 -> 3, lands nothing but New:
/// patched, 0 and 1 Cyclic and 2 Failed stand only on each other's states,
/// 3 included once it reads them.
#[test]
#[cfg_attr(miri, ignore)]
fn wave7_two_thread_oracle_rejects_a_mixed_set_over_a_plain_cycle() {
    let read = |n| Spec::Query {
        expr: Expr::Read(n, false),
        replay: false,
        bare: false,
        strategy: Strat::None,
    };
    let case = Case {
        specs: vec![read(2), read(3), read(3), read(0)],
        panics: false,
        cycles: true,
        queries: 4,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Get(1)],
    };
    let patches = [
        (0, NodeState::Cyclic),
        (1, NodeState::Cyclic),
        (2, NodeState::Failed),
    ];
    match overlap_verdict_of(&case, &patches, true) {
        Some(Err(_)) => {}
        v => panic!("a mixed set over a plain cycle passed: {v:?}"),
    }
}

/// 1 and 2 read each other with no strategy, and 0 reads 1 from outside
/// the cycle: patched Cyclic, 0 is no member of any cycle, whichever of
/// the pair the phase left Cyclic.
#[test]
#[cfg_attr(miri, ignore)]
fn wave7_two_thread_oracle_rejects_a_reader_of_a_cycle_left_cyclic() {
    let read = |n| Spec::Query {
        expr: Expr::Read(n, false),
        replay: false,
        bare: false,
        strategy: Strat::None,
    };
    let case = Case {
        specs: vec![read(1), read(2), read(1)],
        panics: false,
        cycles: true,
        queries: 3,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::TryGet(1), Op::TryGet(2)],
    };
    let patches = [
        (0, NodeState::Cyclic),
        (1, NodeState::Cyclic),
        (2, NodeState::Cyclic),
    ];
    match overlap_verdict_of(&case, &patches, true) {
        Some(Err(m)) => assert!(m.contains("node 0"), "{m}"),
        v => panic!("a reader of a cycle left Cyclic passed: {v:?}"),
    }
}

/// Every order leaves the plain cycle 0 -> 1 -> 0 Cyclic. 2's later runs
/// read 0 by `try_get`, which records the read, then 3, and 3 reads 2:
/// patched Failed, 2 and 3 have no cause but each other.
#[test]
#[cfg_attr(miri, ignore)]
fn wave7_two_thread_oracle_rejects_a_failed_pair_over_a_caught_cyclic_read() {
    let q = |expr| Spec::Query {
        expr,
        replay: false,
        bare: false,
        strategy: Strat::None,
    };
    let case = Case {
        specs: vec![
            q(Expr::Read(1, false)),
            q(Expr::Read(0, false)),
            q(Expr::FirstRun(
                Box::new(Expr::Const(1)),
                Box::new(Expr::Add(
                    Box::new(Expr::Read(0, true)),
                    Box::new(Expr::Read(3, false)),
                )),
            )),
            q(Expr::Read(2, false)),
        ],
        panics: false,
        cycles: true,
        queries: 4,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::TryGet(0), Op::Get(3)],
    };
    let patches = [(2, NodeState::Failed), (3, NodeState::Failed)];
    match overlap_verdict_of(&case, &patches, true) {
        Some(Err(_)) => {}
        v => panic!("a Failed pair over a caught Cyclic read passed: {v:?}"),
    }
}

/// A closure that catches a discard and reads again: at budget 1, 1's
/// speculative run reads bare 2, whose doomed frame opens a context whose
/// every entry is speculative, so the panic of 5, read only off 2's
/// placeholder for 4, unwinds into 1's first read as a discard. 1 catches
/// it, and its second read's compute of 2 popped in its place, so 1 landed
/// 9999 + 4 where the unbounded run never runs 5. 0x58b56de (a base seed)
/// is the same.
#[test]
#[cfg_attr(miri, ignore)]
fn caught_discard_then_read() {
    let read = |n| Box::new(Expr::Read(n, false));
    let case = Case {
        specs: vec![
            replayable(Expr::Add(read(3), read(1)), Strat::None),
            replayable(
                Expr::Add(
                    Box::new(Expr::Catch(Box::new(Expr::Read(2, true)))),
                    Box::new(Expr::Catch(read(2))),
                ),
                Strat::None,
            ),
            Spec::Query {
                expr: Expr::If(read(4), read(5), read(6)),
                replay: true,
                bare: true,
                strategy: Strat::None,
            },
            replayable(Expr::Read(7, false), Strat::None),
            replayable(Expr::Read(8, false), Strat::None),
            replayable(Expr::PanicIf(Box::new(Expr::Const(0)), 0), Strat::None),
            replayable(Expr::Const(4), Strat::None),
            Spec::Input(2),
            Spec::Input(3),
        ],
        panics: true,
        cycles: false,
        queries: 7,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0)],
    };
    quiet(|| {
        differential::<Local>(&case, 0);
        differential::<Shared>(&case, 0);
        let seed = gen_base(0x58b_56de);
        differential::<Local>(&seed, 0x58b_56de);
        differential::<Shared>(&seed, 0x58b_56de);
    });
}

/// 0 catches the cycle of its read of 1, reads 2 and panics on its own:
/// it lands the same whether 2 is Clean or still to compute. A Clean read
/// kept the caught cycle's kind, so 0's panic went on as the cycle and 0
/// landed as a cycle member does. In 0xb5e0053b30 at budget 1 the driver
/// had computed the read node, so 0 landed New where the unbounded run,
/// computing it inline, landed 0 Failed.
#[test]
#[cfg_attr(miri, ignore)]
fn caught_cycle_then_clean_read() {
    let read = |n| Box::new(Expr::Read(n, false));
    let case = |ops| Case {
        specs: vec![
            replayable(
                Expr::Add(
                    Box::new(Expr::Catch(read(1))),
                    Box::new(Expr::Add(
                        read(2),
                        Box::new(Expr::PanicIf(Box::new(Expr::Const(0)), 0)),
                    )),
                ),
                Strat::None,
            ),
            replayable(Expr::Read(0, false), Strat::None),
            replayable(Expr::Const(5), Strat::None),
        ],
        panics: true,
        cycles: true,
        queries: 3,
        specs2: Vec::new(),
        queries2: 0,
        ops,
    };
    let last = |t: Trace| {
        let (out, snap) = t.last().expect("an op ran").clone();
        (out, snap.0[0])
    };
    quiet(|| {
        let (fresh, warm) = (case(vec![Op::Get(0)]), case(vec![Op::Get(2), Op::Get(0)]));
        assert_eq!(
            last(run::<Local>(&warm, None, 0)),
            last(run::<Local>(&fresh, None, 0))
        );
        assert_eq!(
            last(run::<Shared>(&warm, None, 0)),
            last(run::<Shared>(&fresh, None, 0))
        );
        for c in [&fresh, &warm] {
            differential::<Local>(c, 0);
            differential::<Shared>(c, 0);
        }
        let seed = gen_case(0xb5_e005_3b30, false, false);
        differential::<Local>(&seed, 0xb5_e005_3b30);
        differential::<Shared>(&seed, 0xb5_e005_3b30);
    });
}

/// Shared seeds a deferral of group exits to the read's end got wrong on
/// one thread, against Local: in 0x68d3 a verify-only visit of a member
/// whose exit was deferred computed it, and in 0x40d8 a discard dropped
/// the deferred exits of landings it did not undo.
#[test]
#[cfg_attr(miri, ignore)]
fn deferred_exit_seeds() {
    for seed in [0x40d8, 0x68d3] {
        let case = gen_case(seed, false, false);
        quiet(|| {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        });
    }
}

/// A run taken for the value it returned is no own panic: the history
/// checks share one cache across a phase's nodes, so a run one node's read
/// took for its value must not answer a later own-panic check of it.
#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_history_takes_a_valued_run_for_its_value_only() {
    let read = |n| Spec::Query {
        expr: Expr::Read(n, false),
        replay: false,
        bare: false,
        strategy: Strat::None,
    };
    let case = Case {
        specs: vec![read(2), read(2), Spec::Input(5)],
        panics: false,
        cycles: false,
        queries: 2,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Get(1)],
    };
    let phases = phases(&case, 0);
    let serial = Serial::new(&case, &phases);
    let checked = Mutex::new(None);
    quiet(|| {
        run_two_until(&case, &phases, None, 0, &|_| {}, &|_, prefix, live| {
            let mut cx = serial.history("cache", prefix, live).expect("a read phase");
            let (i, run) = Serial::last_run(&cx, 0).expect("0 ran");
            assert_eq!(run.value, Some(5));
            let valued = serial.run_taken(&mut cx, i, &run, Some(5));
            let own = serial.run_taken(&mut cx, i, &run, None);
            *checked.lock().unwrap() = Some((valued, own));
            true
        });
    });
    assert_eq!(checked.into_inner().unwrap(), Some((true, false)));
}

/// 0 and 1 both first-run `Const(1)` then read 2, which panics on its own
/// input. The phase runs both for the first time and their gets return 1,
/// so patched Failed both are untrue of the inputs, and must be judged
/// alike whichever the check takes first.
#[test]
#[cfg_attr(miri, ignore)]
fn wave7_two_thread_oracle_judges_symmetric_failed_extras_alike() {
    let q = |expr| Spec::Query {
        expr,
        replay: false,
        bare: false,
        strategy: Strat::None,
    };
    let first_then_read2 = || {
        q(Expr::FirstRun(
            Box::new(Expr::Const(1)),
            Box::new(Expr::Read(2, false)),
        ))
    };
    let case = Case {
        specs: vec![
            first_then_read2(),
            first_then_read2(),
            q(Expr::PanicIf(Box::new(Expr::Read(3, false)), 0)),
            Spec::Input(7),
        ],
        panics: true,
        cycles: false,
        queries: 4,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Get(1)],
    };
    match overlap_verdict(&case, &[0, 1], NodeState::Failed, true) {
        Some(Err(m)) => assert!(m.contains("node 0") && m.contains("node 1"), "{m}"),
        v => panic!("symmetric Failed extras judged apart: {v:?}"),
    }
}

/// 0's first run returns 1, and any later run of its closure panics. Op 0
/// runs it, and the read phase's `get(0)` reads it Clean in every order,
/// so no order runs the closure again and none leaves 0 Failed. A replay
/// that recomputes 0 as a fresh query runs the branch no order runs, so
/// it must not make 0 patched Failed true of the phase's inputs.
#[test]
#[cfg_attr(miri, ignore)]
fn wave8_two_thread_oracle_rejects_a_failed_node_only_a_rerun_of_it_clean_gives() {
    let case = Case {
        specs: vec![
            Spec::Query {
                expr: Expr::FirstRun(
                    Box::new(Expr::Const(1)),
                    Box::new(Expr::PanicIf(Box::new(Expr::Const(0)), 0)),
                ),
                replay: false,
                bare: false,
                strategy: Strat::None,
            },
            Spec::Input(7),
            Spec::Input(7),
        ],
        panics: true,
        cycles: false,
        queries: 3,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Set(2, 5), Op::Get(0), Op::Get(1)],
    };
    match patched_zero_failed(&case) {
        Some(Err(m)) => assert!(m.contains("node 0"), "{m}"),
        v => panic!("0 Failed, which no order runs its closure for, passed: {v:?}"),
    }
}

/// As above, but 0 reads 3, which always gives 1, so the set of 2 leaves
/// 0 Dirty: every order verifies 0 over 3's unchanged value and keeps its
/// first run's 1, and none runs the closure again.
#[test]
#[cfg_attr(miri, ignore)]
fn wave8_two_thread_oracle_rejects_a_failed_node_an_order_verifies_clean() {
    let query = |expr| Spec::Query {
        expr,
        replay: false,
        bare: false,
        strategy: Strat::None,
    };
    let case = Case {
        specs: vec![
            query(Expr::FirstRun(
                Box::new(Expr::Read(3, false)),
                Box::new(Expr::PanicIf(Box::new(Expr::Const(0)), 0)),
            )),
            Spec::Input(7),
            Spec::Input(7),
            query(Expr::If(
                Box::new(Expr::Read(2, false)),
                Box::new(Expr::Const(1)),
                Box::new(Expr::Const(1)),
            )),
        ],
        panics: true,
        cycles: false,
        queries: 4,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Set(2, 5), Op::Get(0), Op::Get(1)],
    };
    match patched_zero_failed(&case) {
        Some(Err(m)) => assert!(m.contains("node 0"), "{m}"),
        v => panic!("0 Failed, which every order verifies Clean, passed: {v:?}"),
    }
}

/// 0's first run panics over 3's 7, which lands it Failed with nothing an
/// order verifies, so after the set of 2 every order reruns 0 to 5 though
/// 3 still gives 7. A first run counts as kept only if it landed Clean.
#[test]
#[cfg_attr(miri, ignore)]
fn wave8_two_thread_oracle_rejects_a_failed_node_from_a_first_run_no_order_keeps() {
    let query = |expr| Spec::Query {
        expr,
        replay: false,
        bare: false,
        strategy: Strat::None,
    };
    let case = Case {
        specs: vec![
            query(Expr::FirstRun(
                Box::new(Expr::PanicIf(Box::new(Expr::Read(3, false)), 0)),
                Box::new(Expr::Const(5)),
            )),
            Spec::Input(7),
            Spec::Input(7),
            query(Expr::If(
                Box::new(Expr::Read(2, false)),
                Box::new(Expr::Const(7)),
                Box::new(Expr::Const(7)),
            )),
        ],
        panics: true,
        cycles: false,
        queries: 4,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Set(2, 5), Op::Get(0), Op::Get(1)],
    };
    match patched_zero_failed(&case) {
        Some(Err(m)) => assert!(m.contains("node 0"), "{m}"),
        v => panic!("0 Failed, which every order reruns to 5, passed: {v:?}"),
    }
}

/// The two-thread oracle's verdict on `case`'s read phase of ops 2 and 3
/// with node 0 patched Failed after it.
fn patched_zero_failed(case: &Case) -> Option<Result<(), String>> {
    let phases = phases(case, 0);
    let serial = Serial::new(case, &phases);
    let verdict = Mutex::new(None);
    quiet(|| {
        run_two_until(case, &phases, None, 0, &|_| {}, &|_, prefix, live| {
            let mut prefix = prefix.clone();
            for (_, snap) in &mut prefix[2..] {
                snap.0[0] = NodeState::Failed;
            }
            let (rt, h) = (live.world.rt(0), live.world.handle(0, 0));
            rt.nodes
                .get(h.erased().slot())
                .set_state(NodeState::Failed as u8);
            let v = serial.at_overlap("patched", &prefix, live);
            let stop = v.is_some();
            *verdict.lock().unwrap() = v;
            stop
        });
    });
    verdict.into_inner().unwrap()
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
        0xafd,
        0xb0e,
        0xbb6,
        0xbbc,
        0xbc2,
        0xbe7,
        0xcb7,
        0xdd3,
        0xe3f,
        0xe7b,
        0xf41,
        0xfee,
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
        0x551d_e2c5_404a_84b1,
        0x84a1_d45b_e6f9_b165,
        0xa649_6004_aa21_dd3b,
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

/// Seeds that differ under the strict comparator where the unbounded run
/// leaves a node Dirty and the bounded run lands it Clean; 0xa5b and 0x1359
/// return other outcomes too.
#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "T8-strict-group-c-dirty-for-clean"]
fn strict_dirty_for_clean_seeds() {
    let cases = [0x41eb, 0xd60]
        .map(|seed| (seed, gen_case(seed, false, false)))
        .into_iter()
        .chain(
            [0xa5b, 0x1359, 0x0876_ebf0_5e41_2095, 0xa8cc_85b9_bbfd_657d]
                .map(|seed| (seed, gen_base(seed))),
        );
    for (seed, case) in cases {
        quiet(|| {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        });
    }
}

/// Seeds that differ under the strict comparator where the unbounded run
/// computes nodes the bounded run leaves New.
#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "T8-strict-computed-left-new"]
fn strict_computed_left_new_seeds() {
    for seed in [0x2997, 0xd4f2] {
        let case = gen_base(seed);
        quiet(|| {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        });
    }
}

/// 0xd4f2 shrunk: `get(q2)`'s walk computes New former member q1 before
/// q2. At frame budget 3 that compute is doomed at q1's `try q2` and the
/// walk's rest is stashed for q10's retry, which reads q2 after the
/// context landed it Clean. The retry must take the walk so q1 lands
/// Clean as in the unbounded run, not stay New.
#[test]
#[cfg_attr(miri, ignore)]
fn a_walk_stashed_for_a_clean_read_lands_its_new_entries() {
    let q = |expr, bare| Spec::Query {
        expr,
        replay: true,
        bare,
        strategy: Strat::None,
    };
    let mut specs: Vec<Spec> = (0..22).map(|_| q(Expr::Const(0), false)).collect();
    specs[0] = q(Expr::Read(1, false), false);
    specs[1] = q(Expr::Read(2, true), false);
    specs[2] = q(
        Expr::If(
            Box::new(Expr::Read(22, false)),
            Box::new(Expr::Const(0)),
            Box::new(Expr::Catch(Box::new(Expr::Read(1, false)))),
        ),
        false,
    );
    specs[9] = q(Expr::Read(10, false), true);
    specs[10] = q(Expr::Read(2, false), false);
    specs.push(Spec::Input(1));
    let case = Case {
        specs,
        panics: true,
        cycles: true,
        queries: 22,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Set(22, 4), Op::Get(9)],
    };
    quiet(|| {
        differential::<Local>(&case, 0xd4f2);
        differential::<Shared>(&case, 0xd4f2);
    });
}

/// 0x2165's paired case run single-threaded in every order of its read
/// pairs: with the pairs swapped, op 10 `Get(7)` left 19 and 20 Clean at
/// budget 1 where the unbounded run leaves them Dirty: 10's retry reads
/// past the unwind out of 9 that ends the unbounded run's read, and runs
/// 3, which the unbounded run never does.
#[test]
#[ignore = "T8-retry-reads-past-an-unwind-2165"]
fn seed_2165_in_every_order() {
    let case = gen_case(0x2165, false, true);
    let phases = phases(&case, 0x2165);
    let pairs = phases.iter().filter(|p| p.ops().len() == 2).count();
    quiet(|| {
        for swaps in 0..1u64 << pairs {
            let order = Order::new(&case, &phases, swaps);
            differential::<Local>(&order.case, 0x2165);
            differential::<Shared>(&order.case, 0x2165);
        }
    });
}

/// 0x783 at budget 3: in the unbounded run 19 reads 18 before the episode
/// of 16 to 18 forms, lands Dirty over its superseded landing and poisons
/// 14, which reads 19 before 19's later Clean run. In the bounded run a
/// speculative entry reaches 19 while 19's marked frame still holds the
/// Dirty landing for its retry, so that entry must not land 19 Clean.
#[test]
fn seed_783_at_budget_3() {
    let case = gen_case(0x783, false, false);
    quiet(|| {
        differential::<Local>(&case, 0x783);
        differential::<Shared>(&case, 0x783);
    });
}

/// A query whose run read the node being deleted keeps that dependency,
/// so the delete is refused at every budget. In 0x2343 node 15's run
/// completes with the read; in 0xc17 node 7's claim, released by another
/// node's user panic, lands New over the deps its run read.
#[test]
fn delete_of_a_node_a_query_read_is_refused() {
    for (seed, op) in [(0x2343, 9), (0xc17, 5)] {
        let mut case = gen_case(seed, false, false);
        assert!(matches!(case.ops[op], Op::Delete(_)));
        case.ops.truncate(op + 1);
        quiet(|| {
            for (frames, bytes) in std::iter::once((None, 0)).chain(BUDGETS) {
                let local = run::<Local>(&case, frames, bytes);
                let shared = run::<Shared>(&case, frames, bytes);
                for trace in [&local, &shared] {
                    assert_eq!(
                        trace[op].0,
                        Out::Deleted(Err("HasDependents".into())),
                        "seed {seed:#x} at {frames:?}/{bytes}"
                    );
                }
            }
        });
    }
}

/// A diverted frame's Clean read with nothing to take is a fast read: with
/// every retry's frame diverted, each case's bounded runs give the
/// outcomes, states and deps they give with only the frames a delivery
/// waits for diverted, `FirstRun` closures included.
#[test]
#[cfg_attr(miri, ignore)]
fn a_diverted_read_with_no_delivery_is_a_fast_read() {
    fn runs<C: Cells>(case: &Case, frames: u32) {
        let run = || {
            catch_unwind(AssertUnwindSafe(|| run::<C>(case, Some(frames), 0))).map_err(|p| {
                p.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|m| (*m).to_owned()))
            })
        };
        let normal = run();
        crate::runtime::replay_impl::DIVERT_EVERY_RETRY.with(|d| d.set(true));
        let every = run();
        crate::runtime::replay_impl::DIVERT_EVERY_RETRY.with(|d| d.set(false));
        assert_eq!(normal, every, "budget {frames}");
    }
    quiet(|| {
        for seed in 0..u64::from(cases() / 2) {
            let case = gen_case(seed, false, false);
            for frames in [1, 2, 3] {
                runs::<Local>(&case, frames);
                runs::<Shared>(&case, frames);
            }
        }
    });
}

/// Seeds whose retry reads a node after it landed Clean, where the
/// unbounded frame read it before: 0xa5b's read of q7, kept with the
/// poisoned value q7 landed Dirty with, and 0xbe's doomed read of n12,
/// whose outcome was Dirty at 8. The retry must take what the context
/// holds for the read, not the Clean value, so `Get(0)` is 7781 and
/// `Get(9)` 16 as in the unbounded run. 0x41b1 is the same read, whose
/// `Get(4)` returned `Panic("cycle")` for `Value(3)`.
#[test]
#[cfg_attr(miri, ignore)]
fn a_retrys_clean_read_takes_its_delivery_seeds() {
    for seed in [0xa5b, 0xbe, 0x41b1] {
        let case = gen_case(seed, false, false);
        quiet(|| {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        });
    }
}

/// Paired cases run single-threaded, the same read: at budget 1, 0x52b's
/// `Get(3)` left 3 and 4 Clean where the unbounded run leaves them Dirty,
/// under the CI depth-1 environment too, and 0x427d's `TryGet(14)` left
/// 14 Dirty where the unbounded run lands it Failed.
#[test]
#[cfg_attr(miri, ignore)]
fn paired_clean_read_delivery_seeds() {
    for seed in [0x52b, 0x427d] {
        let case = gen_case(seed, false, true);
        quiet(|| {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        });
    }
}

/// High-range strict scan seeds (from 630042492734) that landed Dirty
/// for Clean before a retry's Clean read took its delivery; 0x92b175829f
/// is a stashed walk rest resumed past a root that had landed Clean.
#[test]
#[cfg_attr(miri, ignore)]
fn high_range_clean_read_delivery_seeds() {
    for seed in [
        0x92_b175_80fc,
        0x92_b175_83ef,
        0x92_b175_829f,
        0x92_b175_8479,
    ] {
        let case = gen_case(seed, false, false);
        quiet(|| {
            differential::<Local>(&case, seed);
            differential::<Shared>(&case, seed);
        });
    }
}

/// 0xa02 (see [`cross_seeds`]): a speculative landing read the second
/// runtime's input before a set of it, and a later read returns the old
/// value.
#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "T8-speculative-landings"]
fn speculative_landing_seeds() {
    let case = gen_case(0xa02, true, false);
    quiet(|| {
        differential::<Local>(&case, 0xa02);
        differential::<Shared>(&case, 0xa02);
    });
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
    #[ignore = "T8-strict-group-c-dirty-for-clean"]
    fn replay_differential_local(seed in any::<u64>()) {
        quiet(|| differential::<Local>(&gen_case(seed, false, false), seed));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[ignore = "T8-strict-group-c-dirty-for-clean"]
    fn replay_differential_shared(seed in any::<u64>()) {
        quiet(|| differential::<Shared>(&gen_case(seed, false, false), seed));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[ignore = "T8-strict-group-c-dirty-for-clean, T8-strict-computed-left-new"]
    fn replay_differential_shared_two_threads(seed in any::<u64>()) {
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

/// The case for `seed`, the base one if `REPLAY_BASE` is set, with a
/// second runtime if `REPLAY_CROSS` is, and the two-thread form of the
/// case if `REPLAY_TWO` or `REPLAY_PAIRED` is: `REPLAY_PAIRED` alone runs
/// that case through the single-thread differential.
fn case_of(seed: u64) -> Case {
    if std::env::var_os("REPLAY_BASE").is_some() {
        gen_base(seed)
    } else {
        gen_case(
            seed,
            std::env::var_os("REPLAY_CROSS").is_some(),
            two_threads() || std::env::var_os("REPLAY_PAIRED").is_some(),
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

fn erased<T: Value>(pinned: bool, opts: QueryOpts<T>) -> crate::cycle::ErasedOpts {
    let mut opts = crate::cycle::erase_opts(opts);
    if pinned {
        opts.flags |= crate::segmented_nodes::NOT_REPLAYABLE;
    }
    opts
}

fn has_first_run(e: &Expr) -> bool {
    match e {
        Expr::Const(_) | Expr::Read(..) | Expr::ReadOther(..) => false,
        Expr::FirstRun(..) => true,
        Expr::Add(a, b) => has_first_run(a) || has_first_run(b),
        Expr::If(a, b, c) => has_first_run(a) || has_first_run(b) || has_first_run(c),
        Expr::PanicIf(a, _) | Expr::Catch(a) => has_first_run(a),
    }
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
/// is set, on two threads if `REPLAY_TWO` is, and over the two-thread
/// cases on one thread if `REPLAY_PAIRED` is.
#[test]
#[ignore = "a diagnostic that scans seeds"]
fn replay_scan() {
    let n: u64 = std::env::var("REPLAY_SCAN").map_or(2000, |v| v.parse().expect("a count"));
    if let Ok(seed) = std::env::var("REPLAY_CHECK") {
        let seed = u64::from_str_radix(&seed, 16).expect("a hex seed");
        check(seed);
        return;
    }
    let from: u64 = std::env::var("REPLAY_FROM").map_or(0, |v| v.parse().expect("a seed"));
    scan(from..n, std::env::var_os("REPLAY_ALL").is_some(), check);
}

/// Runs `check` on each seed and panics at the end if any failed, naming
/// the smallest failing case. With `all` every seed runs and each failure
/// is printed; without it, a seed no smaller than the smallest failure so
/// far is skipped.
fn scan(seeds: impl Iterator<Item = u64>, all: bool, check: impl Fn(u64)) {
    let mut best: Option<(usize, u64)> = None;
    let mut failing: Vec<u64> = Vec::new();
    quiet(|| {
        for seed in seeds {
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
                    let head: Vec<&str> = message
                        .lines()
                        .take_while(|l| !l.starts_with("ops:"))
                        .collect();
                    eprintln!("FAIL {seed:x}: {}", head.join(" | "));
                }
                failing.push(seed);
                if best.is_none_or(|(w, _)| weight < w) {
                    best = Some((weight, seed));
                }
            }
        }
    });
    eprintln!("smallest failing: {best:?}");
    eprintln!(
        "comparisons ended at a delete a speculative landing refused: {}",
        SPECULATIVE_DELETES.load(Relaxed)
    );
    if let Some((_, smallest)) = best {
        let skipped = if all {
            ""
        } else {
            " (heavier seeds were skipped; set REPLAY_ALL to run them)"
        };
        panic!(
            "{} seeds failed{skipped}, the first {:x?}; smallest failing: {smallest:#x}",
            failing.len(),
            &failing[..failing.len().min(8)],
        );
    }
}

#[test]
fn scan_passes_when_every_seed_passes() {
    scan(0..4, true, |_| {});
    scan(0..4, false, |_| {});
}

#[test]
#[should_panic(expected = "2 seeds failed, the first [1, 3]")]
fn scan_panics_when_a_seed_fails() {
    scan(0..5, true, |seed| assert!(seed % 2 == 0, "seed {seed}"));
}

#[test]
#[should_panic(expected = "seeds failed (heavier seeds were skipped")]
fn scan_without_all_panics_when_a_seed_fails() {
    scan(0..5, false, |seed| assert!(seed % 2 == 0, "seed {seed}"));
}

fn reads_any(e: &Expr, hit: &impl Fn(usize) -> bool) -> bool {
    match e {
        Expr::Const(_) | Expr::ReadOther(..) => false,
        Expr::Read(target, _) => hit(*target),
        Expr::Add(a, b) | Expr::FirstRun(a, b) => reads_any(a, hit) || reads_any(b, hit),
        Expr::If(c, a, b) => reads_any(c, hit) || reads_any(a, hit) || reads_any(b, hit),
        Expr::PanicIf(inner, _) | Expr::Catch(inner) => reads_any(inner, hit),
    }
}

/// A recreated node that read the dissolving root's template could run the
/// template's first-run-only read for the first time, at a point that
/// differs by budget (seed 0x16eb).
#[test]
fn no_recreated_node_reads_the_template() {
    for seed in (0..4000).chain([0x16eb]) {
        let case = case_of(seed);
        let template = matches!(
            case.specs.get(2),
            Some(Spec::Query {
                expr: Expr::FirstRun(..),
                ..
            })
        );
        if !template {
            continue;
        }
        for op in &case.ops {
            if let Op::Recreate(_, Spec::Query { expr, .. }) = op {
                assert!(
                    !reads_any(expr, &|target| target < 4),
                    "seed {seed:#x}: {expr:?}"
                );
            }
        }
    }
}

/// The ring 0 -> 1 -> 2 -> 3 -> 0, never computed, read by `TryGet(0)` and
/// `TryGet(2)` on the two threads at once: every order and every two-thread
/// run gives both the cycle through all four. `patch` edits op 0's
/// participants, as nodes, before the oracle sees them, standing in for a
/// cycle built from one thread's frames with members another thread landed
/// first left out; returns whether the oracle took the run. A real such
/// cycle needs a member that branches on a cycle error, and which member
/// computes first decides those.
fn cycle_from_own_frames(patch: fn(&mut Vec<usize>)) -> bool {
    let query = |expr| Spec::Query {
        expr,
        replay: true,
        bare: false,
        strategy: Strat::None,
    };
    let read = |n| query(Expr::Read(n, false));
    let case = Case {
        specs: vec![read(1), read(2), read(3), read(0), query(Expr::Const(1))],
        panics: false,
        cycles: true,
        queries: 5,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::TryGet(0), Op::TryGet(2)],
    };
    let phases = [Phase::Read([Some(0), Some(1)])];
    let slots = Mutex::new(Vec::new());
    let fix = |t: &mut Trace| {
        let slot = slots.lock().unwrap();
        let Out::Cycle(c) = &mut t[0].0 else {
            panic!("op 0 got {:?}, not the ring's cycle", t[0].0)
        };
        let mut nodes: Vec<usize> = c
            .iter()
            .map(|p| slot.iter().position(|s| s == p).unwrap())
            .collect();
        nodes.sort_unstable();
        assert_eq!(nodes, [0, 1, 2, 3]);
        patch(&mut nodes);
        *c = nodes.iter().map(|&x| slot[x]).collect();
        c.sort_unstable();
    };
    let mut took = false;
    quiet(|| {
        for mask in [0, 1] {
            let order = Order::new(&case, &phases, mask);
            let ring = run::<Shared>(&order.case, None, 0);
            assert!(ring
                .iter()
                .all(|r| matches!(&r.0, Out::Cycle(c) if c.len() == 4)));
        }
        let serial = Serial::new(&case, &phases);
        let verdict = Mutex::new(None);
        let mut got = run_two_until(&case, &phases, None, 0, &|_| {}, &|_, prefix, live| {
            *slots.lock().unwrap() = (0..case.specs.len())
                .map(|x| live.world.handle(0, x).erased().slot())
                .collect();
            let mut prefix = prefix.clone();
            fix(&mut prefix);
            let v = serial.at_overlap("own-frames ring", &prefix, live);
            let stop = v.is_some();
            *verdict.lock().unwrap() = v;
            stop
        });
        fix(&mut got);
        let verdict = verdict.into_inner().unwrap();
        took = catch_unwind(AssertUnwindSafe(|| {
            serial.check("own-frames ring", &got, verdict)
        }))
        .is_ok();
    });
    took
}

#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_oracle_takes_a_cycle_built_from_this_threads_frames() {
    assert!(cycle_from_own_frames(|c| c.retain(|&x| x != 3)));
}

#[test]
#[cfg_attr(miri, ignore)]
fn two_thread_oracle_rejects_a_made_up_cycle() {
    // 4 is no member of the ring, so a cycle through it is no part of an
    // order's.
    assert!(!cycle_from_own_frames(|c| c.push(4)));
    assert!(!cycle_from_own_frames(|c| c.clear()));
}

/// Seed 0x3904's first phase at a budget of 1 frame: op 0 `Get(0)` on one
/// thread, op 1 `Get(6)` on the other. Some runs give 1's `try_get(2)` a
/// cycle that 6 then reads as 7777, so 5 is 2 and 4 panics, which no order
/// does; the members 4's claims released, read again in the phase, fail
/// through 4, and the phase leaves 1 to 6 Failed. In the first such run,
/// `patch` edits the phase's recorded closure runs before the oracle sees
/// them; returns whether the oracle took the run, every run before it
/// checked as usual, and whether 4's history grounds it.
fn cascade_from_a_history_panic(patch: fn(&mut HashMap<usize, Vec<Eval>>)) -> (bool, bool) {
    let case = gen_case(0x3904, false, true);
    let phases = phases(&case, 0x3904);
    assert_eq!(phases[0].ops(), [0, 1]);
    let mut took = None;
    quiet(|| {
        for _ in 0..2000 {
            let serial = Serial::new(&case, &phases);
            let verdict = Mutex::new(None);
            let hit = AtomicBool::new(false);
            let grounded = AtomicBool::new(false);
            let got = run_two_until(&case, &phases, Some(1), 0, &|_| {}, &|_, prefix, live| {
                let states = &prefix.last().unwrap().1 .0;
                if prefix.len() == 2 && (1..=6).all(|x| states[x] == NodeState::Failed) {
                    hit.store(true, SeqCst);
                    patch(&mut live.evals.lock().unwrap());
                    grounded.store(serial.history_cause("0x3904", prefix, live, 4), SeqCst);
                }
                let v = serial.at_overlap("0x3904", prefix, live);
                let stop = v.is_some();
                *verdict.lock().unwrap() = v;
                stop
            });
            let verdict = verdict.into_inner().unwrap();
            if !hit.load(SeqCst) {
                serial.check("0x3904", &got, verdict);
                continue;
            }
            took = Some((
                catch_unwind(AssertUnwindSafe(|| serial.check("0x3904", &got, verdict))).is_ok(),
                grounded.load(SeqCst),
            ));
            return;
        }
    });
    took.expect("no run of 0x3904 left 1 to 6 Failed")
}

#[test]
#[ignore = "T8-oracle-history-fixture: doom reservations remove the budget-1 runs (0x3904, and 0x527 of 644fcdf7) whose Failed set these fixtures ground"]
#[cfg_attr(miri, ignore)]
fn two_thread_oracle_takes_a_failed_set_grounded_in_its_history() {
    assert_eq!(cascade_from_a_history_panic(|_| {}), (true, true));
}

#[test]
#[ignore = "T8-oracle-history-fixture: doom reservations remove the budget-1 runs (0x3904, and 0x527 of 644fcdf7) whose Failed set these fixtures ground"]
#[cfg_attr(miri, ignore)]
fn two_thread_oracle_rejects_a_history_grounding_over_a_wrong_value() {
    assert_eq!(
        cascade_from_a_history_panic(|evals| {
            for e in evals.values_mut().flatten() {
                if e.node == (0, 5) && e.value == Some(2) {
                    e.value = Some(3);
                }
            }
        }),
        (false, false)
    );
}

/// As above with every value kept but 1's caught cycle naming no
/// participant, which no order's read of 2 gives: 7777 still follows from
/// it, but 6, 5 and 4 ground through a read the oracle does not take.
#[test]
#[ignore = "T8-oracle-history-fixture: doom reservations remove the budget-1 runs (0x3904, and 0x527 of 644fcdf7) whose Failed set these fixtures ground"]
#[cfg_attr(miri, ignore)]
fn two_thread_oracle_rejects_a_history_grounding_over_an_untaken_read() {
    assert_eq!(
        cascade_from_a_history_panic(|evals| {
            for e in evals.values_mut().flatten() {
                if e.node == (0, 1) && e.value == Some(7777) {
                    for r in &mut e.reads {
                        if let ReadOut::Cycle(c) = &mut r.3 {
                            c.clear();
                        }
                    }
                }
            }
        }),
        (false, false)
    );
}

/// Seed 0x55ac771450d16c5b's op 0, `Get(0)` on a fresh runtime: node 8 lands
/// Failed with the union of its deps, which keeps 10 from an earlier
/// cycle-iteration run. Unbounded, 11's walk of Failed 12 memoizes 8 under
/// 0's frame, so 7's rerun reads 8 directly. At a budget of 1 those entries
/// are 0's virtual frame's, a doom in the driver's pop of that frame
/// restarted the driver and dropped them, and 7's retry walked Failed 8 and
/// computed New 10, which landed Failed too.
#[test]
#[cfg_attr(miri, ignore)]
fn a_failed_union_dep_is_not_computed_by_a_retry_walk() {
    let seed = 0x55ac_7714_50d1_6c5b;
    let mut case = gen_case(seed, false, false);
    case.ops.truncate(1);
    assert!(matches!(case.ops[0], Op::Get(0)));
    let failed = |t: &Trace| -> (Out, Vec<usize>) {
        let (out, snap) = &t[0];
        let set = (snap.0.iter().enumerate())
            .filter(|(_, s)| **s == NodeState::Failed)
            .map(|(x, _)| x)
            .collect();
        (out.clone(), set)
    };
    let want = (
        Out::Panic("boom 0:12".to_owned()),
        vec![1, 2, 3, 6, 7, 8, 11, 12],
    );
    quiet(|| {
        assert_eq!(failed(&run::<Local>(&case, None, 0)), want);
        assert_eq!(failed(&run::<Shared>(&case, None, 0)), want);
        let bounded = [
            failed(&run::<Local>(&case, Some(1), 0)),
            failed(&run::<Shared>(&case, Some(1), 0)),
        ];
        assert_eq!(bounded, [want.clone(), want], "budget 1, Local then Shared");
    });
}

/// Seed 0xbd77f8fdde0ce94f's op 0, `Get(0)` on a fresh runtime: 5's member
/// run reads 19, which reads 8, whose own code panics, so unbounded 5 is
/// marked failed and lands Failed. At a budget of 1, 8's run is a stand-in
/// finishing its virtual frame, and its mark counts as a member failure
/// nested in 5's run, so 5 is not marked and lands New.
#[test]
#[cfg_attr(miri, ignore)]
fn a_stand_ins_panic_is_not_a_failure_nested_in_its_readers_member_run() {
    let seed = 0xbd77_f8fd_de0c_e94f;
    let mut case = gen_case(seed, false, false);
    case.ops.truncate(1);
    assert!(matches!(case.ops[0], Op::Get(0)));
    let failed = |t: &Trace| -> (Out, Vec<usize>) {
        let (out, snap) = &t[0];
        let set = (snap.0.iter().enumerate())
            .filter(|(_, s)| **s == NodeState::Failed)
            .map(|(x, _)| x)
            .collect();
        (out.clone(), set)
    };
    let want = (Out::Panic("boom 0:8".to_owned()), vec![5, 8, 19]);
    quiet(|| {
        assert_eq!(failed(&run::<Local>(&case, None, 0)), want);
        assert_eq!(failed(&run::<Shared>(&case, None, 0)), want);
        let bounded = [
            failed(&run::<Local>(&case, Some(1), 0)),
            failed(&run::<Shared>(&case, Some(1), 0)),
        ];
        assert_eq!(bounded, [want.clone(), want], "budget 1, Local then Shared");
    });
}

/// Seed 0x58b5909's base case: at budget 3, op 4 Get(4) must leave the
/// unbounded run's Failed set (bisected: passes on 7df62769, fails from
/// e9eea59d on, where the bounded run lands node 9 Failed too).
#[test]
#[cfg_attr(miri, ignore)]
fn wave8_local_seed_58b5909_failed_set() {
    quiet(|| differential::<Local>(&gen_base(0x58b5909), 0x58b5909));
}

/// As [`wave8_local_seed_58b5909_failed_set`], under `Shared`.
#[test]
#[cfg_attr(miri, ignore)]
fn wave8_shared_seed_58b5909_failed_set() {
    quiet(|| differential::<Shared>(&gen_base(0x58b5909), 0x58b5909));
}

/// Op 6 Get(11): 11's read of 14 walks 3, a dep 14's rerun no longer reads,
/// which catches a cycle and lands Dirty; 11 never reads 3, so it lands
/// Clean.
#[test]
#[cfg_attr(miri, ignore)]
fn local_seed_77d_unread_walk_entry() {
    quiet(|| differential::<Local>(&gen_case(0x77d, false, true), 0x77d));
}

/// As [`local_seed_77d_unread_walk_entry`], under `Shared`.
#[test]
#[cfg_attr(miri, ignore)]
fn shared_seed_77d_unread_walk_entry() {
    quiet(|| differential::<Shared>(&gen_case(0x77d, false, true), 0x77d));
}

/// Seed 0x922f4fd7cf058ec: at budget 1, a walk's entry dooms with a
/// repeat pending entry of its node beneath it. The replayed compute must
/// not pop that repeat from the stashed walk: unbounded, the repeat
/// carries a taint that keeps node 8 from running, so op 11 `TryGet(10)`
/// does not land 8 Failed.
#[test]
#[cfg_attr(miri, ignore)]
fn doomed_entrys_replay_keeps_a_repeat_beneath_it() {
    let seed = 0x0922_f4fd_7cf0_58ec;
    quiet(|| {
        differential::<Local>(&gen_case(seed, false, false), seed);
        differential::<Shared>(&gen_case(seed, false, false), seed);
    });
}

/// Seed 0x92ef8cb9, the same rule: op 6 `Get(7)` must leave the unbounded
/// run's Failed set, on one thread and on two.
#[test]
#[cfg_attr(miri, ignore)]
fn doomed_entrys_replay_keeps_a_repeat_beneath_it_92ef8cb9() {
    let seed = 0x92ef_8cb9;
    quiet(|| {
        differential::<Local>(&gen_case(seed, false, false), seed);
        differential::<Shared>(&gen_case(seed, false, false), seed);
        differential_two(&gen_case(seed, false, true), seed);
    });
}

/// Seed 0x3da5e79115b09817: at budget 1, the context's target n4 ends in
/// a user panic that the native reader catches before it is doomed. Its
/// retry must read n4 as that read did, not walk it again after the
/// context has landed it Failed: op 0 `Get(0)` must not add node 8 to
/// the Failed set.
#[test]
#[cfg_attr(miri, ignore)]
fn targets_unwind_into_the_loops_read_is_kept_for_its_retry() {
    let seed = 0x3da5_e791_15b0_9817;
    quiet(|| {
        differential::<Local>(&gen_case(seed, false, false), seed);
        differential::<Shared>(&gen_case(seed, false, false), seed);
        differential_two(&gen_case(seed, false, true), seed);
    });
}

/// Seed 0x57c24cb at budget 1: op 5's `Get(7)` left queries 15 to 18
/// New or Dirty where the unbounded run fails them. A speculative replay
/// of slot 16's frame memoized slot 15 at owner 1 and was doomed; the real
/// run's next read of 15 took that memo and computed 15 without the walk
/// that finds the unbounded run's cycle.
#[test]
#[cfg_attr(miri, ignore)]
fn wave8_local_seed_57c24cb_failed_set() {
    quiet(|| differential::<Local>(&gen_case(0x57c24cb, false, false), 0x57c24cb));
}

/// As [`wave8_local_seed_57c24cb_failed_set`], under `Shared`.
#[test]
#[cfg_attr(miri, ignore)]
fn wave8_shared_seed_57c24cb_failed_set() {
    quiet(|| differential::<Shared>(&gen_case(0x57c24cb, false, false), 0x57c24cb));
}

/// Seed 0xd6e169f0079b0996 as a paired case, cut to its first four ops,
/// at every budget: op 3's `TryGet(15)` landed slot 10 Dirty where the
/// unbounded run's user-panic release lands it New, with slot 6 Failed for
/// Dirty. Either of two rules keeps it green alone: a walked episode's
/// abort counts no held member as released (478dc453), and a speculative
/// run memoizes nothing (50e5e777).
#[test]
#[cfg_attr(miri, ignore)]
fn wave8_seed_d6e1_released_new_walked_fresh() {
    let mut case = gen_case(0xd6e1_69f0_079b_0996, false, true);
    case.ops.truncate(4);
    quiet(|| {
        differential::<Local>(&case, 0xd6e1_69f0_079b_0996);
        differential::<Shared>(&case, 0xd6e1_69f0_079b_0996);
    });
}

/// Seed 0x148c's base case, shrunk. Unbounded, `Get(7)` leaves node 9 New:
/// no run reads it. At budgets 2 and 3 a stand-in's Dirty landing,
/// reported to its walk as `Unsettled`, must not lead the bounded run to
/// compute node 9.
#[test]
#[cfg_attr(miri, ignore)]
fn review_8b_stand_in_unsettled_landing_computes_no_extra_node() {
    use Expr::{Add, Catch, Const, If, PanicIf, Read};
    let b = Box::new;
    let query = |expr, strategy| Spec::Query {
        expr,
        replay: true,
        bare: false,
        strategy,
    };
    let mut specs: Vec<Spec> = (0..18).map(|_| query(Const(0), Strat::None)).collect();
    specs[0] = query(Read(17, false), Strat::None);
    specs[2] = query(Read(16, false), Strat::None);
    specs[3] = query(Read(5, false), Strat::None);
    specs[5] = query(Read(6, false), Strat::None);
    specs[6] = query(Read(7, false), Strat::None);
    specs[7] = query(Read(8, true), Strat::None);
    specs[8] = query(
        Add(
            b(Add(b(Read(12, true)), b(Read(18, false)))),
            b(If(b(Read(2, false)), b(Read(9, false)), b(Const(0)))),
        ),
        Strat::None,
    );
    specs[9] = query(Add(b(Read(6, false)), b(Read(10, true))), Strat::None);
    specs[10] = query(Catch(b(Read(11, false))), Strat::None);
    specs[11] = query(
        If(b(Read(9, true)), b(Read(12, false)), b(Read(12, false))),
        Strat::None,
    );
    specs[12] = query(
        Add(
            b(Read(13, false)),
            b(If(b(Read(17, false)), b(Read(13, false)), b(Const(0)))),
        ),
        Strat::None,
    );
    specs[13] = query(Read(18, false), Strat::None);
    specs[14] = query(Add(b(Read(10, false)), b(Read(15, false))), Strat::None);
    specs[15] = query(Catch(b(Read(14, false))), Strat::None);
    specs[16] = query(Read(17, false), Strat::None);
    specs[17] = query(
        Add(
            b(Read(15, false)),
            b(PanicIf(
                b(If(
                    b(Read(18, false)),
                    b(Read(18, false)),
                    b(Read(5, false)),
                )),
                2,
            )),
        ),
        Strat::Fixpoint {
            initial: 0,
            limit: 3,
        },
    );
    specs.push(Spec::Input(2));
    let case = Case {
        specs,
        panics: true,
        cycles: true,
        queries: 18,
        specs2: vec![],
        queries2: 0,
        ops: vec![Op::Get(0), Op::Get(3), Op::Set(18, 3), Op::Get(7)],
    };
    let states = |t: Trace| -> Vec<(Out, Vec<NodeState>)> {
        t.into_iter().map(|(out, snap)| (out, snap.0)).collect()
    };
    quiet(|| {
        let unbounded = states(run::<Local>(&case, None, 0));
        assert_eq!(unbounded[3].1[9], NodeState::New, "unbounded");
        for frames in [1, 2, 3] {
            let local = states(run::<Local>(&case, Some(frames), 0));
            assert_eq!(local, unbounded, "Local, budget {frames}");
            let shared = states(run::<Shared>(&case, Some(frames), 0));
            assert_eq!(shared, unbounded, "Shared, budget {frames}");
        }
    });
}

/// Seed 0x9ab9's base case, shrunk. Unbounded, `TryGet(12)` reads 0. At
/// budget 1 a speculative run of node 3 (slot 20) writes its slot and is
/// discarded, so `discard_target` takes the value back and lands the node
/// New where the unbounded run leaves it Dirty; the claim releases and
/// walks that follow differ, and the bounded run reports the cycle
/// [5, 7, 10, 11, 17]. Landing it Dirty passes at every budget.
#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "T8-speculative-landings: a discarded speculative write lands its node New"]
fn review_8b_stand_in_unsettled_landing_reads_as_unbounded() {
    use Expr::{Add, Catch, Const, If, Read};
    let b = Box::new;
    let query = |expr, strategy| Spec::Query {
        expr,
        replay: true,
        bare: false,
        strategy,
    };
    let mut specs: Vec<Spec> = (0..23).map(|_| query(Const(0), Strat::None)).collect();
    specs[2] = query(Read(3, false), Strat::Fallback(1));
    specs[3] = Spec::Query {
        expr: Catch(b(If(b(Read(20, false)), b(Read(4, false)), b(Const(0))))),
        replay: false,
        bare: false,
        strategy: Strat::None,
    };
    specs[4] = query(
        Add(
            b(Add(b(Read(5, true)), b(Read(5, true)))),
            b(Catch(b(Read(5, false)))),
        ),
        Strat::Fixpoint {
            initial: 3,
            limit: 2,
        },
    );
    specs[5] = query(
        Add(
            b(Add(
                b(Read(2, false)),
                b(If(b(Read(6, false)), b(Read(7, false)), b(Read(6, false)))),
            )),
            b(Read(10, true)),
        ),
        Strat::None,
    );
    specs[6] = query(
        If(b(Read(7, false)), b(Const(0)), b(Read(12, false))),
        Strat::Fixpoint {
            initial: 3,
            limit: 3,
        },
    );
    specs[7] = query(
        If(b(Read(8, false)), b(Const(0)), b(Read(22, true))),
        Strat::None,
    );
    specs[8] = query(
        If(
            b(Read(3, true)),
            b(Read(9, false)),
            b(Add(b(Read(8, false)), b(Read(9, true)))),
        ),
        Strat::Fallback(3),
    );
    specs[9] = Spec::Query {
        expr: Read(10, false),
        replay: true,
        bare: true,
        strategy: Strat::Fallback(2),
    };
    specs[10] = query(
        If(b(Read(17, false)), b(Read(14, false)), b(Const(0))),
        Strat::None,
    );
    specs[12] = query(Read(13, false), Strat::None);
    specs[13] = query(Read(14, false), Strat::None);
    specs[14] = query(Read(15, false), Strat::None);
    specs[15] = query(
        If(b(Read(16, false)), b(Read(16, false)), b(Const(0))),
        Strat::None,
    );
    specs[16] = query(Read(22, false), Strat::None);
    specs[17] = query(
        Add(
            b(Read(18, false)),
            b(If(b(Read(14, false)), b(Read(9, false)), b(Const(0)))),
        ),
        Strat::None,
    );
    specs[18] = query(
        If(
            b(If(
                b(Read(19, false)),
                b(Read(19, false)),
                b(Read(17, false)),
            )),
            b(Read(21, false)),
            b(Const(0)),
        ),
        Strat::None,
    );
    specs[19] = query(Read(20, true), Strat::Fallback(3));
    specs[20] = query(Read(21, false), Strat::None);
    specs[21] = query(Read(22, false), Strat::None);
    specs[22] = query(
        If(b(Catch(b(Read(3, false)))), b(Const(0)), b(Const(0))),
        Strat::Fallback(0),
    );
    specs.push(Spec::Input(2));
    let case = Case {
        specs,
        panics: false,
        cycles: true,
        queries: 23,
        specs2: vec![],
        queries2: 0,
        ops: vec![Op::TryGet(15), Op::Get(7), Op::TryGet(12)],
    };
    let outs = |t: Trace| t.into_iter().map(|(out, _)| out).collect::<Vec<_>>();
    quiet(|| {
        let unbounded = outs(run::<Local>(&case, None, 0));
        assert_eq!(unbounded[2], Out::Value(0), "unbounded");
        for frames in [1, 2, 3] {
            let local = outs(run::<Local>(&case, Some(frames), 0));
            assert_eq!(local, unbounded, "Local, budget {frames}");
            let shared = outs(run::<Shared>(&case, Some(frames), 0));
            assert_eq!(shared, unbounded, "Shared, budget {frames}");
        }
    });
}

/// `seed`'s operations with every `Get` read four more times with no set
/// between, and the first get whose repeats differ from it, any panic
/// counting as the same outcome: a Failed node's later reads raise
/// "last compute panicked" in place of the first read's payload.
fn first_changed_get<C: Cells>(seed: u64) -> Option<(usize, Vec<Out>)> {
    let mut case = gen_case(seed, false, false);
    case.ops = case
        .ops
        .drain(..)
        .flat_map(|op| {
            let n = if matches!(op, Op::Get(_)) { 5 } else { 1 };
            std::iter::repeat_n(op, n)
        })
        .collect();
    let mut trace = Vec::new();
    quiet(|| trace = run::<C>(&case, None, 0));
    let outcome = |i: usize| match &trace[i].0 {
        Out::Panic(_) => Out::Panic(String::new()),
        o => o.clone(),
    };
    let mut i = 0;
    while i < case.ops.len() {
        if !matches!(case.ops[i], Op::Get(_)) {
            i += 1;
            continue;
        }
        if (i + 1..i + 5).any(|j| outcome(j) != outcome(i)) {
            return Some((i, trace[i..i + 5].iter().map(|(o, _)| o.clone()).collect()));
        }
        i += 5;
    }
    None
}

/// 0x1a7: q7 catches a co-member's cycle panic and sits in two
/// overlapping episodes, so each episode lands Dirty through real poison
/// and the next get plans the other cycle: 9999, 8, 9999, ... 0x4ea
/// alternates the same way, and 0x701 reads 9999, 4, 9999. Keeping a
/// Dirty landing's outcome for its revision (T8-get-alternates-without-set,
/// rule (b)) is beta.7 work.
#[test]
#[ignore = "beta.7: T8-get-alternates-without-set rule (b)"]
fn a_member_catching_a_co_members_cycle_panic_repeats_its_get() {
    for seed in [0x1a7, 0x4ea, 0x701] {
        assert_eq!(first_changed_get::<Local>(seed), None, "{seed:#x}");
        assert_eq!(first_changed_get::<Shared>(seed), None, "{seed:#x}");
    }
}

/// Runs `case`'s phase of ops `pair` on two threads with thread 0's
/// walk held before its compute of query `held`, until thread 1's read
/// has settled `settled` and left `left` as they are (by query), then
/// checks the run against the orders.
fn hold_planned_walk(
    case: &Case,
    seed: u64,
    pair: [usize; 2],
    held: usize,
    settled: &'static [(usize, NodeState)],
) -> Trace {
    let phases = phases(case, seed);
    let overlap = phases.iter().position(|p| p.ops() == pair).unwrap();
    assert_eq!(phases[overlap].of(0), Some(pair[0]));
    let ready = Arc::new(AtomicBool::new(false));
    let rig = |live: &Live<Shared>| {
        let (ready, world) = (ready.clone(), live.world.clone());
        let slot = move |q| world.handle(0, q).erased().slot();
        let world = live.world.clone();
        let hook: crate::runtime::ParkHook = Box::new(move |p| {
            if TWO_PHASE.with(std::cell::Cell::get) != overlap {
                return;
            }
            let state = |q| world.rt(0).nodes.get(slot(q)).state();
            match TWO_THREAD.with(std::cell::Cell::get) {
                Some(0)
                    if p == ParkPoint::BeforeComputeOne(NodeId(slot(held)))
                        && !ready.swap(true, SeqCst) =>
                {
                    until(|| settled.iter().all(|&(q, s)| state(q) == s));
                }
                Some(1) => until(|| ready.load(SeqCst)),
                _ => {}
            }
        });
        assert!(live.world.rt(0).park_hook.set(hook).is_ok());
    };
    let serial = Serial::new(case, &phases);
    let verdict = Mutex::new(None);
    let got = run_two_until(case, &phases, None, 0, &rig, &|p, prefix, live| {
        if p == overlap {
            assert!(
                ready.load(SeqCst),
                "thread 0's walk reached the held compute"
            );
            assert!(
                live.world.rt(0).unrun_unclaims.load(SeqCst) > 0,
                "thread 0's walk re-planned"
            );
        }
        let v = serial.at_overlap("held walk", prefix, live);
        let stop = v.is_some();
        *verdict.lock().unwrap() = v;
        stop
    });
    serial.check("held walk", &got, verdict.into_inner().unwrap());
    got
}

/// 0x7c3a12bcc's ops 8 (`Get(7)`, thread 0) and 9 (`Get(9)`, thread 1)
/// enter one cycle at different members. Thread 0's walk of 7 plans down
/// to 22 and is held there while thread 1 resolves the cycle alone: it
/// lands 7 Clean by 7's fallback, leaves 22 and 23 New, and panics with
/// the cycle, as `Get(9)` does first in a single thread. The order with
/// `Get(9)` first then reads 7 Clean; a walk that goes on with its plan
/// computes 22 again and panics too.
#[test]
#[cfg_attr(miri, ignore)]
fn a_walk_replans_after_another_thread_lands_its_root_by_a_cycle() {
    let seed = 0x7c3a12bcc;
    let case = gen_case(seed, false, true);
    use NodeState::{Clean, Dirty, New};
    quiet(|| {
        let got = hold_planned_walk(
            &case,
            seed,
            [8, 9],
            22,
            &[(7, Clean), (22, New), (23, New), (21, Dirty)],
        );
        assert_eq!(got[9].0, Out::Panic("cycle".into()));
        assert_eq!(got[8].0, Out::Value(2));
    });
}

/// As above with thread 0 reading 18, rewritten to read 7, so the node
/// thread 1 lands Clean is an intermediate of thread 0's plan, not its
/// root, and 18 stays Dirty.
#[test]
#[cfg_attr(miri, ignore)]
fn a_walk_replans_after_another_thread_lands_an_intermediate_by_a_cycle() {
    let seed = 0x7c3a12bcc;
    let mut case = gen_case(seed, false, true);
    case.specs[18] = Spec::Query {
        expr: Expr::Read(7, false),
        replay: true,
        bare: false,
        strategy: Strat::None,
    };
    case.ops[0] = Op::Get(18);
    case.ops[8] = Op::Get(18);
    use NodeState::{Clean, Dirty, New};
    quiet(|| {
        let got = hold_planned_walk(
            &case,
            seed,
            [8, 9],
            22,
            &[(7, Clean), (22, New), (23, New), (21, Dirty), (18, Dirty)],
        );
        assert_eq!(got[9].0, Out::Panic("cycle".into()));
        assert!(matches!(got[8].0, Out::Value(_)), "{:?}", got[8].0);
    });
}

/// 0x7c3a1173e's ops 2 (`Get(4)`, thread 0) and 3 (`Get(11)`, thread 1)
/// enter the cycle 4, 5, 6, 7, 8, 9, 11, 12 at different members. Thread
/// 0 plans its read while 4 is New and is held at its claim until thread
/// 1's read has claimed 4, so it waits that claim out, and thread 1
/// leaves 4 Dirty. The order with `Get(11)` first then reads 4 Dirty and
/// walks its deps before it, landing 6, 7 and 8 Failed, so op 10
/// (`Get(8)`) panics. Computing 4 from the plan made before the wait
/// enters the cycle at 4 instead and leaves them Dirty, and `Get(8)`
/// returned 10006.
#[test]
#[cfg_attr(miri, ignore)]
fn a_read_replans_after_waiting_out_another_threads_claim_of_its_root() {
    let seed = 0x7c3a1173e;
    let case = gen_case(seed, false, true);
    let phases = phases(&case, seed);
    let overlap = phases.iter().position(|p| p.ops() == [2, 3]).unwrap();
    assert_eq!(phases[overlap].of(0), Some(2));
    quiet(|| {
        for _ in 0..8 {
            let ready = Arc::new(AtomicBool::new(false));
            let rig = |live: &Live<Shared>| {
                let ready = ready.clone();
                let world = live.world.clone();
                let slot = move |q| world.handle(0, q).erased().slot();
                let hook: crate::runtime::ParkHook = Box::new(move |p| {
                    if TWO_PHASE.with(std::cell::Cell::get) != overlap {
                        return;
                    }
                    match TWO_THREAD.with(std::cell::Cell::get) {
                        Some(0) if p == ParkPoint::BeforeClaim(NodeId(slot(4))) => {
                            until(|| ready.load(SeqCst));
                        }
                        Some(1)
                            if p == ParkPoint::BeforeClaim(NodeId(slot(5)))
                                && !ready.swap(true, SeqCst) =>
                        {
                            std::thread::sleep(std::time::Duration::from_millis(30));
                        }
                        _ => {}
                    }
                });
                assert!(live.world.rt(0).park_hook.set(hook).is_ok());
            };
            let serial = Serial::new(&case, &phases);
            let verdict = Mutex::new(None);
            let got = run_two_until(&case, &phases, None, 0, &rig, &|_, prefix, live| {
                let v = serial.at_overlap("held claim", prefix, live);
                let stop = v.is_some();
                *verdict.lock().unwrap() = v;
                stop
            });
            assert!(ready.load(SeqCst), "thread 1's read claimed 4");
            serial.check("held claim", &got, verdict.into_inner().unwrap());
            assert_eq!(got[10].0, Out::Panic("failed".into()));
        }
    });
}

/// A differential panic's message from its budget on, joined to one line
/// up to the trace.
fn failure_message(p: Box<dyn std::any::Any + Send>) -> String {
    let m = p
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    let m: Vec<&str> = m.lines().take_while(|l| !l.starts_with("ops:")).collect();
    let m = m.join(" | ");
    let at = m.find("budget ").unwrap_or(0);
    m[at..].to_string()
}

/// Each expression `e` can become by replacing one of its subexpressions
/// with a constant or with one of that subexpression's operands.
fn shrink_expr(e: &Expr) -> Vec<Expr> {
    let mut out = vec![Expr::Const(0), Expr::Const(1)];
    let b = |e: &Expr| Box::new(e.clone());
    match e {
        Expr::Const(_) => out.clear(),
        Expr::Read(n, true) => out.push(Expr::Read(*n, false)),
        Expr::ReadOther(n, true) => out.push(Expr::ReadOther(*n, false)),
        Expr::Read(..) | Expr::ReadOther(..) => {}
        Expr::Add(x, y) => {
            out.extend([(**x).clone(), (**y).clone()]);
            out.extend(
                shrink_expr(x)
                    .into_iter()
                    .map(|x| Expr::Add(Box::new(x), b(y))),
            );
            out.extend(
                shrink_expr(y)
                    .into_iter()
                    .map(|y| Expr::Add(b(x), Box::new(y))),
            );
        }
        Expr::If(c, t, f) => {
            out.extend([(**c).clone(), (**t).clone(), (**f).clone()]);
            out.extend(
                shrink_expr(c)
                    .into_iter()
                    .map(|c| Expr::If(Box::new(c), b(t), b(f))),
            );
            out.extend(
                shrink_expr(t)
                    .into_iter()
                    .map(|t| Expr::If(b(c), Box::new(t), b(f))),
            );
            out.extend(
                shrink_expr(f)
                    .into_iter()
                    .map(|f| Expr::If(b(c), b(t), Box::new(f))),
            );
        }
        Expr::PanicIf(x, k) => {
            out.push((**x).clone());
            out.extend(
                shrink_expr(x)
                    .into_iter()
                    .map(|x| Expr::PanicIf(Box::new(x), *k)),
            );
        }
        Expr::Catch(x) => {
            out.push((**x).clone());
            out.extend(shrink_expr(x).into_iter().map(|x| Expr::Catch(Box::new(x))));
        }
        Expr::FirstRun(x, y) => {
            out.extend([(**x).clone(), (**y).clone()]);
            out.extend(
                shrink_expr(x)
                    .into_iter()
                    .map(|x| Expr::FirstRun(Box::new(x), b(y))),
            );
            out.extend(
                shrink_expr(y)
                    .into_iter()
                    .map(|y| Expr::FirstRun(b(x), Box::new(y))),
            );
        }
    }
    out
}

/// Each case `c` can become by one step: dropping an op, simplifying one
/// query's expression, or clearing its strategy, `bare` or `replay`.
fn shrink_case(c: &Case) -> Vec<Case> {
    let mut out = Vec::new();
    for i in 0..c.ops.len() {
        let mut d = c.clone();
        d.ops.remove(i);
        out.push(d);
    }
    for (i, s) in c.specs.iter().enumerate() {
        let Spec::Query {
            expr,
            replay,
            bare,
            strategy,
        } = s
        else {
            continue;
        };
        let with = |expr: Expr, replay: bool, bare: bool, strategy: Strat| {
            let mut d = c.clone();
            d.specs[i] = Spec::Query {
                expr,
                replay,
                bare,
                strategy,
            };
            d
        };
        if !matches!(strategy, Strat::None) {
            out.push(with(expr.clone(), *replay, *bare, Strat::None));
        }
        if *bare {
            out.push(with(expr.clone(), *replay, false, *strategy));
        }
        if *replay {
            out.push(with(expr.clone(), false, *bare, *strategy));
        }
        for e in shrink_expr(expr) {
            out.push(with(e, *replay, *bare, *strategy));
        }
    }
    out
}

/// Shrinks the single-thread run of `REPLAY_SEED`'s case (chosen as
/// `replay_scan` chooses it) while its run at
/// budget `REPLAY_FRAMES` (0 for none) frames, `REPLAY_BYTES` bytes, under
/// Shared if `REPLAY_SHARED`, still fails with a message containing
/// `SHRINK_NEEDLE`, and prints the case it ends at. Candidates whose runs
/// panic with another message are counted, not kept.
#[test]
#[ignore = "a diagnostic that shrinks one seed's case"]
fn replay_shrink() {
    let seed = u64::from_str_radix(&std::env::var("REPLAY_SEED").expect("REPLAY_SEED"), 16)
        .expect("a hex seed");
    let frames: u32 = std::env::var("REPLAY_FRAMES").map_or(1, |v| v.parse().expect("frames"));
    let frames = (frames > 0).then_some(frames);
    let bytes: usize = std::env::var("REPLAY_BYTES").map_or(0, |v| v.parse().expect("bytes"));
    let needle = std::env::var("SHRINK_NEEDLE").unwrap_or_else(|_| "differs".into());
    let mut case = case_of(seed);
    let fails = |c: &Case| -> Option<bool> {
        let r = catch_unwind(AssertUnwindSafe(|| {
            let label = budget_label(seed, frames, bytes);
            if shared() {
                let u = run::<Shared>(c, None, 0);
                compare(c, &u, &run::<Shared>(c, frames, bytes), &label);
            } else {
                let u = run::<Local>(c, None, 0);
                compare(c, &u, &run::<Local>(c, frames, bytes), &label);
            }
        }));
        match r {
            Ok(()) => Some(false),
            Err(p) => {
                let m = failure_message(p);
                if m.contains(&needle) {
                    Some(true)
                } else if m.contains("differs") {
                    Some(false)
                } else {
                    None
                }
            }
        }
    };
    let size = |c: &Case| format!("{:?}{:?}", c.specs, c.ops).len();
    let (mut tried, mut other) = (0, 0);
    quiet(|| {
        assert_eq!(
            fails(&case),
            Some(true),
            "seed {seed:#x} does not fail as asked"
        );
        loop {
            let before = size(&case);
            let next = shrink_case(&case).into_iter().find(|d| {
                if size(d) >= before {
                    return false;
                }
                tried += 1;
                match fails(d) {
                    Some(f) => f,
                    None => {
                        other += 1;
                        false
                    }
                }
            });
            match next {
                Some(d) => case = d,
                None => break,
            }
        }
    });
    eprintln!(
        "SHRUNK seed {seed:#x}: {tried} candidates run, {other} panicked otherwise, size {}",
        size(&case)
    );
    for (i, s) in case.specs.iter().enumerate() {
        eprintln!("{i}: {s:?}");
    }
    eprintln!("ops: {:?}", case.ops);
    let r = catch_unwind(AssertUnwindSafe(|| {
        let u = run::<Local>(&case, None, 0);
        compare(
            &case,
            &u,
            &run::<Local>(&case, frames, bytes),
            &budget_label(seed, frames, bytes),
        );
    }));
    eprintln!(
        "SHRUNK failure: {}",
        r.err()
            .map(failure_message)
            .unwrap_or_else(|| "none under Local".into())
    );
}

/// Op 9 of `seed` truncated after it, as `(output, Failed set)`, for the
/// unbounded run and for budget 3, under `C`.
fn op9_at_budget_3<C: Cells>(seed: u64) -> [(Out, Vec<usize>); 2] {
    let mut case = gen_case(seed, false, false);
    case.ops.truncate(10);
    let at9 = |t: &Trace| -> (Out, Vec<usize>) {
        let (out, snap) = &t[9];
        let set = (snap.0.iter().enumerate())
            .filter(|(_, s)| **s == NodeState::Failed)
            .map(|(x, _)| x)
            .collect();
        (out.clone(), set)
    };
    [
        at9(&run::<C>(&case, None, 0)),
        at9(&run::<C>(&case, Some(3), 0)),
    ]
}

/// Seed 0x56cdd78's op 9, `Get(8)`: its read of 4 runs 4's cycle episode,
/// which releases 4, 8, 9 and 10 Dirty with no value, so the read walks 4
/// again and the walk contains the cycle at its entries until 10's
/// strategy closes it, reading 0. At a budget of 3 a doom stops the first
/// walk; after the release, the frame reading 4 must walk again rather
/// than take the `Direct` delivery made for the first walk.
#[test]
#[cfg_attr(miri, ignore)]
fn a_released_episodes_reread_walks_instead_of_taking_the_first_walks_direct() {
    let seed = 0x56c_dd78;
    assert!(matches!(gen_case(seed, false, false).ops[9], Op::Get(8)));
    let want = (Out::Value(0), Vec::new());
    quiet(|| {
        let local = op9_at_budget_3::<Local>(seed);
        let shared = op9_at_budget_3::<Shared>(seed);
        assert_eq!([&local[0], &shared[0]], [&want, &want], "unbounded");
        assert_eq!(
            [&local[1], &shared[1]],
            [&want, &want],
            "budget 3, Local then Shared"
        );
    });
}

/// Seed 0x59afe69's op 9, `Get(12)`: the walk of 11 in 12's read runs 11
/// at position 2 twice, first as a walked entry that catches a cycle out
/// of 10, then, past 10's contained failed read, as the walk's root, which
/// reads 10 Failed and fails 11, 12 and 13. At a budget of 3 a later doom
/// of the first run's maker re-arms the cycle delivered for that run's
/// read of 10, which the root run, a later run of 11 there, must not take.
#[test]
#[cfg_attr(miri, ignore)]
fn a_finished_runs_delivery_is_not_read_by_a_later_run_at_its_position() {
    let seed = 0x59a_fe69;
    assert!(matches!(gen_case(seed, false, false).ops[9], Op::Get(12)));
    let want = (Out::Panic("failed".to_owned()), (12..=18).collect());
    quiet(|| {
        let local = op9_at_budget_3::<Local>(seed);
        let shared = op9_at_budget_3::<Shared>(seed);
        assert_eq!([&local[0], &shared[0]], [&want, &want], "unbounded");
        assert_eq!(
            [&local[1], &shared[1]],
            [&want, &want],
            "budget 3, Local then Shared"
        );
    });
}

/// Seed 0x18aea, shrunk by `replay_shrink`. 20 and 4 are fixpoints; 21
/// reads 22, which reads 10, 19, 25 and on one branch 23, and 13 reaches
/// 11, which reads 12
/// (bare, reading 22) and 4. Unbounded, `TryGet(20)` after `Get(4)` lands
/// 11 Failed with 4; at a budget of 3 the bounded run leaves 11 New.
#[test]
#[cfg_attr(miri, ignore)]
fn a_failed_fixpoints_reader_lands_failed_at_every_budget() {
    let r = |n, t| Box::new(Expr::Read(n, t));
    let c = |v| Box::new(Expr::Const(v));
    let q = |expr, replay, bare, strategy| Spec::Query {
        expr,
        replay,
        bare,
        strategy,
    };
    let zero = || replayable(Expr::Const(0), Strat::None);
    let specs = vec![
        zero(),
        zero(),
        zero(),
        zero(),
        replayable(
            Expr::Read(13, false),
            Strat::Fixpoint {
                initial: 0,
                limit: 3,
            },
        ),
        zero(),
        zero(),
        q(Expr::Const(0), true, true, Strat::None),
        replayable(Expr::Read(21, false), Strat::None),
        zero(),
        replayable(
            Expr::Catch(Box::new(Expr::If(
                Box::new(Expr::Add(r(4, true), c(1))),
                c(0),
                Box::new(Expr::If(r(11, true), c(0), r(12, false))),
            ))),
            Strat::None,
        ),
        q(
            Expr::PanicIf(Box::new(Expr::Add(r(12, false), r(4, false))), 3),
            false,
            false,
            Strat::None,
        ),
        q(Expr::Read(22, false), true, true, Strat::None),
        q(
            Expr::If(
                r(8, false),
                Box::new(Expr::If(r(11, false), r(14, false), c(0))),
                c(0),
            ),
            false,
            false,
            Strat::Fallback(1),
        ),
        replayable(Expr::PanicIf(r(15, true), 0), Strat::None),
        replayable(
            Expr::Read(16, true),
            Strat::Fixpoint {
                initial: 0,
                limit: 2,
            },
        ),
        replayable(Expr::Read(17, false), Strat::None),
        q(Expr::Read(18, false), false, false, Strat::None),
        q(Expr::Read(21, false), false, false, Strat::None),
        replayable(Expr::Catch(r(20, false)), Strat::None),
        replayable(
            Expr::Read(21, false),
            Strat::Fixpoint {
                initial: 0,
                limit: 2,
            },
        ),
        replayable(Expr::Read(22, false), Strat::None),
        q(
            Expr::Add(
                Box::new(Expr::If(r(19, false), c(0), r(23, false))),
                Box::new(Expr::Add(r(25, false), r(10, false))),
            ),
            true,
            true,
            Strat::None,
        ),
        Spec::Input(4),
        Spec::Input(3),
        Spec::Input(3),
        Spec::Input(2),
    ];
    let case = Case {
        specs,
        panics: true,
        cycles: true,
        queries: 23,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(4), Op::TryGet(20)],
    };
    quiet(|| {
        differential::<Local>(&case, 0x18aea);
        differential::<Shared>(&case, 0x18aea);
    });
}

/// Base seed 0x675a, shrunk by `replay_shrink`. 4 and 10 form a ring that
/// 4's read of 5 and 5's of 6 join; 6 reads 7, which panics on 8, and 8
/// reaches 10 through 9. Unbounded, `Get(10)` after `TryGet(4)` lands 7
/// Failed with the others; at budget 3 the bounded run leaves 7 Dirty.
#[test]
#[cfg_attr(miri, ignore)]
fn a_failed_ring_member_lands_failed_at_every_budget() {
    let r = |n, t| Box::new(Expr::Read(n, t));
    let c = |v| Box::new(Expr::Const(v));
    let mut specs: Vec<Spec> = (0..4)
        .map(|_| replayable(Expr::Const(0), Strat::None))
        .collect();
    specs.extend([
        replayable(
            Expr::If(r(10, false), c(0), r(5, false)),
            Strat::Fallback(2),
        ),
        replayable(
            Expr::PanicIf(
                Box::new(Expr::Add(
                    Box::new(Expr::Add(r(6, true), c(1))),
                    Box::new(Expr::If(r(5, false), c(0), r(6, false))),
                )),
                3,
            ),
            Strat::Fallback(1),
        ),
        replayable(Expr::If(r(4, false), r(7, false), c(0)), Strat::Fallback(1)),
        Spec::Query {
            expr: Expr::If(
                Box::new(Expr::If(r(8, true), c(0), r(6, false))),
                Box::new(Expr::PanicIf(r(8, false), 2)),
                r(8, false),
            ),
            replay: true,
            bare: true,
            strategy: Strat::None,
        },
        replayable(Expr::Add(c(1), r(9, false)), Strat::None),
        replayable(Expr::Read(10, true), Strat::None),
        Spec::Query {
            expr: Expr::Read(4, false),
            replay: false,
            bare: false,
            strategy: Strat::Fixpoint {
                initial: 1,
                limit: 2,
            },
        },
        Spec::Input(4),
        Spec::Input(3),
        Spec::Input(0),
        Spec::Input(3),
    ]);
    let case = Case {
        specs,
        panics: true,
        cycles: true,
        queries: 11,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::TryGet(4), Op::Get(10)],
    };
    quiet(|| {
        differential::<Local>(&case, 0x675a);
        differential::<Shared>(&case, 0x675a);
    });
}

/// Seed 0x189fb's op 2, `TryGet(20)`, shrunk: 20 roots an episode whose
/// cycle passes through 6, and at a budget of 2 a doom replays it, so the
/// pop of 20's virtual frame at position 0 runs the episode. It dissolves
/// to 20 alone, which then reruns as a new run, and that run's read of 6,
/// released Dirty, walks again as the unbounded run's does. Taking the
/// `Direct` delivery made for the first run's read instead gave `Value(2)`
/// where the unbounded run panics out of 11.
#[test]
#[cfg_attr(miri, ignore)]
fn a_dissolved_roots_rerun_at_the_bottom_walks_its_reads_again() {
    let specs = vec![
        replayable(Expr::Const(0), Strat::None),
        replayable(Expr::Const(0), Strat::None),
        replayable(Expr::Const(0), Strat::None),
        replayable(Expr::Const(0), Strat::None),
        replayable(
            Expr::Read(5, false),
            Strat::Fixpoint {
                initial: 3,
                limit: 3,
            },
        ),
        replayable(Expr::Read(6, false), Strat::None),
        replayable(Expr::Read(7, true), Strat::Fallback(2)),
        replayable(Expr::Read(13, false), Strat::None),
        replayable(Expr::Const(0), Strat::None),
        replayable(
            Expr::If(
                Box::new(Expr::Read(4, true)),
                Box::new(Expr::Read(10, true)),
                Box::new(Expr::Const(0)),
            ),
            Strat::None,
        ),
        replayable(Expr::Read(20, false), Strat::None),
        replayable(
            Expr::PanicIf(
                Box::new(Expr::Add(
                    Box::new(Expr::Read(12, true)),
                    Box::new(Expr::If(
                        Box::new(Expr::Read(12, false)),
                        Box::new(Expr::Read(9, false)),
                        Box::new(Expr::Const(0)),
                    )),
                )),
                0,
            ),
            Strat::None,
        ),
        replayable(Expr::Read(13, false), Strat::None),
        Spec::Query {
            expr: Expr::If(
                Box::new(Expr::Read(11, false)),
                Box::new(Expr::Const(0)),
                Box::new(Expr::Const(0)),
            ),
            replay: false,
            bare: false,
            strategy: Strat::Fixpoint {
                initial: 4,
                limit: 2,
            },
        },
        replayable(Expr::Const(1), Strat::None),
        replayable(Expr::Const(4), Strat::None),
        replayable(Expr::Const(0), Strat::None),
        Spec::Query {
            expr: Expr::Const(0),
            replay: false,
            bare: false,
            strategy: Strat::None,
        },
        replayable(Expr::Const(0), Strat::None),
        replayable(Expr::Read(24, true), Strat::None),
        replayable(
            Expr::If(
                Box::new(Expr::Read(19, false)),
                Box::new(Expr::Const(1)),
                Box::new(Expr::Read(6, false)),
            ),
            Strat::Fixpoint {
                initial: 1,
                limit: 1,
            },
        ),
        Spec::Query {
            expr: Expr::Const(3),
            replay: false,
            bare: true,
            strategy: Strat::None,
        },
        replayable(Expr::Const(0), Strat::None),
        replayable(Expr::Const(0), Strat::None),
        Spec::Input(4),
        Spec::Input(2),
        Spec::Input(3),
        Spec::Input(2),
    ];
    let case = Case {
        specs,
        panics: true,
        cycles: true,
        queries: 24,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(4), Op::Set(24, 3), Op::TryGet(20)],
    };
    quiet(|| {
        differential::<Local>(&case, 0x189fb);
        differential::<Shared>(&case, 0x189fb);
    });
}

/// Seed 0x381 with `FirstRun` on its Native queries, shrunk by
/// `replay_shrink`: at budget 3, query 0 stayed New where the unbounded run
/// computes it, because 18's Clean read of 19 skipped the walk a doom had
/// stopped there.
#[test]
#[cfg_attr(miri, ignore)]
fn seed_381_first_run_state() {
    let specs: Vec<Spec> = vec![
        Spec::Query {
            expr: Expr::Catch(Box::new(Expr::Read(19, false))),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(0),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(0),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(0),
            replay: false,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(3),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(0),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(0),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(0),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(0),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(0),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(0),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(0),
            replay: false,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(0),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(0),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(1),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(0),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(3),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Read(18, false),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Read(19, false),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::If(
                Box::new(Expr::Read(21, true)),
                Box::new(Expr::Const(0)),
                Box::new(Expr::Catch(Box::new(Expr::Read(0, false)))),
            ),
            replay: false,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Const(3),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Read(22, false),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::FirstRun(Box::new(Expr::Read(23, false)), Box::new(Expr::Const(0))),
            replay: false,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Query {
            expr: Expr::Read(24, false),
            replay: true,
            bare: false,
            strategy: Strat::None,
        },
        Spec::Input(3),
        Spec::Input(2),
        Spec::Input(0),
    ];
    let case = Case {
        specs,
        panics: false,
        cycles: true,
        queries: 24,
        specs2: Vec::new(),
        queries2: 0,
        ops: vec![Op::Get(0), Op::Set(24, 1), Op::TryGet(17)],
    };
    quiet(|| {
        differential::<Local>(&case, 0x381);
        differential::<Shared>(&case, 0x381);
    });
}
