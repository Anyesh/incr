//! `replay_differential`: random graphs of replayable queries, each run at
//! frame budgets 1, 2 and 3, with and without a byte budget, against the
//! same graph and operations run unbounded. Every `get` and `try_get`
//! result (the value, a cycle's participants, or the panic's class or
//! payload) and the sets of Failed and Cyclic nodes after each operation
//! must be the unbounded run's, while every query's state agrees with the
//! unbounded run's before the operation.
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
use std::sync::{Arc, OnceLock};

use proptest::prelude::*;

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{CycleStrategy, Fixpoint, QueryOpts, Replay};
use crate::handle::Incr;
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
    Add(Box<Expr>, Box<Expr>),
    /// The second operand when the first is even, else the third.
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    /// Panics with the node's payload when the operand is `.1` modulo 7.
    PanicIf(Box<Expr>, u64),
    /// The operand, or 9999 if it unwinds.
    Catch(Box<Expr>),
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
/// inputs, and the inputs follow them.
struct Case {
    specs: Vec<Spec>,
    panics: bool,
    cycles: bool,
    queries: usize,
    ops: Vec<Op>,
}

fn gen_expr(
    rng: &mut Rng,
    at: usize,
    queries: usize,
    nodes: usize,
    depth: u32,
    cycles: bool,
) -> Expr {
    let read = |rng: &mut Rng| {
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
    };
    if depth == 0 {
        return read(rng);
    }
    match rng.below(100) {
        0..=7 => Expr::Const(rng.below(5)),
        8..=44 => read(rng),
        45..=64 => Expr::Add(
            Box::new(gen_expr(rng, at, queries, nodes, depth - 1, cycles)),
            Box::new(gen_expr(rng, at, queries, nodes, depth - 1, cycles)),
        ),
        65..=84 => Expr::If(
            Box::new(gen_expr(rng, at, queries, nodes, depth - 1, cycles)),
            Box::new(gen_expr(rng, at, queries, nodes, depth - 1, cycles)),
            Box::new(gen_expr(rng, at, queries, nodes, depth - 1, cycles)),
        ),
        85..=91 => Expr::PanicIf(
            Box::new(gen_expr(rng, at, queries, nodes, depth - 1, cycles)),
            rng.below(7),
        ),
        _ => Expr::Catch(Box::new(gen_expr(
            rng,
            at,
            queries,
            nodes,
            depth - 1,
            cycles,
        ))),
    }
}

fn gen_case(seed: u64) -> Case {
    let mut rng = Rng(seed);
    let cycles = (seed / 3) % 2 == 1;
    let queries = 2 + rng.below(24) as usize;
    let inputs = 1 + rng.below(4) as usize;
    let nodes = queries + inputs;
    let mut specs = Vec::with_capacity(nodes);
    for at in 0..queries {
        let mut expr = gen_expr(&mut rng, at, queries, nodes, 3, cycles);
        if seed.is_multiple_of(3) {
            expr = without_panics(expr);
        }
        let replay = rng.chance(85);
        let bare = rng.chance(12);
        let strategy = if !cycles {
            Strat::None
        } else {
            match rng.below(4) {
                0 | 1 => Strat::None,
                2 => Strat::Fallback(rng.below(5)),
                _ => Strat::Fixpoint {
                    initial: rng.below(5),
                    limit: 1 + rng.below(3) as u32,
                },
            }
        };
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
        ops,
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
    match e {
        Expr::PanicIf(inner, _) => without_panics(*inner),
        Expr::Add(a, b) => Expr::Add(Box::new(without_panics(*a)), Box::new(without_panics(*b))),
        Expr::If(c, a, b) => Expr::If(
            Box::new(without_panics(*c)),
            Box::new(without_panics(*a)),
            Box::new(without_panics(*b)),
        ),
        Expr::Catch(inner) => Expr::Catch(Box::new(without_panics(*inner))),
        leaf => leaf,
    }
}

#[derive(Copy, Clone, Debug)]
enum Handle {
    Plain(Incr<u64>),
    Bare(Incr<Bare>),
}

type Cells_ = Arc<Vec<OnceLock<Handle>>>;

fn eval<C: Cells>(rt: &Runtime<C>, at: usize, cells: &Cells_, e: &Expr) -> u64 {
    match e {
        Expr::Const(v) => *v,
        Expr::Read(target, try_) => match (cells[*target].get().expect("built"), try_) {
            (Handle::Plain(h), false) => rt.get(*h),
            (Handle::Plain(h), true) => rt.try_get(*h).unwrap_or(7777),
            (Handle::Bare(h), false) => rt.get(*h).0,
            (Handle::Bare(h), true) => rt.try_get(*h).map_or(7777, |b| b.0),
        },
        Expr::Add(a, b) => eval(rt, at, cells, a).wrapping_add(eval(rt, at, cells, b)),
        Expr::If(c, a, b) => {
            if eval(rt, at, cells, c) % 2 == 0 {
                eval(rt, at, cells, a)
            } else {
                eval(rt, at, cells, b)
            }
        }
        Expr::PanicIf(c, when) => {
            let v = eval(rt, at, cells, c);
            if v % 7 == *when {
                std::panic::panic_any(format!("boom {at}"));
            }
            v
        }
        Expr::Catch(inner) => {
            catch_unwind(AssertUnwindSafe(|| eval(rt, at, cells, inner))).unwrap_or(9999)
        }
    }
}

fn build<C: Cells>(rt: &Runtime<C>, case: &Case) -> Vec<Handle> {
    let cells: Cells_ = Arc::new((0..case.specs.len()).map(|_| OnceLock::new()).collect());
    for (i, spec) in case.specs.iter().enumerate().rev() {
        let handle = match spec {
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
                let (expr, cells) = (expr.clone(), cells.clone());
                if *bare {
                    Handle::Bare(rt.create_query_with(
                        move |rt| Bare(eval(rt, i, &cells, &expr)),
                        QueryOpts {
                            cycle: strategy.build(Bare),
                            replay,
                        },
                    ))
                } else {
                    Handle::Plain(rt.create_query_with(
                        move |rt| eval(rt, i, &cells, &expr),
                        QueryOpts {
                            cycle: strategy.build(|v| v),
                            replay,
                        },
                    ))
                }
            }
        };
        cells[i].set(handle).expect("set once");
    }
    cells.iter().map(|c| *c.get().expect("built")).collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Out {
    Value(u64),
    Cycle(Vec<u32>),
    Panic(String),
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
        ("stale", "stale"),
        ("own compute", "own-compute"),
    ] {
        if message.contains(needle) {
            return class.to_owned();
        }
    }
    message
}

fn apply<C: Cells>(rt: &Runtime<C>, handles: &[Handle], op: &Op) -> Out {
    match *op {
        Op::Set(i, v) => {
            let Handle::Plain(h) = handles[i] else {
                unreachable!("inputs are plain")
            };
            rt.set(h, v);
            Out::Done
        }
        Op::Get(i) => match catch_unwind(AssertUnwindSafe(|| match handles[i] {
            Handle::Plain(h) => rt.get(h),
            Handle::Bare(h) => rt.get(h).0,
        })) {
            Ok(v) => Out::Value(v),
            Err(p) => Out::Panic(class(p)),
        },
        Op::TryGet(i) => match catch_unwind(AssertUnwindSafe(|| match handles[i] {
            Handle::Plain(h) => rt.try_get(h),
            Handle::Bare(h) => rt.try_get(h).map(|b| b.0),
        })) {
            Ok(Ok(v)) => Out::Value(v),
            Ok(Err(c)) => Out::Cycle(c.participants().iter().map(|p| p.slot()).collect()),
            Err(p) => Out::Panic(class(p)),
        },
    }
}

fn states<C: Cells>(rt: &Runtime<C>, handles: &[Handle], queries: usize) -> Vec<NodeState> {
    handles[..queries]
        .iter()
        .map(|h| match *h {
            Handle::Plain(h) => rt.state_of(h),
            Handle::Bare(h) => rt.state_of(h),
        })
        .collect()
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

/// Every node's dependencies, in the order its last landing installed
/// them.
fn deps<C: Cells>(rt: &Runtime<C>) -> Vec<Vec<u32>> {
    (0..rt.nodes.len())
        .map(|x| {
            let mut d = Vec::new();
            rt.nodes.get(x).for_each_dep(|n| d.push(n.0));
            d
        })
        .collect()
}

/// A run's outcomes, every query's state and every node's dependencies,
/// operation by operation, and whether a read returned a node's Dirty
/// value, or had a delivery, that a later landing made Clean.
type Trace = Vec<(Out, Vec<NodeState>, Vec<Vec<u32>>, bool)>;

fn run<C: Cells>(case: &Case, frames: Option<u32>, bytes: usize) -> Trace {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(frames.unwrap_or(u32::MAX));
    rt.set_max_stack_bytes(bytes);
    let handles = build(&rt, case);
    hook();
    INTERNAL.with(|i| i.borrow_mut().take());
    let mut trace = Vec::new();
    for op in &case.ops {
        let out = apply(&rt, &handles, op);
        rt.assert_invariant_i();
        if let Some(internal) = INTERNAL.with(|i| i.borrow_mut().take()) {
            panic!("operation {op:?} raised an engine panic: {internal}");
        }
        let replaced = std::mem::take(&mut *rt.dirty_reads.lock().unwrap())
            .into_iter()
            .any(|n| rt.nodes.get(n).state() == NodeState::Clean);
        trace.push((
            out,
            states(&rt, &handles, case.queries),
            deps(&rt),
            replaced,
        ));
    }
    trace
}

fn differential<C: Cells>(seed: u64) {
    let case = gen_case(seed);
    let unbounded = run::<C>(&case, None, 0);
    for frames in [1, 2, 3] {
        for bytes in [0, 16 * 1024] {
            let bounded = run::<C>(&case, Some(frames), bytes);
            let mut agree = true;
            for (i, (want, got)) in unbounded.iter().zip(&bounded).enumerate() {
                // Once a speculative landing or a delivery has left the
                // states apart, a panic or a cycle may come out otherwise: a
                // walk of a Dirty node computes its deps before its closure,
                // so it enters a cycle, and so picks its root and strategy,
                // where a compute of the node New would not.
                if !agree && (case.panics || case.cycles) {
                    break;
                }
                // A read that lands its target Dirty returns a value computed
                // against a read that a later landing in the same revision
                // changed. A replayed frame reads after that landing when the
                // read is a Clean `get`, which no delivery can intercept, so
                // the bounded run may return the later value.
                if let Op::Get(q) | Op::TryGet(q) = case.ops[i] {
                    if want.1[q] == NodeState::Dirty {
                        agree = false;
                        continue;
                    }
                }
                // So may a frame that read a node's Dirty value, or took its
                // unwind or a delivery for its read, which a later landing
                // made Clean: its next retry reads it with a Clean `get`.
                if want.3 || got.3 {
                    agree = false;
                    continue;
                }
                let compared = |(out, states, ..): &(Out, Vec<NodeState>, Vec<Vec<u32>>, bool)| {
                    (out.clone(), agree.then(|| settled(states)))
                };
                assert_eq!(
                    compared(want),
                    compared(got),
                    "seed {seed:#x}, budget {frames} frames, {bytes} bytes: operation {i} \
                     ({:?}) differs from the unbounded run\ncase: {:#?}",
                    case.ops[i],
                    case.specs
                );
                // A dep list is history too: under a panic a read can stop
                // the closure short of the reads after it.
                agree &= want.1 == got.1 && want.2 == got.2;
            }
        }
    }
}

/// Seeds that once failed, each with what it caught.
#[test]
fn regressions() {
    // A closure caught the panic of its read of a Failed member in a walk
    // the Failed node's read ran, and landed Clean over it.
    let case = gen_case(0x112);
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
        let case = gen_case(seed);
        quiet(|| {
            run::<Local>(&case, None, 0);
            run::<Shared>(&case, None, 0);
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
        0x00e8_d4a5_2919,
        0x00e8_d4a5_2db3,
        0x00e8_d4a5_33fa,
        0x551d_e2c5_404a_84b1,
        0x84a1_d45b_e6f9_b165,
        0xa649_6004_aa21_dd3b,
        0xa8cc_85b9_bbfd_657d,
        0xe386_b8af_df64_ca89,
        0xf9e8_e4b4_cddf_4529,
    ] {
        quiet(|| {
            differential::<Local>(seed);
            differential::<Shared>(seed);
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
        quiet(|| differential::<Local>(seed));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn replay_differential_shared(seed in any::<u64>()) {
        quiet(|| differential::<Shared>(seed));
    }
}

fn shared() -> bool {
    std::env::var_os("REPLAY_SHARED").is_some()
}

/// One case by its seed (`REPLAY_SEED`, hex), printing both runs' traces,
/// under Shared if `REPLAY_SHARED` is set.
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
    let case = gen_case(seed);
    for (i, s) in case.specs.iter().enumerate() {
        eprintln!("{i}: {s:?}");
    }
    eprintln!("ops: {:?}", case.ops);
    quiet(|| {
        let unbounded = run::<C>(&case, None, 0);
        eprintln!("unbounded: {unbounded:?}");
        let bytes: usize = std::env::var("REPLAY_BYTES").map_or(0, |v| v.parse().expect("bytes"));
        let bounded = catch_unwind(AssertUnwindSafe(|| run::<C>(&case, Some(frames), bytes)));
        eprintln!("budget {frames}: {bounded:?}");
        if std::env::var_os("REPLAY_FULL").is_some() {
            differential::<Local>(seed);
            differential::<Shared>(seed);
        }
    });
}

fn size(e: &Expr) -> usize {
    match e {
        Expr::Const(_) | Expr::Read(..) => 1,
        Expr::Add(a, b) => 1 + size(a) + size(b),
        Expr::If(a, b, c) => 1 + size(a) + size(b) + size(c),
        Expr::PanicIf(a, _) | Expr::Catch(a) => 1 + size(a),
    }
}

/// The smallest failing case among seeds `0..REPLAY_SCAN`, by total
/// expression size and operation count, under Shared if `REPLAY_SHARED`
/// is set.
#[test]
#[ignore = "a diagnostic that scans seeds"]
fn replay_scan() {
    let n: u64 = std::env::var("REPLAY_SCAN").map_or(2000, |v| v.parse().expect("a count"));
    let mut best: Option<(usize, u64)> = None;
    let mut failing: Vec<u64> = Vec::new();
    if let Ok(seed) = std::env::var("REPLAY_CHECK") {
        let seed = u64::from_str_radix(&seed, 16).expect("a hex seed");
        if shared() {
            differential::<Shared>(seed);
        } else {
            differential::<Local>(seed);
        }
        return;
    }
    quiet(|| {
        let from: u64 = std::env::var("REPLAY_FROM").map_or(0, |v| v.parse().expect("a seed"));
        for seed in from..n {
            let case = gen_case(seed);
            let weight: usize = case
                .specs
                .iter()
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
            let failed = if shared() {
                catch_unwind(AssertUnwindSafe(|| differential::<Shared>(seed))).is_err()
            } else {
                catch_unwind(AssertUnwindSafe(|| differential::<Local>(seed))).is_err()
            };
            if failed {
                if all {
                    eprintln!("FAIL {seed:x}");
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
