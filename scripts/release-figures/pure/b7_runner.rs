//! Runs exported differential cases through incr-core's public API only, so
//! the same case runs on v0.2.0-beta.5, main and spreadsheet-t10.
//!
//! `B7_CASE=<name>` picks a case (all by default), `B7_MODE` is `local`,
//! `shared` or `two`, `B7_RUNS` the two-thread run count. Single-thread
//! modes print every op's outcome and, for each read, the outcome of the
//! same read on a fresh world that only had the mutations before it.

#![allow(non_snake_case, dead_code)]

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Barrier, Mutex, RwLock};
use std::time::Duration;

use incr_core::{Cells, Incr, Local, Runtime, Shared};

include!("b7/api.rs");
include!("b7/cases.rs");

#[derive(Clone, Debug)]
pub enum E {
    Const(u64),
    Read(usize, bool),
    ReadOther(usize, bool),
    Add(Box<E>, Box<E>),
    If(Box<E>, Box<E>, Box<E>),
    PanicIf(Box<E>, u64),
    Catch(Box<E>),
    FirstRun(Box<E>, Box<E>),
    FirstLanded(Box<E>, Box<E>),
}

#[derive(Clone, Copy, Debug)]
pub enum Strat {
    None,
    Fallback(u64),
    Fixpoint { initial: u64, limit: u32 },
}

#[derive(Clone, Debug)]
pub enum Spec {
    Input(u64),
    Query(Query),
}

#[derive(Clone, Debug)]
pub struct Query {
    pub expr: E,
    pub replay: bool,
    pub bare: bool,
    pub strategy: Strat,
}

#[derive(Clone, Debug)]
pub enum Op {
    Set(usize, u64),
    Get(usize),
    TryGet(usize),
    Delete(usize),
    Recreate(usize, Spec),
    ReadOld(usize, bool),
    Set2(usize, u64),
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

pub enum Phase {
    M(usize, usize),
    R([Option<usize>; 2]),
}

pub struct Case {
    pub name: &'static str,
    pub specs: Vec<Spec>,
    pub specs2: Vec<Spec>,
    pub ops: Vec<Op>,
    pub phases: Vec<Phase>,
    pub frames: Option<u32>,
}

/// The names exported `{:?}` text uses, so a case file is the harness's
/// Debug output with each spec wrapped in `s(..)`.
mod gen {
    pub use super::Strat::{Fallback, Fixpoint, None};
    #[allow(unused_imports)]
    pub use super::{Case, Phase::M, Phase::R, Query};
    use super::{Op, Spec, E};

    pub fn s(x: impl Into<Spec>) -> Spec {
        x.into()
    }
    impl From<Query> for Spec {
        fn from(q: Query) -> Spec {
            Spec::Query(q)
        }
    }
    fn b(e: E) -> Box<E> {
        Box::new(e)
    }
    pub fn Input(v: u64) -> Spec {
        Spec::Input(v)
    }
    pub fn Const(v: u64) -> E {
        E::Const(v)
    }
    pub fn Read(x: usize, t: bool) -> E {
        E::Read(x, t)
    }
    pub fn ReadOther(x: usize, t: bool) -> E {
        E::ReadOther(x, t)
    }
    pub fn Add(x: E, y: E) -> E {
        E::Add(b(x), b(y))
    }
    pub fn If(c: E, x: E, y: E) -> E {
        E::If(b(c), b(x), b(y))
    }
    pub fn PanicIf(x: E, w: u64) -> E {
        E::PanicIf(b(x), w)
    }
    pub fn Catch(x: E) -> E {
        E::Catch(b(x))
    }
    pub fn FirstRun(x: E, y: E) -> E {
        E::FirstRun(b(x), b(y))
    }
    pub fn FirstLanded(x: E, y: E) -> E {
        E::FirstLanded(b(x), b(y))
    }
    pub fn Set(i: usize, v: u64) -> Op {
        Op::Set(i, v)
    }
    pub fn Get(i: usize) -> Op {
        Op::Get(i)
    }
    pub fn TryGet(i: usize) -> Op {
        Op::TryGet(i)
    }
    pub fn Delete(i: usize) -> Op {
        Op::Delete(i)
    }
    pub fn Recreate(i: usize, s: impl Into<Spec>) -> Op {
        Op::Recreate(i, s.into())
    }
    pub fn ReadOld(k: usize, t: bool) -> Op {
        Op::ReadOld(k, t)
    }
    pub fn Set2(i: usize, v: u64) -> Op {
        Op::Set2(i, v)
    }
    pub fn Get2(i: usize, t: bool) -> Op {
        Op::Get2(i, t)
    }
}

struct Tables {
    handles: [RwLock<Vec<Incr<u64>>>; 2],
    retired: Mutex<Vec<Incr<u64>>>,
}

/// Both runtimes outlive every closure, which reaches the other side's
/// runtime through its address because `Runtime<Local>` is not `Sync`.
struct World<C: Cells> {
    rts: [Box<Runtime<C>>; 2],
    tables: Arc<Tables>,
}

struct Cx<'a, C: Cells> {
    rt: &'a Runtime<C>,
    other: &'a Runtime<C>,
    tables: &'a Tables,
    side: usize,
    at: usize,
    first: bool,
}

fn eval<C: Cells>(cx: &Cx<'_, C>, e: &E) -> u64 {
    let read = |rt: &Runtime<C>, side: usize, x: usize, t: bool| {
        let h = cx.tables.handles[side].read().unwrap()[x];
        if t && !projected() {
            try_read(rt, h).unwrap_or(7777)
        } else {
            rt.get(h)
        }
    };
    match e {
        E::Const(v) => *v,
        E::Read(x, t) => read(cx.rt, cx.side, *x, *t),
        E::ReadOther(x, t) => read(cx.other, 1 - cx.side, *x, *t),
        E::Add(a, b) => eval(cx, a).wrapping_add(eval(cx, b)),
        E::If(c, a, b) => {
            if eval(cx, c) % 2 == 0 {
                eval(cx, a)
            } else {
                eval(cx, b)
            }
        }
        E::PanicIf(a, w) => {
            let v = eval(cx, a);
            if v % 7 == *w {
                std::panic::panic_any(format!("boom {}:{}", cx.side, cx.at));
            }
            v
        }
        E::Catch(a) => catch_unwind(AssertUnwindSafe(|| eval(cx, a))).unwrap_or(9999),
        E::FirstRun(a, b) => eval(cx, if cx.first { a } else { b }),
        E::FirstLanded(..) => panic!("B7: FirstLanded needs a crate-private hook"),
    }
}

fn make<C: Cells>(world: &World<C>, side: usize, at: usize, spec: &Spec) -> Incr<u64> {
    let rt = &*world.rts[side];
    match spec {
        Spec::Input(v) => rt.create_input(*v),
        Spec::Query(q) => {
            let tables = world.tables.clone();
            let other = &*world.rts[1 - side] as *const Runtime<C> as usize;
            let expr = q.expr.clone();
            let runs = AtomicU32::new(0);
            let f = move |rt: &Runtime<C>| {
                // SAFETY: World keeps both runtimes boxed until after every read.
                let other = unsafe { &*(other as *const Runtime<C>) };
                let first = runs.fetch_add(1, Relaxed) == 0 || projected();
                let cx = Cx {
                    rt,
                    other,
                    tables: &tables,
                    side,
                    at,
                    first,
                };
                eval(&cx, &expr)
            };
            let strategy = if projected() { Strat::None } else { q.strategy };
            query(rt, f, strategy, q.replay)
        }
    }
}

fn world<C: Cells>(case: &Case) -> World<C> {
    let rts = [(); 2].map(|()| {
        let rt = Box::new(Runtime::<C>::new());
        configure(&rt, case.frames);
        rt
    });
    let tables = Arc::new(Tables {
        handles: [RwLock::new(Vec::new()), RwLock::new(Vec::new())],
        retired: Mutex::new(Vec::new()),
    });
    let w = World { rts, tables };
    for (side, specs) in [&case.specs, &case.specs2].into_iter().enumerate() {
        // In the harness's order, so slots match its runs.
        let mut hs: Vec<Incr<u64>> = (0..specs.len())
            .rev()
            .map(|i| make(&w, side, i, &specs[i]))
            .collect();
        hs.reverse();
        *w.tables.handles[side].write().unwrap() = hs;
    }
    w
}

fn class(p: Box<dyn std::any::Any + Send>) -> String {
    let m = (p.downcast_ref::<String>().cloned())
        .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default();
    for (needle, class) in [
        ("dependency cycle", "cycle"),
        ("last compute panicked", "failed"),
        ("stale Incr", "stale"),
        ("own compute", "own-compute"),
    ] {
        if m.contains(needle) {
            return class.to_owned();
        }
    }
    m.lines().next().unwrap_or("").chars().take(80).collect()
}

fn read_out<C: Cells>(rt: &Runtime<C>, h: Incr<u64>, t: bool) -> String {
    match catch_unwind(AssertUnwindSafe(|| {
        if t && !projected() {
            try_read(rt, h)
        } else {
            Some(rt.get(h))
        }
    })) {
        Ok(Some(v)) => format!("Value({v})"),
        Ok(None) => "Cycle".to_owned(),
        Err(p) => format!("Panic({:?})", class(p)),
    }
}

fn apply<C: Cells>(w: &World<C>, op: &Op) -> String {
    let h = |side: usize, i: usize| w.tables.handles[side].read().unwrap()[i];
    let set = |side: usize, i: usize, v: u64| match catch_unwind(AssertUnwindSafe(|| {
        w.rts[side].set(h(side, i), v)
    })) {
        Ok(()) => "Done".to_owned(),
        Err(p) => format!("Panic({:?})", class(p)),
    };
    match op {
        Op::Set(i, v) => set(0, *i, *v),
        Op::Set2(i, v) => set(1, *i, *v),
        Op::Get(i) => read_out(&w.rts[0], h(0, *i), false),
        Op::TryGet(i) => read_out(&w.rts[0], h(0, *i), true),
        Op::Get2(i, t) => read_out(&w.rts[1], h(1, *i), *t),
        Op::Delete(i) => match w.rts[0].try_delete_node(h(0, *i)) {
            Ok(()) => "Deleted(Ok)".to_owned(),
            Err(e) => format!("Deleted(Err({e:?}))"),
        },
        Op::Recreate(i, spec) => {
            let old = h(0, *i);
            if w.rts[0].is_alive(old) {
                return "Done".to_owned();
            }
            let new = make(w, 0, *i, spec);
            w.tables.handles[0].write().unwrap()[*i] = new;
            w.tables.retired.lock().unwrap().push(old);
            "Recreated".to_owned()
        }
        Op::ReadOld(k, t) => {
            let old = {
                let r = w.tables.retired.lock().unwrap();
                (!r.is_empty()).then(|| r[k % r.len()])
            };
            old.map_or("Done".to_owned(), |h| read_out(&w.rts[0], h, *t))
        }
    }
}

/// The case as beta.5 can run it, with pure closures: every cycle strategy
/// dropped, every `try_get` a `get`, and every `FirstRun` its first branch.
/// On by `B7_PROJECT=1`, and always on beta.5.
fn projected() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| API == "beta5" || std::env::var("B7_PROJECT").is_ok_and(|v| v == "1"))
}

fn quiet() {
    std::panic::set_hook(Box::new(|_| {}));
}

fn single<C: Cells>(case: &Case, mode: &str) {
    let w = world::<C>(case);
    let mut outs: Vec<String> = Vec::new();
    for (k, op) in case.ops.iter().enumerate() {
        outs.push(apply(&w, op));
        println!("{}\t{mode}\t{k}\t{op:?}\t{}", case.name, outs[k]);
        if op.is_read() {
            // A fresh world records no reads, so it replays only the deletes
            // and recreates that took effect in the live run.
            let f = world::<C>(case);
            for (prev, out) in case.ops[..k].iter().zip(&outs) {
                let took = match prev {
                    Op::Delete(_) => out == "Deleted(Ok)",
                    Op::Recreate(..) => out == "Recreated",
                    _ => !prev.is_read(),
                };
                if took {
                    apply(&f, prev);
                }
            }
            println!("{}\t{mode}-fresh\t{k}\t{op:?}\t{}", case.name, apply(&f, op));
        }
    }
}

/// The read `B7_READ` after a fresh world replays the live mutations, run
/// after each `Get(X)`: which entry orders give the live outcome.
fn entry<C: Cells>(case: &Case, mode: &str) {
    let k: usize = std::env::var("B7_READ").expect("B7_READ").parse().expect("an op index");
    let w = world::<C>(case);
    let outs: Vec<String> = case.ops[..=k].iter().map(|op| apply(&w, op)).collect();
    let fresh = |first: Option<usize>| {
        let f = world::<C>(case);
        for (prev, out) in case.ops[..k].iter().zip(&outs) {
            let took = match prev {
                Op::Delete(_) => out == "Deleted(Ok)",
                Op::Recreate(..) => out == "Recreated",
                _ => !prev.is_read(),
            };
            if took {
                apply(&f, prev);
            }
        }
        if let Some(x) = first {
            apply(&f, &Op::Get(x));
        }
        apply(&f, &case.ops[k])
    };
    println!("{}\t{mode}-entry\t{k}\tlive\t{}", case.name, outs[k]);
    println!("{}\t{mode}-entry\t{k}\tfresh\t{}", case.name, fresh(None));
    for x in 0..case.specs.len() {
        if matches!(case.specs[x], Spec::Query(_)) {
            println!("{}\t{mode}-entry\t{k}\tGet({x})\t{}", case.name, fresh(Some(x)));
        }
    }
}

fn spec_src(s: &Spec) -> String {
    match s {
        Spec::Input(v) => format!("Input({v})"),
        Spec::Query(q) => format!(
            "Query {{ expr: {:?}, replay: {}, bare: {}, strategy: {:?} }}",
            q.expr, q.replay, q.bare, q.strategy
        ),
    }
}

fn op_src(o: &Op) -> String {
    match o {
        Op::Recreate(i, s) => format!("Recreate({i}, {})", spec_src(s)),
        o => format!("{o:?}"),
    }
}

fn case_src(c: &Case) -> String {
    let specs = |v: &[Spec]| v.iter().map(|x| format!("s({})", spec_src(x))).collect::<Vec<_>>().join(", ");
    format!(
        "Case {{ name: \"{}-shrunk\", specs: vec![{}], specs2: vec![{}], ops: vec![{}], phases: vec![], frames: Option::None }},",
        c.name,
        specs(&c.specs),
        specs(&c.specs2),
        c.ops.iter().map(op_src).collect::<Vec<_>>().join(", ")
    )
}

/// The last op's live outcome and fresh outcome, Local.
fn last_read(case: &Case) -> (String, String) {
    let w = world::<Local>(case);
    let mut outs = Vec::new();
    for op in &case.ops {
        outs.push(apply(&w, op));
    }
    let f = world::<Local>(case);
    let k = case.ops.len() - 1;
    for (prev, out) in case.ops[..k].iter().zip(&outs) {
        let took = match prev {
            Op::Delete(_) => out == "Deleted(Ok)",
            Op::Recreate(..) => out == "Recreated",
            _ => !prev.is_read(),
        };
        if took {
            apply(&f, prev);
        }
    }
    (outs[k].clone(), apply(&f, &case.ops[k]))
}

fn simpler(e: &E) -> Vec<E> {
    let b = |x: &E| Box::new(x.clone());
    let mut v = vec![E::Const(0)];
    match e {
        E::Const(_) => v.clear(),
        E::Read(..) | E::ReadOther(..) => {}
        E::Add(x, y) => {
            v.extend([(**x).clone(), (**y).clone()]);
            v.extend(simpler(x).into_iter().map(|s| E::Add(Box::new(s), b(y))));
            v.extend(simpler(y).into_iter().map(|s| E::Add(b(x), Box::new(s))));
        }
        E::If(c, x, y) => {
            v.extend([(**x).clone(), (**y).clone()]);
            v.extend(simpler(c).into_iter().map(|s| E::If(Box::new(s), b(x), b(y))));
            v.extend(simpler(x).into_iter().map(|s| E::If(b(c), Box::new(s), b(y))));
            v.extend(simpler(y).into_iter().map(|s| E::If(b(c), b(x), Box::new(s))));
        }
        E::PanicIf(x, w) => {
            v.push((**x).clone());
            v.extend(simpler(x).into_iter().map(|s| E::PanicIf(Box::new(s), *w)));
        }
        E::Catch(x) => {
            v.push((**x).clone());
            v.extend(simpler(x).into_iter().map(|s| E::Catch(Box::new(s))));
        }
        E::FirstRun(x, y) | E::FirstLanded(x, y) => v.extend([(**x).clone(), (**y).clone()]),
    }
    v
}

/// Shrinks case `B7_CASE` so its op `B7_READ` stays a value where a fresh
/// read gives the cycle, then prints the case source.
fn shrink(case: Case) {
    let k: usize = std::env::var("B7_READ").expect("B7_READ").parse().expect("an op index");
    let holds = |c: &Case| {
        let (live, fresh) = last_read(c);
        live.starts_with("Value(") && fresh == "Panic(\"cycle\")"
    };
    let mut c = case;
    c.ops.truncate(k + 1);
    assert!(holds(&c), "the read does not hold to start with: {:?}", last_read(&c));
    loop {
        let mut changed = false;
        let mut i = 0;
        while i + 1 < c.ops.len() {
            let op = c.ops.remove(i);
            if holds(&c) {
                changed = true;
            } else {
                c.ops.insert(i, op);
                i += 1;
            }
        }
        for x in 0..c.specs.len() {
            let Spec::Query(q) = c.specs[x].clone() else { continue };
            for cand in simpler(&q.expr) {
                let old = std::mem::replace(&mut c.specs[x], Spec::Query(Query { expr: cand, ..q.clone() }));
                if holds(&c) {
                    changed = true;
                    break;
                }
                c.specs[x] = old;
            }
        }
        if !changed {
            break;
        }
    }
    let (live, fresh) = last_read(&c);
    println!("# shrunk: live {live} fresh {fresh}");
    println!("{}", case_src(&c));
}

fn two(case: &Case, runs: usize) {
    let progress = Arc::new(AtomicU64::new(0));
    let tally: Arc<Mutex<BTreeMap<String, usize>>> = Arc::default();
    {
        let (progress, tally, name) = (progress.clone(), tally.clone(), case.name);
        std::thread::spawn(move || {
            let mut last = u64::MAX;
            loop {
                std::thread::sleep(Duration::from_secs(20));
                let now = progress.load(Relaxed);
                if now == last {
                    for (sig, n) in tally.lock().unwrap().iter() {
                        println!("{name}\ttwo\t{n}\t{sig}");
                    }
                    println!("{name}\ttwo-HANG\tafter {now} runs");
                    std::process::exit(3);
                }
                last = now;
            }
        });
    }
    let done = AtomicUsize::new(0);
    for _ in 0..runs {
        let w = world::<Shared>(case);
        let outs: Mutex<Vec<String>> = Mutex::new(vec![String::new(); case.ops.len()]);
        let barrier = Barrier::new(2);
        std::thread::scope(|s| {
            for t in 0..2 {
                let (w, outs, barrier) = (&w, &outs, &barrier);
                s.spawn(move || {
                    for phase in &case.phases {
                        barrier.wait();
                        let mine = match phase {
                            Phase::M(i, who) => (*who == t).then_some(*i),
                            Phase::R(slots) => slots[t],
                        };
                        if let Some(i) = mine {
                            let out = apply(w, &case.ops[i]);
                            outs.lock().unwrap()[i] = out;
                        }
                        barrier.wait();
                    }
                });
            }
        });
        let outs = outs.into_inner().unwrap();
        let sig: Vec<String> = (case.ops.iter().enumerate())
            .filter(|(_, o)| o.is_read())
            .map(|(k, _)| format!("{k}={}", outs[k]))
            .collect();
        *tally.lock().unwrap().entry(sig.join(" ")).or_default() += 1;
        done.fetch_add(1, Relaxed);
        progress.fetch_add(1, Relaxed);
    }
    let serial = serial_sigs(case);
    let tally = tally.lock().unwrap();
    let unserial: usize = (tally.iter()).filter(|(sig, _)| !serial.contains(*sig)).map(|(_, n)| n).sum();
    println!("{}\ttwo-summary\truns {runs} unserial {unserial} orders {}", case.name, serial.len());
    for (sig, n) in tally.iter() {
        let kind = if serial.contains(sig) { "serial" } else { "UNSERIAL" };
        println!("{}\ttwo\t{n}\t{kind}\t{sig}", case.name);
    }
}

/// Every read's outcome in each single-thread order of the case's read
/// pairs, as two-thread runs sign theirs.
fn serial_sigs(case: &Case) -> std::collections::BTreeSet<String> {
    let pairs = (case.phases.iter()).filter(|p| matches!(p, Phase::R([Some(_), Some(_)]))).count();
    (0..1u64 << pairs)
        .map(|mask| {
            let w = world::<Shared>(case);
            let mut outs = vec![String::new(); case.ops.len()];
            let mut pair = 0;
            for phase in &case.phases {
                let order = match *phase {
                    Phase::M(i, _) | Phase::R([Some(i), None]) | Phase::R([None, Some(i)]) => vec![i],
                    Phase::R([Some(a), Some(b)]) => {
                        let (lo, hi) = (a.min(b), a.max(b));
                        pair += 1;
                        if mask >> (pair - 1) & 1 == 1 {
                            vec![hi, lo]
                        } else {
                            vec![lo, hi]
                        }
                    }
                    Phase::R([None, None]) => vec![],
                };
                for i in order {
                    outs[i] = apply(&w, &case.ops[i]);
                }
            }
            (case.ops.iter().enumerate())
                .filter(|(_, o)| o.is_read())
                .map(|(k, _)| format!("{k}={}", outs[k]))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

#[test]
#[ignore = "a tool: B7_CASE, B7_MODE, B7_RUNS"]
fn b7_run() {
    // The generated case list is one large literal, and native reads nest
    // as deep as a case's queries.
    std::thread::Builder::new().stack_size(1 << 30).spawn(run_all).unwrap().join().unwrap();
}

fn run_all() {
    quiet();
    let want = std::env::var("B7_CASE").ok();
    let mode = std::env::var("B7_MODE").unwrap_or_else(|_| "local".to_owned());
    let runs = std::env::var("B7_RUNS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(1000);
    println!("# api {API} projected {}", projected());
    for case in cases() {
        if want.as_deref().is_some_and(|w| w != case.name) {
            continue;
        }
        println!("# case {}", case.name);
        match mode.as_str() {
            "local" => single::<Local>(&case, "local"),
            "shared" => single::<Shared>(&case, "shared"),
            "two" => two(&case, runs),
            "shrink" => {
                shrink(case);
                continue;
            }
            "entry" => {
                entry::<Local>(&case, "local");
                entry::<Shared>(&case, "shared");
            }
            m => panic!("B7_MODE {m}"),
        }
    }
}
