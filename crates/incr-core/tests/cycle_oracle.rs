//! Cycles against an oracle: on a static graph whose reads never depend on
//! values, every node's result is fixed by Tarjan's SCCs and the episode's
//! iteration J(M, v0), whatever node is read first and on whichever
//! thread. The oracle computes those results directly; the engine is read
//! in a shuffled order under `Local`, `Shared` and several `Shared`
//! threads at once, and must match exactly, cycle participants included.
//!
//! Closures read with `try_get` and count an `Err` as a constant, so a
//! read that closes a cycle with no value never panics.
//!
//! A node can also be a cursor-driven operator over the fixed collection
//! `0..S_LEN`, an unseeded member when it is on a cycle: a filter whose
//! `pred` reads the node's one dep, counted, or an aggregate whose
//! `combine` reads it.

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, OnceLock};
use std::thread;

use incr_core::{
    Cells, Cycle, CycleStrategy, Fixpoint, Incr, IncrCollection, Local, PropagationTrace,
    QueryOpts, Runtime, Shared,
};
use proptest::prelude::*;

/// Every value is clamped here, so a Fixpoint over a monotone sum can
/// converge, and one that oscillates or grows reaches its limit.
const CAP: i64 = 40;
const ON_LIMIT: i64 = 1000;
const S_LEN: i64 = 20;

#[derive(Clone, Debug)]
enum Kind {
    None,
    Fallback(i64),
    Fixpoint { initial: i64, limit: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    /// `k` plus the sum of the reads.
    Sum,
    /// `S.filter(|x| x <= read(dep)).count()`.
    FilterCount,
    /// `S.aggregate(0, |_| 1, |a, b| min(a + b, max(read(dep), 0)))`, which
    /// is associative and commutative on nonnegative values, so it lands
    /// `min(|S|, max(v, 0))`.
    Aggregate,
}

#[derive(Clone, Debug)]
struct Node {
    op: Op,
    /// An operator node has exactly one.
    deps: Vec<usize>,
    k: i64,
    /// What an `Err` read counts as.
    err: i64,
    strategy: Kind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Res {
    Value(i64),
    Cyclic(Vec<usize>),
}

fn apply(node: &Node, mut reads: impl Iterator<Item = Option<i64>>) -> i64 {
    let out = match node.op {
        Op::Sum => node.k + reads.map(|r| r.unwrap_or(node.err)).sum::<i64>(),
        Op::FilterCount => filter_count(reads.next().unwrap().unwrap_or(node.err)),
        Op::Aggregate => aggregated(reads.next().unwrap().unwrap_or(node.err)),
    };
    out.clamp(-CAP, CAP)
}

fn filter_count(v: i64) -> i64 {
    (v + 1).clamp(0, S_LEN)
}

fn aggregated(v: i64) -> i64 {
    S_LEN.min(v.max(0))
}

fn tarjan(graph: &[Node]) -> Vec<usize> {
    struct State<'a> {
        graph: &'a [Node],
        index: Vec<Option<usize>>,
        low: Vec<usize>,
        on_stack: Vec<bool>,
        stack: Vec<usize>,
        comp: Vec<usize>,
        next: usize,
        comps: usize,
    }
    fn visit(s: &mut State<'_>, v: usize) {
        s.index[v] = Some(s.next);
        s.low[v] = s.next;
        s.next += 1;
        s.stack.push(v);
        s.on_stack[v] = true;
        for &w in &s.graph[v].deps {
            match s.index[w] {
                None => {
                    visit(s, w);
                    s.low[v] = s.low[v].min(s.low[w]);
                }
                Some(i) if s.on_stack[w] => s.low[v] = s.low[v].min(i),
                Some(_) => {}
            }
        }
        if Some(s.low[v]) == s.index[v] {
            loop {
                let w = s.stack.pop().unwrap();
                s.on_stack[w] = false;
                s.comp[w] = s.comps;
                if w == v {
                    break;
                }
            }
            s.comps += 1;
        }
    }
    let n = graph.len();
    let mut s = State {
        graph,
        index: vec![None; n],
        low: vec![0; n],
        on_stack: vec![false; n],
        stack: Vec::new(),
        comp: vec![0; n],
        next: 0,
        comps: 0,
    };
    for v in 0..n {
        if s.index[v].is_none() {
            visit(&mut s, v);
        }
    }
    s.comp
}

/// The result of every node. Tarjan numbers components in reverse
/// topological order, so a component's deps outside it are final when
/// it is reached.
fn oracle(graph: &[Node]) -> Vec<Res> {
    let n = graph.len();
    let comp = tarjan(graph);
    let comps = comp.iter().copied().max().map_or(0, |c| c + 1);
    let mut res: Vec<Option<Res>> = vec![None; n];
    for c in 0..comps {
        let members: Vec<usize> = (0..n).filter(|&v| comp[v] == c).collect();
        let cyclic = members.len() > 1 || graph[members[0]].deps.contains(&members[0]);
        if !cyclic {
            let v = members[0];
            let value = apply(&graph[v], graph[v].deps.iter().map(|&d| read(&res, d)));
            res[v] = Some(Res::Value(value));
            continue;
        }
        for (v, r) in episode(graph, &members, &res) {
            res[v] = Some(r);
        }
    }
    res.into_iter().map(Option::unwrap).collect()
}

fn read(res: &[Option<Res>], d: usize) -> Option<i64> {
    match res[d]
        .as_ref()
        .expect("a dep outside the component is final")
    {
        Res::Value(v) => Some(*v),
        Res::Cyclic(_) => None,
    }
}

/// J(M, v0) for the component `m`.
fn episode(graph: &[Node], m: &[usize], res: &[Option<Res>]) -> Vec<(usize, Res)> {
    let seeded: Vec<usize> = m
        .iter()
        .copied()
        .filter(|&v| !matches!(graph[v].strategy, Kind::None))
        .collect();
    let unseeded: Vec<usize> = m
        .iter()
        .copied()
        .filter(|&v| matches!(graph[v].strategy, Kind::None))
        .collect();
    // CU: unseeded members on a cycle of reads between unseeded members.
    let sub: Vec<Node> = (0..graph.len())
        .map(|v| Node {
            deps: if unseeded.contains(&v) {
                graph[v]
                    .deps
                    .iter()
                    .copied()
                    .filter(|d| unseeded.contains(d))
                    .collect()
            } else {
                Vec::new()
            },
            ..graph[v].clone()
        })
        .collect();
    let sub_comp = tarjan(&sub);
    let cu: Vec<usize> = unseeded
        .iter()
        .copied()
        .filter(|&u| {
            sub[u].deps.contains(&u)
                || unseeded
                    .iter()
                    .any(|&w| w != u && sub_comp[w] == sub_comp[u])
        })
        .collect();

    // The value of every unseeded member outside CU against `s`, through
    // the acyclic subgraph they form.
    let unseeded_values = |s: &dyn Fn(usize) -> i64| -> Vec<(usize, i64)> {
        let mut memo: Vec<Option<i64>> = vec![None; graph.len()];
        fn eval(
            v: usize,
            graph: &[Node],
            m: &[usize],
            cu: &[usize],
            res: &[Option<Res>],
            s: &dyn Fn(usize) -> i64,
            memo: &mut Vec<Option<i64>>,
        ) -> i64 {
            if let Some(x) = memo[v] {
                return x;
            }
            let reads: Vec<Option<i64>> = graph[v]
                .deps
                .iter()
                .map(|&d| {
                    if !m.contains(&d) {
                        read(res, d)
                    } else if cu.contains(&d) {
                        None
                    } else if !matches!(graph[d].strategy, Kind::None) {
                        Some(s(d))
                    } else {
                        Some(eval(d, graph, m, cu, res, s, memo))
                    }
                })
                .collect();
            let x = apply(&graph[v], reads.into_iter());
            memo[v] = Some(x);
            x
        }
        unseeded
            .iter()
            .copied()
            .filter(|u| !cu.contains(u))
            .map(|u| (u, eval(u, graph, m, &cu, res, s, &mut memo)))
            .collect()
    };

    let limit = seeded
        .iter()
        .filter_map(|&v| match graph[v].strategy {
            Kind::Fixpoint { limit, .. } => Some(limit),
            _ => None,
        })
        .min()
        .unwrap_or(1);
    let mut s: Vec<i64> = vec![0; graph.len()];
    for &v in &seeded {
        s[v] = match graph[v].strategy {
            Kind::Fallback(c) => c,
            Kind::Fixpoint { initial, .. } => initial,
            Kind::None => unreachable!(),
        };
    }
    let mut k = 1;
    loop {
        let prev = s.clone();
        let current = |v: usize| prev[v];
        let u_values = unseeded_values(&current);
        let mut next = prev.clone();
        for &v in &seeded {
            let out = apply(
                &graph[v],
                graph[v].deps.iter().map(|&d| {
                    if !m.contains(&d) {
                        read(res, d)
                    } else if cu.contains(&d) {
                        None
                    } else if !matches!(graph[d].strategy, Kind::None) {
                        Some(prev[d])
                    } else {
                        Some(u_values.iter().find(|(u, _)| *u == d).unwrap().1)
                    }
                }),
            );
            if let Kind::Fixpoint { .. } = graph[v].strategy {
                next[v] = out;
            }
        }
        let fixpoints: Vec<usize> = seeded
            .iter()
            .copied()
            .filter(|&v| matches!(graph[v].strategy, Kind::Fixpoint { .. }))
            .collect();
        let converged = fixpoints.iter().all(|&v| next[v] == prev[v]);
        s = next;
        if converged {
            break;
        }
        if k == limit {
            for &v in &fixpoints {
                if s[v] != prev[v] {
                    s[v] += ON_LIMIT;
                }
            }
            break;
        }
        k += 1;
    }

    let landed = |v: usize| s[v];
    let u_values = unseeded_values(&landed);
    let mut out = Vec::new();
    for &v in &seeded {
        out.push((v, Res::Value(s[v])));
    }
    for (u, x) in u_values {
        out.push((u, Res::Value(x)));
    }
    for &u in &cu {
        out.push((u, Res::Cyclic(m.to_vec())));
    }
    out
}

/// A runtime an operator's user function captures to read nodes, which
/// the operator API's `Fn(&T)` signatures otherwise can't reach.
struct RtRef<C: Cells>(*const Runtime<C>);

// SAFETY: every runtime here is boxed or in an `Arc` and outlives its
// operators' runs, and a `Local` runtime's operators run on its thread.
unsafe impl<C: Cells> Send for RtRef<C> {}
// SAFETY: as above.
unsafe impl<C: Cells> Sync for RtRef<C> {}

impl<C: Cells> RtRef<C> {
    fn rt(&self) -> &Runtime<C> {
        // SAFETY: as above.
        unsafe { &*self.0 }
    }
}

#[derive(Clone, Copy, Debug)]
enum Handle {
    Query(Incr<i64>),
    Count(Incr<u64>),
}

impl Handle {
    fn try_get<C: Cells>(self, rt: &Runtime<C>) -> Result<i64, Cycle> {
        match self {
            Handle::Query(n) => rt.try_get(n),
            Handle::Count(n) => rt.try_get(n).map(|c| c as i64),
        }
    }

    fn cycle_of<C: Cells>(self, rt: &Runtime<C>) -> Option<Cycle> {
        match self {
            Handle::Query(n) => rt.cycle_of(n),
            Handle::Count(n) => rt.cycle_of(n),
        }
    }

    /// The read's propagation trace, which lists every node the read
    /// verified or ran.
    fn traced<C: Cells>(self, rt: &Runtime<C>) -> PropagationTrace {
        match self {
            Handle::Query(n) => rt.get_traced(n).1,
            Handle::Count(n) => rt.get_traced(n).1,
        }
    }
}

struct Built<C: Cells> {
    s: IncrCollection<i64, C>,
    handles: Vec<Handle>,
    /// The slots of the engine nodes each graph node is made of: a filter
    /// count's filter is a cycle participant beside its count.
    slots: Vec<Vec<u32>>,
    /// `pred` and `combine` calls.
    calls: Arc<AtomicUsize>,
}

fn build<C: Cells>(rt: &Runtime<C>, graph: &[Node]) -> Built<C> {
    let cells: Vec<Arc<OnceLock<Handle>>> = (0..graph.len())
        .map(|_| Arc::new(OnceLock::new()))
        .collect();
    let s = rt.create_collection::<i64>();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();
    let mut slots = Vec::new();
    for node in graph {
        let deps: Vec<Arc<OnceLock<Handle>>> =
            node.deps.iter().map(|&d| cells[d].clone()).collect();
        let err = node.err;
        match node.op {
            Op::Sum => {
                let shape = node.clone();
                let f = move |rt: &Runtime<C>| {
                    apply(
                        &shape,
                        deps.iter().map(|d| d.get().unwrap().try_get(rt).ok()),
                    )
                };
                let cycle = match node.strategy {
                    Kind::None => CycleStrategy::None,
                    Kind::Fallback(c) => CycleStrategy::Fallback(Box::new(move |_c: &Cycle| c)),
                    Kind::Fixpoint { initial, limit } => CycleStrategy::Fixpoint(
                        Fixpoint::new(move || initial)
                            .step(|_prev, out, _k| out)
                            .converged(|a, b| a == b)
                            .max_iterations(limit)
                            .on_limit(|_c, s| s + ON_LIMIT),
                    ),
                };
                let q = rt.create_query_with(
                    f,
                    QueryOpts {
                        cycle,
                        ..Default::default()
                    },
                );
                handles.push(Handle::Query(q));
                slots.push(vec![q.slot()]);
            }
            Op::FilterCount => {
                let (r, dep, calls) = (RtRef(rt), deps[0].clone(), calls.clone());
                let f = s.filter(rt, move |&x| {
                    calls.fetch_add(1, SeqCst);
                    x <= dep.get().unwrap().try_get(r.rt()).unwrap_or(err)
                });
                let count = f.count(rt);
                handles.push(Handle::Count(count));
                slots.push(vec![f.version_node().slot(), count.slot()]);
            }
            Op::Aggregate => {
                let (r, dep, calls) = (RtRef(rt), deps[0].clone(), calls.clone());
                let a = s.aggregate(
                    rt,
                    0_i64,
                    |_| 1,
                    move |a, b| {
                        calls.fetch_add(1, SeqCst);
                        let v = dep.get().unwrap().try_get(r.rt()).unwrap_or(err);
                        (a + b).min(v.max(0))
                    },
                );
                handles.push(Handle::Query(a));
                slots.push(vec![a.slot()]);
            }
        }
    }
    for (cell, &n) in cells.iter().zip(&handles) {
        cell.set(n).unwrap();
    }
    for x in 0..S_LEN {
        s.insert(rt, x);
    }
    Built {
        s,
        handles,
        slots,
        calls,
    }
}

fn graph_node(slots: &[Vec<u32>], slot: u32) -> usize {
    slots
        .iter()
        .position(|s| s.contains(&slot))
        .unwrap_or_else(|| panic!("participant {slot} is no graph node's"))
}

fn participants(slots: &[Vec<u32>], cycle: &Cycle) -> Vec<usize> {
    let mut m: Vec<usize> = cycle
        .participants()
        .iter()
        .map(|p| graph_node(slots, p.slot()))
        .collect();
    m.sort_unstable();
    m.dedup();
    m
}

fn observe<C: Cells>(rt: &Runtime<C>, handles: &[Handle], slots: &[Vec<u32>], v: usize) -> Res {
    match handles[v].try_get(rt) {
        Ok(x) => Res::Value(x),
        Err(cycle) => Res::Cyclic(participants(slots, &cycle)),
    }
}

fn check_members<C: Cells>(rt: &Runtime<C>, built: &Built<C>, expected: &[Res], graph: &[Node]) {
    let comp = tarjan(graph);
    for v in 0..graph.len() {
        let in_cycle =
            graph[v].deps.contains(&v) || (0..graph.len()).any(|w| w != v && comp[w] == comp[v]);
        let landed = built.handles[v]
            .cycle_of(rt)
            .map(|c| participants(&built.slots, &c));
        if in_cycle {
            let m: Vec<usize> = (0..graph.len()).filter(|&w| comp[w] == comp[v]).collect();
            assert_eq!(
                landed,
                Some(m),
                "node {v}'s group, expected {:?}",
                expected[v]
            );
        } else {
            assert_eq!(landed, None, "node {v} is in no cycle");
        }
    }
}

/// After the episodes land: a further read of every node runs no closure
/// and verifies nothing, so every member landed `Clean`, and each operator
/// equals a fresh one built over its dep's landed value.
fn check_landed<C: Cells>(rt: &Runtime<C>, built: &Built<C>, expected: &[Res], graph: &[Node]) {
    let calls = built.calls.load(SeqCst);
    for (v, want) in expected.iter().enumerate() {
        match want {
            Res::Value(_) => {
                let trace = built.handles[v].traced(rt);
                assert!(
                    trace.node_traces.is_empty(),
                    "node {v}'s further read touched {:?}",
                    trace.node_traces
                );
            }
            Res::Cyclic(_) => {}
        }
    }
    assert_eq!(
        built.calls.load(SeqCst),
        calls,
        "a further read ran pred or combine"
    );

    for (v, node) in graph.iter().enumerate() {
        let (Res::Value(landed), Op::FilterCount | Op::Aggregate) = (&expected[v], node.op) else {
            continue;
        };
        let dv = built.handles[node.deps[0]].try_get(rt).unwrap_or(node.err);
        let fresh = match node.op {
            Op::Sum => unreachable!(),
            Op::FilterCount => rt.get(built.s.filter(rt, move |&x| x <= dv).count(rt)) as i64,
            Op::Aggregate => rt.get(built.s.aggregate(
                rt,
                0_i64,
                |_| 1,
                move |a, b| (a + b).min(dv.max(0)),
            )),
        };
        assert_eq!(*landed, fresh, "operator {v} against a fresh one over {dv}");
    }
}

fn graph_strategy() -> impl proptest::strategy::Strategy<Value = (Vec<Node>, Vec<usize>)> {
    (1usize..=8).prop_flat_map(|n| {
        let node = (
            prop_oneof![
                4 => Just(Op::Sum),
                1 => Just(Op::FilterCount),
                1 => Just(Op::Aggregate),
            ],
            proptest::collection::vec(0..n, 0..=3),
            0..n,
            -3i64..=5,
            -2i64..=2,
            prop_oneof![
                3 => Just(Kind::None),
                2 => (-5i64..=5).prop_map(Kind::Fallback),
                2 => (0i64..=3, 1u32..=6)
                    .prop_map(|(initial, limit)| Kind::Fixpoint { initial, limit }),
            ],
        )
            .prop_map(|(op, deps, dep, k, err, strategy)| match op {
                Op::Sum => Node {
                    op,
                    deps,
                    k,
                    err,
                    strategy,
                },
                Op::FilterCount | Op::Aggregate => Node {
                    op,
                    deps: vec![dep],
                    k: 0,
                    err,
                    strategy: Kind::None,
                },
            });
        (
            proptest::collection::vec(node, n),
            Just((0..n).collect::<Vec<usize>>()).prop_shuffle(),
        )
    })
}

/// An operator reads its upstream with `get`, which panics on a cycle with
/// no seeded member (R0, R3), and the panic unwinds every reader above it,
/// which the oracle's `try_get` closures don't model. So no case puts an
/// operator on such a cycle.
fn operators_settle(graph: &[Node], expected: &[Res]) -> bool {
    graph
        .iter()
        .zip(expected)
        .all(|(node, res)| node.op == Op::Sum || matches!(res, Res::Value(_)))
}

fn run_one<C: Cells>(graph: &[Node], order: &[usize], expected: &[Res]) {
    let rt = Box::new(Runtime::<C>::new());
    let built = build(&rt, graph);
    for &v in order {
        assert_eq!(
            observe(&rt, &built.handles, &built.slots, v),
            expected[v],
            "node {v}, entered in order {order:?}"
        );
    }
    for (v, want) in expected.iter().enumerate() {
        assert_eq!(
            &observe(&rt, &built.handles, &built.slots, v),
            want,
            "node {v} on a second read"
        );
    }
    check_members(&rt, &built, expected, graph);
    check_landed(&rt, &built, expected, graph);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    #[cfg_attr(miri, ignore)]
    fn local_matches_the_oracle((graph, order) in graph_strategy()) {
        let expected = oracle(&graph);
        prop_assume!(operators_settle(&graph, &expected));
        run_one::<Local>(&graph, &order, &expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shared_matches_the_oracle((graph, order) in graph_strategy()) {
        let expected = oracle(&graph);
        prop_assume!(operators_settle(&graph, &expected));
        run_one::<Shared>(&graph, &order, &expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shared_threads_match_the_oracle((graph, order) in graph_strategy()) {
        let expected = oracle(&graph);
        prop_assume!(operators_settle(&graph, &expected));
        let rt = Arc::new(Runtime::<Shared>::new());
        let built = build(&rt, &graph);
        let threads: Vec<_> = (0..3)
            .map(|t| {
                let (rt, handles, slots, expected) =
                    (rt.clone(), built.handles.clone(), built.slots.clone(), expected.clone());
                let mut order = order.clone();
                let shift = t * order.len() / 3;
                order.rotate_left(shift);
                thread::spawn(move || {
                    for &v in &order {
                        assert_eq!(observe(&rt, &handles, &slots, v), expected[v], "node {v}");
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        check_members(&rt, &built, &expected, &graph);
        check_landed(&rt, &built, &expected, &graph);
    }
}
