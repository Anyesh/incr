//! Cycles against an oracle: on a static graph whose reads never depend on
//! values, every node's result is fixed by Tarjan's SCCs and the episode's
//! iteration J(M, v0), whatever node is read first and on whichever
//! thread. The oracle computes those results directly; the engine is read
//! in a shuffled order under `Local`, `Shared` and several `Shared`
//! threads at once, and must match exactly, cycle participants included.
//!
//! Closures read with `try_get` and count an `Err` as a constant, so a
//! read that closes a cycle with no value never panics.

use std::sync::{Arc, OnceLock};
use std::thread;

use incr_core::{Cells, Cycle, CycleStrategy, Fixpoint, Incr, Local, QueryOpts, Runtime, Shared};
use proptest::prelude::*;

/// Every value is clamped here, so a Fixpoint over a monotone sum can
/// converge, and one that oscillates or grows reaches its limit.
const CAP: i64 = 40;
const ON_LIMIT: i64 = 1000;

#[derive(Clone, Debug)]
enum Kind {
    None,
    Fallback(i64),
    Fixpoint { initial: i64, limit: u32 },
}

#[derive(Clone, Debug)]
struct Node {
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

fn apply(node: &Node, reads: impl Iterator<Item = Option<i64>>) -> i64 {
    let sum: i64 = reads.map(|r| r.unwrap_or(node.err)).sum();
    (node.k + sum).clamp(-CAP, CAP)
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

fn build<C: Cells>(rt: &Runtime<C>, graph: &[Node]) -> Vec<Incr<i64>> {
    let cells: Vec<Arc<OnceLock<Incr<i64>>>> = (0..graph.len())
        .map(|_| Arc::new(OnceLock::new()))
        .collect();
    let mut nodes = Vec::new();
    for node in graph {
        let deps: Vec<Arc<OnceLock<Incr<i64>>>> =
            node.deps.iter().map(|&d| cells[d].clone()).collect();
        let shape = node.clone();
        let f = move |rt: &Runtime<C>| {
            apply(
                &shape,
                deps.iter().map(|d| rt.try_get(*d.get().unwrap()).ok()),
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
        nodes.push(rt.create_query_with(
            f,
            QueryOpts {
                cycle,
                ..Default::default()
            },
        ));
    }
    for (cell, &n) in cells.iter().zip(&nodes) {
        cell.set(n).unwrap();
    }
    nodes
}

fn observe<C: Cells>(rt: &Runtime<C>, nodes: &[Incr<i64>], v: usize) -> Res {
    match rt.try_get(nodes[v]) {
        Ok(x) => Res::Value(x),
        Err(cycle) => {
            let mut slots: Vec<usize> = cycle
                .participants()
                .iter()
                .map(|p| nodes.iter().position(|n| n.slot() == p.slot()).unwrap())
                .collect();
            slots.sort_unstable();
            Res::Cyclic(slots)
        }
    }
}

fn check_members<C: Cells>(rt: &Runtime<C>, nodes: &[Incr<i64>], expected: &[Res], graph: &[Node]) {
    let comp = tarjan(graph);
    for v in 0..graph.len() {
        let in_cycle =
            graph[v].deps.contains(&v) || (0..graph.len()).any(|w| w != v && comp[w] == comp[v]);
        let landed = rt.cycle_of(nodes[v]).map(|c| {
            let mut m: Vec<usize> = c
                .participants()
                .iter()
                .map(|p| nodes.iter().position(|n| n.slot() == p.slot()).unwrap())
                .collect();
            m.sort_unstable();
            m
        });
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

fn graph_strategy() -> impl proptest::strategy::Strategy<Value = (Vec<Node>, Vec<usize>)> {
    (1usize..=8).prop_flat_map(|n| {
        let node = (
            proptest::collection::vec(0..n, 0..=3),
            -3i64..=5,
            -2i64..=2,
            prop_oneof![
                3 => Just(Kind::None),
                2 => (-5i64..=5).prop_map(Kind::Fallback),
                2 => (0i64..=3, 1u32..=6)
                    .prop_map(|(initial, limit)| Kind::Fixpoint { initial, limit }),
            ],
        )
            .prop_map(|(deps, k, err, strategy)| Node {
                deps,
                k,
                err,
                strategy,
            });
        (
            proptest::collection::vec(node, n),
            Just((0..n).collect::<Vec<usize>>()).prop_shuffle(),
        )
    })
}

fn run_one<C: Cells>(graph: &[Node], order: &[usize], expected: &[Res]) {
    let rt = Runtime::<C>::new();
    let nodes = build(&rt, graph);
    for &v in order {
        assert_eq!(
            observe(&rt, &nodes, v),
            expected[v],
            "node {v}, entered in order {order:?}"
        );
    }
    for (v, want) in expected.iter().enumerate() {
        assert_eq!(&observe(&rt, &nodes, v), want, "node {v} on a second read");
    }
    check_members(&rt, &nodes, expected, graph);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    #[cfg_attr(miri, ignore)]
    fn local_matches_the_oracle((graph, order) in graph_strategy()) {
        run_one::<Local>(&graph, &order, &oracle(&graph));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shared_matches_the_oracle((graph, order) in graph_strategy()) {
        run_one::<Shared>(&graph, &order, &oracle(&graph));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shared_threads_match_the_oracle((graph, order) in graph_strategy()) {
        let expected = oracle(&graph);
        let rt = Arc::new(Runtime::<Shared>::new());
        let nodes = build(&rt, &graph);
        let threads: Vec<_> = (0..3)
            .map(|t| {
                let (rt, nodes, expected) = (rt.clone(), nodes.clone(), expected.clone());
                let mut order = order.clone();
                let shift = t * order.len() / 3;
                order.rotate_left(shift);
                thread::spawn(move || {
                    for &v in &order {
                        assert_eq!(observe(&rt, &nodes, v), expected[v], "node {v}");
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        check_members(&rt, &nodes, &expected, &graph);
    }
}
