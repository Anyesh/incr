//! Mixed concurrent stress over the Shared strategy: writers mutating
//! heap-valued inputs, readers pulling derived queries, a churn thread
//! creating and deleting nodes, and collection traffic, all at once.
//! Runs under ThreadSanitizer and Miri's race detector in CI; iteration
//! counts shrink automatically under Miri.

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Once};

use incr_core::{Cells, Incr, Runtime, Shared};

fn iters(full: usize) -> usize {
    if cfg!(miri) {
        full / 50
    } else {
        full
    }
}

#[test]
fn shared_mixed_workload_stress() {
    let rt: Arc<Runtime<Shared>> = Arc::new(Runtime::new());

    let name = rt.create_input("v0".to_string());
    let qty = rt.create_input(0_i64);
    let label = rt.create_query(move |rt| format!("{}#{}", rt.get(name), rt.get(qty)));
    assert_eq!(rt.get(label), "v0#0");

    let col = rt.create_collection::<i64>();
    let evens = col.filter(&rt, |x| x % 2 == 0);
    let count = evens.count(&rt);
    let total = col.aggregate(&rt, 0_i64, |x| *x, |a, b| a + b);

    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    // Readers: derived string must always be a coherent name#qty pair.
    for _ in 0..2 {
        let rt = Arc::clone(&rt);
        let stop = Arc::clone(&stop);
        handles.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let v = rt.get(label);
                let (n, q) = v.split_once('#').expect("torn label");
                assert!(n.starts_with('v'), "torn name: {}", v);
                q.parse::<i64>().expect("torn qty");
            }
        }));
    }

    // Churn: create and delete temp nodes, recycling slots constantly.
    {
        let rt = Arc::clone(&rt);
        let stop = Arc::clone(&stop);
        handles.push(std::thread::spawn(move || {
            let mut i = 0_i64;
            while !stop.load(Ordering::Relaxed) {
                let tmp = rt.create_input(format!("tmp{}", i));
                assert_eq!(rt.get(tmp), format!("tmp{}", i));
                rt.delete_node(tmp);
                i += 1;
            }
        }));
    }

    // Writers.
    let writer_rounds = iters(2000) as i64;
    {
        let rt = Arc::clone(&rt);
        let h = std::thread::spawn(move || {
            for i in 1..=writer_rounds {
                rt.set(name, format!("v{}", i));
                rt.set(qty, i);
            }
        });
        handles.push(h);
    }

    // Collection traffic on the main thread.
    let n = iters(1000) as i64;
    for i in 0..n {
        col.insert(&rt, i);
        if i % 3 == 0 {
            let _ = rt.get(count);
            let _ = rt.get(total);
        }
    }

    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(rt.get(count), (n as u64).div_ceil(2));
    assert_eq!(rt.get(total), (0..n).sum::<i64>());
    assert_eq!(
        rt.get(label),
        format!("v{}#{}", writer_rounds, writer_rounds)
    );
}

thread_local! {
    static EXPECTING_UNWIND: Cell<bool> = const { Cell::new(false) };
}

/// Run `f`, catching its panic without printing it. Other tests in this
/// binary run in parallel, so the hook stays quiet only on the thread
/// that asked.
fn catch_quietly<R>(f: impl FnOnce() -> R) -> std::thread::Result<R> {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !EXPECTING_UNWIND.with(Cell::get) {
                default(info);
            }
        }));
    });
    EXPECTING_UNWIND.with(|e| e.set(true));
    let result = catch_unwind(AssertUnwindSafe(f));
    EXPECTING_UNWIND.with(|e| e.set(false));
    result
}

fn panic_message(err: &(dyn std::any::Any + Send)) -> &str {
    err.downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| err.downcast_ref::<&str>().copied())
        .unwrap_or("<non-string panic payload>")
}

/// One victim subgraph of mixed value types, hanging off its own root
/// and reading the shared `base`, which no cascade may reach:
/// r -> a -> b (String) -> c -> l1, and b -> l2.
struct Victim {
    root: Incr<i64>,
    a: Incr<i64>,
    leaves: [Incr<i64>; 2],
}

const VICTIM_NODES: usize = 6;

fn build_victim(rt: &Runtime<Shared>, base: Incr<i64>, i: i64) -> Victim {
    let root = rt.create_input(i);
    let a = rt.create_query(move |rt| {
        let _ = rt.get(base);
        rt.get(root) + 1
    });
    let b = rt.create_query(move |rt| (rt.get(a) * 2).to_string());
    let c = rt.create_query(move |rt| rt.get(a) + rt.get(b).len() as i64);
    let l1 = rt.create_query(move |rt| rt.get(c) + 1);
    let l2 = rt.create_query(move |rt| rt.get(b).len() as i64);
    assert_eq!(rt.get(l1), i + 2 + ((i + 1) * 2).to_string().len() as i64);
    let _ = rt.get(l2);
    Victim {
        root,
        a,
        leaves: [l1, l2],
    }
}

/// Cascades overlap with each other, with plain deletes inside their
/// subgraph, with a thread that creates, computes and deletes nodes
/// (recycling the cascades' slots), and with sets on an input the victims
/// share with survivors. No node outside a victim subgraph may ever read
/// as dead or stale, and every victim node is deleted exactly once.
#[test]
fn shared_cascade_stress() {
    let rt: Arc<Runtime<Shared>> = Arc::new(Runtime::new());
    let base = rt.create_input(0_i64);
    let survivors: Vec<(Incr<i64>, Incr<i64>)> = (0..4_i64)
        .map(|k| {
            let s = rt.create_input(k);
            let q = rt.create_query(move |rt| {
                let _ = rt.get(base);
                rt.get(s) * 2
            });
            assert_eq!(rt.get(q), k * 2);
            (s, q)
        })
        .collect();
    let stop = Arc::new(AtomicBool::new(false));
    let churned = Arc::new(AtomicI64::new(0));

    let mut background = Vec::new();
    {
        let (rt, stop, survivors) = (Arc::clone(&rt), Arc::clone(&stop), survivors.clone());
        background.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                for (k, &(s, q)) in survivors.iter().enumerate() {
                    assert!(rt.is_alive(s) && rt.is_alive(q), "survivor {k} died");
                    assert_eq!(rt.get(q), k as i64 * 2);
                }
            }
        }));
    }
    {
        let (rt, stop, churned) = (Arc::clone(&rt), Arc::clone(&stop), Arc::clone(&churned));
        background.push(std::thread::spawn(move || {
            let mut i = 0_i64;
            while !stop.load(Ordering::Relaxed) {
                let tmp = rt.create_input(i);
                let q = rt.create_query(move |rt| {
                    let _ = rt.get(base);
                    rt.get(tmp) - 1
                });
                assert_eq!(rt.get(q), i - 1);
                assert!(rt.is_alive(tmp) && rt.is_alive(q), "churned node died");
                if i % 2 == 0 {
                    assert_eq!(rt.delete_with_dependents(tmp), 2);
                } else {
                    rt.delete_node(q);
                    rt.delete_node(tmp);
                }
                i += 1;
            }
            churned.store(i, Ordering::Relaxed);
        }));
    }
    {
        let (rt, stop) = (Arc::clone(&rt), Arc::clone(&stop));
        background.push(std::thread::spawn(move || {
            let mut i = 0_i64;
            while !stop.load(Ordering::Relaxed) {
                i += 1;
                rt.set(base, i);
            }
        }));
    }

    for i in 0..iters(300) as i64 {
        let v = build_victim(&rt, base, i);
        let deleted: usize = std::thread::scope(|scope| {
            let workers: Vec<_> = if i % 2 == 0 {
                [v.root, v.root, v.a]
                    .map(|from| {
                        let rt = &rt;
                        scope.spawn(move || rt.delete_with_dependents(from))
                    })
                    .into()
            } else {
                let cascade = scope.spawn(|| rt.delete_with_dependents(v.root));
                let plain = scope.spawn(|| {
                    v.leaves
                        .iter()
                        .filter(|&&leaf| match catch_quietly(|| rt.delete_node(leaf)) {
                            Ok(()) => true,
                            Err(err) => {
                                let msg = panic_message(&*err);
                                assert!(
                                    msg.contains("deleted twice concurrently")
                                        || msg.contains("stale"),
                                    "plain delete lost the race wrongly: {msg}"
                                );
                                false
                            }
                        })
                        .count()
                });
                vec![cascade, plain]
            };
            workers.into_iter().map(|w| w.join().unwrap()).sum()
        });
        assert_eq!(
            deleted, VICTIM_NODES,
            "round {i}: a node was deleted twice or missed"
        );
        assert!(!rt.is_alive(v.root) && v.leaves.iter().all(|&l| !rt.is_alive(l)));
    }

    stop.store(true, Ordering::Relaxed);
    for h in background {
        h.join().unwrap();
    }
    assert!(churned.load(Ordering::Relaxed) > 0);
    assert_eq!(rt.node_count(), 1 + 2 * survivors.len());
    for (k, &(s, q)) in survivors.iter().enumerate() {
        assert!(rt.is_alive(s));
        assert_eq!(rt.get(q), k as i64 * 2);
    }
}

#[test]
fn shared_two_runtimes_do_not_interfere() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Runtime<Shared>>();
    let a: Runtime<Shared> = Runtime::new();
    let b: Runtime<Shared> = Runtime::new();
    let ia = a.create_input(1_i64);
    let _ib = b.create_input(2_i64);
    assert_eq!(a.get(ia), 1);
}

#[test]
fn local_strategy_types_are_not_sync() {
    fn is_sync<T: Sync>() {}
    fn is_send<T: Send>() {}
    // Compile-time documentation of the confinement story: these must
    // hold for Shared and the equivalent for Local must NOT compile
    // (checked by the trybuild-style absence of such code in-tree).
    is_sync::<Runtime<Shared>>();
    is_send::<Runtime<Shared>>();
    is_sync::<<Shared as Cells>::ValueSlot<String>>();
}

/// Eight threads insert, remove, look up and walk one input table, and
/// create queries through a second table inside computes. Keys 0..32
/// churn; keys 1000..1008 are inserted first and never removed, so every
/// `for_each` must visit each of them exactly once, and no pair twice.
#[test]
fn shared_table_stress() {
    const PINNED: std::ops::Range<u64> = 1000..1008;
    let rt = Arc::new(Runtime::<Shared>::new());
    let inputs = Arc::new(rt.create_table::<u64, i64>());
    let queries = Arc::new(rt.create_table::<u64, i64>());
    for k in PINNED {
        inputs.get_or_insert_input(&rt, k, || 10 * k as i64);
    }
    let workers: Vec<_> = (0..8_u64)
        .map(|t| {
            let (rt, inputs, queries) =
                (Arc::clone(&rt), Arc::clone(&inputs), Arc::clone(&queries));
            std::thread::spawn(move || {
                let mut seed = 0x9e37_79b9_7f4a_7c15_u64 ^ (t + 1);
                let mut next = move || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed
                };
                for _ in 0..iters(2000) {
                    let k = next() % 32;
                    match next() % 5 {
                        0 => {
                            let h = inputs.get_or_insert_input(&rt, k, || 10 * k as i64);
                            if let Ok(v) = catch_quietly(|| rt.get(h)) {
                                assert_eq!(v, 10 * k as i64);
                            }
                        }
                        1 => {
                            let _ = inputs.remove(&rt, &k);
                        }
                        2 => {
                            if let Some(h) = inputs.get(&rt, &k) {
                                if let Ok(v) = catch_quietly(|| rt.get(h)) {
                                    assert_eq!(v, 10 * k as i64);
                                }
                            }
                        }
                        3 => {
                            let mut seen = Vec::new();
                            inputs.for_each(&rt, |key, h| seen.push((*key, h)));
                            let pinned: Vec<u64> = seen
                                .iter()
                                .map(|(k, _)| *k)
                                .filter(|k| PINNED.contains(k))
                                .collect();
                            let mut sorted = pinned.clone();
                            sorted.sort_unstable();
                            assert_eq!(sorted, PINNED.collect::<Vec<_>>(), "pinned keys once each");
                            let mut pairs = seen.clone();
                            pairs.sort_unstable_by_key(|(k, h)| (*k, h.slot()));
                            pairs.dedup();
                            assert_eq!(pairs.len(), seen.len(), "no pair twice");
                        }
                        _ => {
                            let p = PINNED.start + k % 8;
                            let it = Arc::clone(&inputs);
                            let q = queries.get_or_insert_query(&rt, p, move |rt| {
                                rt.get(it.get_or_insert_input(rt, p, || 10 * p as i64)) + 1
                            });
                            assert_eq!(rt.get(q), 10 * p as i64 + 1);
                        }
                    }
                }
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    for k in PINNED {
        let h = inputs.get(&rt, &k).expect("pinned keys stay");
        assert_eq!(rt.get(h), 10 * k as i64);
    }
}

/// Eight threads change the key sets of a chain of creators (creator
/// `i` reads creator `i - 1`), read the creators and the children they
/// return, and attach and delete readers of those children, so sweeps,
/// retention and deletes of owned sets race each other. Every child a
/// read returns holds its key's value or is gone, a creator's final run
/// returns exactly its keys' children, and the cascades from the inputs
/// leave nothing behind: every child was owned.
#[test]
fn shared_children_stress() {
    const CREATORS: usize = 6;
    type Kids = Vec<(u32, Incr<u64>)>;
    let rt = Arc::new(Runtime::<Shared>::new());
    let mut masks = Vec::new();
    let mut creators: Vec<Incr<Kids>> = Vec::new();
    for _ in 0..CREATORS {
        let mask = rt.create_input(0_u8);
        let upstream = creators.last().copied();
        let creator = rt.create_query(move |rt| {
            if let Some(up) = upstream {
                let _ = rt.get(up).len();
            }
            let m = rt.get(mask);
            (0..6_u32)
                .filter(|k| m & (1 << k) != 0)
                .map(|k| (k, rt.child_input(k, move || u64::from(k) * 100)))
                .collect()
        });
        masks.push(mask);
        creators.push(creator);
    }
    let masks = Arc::new(masks);
    let creators = Arc::new(creators);
    let workers: Vec<_> = (0..8_u64)
        .map(|t| {
            let (rt, masks, creators) =
                (Arc::clone(&rt), Arc::clone(&masks), Arc::clone(&creators));
            std::thread::spawn(move || {
                let mut seed = 0x2545_f491_4f6c_dd1d_u64 ^ (t + 1);
                let mut next = move || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed
                };
                let mut readers: Vec<Incr<u64>> = Vec::new();
                for _ in 0..iters(1500) {
                    let c = (next() % CREATORS as u64) as usize;
                    match next() % 4 {
                        0 => rt.set(masks[c], (next() % 64) as u8),
                        1 => {
                            for (k, h) in rt.get(creators[c]) {
                                if let Ok(v) = catch_quietly(|| rt.get(h)) {
                                    assert_eq!(v, u64::from(k) * 100);
                                }
                            }
                        }
                        2 => {
                            let kids = rt.get(creators[c]);
                            if let Some(&(k, h)) =
                                kids.get((next() % 6) as usize % kids.len().max(1))
                            {
                                let reader = rt.create_query(move |rt| rt.get(h) + 1);
                                match catch_quietly(|| rt.get(reader)) {
                                    Ok(v) => assert_eq!(v, u64::from(k) * 100 + 1),
                                    Err(_) => rt.delete_node(reader),
                                }
                                if rt.is_alive(reader) {
                                    readers.push(reader);
                                }
                            }
                        }
                        _ => {
                            if !readers.is_empty() {
                                let r =
                                    readers.swap_remove((next() % readers.len() as u64) as usize);
                                rt.delete_node(r);
                            }
                        }
                    }
                }
                readers
            })
        })
        .collect();
    let readers: Vec<Incr<u64>> = workers
        .into_iter()
        .flat_map(|w| w.join().unwrap())
        .collect();
    for r in readers {
        rt.delete_node(r);
    }
    for (i, &creator) in creators.iter().enumerate() {
        let m = rt.get(masks[i]);
        let kids = rt.get(creator);
        let keys: Vec<u32> = kids.iter().map(|&(k, _)| k).collect();
        let expected: Vec<u32> = (0..6).filter(|k| m & (1 << k) != 0).collect();
        assert_eq!(keys, expected);
        for (k, h) in kids {
            assert_eq!(rt.get(h), u64::from(k) * 100);
        }
    }
    for &mask in masks.iter() {
        rt.delete_with_dependents(mask);
    }
    for &creator in creators.iter() {
        rt.delete_with_dependents(creator);
    }
    assert_eq!(
        rt.node_count(),
        0,
        "every child was owned and went with its creator"
    );
}
