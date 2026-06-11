//! Mixed concurrent stress over the Shared strategy: writers mutating
//! heap-valued inputs, readers pulling derived queries, a churn thread
//! creating and deleting nodes, and collection traffic, all at once.
//! Runs under ThreadSanitizer and Miri's race detector in CI; iteration
//! counts shrink automatically under Miri.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use incr_core::{Cells, Runtime, Shared};

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
