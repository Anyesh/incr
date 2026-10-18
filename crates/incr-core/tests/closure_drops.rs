//! Each query's closure is dropped exactly once: when its node is deleted
//! or when the runtime drops, never on a compute.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use incr_core::{Cells, Local, Runtime, Shared};

struct Counted(Arc<AtomicUsize>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn each_closure_dropped_once<C: Cells>() {
    let drops = Arc::new(AtomicUsize::new(0));
    let rt: Runtime<C> = Runtime::new();
    let x = rt.create_input(1_u64);
    let queries: Vec<_> = (0..10_u64)
        .map(|k| {
            let guard = Counted(Arc::clone(&drops));
            rt.create_query(move |rt| {
                let _ = &guard;
                rt.get(x) + k
            })
        })
        .collect();
    for (k, &q) in queries.iter().enumerate() {
        assert_eq!(rt.get(q), 1 + k as u64);
    }
    rt.set(x, 2);
    for (k, &q) in queries.iter().enumerate() {
        assert_eq!(rt.get(q), 2 + k as u64);
    }
    assert_eq!(
        drops.load(Ordering::Relaxed),
        0,
        "a compute dropped a closure"
    );

    for &q in &queries[..4] {
        rt.delete_node(q);
    }
    assert_eq!(drops.load(Ordering::Relaxed), 4);

    let recycled = {
        let guard = Counted(Arc::clone(&drops));
        rt.create_query(move |rt| {
            let _ = &guard;
            rt.get(x) * 10
        })
    };
    assert_eq!(rt.get(recycled), 20);
    drop(rt);
    assert_eq!(drops.load(Ordering::Relaxed), 11);
}

#[test]
fn local_closures_dropped_once() {
    each_closure_dropped_once::<Local>();
}

#[test]
fn shared_closures_dropped_once() {
    each_closure_dropped_once::<Shared>();
}
