//! `k` queries each reading an input and one query left Failed over a
//! chain of `k` queries, read through one top query that catches each
//! reader's panic: a clean read builds the fan, a read after a `set` below
//! the chain fails the chain, and a read after a `set` of the readers'
//! input reruns the readers over the Failed query.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

use incr_core::{Cells, Incr, Local, Runtime, Shared};

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Each of the three reads: the clean read that builds the fan, the read
/// after `set(fail)` and the read after `set(input)`.
struct Read {
    ms: f64,
    chain_runs: u64,
    reader_runs: u64,
    /// Whether the read of the top query panicked.
    panicked: bool,
}

/// The three reads of a fan of `k`.
fn failed_fan_in<C: Cells>(k: usize) -> [Read; 3] {
    let chain_runs = Arc::new(AtomicU64::new(0));
    let reader_runs = Arc::new(AtomicU64::new(0));
    let rt = Runtime::<C>::new();
    let fail = rt.create_input(0_u64);
    let mut failed = rt.create_query(move |rt| {
        assert_eq!(rt.get(fail), 0, "failed below the fan");
        0
    });
    for _ in 0..k {
        let below = failed;
        let runs = chain_runs.clone();
        failed = rt.create_query(move |rt| {
            runs.fetch_add(1, Relaxed);
            rt.get(below) + 1
        });
    }
    let input = rt.create_input(0_u64);
    let readers: Arc<Vec<Incr<u64>>> = Arc::new(
        (0..k)
            .map(|_| {
                let runs = reader_runs.clone();
                rt.create_query(move |rt| {
                    runs.fetch_add(1, Relaxed);
                    rt.get(input) + rt.get(failed)
                })
            })
            .collect(),
    );
    let top = {
        let readers = readers.clone();
        rt.create_query(move |rt| {
            for &r in readers.iter() {
                let _ = catch_unwind(AssertUnwindSafe(|| rt.get(r)));
            }
            0
        })
    };
    let read = |f: &dyn Fn() -> bool| {
        chain_runs.store(0, Relaxed);
        reader_runs.store(0, Relaxed);
        let t = Instant::now();
        let panicked = f();
        Read {
            ms: ms(t),
            panicked,
            chain_runs: chain_runs.load(Relaxed),
            reader_runs: reader_runs.load(Relaxed),
        }
    };
    let top_read = || catch_unwind(AssertUnwindSafe(|| rt.get(top))).is_err();
    let clean = read(&top_read);
    rt.set(fail, 1);
    let after_fail = read(&top_read);
    rt.set(input, 1);
    let after_input = read(&top_read);
    [clean, after_fail, after_input]
}

#[test]
fn failed_fan_in_probe() {
    let tree = std::env::var("FIGURE_TREE").unwrap_or_default();
    std::panic::set_hook(Box::new(|_| {}));
    std::thread::Builder::new()
        .stack_size(1 << 29)
        .spawn(move || {
            for k in [1000, 2000, 4000, 8000] {
                let mut figs = format!("FIG {tree} failed_fan_in k={k}");
                for (mode, reads) in [
                    ("local", failed_fan_in::<Local>(k)),
                    ("shared", failed_fan_in::<Shared>(k)),
                ] {
                    for (name, r) in ["clean", "after_fail", "after_input"].iter().zip(&reads) {
                        figs += &format!(
                            " {mode}_{name}_ms={:.2} {mode}_{name}_us_per_reader={:.2} {mode}_{name}_chain_runs={} {mode}_{name}_reader_runs={} {mode}_{name}_top_panicked={}",
                            r.ms,
                            r.ms * 1e3 / k as f64,
                            r.chain_runs,
                            r.reader_runs,
                            r.panicked
                        );
                    }
                }
                println!("{figs}");
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
