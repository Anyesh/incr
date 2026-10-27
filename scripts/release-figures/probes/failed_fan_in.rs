//! `k` queries each reading an input and one query left Failed over a
//! chain of `k` queries, read through one top query that catches each
//! reader's panic: the first read after a `set` below the chain fails the
//! chain, the second after a `set` of the readers' input reruns the
//! readers over the Failed query.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::time::Instant;

use incr_core::{Cells, Incr, Local, Runtime, Shared};

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// The clean read that builds the fan, the read after `set(fail)` and the
/// read after `set(input)`, in milliseconds.
fn failed_fan_in<C: Cells>(k: usize) -> (f64, f64, f64) {
    let rt = Runtime::<C>::new();
    let fail = rt.create_input(0_u64);
    let mut failed = rt.create_query(move |rt| {
        assert_eq!(rt.get(fail), 0, "failed below the fan");
        0
    });
    for _ in 0..k {
        let below = failed;
        failed = rt.create_query(move |rt| rt.get(below) + 1);
    }
    let input = rt.create_input(0_u64);
    let readers: Arc<Vec<Incr<u64>>> = Arc::new(
        (0..k)
            .map(|_| rt.create_query(move |rt| rt.get(input) + rt.get(failed)))
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
    let t = Instant::now();
    rt.get(top);
    let clean = ms(t);
    rt.set(fail, 1);
    let t = Instant::now();
    let _ = catch_unwind(AssertUnwindSafe(|| rt.get(top)));
    let first = ms(t);
    rt.set(input, 1);
    let t = Instant::now();
    let _ = catch_unwind(AssertUnwindSafe(|| rt.get(top)));
    (clean, first, ms(t))
}

#[test]
fn failed_fan_in_probe() {
    let tree = std::env::var("FIGURE_TREE").unwrap_or_default();
    std::panic::set_hook(Box::new(|_| {}));
    std::thread::Builder::new()
        .stack_size(1 << 29)
        .spawn(move || {
            for k in [1000, 2000, 4000, 8000] {
                let (lc, lf, ls) = failed_fan_in::<Local>(k);
                let (sc, sf, ss) = failed_fan_in::<Shared>(k);
                println!(
                    "FIG {tree} failed_fan_in k={k} local_clean_ms={lc:.2} local_after_fail_ms={lf:.2} local_after_input_ms={ls:.2} shared_clean_ms={sc:.2} shared_after_fail_ms={sf:.2} shared_after_input_ms={ss:.2} local_after_input_us_per_reader={:.2}",
                    ls * 1e3 / k as f64
                );
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
