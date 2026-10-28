//! First and second read of a chain of `n` queries over a panicking one,
//! with an input `set` between the reads; `frames=none` uses plain
//! queries and no budget, otherwise every query is `Replay::Allowed`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use incr_core::{Cells, Local, QueryOpts, Replay, Runtime, Shared};

fn opts<T>(frames: Option<u32>) -> QueryOpts<T> {
    match frames {
        Some(_) => QueryOpts {
            replay: Replay::Allowed,
            ..Default::default()
        },
        None => QueryOpts::default(),
    }
}

fn contained_chain<C: Cells>(frames: Option<u32>, n: usize) -> (f64, f64) {
    let rt = Runtime::<C>::new();
    if let Some(frames) = frames {
        rt.set_placeholder(0_u64);
        rt.set_max_frames(frames);
    }
    let failed = rt.create_query_with(
        |_| -> u64 { panic!("failed below the chain") },
        opts(frames),
    );
    let input = rt.create_input(0_u64);
    let mut below = rt.create_query_with(move |rt| rt.get(input) + rt.get(failed), opts(frames));
    for _ in 1..n {
        let dep = below;
        below = rt.create_query_with(move |rt| rt.get(dep) + 1, opts(frames));
    }
    let top = below;
    let t = Instant::now();
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(top))).is_err());
    let first = t.elapsed().as_secs_f64() * 1e3;
    rt.set(input, 1);
    let t = Instant::now();
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(top))).is_err());
    (first, t.elapsed().as_secs_f64() * 1e3)
}

#[test]
fn kr_b1_probe() {
    let tree = std::env::var("FIGURE_TREE").unwrap_or_default();
    std::panic::set_hook(Box::new(|_| {}));
    std::thread::Builder::new()
        .stack_size(1 << 29)
        .spawn(move || {
            for frames in [None, Some(1), Some(2), Some(16)] {
                let label = frames.map_or_else(|| "none".to_string(), |f| f.to_string());
                for n in [1000, 2000, 4000, 8000] {
                    let (lf, ls) = contained_chain::<Local>(frames, n);
                    let (sf, ss) = contained_chain::<Shared>(frames, n);
                    println!(
                        "FIG {tree} kr_b1_probe frames={label} n={n} local_first_ms={lf:.2} local_second_ms={ls:.2} shared_first_ms={sf:.2} shared_second_ms={ss:.2} local_second_us_per_level={:.2}",
                        ls * 1e3 / n as f64
                    );
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
