//! A replayable query that catches the panics of `m = n / 2` replayable
//! dependencies and then reads an `m`-deep replayable chain, at a frame
//! budget of 1.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use incr_core::{Cells, Incr, Local, QueryOpts, Replay, Runtime, Shared};

fn replayable<T>() -> QueryOpts<T> {
    QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    }
}

fn probe_fan<C: Cells>(n: u64, frames: u32) -> u64 {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(frames);
    let input = rt.create_input(7_u64);
    let failing: Vec<Incr<u64>> = (0..n / 2)
        .map(|k| {
            rt.create_query_with(
                move |rt: &Runtime<C>| -> u64 { panic!("fan {}", k + rt.get(input)) },
                replayable(),
            )
        })
        .collect();
    let mut top = rt.create_query_with(move |rt| rt.get(input), replayable());
    for _ in 0..n / 2 {
        let below = top;
        top = rt.create_query_with(move |rt| rt.get(below) + 1, replayable());
    }
    let reader = rt.create_query_with(
        move |rt| {
            let caught = failing
                .iter()
                .filter(|&&f| catch_unwind(AssertUnwindSafe(|| rt.get(f))).is_err())
                .count() as u64;
            caught + rt.get(top)
        },
        replayable(),
    );
    rt.get(reader)
}

fn timed<C: Cells>(n: u64, frames: u32) -> (f64, bool) {
    std::thread::Builder::new()
        .stack_size(1 << 26)
        .spawn(move || {
            let t = Instant::now();
            let ok = catch_unwind(|| probe_fan::<C>(n, frames)).map_or(false, |v| v == n + 7);
            (t.elapsed().as_secs_f64(), ok)
        })
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn probe_fan_timing() {
    let tree = std::env::var("FIGURE_TREE").unwrap_or_default();
    std::panic::set_hook(Box::new(|_| {}));
    let frames = 1;
    for n in [1000_u64, 2000, 4000, 8000] {
        let (local, local_ok) = timed::<Local>(n, frames);
        let (shared, shared_ok) = timed::<Shared>(n, frames);
        println!(
            "FIG {tree} probe_fan frames={frames} n={n} m={} local_s={local:.2} shared_s={shared:.2} ok={}",
            n / 2,
            local_ok && shared_ok
        );
    }
}
