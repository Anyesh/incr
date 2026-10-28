use incr_core::{Cells, Local, Runtime, Shared};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};

fn chain<C: Cells>(n: u64) -> Duration {
    let rt = Runtime::<C>::new();
    let input = rt.create_input(7_u64);
    let mut below = rt.create_query(move |rt: &Runtime<C>| rt.get(input));
    for k in 0..n {
        let failing = rt.create_query(move |rt: &Runtime<C>| -> u64 {
            panic!("chain {}", k + rt.get(input))
        });
        let dep = below;
        below = rt.create_query(move |rt: &Runtime<C>| {
            let caught = catch_unwind(AssertUnwindSafe(|| rt.get(failing))).is_err();
            u64::from(caught) + rt.get(dep)
        });
    }
    let start = Instant::now();
    assert_eq!(rt.get(below), n + 7);
    start.elapsed()
}

fn on_big_stack<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> R {
    std::thread::Builder::new()
        .stack_size(4 << 30)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn probe_nested_caught_chain_unbounded() {
    std::panic::set_hook(Box::new(|_| {}));
    let tree = std::env::var("NC_TREE").unwrap_or_default();
    for n in [8000_u64, 16000, 32000, 64000] {
        let local = on_big_stack(move || chain::<Local>(n));
        let shared = on_big_stack(move || chain::<Shared>(n));
        println!("NC {tree} n={n} local_ms={:.1} shared_ms={:.1}", local.as_secs_f64() * 1e3, shared.as_secs_f64() * 1e3);
    }
}
