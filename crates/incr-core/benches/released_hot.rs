use incr_core::{Runtime, Shared};
use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::time::Instant;

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let threads = env("THREADS", 4) as usize;
    let iters = env("ITERS", 200_000);
    let entry = env("ENTRY", 0) == 1;
    let set = env("SET", 1) == 1;
    let rt = Arc::new(Runtime::<Shared>::new());
    if entry {
        let (a, b) = (rt.create_input(0_u64), rt.create_input(0_u64));
        let q = rt.create_query(move |rt| rt.get(a) + rt.get(b));
        rt.get(q);
        rt.hold_released_entry(q);
    }
    let pairs: Vec<_> = (0..threads)
        .map(|_| {
            let i = rt.create_input(0_u64);
            let q = rt.create_query(move |rt| rt.get(i) + 1);
            rt.get(q);
            (i, q)
        })
        .collect();
    let barrier = Arc::new(Barrier::new(threads));
    let handles: Vec<_> = pairs
        .into_iter()
        .map(|(i, q)| {
            let (rt, barrier) = (rt.clone(), barrier.clone());
            std::thread::spawn(move || {
                barrier.wait();
                let start = Instant::now();
                for k in 1..=iters {
                    if set {
                        rt.set(i, k);
                    }
                    black_box(rt.get(q));
                }
                start.elapsed()
            })
        })
        .collect();
    let slowest = handles
        .into_iter()
        .map(|h| h.join().unwrap())
        .max()
        .unwrap();
    println!(
        "threads {threads} entry {} set {} ns/op {:.1}",
        entry as u8,
        set as u8,
        slowest.as_nanos() as f64 / iters.max(1) as f64
    );
}
