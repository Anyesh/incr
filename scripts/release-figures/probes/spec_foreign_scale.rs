//! Wall time of `k` Clean reads of a foreign `Shared` runtime F made from a
//! speculative frame of a budgeted `Shared` runtime R, at several sizes of
//! F, while writer threads keep setting inputs of F.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use incr_core::{Incr, QueryOpts, Replay, Runtime, Shared};

const TIMEOUT: Duration = Duration::from_secs(120);

fn replayable<T>() -> QueryOpts<T> {
    QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    }
}

/// Threads setting inputs of F, each with `fan` dependents it reads again
/// after every set, so the next set walks all of them.
#[derive(Clone, Copy)]
struct Writers {
    threads: usize,
    fan: usize,
}

impl Writers {
    fn label(self) -> String {
        if self.threads == 0 {
            "off".to_string()
        } else {
            format!("{}x{}", self.threads, self.fan)
        }
    }
}

struct Outcome {
    us: Option<u128>,
    sets: u64,
}

/// R reads two leaves and then `spec`, a replayable query making `k` reads
/// of F's Clean inputs. With a frame budget of 1 the first leaf dooms the
/// reader, so `spec` runs as a speculative entry.
fn run(n: usize, k: usize, budget: Option<u32>, writers: Writers) -> Outcome {
    let f = Arc::new(Runtime::<Shared>::new());
    let nodes: Arc<Vec<Incr<u64>>> = Arc::new((0..n as u64).map(|i| f.create_input(i)).collect());
    let inputs: Vec<(Incr<u64>, Vec<Incr<u64>>)> = (0..writers.threads)
        .map(|_| {
            let w = f.create_input(0_u64);
            let fan = (0..writers.fan.max(1) as u64)
                .map(|i| f.create_query(move |rt| rt.get(w) + i))
                .collect::<Vec<_>>();
            for &d in &fan {
                f.get(d);
            }
            (w, fan)
        })
        .collect();

    let r = Arc::new(Runtime::<Shared>::new());
    r.set_placeholder(0_u64);
    r.set_max_frames(budget.unwrap_or(u32::MAX));
    let leaf0 = r.create_query(|_| 1_u64);
    let leaf1 = r.create_query(|_| 2_u64);
    let spec = {
        let (f, nodes) = (f.clone(), nodes.clone());
        r.create_query_with(
            move |_| (0..k).map(|j| f.get(nodes[j % n])).sum::<u64>(),
            replayable(),
        )
    };
    let wide = r.create_query_with(
        move |rt| rt.get(leaf0) + rt.get(leaf1) + rt.get(spec),
        replayable(),
    );
    let expected = 3 + (0..k).map(|j| (j % n) as u64).sum::<u64>();

    let stop = Arc::new(AtomicBool::new(false));
    let sets = Arc::new(AtomicU64::new(0));
    let writer_threads: Vec<_> = inputs
        .into_iter()
        .map(|(w, fan)| {
            let (f, stop, sets) = (f.clone(), stop.clone(), sets.clone());
            thread::spawn(move || {
                let mut i = 0_u64;
                while !stop.load(Ordering::Relaxed) {
                    i += 1;
                    f.set(w, i);
                    for &d in &fan {
                        f.get(d);
                    }
                    sets.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();
    if writers.threads != 0 {
        while sets.load(Ordering::Relaxed) < 100 {
            thread::yield_now();
        }
    }

    let (tx, rx) = mpsc::channel();
    let reader = {
        let r = r.clone();
        thread::spawn(move || {
            let start = Instant::now();
            let value = r.get(wide);
            let _ = tx.send((start.elapsed().as_micros(), value));
        })
    };
    let us = match rx.recv_timeout(TIMEOUT) {
        Ok((us, value)) => {
            assert_eq!(value, expected, "n={n} k={k} budget={budget:?}");
            Some(us)
        }
        Err(_) => None,
    };
    stop.store(true, Ordering::Relaxed);
    for t in writer_threads {
        t.join().expect("writer thread panicked");
    }
    if us.is_some() {
        reader.join().expect("reader thread panicked");
    }
    Outcome {
        us,
        sets: sets.load(Ordering::Relaxed),
    }
}

#[test]
fn spec_foreign_read_scale_probe() {
    let tree = std::env::var("FIGURE_TREE").unwrap_or_default();
    let k = 4000;
    for n in [1000_usize, 16000, 64000] {
        for (budget, writers) in [
            (Some(1), Writers { threads: 1, fan: 1 }),
            (
                Some(1),
                Writers {
                    threads: 3,
                    fan: 256,
                },
            ),
            (Some(1), Writers { threads: 0, fan: 0 }),
            (None, Writers { threads: 1, fan: 1 }),
            (
                None,
                Writers {
                    threads: 3,
                    fan: 256,
                },
            ),
        ] {
            let out = run(n, k, budget, writers);
            let b = budget.map_or_else(|| "none".to_string(), |b| b.to_string());
            let (us, per_read) = out.us.map_or_else(
                || {
                    (
                        format!("TIMEOUT>{}", TIMEOUT.as_micros()),
                        "none".to_string(),
                    )
                },
                |us| (us.to_string(), format!("{:.3}", us as f64 / k as f64)),
            );
            println!(
                "FIG {tree} spec_foreign_scale n={n} k={k} budget={b} writers={} us={us} us_per_read={per_read} sets={}",
                writers.label(),
                out.sets
            );
        }
    }
}
