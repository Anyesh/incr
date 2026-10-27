//! Stack bytes per level of a 400-member replayable `Shared` ring at the
//! default frame budget of 256: the stack pointer each member's closure
//! sees, differenced between consecutive members.

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use incr_core::{CycleStrategy, Fixpoint, Incr, QueryOpts, Replay, Runtime, Shared};

thread_local! {
    static SP: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

#[derive(Copy, Clone)]
enum Strategy {
    Fallback,
    Fixpoint,
}

fn ring(rt: &Runtime<Shared>, n: usize, strategy: Strategy) -> Vec<Incr<i64>> {
    let members: Arc<OnceLock<Vec<Incr<i64>>>> = Arc::new(OnceLock::new());
    let built = (0..n)
        .map(|k| {
            let members = members.clone();
            let cycle = match strategy {
                Strategy::Fallback if k == 0 => CycleStrategy::fallback(|_| 3),
                Strategy::Fixpoint => {
                    CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64).step(|_, out, _| out))
                }
                Strategy::Fallback => CycleStrategy::None,
            };
            rt.create_query_with(
                move |rt: &Runtime<Shared>| {
                    let marker = 0_u8;
                    let sp = std::hint::black_box(&marker) as *const u8 as usize;
                    SP.with(|v| v.borrow_mut().push((k, sp)));
                    (rt.get(members.get().unwrap()[(k + 1) % n]) + 1).min(5)
                },
                QueryOpts {
                    cycle,
                    replay: Replay::Allowed,
                },
            )
        })
        .collect::<Vec<_>>();
    members.set(built.clone()).unwrap();
    built
}

#[test]
fn w10_stack_per_level() {
    let tree = std::env::var("FIGURE_TREE").unwrap_or_default();
    for (name, strategy) in [
        ("fallback", Strategy::Fallback),
        ("fixpoint", Strategy::Fixpoint),
    ] {
        let line = std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || {
                let rt = Runtime::<Shared>::new();
                rt.set_placeholder(0_i64);
                rt.set_max_frames(256);
                let members = ring(&rt, 400, strategy);
                rt.get(members[0]);
                let v = SP.with(|v| std::mem::take(&mut *v.borrow_mut()));
                let mut d: Vec<usize> = v
                    .windows(2)
                    .filter(|w| w[1].0 == w[0].0 + 1 && w[1].1 < w[0].1)
                    .map(|w| w[0].1 - w[1].1)
                    .collect();
                d.sort_unstable();
                format!(
                    "strategy={name} pairs={} min_b={} median_b={} max_b={} deepest_b={}",
                    d.len(),
                    d.first().copied().unwrap_or(0),
                    d.get(d.len() / 2).copied().unwrap_or(0),
                    d.last().copied().unwrap_or(0),
                    v.first().map_or(0, |f| f.1) - v.iter().map(|x| x.1).min().unwrap_or(0)
                )
            })
            .unwrap()
            .join()
            .unwrap();
        println!("FIG {tree} w10_stack_probe {line}");
    }
}
