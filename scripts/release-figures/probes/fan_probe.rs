#![allow(unexpected_cfgs)]
//! First read of a query over `n` new child queries through a 300-query
//! chain. rerun.sh builds the beta.5 tree with `--cfg fig_b5`, where
//! `Replay` does not exist and the queries are plain; elsewhere they are
//! `Replay::Allowed`.

use std::time::Instant;

use incr_core::{Cells, Incr, Local, Runtime, Shared};

#[cfg(not(fig_b5))]
fn query<C: Cells>(
    rt: &Runtime<C>,
    f: impl Fn(&Runtime<C>) -> u64 + Send + Sync + 'static,
) -> Incr<u64> {
    use incr_core::{QueryOpts, Replay};
    rt.create_query_with(
        f,
        QueryOpts {
            replay: Replay::Allowed,
            ..Default::default()
        },
    )
}

#[cfg(fig_b5)]
fn query<C: Cells>(
    rt: &Runtime<C>,
    f: impl Fn(&Runtime<C>) -> u64 + Send + Sync + 'static,
) -> Incr<u64> {
    rt.create_query(f)
}

fn fan<C: Cells>(depth: usize, n: u64) -> f64 {
    let rt = Runtime::<C>::new();
    #[cfg(not(fig_b5))]
    rt.set_placeholder(0_u64);
    let leaves: Vec<_> = (0..n).map(|i| query(&rt, move |_| i)).collect();
    let mut q = query(&rt, move |rt| {
        leaves.iter().map(|&l| rt.get(l)).sum::<u64>()
    });
    for _ in 0..depth {
        let dep = q;
        q = query(&rt, move |rt| rt.get(dep));
    }
    let t = Instant::now();
    assert_eq!(rt.get(q), n * (n - 1) / 2);
    t.elapsed().as_secs_f64() * 1e3
}

#[test]
fn fan_probe() {
    let tree = std::env::var("FIGURE_TREE").unwrap_or_default();
    std::thread::Builder::new()
        .stack_size(1 << 29)
        .spawn(move || {
            for n in [10_000_u64, 20_000, 40_000] {
                let local = fan::<Local>(300, n);
                let shared = fan::<Shared>(300, n);
                let k = n as f64 / 1e3;
                println!(
                    "FIG {tree} fan_probe depth=300 n={n} local_ms={local:.1} shared_ms={shared:.1} local_ms_per_k={:.3} shared_ms_per_k={:.3}",
                    local / k,
                    shared / k
                );
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
