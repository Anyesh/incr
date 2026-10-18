//! Fixed-iteration driver for instruction counting. Runs one workload a
//! warm-up number of times and then a measured number of times, so that
//! `perf stat -e instructions:u` at N iterations minus the same run at 0,
//! divided by N, is the per-iteration count. `icount.sh` does that for
//! every workload.
//!
//! Usage: `icount <workload> <iterations> [warm-up]`, or `icount --list`
//! for each workload's name and default iteration count.

use incr_core::{Local, Shared};
use std::hint::black_box;
use std::process::ExitCode;
use workloads::{
    BatchRebuild, Chain, ChildrenChurn, ChildrenRerun, DeleteCascade, DeleteOwnedTree,
    DeleteSingle, Diamond, EarlyCutoff, Fanout, GetClean, GetContended, InsertThenRead, TableHit,
    TableMiss, ValueSlotRead, Workload,
};

// Allowed because the criterion helpers in the shared workloads are
// unused here.
#[allow(dead_code)]
mod workloads;

const DEFAULT_WARMUP: u64 = 100;

fn drive<W: Workload>(w: W, warmup: u64, iterations: u64) {
    for _ in 0..warmup {
        drop(black_box(w.run(w.setup())));
    }
    for _ in 0..iterations {
        drop(black_box(w.run(w.setup())));
    }
}

/// Names match the criterion bench ids. Default iteration counts are
/// sized so two runs agree to well under one instruction per iteration
/// on the cheap workloads and to a small fraction of a percent on the
/// rest; growing workloads (the collection ones) also change per-run
/// cost with N, so their counts are only comparable at equal N.
macro_rules! registry {
    ($($name:literal, $iterations:literal => $make:expr;)*) => {
        const WORKLOADS: &[(&str, u64)] = &[$(($name, $iterations)),*];

        fn run(name: &str, warmup: u64, iterations: u64) -> bool {
            match name {
                $($name => drive($make, warmup, iterations),)*
                _ => return false,
            }
            true
        }
    };
}

registry! {
    "chain_local/propagate/4", 1_000_000 => Chain::<Local>::new(4);
    "chain_local/propagate/10", 1_000_000 => Chain::<Local>::new(10);
    "chain_local/propagate/100", 100_000 => Chain::<Local>::new(100);
    "chain_shared/propagate/4", 1_000_000 => Chain::<Shared>::new(4);
    "chain_shared/propagate/10", 1_000_000 => Chain::<Shared>::new(10);
    "chain_shared/propagate/100", 100_000 => Chain::<Shared>::new(100);
    "diamond_local", 1_000_000 => Diamond::<Local>::new();
    "diamond_shared", 1_000_000 => Diamond::<Shared>::new();
    "early_cutoff_local", 1_000_000 => EarlyCutoff::<Local>::new();
    "early_cutoff_shared", 1_000_000 => EarlyCutoff::<Shared>::new();
    "collection_insert_then_read/local_incremental/1000", 100_000 =>
        InsertThenRead::<Local>::pipeline(1_000);
    "collection_insert_then_read/shared_incremental/1000", 100_000 =>
        InsertThenRead::<Shared>::pipeline(1_000);
    "collection_insert_then_read/batch/1000", 10_000 => BatchRebuild::new(1_000);
    "collection_insert_then_read/local_incremental/10000", 100_000 =>
        InsertThenRead::<Local>::pipeline(10_000);
    "collection_insert_then_read/shared_incremental/10000", 100_000 =>
        InsertThenRead::<Shared>::pipeline(10_000);
    "collection_insert_then_read/batch/10000", 1_000 => BatchRebuild::new(10_000);
    "collection_insert_then_read/local_incremental/100000", 100_000 =>
        InsertThenRead::<Local>::pipeline(100_000);
    "collection_insert_then_read/shared_incremental/100000", 100_000 =>
        InsertThenRead::<Shared>::pipeline(100_000);
    "collection_insert_then_read/batch/100000", 1_000 => BatchRebuild::new(100_000);
    "simple_count/local/1000", 100_000 => InsertThenRead::<Local>::simple_count(1_000);
    "simple_count/shared/1000", 100_000 => InsertThenRead::<Shared>::simple_count(1_000);
    "simple_count/local/10000", 100_000 => InsertThenRead::<Local>::simple_count(10_000);
    "simple_count/shared/10000", 100_000 => InsertThenRead::<Shared>::simple_count(10_000);
    "get_clean_local", 10_000_000 => GetClean::<Local>::new();
    "get_clean_shared", 10_000_000 => GetClean::<Shared>::new();
    "get_contended_shared", 2_000 => GetContended::new();
    "fanout_local_4", 100_000 => Fanout::<Local>::new(4);
    "fanout_local_5", 100_000 => Fanout::<Local>::new(5);
    "fanout_local_1000", 1_000 => Fanout::<Local>::new(1000);
    "fanout_shared_4", 100_000 => Fanout::<Shared>::new(4);
    "fanout_shared_5", 100_000 => Fanout::<Shared>::new(5);
    "fanout_shared_1000", 1_000 => Fanout::<Shared>::new(1000);
    "delete_single_local", 200_000 => DeleteSingle::<Local>::new();
    "delete_single_shared", 200_000 => DeleteSingle::<Shared>::new();
    "delete_cascade_local", 1_000 => DeleteCascade::<Local>::new();
    "delete_cascade_shared", 1_000 => DeleteCascade::<Shared>::new();
    "delete_owned_tree_local", 10_000 => DeleteOwnedTree::<Local>::new();
    "delete_owned_tree_shared", 10_000 => DeleteOwnedTree::<Shared>::new();
    "table_hit_local", 10_000_000 => TableHit::<Local>::new();
    "table_hit_shared", 10_000_000 => TableHit::<Shared>::new();
    "table_hit_sip_local", 10_000_000 =>
        TableHit::<Local, std::collections::hash_map::RandomState>::with_hasher(Default::default());
    "table_miss_local", 100_000 => TableMiss::<Local>::new();
    "table_miss_shared", 100_000 => TableMiss::<Shared>::new();
    "value_slot_read_local", 10_000_000 => ValueSlotRead::<Local>::new();
    "value_slot_read_shared", 10_000_000 => ValueSlotRead::<Shared>::new();
    "children_rerun_1k_local", 10_000 => ChildrenRerun::<Local>::new(1_000);
    "children_rerun_1k_shared", 10_000 => ChildrenRerun::<Shared>::new(1_000);
    "children_churn_local", 100_000 => ChildrenChurn::<Local>::new();
    "children_churn_shared", 100_000 => ChildrenChurn::<Shared>::new();
}

fn usage() -> ExitCode {
    eprintln!("usage: icount <workload> <iterations> [warm-up]\n       icount --list");
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--list") {
        for (name, iterations) in WORKLOADS {
            println!("{name} {iterations}");
        }
        return ExitCode::SUCCESS;
    }
    let (name, iterations, warmup) = match args.as_slice() {
        [name, n] => (name, n, None),
        [name, n, w] => (name, n, Some(w)),
        _ => return usage(),
    };
    let Ok(iterations) = iterations.parse::<u64>() else {
        return usage();
    };
    let warmup = match warmup.map(|w| w.parse::<u64>()) {
        None => DEFAULT_WARMUP,
        Some(Ok(w)) => w,
        Some(Err(_)) => return usage(),
    };
    if !run(name, warmup, iterations) {
        eprintln!("icount: unknown workload {name}; see --list");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
