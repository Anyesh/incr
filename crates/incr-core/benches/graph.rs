//! Clean reads, fan-out first computes and deletes: the per-operation
//! costs the spreadsheet-grade tasks gate on. Setup for the fan-out and
//! delete benches runs per iteration, untimed; see `workloads`.

use criterion::{criterion_group, criterion_main, Criterion};
use incr_core::{Cells, Local, Shared};
use workloads::{
    ChildrenChurn, ChildrenRerun, DeleteCascade, DeleteOwnedTree, DeleteSingle, Fanout, GetClean,
    GetContended, TableHit, TableMiss, ValueSlotRead, Workload,
};

// Allowed because each bench target uses only its own slice of the
// shared workloads.
#[allow(dead_code)]
mod workloads;

fn bench_get(c: &mut Criterion) {
    let local = GetClean::<Local>::new();
    c.bench_function("get_clean_local", |b| workloads::steady(b, &local));
    let shared = GetClean::<Shared>::new();
    c.bench_function("get_clean_shared", |b| workloads::steady(b, &shared));
    let contended = GetContended::new();
    c.bench_function("get_contended_shared", |b| workloads::steady(b, &contended));
}

/// Table hits and misses beside the value-slot read the hit gate
/// compares against, and the Fx hasher against std's SipHash.
fn bench_table(c: &mut Criterion) {
    let hit = TableHit::<Local>::new();
    c.bench_function("table_hit_local", |b| workloads::steady(b, &hit));
    let hit = TableHit::<Shared>::new();
    c.bench_function("table_hit_shared", |b| workloads::steady(b, &hit));
    let sip =
        TableHit::<Local, std::collections::hash_map::RandomState>::with_hasher(Default::default());
    c.bench_function("table_hit_sip_local", |b| workloads::steady(b, &sip));
    let miss = TableMiss::<Local>::new();
    c.bench_function("table_miss_local", |b| workloads::steady(b, &miss));
    let miss = TableMiss::<Shared>::new();
    c.bench_function("table_miss_shared", |b| workloads::steady(b, &miss));
    let read = ValueSlotRead::<Local>::new();
    c.bench_function("value_slot_read_local", |b| workloads::steady(b, &read));
    let read = ValueSlotRead::<Shared>::new();
    c.bench_function("value_slot_read_shared", |b| workloads::steady(b, &read));
}

/// A creator's rerun over a thousand children it reuses, and a rerun
/// that creates one child and sweeps one.
fn bench_children(c: &mut Criterion) {
    let rerun = ChildrenRerun::<Local>::new(1_000);
    c.bench_function("children_rerun_1k_local", |b| workloads::steady(b, &rerun));
    let rerun = ChildrenRerun::<Shared>::new(1_000);
    c.bench_function("children_rerun_1k_shared", |b| workloads::steady(b, &rerun));
    let churn = ChildrenChurn::<Local>::new();
    c.bench_function("children_churn_local", |b| workloads::steady(b, &churn));
    let churn = ChildrenChurn::<Shared>::new();
    c.bench_function("children_churn_shared", |b| workloads::steady(b, &churn));
}

fn bench_fanout_with<C: Cells>(c: &mut Criterion, strategy: &str) {
    for width in [4, 5, 1000] {
        let w = Fanout::<C>::new(width);
        c.bench_function(&format!("fanout_{strategy}_{width}"), |b| {
            workloads::batched(b, &w)
        });
    }
}

fn bench_fanout(c: &mut Criterion) {
    bench_fanout_with::<Local>(c, "local");
    bench_fanout_with::<Shared>(c, "shared");
}

fn bench_batched<W: Workload>(c: &mut Criterion, name: &str, w: W) {
    c.bench_function(name, |b| workloads::batched(b, &w));
}

fn bench_delete_with<C: Cells>(c: &mut Criterion, strategy: &str) {
    bench_batched(
        c,
        &format!("delete_single_{strategy}"),
        DeleteSingle::<C>::new(),
    );
    bench_batched(
        c,
        &format!("delete_cascade_{strategy}"),
        DeleteCascade::<C>::new(),
    );
    bench_batched(
        c,
        &format!("delete_owned_tree_{strategy}"),
        DeleteOwnedTree::<C>::new(),
    );
}

fn bench_delete(c: &mut Criterion) {
    bench_delete_with::<Local>(c, "local");
    bench_delete_with::<Shared>(c, "shared");
}

criterion_group!(
    graph_benches,
    bench_get,
    bench_table,
    bench_children,
    bench_fanout,
    bench_delete
);
criterion_main!(graph_benches);
