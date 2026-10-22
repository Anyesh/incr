//! Heap bytes a `GridTable` adds beyond its nodes, counted by a global
//! allocator: a column holding one node far down costs about what a
//! column holding one node at the top does, and a dense column costs
//! about one 8 B entry per row.
#![cfg(not(miri))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

use incr_core::{Cells, GridKey, Local, Runtime, Shared};

struct Counting;

static BYTES: AtomicIsize = AtomicIsize::new(0);

// SAFETY: forwards to `System` unchanged; the counter is only
// bookkeeping.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        BYTES.fetch_add(
            new_size as isize - layout.size() as isize,
            Ordering::Relaxed,
        );
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

const COLUMNS: u32 = 256;
const DENSE_ROWS: u32 = 1 << 16;

fn bytes() -> isize {
    BYTES.load(Ordering::Relaxed)
}

/// Bytes per column for one input at `row` in each of `COLUMNS` columns.
fn per_column<C: Cells>(row: u32) -> f64 {
    let rt: Runtime<C> = Runtime::new();
    let t = rt.create_grid_table::<u64>();
    // The sheet's column directory, grown before the count starts.
    t.get_or_insert_input(&rt, GridKey::new(0, 0, COLUMNS), || 0);
    let before = bytes();
    for column in 0..COLUMNS {
        t.get_or_insert_input(&rt, GridKey::new(0, row, column), || 0);
    }
    (bytes() - before) as f64 / f64::from(COLUMNS)
}

/// Bytes per row a dense column adds beyond the same inputs created
/// outside a table.
fn dense_overhead<C: Cells>() -> f64 {
    let plain: Runtime<C> = Runtime::new();
    let before = bytes();
    for _ in 0..DENSE_ROWS {
        plain.create_input(0_u64);
    }
    let nodes = bytes() - before;
    let rt: Runtime<C> = Runtime::new();
    let t = rt.create_grid_table::<u64>();
    let before = bytes();
    for row in 0..DENSE_ROWS {
        t.get_or_insert_input(&rt, GridKey::new(0, row, 0), || 0);
    }
    (bytes() - before - nodes) as f64 / f64::from(DENSE_ROWS)
}

fn grid_columns_are_sparse<C: Cells>() {
    let top = per_column::<C>(0);
    let far = per_column::<C>(1 << 20);
    let dense = dense_overhead::<C>();
    println!(
        "{}: {top:.0} B per column at row 0, {far:.0} B at row 1<<20, \
         {dense:.2} B per row in a dense column",
        std::any::type_name::<C>()
    );
    assert!(
        dense <= 9.0,
        "a dense column costs {dense:.2} B per row beyond its nodes"
    );
    assert!(
        far <= top + 2048.0,
        "a node at row 1<<20 costs {far:.0} B per column, one at row 0 {top:.0} B"
    );
}

// One test, so no other test's allocations land between two snapshots.
#[test]
fn grid_columns_cost_memory_by_their_nodes_not_their_largest_row() {
    grid_columns_are_sparse::<Local>();
    grid_columns_are_sparse::<Shared>();
}
