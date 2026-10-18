//! The wasm32 half of incr-core's `tests/memory_ceiling.rs`: heap bytes
//! requested per input-plus-query pair under `Local`.
//!
//! Expected: two 64 B `NodeData`, two 16 B value slots, two 8 B dependent
//! lists, two 8 B closure entries, a 24 B closure capture (the input
//! handle, 8-byte aligned by its `RuntimeId(u64)`, and the arena `Arc`)
//! and two flag bytes, 218 B per pair.
#![cfg(target_arch = "wasm32")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

use incr_core::{Incr, Local, Runtime};
use wasm_bindgen_test::wasm_bindgen_test;

struct Counting;

static BYTES: AtomicIsize = AtomicIsize::new(0);
static LIVE: AtomicIsize = AtomicIsize::new(0);

// SAFETY: forwards to `System` unchanged; the counters are only
// bookkeeping.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        LIVE.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        LIVE.fetch_sub(1, Ordering::Relaxed);
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

const PAIRS: usize = 523_776;
const CEILING: f64 = 220.0;

#[wasm_bindgen_test]
fn local_input_query_pairs_fit_the_memory_ceiling() {
    let rt: Runtime<Local> = Runtime::new();
    let mut handles: Vec<(Incr<u64>, Incr<u64>)> = Vec::with_capacity(PAIRS);
    let bytes0 = BYTES.load(Ordering::Relaxed);
    let live0 = LIVE.load(Ordering::Relaxed);
    for k in 0..PAIRS as u64 {
        let x = rt.create_input(k);
        let q = rt.create_query(move |rt| rt.get(x) + 1);
        assert_eq!(rt.get(q), k + 1);
        handles.push((x, q));
    }
    let per_pair = (BYTES.load(Ordering::Relaxed) - bytes0) as f64 / PAIRS as f64;
    let allocs = (LIVE.load(Ordering::Relaxed) - live0) as usize;
    let alloc_budget = PAIRS + 8 * (2 * PAIRS).div_ceil(1024) + 64;
    assert!(
        per_pair <= CEILING,
        "{per_pair:.2} B per pair is over the {CEILING} B ceiling"
    );
    assert!(
        allocs <= alloc_budget,
        "{allocs} allocations, budget {alloc_budget}"
    );
    assert_eq!(rt.dependent_list_blocks(), 0);
}
