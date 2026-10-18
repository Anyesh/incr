//! Per-node memory: heap bytes requested per input-plus-query pair, and
//! the number of live allocations, counted by a global allocator.
//!
//! Expected on 64-bit: two 64 B `NodeData`, two 16 B value slots (Shared:
//! an 8 B slot and an 8 B box), two 16 B dependent lists, two 16 B closure
//! entries, a 24 B closure capture (the input handle and the arena `Arc`)
//! and two flag bytes, 250 B per pair.
#![cfg(not(miri))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

use incr_core::{Cells, Incr, Local, Runtime, Shared};

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

#[cfg(target_pointer_width = "64")]
const CEILING: usize = 252;
#[cfg(target_pointer_width = "32")]
const CEILING: usize = 220;

fn snapshot() -> (isize, isize) {
    (BYTES.load(Ordering::Relaxed), LIVE.load(Ordering::Relaxed))
}

fn rss_bytes() -> Option<usize> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4096)
}

fn reclaim<C: Cells>() {
    if std::any::type_name::<C>().ends_with("Shared") {
        haphazard::Domain::global().eager_reclaim();
    }
}

fn pairs_fit_the_ceiling<C: Cells>(closure_allocs: usize) {
    let rt: Runtime<C> = Runtime::new();
    let mut handles: Vec<(Incr<u64>, Incr<u64>)> = Vec::with_capacity(PAIRS);
    let rss0 = rss_bytes();
    let (bytes0, live0) = snapshot();
    for k in 0..PAIRS as u64 {
        let x = rt.create_input(k);
        let q = rt.create_query(move |rt| rt.get(x) + 1);
        assert_eq!(rt.get(q), k + 1);
        handles.push((x, q));
    }
    reclaim::<C>();
    let (bytes1, live1) = snapshot();
    let per_pair = (bytes1 - bytes0) as f64 / PAIRS as f64;
    let allocs = (live1 - live0) as usize;
    let nodes = 2 * PAIRS;
    let alloc_budget = closure_allocs * PAIRS + 8 * nodes.div_ceil(1024) + 64;
    let rss = rss0
        .zip(rss_bytes())
        .map(|(a, b)| (b - a) as f64 / PAIRS as f64);
    println!(
        "{}: {per_pair:.2} B per pair requested, {allocs} live allocations \
         (budget {alloc_budget}), RSS {rss:?} B per pair",
        std::any::type_name::<C>()
    );
    assert!(
        per_pair <= CEILING as f64,
        "{per_pair:.2} B per pair is over the {CEILING} B ceiling"
    );
    assert!(
        allocs <= alloc_budget,
        "{allocs} allocations, budget {alloc_budget}"
    );
    assert_eq!(rt.dependent_list_blocks(), 0);

    let a = handles[0].0;
    let b = handles[1].0;
    let pick = rt.create_input(false);
    let toggled = rt.create_query(move |rt| if rt.get(pick) { rt.get(b) } else { rt.get(a) });
    // A first round, so buffers the first sets grow are in the baseline.
    for i in 0..4 {
        rt.set(pick, i % 2 == 0);
        rt.get(toggled);
    }
    rt.set(pick, false);
    rt.get(toggled);
    reclaim::<C>();
    let before = snapshot().0;
    for i in 0..10_000 {
        rt.set(pick, i % 2 == 0);
        rt.get(toggled);
    }
    rt.set(pick, false);
    rt.get(toggled);
    reclaim::<C>();
    let after = snapshot().0;
    assert_eq!(
        after,
        before,
        "dep toggles changed the byte count by {}",
        after - before
    );
}

// One test, so no other test's allocations land between two snapshots.
#[test]
fn input_query_pairs_fit_the_memory_ceiling() {
    pairs_fit_the_ceiling::<Local>(1);
    pairs_fit_the_ceiling::<Shared>(3);
}
