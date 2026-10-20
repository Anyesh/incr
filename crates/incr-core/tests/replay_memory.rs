//! O(n) memory for a first compute of a 100k-member Fixpoint ring by
//! replay, at budgets 256 and 1: the peak of live heap bytes during the
//! `get`, counted by a global allocator, grows linearly in the ring's size.
// Release only: the same check takes over two minutes in a debug build.
#![cfg(not(any(miri, debug_assertions)))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::{Arc, OnceLock};

use incr_core::{Cells, CycleStrategy, Fixpoint, Incr, Local, QueryOpts, Replay, Runtime, Shared};

struct Counting;

static BYTES: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

fn grew(by: isize) {
    let now = BYTES.fetch_add(by, Ordering::Relaxed) + by;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: forwards to `System` unchanged; the counters are only
// bookkeeping.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grew(layout.size() as isize);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        grew(new_size as isize - layout.size() as isize);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Peak bytes per member above the built ring: about 1000 B measured at
/// 1k to 16k members, Local and Shared, budgets 256 and 1, doubled.
const K: usize = 2048;

fn reclaim<C: Cells>() {
    if std::any::type_name::<C>().ends_with("Shared") {
        haphazard::Domain::global().eager_reclaim();
    }
}

fn ring<C: Cells>(rt: &Runtime<C>, n: usize) -> Vec<Incr<i64>> {
    let members: Arc<OnceLock<Vec<Incr<i64>>>> = Arc::new(OnceLock::new());
    let built = (0..n)
        .map(|k| {
            let members = members.clone();
            rt.create_query_with(
                move |rt| (rt.get(members.get().unwrap()[(k + 1) % n]) + 1).min(5),
                QueryOpts {
                    cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_i64)),
                    replay: Replay::Allowed,
                },
            )
        })
        .collect::<Vec<_>>();
    members.set(built.clone()).unwrap();
    built
}

/// The peak live heap bytes above the built ring during `get(member 0)`
/// of an `n`-member ring at budget `frames`, on an 8 MiB thread. Every
/// member must land Clean on 5: a later `get` returns 5 and traces no
/// compute or verify.
fn peak_of<C: Cells>(n: usize, frames: u32) -> usize {
    std::thread::Builder::new()
        .stack_size(8 << 20)
        .spawn(move || {
            let rt = Runtime::<C>::new();
            rt.set_placeholder(0_i64);
            rt.set_max_frames(frames);
            let members = ring(&rt, n);
            reclaim::<C>();
            let base = BYTES.load(Ordering::Relaxed);
            PEAK.store(base, Ordering::Relaxed);
            assert_eq!(rt.get(members[0]), 5);
            let peak = PEAK.load(Ordering::Relaxed) - base;
            for (k, &m) in members.iter().enumerate() {
                let (v, trace) = rt.get_traced(m);
                assert_eq!(v, 5, "member {k}");
                assert!(trace.node_traces.is_empty(), "member {k} was not Clean");
            }
            peak as usize
        })
        .expect("spawn")
        .join()
        .unwrap_or_else(|e| std::panic::resume_unwind(e))
}

fn linear<C: Cells>(frames: u32) {
    let half = peak_of::<C>(50_000, frames);
    let full = peak_of::<C>(100_000, frames);
    let name = std::any::type_name::<C>();
    assert!(
        full as f64 <= 2.5 * half as f64,
        "{name} at budget {frames}: 100k peak {full} B over 2.5x the 50k peak {half} B"
    );
    assert!(
        full <= K * 100_000,
        "{name} at budget {frames}: 100k peak {full} B over {K} B per member"
    );
}

/// One test, because the counters are process-global.
#[test]
fn fixpoint_ring_replay_memory_is_linear() {
    for frames in [256, 1] {
        linear::<Local>(frames);
        linear::<Shared>(frames);
    }
}
