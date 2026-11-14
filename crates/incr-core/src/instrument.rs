//! Opt-in measurement of computes: the flags `get_traced` and compute
//! stats arm, the per-run timer, and the counters both read.

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::cell::RefCell;
use std::collections::HashMap;

use crate::fx::FxBuildHasher;
use crate::trace::ComputeCounts;

type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

pub(crate) const TRACE: u8 = 1;
pub(crate) const STATS: u8 = 2;

/// Per thread, each open timed run's start and the time of the runs
/// nested in it, so a run's time excludes them and a group's time is not
/// counted again in the groups that read it.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type Open = Vec<(std::time::Instant, u64)>;

#[cfg(all(not(loom), not(all(target_arch = "wasm32", target_os = "unknown"))))]
thread_local! {
    static OPEN: RefCell<Open> = const { RefCell::new(Vec::new()) };
}

#[cfg(all(loom, not(all(target_arch = "wasm32", target_os = "unknown"))))]
loom::thread_local! {
    static OPEN: RefCell<Open> = RefCell::new(Vec::new());
}

/// Opens a timed run; `false` on wasm32-unknown-unknown, where std has
/// no clock, and then [`stop_timer`] must not be called.
#[cold]
#[inline(never)]
pub(crate) fn start_timer() -> bool {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        OPEN.with(|o| o.borrow_mut().push((std::time::Instant::now(), 0)));
        true
    }
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    false
}

/// Closes this thread's innermost timed run and returns its own
/// nanoseconds.
pub(crate) fn stop_timer() -> u64 {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        OPEN.with(|o| {
            let mut o = o.borrow_mut();
            let Some((start, nested)) = o.pop() else {
                return 0;
            };
            let total = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            if let Some(parent) = o.last_mut() {
                parent.1 = parent.1.saturating_add(total);
            }
            total.saturating_sub(nested)
        })
    }
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    0
}

/// What the instrumentation flags have collected. Held under
/// `LockKind::Instrument`, under which nothing is taken.
#[derive(Default)]
pub(crate) struct InstrumentState {
    /// Closure time per slot since the current trace began.
    pub(crate) trace_times: FxHashMap<u32, u64>,
    /// Counts per live slot since stats were last reset.
    pub(crate) counts: FxHashMap<u32, ComputeCounts>,
    /// Counts of released slots, by the label each had.
    pub(crate) released: FxHashMap<Option<String>, ComputeCounts>,
}

impl InstrumentState {
    pub(crate) fn add_time(&mut self, flags: u8, slot: u32, ns: u64) {
        if flags & TRACE != 0 {
            let t = self.trace_times.entry(slot).or_default();
            *t = t.saturating_add(ns);
        }
        if flags & STATS != 0 {
            let c = self.counts.entry(slot).or_default();
            c.elapsed_ns = c.elapsed_ns.saturating_add(ns);
        }
    }

    /// Moves a released slot's counts to its label, so a recycled slot
    /// starts from zero.
    pub(crate) fn release(&mut self, slot: u32, label: Option<String>) {
        self.trace_times.remove(&slot);
        if let Some(counts) = self.counts.remove(&slot) {
            self.released.entry(label).or_default().add(&counts);
        }
    }
}
