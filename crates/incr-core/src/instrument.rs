//! Opt-in measurement of computes: the flags `get_traced` and compute
//! stats arm, the per-run timer, and the counters both read.

use std::cell::RefCell;

use std::collections::HashMap;

use crate::fx::FxBuildHasher;
use crate::trace::ComputeCounts;

type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

pub(crate) const TRACE: u8 = 1;
pub(crate) const STATS: u8 = 2;

#[cfg(not(loom))]
thread_local! {
    static NESTED: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

#[cfg(loom)]
loom::thread_local! {
    static NESTED: RefCell<Vec<u64>> = RefCell::new(Vec::new());
}

/// Times one run of a compute closure, net of the runs nested in it on
/// this thread, so a group's time is not counted again in the groups that
/// read it.
pub(crate) struct Timer {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    start: std::time::Instant,
}

impl Timer {
    /// `None` on wasm32-unknown-unknown, where std has no clock.
    #[cold]
    #[inline(never)]
    pub(crate) fn start() -> Option<Timer> {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            NESTED.with(|n| n.borrow_mut().push(0));
            Some(Timer {
                start: std::time::Instant::now(),
            })
        }
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        None
    }

    /// The run's own nanoseconds.
    #[cold]
    #[inline(never)]
    pub(crate) fn stop(self) -> u64 {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            let total = u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            NESTED.with(|n| {
                let mut n = n.borrow_mut();
                let nested = n.pop().unwrap_or(0);
                if let Some(parent) = n.last_mut() {
                    *parent = parent.saturating_add(total);
                }
                total.saturating_sub(nested)
            })
        }
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        0
    }
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
