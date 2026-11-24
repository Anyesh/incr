use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use super::{MemoCheckpoint, Stage};
use crate::memo::table_core::TableCore;
use crate::memo::FileMemo;
use crate::memo::{
    BuildId, Chain, CheckpointStats, DeltaHeader, EncodedEntry, GenHeader, GenId, Generation,
    IndexItem, Memo, MemoBackend, MemoBuilder, MemoOptions, MemoTable, MemoryMemo, Tombstone,
};
use crate::value_context::wait_probe::{self, WaitProbe};
use crate::{Error, Input, Local, MemoryBackend, Options, Result, Store};

const WAIT: Duration = Duration::from_secs(20);

/// `leaf(k) = k + y` and `top(k) = 10 * leaf(k)` on a Local store.
struct Fx {
    store: Arc<Store<Local, MemoryBackend>>,
    memo: Memo<Local, MemoryBackend>,
    y: Input<u64>,
    w: Input<u64>,
    leaf: MemoTable<u32, u64>,
    top: MemoTable<u32, u64>,
    runs: Arc<AtomicUsize>,
}

impl Fx {
    fn open(log: &MemoryBackend, backend: impl MemoBackend + 'static) -> Fx {
        Fx::with(log, backend, MemoOptions::new(BuildId::Fixed(0x24c)))
    }

    fn with(log: &MemoryBackend, backend: impl MemoBackend + 'static, options: MemoOptions) -> Fx {
        Fx::with_store(log, backend, options, Options::default())
    }

    // A memo opens over an `Arc` of its store, a Local one included.
    #[allow(clippy::arc_with_non_send_sync)]
    fn with_store(
        log: &MemoryBackend,
        backend: impl MemoBackend + 'static,
        options: MemoOptions,
        store_options: Options,
    ) -> Fx {
        let store = Arc::new(Store::<Local, _>::open(log.clone(), store_options).unwrap());
        let y = store.input("y", 1_u64).unwrap();
        let w = store.input("w", 0_u64).unwrap();
        let (yh, wh) = (y.handle(), w.handle());
        let runs = Arc::new(AtomicUsize::new(0));
        let mut builder = MemoBuilder::new(options);
        let leaf = builder.table::<u32, u64>("leaf", 1);
        let top = builder.table::<u32, u64>("top", 1);
        let counted = Arc::clone(&runs);
        builder.define(leaf, move |ctx, &k| {
            counted.fetch_add(1, SeqCst);
            let w = if k < 10 { ctx.rt().get(wh) } else { 0 };
            u64::from(k) + ctx.rt().get(yh) + w
        });
        let counted = Arc::clone(&runs);
        builder.define(top, move |ctx, &k| {
            counted.fetch_add(1, SeqCst);
            10 * ctx.get(leaf, k)
        });
        let memo = builder.open(&store, backend).unwrap();
        Fx {
            store,
            memo,
            y,
            w,
            leaf,
            top,
            runs,
        }
    }

    fn leaf(&self, k: u32) -> u64 {
        self.memo
            .runtime()
            .get(self.memo.get_or_insert(self.leaf, k))
    }

    fn top(&self, k: u32) -> u64 {
        self.memo
            .runtime()
            .get(self.memo.get_or_insert(self.top, k))
    }

    fn set_y(&self, y: u64) {
        self.store.set(self.y, y).unwrap();
    }

    /// Recomputes `leaf(0..10)` alone, leaving no entry stale.
    fn bump_w(&self) {
        let w = self.memo.runtime().get(self.w.handle());
        self.store.set(self.w, w + 1).unwrap();
        for k in 0..10 {
            self.leaf(k);
        }
    }

    fn finish(&self, flight: &MemoCheckpoint) -> CheckpointStats {
        flight.result(&self.memo, Some(WAIT)).unwrap().unwrap()
    }

    /// Where the open flight is: `None` when none is open.
    fn stage(&self) -> Option<&'static str> {
        let slot = self.memo.core().lock_checkpoint();
        slot.open.as_ref().map(|flight| match &flight.stage {
            Stage::Prune(_) => "prune",
            Stage::Inspect { .. } => "inspect",
            Stage::Sealed => "sealed",
            Stage::Finish(_) => "finish",
        })
    }

    /// The shard inspection takes next, and how many of the taken one's
    /// entries are left.
    fn inspect_at(&self) -> Option<(usize, usize)> {
        let slot = self.memo.core().lock_checkpoint();
        match &slot.open.as_ref()?.stage {
            Stage::Inspect { shard, pending } => Some((*shard, pending.len())),
            _ => None,
        }
    }

    fn wait_worker(&self) {
        let slot = self.memo.core().lock_checkpoint();
        let flight = slot.open.as_ref().expect("a flight is open");
        assert!(flight.flight.wait(Some(WAIT)), "the worker gave no verdict");
    }

    /// Polls with no budget, one step a call, until `done` holds.
    fn step_until(&self, flight: &MemoCheckpoint, done: impl Fn(&Fx) -> bool) {
        let started = Instant::now();
        while !done(self) {
            assert!(started.elapsed() < WAIT, "the flight never got there");
            assert!(!flight.poll(&self.memo, Duration::ZERO).unwrap());
        }
    }

    /// Restore counts since open: entries reused or revalidated, and
    /// entries created cold.
    fn restored(&self) -> (u64, u64) {
        let stats = self.memo.stats();
        (stats.reused + stats.revalidated, stats.cold)
    }
}

/// Removes `leaf(k)`'s node, so a later read restores it.
fn forget(fx: &Fx, k: u32) {
    let table: &TableCore<u32, u64, Local> = fx.memo.core().table(fx.leaf.memo, fx.leaf.index);
    table.nodes.remove(fx.memo.runtime(), &k).unwrap();
}

fn shard_of(fx: &Fx, table: MemoTable<u32, u64>, k: u32) -> usize {
    fx.memo.get_or_insert(table, k).erase().slot() as usize % super::SHARDS
}

#[test]
fn a_flight_with_units_running_restores_equal_to_a_control() {
    let (log, backend) = (MemoryBackend::new(), MemoryMemo::new());
    let flown = {
        let fx = Fx::open(&log, backend.clone());
        for k in 0..300 {
            fx.top(k);
        }
        let flight = fx.memo.begin_checkpoint().unwrap();
        let mut k = 300;
        while fx.stage() != Some("sealed") {
            assert!(!flight.poll(&fx.memo, Duration::ZERO).unwrap());
            if k < 350 {
                fx.top(k);
                k += 1;
            }
        }
        fx.bump_w();
        for k in (0..10).chain(k..400) {
            fx.top(k);
        }
        fx.wait_worker();
        fx.finish(&flight)
    };
    assert_eq!((flown.written, flown.fresh_written), (600, 600));
    let fx = Fx::open(&log, backend.clone());
    for k in 0..400 {
        let w = if k < 10 { 1 } else { 0 };
        assert_eq!(fx.top(k), 10 * (u64::from(k) + 1 + w), "top({k})");
    }
    assert_eq!(fx.restored().0, 600, "the flight's entries restore");
    assert_eq!(
        fx.memo.stats().unsaved_entries,
        220,
        "entries made after the begin, and those reading the edited input"
    );
    fx.memo.checkpoint().unwrap();
    drop(fx);
    let fx = Fx::open(&log, backend);
    for k in 0..400 {
        let w = if k < 10 { 1 } else { 0 };
        assert_eq!(fx.top(k), 10 * (u64::from(k) + 1 + w), "top({k})");
    }
    let (hits, cold) = fx.restored();
    assert_eq!(hits + cold, 800, "every entry restored or computed once");
    assert_eq!(cold, 0, "every entry was saved");
}

/// A key whose `top` entry and `leaf` entry fall in shards in the order
/// `first`, then the other.
fn ordered_key(fx: &Fx, top_first: bool) -> u32 {
    (0..256)
        .find(|&k| {
            if top_first {
                let t = shard_of(fx, fx.top, k);
                t < shard_of(fx, fx.leaf, k)
            } else {
                let l = shard_of(fx, fx.leaf, k);
                l < shard_of(fx, fx.top, k)
            }
        })
        .expect("a key in each shard order")
}

/// Inspects the first of `k`'s two entries, takes the second's shard,
/// recomputes with `y = 2` before the second is inspected, and finishes. A dependent whose dep was inspected at
/// another stamp must not be stored paired with that dep's value.
fn sibling_recomputes_between_inspections(top_first: bool) {
    let (log, backend) = (MemoryBackend::new(), MemoryMemo::without_deltas());
    let k;
    {
        let fx = Fx::open(&log, backend.clone());
        k = ordered_key(&fx, top_first);
        fx.top(k);
        let second = shard_of(&fx, if top_first { fx.leaf } else { fx.top }, k);
        let flight = fx.memo.begin_checkpoint().unwrap();
        fx.step_until(&flight, |fx| fx.inspect_at() == Some((second + 1, 1)));
        fx.set_y(2);
        if top_first {
            fx.leaf(k);
        } else {
            fx.top(k);
        }
        fx.finish(&flight);
        if !top_first {
            fx.set_y(1);
        }
    }
    let fx = Fx::open(&log, backend);
    let y = if top_first { 2 } else { 1 };
    assert_eq!(fx.leaf(k), u64::from(k) + y);
    assert_eq!(fx.top(k), 10 * (u64::from(k) + y));
}

#[test]
fn a_dep_recomputed_after_its_dependent_was_inspected_is_not_paired_with_it() {
    sibling_recomputes_between_inspections(true);
}

#[test]
fn a_dependent_recomputed_after_its_dep_was_inspected_is_not_paired_with_it() {
    sibling_recomputes_between_inspections(false);
}

#[test]
fn a_dependent_left_unpaired_stays_dirty_and_is_saved_later() {
    let (log, backend) = (MemoryMemo::without_deltas(), MemoryBackend::new());
    let k;
    {
        let fx = Fx::open(&backend, log.clone());
        k = ordered_key(&fx, true);
        fx.top(k);
        let second = shard_of(&fx, fx.leaf, k);
        let flight = fx.memo.begin_checkpoint().unwrap();
        fx.step_until(&flight, |fx| fx.inspect_at() == Some((second + 1, 1)));
        fx.set_y(2);
        fx.leaf(k);
        fx.finish(&flight);
        assert!(fx.memo.stats().unsaveable >= 1);
        fx.top(k);
        let later = fx.memo.checkpoint().unwrap();
        assert!(later.fresh_written >= 1, "{later:?}");
    }
    let fx = Fx::open(&backend, log);
    assert_eq!(fx.top(k), 10 * (u64::from(k) + 2));
    assert_eq!(fx.restored(), (2, 0));
}

#[test]
fn inspection_waits_while_an_unlogged_scope_is_open() {
    let (log, backend) = (MemoryBackend::new(), MemoryMemo::new());
    {
        let fx = Fx::open(&log, backend.clone());
        for k in 0..64 {
            fx.top(k);
        }
        let flight = fx.memo.begin_checkpoint().unwrap();
        fx.step_until(&flight, |fx| {
            fx.inspect_at().is_some_and(|(shard, _)| shard >= 4)
        });
        let at = fx.inspect_at();
        {
            let _scope = fx.store.unlogged_scope().unwrap();
            for _ in 0..8 {
                assert!(!flight.poll(&fx.memo, Duration::from_millis(1)).unwrap());
            }
            assert_eq!(fx.inspect_at(), at, "inspected inside a scope");
            assert!(matches!(
                flight.result(&fx.memo, Some(WAIT)),
                Err(Error::UnloggedScopeOpen)
            ));
            _scope.close().unwrap();
        }
        assert_eq!(fx.finish(&flight).fresh_written, 128);
    }
    let fx = Fx::open(&log, backend);
    for k in 0..64 {
        assert_eq!(fx.top(k), 10 * (u64::from(k) + 1));
    }
    assert_eq!(fx.restored(), (128, 0));
}

/// Computes 100 entries, begins, and between inspection steps computes 20
/// new ones and recomputes 10 of the first: those 30 are unsaved after.
fn computes_during_inspection(fx: &Fx, flight: &MemoCheckpoint) {
    fx.step_until(flight, |fx| {
        fx.inspect_at().is_some_and(|(shard, _)| shard >= 8)
    });
    for k in 100..120 {
        fx.leaf(k);
    }
    fx.bump_w();
}

fn unsaved_is_exact(fx: &Fx) -> u64 {
    let unsaved = fx.memo.stats().unsaved_entries;
    assert_eq!(unsaved, fx.memo.core().dirty.above_attempt());
    unsaved
}

#[test]
fn unsaved_counts_exactly_the_entries_computed_since_begin() {
    let fx = Fx::open(&MemoryBackend::new(), MemoryMemo::new());
    for k in 0..100 {
        fx.leaf(k);
    }
    let flight = fx.memo.begin_checkpoint().unwrap();
    computes_during_inspection(&fx, &flight);
    assert_eq!(unsaved_is_exact(&fx), 30);
    let written = fx.finish(&flight).fresh_written;
    assert!((90..=100).contains(&written), "{written} written");
    assert_eq!(unsaved_is_exact(&fx), 30);
    assert_eq!(fx.memo.checkpoint().unwrap().fresh_written, 30);
    assert_eq!(unsaved_is_exact(&fx), 0);
}

#[test]
fn a_cancelled_flight_leaves_every_dirty_entry_unsaved() {
    let fx = Fx::open(&MemoryBackend::new(), MemoryMemo::new());
    for k in 0..100 {
        fx.leaf(k);
    }
    let flight = fx.memo.begin_checkpoint().unwrap();
    computes_during_inspection(&fx, &flight);
    flight.cancel();
    assert!(matches!(
        flight.poll(&fx.memo, Duration::ZERO),
        Err(Error::Cancelled)
    ));
    assert_eq!(unsaved_is_exact(&fx), 120);
    assert_eq!(fx.memo.checkpoint().unwrap().fresh_written, 120);
    assert_eq!(unsaved_is_exact(&fx), 0);
}

/// Stores `top(0..100)` and 10,000 leaves, then reopens with every leaf
/// restored and so seeded and the last 5,000 removed, so the seed maps
/// hold 5,000 dead entries and the next checkpoint prunes in three steps.
fn seeded(log: &MemoryBackend, backend: &MemoryMemo) -> Fx {
    {
        let fx = Fx::open(log, backend.clone());
        for k in 0..10_000 {
            fx.leaf(k);
        }
        for k in 0..100 {
            fx.top(k);
        }
        fx.memo.checkpoint().unwrap();
    }
    let fx = Fx::open(log, backend.clone());
    for k in 0..10_000 {
        fx.leaf(k);
    }
    assert_eq!(fx.restored(), (10_000, 0));
    for k in 5_000..10_000 {
        forget(&fx, k);
    }
    fx
}

fn seeds(fx: &Fx) -> (usize, usize) {
    let state = fx.memo.core().lock_memo();
    (state.seed_info.len(), state.seed_keys.len())
}

/// Live leaves `0..5_000` whose seed is in both maps.
fn live_seeds(fx: &Fx) -> usize {
    let state = fx.memo.core().lock_memo();
    (0..5_000)
        .filter(|k| {
            let handle = fx.memo.get(fx.leaf, k).unwrap().erase();
            state.seed_info.contains_key(&handle) && state.seed_keys.contains_key(&handle)
        })
        .count()
}

#[test]
fn a_lookup_in_the_middle_of_the_prune_falls_back_and_the_prune_keeps_live_seeds() {
    let (log, backend) = (MemoryBackend::new(), MemoryMemo::new());
    let fx = seeded(&log, &backend);
    let flight = fx.memo.begin_checkpoint().unwrap();
    assert!(!flight.poll(&fx.memo, Duration::ZERO).unwrap());
    assert_eq!(fx.stage(), Some("prune"));
    let moved = seeds(&fx).0;
    assert!(0 < moved && moved < 5_000, "{moved} seeds moved");
    let runs = fx.runs.load(SeqCst);
    for k in 0..100 {
        assert_eq!(fx.top(k), 10 * (u64::from(k) + 1));
    }
    assert!(fx.runs.load(SeqCst) - runs < 100, "every restore ran");
    fx.finish(&flight);
    assert_eq!(live_seeds(&fx), 5_000);
    assert_eq!(seeds(&fx), (5_000 + 100, 5_000 + 100));
}

#[test]
fn a_flight_abandoned_in_the_middle_of_the_prune_loses_no_seed() {
    let (log, backend) = (MemoryBackend::new(), MemoryMemo::new());
    let fx = seeded(&log, &backend);
    let flight = fx.memo.begin_checkpoint().unwrap();
    assert!(!flight.poll(&fx.memo, Duration::ZERO).unwrap());
    assert_eq!(fx.stage(), Some("prune"));
    assert!(live_seeds(&fx) < 5_000);
    drop(flight);
    fx.wait_worker();
    assert!(fx.memo.checkpoint_if_due().unwrap().is_none());
    assert!(fx.stage().is_none());
    assert_eq!(live_seeds(&fx), 5_000);
}

/// Fails its writes while `failing` is set; with a `probe`, its first write
/// records there whether the caller waits for it inside the value scope's
/// `wait`.
#[derive(Clone)]
struct Failing {
    inner: MemoryMemo,
    failing: Arc<AtomicBool>,
    probe: Option<Arc<WaitProbe>>,
    flagged: bool,
}

impl Failing {
    fn new() -> Self {
        Failing {
            inner: MemoryMemo::new(),
            failing: Arc::new(AtomicBool::new(true)),
            probe: None,
            flagged: true,
        }
    }

    fn check(&self) -> Result<()> {
        if let Some(probe) = &self.probe {
            probe.record(WAIT);
        }
        if self.failing.load(SeqCst) {
            Err(Error::ReadOnly)
        } else {
            Ok(())
        }
    }
}

impl MemoBackend for Failing {
    type Guard = <MemoryMemo as MemoBackend>::Guard;

    fn lock_checkpoint(&self) -> Result<Self::Guard> {
        self.inner.lock_checkpoint()
    }

    fn load(&self) -> Result<Option<Generation>> {
        self.inner.load()
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        self.inner.read_entry(generation, item)
    }

    fn write_generation(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &[EncodedEntry],
    ) -> Result<Generation> {
        self.check()?;
        self.inner.write_generation(base, id, header, entries)
    }

    fn write_generation_streamed(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &mut dyn Iterator<Item = Result<EncodedEntry>>,
    ) -> Result<Generation> {
        self.check()?;
        self.inner
            .write_generation_streamed(base, id, header, entries)
    }

    fn read_entries(&self, generation: GenId, items: &[IndexItem]) -> Result<Vec<Option<Vec<u8>>>> {
        self.inner.read_entries(generation, items)
    }

    fn supports_delta(&self) -> bool {
        self.inner.supports_delta()
    }

    fn concurrent_compaction(&self) -> bool {
        self.flagged
    }

    fn load_chain(&self, known: &[Arc<Generation>]) -> Result<Option<Chain>> {
        self.inner.load_chain(known)
    }

    fn write_delta(
        &self,
        tip: GenId,
        id: GenId,
        header: &DeltaHeader,
        entries: &[EncodedEntry],
        tombstones: &[Tombstone],
    ) -> Result<Generation> {
        self.check()?;
        self.inner.write_delta(tip, id, header, entries, tombstones)
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        self.inner.remove_generation(generation)
    }
}

/// How a test lets go of a flight's handle.
#[derive(Clone, Copy, Debug)]
enum Dropped {
    AfterCommit,
    InTheFinish,
    AfterAWorkerError,
    OnAnotherThread,
}

/// 5,000 dirty entries, a flight dropped as `how`, then 10 new entries
/// and 10 recomputes: the next `checkpoint()` settles the flight and
/// writes every dirty entry, `unsaved` is exact throughout, and a reopen
/// restores every entry.
fn dropped_handle(how: Dropped) {
    let (log, backend) = (MemoryBackend::new(), Failing::new());
    backend
        .failing
        .store(matches!(how, Dropped::AfterAWorkerError), SeqCst);
    {
        let fx = Fx::open(&log, backend.clone());
        for k in 0..5_000 {
            fx.leaf(k);
        }
        let flight = fx.memo.begin_checkpoint().unwrap();
        fx.step_until(&flight, |fx| fx.stage() == Some("sealed"));
        fx.wait_worker();
        match how {
            Dropped::InTheFinish => {
                assert!(!flight.poll(&fx.memo, Duration::ZERO).unwrap());
                assert_eq!(fx.stage(), Some("finish"));
                drop(flight);
            }
            Dropped::OnAnotherThread => thread::spawn(move || drop(flight)).join().unwrap(),
            Dropped::AfterCommit | Dropped::AfterAWorkerError => drop(flight),
        }
        backend.failing.store(false, SeqCst);
        for k in 5_000..5_010 {
            fx.leaf(k);
        }
        fx.bump_w();
        assert_eq!(unsaved_is_exact(&fx), 20, "{how:?}");
        fx.memo.checkpoint().unwrap();
        assert!(fx.stage().is_none());
        assert_eq!(unsaved_is_exact(&fx), 0, "{how:?}");
    }
    let fx = Fx::open(&log, backend);
    for k in 0..5_010 {
        let w = if k < 10 { 1 } else { 0 };
        assert_eq!(fx.leaf(k), u64::from(k) + 1 + w, "{how:?}");
    }
    assert_eq!(fx.restored(), (5_010, 0), "{how:?}");
}

#[test]
fn a_handle_dropped_after_the_commit_is_settled_by_the_next_checkpoint() {
    dropped_handle(Dropped::AfterCommit);
}

#[test]
fn a_handle_dropped_in_the_finish_is_settled_by_the_next_checkpoint() {
    dropped_handle(Dropped::InTheFinish);
}

#[test]
fn a_handle_dropped_after_a_worker_error_is_settled_by_the_next_checkpoint() {
    dropped_handle(Dropped::AfterAWorkerError);
}

#[test]
fn a_handle_freed_on_another_thread_is_settled_by_the_next_checkpoint() {
    dropped_handle(Dropped::OnAnotherThread);
}

#[test]
fn an_edit_after_the_commit_and_before_the_next_poll_settles_nothing() {
    let options = MemoOptions::new(BuildId::Fixed(0x24c)).checkpoint_after_entries(1);
    let fx = Fx::with(&MemoryBackend::new(), MemoryMemo::new(), options);
    for k in 0..100 {
        fx.leaf(k);
    }
    let flight = fx.memo.begin_checkpoint().unwrap();
    fx.step_until(&flight, |fx| fx.stage() == Some("sealed"));
    fx.wait_worker();
    fx.leaf(1_000);
    assert!(fx.memo.checkpoint_if_due().unwrap().is_none());
    assert!(matches!(fx.memo.checkpoint(), Err(Error::Busy)));
    assert_eq!(fx.stage(), Some("sealed"));
    assert_eq!(fx.finish(&flight).fresh_written, 100);
    assert_eq!(unsaved_is_exact(&fx), 1);
}

#[test]
fn a_flight_cancelled_after_the_commit_keeps_its_generation_without_its_finish() {
    let fx = Fx::open(&MemoryBackend::new(), MemoryMemo::new());
    for k in 0..100 {
        fx.leaf(k);
    }
    let flight = fx.memo.begin_checkpoint().unwrap();
    fx.step_until(&flight, |fx| fx.stage() == Some("sealed"));
    fx.wait_worker();
    flight.cancel();
    assert_eq!(fx.memo.checkpoint().unwrap().fresh_written, 100);
    assert_eq!(
        fx.memo.stats().checkpoints,
        2,
        "the flight's generation counts"
    );
    assert_eq!(unsaved_is_exact(&fx), 0);
    assert!(matches!(
        flight.poll(&fx.memo, Duration::ZERO),
        Err(Error::Cancelled)
    ));
}

#[test]
fn an_edit_between_finish_slices_runs_no_finish_work() {
    let options = MemoOptions::new(BuildId::Fixed(0x24c)).checkpoint_after_entries(1);
    let fx = Fx::with(&MemoryBackend::new(), MemoryMemo::new(), options);
    for k in 0..10_000 {
        fx.leaf(k);
    }
    let flight = fx.memo.begin_checkpoint().unwrap();
    fx.step_until(&flight, |fx| fx.stage() == Some("sealed"));
    fx.wait_worker();
    assert!(!flight.poll(&fx.memo, Duration::ZERO).unwrap());
    let finished = || {
        let slot = fx.memo.core().lock_checkpoint();
        match &slot.open.as_ref().unwrap().stage {
            Stage::Finish(finishing) => finishing.next,
            _ => panic!("not finishing"),
        }
    };
    let at = finished();
    assert!(at < 10_000);
    let runs = fx.runs.load(SeqCst);
    fx.leaf(20_000);
    assert_eq!(fx.runs.load(SeqCst), runs + 1);
    assert!(fx.memo.checkpoint_if_due().unwrap().is_none());
    assert!(matches!(fx.memo.checkpoint(), Err(Error::Busy)));
    assert_eq!(finished(), at);
    fx.finish(&flight);
    assert_eq!(fx.memo.stats().unsaved_entries, 1);
}

#[test]
fn a_flight_abandoned_after_the_commit_keeps_reads_working_and_removes_what_it_superseded() {
    let (log, backend) = (MemoryBackend::new(), MemoryMemo::without_deltas());
    let fx = Fx::open(&log, backend.clone());
    for k in 0..200 {
        fx.leaf(k);
    }
    fx.memo.checkpoint().unwrap();
    let old = fx.memo.core().index().layers[0].id();
    let item = *fx.memo.core().index().layers[0].items().first().unwrap();
    fx.leaf(200);
    let flight = fx.memo.begin_checkpoint().unwrap();
    fx.step_until(&flight, |fx| fx.stage() == Some("sealed"));
    fx.wait_worker();
    drop(flight);
    assert!(fx.memo.checkpoint_if_due().unwrap().is_none());
    assert!(fx.stage().is_none());
    assert_ne!(fx.memo.core().index().layers[0].id(), old);
    let started = Instant::now();
    while backend.read_entry(old, &item).unwrap().is_some() {
        assert!(started.elapsed() < WAIT, "the superseded generation stayed");
        thread::sleep(Duration::from_millis(5));
    }
    for k in 0..201 {
        forget(&fx, k);
    }
    let runs = fx.runs.load(SeqCst);
    for k in 0..201 {
        assert_eq!(fx.leaf(k), u64::from(k) + 1);
    }
    assert_eq!(fx.runs.load(SeqCst), runs, "reads restored every entry");
}

/// Takes one entry of a streamed write and returns `Ok`.
#[derive(Clone)]
struct StopsEarly(MemoryMemo);

impl MemoBackend for StopsEarly {
    type Guard = <MemoryMemo as MemoBackend>::Guard;

    fn lock_checkpoint(&self) -> Result<Self::Guard> {
        self.0.lock_checkpoint()
    }

    fn load(&self) -> Result<Option<Generation>> {
        self.0.load()
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        self.0.read_entry(generation, item)
    }

    fn write_generation(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &[EncodedEntry],
    ) -> Result<Generation> {
        self.0.write_generation(base, id, header, entries)
    }

    fn write_generation_streamed(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &mut dyn Iterator<Item = Result<EncodedEntry>>,
    ) -> Result<Generation> {
        let first: Vec<_> = entries.take(1).collect::<Result<_>>()?;
        self.0.write_generation(base, id, header, &first)
    }

    fn read_entries(&self, generation: GenId, items: &[IndexItem]) -> Result<Vec<Option<Vec<u8>>>> {
        self.0.read_entries(generation, items)
    }

    fn concurrent_compaction(&self) -> bool {
        true
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        self.0.remove_generation(generation)
    }
}

#[test]
fn a_backend_that_stops_pulling_early_fails_the_flight() {
    let fx = Fx::open(&MemoryBackend::new(), StopsEarly(MemoryMemo::new()));
    for k in 0..100 {
        fx.leaf(k);
    }
    let flight = fx.memo.begin_checkpoint().unwrap();
    assert!(matches!(
        flight.result(&fx.memo, Some(WAIT)),
        Err(Error::TruncatedWrite)
    ));
    assert_eq!(fx.memo.stats().unsaved_entries, 100);
    assert!(fx.stage().is_none());
}

#[test]
fn result_waits_for_the_worker_inside_the_scope_wait() {
    let probe = Arc::new(WaitProbe::default());
    let backend = Failing {
        probe: Some(Arc::clone(&probe)),
        ..Failing::new()
    };
    backend.failing.store(false, SeqCst);
    let store_options = Options {
        value_scope: Some(wait_probe::counting_waits()),
        ..Options::default()
    };
    let fx = Fx::with_store(
        &MemoryBackend::new(),
        backend,
        MemoOptions::new(BuildId::Fixed(0x24c)),
        store_options,
    );
    for k in 0..100 {
        fx.leaf(k);
    }
    let flight = fx.memo.begin_checkpoint().unwrap();
    wait_probe::watch(Some(Arc::clone(&probe)));
    let stats = flight.result(&fx.memo, Some(WAIT));
    let waits = wait_probe::waits();
    wait_probe::watch(None);
    assert!(waits > 0, "result entered the scope's wait");
    assert_eq!(
        probe.seen(),
        Some(true),
        "the worker finished while result waited inside the scope's wait"
    );
    assert!(matches!(stats, Ok(Some(_))), "{stats:?}");
}

#[test]
fn a_flight_is_refused_while_another_runs_and_on_a_backend_without_the_flag() {
    let fx = Fx::open(&MemoryBackend::new(), MemoryMemo::new());
    fx.leaf(1);
    let flight = fx.memo.begin_checkpoint().unwrap();
    assert!(matches!(fx.memo.begin_checkpoint(), Err(Error::Busy)));
    assert!(matches!(fx.memo.checkpoint(), Err(Error::Busy)));
    fx.finish(&flight);
    assert!(flight.poll(&fx.memo, Duration::ZERO).unwrap());
    fx.memo.checkpoint().unwrap();
    let unflagged = Failing {
        flagged: false,
        ..Failing::new()
    };
    let other = Fx::open(&MemoryBackend::new(), unflagged);
    assert!(matches!(
        other.memo.begin_checkpoint(),
        Err(Error::ConcurrentCompactionUnsupported)
    ));
    assert!(matches!(
        flight.poll(&other.memo, Duration::ZERO),
        Err(Error::UnknownHandle)
    ));
}

const CHILD_DIR: &str = "INCR_MEMO_FLIGHT_CHILD_DIR";

/// In the child: notes that the worker is writing, then waits to be
/// killed.
fn parked_for_good() {
    let dir = std::env::var(CHILD_DIR).unwrap();
    std::fs::write(std::path::Path::new(&dir).join("parked"), b"").unwrap();
    loop {
        thread::park();
    }
}

/// The child checkpoints 100 entries under build 1, then begins a flight
/// under build 2, which writes a full generation, and is killed while the
/// worker pulls it. The memo reopens on build 1's chain and restores it.
#[test]
fn a_child_killed_while_the_worker_writes_reopens_on_the_previous_chain() {
    let build = |id| MemoOptions::new(BuildId::Fixed(id));
    if let Ok(dir) = std::env::var(CHILD_DIR) {
        let dir = std::path::Path::new(&dir);
        let previous = {
            let fx = Fx::with(
                &MemoryBackend::new(),
                FileMemo::open(dir.join("memo")).unwrap(),
                build(1),
            );
            for k in 0..100 {
                fx.leaf(k);
            }
            fx.memo.checkpoint().unwrap();
            fx.memo.core().index().layers[0].id()
        };
        std::fs::write(dir.join("previous"), previous.0.to_string()).unwrap();
        let fx = Fx::with(
            &MemoryBackend::new(),
            FileMemo::open(dir.join("memo")).unwrap(),
            build(2),
        );
        for k in 0..100 {
            fx.leaf(k);
        }
        super::super::checkpoint::WRITE_PARK
            .set(parked_for_good)
            .unwrap();
        let flight = fx.memo.begin_checkpoint().unwrap();
        let _ = flight.result(&fx.memo, None);
        unreachable!("the worker parks for good");
    }
    let dir = tempfile::tempdir().unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "memo::flight::tests::a_child_killed_while_the_worker_writes_reopens_on_the_previous_chain",
            "--exact",
            "--test-threads=1",
        ])
        .env(CHILD_DIR, dir.path())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let started = Instant::now();
    while !dir.path().join("parked").exists() {
        if let Some(status) = child.try_wait().unwrap() {
            panic!("the child exited before the write park: {status}");
        }
        assert!(
            started.elapsed() < WAIT,
            "the child never reached the write park"
        );
        thread::sleep(Duration::from_millis(10));
    }
    child.kill().unwrap();
    child.wait().unwrap();
    let previous: u64 = std::fs::read_to_string(dir.path().join("previous"))
        .unwrap()
        .parse()
        .unwrap();
    let memo = FileMemo::open(dir.path().join("memo")).unwrap();
    let chain = memo.load_chain(&[]).unwrap().unwrap();
    assert_eq!(chain.tip().id(), GenId(previous));
    let fx = Fx::with(&MemoryBackend::new(), memo, build(1));
    for k in 0..100 {
        assert_eq!(fx.leaf(k), u64::from(k) + 1);
    }
    assert_eq!(fx.restored(), (100, 0));
}

#[test]
fn the_reference_memo_backends_declare_concurrent_compaction() {
    assert!(MemoryMemo::new().concurrent_compaction());
    let dir = tempfile::tempdir().unwrap();
    assert!(FileMemo::open(dir.path()).unwrap().concurrent_compaction());
}

/// Reports how long the one-step settle of an abandoned flight takes over
/// 531,000 dirty marks, the client's fresh entries at L scale.
#[test]
#[ignore = "a timing at scale, run in release"]
fn settle_of_an_abandoned_flight_at_531k_marks() {
    let fx = Fx::open(&MemoryBackend::new(), MemoryMemo::new());
    for k in 0..531_000 {
        fx.leaf(k);
    }
    drop(fx.memo.begin_checkpoint().unwrap());
    fx.wait_worker();
    let started = Instant::now();
    assert!(fx
        .memo
        .core()
        .settle_idle(&mut fx.memo.core().lock_checkpoint()));
    let took = started.elapsed();
    assert_eq!(unsaved_is_exact(&fx), 531_000);
    eprintln!("settled an abandoned flight over 531,000 marks in {took:?}");
}
