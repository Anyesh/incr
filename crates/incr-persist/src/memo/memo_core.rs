use std::any::Any;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, RwLock};
use std::time::{Duration, Instant};

use incr_core::lock_order::{LockKind, Ordered};
use incr_core::persist_hooks::changed_at;
use incr_core::{Cells, ErasedIncr, Incr, Local, NodeTable, Runtime, Shared};

use super::dirty::Dirty;

use super::checkpoint::write_known;
use super::layers::{self, Live, Probe};
use super::table_core::{ErasedTable, KeyPart};
use super::{
    key_hash, Chain, CheckpointInput, Checkpointed, Entry, GenId, Generation, IndexItem,
    MemoBackend, TableDesc,
};
use crate::store::dep_index::{DepIndex, External};
use crate::value_context::{EncodeMismatch, ValueScope};
use crate::Result;

/// Restores and checkpoints by count since the memo opened, and what the
/// backend holds and lacks now.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoStats {
    /// Entries in the current generation.
    pub entries: u64,
    /// The current generation's summed entry frame lengths, the size
    /// `max_bytes` bounds.
    pub bytes: u64,
    /// Entries computed since the last checkpoint attempt that no
    /// checkpoint has saved, and clean entries a checkpoint found missing
    /// from the base it replaced. Entries left dirty by an earlier
    /// checkpoint, unsaveable or evicted, are not counted until they
    /// compute again.
    pub unsaved_entries: u64,
    /// Entries the last checkpoint found but could not save: a dep that
    /// maps to nothing or to an entry not kept, a cycle group, a key or
    /// value that does not round-trip, or a frame too large. They stay
    /// dirty, so every checkpoint tries them again.
    pub unsaveable: u64,
    /// Entries the last checkpoint evicted to fit `max_bytes`. Live ones
    /// are dirty again, and a reopen recomputes the others.
    pub evicted_last_checkpoint: u64,
    /// Entries seeded to verify without running when their deps are
    /// unchanged.
    pub reused: u64,
    /// Entries seeded to run once and cut off if their value is unchanged.
    pub revalidated: u64,
    /// Stored entries created cold instead: an unknown or untrusted dep, a
    /// query dep that could not be restored, a dep deleted before the seed,
    /// or a corrupt frame.
    pub cold: u64,
    /// Frames, keys or values that failed to decode, a subset of `cold`.
    pub corrupt: u64,
    /// Checkpoints that wrote a generation.
    pub checkpoints: u64,
    /// Automatic checkpoints that failed with an error other than `Busy` or
    /// an open unlogged scope or staged batch, or panicked.
    pub auto_checkpoint_errors: u64,
    /// Generations a checkpoint replaced but could not remove.
    pub stale_generations: u64,
    /// Deltas in the current chain.
    pub chain_depth: u64,
    /// The summed entry frame lengths of the current chain's deltas.
    pub chain_bytes: u64,
    /// Checkpoints that wrote a delta, a part of `checkpoints`.
    pub delta_checkpoints: u64,
    /// Checkpoints whose full write replaced a chain holding deltas.
    pub compactions: u64,
    /// [`CheckpointStats::bytes_appended`](super::CheckpointStats::bytes_appended)
    /// of the last checkpoint that succeeded, 0 when it found nothing
    /// fresh.
    pub last_bytes_appended: u64,
}

/// [`MemoBackend`] with its guard type erased.
pub(crate) trait ErasedBackend: Send + Sync {
    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>>;
    /// `known` holds the layers this handle's index already loaded.
    fn checkpoint(&self, input: CheckpointInput, known: &[Arc<Generation>])
        -> Result<Checkpointed>;
    fn remove_generations(&self, generations: &[GenId]) -> Vec<GenId>;
}

impl<M: MemoBackend> ErasedBackend for M {
    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        MemoBackend::read_entry(self, generation, item)
    }

    fn checkpoint(
        &self,
        input: CheckpointInput,
        known: &[Arc<Generation>],
    ) -> Result<Checkpointed> {
        write_known(self, input, known)
    }

    fn remove_generations(&self, generations: &[GenId]) -> Vec<GenId> {
        MemoBackend::remove_generations(self, generations)
    }
}

/// The current chain as the live catalog sees it. Every layer of a chain
/// shares its base's catalog, so one table mapping serves them all.
pub(crate) struct IndexGen {
    /// Oldest first.
    pub(crate) layers: Vec<Arc<Generation>>,
    /// Per live table, its index in the base's catalog: `None` for a table
    /// the base doesn't hold under the same descriptor, and for every table
    /// when the build differs.
    to_gen: Vec<Option<u32>>,
    to_live: HashMap<u32, u32>,
    /// What the chain shows its readers; `bytes` is the size `max_bytes`
    /// bounds.
    pub(crate) live: Live,
    /// The summed entry frame lengths of the deltas.
    pub(crate) chain_bytes: u64,
}

impl IndexGen {
    pub(crate) fn new(chain: Chain, build: u128, catalog: &[TableDesc]) -> Self {
        let layers = chain.into_layers();
        let header = layers[0].header();
        let to_gen: Vec<Option<u32>> = catalog
            .iter()
            .map(|desc| {
                (header.build == build)
                    .then(|| header.table_index(desc))
                    .flatten()
            })
            .collect();
        let to_live = to_gen
            .iter()
            .enumerate()
            .filter_map(|(live, gen)| Some((((*gen)?), live as u32)))
            .collect();
        IndexGen {
            live: Live::of_chain(&layers),
            chain_bytes: layers[1..].iter().map(|layer| layer.entry_bytes()).sum(),
            layers,
            to_gen,
            to_live,
        }
    }

    pub(crate) fn depth(&self) -> usize {
        self.layers.len() - 1
    }

    pub(crate) fn live_table(&self, gen_table: u32) -> Option<u32> {
        self.to_live.get(&gen_table).copied()
    }

    /// Whether the chain may hold an entry for `key` of live `table`.
    pub(crate) fn holds(&self, table: u32, key: &[u8]) -> bool {
        self.to_gen[table as usize]
            .is_some_and(|gen| layers::may_hold(&self.layers, gen, key_hash(key)))
    }

    /// The stored entry for `key` of live `table`: `Ok(None)` if there is
    /// none or a layer it needs has been removed, `Err(())` for a frame
    /// that fails to decode or to read.
    pub(crate) fn read(
        &self,
        backend: &dyn ErasedBackend,
        table: u32,
        key: &[u8],
    ) -> std::result::Result<Option<Entry>, ()> {
        let Some(gen) = self.to_gen[table as usize] else {
            return Ok(None);
        };
        match layers::probe(&self.layers, gen, key, |id, item| {
            backend.read_entry(id, item)
        }) {
            Ok(Probe::Found(entry)) => Ok(Some(entry)),
            Ok(Probe::Missing | Probe::Gone) => Ok(None),
            Ok(Probe::Corrupt) | Err(_) => Err(()),
        }
    }
}

/// What M guards.
#[derive(Default)]
pub(crate) struct MemoState {
    /// Per node a restore seeded: the fingerprint of the value it seeded
    /// and the stamp it seeded at. A removed or recycled node never matches
    /// a live handle; checkpoints prune them.
    pub(crate) seed_info: HashMap<ErasedIncr, (u128, u64)>,
    /// Per node in `seed_info`, its live table and stored key bytes.
    pub(crate) seed_keys: HashMap<ErasedIncr, (u32, Vec<u8>)>,
    /// `seed_info`'s length after checkpoints last pruned it.
    pub(crate) pruned_at: usize,
    pub(crate) stats: MemoStats,
    pub(crate) mismatches: Vec<EncodeMismatch>,
}

impl MemoState {
    pub(crate) fn record_mismatch(&mut self, mismatch: EncodeMismatch) {
        if !self.mismatches.contains(&mismatch) {
            self.mismatches.push(mismatch);
        }
    }
}

/// Everything a restore, a checkpoint and a memo closure share.
pub(crate) struct MemoCore<C: MemoCells> {
    pub(crate) id: u64,
    pub(crate) build: u128,
    pub(crate) catalog: Vec<TableDesc>,
    pub(crate) tables: Vec<Box<dyn ErasedTable<C>>>,
    /// The store's value scope, whose hold batches a checkpoint's encodes.
    pub(crate) scope: Option<ValueScope>,
    pub(crate) backend: Box<dyn ErasedBackend>,
    pub(crate) deps: Arc<DepIndex>,
    externals: NodeTable<String, u128, C>,
    pub(crate) max_bytes: u64,
    checkpoint: Mutex<()>,
    memo: Mutex<MemoState>,
    index: RwLock<Arc<IndexGen>>,
    pub(crate) dirty: Dirty,
    /// `MemoOptions::checkpoint_after_entries`.
    pub(crate) trigger: Option<u64>,
    pub(crate) wake: Arc<Wake>,
}

/// What the automatic-checkpoint thread waits on. A leaf lock outside the
/// lock order.
pub(crate) struct Wake {
    state: Mutex<Waking>,
    cond: Condvar,
    /// Set while a compute that finds the trigger reached may wake the
    /// thread; the first one to do so clears it, and the thread sets it
    /// again before each wait.
    armed: AtomicBool,
    /// Set by the thread after a checkpoint skipped for an open staged
    /// batch or unlogged scope; the close of the last one, or the thread
    /// itself when it finds both closed already, clears it and wakes the
    /// thread. SeqCst, see `DepIndex::unlogged`.
    skipped: AtomicBool,
}

impl Default for Wake {
    fn default() -> Self {
        Wake {
            state: Mutex::default(),
            cond: Condvar::new(),
            armed: AtomicBool::new(true),
            skipped: AtomicBool::new(false),
        }
    }
}

#[derive(Default)]
struct Waking {
    stop: bool,
    due: bool,
    /// What a checkpoint since the thread's last [`Wake::take_landing`]
    /// left landing.
    landing: Option<Landing>,
    #[cfg(test)]
    dues: u64,
    #[cfg(test)]
    held_skips: u64,
    #[cfg(test)]
    blocked: bool,
    #[cfg(test)]
    landings: u64,
}

impl Wake {
    fn lock(&self) -> MutexGuard<'_, Waking> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn due(&self) {
        let mut waking = self.lock();
        waking.due = true;
        #[cfg(test)]
        {
            waking.dues += 1;
        }
        drop(waking);
        self.cond.notify_all();
    }

    /// How many times [`Self::due`] ran.
    #[cfg(test)]
    pub(crate) fn dues(&self) -> u64 {
        self.lock().dues
    }

    /// Counts a skipped checkpoint after which the thread waits for a close.
    #[cfg(test)]
    pub(crate) fn held_skip(&self) {
        self.lock().held_skips += 1;
        self.cond.notify_all();
    }

    /// Waits up to `timeout` until [`Self::held_skip`] ran `n` times:
    /// whether it did.
    #[cfg(test)]
    pub(crate) fn wait_for_held_skips(&self, n: u64, timeout: Duration) -> bool {
        let waking = self.lock();
        let (waking, _) = self
            .cond
            .wait_timeout_while(waking, timeout, |w| w.held_skips < n)
            .unwrap_or_else(PoisonError::into_inner);
        waking.held_skips >= n
    }

    /// Waits up to `timeout` until the thread is blocked with nothing
    /// due: whether it is.
    #[cfg(test)]
    pub(crate) fn wait_until_blocked(&self, timeout: Duration) -> bool {
        let waking = self.lock();
        let (waking, _) = self
            .cond
            .wait_timeout_while(waking, timeout, |w| !w.blocked || w.due)
            .unwrap_or_else(PoisonError::into_inner);
        waking.blocked && !waking.due
    }

    /// Waits up to `timeout` until [`Self::wait`] found a carried entry
    /// landed `n` times: whether it did.
    #[cfg(test)]
    pub(crate) fn wait_for_landings(&self, n: u64, timeout: Duration) -> bool {
        let waking = self.lock();
        let (waking, _) = self
            .cond
            .wait_timeout_while(waking, timeout, |w| w.landings < n)
            .unwrap_or_else(PoisonError::into_inner);
        waking.landings >= n
    }

    /// Records a checkpoint skipped for an open staged batch or unlogged
    /// scope, for [`Self::resume`].
    pub(crate) fn skip(&self) {
        self.skipped.store(true, Ordering::SeqCst);
    }

    /// [`Self::due`] if a skip is recorded, clearing it.
    pub(crate) fn resume(&self) {
        if self.skipped.swap(false, Ordering::SeqCst) {
            self.due();
        }
    }

    /// Lets the next compute that finds the trigger reached wake the
    /// thread. A swap, not a store: when it reads a compute's
    /// [`Self::due_once`], the thread's next unsaved count sees that
    /// compute's.
    pub(crate) fn arm(&self) {
        self.armed.swap(true, Ordering::AcqRel);
    }

    /// [`Self::due`] once per [`Self::arm`].
    pub(crate) fn due_once(&self) {
        if self.armed.swap(false, Ordering::AcqRel) {
            self.due();
        }
    }

    pub(crate) fn stop(&self) {
        self.lock().stop = true;
        self.cond.notify_all();
    }

    /// Hands the thread what a checkpoint left landing, replacing what an
    /// earlier one left.
    pub(crate) fn await_landing(&self, landing: Landing) {
        self.lock().landing = Some(landing);
        self.cond.notify_all();
    }

    /// What a checkpoint left landing since the last call, if one ran.
    pub(crate) fn take_landing(&self) -> Option<Landing> {
        self.lock().landing.take()
    }

    /// Waits until a checkpoint is due or `deadline` has passed
    /// ([`Woke::Due`]), a checkpoint left new entries landing, a stop, or
    /// `poll` returns `true`. It calls `poll` after [`FIRST_POLL`], then
    /// at gaps doubling up to [`LAST_POLL`].
    pub(crate) fn wait(
        &self,
        deadline: Option<Instant>,
        mut poll: Option<&mut dyn FnMut() -> bool>,
    ) -> Woke {
        let mut gap = FIRST_POLL;
        let mut waking = self.lock();
        loop {
            if waking.stop {
                return Woke::Stop;
            }
            if std::mem::take(&mut waking.due) {
                return Woke::Due;
            }
            if waking.landing.is_some() {
                return Woke::Replaced;
            }
            let now = Instant::now();
            if deadline.is_some_and(|deadline| deadline <= now) {
                return Woke::Due;
            }
            let until = match (deadline, poll.is_some()) {
                (Some(deadline), true) => Some(deadline.min(now + gap)),
                (None, true) => Some(now + gap),
                (deadline, false) => deadline,
            };
            #[cfg(test)]
            {
                waking.blocked = true;
                self.cond.notify_all();
            }
            waking = match until {
                None => self
                    .cond
                    .wait(waking)
                    .unwrap_or_else(PoisonError::into_inner),
                Some(until) => {
                    self.cond
                        .wait_timeout(waking, until.saturating_duration_since(now))
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
            };
            #[cfg(test)]
            {
                waking.blocked = false;
            }
            if let Some(poll) = poll.as_mut() {
                if Instant::now() >= now + gap {
                    drop(waking);
                    let landed = poll();
                    waking = self.lock();
                    if landed {
                        #[cfg(test)]
                        {
                            waking.landings += 1;
                            self.cond.notify_all();
                        }
                        return Woke::Landed;
                    }
                    gap = (gap * 2).min(LAST_POLL);
                }
            }
        }
    }
}

/// Why [`Wake::wait`] returned.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Woke {
    Stop,
    Due,
    /// A checkpoint left new entries landing.
    Replaced,
    /// `poll` found an entry landed.
    Landed,
}

const FIRST_POLL: Duration = Duration::from_millis(1);
const LAST_POLL: Duration = Duration::from_secs(1);

/// The entries a checkpoint carried and could not take, each with whether
/// it counted that entry unsaved: no compute reports their landing.
#[derive(Default)]
pub(crate) struct Landing {
    entries: Vec<(ErasedIncr, bool)>,
}

impl Landing {
    pub(crate) fn push(&mut self, handle: ErasedIncr, counted: bool) {
        self.entries.push((handle, counted));
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many of the entries the checkpoint counted unsaved.
    pub(crate) fn counted(&self) -> u64 {
        self.entries.iter().filter(|&&(_, counted)| counted).count() as u64
    }

    /// Drops the entries `landed` accepts, handing those the checkpoint
    /// did not count to `uncounted`: whether there were any.
    pub(crate) fn drop_landed(
        &mut self,
        landed: impl Fn(ErasedIncr) -> bool,
        mut uncounted: impl FnMut(ErasedIncr),
    ) -> bool {
        let before = self.entries.len();
        self.entries.retain(|&(handle, counted)| {
            let dropped = landed(handle);
            if dropped && !counted {
                uncounted(handle);
            }
            !dropped
        });
        self.entries.len() < before
    }
}

impl<C: MemoCells> MemoCore<C> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        id: u64,
        build: u128,
        catalog: Vec<TableDesc>,
        tables: Vec<Box<dyn ErasedTable<C>>>,
        scope: Option<ValueScope>,
        backend: Box<dyn ErasedBackend>,
        deps: Arc<DepIndex>,
        externals: NodeTable<String, u128, C>,
        max_bytes: u64,
        trigger: Option<u64>,
        index: IndexGen,
    ) -> Self {
        MemoCore {
            id,
            build,
            catalog,
            tables,
            scope,
            backend,
            deps,
            externals,
            max_bytes,
            checkpoint: Mutex::new(()),
            memo: Mutex::new(MemoState::default()),
            index: RwLock::new(Arc::new(index)),
            dirty: Dirty::new(),
            trigger,
            wake: Arc::default(),
        }
    }

    // Poisoning is ignored on all three: a restore that unwinds part-way
    // leaves only seeded entries, which were verified, and cold ones, and
    // `seed_info` is written only after a seed succeeds.
    pub(crate) fn lock_checkpoint(&self) -> Ordered<MutexGuard<'_, ()>> {
        Ordered::take(LockKind::MemoCheckpoint, || {
            self.checkpoint
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
        })
    }

    pub(crate) fn lock_memo(&self) -> Ordered<MutexGuard<'_, MemoState>> {
        Ordered::take(LockKind::Memo, || {
            self.memo.lock().unwrap_or_else(PoisonError::into_inner)
        })
    }

    /// Records that `table`'s entry `handle` computed, and wakes the
    /// automatic-checkpoint thread when a checkpoint is due. A count that
    /// reached the trigger without a compute, carried or left by a failed
    /// checkpoint, wakes it at the next compute; a carried entry's landing
    /// also wakes it when that leaves a checkpoint due.
    pub(crate) fn computed(&self, table: u32, handle: ErasedIncr, key: &KeyPart) {
        let unsaved = self.dirty.computed(table, handle, key);
        self.reached(unsaved);
    }

    /// [`Self::computed`] for a clean entry the backend lost.
    pub(crate) fn lost(&self, table: u32, handle: ErasedIncr, key: Vec<u8>) {
        let unsaved = self.dirty.lost(table, handle, key);
        self.reached(unsaved);
    }

    fn reached(&self, unsaved: u64) {
        if self.trigger.is_some_and(|trigger| unsaved >= trigger) {
            self.wake.due_once();
        }
    }

    /// Whether `checkpoint_after_entries` entries are unsaved.
    pub(crate) fn checkpoint_due(&self) -> bool {
        self.checkpoint_due_besides(0)
    }

    /// [`Self::checkpoint_due`] leaving out `carried` of the unsaved
    /// entries.
    pub(crate) fn checkpoint_due_besides(&self, carried: u64) -> bool {
        self.trigger
            .is_some_and(|trigger| self.dirty.unsaved().saturating_sub(carried) >= trigger)
    }

    /// The stats with what the backend holds and lacks now.
    pub(crate) fn stats(&self) -> MemoStats {
        let mut stats = self.lock_memo().stats.clone();
        let index = self.index();
        stats.entries = index.live.entries;
        stats.bytes = index.live.bytes;
        stats.chain_depth = index.depth() as u64;
        stats.chain_bytes = index.chain_bytes;
        stats.unsaved_entries = self.dirty.unsaved();
        stats
    }

    pub(crate) fn index(&self) -> Arc<IndexGen> {
        let index = Ordered::take(LockKind::MemoIndex, || {
            self.index.read().unwrap_or_else(PoisonError::into_inner)
        });
        Arc::clone(&index)
    }

    /// Swaps in `index`; the caller holds M.
    pub(crate) fn swap_index(&self, index: IndexGen) {
        let mut current = Ordered::take(LockKind::MemoIndex, || {
            self.index.write().unwrap_or_else(PoisonError::into_inner)
        });
        *current = Arc::new(index);
    }

    pub(crate) fn table<T: Any>(&self, memo: u64, index: u32) -> &T {
        assert_eq!(
            memo, self.id,
            "incr-persist: a MemoTable used with a memo other than the one that declared it"
        );
        self.tables[index as usize]
            .as_any()
            .downcast_ref()
            .expect("incr-persist: a MemoTable's types match the table it names")
    }
}

impl<C: MemoCells> MemoCore<C> {
    /// External `name`'s input, created at fp `0` if no call has set it.
    /// Its value changes only under the `DepIndex` write hold, so the fp
    /// registered there always equals it.
    pub(crate) fn external(&self, rt: &Runtime<C>, name: &str) -> Incr<u128> {
        if let Some(external) = self.deps.read().external(self.id, name) {
            return external.handle;
        }
        // Created before the hold: a table miss ranks below `DepIndex`.
        let handle = self
            .externals
            .get_or_insert_input(rt, name.to_owned(), || 0);
        let mut map = self.deps.write();
        if map.external(self.id, name).is_none() {
            let stamp = changed_at(rt, handle);
            map.set_external(
                self.id,
                name,
                External {
                    handle,
                    fp: 0,
                    stamp,
                },
            );
        }
        handle
    }

    /// Sets external `name` to `fp`, creating its input on the first call.
    pub(crate) fn set_external(&self, rt: &Runtime<C>, name: &str, fp: u128) -> Incr<u128> {
        // Created before the hold: a table miss ranks below `DepIndex`.
        let handle = self
            .externals
            .get_or_insert_input(rt, name.to_owned(), || 0);
        let _routing = rt.hold_routing();
        let mut map = self.deps.write();
        rt.set(handle, fp);
        let stamp = changed_at(rt, handle);
        map.set_external(self.id, name, External { handle, fp, stamp });
        handle
    }
}

/// A [`MemoCore`] for memo closures, which core requires to be `Send` and
/// `Sync` under every `Cells`.
pub(crate) struct CoreRef<C: MemoCells>(pub(crate) Arc<MemoCore<C>>);

impl<C: MemoCells> Clone for CoreRef<C> {
    fn clone(&self) -> Self {
        CoreRef(Arc::clone(&self.0))
    }
}

impl<C: MemoCells> Deref for CoreRef<C> {
    type Target = MemoCore<C>;

    fn deref(&self) -> &MemoCore<C> {
        &self.0
    }
}

/// The `Cells` a memo runs on: [`Local`] and [`Shared`]. Sealed, because
/// memo closures share the memo's state under exactly those two models.
pub trait MemoCells: Cells + sealed::Sealed {}

impl MemoCells for Local {}
impl MemoCells for Shared {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for incr_core::Local {}
    impl Sealed for incr_core::Shared {}
}

// SAFETY: `MemoCells` is sealed to `Local` and `Shared`, and for both
// every field of `MemoCore` is `Send + Sync` on its own: the externals'
// `NodeTable<String, u128, C>` by core's impls for either `Cells`, the
// backend by `ErasedBackend`'s supertraits, the locks and `Arc<DepIndex>`
// by std. The one field that hides its auto traits is the boxed
// `dyn ErasedTable<C>`, and its only implementor is `TableCore<K, T, C>`:
// a `NodeTable<K, T, C>` (core's impls again, as `K: MemoKey` and
// `T: MemoValue` are `Send + Sync`) and an `Arc` of a `Send + Sync`
// closure. `table_core::tests` checks both under each `Cells`.
unsafe impl<C: MemoCells> Send for CoreRef<C> {}
// SAFETY: as for `Send`.
unsafe impl<C: MemoCells> Sync for CoreRef<C> {}
