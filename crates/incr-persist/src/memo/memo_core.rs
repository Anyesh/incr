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

use super::table_core::{ErasedTable, KeyPart};
use super::{
    decode_entry, key_hash, write_checkpoint, CheckpointInput, Checkpointed, Entry, GenId,
    Generation, IndexItem, MemoBackend, TableDesc,
};
use crate::store::dep_index::{DepIndex, External};
use crate::value_context::EncodeMismatch;
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
}

/// [`MemoBackend`] with its guard type erased.
pub(crate) trait ErasedBackend: Send + Sync {
    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>>;
    fn checkpoint(&self, input: CheckpointInput) -> Result<Checkpointed>;
    fn remove_generation(&self, generation: GenId) -> Result<()>;
}

impl<M: MemoBackend> ErasedBackend for M {
    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        MemoBackend::read_entry(self, generation, item)
    }

    fn checkpoint(&self, input: CheckpointInput) -> Result<Checkpointed> {
        write_checkpoint(self, input)
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        MemoBackend::remove_generation(self, generation)
    }
}

/// The current generation as the live catalog sees it.
pub(crate) struct IndexGen {
    pub(crate) generation: Generation,
    /// Per live table, its index in the generation's catalog: `None` for a
    /// table the generation doesn't hold under the same descriptor, and for
    /// every table when the build differs.
    to_gen: Vec<Option<u32>>,
    to_live: HashMap<u32, u32>,
    /// The generation's [`Generation::entry_bytes`].
    pub(crate) bytes: u64,
}

impl IndexGen {
    pub(crate) fn new(generation: Generation, build: u128, catalog: &[TableDesc]) -> Self {
        let header = generation.header();
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
            bytes: generation.entry_bytes(),
            generation,
            to_gen,
            to_live,
        }
    }

    pub(crate) fn live_table(&self, gen_table: u32) -> Option<u32> {
        self.to_live.get(&gen_table).copied()
    }

    /// Whether the generation may hold an entry for `key` of live `table`.
    pub(crate) fn holds(&self, table: u32, key: &[u8]) -> bool {
        self.to_gen[table as usize]
            .is_some_and(|gen| !self.generation.find(gen, key_hash(key)).is_empty())
    }

    /// The stored entry for `key` of live `table`: `Ok(None)` if there is
    /// none or its generation has been removed, `Err(())` for a frame that
    /// fails to decode.
    pub(crate) fn read(
        &self,
        backend: &dyn ErasedBackend,
        table: u32,
        key: &[u8],
    ) -> std::result::Result<Option<Entry>, ()> {
        let Some(gen) = self.to_gen[table as usize] else {
            return Ok(None);
        };
        let mut corrupt = false;
        for item in self.generation.find(gen, key_hash(key)) {
            match backend.read_entry(self.generation.id(), item) {
                Ok(Some(frame)) => match decode_entry(&frame, self.generation.frame_tag()) {
                    Ok(entry) if entry.table == gen && entry.key == key => return Ok(Some(entry)),
                    Ok(_) => {}
                    Err(_) => corrupt = true,
                },
                Ok(None) => return Ok(None),
                Err(_) => corrupt = true,
            }
        }
        if corrupt {
            Err(())
        } else {
            Ok(None)
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
    #[cfg(test)]
    dues: u64,
    #[cfg(test)]
    held_skips: u64,
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
    /// thread.
    pub(crate) fn arm(&self) {
        self.armed.store(true, Ordering::Release);
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

    /// Waits until a checkpoint is due, `every` has passed, or a stop:
    /// `true` for a stop. `None` waits without an interval.
    pub(crate) fn wait(&self, every: Option<Duration>) -> bool {
        let deadline = every.map(|every| Instant::now() + every);
        let mut waking = self.lock();
        loop {
            if waking.stop {
                return true;
            }
            if std::mem::take(&mut waking.due) {
                return false;
            }
            waking = match deadline {
                None => self
                    .cond
                    .wait(waking)
                    .unwrap_or_else(PoisonError::into_inner),
                Some(deadline) => {
                    let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                        return false;
                    };
                    self.cond
                        .wait_timeout(waking, left)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
            };
        }
    }
}

impl<C: MemoCells> MemoCore<C> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        id: u64,
        build: u128,
        catalog: Vec<TableDesc>,
        tables: Vec<Box<dyn ErasedTable<C>>>,
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
    /// checkpoint, wakes it at the next compute.
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
        self.trigger
            .is_some_and(|trigger| self.dirty.unsaved() >= trigger)
    }

    /// The stats with what the backend holds and lacks now.
    pub(crate) fn stats(&self) -> MemoStats {
        let mut stats = self.lock_memo().stats.clone();
        let index = self.index();
        stats.entries = index.generation.items().len() as u64;
        stats.bytes = index.bytes;
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
