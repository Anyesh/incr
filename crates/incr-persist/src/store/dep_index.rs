use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{
    Arc, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError,
    Weak,
};

use incr_core::lock_order::{LockKind, Ordered};
use incr_core::persist_hooks::Deferred;
use incr_core::{Cells, ErasedIncr, Incr, Runtime};

use crate::memo::Wake;
use crate::Kind;

/// The xxh3-128 of `bytes`: every fingerprint the memo compares.
pub(crate) fn fp_of(bytes: &[u8]) -> u128 {
    #[cfg(test)]
    FP_BYTES.with(|hashed| hashed.set(hashed.get() + bytes.len()));
    xxhash_rust::xxh3::xxh3_128(bytes)
}

#[cfg(test)]
thread_local! {
    pub(crate) static FP_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// What a key's runtime node holds, as far as a memo can tell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DepFp {
    Known(u128),
    /// A mapped write of the key panicked part-way, so nothing that depends
    /// on it is trusted again in this process.
    Untrusted,
}

/// Computes a collection's fingerprint when a memo first needs it. Called
/// under the memo lock and a `DepIndex` read hold, which keeps every mapped
/// write out, and possibly inside a compute.
pub(crate) trait FpSource<C: Cells>: Send + Sync {
    /// Reads the collection without recording a dependency, running its
    /// value code in the store's value scope, and moves every value it
    /// materialises into `bag`. `Untrusted` when an element does not
    /// encode or the node was deleted through the runtime.
    fn fp(&self, rt: &Runtime<C>, bag: &mut Deferred<C>) -> DepFp;
}

/// An [`FpSource`] as a `DepIndex` holds it, behind `Any`.
pub(crate) type Fps<C> = Arc<dyn FpSource<C>>;

fn fps_of<C: Cells>(source: &(dyn Any + Send + Sync)) -> &Fps<C> {
    source
        .downcast_ref::<Fps<C>>()
        .expect("a store registers fingerprint sources of its own cells")
}

/// How a key's fingerprint is registered.
pub(crate) enum Registered {
    Now(DepFp),
    /// A collection's [`Fps`], computed when first read.
    Later(Box<dyn Any + Send + Sync>),
}

#[derive(Debug)]
enum FpState {
    Ready(DepFp),
    /// A leaf lock, held while the first reader computes the fingerprint
    /// so that concurrent readers wait for it instead of computing it too.
    Pending {
        source: Box<dyn Any + Send + Sync>,
        computed: Mutex<Option<DepFp>>,
    },
}

#[derive(Debug)]
pub(crate) struct DepEntry {
    pub(crate) handle: ErasedIncr,
    pub(crate) kind: Kind,
    fp: FpState,
    /// An input's mapped stamp: its `changed_at` after the last mapped write
    /// that wrote. `None` for a collection and for an untrusted key.
    pub(crate) stamp: Option<u64>,
}

impl DepEntry {
    /// The fingerprint, unless it is pending and no reader has computed it
    /// or one is computing it now.
    fn known_fp(&self) -> Option<DepFp> {
        match &self.fp {
            FpState::Ready(fp) => Some(*fp),
            FpState::Pending { computed, .. } => match computed.try_lock() {
                Ok(computed) => *computed,
                Err(TryLockError::Poisoned(computed)) => *computed.into_inner(),
                Err(TryLockError::WouldBlock) => None,
            },
        }
    }

    /// Whether no panic or failed encode untrusted the key. A pending
    /// fingerprint counts as trusted without being computed.
    pub(crate) fn trusted(&self) -> bool {
        self.known_fp() != Some(DepFp::Untrusted)
    }

    /// Whether writes keep the fingerprint current: it is known, and
    /// neither pending nor untrusted.
    fn tracked(&mut self) -> bool {
        if let FpState::Pending { computed, .. } = &mut self.fp {
            let computed = computed.get_mut().unwrap_or_else(PoisonError::into_inner);
            if let Some(fp) = computed.take() {
                self.fp = FpState::Ready(fp);
            }
        }
        matches!(self.fp, FpState::Ready(DepFp::Known(_)))
    }
}

/// A memo external's input, its fp and the stamp the fp vouches for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct External {
    pub(crate) handle: Incr<u128>,
    pub(crate) fp: u128,
    pub(crate) stamp: u64,
}

/// Every declared key of a store, with its node and fingerprint, while a
/// memo is open over the store.
pub(crate) struct DepIndex {
    map: RwLock<DepMap>,
    /// Set while an unlogged scope is open on the store: a memo then
    /// neither restores nor checkpoints, because its fingerprints describe
    /// the logged state while the runtime holds something else. Shared
    /// with the map, whose pending fingerprints it guards.
    unlogged: Arc<AtomicBool>,
    /// Set while a staged batch is open on the store: a memo then refuses
    /// to checkpoint, because the runtime holds writes the log may never
    /// receive.
    staged: AtomicBool,
    /// Set once the store is poisoned: a memo then refuses to checkpoint,
    /// because the runtime may hold changes the log never received.
    poisoned: AtomicBool,
    /// The automatic-checkpoint threads of the memos open over the store,
    /// resumed when the last staged batch or unlogged scope closes. A leaf
    /// lock outside the lock order, held only to take a [`Wake`]'s.
    waiters: Mutex<Vec<Weak<Wake>>>,
}

impl Default for DepIndex {
    fn default() -> Self {
        let unlogged = Arc::<AtomicBool>::default();
        DepIndex {
            map: RwLock::new(DepMap {
                unlogged: Arc::clone(&unlogged),
                ..DepMap::default()
            }),
            unlogged,
            staged: AtomicBool::default(),
            poisoned: AtomicBool::default(),
            waiters: Mutex::default(),
        }
    }
}

#[derive(Default)]
pub(crate) struct DepMap {
    by_key: HashMap<String, DepEntry>,
    by_handle: HashMap<ErasedIncr, String>,
    /// Each memo's externals by (memo id, name): an input whose value is
    /// its fp. Keyed by memo too, because every memo on a store shares
    /// this map but owns its own external nodes.
    externals: HashMap<(u64, String), External>,
    external_names: HashMap<ErasedIncr, (u64, String)>,
    /// Each keyed collection's rows by its key, a `keyed::Rows<C>` of the
    /// store's cells.
    rows: HashMap<String, Box<dyn Any + Send + Sync>>,
    /// The index's unlogged flag. An unlogged scope's writes bypass the
    /// index's hold, and its close takes the hold for writing, so a
    /// fingerprint computed while the flag is clear afterwards saw none.
    unlogged: Arc<AtomicBool>,
}

impl DepIndex {
    // `unlogged` and `staged` are SeqCst, like a `Wake`'s skip: a thread
    // that records a skip and then finds both clear, and a close that
    // clears one and then finds the skip, cannot both miss the other.
    pub(crate) fn unlogged(&self) -> bool {
        self.unlogged.load(Ordering::SeqCst)
    }

    /// Clearing it resumes the threads that skipped a checkpoint, when no
    /// staged batch is open either.
    pub(super) fn set_unlogged(&self, open: bool) {
        self.unlogged.store(open, Ordering::SeqCst);
        if !open {
            self.closed();
        }
    }

    pub(crate) fn staged(&self) -> bool {
        self.staged.load(Ordering::SeqCst)
    }

    /// Clearing it resumes the threads that skipped a checkpoint, when no
    /// unlogged scope is open either.
    pub(super) fn set_staged(&self, open: bool) {
        self.staged.store(open, Ordering::SeqCst);
        if !open {
            self.closed();
        }
    }

    /// Whether a staged batch or an unlogged scope is open, so a memo
    /// refuses to checkpoint.
    pub(crate) fn holds_checkpoints(&self) -> bool {
        self.unlogged() || self.staged()
    }

    fn closed(&self) {
        if self.holds_checkpoints() {
            return;
        }
        for waiter in self.lock_waiters().iter() {
            if let Some(wake) = waiter.upgrade() {
                wake.resume();
            }
        }
    }

    fn lock_waiters(&self) -> MutexGuard<'_, Vec<Weak<Wake>>> {
        self.waiters.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Resumes `wake` at every close after it [skipped](Wake::skip) a
    /// checkpoint, until [`Self::unwatch`].
    pub(crate) fn watch(&self, wake: &Arc<Wake>) {
        let mut waiters = self.lock_waiters();
        waiters.retain(|waiter| waiter.strong_count() > 0);
        waiters.push(Arc::downgrade(wake));
    }

    pub(crate) fn unwatch(&self, wake: &Arc<Wake>) {
        self.lock_waiters()
            .retain(|waiter| waiter.strong_count() > 0 && waiter.as_ptr() != Arc::as_ptr(wake));
    }

    pub(crate) fn poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    pub(super) fn set_poisoned(&self) {
        self.poisoned.store(true, Ordering::Release);
    }

    // Poisoning is ignored: a mapped write that panics marks the one key it
    // was writing untrusted before the hold is released.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn read(&self) -> Ordered<RwLockReadGuard<'_, DepMap>> {
        Ordered::take(LockKind::DepIndex, || {
            self.map.read().unwrap_or_else(PoisonError::into_inner)
        })
    }

    pub(crate) fn write(&self) -> Ordered<RwLockWriteGuard<'_, DepMap>> {
        Ordered::take(LockKind::DepIndex, || {
            self.map.write().unwrap_or_else(PoisonError::into_inner)
        })
    }

    /// The fingerprint of `key`, computed if pending.
    #[cfg(test)]
    pub(crate) fn fp<C: Cells>(&self, rt: &Runtime<C>, key: &str) -> Option<DepFp> {
        let mut bag = Deferred::new();
        let map = self.read();
        let fp = map.entry(key).map(|entry| map.fp(entry, rt, &mut bag));
        drop(map);
        bag.drain(rt);
        fp
    }

    /// Whether `key` is registered with a fingerprint no reader computed.
    #[cfg(test)]
    pub(crate) fn pending(&self, key: &str) -> bool {
        self.read()
            .entry(key)
            .is_some_and(|entry| entry.known_fp().is_none())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn stamp(&self, key: &str) -> Option<u64> {
        self.read().entry(key).and_then(|entry| entry.stamp)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn key_of(&self, handle: ErasedIncr) -> Option<String> {
        self.read().by_handle.get(&handle).cloned()
    }

    #[cfg(test)]
    pub(crate) fn is_held(&self) -> bool {
        self.map.try_write().is_err()
    }
}

impl DepMap {
    pub(crate) fn entry(&self, key: &str) -> Option<&DepEntry> {
        self.by_key.get(key)
    }

    /// `entry`'s fingerprint. A pending one is computed by the first caller,
    /// while concurrent callers wait for it, and kept unless an unlogged
    /// scope is open, whose writes it may have seen: it then reads
    /// `Untrusted` for this caller and stays pending.
    pub(crate) fn fp<C: Cells>(
        &self,
        entry: &DepEntry,
        rt: &Runtime<C>,
        bag: &mut Deferred<C>,
    ) -> DepFp {
        let FpState::Pending { source, computed } = &entry.fp else {
            return entry.known_fp().expect("a ready fingerprint is known");
        };
        let mut computed = computed.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(fp) = *computed {
            return fp;
        }
        if self.unlogged.load(Ordering::SeqCst) {
            return DepFp::Untrusted;
        }
        let fp = fps_of::<C>(&**source).fp(rt, bag);
        if self.unlogged.load(Ordering::SeqCst) {
            return DepFp::Untrusted;
        }
        *computed = Some(fp);
        fp
    }

    pub(crate) fn register(
        &mut self,
        key: &str,
        handle: ErasedIncr,
        kind: Kind,
        fp: Registered,
        stamp: Option<u64>,
    ) {
        let fp = match fp {
            Registered::Now(fp) => FpState::Ready(fp),
            Registered::Later(source) => FpState::Pending {
                source,
                computed: Mutex::new(None),
            },
        };
        let stamp = stamp.filter(|_| !matches!(fp, FpState::Ready(DepFp::Untrusted)));
        let entry = DepEntry {
            handle,
            kind,
            fp,
            stamp,
        };
        if let Some(old) = self.by_key.insert(key.to_owned(), entry) {
            self.by_handle.remove(&old.handle);
        }
        self.by_handle.insert(handle, key.to_owned());
    }

    pub(crate) fn unregister(&mut self, key: &str) -> Option<DepEntry> {
        self.rows.remove(key);
        let entry = self.by_key.remove(key)?;
        self.by_handle.remove(&entry.handle);
        Some(entry)
    }

    /// Whether writes to `key` keep its fingerprint current, so a write
    /// works out the change: not for a pending or untrusted one, which a
    /// write leaves as it is.
    pub(crate) fn tracks(&mut self, key: &str) -> bool {
        self.by_key.get_mut(key).is_some_and(DepEntry::tracked)
    }

    /// Sets the fingerprint of a registered key whose fingerprint is
    /// [tracked](Self::tracks); a pending or untrusted one stays so.
    pub(crate) fn set_fp(&mut self, key: &str, fp: u128) {
        if let Some(entry) = self.by_key.get_mut(key) {
            if entry.tracked() {
                entry.fp = FpState::Ready(DepFp::Known(fp));
            }
        }
    }

    /// Moves a tracked fingerprint by `delta`.
    pub(crate) fn shift_fp(&mut self, key: &str, delta: u128) {
        if let Some(entry) = self.by_key.get_mut(key) {
            if entry.tracked() {
                if let FpState::Ready(DepFp::Known(fp)) = &mut entry.fp {
                    *fp = fp.wrapping_add(delta);
                }
            }
        }
    }

    /// Sets a registered input's fingerprint and mapped stamp after a write
    /// that wrote; an untrusted one stays so.
    pub(crate) fn set_written(&mut self, key: &str, fp: u128, stamp: u64) {
        if let Some(entry) = self.by_key.get_mut(key) {
            if entry.tracked() {
                entry.fp = FpState::Ready(DepFp::Known(fp));
                entry.stamp = Some(stamp);
            }
        }
    }

    pub(crate) fn store_fp(&mut self, key: &str, fp: DepFp) {
        match fp {
            DepFp::Known(fp) => self.set_fp(key, fp),
            DepFp::Untrusted => self.untrust(key),
        }
    }

    pub(crate) fn register_rows(&mut self, key: &str, rows: Box<dyn Any + Send + Sync>) {
        self.rows.insert(key.to_owned(), rows);
    }

    pub(crate) fn rows(&self, key: &str) -> Option<&(dyn Any + Send + Sync)> {
        self.rows.get(key).map(|rows| &**rows)
    }

    pub(crate) fn all_rows(&self) -> impl Iterator<Item = (&str, &(dyn Any + Send + Sync))> {
        self.rows.iter().map(|(key, rows)| (key.as_str(), &**rows))
    }

    pub(crate) fn key_of(&self, handle: ErasedIncr) -> Option<&str> {
        self.by_handle.get(&handle).map(String::as_str)
    }

    pub(crate) fn external(&self, memo: u64, name: &str) -> Option<External> {
        self.externals.get(&(memo, name.to_owned())).copied()
    }

    /// The memo and name of the external whose input is `handle`.
    pub(crate) fn external_of(&self, handle: ErasedIncr) -> Option<(u64, &str)> {
        self.external_names
            .get(&handle)
            .map(|(memo, name)| (*memo, name.as_str()))
    }

    /// Records the input of `memo`'s external `name`, the fp it was just
    /// set to and its `changed_at` after that set.
    pub(crate) fn set_external(&mut self, memo: u64, name: &str, external: External) {
        self.externals.insert((memo, name.to_owned()), external);
        self.external_names
            .insert(external.handle.erase(), (memo, name.to_owned()));
    }

    pub(crate) fn untrust(&mut self, key: &str) {
        if let Some(entry) = self.by_key.get_mut(key) {
            entry.fp = FpState::Ready(DepFp::Untrusted);
            entry.stamp = None;
        }
    }
}

/// Marks `key` untrusted if dropped while unwinding, so a panic in user
/// code inside a mapped write never leaves a fingerprint that disagrees
/// with the runtime.
pub(crate) struct PanicGuard<'a> {
    deps: Option<&'a mut DepMap>,
    key: &'a str,
}

impl<'a> PanicGuard<'a> {
    pub(crate) fn new(deps: Option<&'a mut DepMap>, key: &'a str) -> Self {
        PanicGuard { deps, key }
    }

    pub(crate) fn deps(&mut self) -> Option<&mut DepMap> {
        self.deps.as_deref_mut()
    }

    /// Disarms the guard once the user code has returned.
    pub(crate) fn disarm(mut self) {
        self.deps = None;
    }
}

impl Drop for PanicGuard<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            if let Some(deps) = self.deps.as_deref_mut() {
                deps.untrust(self.key);
            }
        }
    }
}
