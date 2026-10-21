//! `NodeTable<K, T, C, S>`: a map from keys to nodes of one runtime, so a
//! compute can find or create the node for a key instead of keeping
//! handles itself.
//!
//! The table is open-addressed with linear probing over a power-of-two
//! array of inline buckets. A hit is lock-free: it loads a bucket's `meta`
//! with Acquire and reads the bucket's hash, handle and key, which were
//! written before `meta` was published with Release and are never written
//! again in that array. Every write (an insert, a tombstone, a rehash)
//! runs under the table's miss mutex. A rehash clones the live keys into
//! a new array, publishes it, and retires the old one after the mutex is
//! released, so a reader that still holds the old array keeps comparing
//! against keys nobody has dropped. The array pointer is protected by an
//! `Rc` under `Local`, a hazard pointer under `Shared` and an `RwLock`ed
//! `Arc` under loom.
//!
//! A lookup is not a dependency: `rt.get` of the handle records one.
//! `Hash`, `Eq` and `Clone` of a key run as value code, so a key that
//! calls into the runtime other than to read a Clean node panics with the
//! contract message, never re-entering the miss mutex.

use std::cell::{RefCell, UnsafeCell};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use haphazard::{AtomicPtr as HzAtomicPtr, Global, Replaced};

use crate::cells::{Cells, Local, Shared};
use crate::cycle::{self, ErasedOpts, QueryOpts};
use crate::deferred::{isolate, Deferred};
use crate::dep_stack::DepStack;
use crate::fx::FxBuildHasher;
use crate::handle::{ErasedIncr, Incr, RuntimeId};
use crate::lock_order::LockKind;
use crate::locks::OrderedLock;
use crate::reclaim_probe::{self, RetireSite};
use crate::runtime::{DeleteError, Runtime};
use crate::value::Value;
use crate::value_code;
use crate::value_slot::with_pooled_hazard;

/// What a table key must be.
pub trait TableKey: Hash + Eq + Clone + Send + Sync + 'static {}

impl<K> TableKey for K where K: Hash + Eq + Clone + Send + Sync + 'static {}

/// The protected pointer to a table's current array, through
/// `Cells::TableArray`.
#[doc(hidden)]
pub trait ArrayCell<A: 'static>: 'static {
    /// The replaced array; dropping it frees the array once no reader
    /// holds it.
    type Retired: 'static;

    fn new(array: A) -> Self;

    /// Runs `f` on the current array, which stays allocated while `f`
    /// runs even if another call replaces it.
    fn read<R>(&self, f: impl FnOnce(&A) -> R) -> R;

    /// Publishes `array`. Callers serialize replaces through the table's
    /// miss mutex.
    fn replace(&self, array: A) -> Self::Retired;
}

/// `Local`: a hit clones the `Rc`, so a callback that inserts cannot
/// free the array under the reader.
#[doc(hidden)]
pub struct RcArrayCell<A>(RefCell<Rc<A>>);

impl<A: 'static> ArrayCell<A> for RcArrayCell<A> {
    type Retired = Rc<A>;

    fn new(array: A) -> Self {
        Self(RefCell::new(Rc::new(array)))
    }

    #[inline]
    fn read<R>(&self, f: impl FnOnce(&A) -> R) -> R {
        let array = self.0.borrow().clone();
        f(&array)
    }

    fn replace(&self, array: A) -> Rc<A> {
        self.0.replace(Rc::new(array))
    }
}

/// `Shared`: one hazard pointer from the thread's pool per read.
#[doc(hidden)]
pub struct HzArrayCell<A>(HzAtomicPtr<A>);

impl<A: Send + Sync + 'static> ArrayCell<A> for HzArrayCell<A> {
    type Retired = RetiredArray<A>;

    fn new(array: A) -> Self {
        Self(HzAtomicPtr::from(Box::new(array)))
    }

    #[inline]
    fn read<R>(&self, f: impl FnOnce(&A) -> R) -> R {
        with_pooled_hazard(|hp| {
            // SAFETY: every pointer stored here came from `Box::into_raw`
            // in `new` or `replace`, and a replaced one is retired through
            // the global domain, so it stays valid while `hp` protects it.
            let array = unsafe { self.0.load(hp) }.expect("a table always has an array");
            f(array)
        })
    }

    fn replace(&self, array: A) -> RetiredArray<A> {
        RetiredArray(
            self.0
                .swap(Box::new(array))
                .expect("a table always has an array"),
        )
    }
}

impl<A> Drop for HzArrayCell<A> {
    fn drop(&mut self) {
        // SAFETY: the cell owns its current array, and no reader can be
        // inside `read` while the table is being dropped.
        drop(unsafe { Box::from_raw(self.0.load_ptr()) });
    }
}

/// A replaced `Shared` array. Dropping it retires the array through the
/// global haphazard domain, which drops it, keys included, once no hazard
/// pointer protects it.
#[doc(hidden)]
pub struct RetiredArray<A: Send>(Replaced<A, Global, Box<A>>);

impl<A: Send> Drop for RetiredArray<A> {
    fn drop(&mut self) {
        // SAFETY: the array came from `Box::into_raw` in `new` or
        // `replace`, was displaced exactly once by the swap that made this
        // token, and is never loaded from the cell again.
        unsafe { self.0.retire() };
        reclaim_probe::reached(RetireSite::TableRetire);
    }
}

/// A bucket's data, through `Cells::DataCell`: written once under the miss
/// mutex before the bucket's `meta` is published, then only read.
#[doc(hidden)]
pub trait DataCell<T>: 'static {
    fn new(value: T) -> Self;
    fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R;
    fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R;
}

#[doc(hidden)]
pub struct StdDataCell<T>(UnsafeCell<T>);

// SAFETY: a bucket's data is written only under the miss mutex and only
// while its `meta` is Empty, and read only after an Acquire load of
// `meta` has seen the Release store that published it, so no write is
// ever concurrent with another access.
unsafe impl<T: Send + Sync> Sync for StdDataCell<T> {}

impl<T: 'static> DataCell<T> for StdDataCell<T> {
    fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    #[inline(always)]
    fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R {
        f(self.0.get())
    }

    #[inline(always)]
    fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        f(self.0.get())
    }
}

const LIVE: u32 = 1 << 30;
const TOMBSTONE: u32 = 2 << 30;
const STATE_MASK: u32 = 3 << 30;
const FRAGMENT_MASK: u32 = (1 << 30) - 1;
#[cfg(not(loom))]
const MIN_CAPACITY: usize = 16;
// Four buckets under loom, so a model reaches a rehash in three inserts.
#[cfg(loom)]
const MIN_CAPACITY: usize = 4;

/// The 30 hash bits the bucket index does not use.
#[inline(always)]
fn fragment(hash: u64) -> u32 {
    (hash >> 34) as u32 & FRAGMENT_MASK
}

#[doc(hidden)]
pub struct Entry<K> {
    hash: u64,
    slot: u32,
    generation: u32,
    /// Initialized in every bucket that is not Empty.
    key: MaybeUninit<K>,
}

#[doc(hidden)]
pub struct Bucket<K: TableKey, C: Cells> {
    /// Two state bits (Empty, Live or Tombstone) and a hash fragment.
    meta: C::U32,
    data: C::DataCell<Entry<K>>,
}

/// One array of buckets. It owns the keys of its non-Empty buckets and
/// drops them with itself, so a key outlives every reader of its array.
#[doc(hidden)]
pub struct Array<K: TableKey, C: Cells> {
    buckets: Box<[Bucket<K, C>]>,
}

/// A rehash's result: the new array and its live count.
struct Rehashed<K: TableKey, C: Cells> {
    array: Array<K, C>,
    live: usize,
}

/// What the probe under the miss mutex found.
enum Locked {
    Hit(u32, u32),
    /// A miss, after tombstoning this many dead buckets of the key.
    Miss(usize),
}

impl<K: TableKey, C: Cells> Array<K, C> {
    fn with_capacity(capacity: usize) -> Self {
        debug_assert!(capacity.is_power_of_two());
        Self {
            buckets: (0..capacity)
                .map(|_| Bucket {
                    meta: C::new_u32(0),
                    data: <C::DataCell<Entry<K>> as DataCell<_>>::new(Entry {
                        hash: 0,
                        slot: 0,
                        generation: 0,
                        key: MaybeUninit::uninit(),
                    }),
                })
                .collect(),
        }
    }

    #[inline(always)]
    fn capacity(&self) -> usize {
        self.buckets.len()
    }

    /// Runs `visit` on each Live bucket of `hash`'s fragment along the
    /// probe sequence, until it returns `Some`. The probe ends at an Empty
    /// bucket or after `capacity` steps, so even a full array is a miss,
    /// never a hang.
    #[inline(always)]
    fn probe<R>(
        &self,
        hash: u64,
        mut visit: impl FnMut(&Bucket<K, C>, &Entry<K>) -> Option<R>,
    ) -> Option<R> {
        let mask = self.capacity() - 1;
        let fragment = fragment(hash);
        let mut i = hash as usize & mask;
        for _ in 0..self.capacity() {
            let bucket = &self.buckets[i];
            let meta = C::u32_load_acquire(&bucket.meta);
            match meta & STATE_MASK {
                0 => return None,
                LIVE if meta & FRAGMENT_MASK == fragment => {
                    // SAFETY: the Acquire load saw the Release store that
                    // published this bucket's data, which is never
                    // written again in this array.
                    let found = bucket.data.with(|e| visit(bucket, unsafe { &*e }));
                    if found.is_some() {
                        return found;
                    }
                }
                _ => {}
            }
            i = (i + 1) & mask;
        }
        None
    }

    /// The live handle of `key`, if any. Lock-free.
    #[inline]
    fn find(&self, rt: &Runtime<C>, hash: u64, key: &K) -> Option<(u32, u32)> {
        self.probe(hash, |_, e| {
            // SAFETY: a Live bucket's key is initialized.
            let equal =
                e.hash == hash && value_code::run(|| unsafe { e.key.assume_init_ref() } == key);
            (equal && rt.is_alive(ErasedIncr::new(e.slot, e.generation, rt.runtime_id)))
                .then_some((e.slot, e.generation))
        })
    }

    /// [`Self::find`] under the miss mutex, which also tombstones the
    /// key's dead buckets.
    fn find_locked(&self, rt: &Runtime<C>, hash: u64, key: &K) -> Locked {
        let mut tombstoned = 0;
        let hit = self.probe(hash, |bucket, e| {
            // SAFETY: a Live bucket's key is initialized.
            if e.hash != hash || !value_code::run(|| unsafe { e.key.assume_init_ref() } == key) {
                return None;
            }
            if rt.is_alive(ErasedIncr::new(e.slot, e.generation, rt.runtime_id)) {
                return Some((e.slot, e.generation));
            }
            Self::tombstone(bucket);
            tombstoned += 1;
            None
        });
        match hit {
            Some((slot, generation)) => Locked::Hit(slot, generation),
            None => Locked::Miss(tombstoned),
        }
    }

    /// Tombstones the Live bucket holding `(slot, generation)` under
    /// `hash`, if one does. Runs no user code. Under the miss mutex.
    fn tombstone_handle(&self, hash: u64, slot: u32, generation: u32) -> bool {
        self.probe(hash, |bucket, e| {
            (e.hash == hash && e.slot == slot && e.generation == generation)
                .then(|| Self::tombstone(bucket))
        })
        .is_some()
    }

    fn tombstone(bucket: &Bucket<K, C>) {
        let meta = C::u32_load_relaxed(&bucket.meta);
        C::u32_store_release(&bucket.meta, TOMBSTONE | (meta & FRAGMENT_MASK));
    }

    /// Writes an entry into the first Empty bucket of `hash`'s probe
    /// sequence and publishes it. Under the miss mutex, with room left by
    /// the load factor. Runs no user code.
    fn insert(&self, hash: u64, slot: u32, generation: u32, key: K) {
        let mask = self.capacity() - 1;
        let mut i = hash as usize & mask;
        for _ in 0..self.capacity() {
            let bucket = &self.buckets[i];
            if C::u32_load_relaxed(&bucket.meta) & STATE_MASK == 0 {
                // SAFETY: the bucket is Empty, so no reader reads its data
                // (a probe stops at Empty), and the miss mutex keeps every
                // other writer out.
                bucket.data.with_mut(|e| unsafe {
                    (*e).hash = hash;
                    (*e).slot = slot;
                    (*e).generation = generation;
                    (*e).key.write(key);
                });
                C::u32_store_release(&bucket.meta, LIVE | fragment(hash));
                return;
            }
            i = (i + 1) & mask;
        }
        unreachable!("incr-core: a table array filled past its load factor");
    }

    /// A new array holding a clone of each Live entry whose node is
    /// alive, sized for twice that many plus one. A `K::clone` that
    /// panics leaves the partial array to `bag`, so its cloned keys drop
    /// after the miss mutex is released.
    fn rehashed(&self, rt: &Runtime<C>, bag: &mut Deferred<C>) -> Rehashed<K, C> {
        let mut alive = Vec::new();
        for bucket in self.buckets.iter() {
            if C::u32_load_relaxed(&bucket.meta) & STATE_MASK == LIVE {
                // SAFETY: a Live bucket's data is published and never
                // written again in this array.
                let e = bucket.data.with(|e| unsafe { &*e });
                if rt.is_alive(ErasedIncr::new(e.slot, e.generation, rt.runtime_id)) {
                    alive.push(e);
                }
            }
        }
        let capacity = (2 * (alive.len() + 1))
            .max(MIN_CAPACITY)
            .next_power_of_two();

        struct Partial<'b, K: TableKey, C: Cells> {
            array: Option<Array<K, C>>,
            bag: &'b mut Deferred<C>,
        }
        impl<K: TableKey, C: Cells> Drop for Partial<'_, K, C> {
            fn drop(&mut self) {
                if let Some(array) = self.array.take() {
                    self.bag.push_erased(Box::new(array));
                }
            }
        }

        let mut partial = Partial {
            array: Some(Array::with_capacity(capacity)),
            bag,
        };
        let array = partial.array.as_ref().expect("set above");
        for e in &alive {
            // SAFETY: a Live bucket's key is initialized.
            let key = value_code::run(|| unsafe { e.key.assume_init_ref() }.clone());
            array.insert(e.hash, e.slot, e.generation, key);
        }
        Rehashed {
            array: partial.array.take().expect("set above"),
            live: alive.len(),
        }
    }
}

impl<K: TableKey, C: Cells> Drop for Array<K, C> {
    fn drop(&mut self) {
        drop_keys(&mut self.buckets);
    }
}

/// Drops the key of every non-Empty bucket. A key whose `Drop` panics
/// leaves the rest to a guard that drops them during the unwind, so a
/// second panicking key aborts, as it would in a `Vec`.
fn drop_keys<K: TableKey, C: Cells>(buckets: &mut [Bucket<K, C>]) {
    struct Rest<'a, K: TableKey, C: Cells>(&'a mut [Bucket<K, C>]);
    impl<K: TableKey, C: Cells> Drop for Rest<'_, K, C> {
        fn drop(&mut self) {
            if !self.0.is_empty() {
                drop_keys(&mut *self.0);
            }
        }
    }

    let mut rest = Rest(buckets);
    while let Some((first, tail)) = std::mem::take(&mut rest.0).split_first_mut() {
        rest.0 = tail;
        if C::u32_load_relaxed(&first.meta) & STATE_MASK != 0 {
            // SAFETY: a non-Empty bucket's key is initialized, and the
            // array is being dropped, so nothing reads it again.
            first
                .data
                .with_mut(|e| unsafe { (*e).key.assume_init_drop() });
        }
    }
}

/// The Live and Tombstone buckets of the current array, under the miss
/// mutex.
struct Counts {
    live: usize,
    tombstones: usize,
}

/// A map from keys to nodes of the runtime that created it, through
/// [`Runtime::create_table`]. A hit returns the existing entry whichever
/// call created it. Dropping the table leaves its nodes alive.
pub struct NodeTable<K: TableKey, T: Value, C: Cells, S = FxBuildHasher> {
    runtime_id: RuntimeId,
    array: C::TableArray<K>,
    counts: OrderedLock<Counts, C>,
    hasher: S,
    _value: PhantomData<fn() -> T>,
}

// SAFETY: every method takes the runtime and checks that it is the
// table's own before it touches any other field, and a `Runtime<Local>` is
// neither `Send` nor `Sync`, so the `Rc` and `RefCell`s inside are only
// ever used on the one thread that runtime lives on. Another thread can
// only reach that check and panic, or drop the table, which needs it
// owned, so nothing on the runtime's thread can be using it then. This is
// what lets a Local compute closure, which must be `Send + Sync`, capture
// a table.
unsafe impl<K: TableKey, T: Value, S: Send + Sync> Send for NodeTable<K, T, Local, S> {}
// SAFETY: see the `Send` impl.
unsafe impl<K: TableKey, T: Value, S: Send + Sync> Sync for NodeTable<K, T, Local, S> {}

// The `Local` impls above turn off the automatic ones for every strategy,
// so the thread-safe strategies state theirs. SAFETY: every field is
// `Send` and `Sync` under them on its own, which `_fields_are_thread_safe`
// checks.
unsafe impl<K: TableKey, T: Value, S: Send + Sync> Send for NodeTable<K, T, Shared, S> {}
// SAFETY: as for `Send`.
unsafe impl<K: TableKey, T: Value, S: Send + Sync> Sync for NodeTable<K, T, Shared, S> {}
// SAFETY: as for `Shared`.
#[cfg(loom)]
unsafe impl<K: TableKey, T: Value, S: Send + Sync> Send
    for NodeTable<K, T, crate::loom_cells::Loom, S>
{
}
// SAFETY: as for `Shared`.
#[cfg(loom)]
unsafe impl<K: TableKey, T: Value, S: Send + Sync> Sync
    for NodeTable<K, T, crate::loom_cells::Loom, S>
{
}

#[allow(dead_code)]
fn _fields_are_thread_safe<K: TableKey>() {
    fn thread_safe<X: Send + Sync>() {}
    thread_safe::<<Shared as Cells>::TableArray<K>>();
    thread_safe::<OrderedLock<Counts, Shared>>();
    #[cfg(loom)]
    thread_safe::<<crate::loom_cells::Loom as Cells>::TableArray<K>>();
    #[cfg(loom)]
    thread_safe::<OrderedLock<Counts, crate::loom_cells::Loom>>();
}

impl<K: TableKey, T: Value, C: Cells, S: BuildHasher> NodeTable<K, T, C, S> {
    pub(crate) fn new(runtime_id: RuntimeId, hasher: S) -> Self {
        Self {
            runtime_id,
            array: <C::TableArray<K> as ArrayCell<_>>::new(Array::with_capacity(MIN_CAPACITY)),
            counts: OrderedLock::new(
                LockKind::TableMiss,
                Counts {
                    live: 0,
                    tombstones: 0,
                },
            ),
            hasher,
            _value: PhantomData,
        }
    }

    /// The input node for `key`, created with `init`'s value on a miss.
    /// `init` runs with no lock held and may use any table.
    pub fn get_or_insert_input(
        &self,
        rt: &Runtime<C>,
        key: K,
        init: impl FnOnce() -> T,
    ) -> Incr<T> {
        self.check_runtime(rt);
        let hash = self.hash(&key);
        if let Some(handle) = self.find(rt, hash, &key) {
            return handle;
        }
        if rt.frame_doomed() {
            return rt.pending_handle();
        }
        let value = init();
        self.miss_dropping(rt, hash, key, value, |rt, value| {
            rt.create_input_unchecked(value)
        })
    }

    /// The query node for `key`, created with `f` on a miss.
    pub fn get_or_insert_query<F>(&self, rt: &Runtime<C>, key: K, f: F) -> Incr<T>
    where
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.query(rt, key, f, ErasedOpts::NONE)
    }

    /// [`Self::get_or_insert_query`] with options for the query a miss
    /// creates.
    pub fn get_or_insert_query_with<F>(
        &self,
        rt: &Runtime<C>,
        key: K,
        f: F,
        opts: QueryOpts<T>,
    ) -> Incr<T>
    where
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.query(rt, key, f, cycle::erase_opts(opts))
    }

    fn query<F>(&self, rt: &Runtime<C>, key: K, f: F, opts: ErasedOpts) -> Incr<T>
    where
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.check_runtime(rt);
        let hash = self.hash(&key);
        if let Some(handle) = self.find(rt, hash, &key) {
            return handle;
        }
        if rt.frame_doomed() {
            return rt.pending_handle();
        }
        self.miss_dropping(rt, hash, key, (f, opts), |rt, (f, opts)| {
            rt.create_query_erased::<false, _, _>(f, opts)
        })
    }

    /// [`Self::get_or_insert_query_with`] that hands a node the miss
    /// creates to `seed` under the miss mutex, before the node is in the
    /// table, so no reader has claimed it. If `seed` panics, the node is
    /// deleted after the mutex and the panic resumes; a panic after the
    /// insert leaves it. Whatever the call does not keep (the key and `f`
    /// on a hit, a value `seed` returns, what a growth retires) goes to
    /// `bag`, for a caller that holds its own lock to drain after it.
    pub(crate) fn get_or_insert_query_seeded<F>(
        &self,
        rt: &Runtime<C>,
        key: K,
        f: F,
        opts: QueryOpts<T>,
        seed: impl FnOnce(&Runtime<C>, Incr<T>) -> Option<T>,
        bag: &mut Deferred<C>,
    ) -> Incr<T>
    where
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.check_runtime(rt);
        let hash = self.hash(&key);
        let found = self.find(rt, hash, &key);
        if found.is_none() && !rt.frame_doomed() {
            return self.seeded_miss(rt, hash, key, f, opts, seed, bag);
        }
        bag.push((key, f, opts));
        found.unwrap_or_else(|| rt.pending_handle())
    }

    #[allow(clippy::too_many_arguments)]
    fn seeded_miss<F>(
        &self,
        rt: &Runtime<C>,
        hash: u64,
        key: K,
        f: F,
        opts: QueryOpts<T>,
        seed: impl FnOnce(&Runtime<C>, Incr<T>) -> Option<T>,
        bag: &mut Deferred<C>,
    ) -> Incr<T>
    where
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        let opts = cycle::erase_opts(opts);
        let mut created = None;
        // Once `seed` returns the node may be in the table, so an unwind
        // after that must leave the published node alone.
        let mut seeded = false;
        let mut unseeded = None;
        let missed = catch_unwind(AssertUnwindSafe(|| {
            self.miss(rt, hash, key, (f, opts), &mut *bag, |rt, (f, opts)| {
                let q = rt.create_query_erased::<false, _, _>(f, opts);
                created = Some(q);
                unseeded = seed(rt, q);
                seeded = true;
                q
            })
        }));
        if let Some(value) = unseeded {
            bag.push(value);
        }
        match missed {
            Ok(Ok(handle)) => handle,
            Ok(Err((handle, key, payload))) => {
                bag.push((key, payload));
                handle
            }
            Err(unwind) => {
                if let (Some(q), false) = (created, seeded) {
                    isolate(|| match rt.try_delete_node(q) {
                        // q never reached the table, so only the hook could
                        // have deleted it or computed a node against it.
                        Ok(()) | Err(DeleteError::NotAlive | DeleteError::HasDependents) => {}
                    });
                }
                bag.drain_carried(unwind)
            }
        }
    }

    pub(crate) fn probe_unchecked(&self, rt: &Runtime<C>, key: &K) -> Option<Incr<T>> {
        self.check_runtime(rt);
        self.find(rt, self.hash(key), key)
    }

    /// The node for `key`, if the table holds a live one. Not a
    /// dependency, and so not allowed inside a compute of this runtime.
    pub fn get(&self, rt: &Runtime<C>, key: &K) -> Option<Incr<T>> {
        self.check_runtime(rt);
        self.refuse_in_compute(rt, "get");
        self.find(rt, self.hash(key), key)
    }

    /// Deletes `key`'s node with [`Runtime::try_delete_node`] and removes
    /// the entry. `Err(NotAlive)` if the table holds no live node for it.
    pub fn remove(&self, rt: &Runtime<C>, key: &K) -> Result<(), DeleteError> {
        self.check_runtime(rt);
        let hash = self.hash(key);
        let handle = self.find(rt, hash, key).ok_or(DeleteError::NotAlive)?;
        rt.try_delete_node(handle)?;
        let mut counts = self.counts.write();
        if self
            .array
            .read(|array| array.tombstone_handle(hash, handle.slot(), handle.generation()))
        {
            counts.live -= 1;
            counts.tombstones += 1;
        }
        Ok(())
    }

    /// Calls `f` for each entry whose node is alive, over one snapshot
    /// of the array: an entry live for the whole call is visited exactly
    /// once, each pair at most once, and entries inserted or removed
    /// during the call may or may not be. `f` may insert and remove.
    pub fn for_each(&self, rt: &Runtime<C>, mut f: impl FnMut(&K, Incr<T>)) {
        self.check_runtime(rt);
        self.refuse_in_compute(rt, "for_each");
        self.array.read(|array| {
            for bucket in array.buckets.iter() {
                if C::u32_load_acquire(&bucket.meta) & STATE_MASK != LIVE {
                    continue;
                }
                bucket.data.with(|e| {
                    // SAFETY: the Acquire load saw this bucket published,
                    // and the array stays allocated for the whole read.
                    let e = unsafe { &*e };
                    let handle = Incr::new(e.slot, e.generation, self.runtime_id);
                    if rt.is_alive(handle) {
                        // SAFETY: a Live bucket's key is initialized.
                        f(unsafe { e.key.assume_init_ref() }, handle);
                    }
                });
            }
        });
    }

    /// The current array's capacity and its Live and Tombstone counts.
    #[cfg(all(test, not(loom)))]
    pub(crate) fn shape(&self) -> (usize, usize, usize) {
        let counts = self.counts.read();
        (
            self.array.read(Array::capacity),
            counts.live,
            counts.tombstones,
        )
    }

    #[inline]
    fn hash(&self, key: &K) -> u64 {
        value_code::run(|| self.hasher.hash_one(key))
    }

    #[inline]
    fn find(&self, rt: &Runtime<C>, hash: u64, key: &K) -> Option<Incr<T>> {
        self.array
            .read(|array| array.find(rt, hash, key))
            .map(|(slot, generation)| Incr::new(slot, generation, self.runtime_id))
    }

    /// [`Self::miss`] that drops what a hit leaves unused and drains its
    /// own bag.
    fn miss_dropping<P>(
        &self,
        rt: &Runtime<C>,
        hash: u64,
        key: K,
        payload: P,
        create: impl FnOnce(&Runtime<C>, P) -> Incr<T>,
    ) -> Incr<T> {
        let mut bag = Deferred::<C>::new();
        let handle = match self.miss(rt, hash, key, payload, &mut bag, create) {
            Ok(handle) => handle,
            Err((handle, key, payload)) => {
                drop(payload);
                drop(key);
                handle
            }
        };
        bag.drain(rt);
        handle
    }

    /// Steps 4 to 7 of a miss: probe again under the miss mutex, rehash
    /// if the insert needs room, create the node with `create` and insert
    /// it. A live hit under the mutex unlocks and returns the winner in
    /// `Err` with the unused `key` and `payload`. What a rehash retires
    /// goes to `bag`, for the caller to drain.
    #[allow(clippy::type_complexity)]
    fn miss<P>(
        &self,
        rt: &Runtime<C>,
        hash: u64,
        key: K,
        payload: P,
        bag: &mut Deferred<C>,
        create: impl FnOnce(&Runtime<C>, P) -> Incr<T>,
    ) -> Result<Incr<T>, (Incr<T>, K, P)> {
        // Checked before the mutex, so a key method that misses panics
        // with the contract message instead of re-entering it.
        value_code::check();
        let mut counts = self.counts.write();
        match self.array.read(|array| array.find_locked(rt, hash, &key)) {
            Locked::Hit(slot, generation) => {
                drop(counts);
                return Err((Incr::new(slot, generation, self.runtime_id), key, payload));
            }
            Locked::Miss(tombstoned) => {
                counts.live -= tombstoned;
                counts.tombstones += tombstoned;
            }
        }
        let capacity = self.array.read(Array::capacity);
        if (counts.live + counts.tombstones + 1) * 8 > capacity * 7 {
            match catch_unwind(AssertUnwindSafe(|| {
                self.array.read(|old| old.rehashed(rt, &mut *bag))
            })) {
                Ok(rehashed) => {
                    bag.push_erased(Box::new(self.array.replace(rehashed.array)));
                    counts.live = rehashed.live;
                    counts.tombstones = 0;
                }
                Err(unwind) => {
                    drop(counts);
                    drop(payload);
                    drop(key);
                    bag.drain_carried(unwind);
                }
            }
        }
        let handle = create(rt, payload);
        self.array
            .read(|array| array.insert(hash, handle.slot(), handle.generation(), key));
        counts.live += 1;
        drop(counts);
        Ok(handle)
    }

    fn check_runtime(&self, rt: &Runtime<C>) {
        if rt.runtime_id != self.runtime_id {
            panic!(
                "incr-core: NodeTable of runtime {} used with runtime {}",
                self.runtime_id.get(),
                rt.runtime_id.get(),
            );
        }
    }

    fn refuse_in_compute(&self, rt: &Runtime<C>, call: &str) {
        assert!(
            !rt.dep_stack.current_frame_active(rt.runtime_id),
            "incr-core: NodeTable::{call} called during compute; a lookup is not a \
             dependency, so a compute finds its nodes through get_or_insert_*",
        );
    }
}

#[cfg(all(test, not(loom)))]
#[path = "table_tests.rs"]
mod tests;
