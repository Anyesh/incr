//! `GridTable<T, C>`: a [`NodeTable`](crate::NodeTable) for positional
//! keys, `(sheet, row, column)`, that finds a node by indexing instead of
//! hashing.
//!
//! Each sheet has a directory of columns. A column reaches its pages of
//! [`PAGE_ROWS`] consecutive rows through a directory and two fixed
//! levels of 64 entries, each indexed by six more bits of the row, so it
//! costs memory by the pages in use rather than by its largest row. A
//! walk down a column reads adjacent memory and an entry costs 8 bytes in
//! a page that is in use.
//! An entry is one word, the node's slot and generation, written under
//! the table's miss mutex with a Release store and read with an Acquire
//! load. Directories, pages and the column and sheet levels are never
//! freed before the table drops (see [`Directory`]), so a lock-free hit
//! reads no memory a writer can free, under `Local` and `Shared` alike.
//!
//! A lookup is not a dependency, as for `NodeTable`: a hit returns a node
//! whatever created it, a miss in a doomed frame returns a pending handle
//! and creates nothing, and `get` refuses a compute of its runtime.

use std::marker::PhantomData;

use crate::cells::{Cells, Local, PtrCell, Shared};
use crate::cycle::{ErasedOpts, QueryOpts};
use crate::dep_stack::DepStack;
use crate::directory::Directory;
use crate::handle::{Incr, RuntimeId};
use crate::lock_order::LockKind;
use crate::locks::OrderedLock;
use crate::runtime::Runtime;
use crate::value::Value;
use crate::value_code;

/// Rows per page.
pub const PAGE_ROWS: u32 = 64;
const PAGE_SHIFT: u32 = PAGE_ROWS.trailing_zeros();
/// Row bits each directory level of a column indexes.
const LEVEL_SHIFT: u32 = 6;
const LEVEL_MASK: u32 = (1 << LEVEL_SHIFT) - 1;
const CHUNK_SHIFT: u32 = PAGE_SHIFT + LEVEL_SHIFT;
const MID_SHIFT: u32 = CHUNK_SHIFT + LEVEL_SHIFT;

/// The key of a [`GridTable`] entry. Each coordinate has a bound, so a
/// stray key cannot make the table allocate gigabytes of directory: a
/// sheet's directory grows with its largest column, and a column's top
/// level, at most 64 entries under `ROWS`, with its largest row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GridKey {
    pub sheet: u32,
    pub row: u32,
    pub column: u32,
}

impl GridKey {
    /// One more than the largest sheet a key may name.
    pub const SHEETS: u32 = 1 << 16;
    /// One more than the largest row a key may name.
    pub const ROWS: u32 = 1 << 24;
    /// One more than the largest column a key may name.
    pub const COLUMNS: u32 = 1 << 16;

    pub fn new(sheet: u32, row: u32, column: u32) -> Self {
        Self { sheet, row, column }
    }

    #[inline(always)]
    fn check(self) {
        if self.sheet >= Self::SHEETS || self.row >= Self::ROWS || self.column >= Self::COLUMNS {
            out_of_range(self);
        }
    }
}

#[cold]
#[inline(never)]
fn out_of_range(key: GridKey) -> ! {
    panic!(
        "incr-core: GridKey {key:?} out of range (sheet < {}, row < {}, column < {})",
        GridKey::SHEETS,
        GridKey::ROWS,
        GridKey::COLUMNS,
    );
}

const EMPTY: u64 = 0;
/// An entry an insert of a batch holds until its node exists. No node
/// encodes to it, since a slot plus one is below `u32::MAX`.
const CLAIMED: u64 = u64::MAX;

/// A node as an entry: its generation above its slot plus one, so that no
/// node encodes to `EMPTY`. A slot is below `MAX_NODES`, so plus one fits.
#[inline(always)]
fn encode<T>(handle: Incr<T>) -> u64 {
    (u64::from(handle.generation()) << 32) | u64::from(handle.slot() + 1)
}

/// A directory that owns its entries: each non-null one came from
/// `Box::into_raw` and is freed when the directory drops.
struct Owning<T: 'static, C: Cells>(Directory<T, C>);

impl<T: 'static, C: Cells> Owning<T, C> {
    fn new(len: usize) -> Self {
        Self(Directory::with_len(len))
    }

    #[inline(always)]
    fn get(&self, index: u32) -> Option<&T> {
        let p = self.0.get(index as usize)?.load_acquire();
        // SAFETY: a non-null entry came from `Box::into_raw` in
        // `get_or_create` and is freed only when `self` drops.
        (!p.is_null()).then(|| unsafe { &*p })
    }

    /// The entry at `index`, created with `new` if it is null. Under the
    /// table's miss mutex, which serializes every write to the directory.
    fn get_or_create(&self, index: u32, new: impl FnOnce() -> T) -> &T {
        self.0.grow_to(index as usize);
        let cell = self.0.get(index as usize).expect("grown to cover index");
        let mut p = cell.load_relaxed();
        if p.is_null() {
            p = Box::into_raw(Box::new(new()));
            cell.store_release(p);
        }
        // SAFETY: as in `get`.
        unsafe { &*p }
    }
}

impl<T: 'static, C: Cells> Drop for Owning<T, C> {
    fn drop(&mut self) {
        for i in 0..self.0.len() {
            let p = self.0.get(i).expect("below len").load_relaxed();
            if !p.is_null() {
                // SAFETY: see `get`; `&mut self` rules out readers.
                drop(unsafe { Box::from_raw(p) });
            }
        }
    }
}

/// A level of `1 << LEVEL_SHIFT` entries that owns them as [`Owning`]
/// does. Fixed in size, so an index is one load with no length to check.
struct Fixed<T: 'static, C: Cells>([C::Ptr<T>; 1 << LEVEL_SHIFT]);

impl<T: 'static, C: Cells> Fixed<T, C> {
    fn new() -> Self {
        Self(std::array::from_fn(|_| C::Ptr::new_null()))
    }

    #[inline(always)]
    fn get(&self, index: u32) -> Option<&T> {
        let p = self.0[(index & LEVEL_MASK) as usize].load_acquire();
        // SAFETY: as in `Owning::get`.
        (!p.is_null()).then(|| unsafe { &*p })
    }

    /// As [`Owning::get_or_create`], under the table's miss mutex.
    fn get_or_create(&self, index: u32, new: impl FnOnce() -> T) -> &T {
        let cell = &self.0[(index & LEVEL_MASK) as usize];
        let mut p = cell.load_relaxed();
        if p.is_null() {
            p = Box::into_raw(Box::new(new()));
            cell.store_release(p);
        }
        // SAFETY: as in `Owning::get`.
        unsafe { &*p }
    }
}

impl<T: 'static, C: Cells> Drop for Fixed<T, C> {
    fn drop(&mut self) {
        for cell in &self.0 {
            let p = cell.load_relaxed();
            if !p.is_null() {
                // SAFETY: see `Owning::get`; `&mut self` rules out readers.
                drop(unsafe { Box::from_raw(p) });
            }
        }
    }
}

struct Page<C: Cells>([C::U64; PAGE_ROWS as usize]);

type Chunk<C> = Fixed<Page<C>, C>;
type Mid<C> = Fixed<Chunk<C>, C>;
type Column<C> = Owning<Mid<C>, C>;
type Sheet<C> = Owning<Column<C>, C>;

/// A map from `(sheet, row, column)` to nodes of the runtime that created
/// it, through [`Runtime::create_grid_table`]. A hit returns the existing
/// entry whichever call created it. Dropping the table leaves its nodes
/// alive.
pub struct GridTable<T: Value, C: Cells> {
    runtime_id: RuntimeId,
    sheets: Owning<Sheet<C>, C>,
    miss: OrderedLock<(), C>,
    _value: PhantomData<fn() -> T>,
}

// SAFETY: every method takes the runtime and checks that it is the
// table's own before it touches any other field, and a `Runtime<Local>` is
// neither `Send` nor `Sync`, so the `Cell`s inside are only ever used on
// the one thread that runtime lives on. Another thread can only reach that
// check and panic, or drop the table, which needs it owned and frees only
// plain allocations. This lets a Local compute closure, which must be
// `Send + Sync`, capture a table.
unsafe impl<T: Value> Send for GridTable<T, Local> {}
// SAFETY: see the `Send` impl.
unsafe impl<T: Value> Sync for GridTable<T, Local> {}
// The `Local` impls above turn off the automatic ones for every strategy.
// SAFETY: every field is `Send` and `Sync` under `Shared` on its own,
// which `_fields_are_thread_safe` checks, and the boxes the directories
// own hold only those fields and atomics.
unsafe impl<T: Value> Send for GridTable<T, Shared> {}
// SAFETY: as for `Send`.
unsafe impl<T: Value> Sync for GridTable<T, Shared> {}
// SAFETY: as for `Shared`.
#[cfg(loom)]
unsafe impl<T: Value> Send for GridTable<T, crate::loom_cells::Loom> {}
// SAFETY: as for `Shared`.
#[cfg(loom)]
unsafe impl<T: Value> Sync for GridTable<T, crate::loom_cells::Loom> {}

#[allow(dead_code)]
fn _fields_are_thread_safe() {
    fn thread_safe<X: Send + Sync>() {}
    thread_safe::<Directory<Sheet<Shared>, Shared>>();
    thread_safe::<<Shared as Cells>::U64>();
    thread_safe::<OrderedLock<(), Shared>>();
}

impl<T: Value, C: Cells> GridTable<T, C> {
    pub(crate) fn new(runtime_id: RuntimeId) -> Self {
        Self {
            runtime_id,
            sheets: Owning::new(1),
            miss: OrderedLock::new(LockKind::TableMiss, ()),
            _value: PhantomData,
        }
    }

    /// The input node for `key`, created with `init`'s value on a miss.
    /// `init` runs with no lock held and may use any table.
    pub fn get_or_insert_input(
        &self,
        rt: &Runtime<C>,
        key: GridKey,
        init: impl FnOnce() -> T,
    ) -> Incr<T> {
        self.check_runtime(rt);
        // Only `entry_or_create` makes entries, after this check, so an
        // out-of-range key never hits and is checked on the miss.
        if let Some(handle) = self.find(rt, key) {
            return handle;
        }
        key.check();
        if rt.frame_doomed() {
            return rt.pending_handle();
        }
        let value = init();
        self.miss(rt, key, value, |rt, value| rt.create_input_unchecked(value))
    }

    /// The query node for `key`, created with `f` on a miss.
    pub fn get_or_insert_query<F>(&self, rt: &Runtime<C>, key: GridKey, f: F) -> Incr<T>
    where
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.query(rt, key, || (f, ErasedOpts::NONE))
    }

    /// [`Self::get_or_insert_query`] with options for the query a miss
    /// creates.
    pub fn get_or_insert_query_with<F>(
        &self,
        rt: &Runtime<C>,
        key: GridKey,
        f: F,
        opts: QueryOpts<T>,
    ) -> Incr<T>
    where
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.query(rt, key, || (f, rt.erase_query_opts(opts)))
    }

    /// [`Self::get_or_insert_query_with`] with the compute and options
    /// built by `make` only on a miss, so a hit pays nothing for options
    /// that allocate. `make` runs with no lock held, so it may run on a
    /// miss that does not create the query: when threads miss the same
    /// key together each runs `make`, or when `make` itself creates the
    /// key; every build but the one that created the node is dropped
    /// unused and the call returns the created node.
    pub fn get_or_insert_query_lazy<F>(
        &self,
        rt: &Runtime<C>,
        key: GridKey,
        make: impl FnOnce() -> (F, QueryOpts<T>),
    ) -> Incr<T>
    where
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.query(rt, key, || {
            let (f, opts) = make();
            (f, rt.erase_query_opts(opts))
        })
    }

    #[inline]
    fn query<F>(
        &self,
        rt: &Runtime<C>,
        key: GridKey,
        make: impl FnOnce() -> (F, ErasedOpts),
    ) -> Incr<T>
    where
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.check_runtime(rt);
        // Only `entry_or_create` makes entries, after this check, so an
        // out-of-range key never hits and is checked on the miss.
        if let Some(handle) = self.find(rt, key) {
            return handle;
        }
        key.check();
        if rt.frame_doomed() {
            return rt.pending_handle();
        }
        let (f, opts) = make();
        self.miss(rt, key, (f, opts), |rt, (f, opts)| {
            rt.create_query_erased::<false, _, _>(f, opts)
        })
    }

    /// The node for `key`, if the table holds a live one. Not a
    /// dependency, and so not allowed inside a compute of this runtime.
    pub fn get(&self, rt: &Runtime<C>, key: GridKey) -> Option<Incr<T>> {
        self.check_runtime(rt);
        key.check();
        self.refuse_in_compute(rt, "get");
        self.find(rt, key)
    }

    /// Creates an input node for each `(key, value)`, all under one hold
    /// of the runtime's lock, and enters them. Returns the handles in
    /// order. Not allowed inside a compute; panics, creating nothing, if
    /// a key has a live node or appears twice.
    pub fn insert_inputs(&self, rt: &Runtime<C>, entries: Vec<(GridKey, T)>) -> Vec<Incr<T>> {
        self.check_runtime(rt);
        self.refuse_in_compute(rt, "insert_inputs");
        let (keys, values): (Vec<GridKey>, Vec<T>) = entries.into_iter().unzip();
        self.insert_many(rt, &keys, values, |rt, values| rt.create_inputs(values))
    }

    /// Creates a query node for each `(key, f, opts)`, all under one hold
    /// of the runtime's lock, watches each as [`Runtime::watch`] does, and
    /// enters them. Returns the handles in order. Not allowed inside a
    /// compute; panics, creating nothing, if a key has a live node or
    /// appears twice.
    pub fn insert_queries_watched<F>(
        &self,
        rt: &Runtime<C>,
        entries: Vec<(GridKey, F, QueryOpts<T>)>,
    ) -> Vec<Incr<T>>
    where
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.check_runtime(rt);
        self.refuse_in_compute(rt, "insert_queries_watched");
        let mut keys = Vec::with_capacity(entries.len());
        let queries: Vec<(F, ErasedOpts)> = entries
            .into_iter()
            .map(|(key, f, opts)| {
                keys.push(key);
                (f, rt.erase_query_opts(opts))
            })
            .collect();
        self.insert_many(rt, &keys, queries, |rt, queries| {
            rt.create_queries_watched(queries)
        })
    }

    /// Checks every key under the miss mutex before `create` makes any
    /// node, so a refused batch leaves nothing behind.
    fn insert_many<P>(
        &self,
        rt: &Runtime<C>,
        keys: &[GridKey],
        payload: P,
        create: impl FnOnce(&Runtime<C>, P) -> Vec<Incr<T>>,
    ) -> Vec<Incr<T>> {
        value_code::check();
        let miss = self.miss.write();
        // Each entry of the batch holds `CLAIMED` until its node exists,
        // so a repeated key finds its entry taken; dropping the claims
        // without committing them puts the old words back.
        struct Claims<'a, C: Cells>(Vec<(&'a C::U64, u64)>);
        impl<C: Cells> Drop for Claims<'_, C> {
            fn drop(&mut self) {
                for &(entry, word) in self.0.iter().rev() {
                    C::u64_store_relaxed(entry, word);
                }
            }
        }
        let mut claims = Claims::<C>(Vec::with_capacity(keys.len()));
        for &key in keys {
            key.check();
            let entry = self.entry_or_create(key);
            let word = C::u64_load_relaxed(entry);
            let refused = if word == CLAIMED {
                Some("appears twice")
            } else if self.live(rt, word).is_some() {
                Some("has a live node")
            } else {
                None
            };
            if let Some(why) = refused {
                drop(claims);
                drop(miss);
                panic!("incr-core: GridTable insert refused: {key:?} {why}");
            }
            C::u64_store_relaxed(entry, CLAIMED);
            claims.0.push((entry, word));
        }
        let handles = create(rt, payload);
        for ((entry, _), &handle) in std::mem::take(&mut claims.0).into_iter().zip(&handles) {
            C::u64_store_release(entry, encode(handle));
        }
        drop(miss);
        handles
    }

    /// The handle in `word`, if it names a live node of this runtime.
    #[inline(always)]
    fn live(&self, rt: &Runtime<C>, word: u64) -> Option<Incr<T>> {
        if word == EMPTY || word == CLAIMED {
            return None;
        }
        let (slot, generation) = ((word as u32) - 1, (word >> 32) as u32);
        // `rt` is this table's runtime: every caller ran `check_runtime`.
        rt.is_alive_at(slot, generation)
            .then(|| Incr::new(slot, generation, self.runtime_id))
    }

    #[inline(always)]
    fn entry(&self, key: GridKey) -> Option<&C::U64> {
        let page = self
            .sheets
            .get(key.sheet)?
            .get(key.column)?
            .get(key.row >> MID_SHIFT)?
            .get(key.row >> CHUNK_SHIFT)?
            .get(key.row >> PAGE_SHIFT)?;
        Some(&page.0[(key.row & (PAGE_ROWS - 1)) as usize])
    }

    /// [`Self::entry`], creating the levels it passes through. Under the
    /// miss mutex.
    fn entry_or_create(&self, key: GridKey) -> &C::U64 {
        let page = self
            .sheets
            .get_or_create(key.sheet, || Owning::new(1))
            .get_or_create(key.column, || Owning::new(1))
            .get_or_create(key.row >> MID_SHIFT, Fixed::new)
            .get_or_create(key.row >> CHUNK_SHIFT, Fixed::new)
            .get_or_create(key.row >> PAGE_SHIFT, || {
                Page(std::array::from_fn(|_| C::new_u64(0)))
            });
        &page.0[(key.row & (PAGE_ROWS - 1)) as usize]
    }

    #[inline]
    fn find(&self, rt: &Runtime<C>, key: GridKey) -> Option<Incr<T>> {
        self.live(rt, C::u64_load_acquire(self.entry(key)?))
    }

    /// Probes again under the miss mutex and creates the node with
    /// `create` if the entry still has no live one. A live hit there
    /// drops `payload` after unlocking and returns the winner. A frame
    /// doomed while `payload` was built drops it and gets a pending
    /// handle, since the build may hold pending handles of its own.
    fn miss<P>(
        &self,
        rt: &Runtime<C>,
        key: GridKey,
        payload: P,
        create: impl FnOnce(&Runtime<C>, P) -> Incr<T>,
    ) -> Incr<T> {
        if rt.frame_doomed() {
            drop(payload);
            return rt.pending_handle();
        }
        // Checked before the mutex, so a value method that misses panics
        // with the contract message instead of re-entering it.
        value_code::check();
        let miss = self.miss.write();
        let entry = self.entry_or_create(key);
        if let Some(handle) = self.live(rt, C::u64_load_relaxed(entry)) {
            drop(miss);
            drop(payload);
            return handle;
        }
        let handle = create(rt, payload);
        C::u64_store_release(entry, encode(handle));
        drop(miss);
        handle
    }

    fn check_runtime(&self, rt: &Runtime<C>) {
        if rt.runtime_id != self.runtime_id {
            panic!(
                "incr-core: GridTable of runtime {} used with runtime {}",
                self.runtime_id.get(),
                rt.runtime_id.get(),
            );
        }
    }

    fn refuse_in_compute(&self, rt: &Runtime<C>, call: &str) {
        assert!(
            !rt.dep_stack.current_frame_active(rt.runtime_id),
            "incr-core: GridTable::{call} called during compute; a lookup is not a \
             dependency, so a compute finds its nodes through get_or_insert_*",
        );
    }
}

#[cfg(all(test, not(loom)))]
#[path = "grid_table_tests.rs"]
mod tests;
