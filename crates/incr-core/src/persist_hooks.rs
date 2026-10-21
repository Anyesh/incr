//! Hooks for incr-persist's mapped writes, restores and checkpoints.
//! Unstable: incr-persist pins this crate's exact version, and nothing
//! else should call them.

use std::hash::{BuildHasher, Hash};

use crate::cells::Cells;
pub use crate::collection::Displaced;
use crate::collection::IncrCollection;
use crate::cycle::QueryOpts;
pub use crate::deferred::Deferred;
use crate::handle::{ErasedIncr, Incr, RuntimeId};
use crate::runtime::Runtime;
use crate::table::{NodeTable, TableKey};
use crate::value::Value;
use crate::value_slot::{self, Compared};

/// Which side of a collection mutation a representative is reported for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepSide {
    /// An occurrence of the element is removed; the representative is the
    /// value stored before the mutation.
    Removed,
    /// An occurrence of the element is added; the representative is the
    /// value stored after it.
    Added,
}

/// Receives the stored representative of each element occurrence a
/// collection mutation removes or adds. It is called under the log guard,
/// as value code, before the mutation changes anything, so a panic leaves
/// the collection as it was.
pub trait RepSink<T> {
    /// False only for `()`, whose reports compile away.
    const ACTIVE: bool = true;

    fn on_rep(&mut self, side: RepSide, rep: &T);
}

impl<T> RepSink<T> for () {
    const ACTIVE: bool = false;

    #[inline(always)]
    fn on_rep(&mut self, _: RepSide, _: &T) {}
}

/// [`Runtime::set`] that hands back what it would drop, for the caller to
/// drop after releasing its own locks: `Replaced` with the displaced value
/// if it wrote, `Equal` with the rejected value if it did not.
pub fn set_returning_displaced<T: Value, C: Cells>(
    rt: &Runtime<C>,
    handle: Incr<T>,
    value: T,
) -> Compared<T, value_slot::Displaced<T, C>> {
    rt.set_returning_displaced(handle, value)
}

/// An input's `changed_at`, for the stamp a mapped write records and the
/// bypass check that compares it. Settled once that input's `set` has
/// returned; `u64::MAX`, which matches no recorded stamp, while another
/// thread's `set` of it is in flight. Panics on a stale or foreign handle.
pub fn changed_at<T: Value, C: Cells>(rt: &Runtime<C>, handle: Incr<T>) -> u64 {
    rt.changed_at(handle)
}

/// [`changed_at`] of a handle that may be dead, recycled or foreign:
/// `None` for those.
pub fn changed_at_erased<C: Cells>(rt: &Runtime<C>, handle: ErasedIncr) -> Option<u64> {
    rt.changed_at_erased(handle)
}

/// Whether `rt`'s active frame on this thread is doomed, so a table miss
/// there returns [`pending_handle`] instead of creating a node.
pub fn frame_doomed<C: Cells>(rt: &Runtime<C>) -> bool {
    rt.frame_doomed()
}

/// The handle a table miss returns in a doomed frame: it reads `T`'s
/// placeholder there. Panics if `T` has no placeholder.
pub fn pending_handle<T: Value, C: Cells>(rt: &Runtime<C>) -> Incr<T> {
    rt.pending_handle()
}

/// [`IncrCollection::insert`] reporting to `on_rep` and keeping what it
/// displaced in `keep`.
pub fn collection_insert<T: Value + Hash + Eq, C: Cells, R: RepSink<T>>(
    rt: &Runtime<C>,
    collection: &IncrCollection<T, C>,
    value: T,
    on_rep: &mut R,
    keep: &mut Vec<Displaced<T>>,
) {
    collection.insert_with(rt, value, on_rep, |d| d.keep_into(keep));
}

/// [`IncrCollection::delete`] as [`collection_insert`].
pub fn collection_delete<T: Value + Hash + Eq, C: Cells, R: RepSink<T>>(
    rt: &Runtime<C>,
    collection: &IncrCollection<T, C>,
    value: &T,
    on_rep: &mut R,
    keep: &mut Vec<Displaced<T>>,
) -> bool {
    collection.delete_with(rt, value, on_rep, |d| d.keep_into(keep))
}

/// [`IncrCollection::replace`] as [`collection_insert`].
pub fn collection_replace<T: Value + Hash + Eq, C: Cells, R: RepSink<T>>(
    rt: &Runtime<C>,
    collection: &IncrCollection<T, C>,
    old: Option<&T>,
    new: T,
    on_rep: &mut R,
    keep: &mut Vec<Displaced<T>>,
) -> bool {
    collection.replace_with(rt, old, new, on_rep, |d| d.keep_into(keep))
}

/// The stamp a restore seeds every entry with, `walked - 1`: odd, so at
/// or above every input whose set has walked and below the revision of
/// every later claim, which is at least `walked`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeedBase {
    pub w0: u64,
    pub rid: RuntimeId,
}

/// One dep of a seeded query. `trusted_stamp` is a query dep's seed
/// stamp, which it must still carry for the seed to be `IfUnchanged`;
/// `None` is a dep the caller has checked itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeedDep {
    pub dep: ErasedIncr,
    pub trusted_stamp: Option<u64>,
}

/// How a seeded query is first read: `IfUnchanged` verifies Clean without
/// running when no dep changed since the base; `RecomputeFirst` runs the
/// closure, and an equal value is a cutoff.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reuse {
    IfUnchanged,
    RecomputeFirst,
}

/// Why [`seed_query`] did not seed; the query is left New.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedError {
    /// The query is not a New node with no deps and no dependents.
    NotFresh,
    /// A dep was deleted, or belongs to another runtime.
    DepNotAlive,
    /// `RecomputeFirst` with no deps, which no change could ever rerun.
    NoDeps,
    /// The base or the query belongs to another runtime.
    ForeignBase,
}

/// A Clean query's value and the deps of the run that produced it.
#[derive(Clone, Debug)]
pub struct Inspected<T> {
    pub value: T,
    pub deps: Vec<ErasedIncr>,
    pub in_group: bool,
}

pub fn seed_base<C: Cells>(rt: &Runtime<C>) -> SeedBase {
    rt.seed_base()
}

/// Lands New query `q` Dirty with `value` and `deps` at `base.w0`, as of
/// a run that read those deps; `IfUnchanged` falls back to
/// `RecomputeFirst` when a trusted dep no longer holds its stamp. On
/// failure `value` comes back for the caller to drop where its protocol
/// allows. Panics in a doomed frame.
pub fn seed_query<T: Value, C: Cells>(
    rt: &Runtime<C>,
    base: &SeedBase,
    q: Incr<T>,
    value: T,
    deps: &[SeedDep],
    reuse: Reuse,
) -> Result<Reuse, (SeedError, T)> {
    rt.seed_query(base, q, value, deps, reuse)
}

/// `None` unless `q` is Clean through both reads, with its stamp and
/// deps unmoved between them. Panics inside a compute of `rt`.
pub fn inspect<T: Value, C: Cells>(rt: &Runtime<C>, q: Incr<T>) -> Option<Inspected<T>> {
    rt.inspect(q)
}

/// [`NodeTable::get_or_insert_query_with`] that hands a node the miss
/// creates to `seed` before publishing it; a value `seed` returns, a
/// failed seed's, drops after the miss mutex. A hit never calls `seed`.
pub fn get_or_insert_query_seeded<K, T, C, S, F>(
    table: &NodeTable<K, T, C, S>,
    rt: &Runtime<C>,
    key: K,
    f: F,
    opts: QueryOpts<T>,
    seed: impl FnOnce(&Runtime<C>, Incr<T>) -> Option<T>,
) -> Incr<T>
where
    K: TableKey,
    T: Value,
    C: Cells,
    S: BuildHasher,
    F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
{
    table.get_or_insert_query_seeded(rt, key, f, opts, seed)
}

/// The live node for `key`, if any, without `NodeTable::get`'s refusal
/// inside a compute: a restore records no dependency on the lookup.
pub fn probe_unchecked<K: TableKey, T: Value, C: Cells, S: BuildHasher>(
    table: &NodeTable<K, T, C, S>,
    rt: &Runtime<C>,
    key: &K,
) -> Option<Incr<T>> {
    table.probe_unchecked(rt, key)
}

#[cfg(all(test, not(loom)))]
#[path = "persist_hooks_tests.rs"]
mod tests;
