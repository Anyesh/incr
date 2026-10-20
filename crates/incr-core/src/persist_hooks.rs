//! Hooks for incr-persist's mapped writes. Unstable: incr-persist pins
//! this crate's exact version, and nothing else should call them.

use std::hash::Hash;

use crate::cells::Cells;
pub use crate::collection::Displaced;
use crate::collection::IncrCollection;
use crate::handle::Incr;
use crate::runtime::Runtime;
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

#[cfg(all(test, not(loom)))]
#[path = "persist_hooks_tests.rs"]
mod tests;
