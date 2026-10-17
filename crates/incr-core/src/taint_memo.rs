//! The thread-local taint memo: nodes an `ensure_clean` walk verified
//! only, or left New, under taint, which the root's own compute is about
//! to re-read through nested gets. An `ensure_clean` of a memoized node
//! computes it directly instead of expanding its installed deps again,
//! which would re-run the walk that tainted it once per chain level.
//!
//! Entries are owned by the frame of the node whose skip tainted them
//! (the lowest such frame when there are several) and cleared when it
//! pops. Clearing early is sound: a node with no entry is walked and
//! verified as it would be without the memo, and computing a memoized
//! node directly is always a legal landing, so an entry can only cost a
//! closure run where a verify would have passed, never a wrong value.
//!
//! The memo's non-empty state is a bit of `THREAD_WORD`, whose low bits
//! are reserved for the value-code counter, so a single load and compare
//! against zero on the non-Clean get path covers both and only a
//! non-zero word takes the cold path that tells them apart.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use crate::handle::RuntimeId;
use crate::node::NodeId;

/// Set in `THREAD_WORD` while the memo has entries.
const MEMO_BIT: u64 = 1 << 63;

/// The value-code counter's bits of `THREAD_WORD`. Its check sites test
/// `word & COUNTER_MASK`, so a live memo never trips the contract panic.
pub(crate) const COUNTER_MASK: u64 = !MEMO_BIT;

#[cfg(not(loom))]
thread_local! {
    static THREAD_WORD: Cell<u64> = const { Cell::new(0) };
    static MEMO: RefCell<HashMap<(RuntimeId, NodeId), u32>> = RefCell::new(HashMap::new());
}

#[cfg(loom)]
loom::thread_local! {
    static THREAD_WORD: Cell<u64> = Cell::new(0);
    static MEMO: RefCell<HashMap<(RuntimeId, NodeId), u32>> = RefCell::new(HashMap::new());
}

#[inline(always)]
pub(crate) fn thread_word() -> u64 {
    THREAD_WORD.with(Cell::get)
}

/// Whether `id` of `rt` is memoized, given a non-zero `word`.
#[cold]
#[inline(never)]
pub(crate) fn memoized(word: u64, rt: RuntimeId, id: NodeId) -> bool {
    word & MEMO_BIT != 0 && MEMO.with(|m| m.borrow().contains_key(&(rt, id)))
}

/// Memoize `id` of `rt` under the frame at depth `owner`; an existing
/// entry keeps the lower of the two owners, the longer-lived frame.
pub(crate) fn memoize(rt: RuntimeId, id: NodeId, owner: u32) {
    MEMO.with(|m| {
        let mut m = m.borrow_mut();
        let entry = m.entry((rt, id)).or_insert(owner);
        *entry = (*entry).min(owner);
    });
    THREAD_WORD.with(|w| w.set(w.get() | MEMO_BIT));
}

/// Clear the entries owned by the frame at `depth` or deeper, when that
/// frame pops.
pub(crate) fn clear_from(depth: u32) {
    MEMO.with(|m| {
        let mut m = m.borrow_mut();
        m.retain(|_, owner| *owner < depth);
        if m.is_empty() {
            THREAD_WORD.with(|w| w.set(w.get() & COUNTER_MASK));
        }
    });
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn entries_keep_the_lowest_owner_and_clear_from_a_depth() {
        let rt = RuntimeId::allocate();
        assert_eq!(thread_word(), 0);
        memoize(rt, NodeId(1), 3);
        memoize(rt, NodeId(1), 1);
        memoize(rt, NodeId(2), 3);
        assert_ne!(thread_word(), 0);
        assert!(memoized(thread_word(), rt, NodeId(1)));
        assert!(!memoized(thread_word(), RuntimeId::allocate(), NodeId(1)));
        clear_from(3);
        assert!(memoized(thread_word(), rt, NodeId(1)));
        assert!(!memoized(thread_word(), rt, NodeId(2)));
        clear_from(1);
        assert_eq!(thread_word(), 0);
        assert!(!memoized(MEMO_BIT, rt, NodeId(1)));
    }
}
