//! The thread-local taint memo: nodes an `ensure_clean` walk verified
//! only, or left New, under taint, which the root's own compute is about
//! to re-read through nested gets. An `ensure_clean` of a memoized node
//! computes it directly instead of expanding its installed deps again,
//! which would re-run the walk that tainted it once per chain level.
//!
//! Entries are owned by the frame of the node whose skip tainted them
//! (the lowest such frame when there are several) and cleared when it
//! pops. A pop finds the memo through the thread word's bit alone, so a
//! frame that never tainted carries no owner flag; while the memo is
//! non-empty, every pop compares its depth with the deepest owner, and
//! only an owner's pop scans the entries. Clearing early is sound: a node with no entry is walked and
//! verified as it would be without the memo, and computing a memoized
//! node directly is always a legal landing, so an entry can only cost a
//! closure run where a verify would have passed, never a wrong value.
//!
//! The memo's non-empty state is a bit of the thread word in
//! `value_code`, whose low bits count value-code calls, so a single load
//! and compare against zero on the non-Clean get path covers both and
//! only a non-zero word takes the cold path that tells them apart.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::handle::RuntimeId;
use crate::node::NodeId;
use crate::value_code::{self, MEMO_BIT};

#[derive(Default)]
struct Memo {
    owners: HashMap<(RuntimeId, NodeId), u32>,
    /// At least every entry's owner, so a pop above it skips the scan.
    deepest: u32,
}

#[cfg(not(loom))]
thread_local! {
    static MEMO: RefCell<Memo> = RefCell::new(Memo::default());
}

#[cfg(loom)]
loom::thread_local! {
    static MEMO: RefCell<Memo> = RefCell::new(Memo::default());
}

/// Whether `id` of `rt` is memoized, given a non-zero `word`.
#[cold]
#[inline(never)]
pub(crate) fn memoized(word: u64, rt: RuntimeId, id: NodeId) -> bool {
    word & MEMO_BIT != 0 && MEMO.with(|m| m.borrow().owners.contains_key(&(rt, id)))
}

/// Memoize `id` of `rt` under the frame at depth `owner`; an existing
/// entry keeps the lower of the two owners, the longer-lived frame.
pub(crate) fn memoize(rt: RuntimeId, id: NodeId, owner: u32) {
    MEMO.with(|m| {
        let mut m = m.borrow_mut();
        let entry = m.owners.entry((rt, id)).or_insert(owner);
        *entry = (*entry).min(owner);
        let kept = *entry;
        m.deepest = m.deepest.max(kept);
    });
    value_code::set_memo_bit();
}

/// Clear the entries owned by the frame at `depth` or deeper, when that
/// frame pops with the memo bit set.
#[cold]
#[inline(never)]
pub(crate) fn clear_from(depth: u32) {
    MEMO.with(|m| {
        let mut m = m.borrow_mut();
        if m.deepest < depth {
            return;
        }
        m.owners.retain(|_, owner| *owner < depth);
        m.deepest = m.owners.values().copied().max().unwrap_or(0);
        if m.owners.is_empty() {
            value_code::clear_memo_bit();
        }
    });
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::value_code::thread_word;

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
