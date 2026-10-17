//! The per-thread stack of inputs being set. `set` pushes an entry
//! before it runs any user code and pops it through a guard; the entry
//! is marked claimed while `set` holds the input's `Computing` claim.
//! Only claimed entries count as held by this thread, so a `set` waiting
//! on another thread's claim never mistakes its own unclaimed entry for
//! a claim it holds.

use std::cell::RefCell;

use crate::handle::RuntimeId;

struct Entry {
    rt: RuntimeId,
    slot: u32,
    claimed: bool,
}

#[cfg(not(loom))]
thread_local! {
    static SET_STACK: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
}

#[cfg(loom)]
loom::thread_local! {
    static SET_STACK: RefCell<Vec<Entry>> = RefCell::new(Vec::new());
}

/// One `set` in progress on this thread; pops its entry when dropped.
pub(crate) struct SetEntry {
    depth: usize,
}

pub(crate) fn push(rt: RuntimeId, slot: u32) -> SetEntry {
    SET_STACK.with(|s| {
        let mut s = s.borrow_mut();
        s.push(Entry {
            rt,
            slot,
            claimed: false,
        });
        SetEntry { depth: s.len() - 1 }
    })
}

impl SetEntry {
    pub(crate) fn set_claimed(&self, claimed: bool) {
        SET_STACK.with(|s| s.borrow_mut()[self.depth].claimed = claimed);
    }
}

impl Drop for SetEntry {
    fn drop(&mut self) {
        SET_STACK.with(|s| {
            let mut s = s.borrow_mut();
            debug_assert_eq!(s.len(), self.depth + 1, "set stack popped out of order");
            s.pop();
        });
    }
}

/// True iff a `set` on this thread holds the claim on `slot` of `rt`.
pub(crate) fn holds(rt: RuntimeId, slot: u32) -> bool {
    SET_STACK.with(|s| {
        s.borrow()
            .iter()
            .any(|e| e.claimed && e.rt == rt && e.slot == slot)
    })
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn only_claimed_entries_are_held_and_the_guard_pops() {
        let rt = RuntimeId::allocate();
        let entry = push(rt, 3);
        assert!(!holds(rt, 3));
        entry.set_claimed(true);
        assert!(holds(rt, 3));
        assert!(!holds(rt, 4));
        assert!(!holds(RuntimeId::allocate(), 3));
        entry.set_claimed(false);
        assert!(!holds(rt, 3));
        entry.set_claimed(true);
        drop(entry);
        assert!(!holds(rt, 3));
    }
}
