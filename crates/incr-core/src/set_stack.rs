//! The per-thread stack of inputs whose claim (`NodeData::SET_CLAIM`) a
//! `set` on this thread holds. `set` pushes an entry once its claim
//! succeeds and pops it when it releases the claim, so every entry is a
//! held claim, and a `set` waiting on another thread's claim has no
//! entry of its own to mistake for one.

use std::cell::RefCell;

use crate::handle::RuntimeId;

struct Entry {
    rt: RuntimeId,
    slot: u32,
}

#[cfg(not(loom))]
thread_local! {
    static SET_STACK: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
}

#[cfg(loom)]
loom::thread_local! {
    static SET_STACK: RefCell<Vec<Entry>> = RefCell::new(Vec::new());
}

/// Record that a `set` on this thread now holds the claim on `slot` of
/// `rt`. Claims nest only inside one another, so the matching [`pop`]
/// always finds this entry on top.
#[inline]
pub(crate) fn push(rt: RuntimeId, slot: u32) {
    SET_STACK.with(|s| s.borrow_mut().push(Entry { rt, slot }));
}

/// Release the entry [`push`] made for `slot` of `rt`.
#[inline]
pub(crate) fn pop(rt: RuntimeId, slot: u32) {
    SET_STACK.with(|s| {
        let top = s.borrow_mut().pop();
        debug_assert!(
            top.is_some_and(|e| e.rt == rt && e.slot == slot),
            "set stack popped out of order"
        );
    });
}

/// True iff a `set` on this thread holds the claim on `slot` of `rt`.
/// Every input this thread's sets hold claimed.
pub(crate) fn claims() -> Vec<(RuntimeId, u32)> {
    SET_STACK.with(|s| s.borrow().iter().map(|e| (e.rt, e.slot)).collect())
}

pub(crate) fn holds(rt: RuntimeId, slot: u32) -> bool {
    SET_STACK.with(|s| s.borrow().iter().any(|e| e.rt == rt && e.slot == slot))
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn an_entry_is_held_from_push_to_pop() {
        let rt = RuntimeId::allocate();
        assert!(!holds(rt, 3));
        push(rt, 3);
        assert!(holds(rt, 3));
        assert!(!holds(rt, 4));
        assert!(!holds(RuntimeId::allocate(), 3));
        push(rt, 4);
        assert!(holds(rt, 3) && holds(rt, 4));
        pop(rt, 4);
        assert!(holds(rt, 3) && !holds(rt, 4));
        pop(rt, 3);
        assert!(!holds(rt, 3));
    }
}
