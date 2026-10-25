//! The debug lock-order checker. Every incr lock has a [`LockKind`], and a
//! thread takes locks only in rising rank within one compute frame; an
//! acquisition out of order panics before it can block. A nested compute
//! starts a fresh scope, which is sound only because what a thread may hold
//! across a compute (the Store mutex and an `aggregate` state lock) is never
//! requested by a thread that holds a claim. Release builds check nothing.
//!
//! Loom runs its threads on one OS thread, so the thread-local stack would
//! mix them; the checker is off under loom.

/// The incr locks, in the order a thread takes them.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LockKind {
    /// incr-persist's Store mutex S.
    Store,
    MemoCheckpoint,
    Memo,
    /// The memo index lock, held only to clone or swap its `Arc`.
    MemoIndex,
    /// A `NodeTable`'s or `GridTable`'s miss mutex; a thread holds at
    /// most one.
    TableMiss,
    /// incr-persist's `DepIndex`.
    DepIndex,
    /// `join`'s state or a positional operator's mirror.
    OperatorState,
    /// `group_by`'s `groups`, which the router's `rt.set` runs under.
    GroupState,
    /// `aggregate`'s state lock, which `combine` runs under.
    AggregateState,
    /// `group_by`'s `key_cache`, taken after its `groups`.
    KeyCache,
    /// A collection log guard or a sorted collection's `state`; a thread
    /// holds at most one.
    CollectionLog,
    Inner,
    /// A `GenericArena`'s free-list lock.
    Arena,
    /// A directory's retired arrays, taken under the arena lock.
    Directory,
    /// T9's watch queue.
    WatchQueue,
    /// A runtime's stashes of speculative runs. Only value code, which reads
    /// Clean nodes without a lock, runs under it.
    Stashes,
}

impl LockKind {
    #[cfg_attr(not(all(debug_assertions, not(loom))), allow(dead_code))]
    fn rank(self) -> u16 {
        match self {
            LockKind::Store => 0,
            LockKind::MemoCheckpoint => 10,
            LockKind::Memo => 20,
            LockKind::MemoIndex => 30,
            LockKind::TableMiss => 40,
            LockKind::DepIndex => 50,
            LockKind::OperatorState | LockKind::GroupState | LockKind::AggregateState => 55,
            LockKind::KeyCache => 56,
            LockKind::CollectionLog => 58,
            LockKind::Inner => 60,
            LockKind::Arena => 65,
            LockKind::Directory => 66,
            LockKind::WatchQueue => 70,
            LockKind::Stashes => 75,
        }
    }
}

/// Keeps a lock's place on the thread's stack until the lock is released.
/// Dropped after the lock's own guard.
#[doc(hidden)]
#[must_use]
pub struct Held {
    #[cfg(all(debug_assertions, not(loom)))]
    id: u64,
}

#[cfg(all(debug_assertions, not(loom)))]
mod stack {
    use super::{Held, LockKind};
    use std::cell::{Cell, RefCell};

    pub(super) struct Entry {
        pub(super) kind: LockKind,
        pub(super) frame: u32,
        pub(super) id: u64,
    }

    thread_local! {
        pub(super) static HELD: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
        pub(super) static FRAME: Cell<u32> = const { Cell::new(0) };
        static NEXT_ID: Cell<u64> = const { Cell::new(0) };
    }

    pub(super) fn acquire(kind: LockKind) -> Held {
        let frame = FRAME.with(Cell::get);
        let id = NEXT_ID.with(|next| next.replace(next.get() + 1));
        HELD.with(|held| {
            let mut held = held.borrow_mut();
            if let Some(above) = held
                .iter()
                .filter(|e| e.frame == frame)
                .find(|e| e.kind.rank() >= kind.rank())
            {
                let (above, held_kinds) = (above.kind, kinds(&held));
                drop(held);
                panic!(
                    "incr-core: lock order: taking {kind:?} while holding {above:?} \
                     (held: {held_kinds:?})"
                );
            }
            held.push(Entry { kind, frame, id });
        });
        Held { id }
    }

    pub(super) fn release(id: u64) {
        // During thread teardown the stack may be gone already.
        let _ = HELD.try_with(|held| {
            let mut held = held.borrow_mut();
            if let Some(at) = held.iter().rposition(|e| e.id == id) {
                held.remove(at);
            }
        });
    }

    pub(super) fn kinds(held: &[Entry]) -> Vec<LockKind> {
        held.iter().map(|e| e.kind).collect()
    }
}

/// Puts `kind` on the thread's stack, panicking if the current frame holds
/// a lock of the same or a later rank. Take it before the lock itself, so
/// that a bad order panics instead of deadlocking.
#[doc(hidden)]
#[inline]
pub fn acquire(kind: LockKind) -> Held {
    #[cfg(all(debug_assertions, not(loom)))]
    return stack::acquire(kind);
    #[cfg(not(all(debug_assertions, not(loom))))]
    {
        let _ = kind;
        Held {}
    }
}

impl Drop for Held {
    #[inline]
    fn drop(&mut self) {
        #[cfg(all(debug_assertions, not(loom)))]
        stack::release(self.id);
    }
}

/// A guard with its place in the lock order. The lock is released before
/// its place, so the two never disagree about what is held.
#[doc(hidden)]
pub struct Ordered<G> {
    guard: G,
    _order: Held,
}

impl<G> Ordered<G> {
    /// Takes `kind`'s place in the order, then the lock, so that a bad
    /// order panics instead of blocking.
    #[inline(always)]
    pub fn take(kind: LockKind, lock: impl FnOnce() -> G) -> Self {
        let order = acquire(kind);
        Ordered {
            guard: lock(),
            _order: order,
        }
    }

    /// [`Self::take`] for a lock that can fail.
    pub fn try_take<E>(kind: LockKind, lock: impl FnOnce() -> Result<G, E>) -> Result<Self, E> {
        let order = acquire(kind);
        Ok(Ordered {
            guard: lock()?,
            _order: order,
        })
    }
}

impl<G: std::ops::Deref> std::ops::Deref for Ordered<G> {
    type Target = G::Target;

    #[inline(always)]
    fn deref(&self) -> &G::Target {
        &self.guard
    }
}

impl<G: std::ops::DerefMut> std::ops::DerefMut for Ordered<G> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut G::Target {
        &mut self.guard
    }
}

/// A compute frame is pushed: the only locks it may sit above are the Store
/// mutex (a subscriber's work) and an `aggregate` state lock (`combine`).
#[inline]
pub(crate) fn enter_frame() {
    #[cfg(all(debug_assertions, not(loom)))]
    {
        stack::HELD.with(|held| {
            let held = held.borrow();
            let kinds = stack::kinds(&held);
            drop(held);
            assert!(
                kinds
                    .iter()
                    .all(|k| matches!(k, LockKind::Store | LockKind::AggregateState)),
                "incr-core: lock order: a compute frame starts while holding {kinds:?}"
            );
        });
        stack::FRAME.with(|frame| frame.set(frame.get() + 1));
    }
}

#[inline]
pub(crate) fn leave_frame() {
    #[cfg(all(debug_assertions, not(loom)))]
    stack::FRAME.with(|frame| frame.set(frame.get() - 1));
}

/// An operator closure (`pred`, `f`, `key_fn`, `lift`, a sort key) is about
/// to run: it is compute code, so the current frame holds no incr lock but
/// the Store mutex, which a closure that computes can never request.
#[cfg(all(debug_assertions, not(loom)))]
pub(crate) fn assert_unlocked(closure: &str) {
    let frame = stack::FRAME.with(std::cell::Cell::get);
    stack::HELD.with(|held| {
        let held = held.borrow();
        let locked: Vec<LockKind> = held
            .iter()
            .filter(|e| e.frame == frame && e.kind != LockKind::Store)
            .map(|e| e.kind)
            .collect();
        drop(held);
        assert!(
            locked.is_empty(),
            "incr-core: lock order: {closure} runs while holding {locked:?}"
        );
    });
}

/// `f`, checking at each call that it runs with no incr lock held (see
/// [`assert_unlocked`]).
#[cfg(all(debug_assertions, not(loom)))]
pub(crate) fn unlocked<A: ?Sized, R>(
    closure: &'static str,
    f: impl Fn(&A) -> R + Send + Sync + 'static,
) -> impl Fn(&A) -> R + Send + Sync + 'static {
    move |a| {
        assert_unlocked(closure);
        f(a)
    }
}

#[cfg(not(all(debug_assertions, not(loom))))]
#[inline(always)]
pub(crate) fn unlocked<F>(_closure: &'static str, f: F) -> F {
    f
}

/// A thread is about to wait on a claim. It may hold only the Store mutex,
/// the `DepIndex` (a `set` waiting on another `set`), `groups` and `key_cache`
/// (the router's `rt.set`) and an aggregate's state (`combine`): no claim
/// holder ever requests these.
#[inline]
pub(crate) fn assert_may_wait_on_claim() {
    #[cfg(all(debug_assertions, not(loom)))]
    stack::HELD.with(|held| {
        let held = held.borrow();
        let kinds = stack::kinds(&held);
        drop(held);
        assert!(
            kinds.iter().all(|k| matches!(
                k,
                LockKind::Store
                    | LockKind::DepIndex
                    | LockKind::GroupState
                    | LockKind::AggregateState
                    | LockKind::KeyCache
            )),
            "incr-core: lock order: waiting on a claim while holding {kinds:?}"
        );
    });
}

#[cfg(all(test, debug_assertions, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn locks_taken_in_rising_rank_are_allowed() {
        let _s = acquire(LockKind::Store);
        let _d = acquire(LockKind::DepIndex);
        let _g = acquire(LockKind::GroupState);
        let _k = acquire(LockKind::KeyCache);
        let _l = acquire(LockKind::CollectionLog);
        let _i = acquire(LockKind::Inner);
        let _a = acquire(LockKind::Arena);
        let _r = acquire(LockKind::Directory);
        let _w = acquire(LockKind::WatchQueue);
    }

    #[test]
    #[should_panic(expected = "taking DepIndex while holding Inner")]
    fn a_lower_rank_under_a_higher_one_panics() {
        let _i = acquire(LockKind::Inner);
        let _d = acquire(LockKind::DepIndex);
    }

    #[test]
    #[should_panic(expected = "taking TableMiss while holding TableMiss")]
    fn two_locks_of_one_rank_panic() {
        let _a = acquire(LockKind::TableMiss);
        let _b = acquire(LockKind::TableMiss);
    }

    #[test]
    fn locks_may_be_released_out_of_order() {
        let groups = acquire(LockKind::GroupState);
        let _keys = acquire(LockKind::KeyCache);
        drop(groups);
        let _log = acquire(LockKind::CollectionLog);
    }

    #[test]
    fn a_released_lock_no_longer_counts() {
        drop(acquire(LockKind::Inner));
        let _d = acquire(LockKind::DepIndex);
    }

    #[test]
    fn a_nested_frame_starts_a_fresh_scope() {
        let _agg = acquire(LockKind::AggregateState);
        enter_frame();
        let log = acquire(LockKind::CollectionLog);
        let inner = acquire(LockKind::Inner);
        drop((inner, log));
        let _state = acquire(LockKind::AggregateState);
        leave_frame();
    }

    #[test]
    #[should_panic(expected = "a compute frame starts while holding [Inner]")]
    fn a_frame_under_inner_panics() {
        let _i = acquire(LockKind::Inner);
        enter_frame();
    }

    #[test]
    #[should_panic(expected = "pred runs while holding [CollectionLog]")]
    fn a_closure_under_a_lock_panics() {
        let _l = acquire(LockKind::CollectionLog);
        assert_unlocked("pred");
    }

    #[test]
    fn a_closure_may_run_under_the_store_or_a_lower_frames_lock() {
        let _s = acquire(LockKind::Store);
        assert_unlocked("key_fn");
        let _agg = acquire(LockKind::AggregateState);
        enter_frame();
        assert_unlocked("lift");
        leave_frame();
    }

    #[test]
    #[should_panic(expected = "waiting on a claim while holding [Inner]")]
    fn waiting_on_a_claim_under_inner_panics() {
        let _i = acquire(LockKind::Inner);
        assert_may_wait_on_claim();
    }

    #[test]
    #[should_panic(expected = "waiting on a claim while holding [OperatorState]")]
    fn waiting_on_a_claim_under_a_join_state_panics() {
        let _j = acquire(LockKind::OperatorState);
        assert_may_wait_on_claim();
    }

    #[test]
    fn waiting_on_a_claim_under_the_routers_locks_is_allowed() {
        let _g = acquire(LockKind::GroupState);
        let _k = acquire(LockKind::KeyCache);
        assert_may_wait_on_claim();
    }

    #[test]
    fn a_panic_releases_what_it_held() {
        let caught = std::panic::catch_unwind(|| {
            let _i = acquire(LockKind::Inner);
            panic!("in a critical section");
        });
        assert!(caught.is_err());
        let _d = acquire(LockKind::DepIndex);
    }
}
