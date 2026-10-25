//! The per-thread word behind two checks that share one load on the
//! non-Clean get path: its low bits count the user trait calls the
//! engine is inside (`Clone`, `PartialEq`, `Hash`, `Ord`, `Serialize`
//! and so on, of a value, key or element type), its top bit says the
//! taint memo has entries, and the two bits below it hold the kind of an
//! engine unwind in flight.
//!
//! Every engine path that can claim, wait, lock or create panics while
//! the counter is non-zero, because such a call from inside a user trait
//! method would re-enter a lock its own thread holds or wait on a claim
//! it cannot release. Operator closures are compute code, not value
//! code, and run outside the counter; so does user `Drop`, which the
//! engine cannot wrap because a retire may run it on any thread.
//!
//! A Clean `get` takes no lock and never waits, so it serves value code
//! like any caller. A `get` that misses the Clean fast path tests the
//! counter in `get_slow`, and from value code it only reloads the state
//! and reads, or panics.

use std::cell::Cell;

/// Set while the taint memo has entries.
pub(crate) const MEMO_BIT: u64 = 1 << 63;

/// The kind of an engine unwind in flight (`cycle_state::UnwindKind`),
/// kept in the word so that a frame pop tests it with the memo bit in
/// one load.
pub(crate) const UNWIND_SHIFT: u32 = 61;
pub(crate) const UNWIND_MASK: u64 = 3 << UNWIND_SHIFT;

/// Set while a replay context of some runtime is open on this thread,
/// so a non-Clean read finds out with the load it already makes for the
/// counter whether it must look for virtual frames.
pub(crate) const REPLAY_BIT: u64 = 1 << 60;

/// Set while this thread holds notes of Failed landings a panic in flight
/// made (`runtime_provenance`), so that a frame pop whose closure caught
/// that panic retires them.
pub(crate) const NOTES_BIT: u64 = 1 << 59;

/// Set while this thread keeps reads for a retry (`replay::keep_read`), so
/// that a frame pop drops the ones kept for frames it ends.
pub(crate) const KEPT_BIT: u64 = 1 << 58;

/// Set by a non-Clean read whose top frame is another runtime's, from
/// recording the read until its cold path clears it.
pub(crate) const FOREIGN_BIT: u64 = 1 << 57;

/// Set while a dep's closure runs again to regenerate a panic payload no
/// copy can stand for (`Runtime::regenerate`): a read that would compute,
/// wait or create aborts that run instead.
pub(crate) const REGEN_BIT: u64 = 1 << 56;

/// Set when a read aborted the run `REGEN_BIT` marks.
pub(crate) const REGEN_ABORTED: u64 = 1 << 55;

/// Set while this thread holds records of what runs' reads took, for a
/// regeneration in the engine call (`replay::note_took`).
pub(crate) const TOOK_BIT: u64 = 1 << 54;

/// The value-code counter's bits. Check sites test `word & COUNTER_MASK`,
/// so a live memo, unwind kind, replay context or note never trips the
/// contract panic.
pub(crate) const COUNTER_MASK: u64 = !(MEMO_BIT
    | UNWIND_MASK
    | REPLAY_BIT
    | NOTES_BIT
    | KEPT_BIT
    | FOREIGN_BIT
    | REGEN_BIT
    | REGEN_ABORTED
    | TOOK_BIT);

const _: () = assert!(crate::bits::disjoint(&[
    MEMO_BIT,
    UNWIND_MASK,
    REPLAY_BIT,
    NOTES_BIT,
    KEPT_BIT,
    FOREIGN_BIT,
    REGEN_BIT,
    REGEN_ABORTED,
    TOOK_BIT,
]));

pub(crate) const CONTRACT_MESSAGE: &str = "value and key methods must not call into the runtime";

#[cfg(not(loom))]
thread_local! {
    static THREAD_WORD: Cell<u64> = const { Cell::new(0) };
}

#[cfg(loom)]
loom::thread_local! {
    static THREAD_WORD: Cell<u64> = Cell::new(0);
}

#[inline(always)]
pub(crate) fn thread_word() -> u64 {
    THREAD_WORD.with(Cell::get)
}

pub(crate) fn set_memo_bit() {
    THREAD_WORD.with(|w| w.set(w.get() | MEMO_BIT));
}

pub(crate) fn clear_memo_bit() {
    THREAD_WORD.with(|w| w.set(w.get() & !MEMO_BIT));
}

/// Sets or clears `bits` in this thread's word.
pub(crate) fn set_bits(bits: u64, on: bool) {
    THREAD_WORD.with(|w| w.set(if on { w.get() | bits } else { w.get() & !bits }));
}

pub(crate) fn set_foreign_bit() {
    THREAD_WORD.with(|w| w.set(w.get() | FOREIGN_BIT));
}

pub(crate) fn clear_foreign_bit() {
    THREAD_WORD.with(|w| w.set(w.get() & !FOREIGN_BIT));
}

pub(crate) fn set_replay_bit(open: bool) {
    THREAD_WORD.with(|w| {
        w.set(if open {
            w.get() | REPLAY_BIT
        } else {
            w.get() & !REPLAY_BIT
        })
    });
}

pub(crate) fn set_notes_bit(held: bool) {
    THREAD_WORD.with(|w| {
        w.set(if held {
            w.get() | NOTES_BIT
        } else {
            w.get() & !NOTES_BIT
        })
    });
}

pub(crate) fn set_kept_bit(kept: bool) {
    THREAD_WORD.with(|w| {
        w.set(if kept {
            w.get() | KEPT_BIT
        } else {
            w.get() & !KEPT_BIT
        })
    });
}

/// Replaces the unwind kind's bits with `bits`, already shifted.
pub(crate) fn set_unwind_bits(bits: u64) {
    THREAD_WORD.with(|w| w.set((w.get() & !UNWIND_MASK) | bits));
}

/// Holds one count on the counter; dropping it, by a panic included,
/// gives the count back.
pub(crate) struct ValueCode(());

impl Drop for ValueCode {
    #[inline(always)]
    fn drop(&mut self) {
        THREAD_WORD.with(|w| w.set(w.get() - 1));
    }
}

/// Enter value code for the guard's lifetime: the engine is about to
/// call a user trait method of a value, key or element type.
#[inline(always)]
pub(crate) fn enter() -> ValueCode {
    THREAD_WORD.with(|w| w.set(w.get() + 1));
    ValueCode(())
}

/// Run `f`, a call into user trait methods, inside the counter.
#[inline(always)]
pub(crate) fn run<R>(f: impl FnOnce() -> R) -> R {
    let _guard = enter();
    f()
}

/// Runs `f` as value code, so that a call it makes into the runtime
/// panics with the contract message instead of deadlocking. For the
/// crates that serialize and compare values on the engine's behalf.
#[doc(hidden)]
pub fn with_value_code<R>(f: impl FnOnce() -> R) -> R {
    run(f)
}

/// Panics with the contract message if this thread is inside value code,
/// for the crates whose own locks value code can run under, before they
/// take one.
#[doc(hidden)]
#[inline(always)]
pub fn check_value_code() {
    check();
}

/// The contract panic of a checked path reached from value code.
#[cold]
#[inline(never)]
pub(crate) fn contract_panic() -> ! {
    panic!("incr-core: {}", CONTRACT_MESSAGE);
}

/// Panics if the current thread is inside value code. One load and
/// branch; the panic is out of line.
#[inline(always)]
pub(crate) fn check() {
    if thread_word() & COUNTER_MASK != 0 {
        contract_panic();
    }
}

/// The thread word, after refusing value code. One load serves the
/// check and a caller's memo test: a non-zero word is either, and only
/// then does the cold arm tell them apart.
#[inline(always)]
pub(crate) fn checked_word() -> u64 {
    let word = thread_word();
    if word != 0 {
        check_nonzero(word);
    }
    word
}

/// The cold arm of a site that already found `word` non-zero: tells a
/// live memo from a value-code count and panics for the count.
#[cold]
#[inline(never)]
fn check_nonzero(word: u64) {
    if word & COUNTER_MASK != 0 {
        contract_panic();
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn the_counter_balances_across_a_panic_and_leaves_the_memo_bit() {
        assert_eq!(thread_word(), 0);
        set_memo_bit();
        let outer = enter();
        assert_eq!(thread_word(), MEMO_BIT | 1);
        let unwound = std::panic::catch_unwind(|| {
            let _inner = enter();
            assert_eq!(thread_word(), MEMO_BIT | 2);
            check();
        });
        assert!(unwound.is_err());
        assert_eq!(thread_word(), MEMO_BIT | 1);
        drop(outer);
        assert_eq!(thread_word(), MEMO_BIT);
        check();
        check_nonzero(thread_word());
        clear_memo_bit();
        assert_eq!(thread_word(), 0);
        assert_eq!(with_value_code(thread_word), 1);
        assert_eq!(thread_word(), 0);
    }

    #[test]
    fn check_value_code_refuses_only_a_count() {
        set_memo_bit();
        check_value_code();
        clear_memo_bit();
        let err = std::panic::catch_unwind(|| with_value_code(check_value_code))
            .expect_err("value code was let through");
        let message = err.downcast::<String>().map(|s| *s).unwrap_or_default();
        assert!(message.contains(CONTRACT_MESSAGE), "{message}");
        assert_eq!(thread_word(), 0);
    }
}
