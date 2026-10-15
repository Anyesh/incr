//! The walked-revision watermark. `set` bumps the revision before its
//! dirty walk, so the current revision can name a change whose walk has
//! not dirtied anything yet. A compute that stamped `verified_at` with
//! that revision could verify `Clean` past the change forever. Computes
//! stamp `verified_at` with `walked` instead: every revision `<= walked`
//! has finished its walk.
//!
//! Setters finish out of order, so each one records its revision in a
//! ring slot and `walked` advances over the contiguous prefix of recorded
//! slots. Whoever finds the next slot recorded advances past it, which is
//! often not the thread that walked it.

use crate::cells::Cells;

pub struct WalkRing<C: Cells, const N: usize> {
    walked: C::U64,
    slots: Box<[C::U64]>,
}

impl<C: Cells, const N: usize> WalkRing<C, N> {
    pub fn new(initial: u64) -> Self {
        Self {
            walked: C::new_u64(initial),
            slots: (0..N).map(|_| C::new_u64(0)).collect(),
        }
    }

    /// Acquire, so a reader that sees `walked = r` also sees every
    /// `Clean -> Dirty` CAS of walks up to r. A Relaxed load would let a
    /// compute stamp r while still seeing a dep `Clean` that r's walk
    /// dirtied.
    pub fn walked(&self) -> u64 {
        C::u64_load_acquire(&self.walked)
    }

    /// Record that revision `rev`'s walk has finished. The revision was
    /// already owned by an unconditional `fetch_add`, so a setter waits
    /// here, after its walk, until `rev`'s slot has been consumed. A lower
    /// revision never waits on a higher one, so this cannot deadlock.
    pub fn complete(&self, rev: u64) {
        let mut spins: u32 = 0;
        while self.walked() + (N as u64) < rev {
            // Advance rather than only poll, so a waiter never depends on
            // another setter to consume the slot it needs.
            self.advance();
            spins += 1;
            C::backoff(spins);
        }
        // Release, so the advancing thread's Acquire load of this slot
        // carries this walk's Clean -> Dirty CASes on to `walked`.
        C::u64_store_release(&self.slots[Self::index(rev)], rev);
        // Setter A (rev 5) stores slot 5 and then loads slot 4, while
        // setter B (rev 4) stores slot 4, advances to 4, and loads slot 5.
        // With only Release/Acquire both loads may miss the other's store
        // (store buffering), leaving `walked` at 4 after rev 5's walk
        // finished, and every compute until the next `set` would recompute
        // against a dep stamped 5. A SeqCst fence between each setter's
        // slot store and its loads guarantees at least one of the two sees
        // the other, so the last setter to finish always advances past
        // every finished revision. A fence rather than SeqCst accesses
        // because loom models SeqCst loads and stores as AcqRel, and would
        // report this stall as reachable.
        C::fence_seqcst();
        self.advance();
    }

    /// Move `walked` over every recorded slot that follows it.
    fn advance(&self) {
        loop {
            let w = self.walked();
            let next = w + 1;
            if C::u64_load_acquire(&self.slots[Self::index(next)]) != next {
                return;
            }
            // A failed CAS means another thread advanced; re-examine.
            let _ = C::u64_compare_exchange(&self.walked, w, next);
        }
    }

    fn index(rev: u64) -> usize {
        (rev % N as u64) as usize
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::cells::Shared;
    use std::sync::Arc;

    #[test]
    fn out_of_order_completion_advances_over_the_prefix() {
        let ring: WalkRing<Shared, 4> = WalkRing::new(1);
        ring.complete(3);
        assert_eq!(ring.walked(), 1);
        ring.complete(2);
        assert_eq!(ring.walked(), 3);
    }

    #[test]
    fn slots_are_reused_after_consumption() {
        let ring: WalkRing<Shared, 2> = WalkRing::new(1);
        for rev in 2..10 {
            ring.complete(rev);
            assert_eq!(ring.walked(), rev);
        }
    }

    #[test]
    fn concurrent_setters_leave_walked_at_the_last_revision() {
        let ring: Arc<WalkRing<Shared, 4>> = Arc::new(WalkRing::new(1));
        let next = Arc::new(std::sync::atomic::AtomicU64::new(2));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let ring = Arc::clone(&ring);
                let next = Arc::clone(&next);
                std::thread::spawn(move || {
                    for _ in 0..500 {
                        let rev = next.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                        ring.complete(rev);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(ring.walked(), 1 + 8 * 500);
    }
}
