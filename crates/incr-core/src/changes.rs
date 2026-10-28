//! A count of the changes a Clean read of a runtime could see: sets,
//! deletes and the runtime's drop. Another runtime's speculative run keeps
//! what the count was at each Clean read it made of this one, and its stash
//! is adopted only while the count still is (`Runtime::adopt_stash`), so the
//! reads stand for the ones the unbounded run makes at the adoption. The
//! reader reaches the count through an `Arc`, never through the runtime,
//! which may have moved or dropped by then.

#[cfg(loom)]
use loom::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
#[cfg(not(loom))]
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

/// `word`'s high half counts changes begun, its low half changes in
/// progress, so a read sees both at once. A change begins before it
/// publishes anything and ends after, so a reader that finds none in
/// progress has seen every change the count holds, and one that has seen
/// any part of a later change finds the count moved. `speculative` counts
/// the changes speculative runs made, each before it ends.
pub(crate) struct Changes(Arc<Words>);

struct Words {
    word: AtomicU64,
    speculative: AtomicU64,
}

const BEGUN: u64 = 1 << 32;

impl Changes {
    pub(crate) fn new() -> Self {
        Self(Arc::new(Words {
            word: AtomicU64::new(0),
            speculative: AtomicU64::new(0),
        }))
    }

    /// Before the change stores anything a read could see, saying whether a
    /// speculative run makes it. Each call needs its [`Self::end`];
    /// [`Self::change`] pairs them.
    #[inline]
    pub(crate) fn begin(&self, speculative: bool) {
        self.0.word.fetch_add(BEGUN | 1, Ordering::AcqRel);
        if speculative {
            self.0.speculative.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// After the change is published.
    #[inline]
    pub(crate) fn end(&self) {
        self.0.word.fetch_sub(1, Ordering::AcqRel);
    }

    pub(crate) fn change(&self, speculative: bool) -> Change<'_> {
        self.begin(speculative);
        Change(self)
    }

    /// The count a Clean read made after this checks against, or `None`
    /// while a change is in progress.
    pub(crate) fn count(&self) -> Option<Count> {
        let at = self.0.word.load(Ordering::Acquire);
        // After `word`, so it holds every speculative change `at` does; one
        // it holds beyond them only makes `only_speculative_since` refuse.
        let speculative = self.0.speculative.load(Ordering::Acquire);
        (at as u32 == 0).then(|| Count {
            changes: Arc::clone(&self.0),
            at,
            speculative,
            mixed: false,
        })
    }
}

impl Drop for Changes {
    fn drop(&mut self) {
        self.0.word.fetch_add(BEGUN, Ordering::AcqRel);
    }
}

/// A change in progress until dropped.
pub(crate) struct Change<'a>(&'a Changes);

impl Drop for Change<'_> {
    fn drop(&mut self) {
        self.0.end();
    }
}

/// A runtime's [`Changes`] as a Clean read of it found them.
#[derive(Clone)]
pub(crate) struct Count {
    changes: Arc<Words>,
    at: u64,
    speculative: u64,
    /// Whether reads folded in by [`Count::absorb`] saw different numbers
    /// of changes that were not speculative.
    mixed: bool,
}

impl std::fmt::Debug for Count {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Count")
            .field("at", &self.at)
            .field("speculative", &self.speculative)
            .field("mixed", &self.mixed)
            .finish()
    }
}

impl Count {
    /// Whether the runtime has not changed since the read.
    pub(crate) fn holds(&self) -> bool {
        self.changes.word.load(Ordering::Acquire) == self.at
    }

    /// Whether every change since the read was a speculative run's.
    pub(crate) fn only_speculative_since(&self) -> bool {
        // Before `word`: a speculative change it holds beyond `word`'s is
        // still in progress there, and one `word` holds beyond it only
        // makes this refuse.
        let speculative = self.changes.speculative.load(Ordering::Acquire);
        let now = self.changes.word.load(Ordering::Acquire);
        !self.mixed
            && now as u32 == 0
            && (now >> 32).wrapping_sub(self.at >> 32) == speculative.wrapping_sub(self.speculative)
    }

    /// Whether the read found neither word moved by any change.
    #[cfg(all(test, loom))]
    pub(crate) fn before_any(&self) -> bool {
        self.at == 0 && self.speculative == 0
    }

    /// Folds in `other` if it is of the same runtime, so that `holds` and
    /// `only_speculative_since` answer for both reads: each holds for every
    /// read only if it holds for the oldest and, for the second, every read
    /// saw the same number of changes that were not speculative.
    pub(crate) fn absorb(&mut self, other: &Count) -> bool {
        if !Arc::ptr_eq(&self.changes, &other.changes) {
            return false;
        }
        self.mixed |= other.mixed || self.others() != other.others();
        if other.at < self.at {
            self.at = other.at;
            self.speculative = other.speculative;
        }
        true
    }

    fn others(&self) -> u64 {
        (self.at >> 32).wrapping_sub(self.speculative)
    }
}

/// Adds `count` to `counts`, folded into the count of its runtime there if
/// there is one, so that `counts` holds one count per runtime however often
/// a runtime changed between the reads.
pub(crate) fn fold(counts: &mut Vec<Count>, count: &Count) {
    if !counts.iter_mut().any(|c| c.absorb(count)) {
        counts.push(count.clone());
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    /// Over every order of four changes, speculative or not, the folded
    /// counts of any reads between them decide as all of the reads do.
    #[test]
    fn folded_counts_decide_as_their_reads() {
        for order in 0..16_u32 {
            let changes = Changes::new();
            let other = Changes::new();
            let mut reads = vec![changes.count().unwrap()];
            for step in 0..4 {
                drop(changes.change((order >> step) & 1 == 1));
                reads.push(changes.count().unwrap());
                for subset in 1..1_u32 << reads.len() {
                    let picked: Vec<&Count> = (0..reads.len())
                        .filter(|&i| (subset >> i) & 1 == 1)
                        .map(|i| &reads[i])
                        .collect();
                    let mut folded = vec![other.count().unwrap()];
                    for c in &picked {
                        fold(&mut folded, c);
                    }
                    assert_eq!(folded.len(), 2, "order {order:b} subset {subset:b}");
                    let c = &folded[1];
                    assert_eq!(
                        c.holds(),
                        picked.iter().all(|c| c.holds()),
                        "order {order:b} step {step} subset {subset:b}"
                    );
                    assert_eq!(
                        c.only_speculative_since(),
                        picked.iter().all(|c| c.only_speculative_since()),
                        "order {order:b} step {step} subset {subset:b}"
                    );
                }
            }
        }
    }
}
