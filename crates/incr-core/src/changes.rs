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
pub(crate) struct Count {
    changes: Arc<Words>,
    at: u64,
    speculative: u64,
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
        now as u32 == 0
            && (now >> 32).wrapping_sub(self.at >> 32) == speculative.wrapping_sub(self.speculative)
    }

    /// Whether the read found neither word moved by any change.
    #[cfg(all(test, loom))]
    pub(crate) fn before_any(&self) -> bool {
        self.at == 0 && self.speculative == 0
    }

    /// Whether `self` and `other` are of one runtime at one count.
    pub(crate) fn same(&self, other: &Count) -> bool {
        Arc::ptr_eq(&self.changes, &other.changes) && self.at == other.at
    }
}
