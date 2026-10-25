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

/// The high half counts changes begun, the low half changes in progress,
/// in one word so a read sees both at once. A change begins before it
/// publishes anything and ends after, so a reader that finds none in
/// progress has seen every change the count holds, and one that has seen
/// any part of a later change finds the count moved.
pub(crate) struct Changes(Arc<AtomicU64>);

const BEGUN: u64 = 1 << 32;

impl Changes {
    pub(crate) fn new() -> Self {
        Self(Arc::new(AtomicU64::new(0)))
    }

    /// Before the change stores anything a read could see. Each call needs
    /// its [`Self::end`]; [`Self::change`] pairs them.
    #[inline]
    pub(crate) fn begin(&self) {
        self.0.fetch_add(BEGUN | 1, Ordering::AcqRel);
    }

    /// After the change is published.
    #[inline]
    pub(crate) fn end(&self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }

    pub(crate) fn change(&self) -> Change<'_> {
        self.begin();
        Change(self)
    }

    /// The count a Clean read made after this checks against, or `None`
    /// while a change is in progress.
    pub(crate) fn count(&self) -> Option<Count> {
        let at = self.0.load(Ordering::Acquire);
        (at as u32 == 0).then(|| Count {
            changes: Arc::clone(&self.0),
            at,
        })
    }
}

impl Drop for Changes {
    fn drop(&mut self) {
        self.0.fetch_add(BEGUN, Ordering::AcqRel);
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
    changes: Arc<AtomicU64>,
    at: u64,
}

impl Count {
    /// Whether the runtime has not changed since the read.
    pub(crate) fn holds(&self) -> bool {
        self.changes.load(Ordering::Acquire) == self.at
    }

    /// Whether the read found the count before any change began.
    #[cfg(all(test, loom))]
    pub(crate) fn before_any(&self) -> bool {
        self.at == 0
    }

    /// Whether `self` and `other` are of one runtime at one count.
    pub(crate) fn same(&self, other: &Count) -> bool {
        Arc::ptr_eq(&self.changes, &other.changes) && self.at == other.at
    }
}
