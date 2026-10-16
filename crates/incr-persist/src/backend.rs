use std::fmt;

use crate::{Record, Result};

/// A log sequence number.
///
/// Positions are handed out by a [`Backend`] in strictly increasing order
/// and are never reused, not even after compaction removes the records that
/// held them. [`Position::ZERO`] precedes every record, so
/// `read_from(Position::ZERO, ..)` asks for the whole log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Position(pub u64);

impl Position {
    /// The position before the first record of every log.
    pub const ZERO: Position = Position(0);
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// What [`Backend::read_from`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadOutcome {
    /// Every record after the requested position was pushed onto `out`.
    Records,
    /// Records after the requested position were folded into the snapshot
    /// at `snapshot_at`, so nothing was pushed. The caller loads the snapshot
    /// and reads again from its position.
    Compacted { snapshot_at: Position },
}

/// Names one snapshot of a log, as [`Backend::snapshot_id`] reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SnapshotId {
    /// The position the snapshot covers.
    pub position: Position,
    /// How many compactions the log had seen once this snapshot replaced
    /// the previous one: 1 for the first, one more for each after it, a
    /// replacement at the same position included.
    pub generation: u64,
}

/// Storage for one totally ordered log of [`Record`]s and the snapshot that
/// folds its prefix.
///
/// A handle may be one of several writers on the same log (another thread,
/// another process). Every method takes `&self`; implementations serialize
/// internally.
pub trait Backend: Send + Sync {
    /// Appends `records` as one contiguous run and returns the position of
    /// the last one.
    ///
    /// The run occupies positions `last - (len - 1) ..= last`, each greater
    /// than every position this log has handed out before, compacted ones
    /// included. An append from another writer lands wholly before or
    /// wholly after it. An empty slice writes nothing and returns the
    /// current tail ([`Position::ZERO`] for a log that never held a record).
    ///
    /// Fails with [`Error::ReadOnly`](crate::Error::ReadOnly) on a
    /// read-only handle. On any other error the records may or may not have
    /// reached the log, so a caller must not blindly retry.
    fn append(&self, records: &[Record]) -> Result<Position>;

    /// Makes every record at or below `upto` durable: once this returns
    /// `Ok`, a later open of the same log sees them even if this process
    /// crashes. `upto` beyond the tail covers the whole log. A no-op for
    /// backends whose `append` is already durable.
    fn sync(&self, upto: Position) -> Result<()>;

    /// Pushes every record after `after` onto `out` in position order,
    /// without clearing `out`, and returns [`ReadOutcome::Records`].
    ///
    /// The run is gap-free: no record ever appears later at a position
    /// between `after` and the last position pushed, so a caller may treat
    /// that last position as fully read. Appends racing with the read may
    /// or may not be included.
    ///
    /// If `after` is below the snapshot position, some records the caller
    /// has not seen may survive only in the snapshot. Unless this handle can
    /// still supply every one of them, nothing is pushed and the result is
    /// [`ReadOutcome::Compacted`], even when compaction left no log records
    /// at all. A handle that still holds them, such as a file-backend reader
    /// draining a journal it opened before the compaction, may return them
    /// as `Records` instead, so callers must accept either outcome there.
    /// `Compacted` is never returned for `after` at or above the snapshot
    /// position.
    fn read_from(&self, after: Position, out: &mut Vec<(Position, Record)>) -> Result<ReadOutcome>;

    /// Returns the current snapshot and the position it covers, or `None`
    /// for a log that was never compacted. A snapshot of zero records is
    /// `Some((position, vec![]))`.
    fn load_snapshot(&self) -> Result<Option<(Position, Vec<Record>)>>;

    /// Names the current snapshot without loading it, or returns `None` for
    /// a log that was never compacted. Cheap enough to call on every
    /// catch-up.
    ///
    /// The position alone cannot tell snapshots apart, because `compact`
    /// may replace a snapshot at its own position, so the id carries the
    /// generation too. Every handle on the log, read-only ones included,
    /// reports the same id, and a reopen reports it again.
    fn snapshot_id(&self) -> Result<Option<SnapshotId>>;

    /// Replaces the snapshot with `folded`, recorded as covering every
    /// position through `upto`, and drops the log records at or below
    /// `upto`. Records above `upto`, such as ones other writers appended
    /// after the caller folded, stay in the log.
    ///
    /// The swap is atomic for readers: they see the old snapshot and log or
    /// the new pair, never a mix. `folded` is opaque here; omitting a key's
    /// records from it is how keys are forgotten. Positions handed out
    /// afterwards remain above `upto`.
    ///
    /// Fails with [`Error::ReadOnly`](crate::Error::ReadOnly) on a
    /// read-only handle, and with
    /// [`Error::InvalidCompaction`](crate::Error::InvalidCompaction) when
    /// `upto` is below the current snapshot position or beyond the tail.
    /// `upto` equal to the current snapshot position replaces the snapshot.
    /// A successful compaction advances the [`SnapshotId::generation`] by
    /// exactly one; a failed one leaves it unchanged.
    fn compact(&self, upto: Position, folded: Vec<Record>) -> Result<()>;

    /// `false` for a read-only open, on which `append` and `compact` fail.
    fn writable(&self) -> bool;
}
