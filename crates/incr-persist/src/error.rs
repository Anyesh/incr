use crate::{Kind, Position};

/// Shorthand for results whose error is this crate's [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Every failure a backend or record codec can report.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A write (`append` or `compact`) went to a handle opened read-only.
    #[error("backend handle is read-only")]
    ReadOnly,

    /// `compact` was asked to fold through a position below the current
    /// snapshot or beyond the log tail. Another writer may have compacted
    /// further in the meantime; the caller should read again before retrying.
    #[error(
        "cannot compact through position {upto}: it must lie between the snapshot position {snapshot_at} and the log tail {tail}"
    )]
    InvalidCompaction {
        upto: Position,
        snapshot_at: Position,
        tail: Position,
    },

    /// A record carries a format version this build does not read, usually
    /// because a newer incr-persist wrote it.
    #[error(
        "record format version {found} is not supported (this build reads version {supported})"
    )]
    UnsupportedFormat { found: u8, supported: u8 },

    /// A record failed to serialize.
    #[error("failed to encode record")]
    Encode(#[source] postcard::Error),

    /// Record bytes did not deserialize: the bytes are damaged or were not
    /// written as a record.
    #[error("failed to decode record")]
    Decode(#[source] postcard::Error),

    /// Record bytes are structurally invalid in a way the codec detects
    /// itself, such as an empty buffer or trailing bytes after the record.
    #[error("corrupt record: {0}")]
    Corrupt(&'static str),

    /// An I/O failure in an on-disk backend.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// A failure from a backend implemented outside this crate.
    #[error("backend failure")]
    Backend(#[source] Box<dyn std::error::Error + Send + Sync>),

    /// A store write failed after its records reached the log, so the store
    /// can no longer tell which of them it has applied. Every later call
    /// fails with this until the log is opened again, which replays it.
    #[error("store is poisoned by a failure after a write reached the log; reopen it")]
    Poisoned,

    /// A key holds a value of a different type than the one requested: its
    /// logged bytes do not decode as that type, or this store already bound
    /// the key with another type. `postcard` is not self-describing, so a
    /// mismatch whose bytes happen to decode goes undetected.
    #[error("key {key:?} holds a value of a different type than requested")]
    TypeMismatch {
        key: String,
        #[source]
        source: Option<postcard::Error>,
    },

    /// A key was requested as one kind (input or collection) but the log
    /// declares it as the other.
    #[error("key {key:?} is declared as a {declared:?}")]
    KindMismatch { key: String, declared: Kind },

    /// [`Store::undeclare`](crate::Store::undeclare) was asked to remove a
    /// key that queries in this process still depend on.
    #[error("key {key:?} still has dependents in this process; delete them before undeclaring it")]
    HasDependents { key: String },

    /// An input holds a value in the runtime that differs from the last
    /// record the store applied to it, because something wrote it through
    /// the runtime instead of the store. The log never saw that write, so a
    /// restart would lose it. Reported by `Store::set`, `Batch::set` and
    /// `Store::checkpoint` in debug builds only.
    #[error("input {key:?} was written through the runtime, bypassing the store's log")]
    Bypassed { key: String },

    /// A store operation was attempted while an
    /// [`UnloggedScope`](crate::UnloggedScope) is open on the store, or a
    /// second scope was opened. Close the scope first.
    #[error("an unlogged scope is open on this store; close it before this call")]
    UnloggedScopeOpen,

    /// Another handle holds the memo checkpoint lock, or wrote a newer
    /// memo generation since this one loaded its base. The checkpoint wrote
    /// nothing; a later one can try again.
    #[error("another memo checkpoint holds the lock or has replaced the generation")]
    Busy,

    /// A handle passed to a store was not issued by that store.
    #[error("handle was not issued by this store")]
    UnknownHandle,

    /// A keyed write found other writers changing the keys it wrote
    /// through every repair round, so some of those keys may still hold
    /// another row, or more than one pair. Refresh and retry.
    #[error("keyed collection {key:?} kept changing under other writers: refresh and retry")]
    KeyedContention { key: String },

    /// A keyed write would remove a held pair whose codec now encodes it to
    /// other bytes than it was applied from, and whose decoded copy is not
    /// `==` to it, so a replay of the log (a reopen, or another process)
    /// could not match the removal to the pair and would keep it. Nothing
    /// was written and the store is not poisoned.
    #[error(
        "keyed collection {key:?} holds a row that does not equal its decoded copy, and its codec \
         encodes it differently from the logged bytes, so a replay would not find it: the write \
         was refused"
    )]
    UnstableRow {
        /// The collection's key.
        key: String,
        /// The key of the row, a clone of the collection's key type.
        row_key: Box<dyn std::any::Any + Send + Sync>,
    },

    /// [`ConditionalAppend::append_at`](crate::ConditionalAppend::append_at)
    /// found the log's tail somewhere other than `after`, and wrote nothing.
    #[error("the log's tail is no longer at position {after}")]
    TailMoved { after: Position },

    /// A store operation other than the batch's own writes was attempted
    /// while a [`StagedBatch`](crate::StagedBatch) is open on the store, or
    /// a second one was opened. Commit or drop the batch first.
    #[error("a staged batch is open on this store; commit it before this call")]
    StagedBatchOpen,

    /// A [`StagedBatch`](crate::StagedBatch) was written to after its
    /// commit was attempted, or committed again after it landed.
    #[error("the staged batch has been committed or awaits retry; it takes no more writes")]
    StagedBatchClosed,

    /// [`Store::staged_batch`](crate::Store::staged_batch) was called on a
    /// backend without a [`ConditionalAppend`](crate::ConditionalAppend).
    #[error("the backend has no conditional append, which a staged batch needs")]
    ConditionalAppendUnsupported,

    /// `BuildId::CurrentExe` could not read the running executable.
    #[error("cannot fingerprint the running executable for the memo build id")]
    BuildIdUnavailable(#[source] std::io::Error),

    /// A memo table was declared with `MemoBuilder::table` but never given
    /// a closure with `MemoBuilder::define`.
    #[error("memo table {0:?} is declared but not defined")]
    UndefinedTable(String),

    /// `MemoOptions::auto_checkpoint_secs` was set on a `Local` runtime,
    /// which has no thread to run it on.
    #[error("automatic memo checkpoints need a Shared runtime")]
    AutoCheckpointNeedsShared,

    /// A memo entry, header or index encodes to a frame whose length does
    /// not fit the `u32` a frame records. A checkpoint drops such an entry,
    /// with every entry that reads it, instead of failing.
    #[error("a memo frame of {len} bytes exceeds the 4 GiB frame limit")]
    FrameTooLarge { len: u64 },
}
