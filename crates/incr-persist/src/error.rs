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

    /// [`Store::undeclare`](crate::Store::undeclare) was asked to remove an
    /// input that queries in this process still depend on.
    #[error("key {key:?} still has dependents in this process; delete them before undeclaring it")]
    HasDependents { key: String },

    /// A handle passed to a store was not issued by that store.
    #[error("handle was not issued by this store")]
    UnknownHandle,
}
