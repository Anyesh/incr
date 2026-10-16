use crate::Position;

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
}
