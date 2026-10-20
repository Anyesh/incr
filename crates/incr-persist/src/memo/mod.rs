//! Storage for persistent memoization, shared by every memo backend.
//!
//! A memo store holds immutable generations, one of them current. A
//! generation is a [`GenHeader`] (build fingerprint, checkpoint sequence
//! number and table catalog) plus one crc-checked frame per [`Entry`].
//! [`MemoBackend`] is what a store implements; [`write_checkpoint`] merges
//! a checkpoint's live entries with the current generation, evicts down to
//! a byte budget and writes the next generation through it. [`FileMemo`]
//! keeps generations as files in a directory; `SqliteMemo`, behind the
//! `sqlite` feature, keeps them in an SQLite database; [`MemoryMemo`] is an
//! in-process store for tests.

mod backend;
mod checkpoint;
mod codec;
#[cfg(feature = "backend-tests")]
pub mod conformance;
mod file;
mod generation;
mod memory;
#[cfg(feature = "sqlite")]
mod sqlite;

pub use backend::{EncodedEntry, MemoBackend};
pub use checkpoint::{
    write_checkpoint, CheckpointInput, CheckpointStats, Checkpointed, DEFAULT_MAX_BYTES,
};
pub use codec::*;
pub use file::{FileMemo, FileMemoGuard};
pub use generation::{GenId, Generation};
pub use memory::MemoryMemo;
#[cfg(feature = "sqlite")]
pub use sqlite::{SqliteMemo, SqliteMemoGuard, SqliteMemoOptions};
