//! Storage for persistent memoization, shared by every memo backend.
//!
//! A memo store holds immutable generations, one of them current. A
//! generation is a [`GenHeader`] (build fingerprint, checkpoint sequence
//! number and table catalog) plus one crc-checked frame per [`Entry`].
//! [`MemoBackend`] is what a store implements; [`write_checkpoint`] merges
//! a checkpoint's live entries with the current generation, evicts down to
//! a byte budget and writes the next generation through it. [`MemoryMemo`]
//! is an in-process store for tests.

mod backend;
mod checkpoint;
mod codec;
#[cfg(feature = "backend-tests")]
pub mod conformance;
mod generation;
mod memory;

pub use backend::{EncodedEntry, MemoBackend};
pub use checkpoint::{
    write_checkpoint, CheckpointInput, CheckpointStats, Checkpointed, DEFAULT_MAX_BYTES,
};
pub use codec::*;
pub use generation::{GenId, Generation};
pub use memory::MemoryMemo;
