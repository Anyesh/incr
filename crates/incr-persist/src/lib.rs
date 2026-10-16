//! `incr-persist`: durable, shareable state for incr runtimes.
//!
//! The storage seam is [`Backend`]: a totally ordered log of [`Record`]s at
//! increasing [`Position`]s, plus one snapshot that folds a prefix of the
//! log. Record values are opaque serialized bytes at this layer; typed
//! inputs and collections are layered on top by a `Store`.
//!
//! [`MemoryBackend`] keeps the log in process and can be shared by several
//! writers. The `backend-tests` feature exports `conformance`, the suite
//! every backend must pass, and the `backend_conformance!` macro that runs
//! it.

mod backend;
#[cfg(feature = "backend-tests")]
pub mod conformance;
mod error;
mod file;
mod memory;
mod record;

pub use backend::{Backend, Position, ReadOutcome};
pub use error::{Error, Result};
pub use file::FileBackend;
pub use memory::MemoryBackend;
pub use record::{Kind, Record};
