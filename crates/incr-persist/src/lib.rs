//! `incr-persist`: durable, shareable state for incr runtimes.
//!
//! The storage seam is [`Backend`]: a totally ordered log of [`Record`]s at
//! increasing [`Position`]s, plus one snapshot that folds a prefix of the
//! log. Record values are opaque serialized bytes at this layer. A
//! [`Store`] layers typed inputs over it: it owns a runtime and keeps it in
//! sync with the log, so restarted and concurrent processes agree on state.
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
#[cfg(feature = "sqlite")]
mod sqlite;
mod store;

pub use backend::{Backend, Position, ReadOutcome};
pub use error::{Error, Result};
pub use file::FileBackend;
pub use incr_core::{Cells, Local, Shared};
pub use memory::MemoryBackend;
pub use record::{Kind, Record};
#[cfg(feature = "sqlite")]
pub use sqlite::{SqliteBackend, SqliteDurability, SqliteOptions};
pub use store::{Batch, Durability, Input, Options, PersistValue, PersistedCollection, Store};
