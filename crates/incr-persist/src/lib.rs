//! `incr-persist`: durable, shareable state for incr runtimes.
//!
//! The storage seam is [`Backend`]: a totally ordered log of [`Record`]s at
//! increasing [`Position`]s, plus one snapshot that folds a prefix of the
//! log. Record values are opaque serialized bytes at this layer; typed
//! inputs and collections are layered on top by a `Store`.

mod backend;
mod error;
mod record;

pub use backend::{Backend, Position, ReadOutcome};
pub use error::{Error, Result};
pub use record::{Kind, Record};
