//! Concurrent core v2.
//!
//! This module is the in-progress rewrite of the incr core per
//! `docs/superpowers/specs/2026-04-05-concurrent-core-rewrite.md`.
//!
//! Design values this module is accountable to:
//!
//! 1. **Neutral across use cases.** One `Runtime` type that serves both
//!    single-threaded and concurrent workloads without favoring either.
//! 2. **Blazingly fast.** Atomic fields with Relaxed ordering on the hot path,
//!    single Release-Acquire pair per compute completion. Zero cost for
//!    single-threaded users.
//! 3. **Easy to use.** Public API mirrors v1 exactly; the fact that the
//!    engine is `Send + Sync` is invisible to code that does not need it.
//!
//! This module is built alongside v1, not in place of it. v1 remains the
//! public Runtime until v2 passes all milestone gates in the rewrite
//! sequencing section of the spec.
//!
//! Status as of first commit: skeleton, state machine, typed arenas. No
//! Runtime yet, no compute, no dirty propagation.

pub(crate) mod arena;
pub(crate) mod state;
