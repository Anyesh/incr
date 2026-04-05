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
//!
//! The module is under active construction. Items land ahead of their
//! first consumer (state machine before NodeData, arenas before Runtime,
//! etc.), which trips dead-code warnings on every partial commit. We
//! suppress them here for the duration of the rewrite; individual items
//! are still exercised by their own unit tests. The allow comes off when
//! v2 is wired into `lib.rs` at Gate 5 per the spec.

#![allow(dead_code)]

pub(crate) mod arena;
pub(crate) mod handle;
pub(crate) mod node;
pub(crate) mod registry;
pub(crate) mod runtime;
#[cfg(test)]
mod runtime_concurrent_test;
#[cfg(test)]
mod runtime_proptest;
#[cfg(test)]
mod runtime_vs_v1_bench;
pub(crate) mod state;
pub(crate) mod value;
