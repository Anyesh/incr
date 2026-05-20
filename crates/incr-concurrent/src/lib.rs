//! `incr-concurrent`: thread-safe incremental computation engine.
//!
//! Since 0.2, this crate is a thin re-export of [`incr_core`] with the
//! [`Shared`] strategy. The `Runtime` type is `Send + Sync`: wrap it in
//! `Arc`, share it across threads, have one writer thread call `set`
//! while many reader threads call `get` on derived nodes. Same API
//! surface as the single-threaded sibling [`incr-compute`]: switching
//! is a one-line dependency swap.
//!
//! ## API status
//!
//! - Function DAG: `Runtime`, `Incr<T>`, `create_input`, `create_query`,
//!   `get`, `set`, `node_count`, `graph_snapshot`, `get_traced`. All
//!   functional. `get_traced` returns timing data but not per-node
//!   trace events; full tracing lands alongside the dashboard demo.
//! - Operators: `filter`, `map`, `count`, `reduce`, `sort_by_key`,
//!   `pairwise`, `window`, `group_by`, `join`. All functional under
//!   `Shared`.
//! - Soundness: `set()` on a query node panics with a clear message.
//!
//! Migration from 0.1: the `Value` trait surface is now shared with
//! `incr-compute`. Most user types (primitives, String, Vec, Option,
//! tuples) implement it automatically.

#![doc(html_no_source)]

use incr_core::Shared;

pub use incr_core::{
    Delta, GroupedCollection as GroupedCollectionInner, Incr,
    IncrCollection as IncrCollectionInner, NodeId, NodeInfo, NodeKindInfo, NodeState, NodeTrace,
    PropagationTrace, RuntimeId, SortDelta, SortedCollection as SortedCollectionInner, TraceAction,
    Value,
};

/// Multi-threaded runtime: `Runtime<Shared>`. `Send + Sync`; wrap in
/// `Arc` to share across threads.
pub type Runtime = incr_core::Runtime<Shared>;

/// Thread-safe incremental collection: `IncrCollection<T, Shared>`.
pub type IncrCollection<T> = IncrCollectionInner<T, Shared>;

/// Thread-safe grouped collection: `GroupedCollection<K, T, Shared>`.
pub type GroupedCollection<K, T> = GroupedCollectionInner<K, T, Shared>;

/// Thread-safe sorted collection: `SortedCollection<T, K, Shared>`.
pub type SortedCollection<T, K> = SortedCollectionInner<T, K, Shared>;
