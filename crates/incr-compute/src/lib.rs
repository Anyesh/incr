//! `incr-compute`: single-threaded incremental computation engine.
//!
//! Since 0.2, this crate is a thin re-export of [`incr_core`] with the
//! [`Local`] strategy. The `Runtime` type is single-threaded (`!Sync`),
//! pays no atomic-fence cost on its hot path, and has zero atomic ops in
//! its uncontended access patterns. Same API surface as the concurrent
//! sibling [`incr-concurrent`]: switching is a one-line dependency swap.
//!
//! ## API status
//!
//! - Function DAG: `Runtime`, `Incr<T>`, `create_input`, `create_query`,
//!   `get`, `set`, `node_count`, `graph_snapshot`, `get_traced`. All
//!   functional. `get_traced` returns timing data but not per-node
//!   trace events; full tracing lands alongside the dashboard demo.
//! - Operators: `filter`, `map`, `count`, `reduce`, `sort_by_key`,
//!   `pairwise`, `window`, `group_by`, `join`. All functional under
//!   `Local`.
//! - Soundness: `set()` on a query node panics with a clear message
//!   (was undefined behavior in v0.1).
//!
//! Migration from 0.1: a single import. Closure bounds tightened to
//! `Fn + Send + Sync + 'static` for uniformity with [`incr-concurrent`];
//! most user types already meet these bounds.

#![doc(html_no_source)]

use incr_core::Local;

pub use incr_core::{
    Delta, GroupedCollection as GroupedCollectionInner, Incr,
    IncrCollection as IncrCollectionInner, NodeId, NodeInfo, NodeKindInfo, NodeState, NodeTrace,
    ObserverId, PropagationTrace, RuntimeId, SortDelta, SortedCollection as SortedCollectionInner,
    TraceAction,
    Value,
};

/// Single-threaded runtime: `Runtime<Local>`. Not `Send`/`Sync`. Use the
/// `incr-concurrent` crate for the multi-threaded equivalent.
pub type Runtime = incr_core::Runtime<Local>;

/// Single-threaded incremental collection: `IncrCollection<T, Local>`.
pub type IncrCollection<T> = IncrCollectionInner<T, Local>;

/// Single-threaded grouped collection: `GroupedCollection<K, T, Local>`.
pub type GroupedCollection<K, T> = GroupedCollectionInner<K, T, Local>;

/// Single-threaded sorted collection: `SortedCollection<T, K, Local>`.
pub type SortedCollection<T, K> = SortedCollectionInner<T, K, Local>;
