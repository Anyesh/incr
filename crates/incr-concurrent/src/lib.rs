//! `incr-concurrent`: thread-safe incremental computation engine.
//!
//! Since 0.2, this crate is a thin re-export of [`incr_core`] with the
//! [`Shared`] strategy. The `Runtime` type is `Send + Sync`: wrap it in
//! `Arc`, share it across threads, have one writer thread call `set`
//! while many reader threads call `get` on derived nodes. Same API
//! surface as the single-threaded sibling [incr-compute](https://crates.io/crates/incr-compute): switching
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

#[doc(hidden)]
pub use incr_core::{engine_unwind, take_last_cycle, with_value_code};
pub use incr_core::{
    Cycle, CycleStrategy, DeleteError, Delta, ErasedIncr, Fixpoint,
    GroupedCollection as GroupedCollectionInner, Incr, IncrCollection as IncrCollectionInner,
    NodeId, NodeInfo, NodeKindInfo, NodeState, NodeTrace, ObserverId, PropagationTrace, QueryOpts,
    RuntimeId, SortDelta, SortedCollection as SortedCollectionInner, TraceAction, Value,
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

#[cfg(test)]
mod reexports {
    /// Names every item the facade re-exports from incr-core, so a
    /// missing one fails to compile.
    #[test]
    fn every_incr_core_item_is_reexported() {
        use crate::{
            engine_unwind, take_last_cycle, with_value_code, Cycle, CycleStrategy, DeleteError,
            Delta, ErasedIncr, Fixpoint, GroupedCollection, GroupedCollectionInner, Incr,
            IncrCollection, IncrCollectionInner, NodeId, NodeInfo, NodeKindInfo, NodeState,
            NodeTrace, ObserverId, PropagationTrace, QueryOpts, Runtime, RuntimeId, SortDelta,
            SortedCollection, SortedCollectionInner, TraceAction, Value,
        };

        fn named<T: ?Sized>() {}
        fn value<T: Value>() {}
        let _ = (engine_unwind, take_last_cycle);
        named::<Cycle>();
        named::<CycleStrategy<u32>>();
        named::<Fixpoint<u32>>();
        named::<QueryOpts<u32>>();
        named::<DeleteError>();
        named::<Delta<u32>>();
        named::<ErasedIncr>();
        named::<GroupedCollection<u32, u32>>();
        named::<GroupedCollectionInner<u32, u32, incr_core::Shared>>();
        named::<Incr<u32>>();
        named::<IncrCollection<u32>>();
        named::<IncrCollectionInner<u32, incr_core::Shared>>();
        named::<NodeId>();
        named::<NodeInfo>();
        named::<NodeKindInfo>();
        named::<NodeState>();
        named::<NodeTrace>();
        named::<ObserverId>();
        named::<PropagationTrace>();
        named::<Runtime>();
        named::<RuntimeId>();
        named::<SortDelta<u32>>();
        named::<SortedCollection<u32, u32>>();
        named::<SortedCollectionInner<u32, u32, incr_core::Shared>>();
        named::<TraceAction>();
        value::<u32>();
        with_value_code(|| ());
    }
}
