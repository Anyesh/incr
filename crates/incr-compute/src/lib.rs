//! `incr-compute`: single-threaded incremental computation engine.
//!
//! Since 0.2, this crate is a thin re-export of [`incr_core`] with the
//! [`Local`] strategy. The `Runtime` type is single-threaded (`!Sync`),
//! pays no atomic-fence cost on its hot path, and has zero atomic ops in
//! its uncontended access patterns. Same API surface as the concurrent
//! sibling [incr-concurrent](https://crates.io/crates/incr-concurrent): switching is a one-line dependency swap.
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
//! `Fn + Send + Sync + 'static` for uniformity with [incr-concurrent](https://crates.io/crates/incr-concurrent);
//! most user types already meet these bounds.

#![doc(html_no_source)]

use incr_core::Local;

#[doc(hidden)]
pub use incr_core::persist_hooks;
#[doc(hidden)]
pub use incr_core::{
    active_frame_doomed, engine_unwind, engine_unwind_in_progress, take_last_cycle, with_value_code,
};
pub use incr_core::{
    Cycle, CycleStrategy, DeleteError, Delta, ErasedIncr, Fixpoint, FxBuildHasher, FxHasher,
    GroupedCollection as GroupedCollectionInner, Incr, IncrCollection as IncrCollectionInner,
    NodeId, NodeInfo, NodeKindInfo, NodeState, NodeTable as NodeTableInner, NodeTrace, ObserverId,
    PropagationTrace, QueryOpts, Replay, RuntimeId, SortDelta,
    SortedCollection as SortedCollectionInner, TableKey, TraceAction, Value, DEFAULT_MAX_FRAMES,
};

/// Single-threaded runtime: `Runtime<Local>`. Not `Send`/`Sync`. Use the
/// `incr-concurrent` crate for the multi-threaded equivalent.
pub type Runtime = incr_core::Runtime<Local>;

/// A map from keys to nodes of one runtime: `NodeTable<K, T, Local, S>`.
pub type NodeTable<K, T, S = FxBuildHasher> = NodeTableInner<K, T, Local, S>;

/// Single-threaded incremental collection: `IncrCollection<T, Local>`.
pub type IncrCollection<T> = IncrCollectionInner<T, Local>;

/// Single-threaded grouped collection: `GroupedCollection<K, T, Local>`.
pub type GroupedCollection<K, T> = GroupedCollectionInner<K, T, Local>;

/// Single-threaded sorted collection: `SortedCollection<T, K, Local>`.
pub type SortedCollection<T, K> = SortedCollectionInner<T, K, Local>;

#[cfg(test)]
mod reexports {
    /// Names every item the facade re-exports from incr-core, so a
    /// missing one fails to compile.
    #[test]
    fn every_incr_core_item_is_reexported() {
        use crate::{
            active_frame_doomed, engine_unwind, engine_unwind_in_progress, take_last_cycle,
            with_value_code, Cycle, CycleStrategy, DeleteError, Delta, ErasedIncr, Fixpoint,
            FxBuildHasher, FxHasher, GroupedCollection, GroupedCollectionInner, Incr,
            IncrCollection, IncrCollectionInner, NodeId, NodeInfo, NodeKindInfo, NodeState,
            NodeTable, NodeTableInner, NodeTrace, ObserverId, PropagationTrace, QueryOpts, Replay,
            Runtime, RuntimeId, SortDelta, SortedCollection, SortedCollectionInner, TableKey,
            TraceAction, Value, DEFAULT_MAX_FRAMES,
        };

        fn named<T: ?Sized>() {}
        named::<crate::persist_hooks::SeedBase>();
        fn value<T: Value>() {}
        let _ = (
            active_frame_doomed,
            engine_unwind,
            engine_unwind_in_progress,
            take_last_cycle,
        );
        named::<Cycle>();
        named::<FxBuildHasher>();
        named::<FxHasher>();
        named::<NodeTable<u32, u32>>();
        named::<NodeTableInner<u32, u32, incr_core::Local>>();
        fn key<K: TableKey>() {}
        key::<u32>();
        named::<CycleStrategy<u32>>();
        named::<Fixpoint<u32>>();
        named::<QueryOpts<u32>>();
        named::<Replay>();
        let _: u32 = DEFAULT_MAX_FRAMES;
        named::<DeleteError>();
        named::<Delta<u32>>();
        named::<ErasedIncr>();
        named::<GroupedCollection<u32, u32>>();
        named::<GroupedCollectionInner<u32, u32, incr_core::Local>>();
        named::<Incr<u32>>();
        named::<IncrCollection<u32>>();
        named::<IncrCollectionInner<u32, incr_core::Local>>();
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
        named::<SortedCollectionInner<u32, u32, incr_core::Local>>();
        named::<TraceAction>();
        value::<u32>();
        with_value_code(|| ());
    }
}
