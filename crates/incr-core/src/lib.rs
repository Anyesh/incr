//! `incr-core`: the shared engine behind `incr-compute` and `incr-concurrent`.
//!
//! Both surface crates re-export the same `Runtime` parameterized over a
//! [`Cells`] strategy:
//! - `incr-compute` uses [`Local`], which backs every cell with
//!   `std::cell::Cell`. The single-threaded variant is `!Send + !Sync` and
//!   pays no atomic-fence cost.
//! - `incr-concurrent` uses [`Shared`], which backs every cell with
//!   `std::sync::atomic::Atomic*` types and explicit Acquire/Release
//!   ordering. The concurrent variant is `Send + Sync` and supports a
//!   writer thread plus arbitrary reader threads on the same graph.
//!
//! The validation that this parameterization carries zero overhead on the
//! single-threaded path lives in the spike crate's RESULTS.md (preserved on
//! the `spike/incr-core-monomorphization` branch). Short version: under
//! `Local`, every trait method inlines to the same code a direct
//! `Cell::get()` would emit; under `Shared`, every Acquire load compiles
//! to a plain `mov` on x86 with no `lock` prefixes or fences.

pub mod arena;
mod arena_registry;
mod attempt;
mod bits;
pub mod cells;
mod changes;
mod children;
pub mod collection;
pub mod cycle;
mod cycle_state;
mod deferred;
mod dep_set;
pub mod dep_stack;
mod dependent_list;
mod directory;
mod few;
pub mod fx;
pub mod generic_arena;
pub mod handle;
mod hashed_multiset;
mod hold;
pub mod lock_order;
pub mod locks;
#[cfg(loom)]
pub mod loom_cells;
#[cfg(all(test, loom))]
mod loom_models;
#[cfg(all(test, loom))]
mod loom_seed_models;
pub mod node;
mod notes;
#[doc(hidden)]
pub mod persist_hooks;
mod reclaim_probe;
mod replay;
mod reserve;
pub mod runtime;
pub mod segmented_nodes;
mod set_stack;
mod slot_vec;
pub mod sorted_collection;
pub mod state;
pub mod table;
mod taint_memo;
#[cfg(test)]
mod test_size;
pub mod trace;
pub mod value;
mod value_code;
pub mod value_slot;
mod waits;
pub mod walk_ring;

pub use arena::PrimitiveArena;
pub use cells::{Cells, Local, LocalPtrCell, PtrCell, Shared};
pub use collection::{CollectionLog, Delta, GroupedCollection, IncrCollection};
pub use cycle::{Cycle, CycleStrategy, Fixpoint, QueryOpts, Replay};
#[doc(hidden)]
pub use cycle_state::{engine_unwind, take_last_cycle};
pub use dep_stack::{DepStack, LocalDepStack, SharedDepStack};
pub use fx::{FxBuildHasher, FxHasher};
pub use generic_arena::GenericArena;
pub use handle::{ErasedIncr, Incr, RuntimeId};
pub use locks::{LocalLock, Lock};
pub use node::{NodeData, NodeId};
pub use runtime::{DeleteError, ObserverId, Runtime, DEFAULT_MAX_FRAMES};
pub use segmented_nodes::{SegmentedNodes, MAX_NODES};
pub use sorted_collection::{SortDelta, SortedCollection};
pub use state::NodeState;
pub use table::{NodeTable, TableKey};
pub use trace::{NodeInfo, NodeKindInfo, NodeTrace, PropagationTrace, TraceAction};
pub use value::Value;
#[doc(hidden)]
pub use value_code::{check_value_code, with_value_code};
pub use value_slot::{LocalValueSlot, SharedValueSlot, ValueSlot};

#[cfg(test)]
#[path = "../tests/common/child.rs"]
pub(crate) mod child_test;

#[cfg(test)]
mod child_selftest;
