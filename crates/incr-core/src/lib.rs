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
pub mod cells;
pub mod collection;
mod deferred;
pub mod dep_stack;
mod directory;
pub mod generic_arena;
pub mod handle;
mod hashed_multiset;
pub mod locks;
#[cfg(loom)]
pub mod loom_cells;
#[cfg(all(test, loom))]
mod loom_models;
pub mod node;
mod reclaim_probe;
pub mod runtime;
pub mod segmented_nodes;
mod set_stack;
pub mod sorted_collection;
pub mod state;
mod taint_memo;
#[cfg(test)]
mod test_size;
pub mod trace;
pub mod value;
mod value_code;
pub mod value_slot;
pub mod walk_ring;

pub use arena::PrimitiveArena;
pub use cells::{Cells, Local, LocalPtrCell, PtrCell, Shared};
pub use collection::{CollectionLog, Delta, GroupedCollection, IncrCollection};
pub use dep_stack::{DepStack, LocalDepStack, PoppedFrame, SharedDepStack};
pub use generic_arena::GenericArena;
pub use handle::{ErasedIncr, Incr, RuntimeId};
pub use locks::{LocalLock, Lock};
pub use node::{NodeData, NodeId};
pub use runtime::{DeleteError, ObserverId, Runtime};
pub use segmented_nodes::{SegmentedNodes, MAX_NODES};
pub use sorted_collection::{SortDelta, SortedCollection};
pub use state::NodeState;
pub use trace::{NodeInfo, NodeKindInfo, NodeTrace, PropagationTrace, TraceAction};
pub use value::Value;
#[doc(hidden)]
pub use value_code::with_value_code;
pub use value_slot::{LocalValueSlot, SharedValueSlot, ValueSlot};

#[cfg(test)]
#[path = "../tests/common/child.rs"]
pub(crate) mod child_test;

#[cfg(test)]
mod child_selftest;
