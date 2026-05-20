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
pub mod arena_registry;
pub mod cells;
pub mod collection;
pub mod dep_stack;
pub mod generic_arena;
pub mod handle;
pub mod locks;
pub mod node;
pub mod runtime;
pub mod segmented_nodes;
pub mod sorted_collection;
pub mod state;
pub mod value;

pub use arena::PrimitiveArena;
pub use arena_registry::{ArenaRegistry, ErasedArena};
pub use cells::{Cells, Local, LocalPtrCell, PtrCell, Shared};
pub use collection::{CollectionLog, Delta, GroupedCollection, IncrCollection};
pub use dep_stack::{DepStack, LocalDepStack, SharedDepStack};
pub use generic_arena::GenericArena;
pub use handle::{Incr, RuntimeId};
pub use locks::{LocalLock, Lock};
pub use node::{NodeData, NodeId};
pub use runtime::Runtime;
pub use segmented_nodes::{SegmentedNodes, MAX_NODES};
pub use sorted_collection::{SortDelta, SortedCollection};
pub use state::NodeState;
pub use value::Value;
