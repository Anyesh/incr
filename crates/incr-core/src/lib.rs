mod collection;
mod graph;
mod runtime;
mod sorted_collection;
mod types;

// Concurrent core rewrite. See crates/incr-core/src/v2/mod.rs for status.
// v2 is not publicly exported yet; it lives alongside v1 until it passes
// all milestone gates in the rewrite sequencing spec.
mod v2;

pub use collection::IncrCollection;
pub use runtime::Runtime;
pub use sorted_collection::{SortDelta, SortedCollection};
pub use types::{
    Incr, NodeId, NodeInfo, NodeKindInfo, NodeTrace, PropagationTrace, Revision, TraceAction,
};
