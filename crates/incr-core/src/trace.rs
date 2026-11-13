//! Tracing types: structural snapshots and per-node propagation traces.
//!
//! The wrappers re-export these types under the same names the original
//! `incr-compute` and `incr-concurrent` crates use, so user code that
//! constructed `NodeInfo`/`PropagationTrace` continues to compile.
//!
//! Full implementation status:
//! - `graph_snapshot()` on `Runtime<C>` returns real per-node `NodeInfo`
//!   data with dependencies and dependents.
//! - `get_traced()` populates `PropagationTrace` with totals and the
//!   per-node trace log when tracing is enabled. (Stub in this slice;
//!   real implementation lands alongside the dashboard demo work.)

use crate::node::NodeId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeKindInfo {
    Input,
    Compute,
}

#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub id: NodeId,
    pub kind: NodeKindInfo,
    pub label: Option<String>,
    pub dependencies: Vec<NodeId>,
    pub dependents: Vec<NodeId>,
}

/// Which edges a bounded graph walk follows from each visited node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// The nodes a node reads.
    Dependencies,
    /// The nodes that read a node.
    Dependents,
    /// Both.
    Both,
}

/// What `Runtime::graph_snapshot_bounded` walks and returns.
///
/// The walk is breadth-first from `roots`, ignoring dead nodes. `depth`
/// counts edges from a root: `Some(0)` visits only the roots, `Some(1)`
/// adds their direct neighbours in `direction`, `None` is unbounded.
#[derive(Clone, Debug)]
pub struct GraphQuery<'a> {
    pub roots: &'a [NodeId],
    pub direction: Direction,
    pub depth: Option<u32>,
    /// Only visited nodes whose label starts with this are returned
    /// (unlabelled nodes never match); the walk still passes through the
    /// others.
    pub label_prefix: Option<&'a str>,
    /// Most nodes returned; `truncated` is set when more matched.
    pub limit: Option<usize>,
    /// Most nodes visited, returned or not; `truncated` is set when the
    /// walk stopped on it. Defaults to `limit * 64` when only `limit` is set.
    pub max_visited: Option<usize>,
}

/// The nodes a bounded walk returned, in breadth-first order. Each keeps
/// its full dependency and dependent lists, including ids outside `nodes`.
#[derive(Clone, Debug)]
pub struct BoundedSnapshot {
    pub nodes: Vec<NodeInfo>,
    pub truncated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceAction {
    /// Node was dirty but its dependencies hadn't actually changed.
    VerifiedClean,
    /// Node was recomputed. `value_changed` is false when early cutoff occurred.
    Recomputed { value_changed: bool },
}

#[derive(Clone, Debug)]
pub struct NodeTrace {
    pub id: NodeId,
    pub action: TraceAction,
}

#[derive(Clone, Debug)]
pub struct PropagationTrace {
    pub target: NodeId,
    pub node_traces: Vec<NodeTrace>,
    pub total_nodes: usize,
    pub nodes_recomputed: usize,
    pub nodes_cutoff: usize,
    /// Always 0 on wasm32-unknown-unknown, which has no clock std can read.
    pub elapsed_ns: u64,
}
