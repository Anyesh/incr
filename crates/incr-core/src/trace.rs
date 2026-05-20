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
    pub elapsed_ns: u64,
}
