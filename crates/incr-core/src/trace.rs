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
    /// The generation of the node at `id`'s slot: a slot that was deleted
    /// and reused reports the new occupant's, so `(id, generation)` names
    /// one node for good.
    pub generation: u32,
    /// The node's state when the walk read it.
    pub state: crate::state::NodeState,
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

/// The group a label reports under: the label up to its first `/`. A
/// label with no `/` is its own group. Every label grouping in the
/// bindings (`graph_summary`, `memory_stats`, `compute_stats`) uses this.
pub fn label_group(label: &str) -> &str {
    label.split('/').next().unwrap_or(label)
}

/// What the nodes of one summary group are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupKind {
    /// Every node in the group is an input.
    Input,
    /// Every node in the group is a compute.
    Compute,
    /// The group holds both inputs and computes.
    Mixed,
}

/// One label group of a [`GraphSummary`]: `label` is the group name
/// ([`label_group`] of the node labels), `None` for unlabelled nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupSummary {
    pub label: Option<String>,
    pub kind: GroupKind,
    pub nodes: usize,
}

/// Dependency edges from one group to another: `count` is the number of
/// (dependency, dependent) node pairs whose dependency is in `from` and
/// whose dependent is in `to`. An edge between two nodes of one group has
/// `from == to` (a self edge); it is reported, never omitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupEdge {
    pub from: Option<String>,
    pub to: Option<String>,
    pub count: usize,
}

/// The graph grouped by label, from [`crate::Runtime::graph_summary`].
/// Groups and edges come in no particular order; a group or edge with no
/// live nodes is absent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphSummary {
    pub groups: Vec<GroupSummary>,
    pub edges: Vec<GroupEdge>,
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
    /// The nodes this one read that also appear in the trace, as of the
    /// trace's end.
    pub dependencies: Vec<NodeId>,
    /// Time its closure ran during the trace, net of the nodes it computed
    /// in turn, summed over every run (a retried or speculative run
    /// included); 0 for a node verified clean, and always 0 on
    /// wasm32-unknown-unknown.
    pub elapsed_ns: u64,
}

/// What compute stats counted for one node or label, from when they were
/// enabled or last reset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputeCounts {
    /// Runs that landed a value, cut off or not.
    pub recomputed: u64,
    /// The `recomputed` runs whose value was equal to the old one.
    pub cutoff: u64,
    /// Verifications that found the node clean without running it.
    pub verified: u64,
    /// Closure time net of nested computes, over every run including
    /// retried and speculative ones.
    pub elapsed_ns: u64,
}

impl ComputeCounts {
    /// Adds `other`'s counts to these.
    pub fn add(&mut self, other: &ComputeCounts) {
        self.recomputed += other.recomputed;
        self.cutoff += other.cutoff;
        self.verified += other.verified;
        self.elapsed_ns = self.elapsed_ns.saturating_add(other.elapsed_ns);
    }

    pub(crate) fn count(&mut self, action: &TraceAction) {
        match action {
            TraceAction::VerifiedClean => self.verified += 1,
            TraceAction::Recomputed { value_changed } => {
                self.recomputed += 1;
                if !value_changed {
                    self.cutoff += 1;
                }
            }
        }
    }
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

/// What a binding keeps for a `get_traced` root, rendered from its trace;
/// [`crate::Runtime::retain_trace`] stores it and
/// [`crate::Runtime::last_trace`] hands it back.
pub type TracePayload = std::sync::Arc<dyn std::any::Any + Send + Sync>;

/// A node's `verified_at` (memo bit masked) and `changed_at`. Zero in
/// either names no landing a trace could describe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StampPair {
    pub(crate) verified_at: u64,
    pub(crate) changed_at: u64,
}

impl StampPair {
    pub(crate) fn is_landed(self) -> bool {
        self.verified_at != 0 && self.changed_at != 0
    }
}

/// The traced root's entries as the trace log recorded them, each with
/// the root's stamps read when it was recorded.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RootEntries {
    pub(crate) count: u32,
    pub(crate) last: Option<(bool, StampPair)>,
}

impl RootEntries {
    pub(crate) fn see(&mut self, action: &TraceAction, stamps: StampPair) {
        self.count += 1;
        self.last = Some((matches!(action, TraceAction::Recomputed { .. }), stamps));
    }
}

/// What one `get_traced_retaining` saw of its root, for
/// [`crate::Runtime::retain_trace`].
#[must_use]
#[derive(Debug)]
pub struct TraceCapture {
    pub(crate) slot: u32,
    pub(crate) generation: u32,
    pub(crate) before: StampPair,
    pub(crate) root: RootEntries,
}

/// What a store does with a capture: keep a new payload at these stamps,
/// move the kept payload to them, or leave everything as it is.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Retain {
    Keep(StampPair),
    Restamp { from: StampPair, to: StampPair },
    Nothing,
}

impl TraceCapture {
    pub(crate) fn decide(&self) -> Retain {
        match self.root.last {
            Some((true, stamps)) if stamps.is_landed() => Retain::Keep(stamps),
            Some((false, stamps))
                if self.root.count == 1
                    && stamps.is_landed()
                    && stamps.changed_at == self.before.changed_at =>
            {
                Retain::Restamp {
                    from: self.before,
                    to: stamps,
                }
            }
            _ => Retain::Nothing,
        }
    }
}

/// A root's kept payload and the stamps it describes.
pub(crate) struct RetainedTrace {
    pub(crate) generation: u32,
    pub(crate) stamps: StampPair,
    pub(crate) payload: TracePayload,
}

/// [`crate::Runtime::last_trace`]'s answer for a live node.
pub struct LastTrace {
    /// The node's state when read.
    pub state: crate::state::NodeState,
    /// The kept payload, when the node is Clean at the stamps it was kept
    /// with.
    pub payload: Option<TracePayload>,
}
