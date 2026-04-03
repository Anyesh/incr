use std::marker::PhantomData;

/// Index into the node arena. Cheap to copy and compare.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct NodeId(pub(crate) u32);

impl NodeId {
    pub fn raw(self) -> u32 {
        self.0
    }

    pub fn from_raw(id: u32) -> Self {
        NodeId(id)
    }
}

/// Monotonically increasing counter. Incremented on every input mutation.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Default)]
pub struct Revision(pub(crate) u64);

impl Revision {
    pub(crate) fn increment(&mut self) {
        self.0 += 1;
    }
}

/// Typed handle to a node in the incremental graph. `T` is the value type.
/// Cheap to copy — it's just a u32 index + phantom type.
#[derive(Debug)]
pub struct Incr<T> {
    pub(crate) id: NodeId,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T> Incr<T> {
    pub fn node_id(self) -> NodeId {
        self.id
    }
}

// Manual impls because derive would add T: Copy/Clone bounds
impl<T> Copy for Incr<T> {}
impl<T> Clone for Incr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

// ── Introspection types ─────────────────────────────────────────────────────

/// Whether a node is an input or a computed value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeKindInfo {
    Input,
    Compute,
}

/// Structural metadata about a single node, for visualization/debugging.
#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub id: NodeId,
    pub kind: NodeKindInfo,
    pub label: Option<String>,
    pub dependencies: Vec<NodeId>,
    pub dependents: Vec<NodeId>,
}

/// What happened to a node during a traced get() call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceAction {
    /// Node was dirty but its dependencies hadn't actually changed.
    VerifiedClean,
    /// Node was recomputed. `value_changed` is false when early cutoff occurred.
    Recomputed { value_changed: bool },
}

/// Trace entry for a single node during propagation.
#[derive(Clone, Debug)]
pub struct NodeTrace {
    pub id: NodeId,
    pub action: TraceAction,
}

/// Summary of what happened during a single get() call.
#[derive(Clone, Debug)]
pub struct PropagationTrace {
    pub target: NodeId,
    pub node_traces: Vec<NodeTrace>,
    pub total_nodes: usize,
    pub nodes_recomputed: usize,
    pub nodes_cutoff: usize,
    pub elapsed_ns: u64,
}
