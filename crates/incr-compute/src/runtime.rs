use crate::collection::{CollectionLog, IncrCollection};
use crate::graph::{ComputeEntry, NodeKind, NodeState};
use crate::types::{
    Incr, NodeId, NodeInfo, NodeKindInfo, NodeTrace, PropagationTrace, Revision, TraceAction,
};
use std::any::Any;
use std::cell::{Cell, RefCell};
use std::hash::Hash;
use std::marker::PhantomData;
use std::rc::Rc;

/// The incremental computation runtime.
///
/// Creation methods (`create_input`, `create_query`) take `&self`.
/// Access methods (`get`, `set`) take `&self` using interior mutability.
pub struct Runtime {
    /// Node data (values, state, edges). Interior mutability for access during compute.
    nodes: RefCell<Vec<crate::graph::NodeData>>,
    /// Node kinds (Input or Compute). RefCell for &self creation methods.
    kinds: RefCell<Vec<NodeKind>>,
    /// Compute functions. RefCell for &self creation methods. Stored separately from nodes
    /// to avoid borrow conflicts: we read a function while mutating node data.
    funcs: RefCell<Vec<ComputeEntry>>,
    /// Global revision counter. Incremented on every input mutation.
    revision: Cell<Revision>,
    /// Stack of dependency recordings. Each frame records which nodes are read
    /// during a compute function's execution. Stack handles nested compute calls.
    dep_stack: RefCell<Vec<Vec<NodeId>>>,
    /// Set of nodes currently being computed. Used for cycle detection.
    computing: RefCell<Vec<NodeId>>,
    /// Optional display labels for nodes (for introspection/debugging).
    labels: RefCell<Vec<Option<String>>>,
    /// When true, compute_node records trace events into trace_log.
    tracing_enabled: Cell<bool>,
    /// Trace events collected during the current get_traced() call.
    trace_log: RefCell<Vec<NodeTrace>>,
}

impl Runtime {
    pub fn new() -> Self {
        Runtime {
            nodes: RefCell::new(Vec::new()),
            kinds: RefCell::new(Vec::new()),
            funcs: RefCell::new(Vec::new()),
            revision: Cell::new(Revision(1)), // Start at 1; default 0 means "never verified"
            dep_stack: RefCell::new(Vec::new()),
            computing: RefCell::new(Vec::new()),
            labels: RefCell::new(Vec::new()),
            tracing_enabled: Cell::new(false),
            trace_log: RefCell::new(Vec::new()),
        }
    }

    /// Create an input node with an initial value.
    pub fn create_input<T>(&self, value: T) -> Incr<T>
    where
        T: Any + Clone + PartialEq + 'static,
    {
        assert!(
            self.dep_stack.borrow().is_empty(),
            "cannot create nodes during computation"
        );
        let revision = self.revision.get();
        let id = {
            let mut nodes = self.nodes.borrow_mut();
            let id = NodeId(nodes.len() as u32);
            nodes.push(crate::graph::NodeData {
                state: NodeState::Clean,
                value: Some(Box::new(value)),
                verified_at: revision,
                changed_at: revision,
                dependents: Vec::new(),
                dependencies: Vec::new(),
            });
            id
        };
        self.kinds.borrow_mut().push(NodeKind::Input);
        self.labels.borrow_mut().push(None);
        Incr {
            id,
            _phantom: PhantomData,
        }
    }

    /// Create a compute node defined by a pure function.
    /// The function receives `&Runtime` and calls `rt.get()` to read dependencies.
    /// Dependencies are automatically tracked — no manual wiring needed.
    pub fn create_query<T, F>(&self, f: F) -> Incr<T>
    where
        T: Any + Clone + PartialEq + 'static,
        F: Fn(&Runtime) -> T + 'static,
    {
        assert!(
            self.dep_stack.borrow().is_empty(),
            "cannot create nodes during computation"
        );
        let func = Box::new(move |rt: &Runtime| -> Box<dyn Any> { Box::new(f(rt)) });
        let eq_fn = Box::new(|a: &dyn Any, b: &dyn Any| -> bool {
            a.downcast_ref::<T>().unwrap() == b.downcast_ref::<T>().unwrap()
        });
        let entry = ComputeEntry { func, eq_fn };

        let func_idx = self.funcs.borrow().len();
        self.funcs.borrow_mut().push(entry);
        let id = {
            let mut nodes = self.nodes.borrow_mut();
            let id = NodeId(nodes.len() as u32);
            nodes.push(crate::graph::NodeData {
                state: NodeState::New,
                value: None,
                verified_at: Revision::default(),
                changed_at: Revision::default(),
                dependents: Vec::new(),
                dependencies: Vec::new(),
            });
            id
        };
        self.kinds.borrow_mut().push(NodeKind::Compute(func_idx));
        self.labels.borrow_mut().push(None);

        Incr {
            id,
            _phantom: PhantomData,
        }
    }

    /// Read the current value of a node. If the node is dirty or new,
    /// triggers recomputation of the minimum necessary subgraph.
    pub fn get<T>(&self, node: Incr<T>) -> T
    where
        T: Any + Clone + 'static,
    {
        // Record dependency if we're inside a compute function
        {
            let mut stack = self.dep_stack.borrow_mut();
            if let Some(frame) = stack.last_mut() {
                frame.push(node.id);
            }
        }

        // Ensure the node is up-to-date
        self.ensure_clean(node.id);

        // Read and clone the value
        let nodes = self.nodes.borrow();
        let node_data = &nodes[node.id.0 as usize];
        node_data
            .value
            .as_ref()
            .expect("node has no value after ensure_clean")
            .downcast_ref::<T>()
            .expect("type mismatch in get()")
            .clone()
    }

    /// Set a new value for an input node. If the value differs from the current one,
    /// increments the global revision and marks all transitive dependents as dirty.
    pub fn set<T>(&self, node: Incr<T>, value: T)
    where
        T: Any + Clone + PartialEq + 'static,
    {
        // Check if the value actually changed
        {
            let nodes = self.nodes.borrow();
            let node_data = &nodes[node.id.0 as usize];
            if let Some(old) = &node_data.value {
                if let Some(old_val) = old.downcast_ref::<T>() {
                    if *old_val == value {
                        return; // Same value — no-op
                    }
                }
            }
        }

        // Increment revision
        let mut rev = self.revision.get();
        rev.increment();
        self.revision.set(rev);

        // Update the input's value and timestamps
        let dependents = {
            let mut nodes = self.nodes.borrow_mut();
            let node_data = &mut nodes[node.id.0 as usize];
            node_data.value = Some(Box::new(value));
            node_data.changed_at = rev;
            node_data.verified_at = rev;
            node_data.dependents.clone()
        };

        // Mark all transitive dependents as dirty
        self.mark_dirty_transitive(&dependents);
    }

    pub fn create_collection<T>(&self) -> IncrCollection<T>
    where
        T: Any + Clone + Hash + Eq + 'static,
    {
        assert!(
            self.dep_stack.borrow().is_empty(),
            "cannot create nodes during computation"
        );
        let log = Rc::new(RefCell::new(CollectionLog::new()));
        let version_node = self.create_input(0_u64);
        IncrCollection { log, version_node }
    }

    /// Like `create_collection` but skips the dep_stack assertion. Used internally
    /// by operators (e.g. group_by) that lazily create sub-collections during
    /// compute closures. The caller is responsible for not using the resulting
    /// version_node as a tracked dependency of the current computation.
    pub(crate) fn create_collection_in_compute<T>(&self) -> IncrCollection<T>
    where
        T: Any + Clone + Hash + Eq + 'static,
    {
        let log = Rc::new(RefCell::new(CollectionLog::new()));
        let version_node = self.create_input_in_compute(0_u64);
        IncrCollection { log, version_node }
    }

    /// Like `create_input` but skips the dep_stack assertion. Used by operators
    /// that need to create input nodes during compute closures.
    pub(crate) fn create_input_in_compute<T>(&self, value: T) -> Incr<T>
    where
        T: Any + Clone + PartialEq + 'static,
    {
        let revision = self.revision.get();
        let id = {
            let mut nodes = self.nodes.borrow_mut();
            let id = NodeId(nodes.len() as u32);
            nodes.push(crate::graph::NodeData {
                state: NodeState::Clean,
                value: Some(Box::new(value)),
                verified_at: revision,
                changed_at: revision,
                dependents: Vec::new(),
                dependencies: Vec::new(),
            });
            id
        };
        self.kinds.borrow_mut().push(NodeKind::Input);
        self.labels.borrow_mut().push(None);
        Incr {
            id,
            _phantom: PhantomData,
        }
    }

    // ── Introspection API ───────────────────────────────────────────────────

    /// Assign a human-readable label to a node for visualization/debugging.
    pub fn set_label(&self, id: NodeId, label: String) {
        self.labels.borrow_mut()[id.0 as usize] = Some(label);
    }

    /// Enable or disable execution tracing. When enabled, compute_node records
    /// which nodes were visited, recomputed, or cut off during get() calls.
    pub fn set_tracing(&self, enabled: bool) {
        self.tracing_enabled.set(enabled);
    }

    /// Like get(), but also returns a trace of which nodes were processed.
    /// Clears the trace log before running, so the trace reflects only this call.
    pub fn get_traced<T>(&self, node: Incr<T>) -> (T, PropagationTrace)
    where
        T: Any + Clone + 'static,
    {
        let was_enabled = self.tracing_enabled.get();
        self.tracing_enabled.set(true);
        self.trace_log.borrow_mut().clear();

        let start = std::time::Instant::now();
        let value = self.get(node);
        let elapsed_ns = start.elapsed().as_nanos() as u64;

        self.tracing_enabled.set(was_enabled);

        let log = self.trace_log.borrow();
        let total_nodes = self.nodes.borrow().len();
        let nodes_recomputed = log
            .iter()
            .filter(|t| matches!(t.action, TraceAction::Recomputed { .. }))
            .count();
        let nodes_cutoff = log
            .iter()
            .filter(|t| {
                matches!(
                    t.action,
                    TraceAction::Recomputed {
                        value_changed: false
                    }
                )
            })
            .count();

        let trace = PropagationTrace {
            target: node.id,
            node_traces: log.clone(),
            total_nodes,
            nodes_recomputed,
            nodes_cutoff,
            elapsed_ns,
        };

        (value, trace)
    }

    /// Return structural info about every node in the graph.
    pub fn graph_snapshot(&self) -> Vec<NodeInfo> {
        let nodes = self.nodes.borrow();
        let kinds = self.kinds.borrow();
        let labels = self.labels.borrow();

        (0..nodes.len())
            .map(|i| {
                let id = NodeId(i as u32);
                NodeInfo {
                    id,
                    kind: match &kinds[i] {
                        NodeKind::Input => NodeKindInfo::Input,
                        NodeKind::Compute(_) => NodeKindInfo::Compute,
                    },
                    label: labels[i].clone(),
                    dependencies: nodes[i].dependencies.clone(),
                    dependents: nodes[i].dependents.clone(),
                }
            })
            .collect()
    }

    /// Return the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// Walk forward from the given nodes, marking all reachable compute nodes as Dirty.
    fn mark_dirty_transitive(&self, start: &[NodeId]) {
        let mut queue: std::collections::VecDeque<NodeId> = start.iter().copied().collect();
        let mut nodes = self.nodes.borrow_mut();
        while let Some(id) = queue.pop_front() {
            let node = &mut nodes[id.0 as usize];
            if node.state == NodeState::Clean || node.state == NodeState::New {
                if node.state == NodeState::Clean {
                    node.state = NodeState::Dirty;
                }
                for i in 0..node.dependents.len() {
                    queue.push_back(node.dependents[i]);
                }
            }
        }
    }

    /// Ensure a node's value is up-to-date. For inputs, this is always true.
    /// For compute nodes, iteratively ensures dependencies are clean in
    /// post-order (dependencies before dependents), then recomputes if necessary.
    fn ensure_clean(&self, id: NodeId) {
        // Fast path: already clean
        if self.nodes.borrow()[id.0 as usize].state == NodeState::Clean {
            return;
        }

        // Collect the post-order traversal of nodes that need processing.
        // Each stack entry is (node_id, visited) where visited=false means
        // "push deps first", visited=true means "now process this node".
        let mut work_stack: Vec<(NodeId, bool)> = vec![(id, false)];

        while let Some((cur, visited)) = work_stack.pop() {
            if visited {
                // Second visit: all deps should now be clean; process this node
                self.compute_node(cur);
                continue;
            }

            // Single borrow to check state and gather dirty deps
            let nodes = self.nodes.borrow();
            let state = nodes[cur.0 as usize].state;

            // Inputs and already-clean nodes need no work
            if state == NodeState::Clean {
                continue;
            }
            if matches!(self.kinds.borrow()[cur.0 as usize], NodeKind::Input) {
                continue;
            }

            // First visit: push self again (to process after deps), then push dirty deps
            work_stack.push((cur, true));
            let deps = &nodes[cur.0 as usize].dependencies;
            for &dep_id in deps {
                if nodes[dep_id.0 as usize].state != NodeState::Clean {
                    work_stack.push((dep_id, false));
                }
            }
        }
    }

    /// Compute (or verify) a single node, assuming all its known dependencies are already clean.
    fn compute_node(&self, id: NodeId) {
        // Single borrow to gather state, kind, cycle check, and needs_recompute
        let (func_idx, needs_recompute) = {
            let nodes = self.nodes.borrow();
            let node = &nodes[id.0 as usize];

            // Re-check state (may have been cleaned by an earlier iteration)
            if node.state == NodeState::Clean {
                return;
            }

            let func_idx = match &self.kinds.borrow()[id.0 as usize] {
                NodeKind::Input => return,
                NodeKind::Compute(idx) => *idx,
            };

            // Cycle detection
            {
                let computing = self.computing.borrow();
                if computing.contains(&id) {
                    panic!("Cycle detected: node {:?} is already being computed", id);
                }
            }

            // Check if recomputation is actually needed
            let needs_recompute = match node.state {
                NodeState::New => true,
                NodeState::Dirty => {
                    // Recompute only if a dependency actually changed since last verification
                    node.dependencies
                        .iter()
                        .any(|dep_id| nodes[dep_id.0 as usize].changed_at > node.verified_at)
                }
                NodeState::Clean => false,
            };

            (func_idx, needs_recompute)
        };

        if !needs_recompute {
            // Dependencies haven't changed — skip recomputation
            let mut nodes = self.nodes.borrow_mut();
            let node = &mut nodes[id.0 as usize];
            node.state = NodeState::Clean;
            node.verified_at = self.revision.get();
            if self.tracing_enabled.get() {
                self.trace_log.borrow_mut().push(NodeTrace {
                    id,
                    action: TraceAction::VerifiedClean,
                });
            }
            return;
        }

        // Step 2: Execute the compute function
        self.computing.borrow_mut().push(id);
        self.dep_stack.borrow_mut().push(Vec::with_capacity(4));

        let new_value = {
            let funcs = self.funcs.borrow();
            (funcs[func_idx].func)(self)
        };

        let new_deps = self.dep_stack.borrow_mut().pop().unwrap();
        // LIFO pop instead of O(n) retain — computing is always used as a stack
        self.computing.borrow_mut().pop();

        // Step 3: Check equality BEFORE borrowing nodes mutably
        // This avoids holding nodes borrow_mut and funcs borrow simultaneously
        let value_changed = {
            let nodes = self.nodes.borrow();
            let node = &nodes[id.0 as usize];
            match &node.value {
                Some(old_value) => {
                    let funcs = self.funcs.borrow();
                    !(funcs[func_idx].eq_fn)(old_value.as_ref(), new_value.as_ref())
                }
                None => true, // First computation
            }
        };

        if self.tracing_enabled.get() {
            self.trace_log.borrow_mut().push(NodeTrace {
                id,
                action: TraceAction::Recomputed { value_changed },
            });
        }

        // Step 4: Update node state and dependency edges in a single mutable borrow
        let mut nodes = self.nodes.borrow_mut();
        let revision = self.revision.get();

        // Update value and timestamps
        {
            let node = &mut nodes[id.0 as usize];
            if value_changed {
                node.value = Some(new_value);
                node.changed_at = revision;
            }
            node.verified_at = revision;
            node.state = NodeState::Clean;
        }

        // Update dependency edges — move new_deps in, take old_deps out
        let old_deps = std::mem::replace(&mut nodes[id.0 as usize].dependencies, new_deps);

        // Diff edges using the stored new_deps (now at nodes[id].dependencies)
        // Remove self from dependents of old deps no longer needed
        for old_dep in &old_deps {
            if !nodes[id.0 as usize].dependencies.contains(old_dep) {
                nodes[old_dep.0 as usize].dependents.retain(|d| *d != id);
            }
        }
        // Add self to dependents of new deps not previously present
        // Must collect indices first since we need to read nodes[id] then mutate others
        let new_dep_ids: Vec<NodeId> = nodes[id.0 as usize]
            .dependencies
            .iter()
            .filter(|new_dep| !old_deps.contains(new_dep))
            .copied()
            .collect();
        for dep in new_dep_ids {
            nodes[dep.0 as usize].dependents.push(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn create_and_get_input() {
        let rt = Runtime::new();
        let x = rt.create_input(42_i64);
        assert_eq!(rt.get(x), 42);
    }

    #[test]
    fn set_input_and_get_new_value() {
        let rt = Runtime::new();
        let x = rt.create_input(10_i64);
        assert_eq!(rt.get(x), 10);
        rt.set(x, 20);
        assert_eq!(rt.get(x), 20);
    }

    #[test]
    fn set_same_value_is_noop() {
        let rt = Runtime::new();
        let x = rt.create_input(5_i64);
        let rev_before = rt.revision.get();
        rt.set(x, 5);
        let rev_after = rt.revision.get();
        assert_eq!(rev_before, rev_after);
    }

    #[test]
    fn multiple_inputs() {
        let rt = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_input(2_i64);
        let c = rt.create_input(3_i64);
        assert_eq!(rt.get(a), 1);
        assert_eq!(rt.get(b), 2);
        assert_eq!(rt.get(c), 3);
    }

    #[test]
    fn simple_compute_node() {
        let rt = Runtime::new();
        let a = rt.create_input(10_i64);
        let b = rt.create_query(move |rt| rt.get(a) * 2);
        assert_eq!(rt.get(b), 20);
    }

    #[test]
    fn compute_reads_multiple_inputs() {
        let rt = Runtime::new();
        let x = rt.create_input(3_i64);
        let y = rt.create_input(4_i64);
        let sum = rt.create_query(move |rt| rt.get(x) + rt.get(y));
        assert_eq!(rt.get(sum), 7);
    }

    #[test]
    fn chained_compute_nodes() {
        let rt = Runtime::new();
        let a = rt.create_input(5_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 1);
        let c = rt.create_query(move |rt| rt.get(b) * 2);
        assert_eq!(rt.get(c), 12); // (5 + 1) * 2
    }

    #[test]
    fn diamond_dependency_first_computation() {
        let rt = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 10);
        let c = rt.create_query(move |rt| rt.get(a) + 100);
        let d = rt.create_query(move |rt| rt.get(b) + rt.get(c));
        assert_eq!(rt.get(d), 112); // (1+10) + (1+100)
    }

    // ── Task 5: Dirty Marking and Incremental Recomputation ──────────────────

    #[test]
    fn input_change_triggers_recomputation() {
        let rt = Runtime::new();
        let a = rt.create_input(10_i64);
        let b = rt.create_query(move |rt| rt.get(a) * 2);
        assert_eq!(rt.get(b), 20);

        rt.set(a, 15);
        assert_eq!(rt.get(b), 30);
    }

    #[test]
    fn chain_recomputation() {
        let rt = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 10);
        let c = rt.create_query(move |rt| rt.get(b) * 2);
        assert_eq!(rt.get(c), 22); // (1+10)*2

        rt.set(a, 5);
        assert_eq!(rt.get(c), 30); // (5+10)*2
    }

    #[test]
    fn diamond_recomputation() {
        let rt = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 10);
        let c = rt.create_query(move |rt| rt.get(a) + 100);
        let d = rt.create_query(move |rt| rt.get(b) + rt.get(c));

        assert_eq!(rt.get(d), 112); // (1+10) + (1+100)
        rt.set(a, 2);
        assert_eq!(rt.get(d), 114); // (2+10) + (2+100)
    }

    #[test]
    fn only_affected_nodes_recompute() {
        let rt = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_input(2_i64);

        let count_a = Rc::new(Cell::new(0_u32));
        let count_b = Rc::new(Cell::new(0_u32));

        let ca = count_a.clone();
        let derived_a = rt.create_query(move |rt| {
            ca.set(ca.get() + 1);
            rt.get(a) * 10
        });

        let cb = count_b.clone();
        let derived_b = rt.create_query(move |rt| {
            cb.set(cb.get() + 1);
            rt.get(b) * 10
        });

        // Initial computation
        assert_eq!(rt.get(derived_a), 10);
        assert_eq!(rt.get(derived_b), 20);
        assert_eq!(count_a.get(), 1);
        assert_eq!(count_b.get(), 1);

        // Change only input a — derived_b should NOT recompute
        rt.set(a, 5);
        assert_eq!(rt.get(derived_a), 50);
        assert_eq!(rt.get(derived_b), 20);
        assert_eq!(count_a.get(), 2); // recomputed
        assert_eq!(count_b.get(), 1); // NOT recomputed
    }

    #[test]
    fn multiple_mutations_before_get() {
        let rt = Runtime::new();
        let a = rt.create_input(1_i64);
        let compute_count = Rc::new(Cell::new(0_u32));
        let cc = compute_count.clone();
        let b = rt.create_query(move |rt| {
            cc.set(cc.get() + 1);
            rt.get(a) + 100
        });

        assert_eq!(rt.get(b), 101);
        assert_eq!(compute_count.get(), 1);

        // Multiple sets before reading — only one recomputation on get
        rt.set(a, 2);
        rt.set(a, 3);
        rt.set(a, 4);
        assert_eq!(rt.get(b), 104);
        assert_eq!(compute_count.get(), 2); // Only one recomputation, not three
    }

    // ── Task 6: Early Cutoff ─────────────────────────────────────────────────

    #[test]
    fn early_cutoff_stops_propagation() {
        let rt = Runtime::new();
        let a = rt.create_input(50_i64);

        let b_count = Rc::new(Cell::new(0_u32));
        let bc = b_count.clone();
        let b = rt.create_query(move |rt| {
            bc.set(bc.get() + 1);
            rt.get(a).min(100) // Clamp to max 100
        });

        let c_count = Rc::new(Cell::new(0_u32));
        let cc = c_count.clone();
        let c = rt.create_query(move |rt| {
            cc.set(cc.get() + 1);
            rt.get(b) + 1
        });

        // Initial
        assert_eq!(rt.get(c), 51); // min(50, 100) + 1
        assert_eq!(b_count.get(), 1);
        assert_eq!(c_count.get(), 1);

        // Change A to 60 — B changes (60 != 50), C recomputes
        rt.set(a, 60);
        assert_eq!(rt.get(c), 61);
        assert_eq!(b_count.get(), 2);
        assert_eq!(c_count.get(), 2);

        // Change A to 200 — B produces 100
        rt.set(a, 200);
        assert_eq!(rt.get(c), 101); // 100 + 1
        assert_eq!(b_count.get(), 3);
        assert_eq!(c_count.get(), 3);

        // Change A to 300 — B still 100 (clamped), SAME as before! Early cutoff!
        rt.set(a, 300);
        assert_eq!(rt.get(c), 101); // Still 100 + 1
        assert_eq!(b_count.get(), 4); // B recomputed (has to check)
        assert_eq!(c_count.get(), 3); // C did NOT recompute — early cutoff!
    }

    #[test]
    fn verification_skip_without_recomputation() {
        let rt = Runtime::new();
        let a = rt.create_input(5_i64);
        let unrelated = rt.create_input(100_i64);

        let b = rt.create_query(move |rt| rt.get(a).min(10)); // Clamped

        let d_count = Rc::new(Cell::new(0_u32));
        let dc = d_count.clone();
        let d = rt.create_query(move |rt| {
            dc.set(dc.get() + 1);
            rt.get(unrelated) + rt.get(b)
        });

        assert_eq!(rt.get(d), 105); // 100 + 5
        assert_eq!(d_count.get(), 1);

        // Change A from 5 to 8 — B changes from 5 to 8
        rt.set(a, 8);
        assert_eq!(rt.get(d), 108);
        assert_eq!(d_count.get(), 2);

        // Change A from 8 to 15 — B clamped to 10
        rt.set(a, 15);
        assert_eq!(rt.get(d), 110);
        assert_eq!(d_count.get(), 3);

        // Change A from 15 to 20 — B still clamped to 10, SAME value
        rt.set(a, 20);
        assert_eq!(rt.get(d), 110);
        assert_eq!(d_count.get(), 3); // D did not recompute
    }

    // ── Task 7: Dynamic Dependencies ─────────────────────────────────────────

    #[test]
    fn dynamic_dependency_switch() {
        let rt = Runtime::new();
        let flag = rt.create_input(true);
        let a = rt.create_input(10_i64);
        let b = rt.create_input(20_i64);

        let a_count = Rc::new(Cell::new(0_u32));
        let b_count = Rc::new(Cell::new(0_u32));
        let ac = a_count.clone();
        let bc = b_count.clone();

        let result = rt.create_query(move |rt| {
            if rt.get(flag) {
                ac.set(ac.get() + 1);
                rt.get(a)
            } else {
                bc.set(bc.get() + 1);
                rt.get(b)
            }
        });

        // Flag is true — reads A
        assert_eq!(rt.get(result), 10);

        // Switch flag to false — now reads B
        rt.set(flag, false);
        assert_eq!(rt.get(result), 20);

        // Change A — result should NOT recompute (no longer depends on A)
        rt.set(a, 99);
        assert_eq!(rt.get(result), 20);
    }

    #[test]
    fn cycle_detection_no_false_positives() {
        let rt = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 1);
        let c = rt.create_query(move |rt| rt.get(b) + 1);
        assert_eq!(rt.get(c), 3); // No cycle panic
    }
}
