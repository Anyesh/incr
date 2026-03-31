use crate::graph::{ComputeEntry, Graph, NodeKind, NodeState};
use crate::types::{Incr, NodeId, Revision};
use std::any::Any;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;

/// The incremental computation runtime.
///
/// Creation methods (`create_input`, `create_query`) take `&mut self`.
/// Access methods (`get`, `set`) take `&self` using interior mutability.
pub struct Runtime {
    /// Node data (values, state, edges). Interior mutability for access during compute.
    nodes: RefCell<Vec<crate::graph::NodeData>>,
    /// Node kinds (Input or Compute). Immutable after creation.
    kinds: Vec<NodeKind>,
    /// Compute functions. Immutable after creation. Stored separately from nodes
    /// to avoid borrow conflicts: we read a function while mutating node data.
    funcs: Vec<ComputeEntry>,
    /// Global revision counter. Incremented on every input mutation.
    revision: Cell<Revision>,
    /// Stack of dependency recordings. Each frame records which nodes are read
    /// during a compute function's execution. Stack handles nested compute calls.
    dep_stack: RefCell<Vec<Vec<NodeId>>>,
    /// Set of nodes currently being computed. Used for cycle detection.
    computing: RefCell<Vec<NodeId>>,
}

impl Runtime {
    pub fn new() -> Self {
        Runtime {
            nodes: RefCell::new(Vec::new()),
            kinds: Vec::new(),
            funcs: Vec::new(),
            revision: Cell::new(Revision(1)), // Start at 1; default 0 means "never verified"
            dep_stack: RefCell::new(Vec::new()),
            computing: RefCell::new(Vec::new()),
        }
    }

    /// Create an input node with an initial value.
    pub fn create_input<T>(&mut self, value: T) -> Incr<T>
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
        self.kinds.push(NodeKind::Input);
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

    /// Walk forward from the given nodes, marking all reachable compute nodes as Dirty.
    fn mark_dirty_transitive(&self, start: &[NodeId]) {
        let mut queue: std::collections::VecDeque<NodeId> = start.iter().copied().collect();
        while let Some(id) = queue.pop_front() {
            let mut nodes = self.nodes.borrow_mut();
            let node = &mut nodes[id.0 as usize];
            if node.state == NodeState::Clean || node.state == NodeState::New {
                if node.state == NodeState::Clean {
                    node.state = NodeState::Dirty;
                }
                let dependents = node.dependents.clone();
                drop(nodes);
                for dep in dependents {
                    queue.push_back(dep);
                }
            }
        }
    }

    /// Ensure a node's value is up-to-date. For inputs, this is always true.
    /// For compute nodes, recursively ensures dependencies are clean,
    /// then recomputes if necessary.
    fn ensure_clean(&self, id: NodeId) {
        let state = self.nodes.borrow()[id.0 as usize].state;
        if state == NodeState::Clean {
            return;
        }

        match &self.kinds[id.0 as usize] {
            NodeKind::Input => return, // Inputs are always up-to-date after set()
            NodeKind::Compute(_) => {
                // Will be fully implemented in Task 4
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_get_input() {
        let mut rt = Runtime::new();
        let x = rt.create_input(42_i64);
        assert_eq!(rt.get(x), 42);
    }

    #[test]
    fn set_input_and_get_new_value() {
        let mut rt = Runtime::new();
        let x = rt.create_input(10_i64);
        assert_eq!(rt.get(x), 10);
        rt.set(x, 20);
        assert_eq!(rt.get(x), 20);
    }

    #[test]
    fn set_same_value_is_noop() {
        let mut rt = Runtime::new();
        let x = rt.create_input(5_i64);
        let rev_before = rt.revision.get();
        rt.set(x, 5);
        let rev_after = rt.revision.get();
        assert_eq!(rev_before, rev_after);
    }

    #[test]
    fn multiple_inputs() {
        let mut rt = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_input(2_i64);
        let c = rt.create_input(3_i64);
        assert_eq!(rt.get(a), 1);
        assert_eq!(rt.get(b), 2);
        assert_eq!(rt.get(c), 3);
    }
}
