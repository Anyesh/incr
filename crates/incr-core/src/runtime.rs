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

    /// Create a compute node defined by a pure function.
    /// The function receives `&Runtime` and calls `rt.get()` to read dependencies.
    /// Dependencies are automatically tracked — no manual wiring needed.
    pub fn create_query<T, F>(&mut self, f: F) -> Incr<T>
    where
        T: Any + Clone + PartialEq + 'static,
        F: Fn(&Runtime) -> T + 'static,
    {
        let func = Box::new(move |rt: &Runtime| -> Box<dyn Any> { Box::new(f(rt)) });
        let eq_fn = Box::new(|a: &dyn Any, b: &dyn Any| -> bool {
            a.downcast_ref::<T>().unwrap() == b.downcast_ref::<T>().unwrap()
        });
        let entry = ComputeEntry { func, eq_fn };

        let func_idx = self.funcs.len();
        self.funcs.push(entry);
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
        self.kinds.push(NodeKind::Compute(func_idx));

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

        let func_idx = match &self.kinds[id.0 as usize] {
            NodeKind::Input => return,
            NodeKind::Compute(idx) => *idx,
        };

        // Cycle detection
        {
            let computing = self.computing.borrow();
            if computing.contains(&id) {
                panic!(
                    "Cycle detected: node {:?} is already being computed",
                    id
                );
            }
        }

        // Step 1: Ensure all current dependencies are clean
        let deps: Vec<NodeId> = self.nodes.borrow()[id.0 as usize].dependencies.clone();
        for dep_id in &deps {
            self.ensure_clean(*dep_id);
        }

        // Step 2: Check if recomputation is actually needed
        let needs_recompute = {
            let nodes = self.nodes.borrow();
            let node = &nodes[id.0 as usize];
            match node.state {
                NodeState::New => true,
                NodeState::Dirty => {
                    // Recompute only if a dependency actually changed since last verification
                    node.dependencies.iter().any(|dep_id| {
                        nodes[dep_id.0 as usize].changed_at > node.verified_at
                    })
                }
                NodeState::Clean => false,
            }
        };

        if !needs_recompute {
            // Dependencies haven't changed — skip recomputation
            let mut nodes = self.nodes.borrow_mut();
            let node = &mut nodes[id.0 as usize];
            node.state = NodeState::Clean;
            node.verified_at = self.revision.get();
            return;
        }

        // Step 3: Execute the compute function
        self.computing.borrow_mut().push(id);
        self.dep_stack.borrow_mut().push(Vec::new());

        let new_value = (self.funcs[func_idx].func)(self);

        let new_deps = self.dep_stack.borrow_mut().pop().unwrap();
        self.computing.borrow_mut().retain(|n| *n != id);

        // Step 4: Update node — check for early cutoff
        let mut nodes = self.nodes.borrow_mut();
        let node = &mut nodes[id.0 as usize];
        let revision = self.revision.get();

        let value_changed = match &node.value {
            Some(old_value) => !(self.funcs[func_idx].eq_fn)(old_value.as_ref(), new_value.as_ref()),
            None => true, // First computation
        };

        if value_changed {
            node.value = Some(new_value);
            node.changed_at = revision;
        }
        node.verified_at = revision;
        node.state = NodeState::Clean;

        // Step 5: Update dependency edges
        let old_deps = std::mem::replace(&mut node.dependencies, new_deps.clone());

        // Remove self from dependents of old deps no longer needed
        for old_dep in &old_deps {
            if !new_deps.contains(old_dep) {
                nodes[old_dep.0 as usize].dependents.retain(|d| *d != id);
            }
        }
        // Add self to dependents of new deps
        for new_dep in &new_deps {
            if !old_deps.contains(new_dep) {
                nodes[new_dep.0 as usize].dependents.push(id);
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

    #[test]
    fn simple_compute_node() {
        let mut rt = Runtime::new();
        let a = rt.create_input(10_i64);
        let b = rt.create_query(move |rt| rt.get(a) * 2);
        assert_eq!(rt.get(b), 20);
    }

    #[test]
    fn compute_reads_multiple_inputs() {
        let mut rt = Runtime::new();
        let x = rt.create_input(3_i64);
        let y = rt.create_input(4_i64);
        let sum = rt.create_query(move |rt| rt.get(x) + rt.get(y));
        assert_eq!(rt.get(sum), 7);
    }

    #[test]
    fn chained_compute_nodes() {
        let mut rt = Runtime::new();
        let a = rt.create_input(5_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 1);
        let c = rt.create_query(move |rt| rt.get(b) * 2);
        assert_eq!(rt.get(c), 12); // (5 + 1) * 2
    }

    #[test]
    fn diamond_dependency_first_computation() {
        let mut rt = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 10);
        let c = rt.create_query(move |rt| rt.get(a) + 100);
        let d = rt.create_query(move |rt| rt.get(b) + rt.get(c));
        assert_eq!(rt.get(d), 112); // (1+10) + (1+100)
    }
}
