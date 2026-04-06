use crate::types::{NodeId, Revision};
use std::any::Any;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NodeState {
    Clean,
    Dirty,
    New,
}

#[derive(Debug)]
pub(crate) enum NodeKind {
    Input,
    Compute(usize), // Index into Graph::funcs
}

pub(crate) struct NodeData {
    pub state: NodeState,
    pub value: Option<Box<dyn Any>>,
    pub verified_at: Revision,
    pub changed_at: Revision,
    pub dependents: Vec<NodeId>,   // Forward edges: who depends on me
    pub dependencies: Vec<NodeId>, // Backward edges: who do I depend on
}

#[allow(clippy::type_complexity)]
pub(crate) struct ComputeEntry {
    pub func: Box<dyn Fn(&crate::runtime::Runtime) -> Box<dyn Any>>,
    pub eq_fn: Box<dyn Fn(&dyn Any, &dyn Any) -> bool>,
}

#[allow(dead_code)]
pub(crate) struct Graph {
    pub nodes: Vec<NodeData>,
    pub kinds: Vec<NodeKind>,
    pub funcs: Vec<ComputeEntry>,
}

#[allow(dead_code)]
impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            kinds: Vec::new(),
            funcs: Vec::new(),
        }
    }

    pub fn add_input(&mut self, value: Box<dyn Any>, revision: Revision) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(NodeData {
            state: NodeState::Clean,
            value: Some(value),
            verified_at: revision,
            changed_at: revision,
            dependents: Vec::new(),
            dependencies: Vec::new(),
        });
        self.kinds.push(NodeKind::Input);
        id
    }

    pub fn add_compute(&mut self, entry: ComputeEntry) -> NodeId {
        let func_idx = self.funcs.len();
        self.funcs.push(entry);
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(NodeData {
            state: NodeState::New,
            value: None,
            verified_at: Revision::default(),
            changed_at: Revision::default(),
            dependents: Vec::new(),
            dependencies: Vec::new(),
        });
        self.kinds.push(NodeKind::Compute(func_idx));
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Revision;

    #[test]
    fn create_input_node() {
        let mut graph = Graph::new();
        let id = graph.add_input(Box::new(42_i64), Revision(1));
        assert_eq!(id, NodeId(0));
        assert_eq!(graph.nodes[0].state, NodeState::Clean);
        assert!(graph.nodes[0].value.as_ref().unwrap().downcast_ref::<i64>() == Some(&42));
    }

    #[test]
    fn create_compute_node() {
        let mut graph = Graph::new();
        let entry = ComputeEntry {
            func: Box::new(|_| Box::new(0_i64)),
            eq_fn: Box::new(|a, b| a.downcast_ref::<i64>() == b.downcast_ref::<i64>()),
        };
        let id = graph.add_compute(entry);
        assert_eq!(id, NodeId(0));
        assert_eq!(graph.nodes[0].state, NodeState::New);
        assert!(graph.nodes[0].value.is_none());
    }

    #[test]
    fn nodes_get_sequential_ids() {
        let mut graph = Graph::new();
        let a = graph.add_input(Box::new(1_i64), Revision(1));
        let b = graph.add_input(Box::new(2_i64), Revision(1));
        let c = graph.add_compute(ComputeEntry {
            func: Box::new(|_| Box::new(0_i64)),
            eq_fn: Box::new(|a, b| a.downcast_ref::<i64>() == b.downcast_ref::<i64>()),
        });
        assert_eq!(a, NodeId(0));
        assert_eq!(b, NodeId(1));
        assert_eq!(c, NodeId(2));
    }
}
