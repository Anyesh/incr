//! `Runtime.memory_stats()`: the engine's figures from
//! `incr_compute::Runtime::memory_stats`, the Python objects in them
//! counted by type, and the Python callables the engine holds, which it
//! cannot see inside its closures ([`crate::state::RtState`] counts them).

use std::any::TypeId;
use std::collections::HashMap;

use incr_compute::{MemoryStats, NodeKind};
use incr_python_protocol::inspect::label_group;
use incr_python_protocol::PyValue;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// The Python references one value of `type_id` holds, for the types this
/// binding puts in the engine. A `window`'s `Vec<PyValue>` is not among
/// them, as its length is not known from its type.
fn py_refs_of(type_id: TypeId) -> usize {
    if type_id == TypeId::of::<PyValue>() || type_id == TypeId::of::<Py<PyTuple>>() {
        1
    } else if type_id == TypeId::of::<(PyValue, PyValue)>() {
        2
    } else {
        0
    }
}

#[derive(Default)]
struct Totals {
    nodes: usize,
    bytes: usize,
    py_refs: usize,
}

impl Totals {
    fn add(&mut self, nodes: usize, bytes: usize, py_refs: usize) {
        self.nodes += nodes;
        self.bytes += bytes;
        self.py_refs += py_refs;
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("nodes", self.nodes)?;
        d.set_item("bytes", self.bytes)?;
        d.set_item("py_refs", self.py_refs)?;
        Ok(d)
    }
}

/// The dict `Runtime.memory_stats()` returns. `bridges` maps a
/// collection's version node to the node its `version_node` getter hands
/// out, so a label set on that node names the collection's state.
pub(crate) fn stats_dict<'py>(
    py: Python<'py>,
    stats: &MemoryStats,
    callables: usize,
    bridges: &HashMap<u32, u32>,
) -> PyResult<Bound<'py, PyDict>> {
    let py_value: Vec<bool> = stats
        .value_types
        .iter()
        .map(|t| t.type_id == TypeId::of::<PyValue>())
        .collect();
    let mut by_kind: [Totals; 3] = Default::default();
    let mut by_label: HashMap<&str, Totals> = HashMap::new();
    for node in &stats.nodes {
        let refs = usize::from(node.has_value && py_value[node.type_tag as usize]);
        let kind = match node.kind {
            NodeKind::Input => 0,
            NodeKind::Query => 1,
            NodeKind::Collection => 2,
        };
        by_kind[kind].add(1, node.bytes, refs);
        if let Some(label) = &node.label {
            by_label
                .entry(label_group(label))
                .or_default()
                .add(1, node.bytes, refs);
        }
    }

    let label_of = |slot: u32| -> Option<&str> {
        let i = stats.nodes.binary_search_by_key(&slot, |n| n.slot).ok()?;
        stats.nodes[i].label.as_deref()
    };
    for state in &stats.states {
        let refs: usize = state
            .held
            .iter()
            .map(|h| h.count * py_refs_of(h.type_id))
            .sum();
        by_kind[2].add(0, state.bytes, refs);
        let label = state
            .node
            .and_then(|n| label_of(n.0).or_else(|| bridges.get(&n.0).and_then(|&b| label_of(b))));
        if let Some(label) = label {
            by_label
                .entry(label_group(label))
                .or_default()
                .add(0, state.bytes, refs);
        }
    }

    let kinds = PyDict::new(py);
    for (name, totals) in ["input", "query", "collection"].iter().zip(&by_kind) {
        kinds.set_item(name, totals.to_dict(py)?)?;
    }
    let labels = PyDict::new(py);
    for (prefix, totals) in &by_label {
        labels.set_item(prefix, totals.to_dict(py)?)?;
    }
    let d = PyDict::new(py);
    d.set_item("bytes", stats.bytes())?;
    d.set_item("nodes", stats.nodes.len())?;
    d.set_item(
        "py_refs",
        by_kind.iter().map(|t| t.py_refs).sum::<usize>() + callables,
    )?;
    d.set_item("by_kind", kinds)?;
    d.set_item("by_label", labels)?;
    Ok(d)
}
