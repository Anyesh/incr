//! The Python face of `get_traced` traces and compute stats, shared by
//! both bindings, and the label grouping they and `memory_stats` use.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub use incr_core::label_group;
use incr_core::{
    ComputeCounts, GraphSummary, GroupKind, KeptStep, KeptTrace, KeptValue, TraceAction,
};

/// A kept trace as `get_traced` returns it: `target`, the counts,
/// `elapsed_ns` and `node_traces`, one [`trace_entry`] per step, to which
/// `extra` adds what a binding shows beyond them.
pub fn trace_dict<'py>(
    py: Python<'py>,
    target: u32,
    trace: &KeptTrace,
    mut extra: impl FnMut(&Bound<'py, PyDict>, &KeptStep<'_>) -> PyResult<()>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = trace_header(py, target, trace)?;
    let node_traces = PyList::empty(py);
    for step in trace.steps() {
        let entry = trace_entry(py, &step)?;
        extra(&entry, &step)?;
        node_traces.append(entry)?;
    }
    d.set_item("node_traces", node_traces)?;
    Ok(d)
}

fn trace_header<'py>(
    py: Python<'py>,
    target: u32,
    trace: &KeptTrace,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("target", target)?;
    d.set_item("total_nodes", trace.total_nodes())?;
    d.set_item("nodes_recomputed", trace.nodes_recomputed())?;
    d.set_item("nodes_cutoff", trace.nodes_cutoff())?;
    d.set_item("elapsed_ns", trace.elapsed_ns())?;
    Ok(d)
}

fn trace_entry<'py>(py: Python<'py>, step: &KeptStep<'_>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", step.id.0)?;
    d.set_item(
        "action",
        match step.action {
            TraceAction::VerifiedClean => "verified_clean",
            TraceAction::Recomputed {
                value_changed: true,
            } => "recomputed_changed",
            TraceAction::Recomputed {
                value_changed: false,
            } => "recomputed_cutoff",
        },
    )?;
    d.set_item("dependencies", step.dependencies)?;
    d.set_item("elapsed_ns", step.elapsed_ns)?;
    Ok(d)
}

/// A step's kept value as Python sees it: a reference as kept, a count as
/// an `int`, a pair as a tuple.
pub fn kept_value(py: Python<'_>, value: &KeptValue<'_>) -> Py<PyAny> {
    match value {
        KeptValue::None => py.None(),
        KeptValue::Count(n) => n.into_pyobject(py).unwrap().into_any().unbind(),
        KeptValue::Pair(a, b) => (*a, *b).into_pyobject(py).unwrap().into_any().unbind(),
        KeptValue::Ref(r) => r.get::<Py<PyAny>>().clone_ref(py),
    }
}

/// `{group: {"recomputed", "cutoff", "verified", "elapsed_ns"}}`, with
/// unlabelled nodes under `None`.
pub fn compute_stats_dict<'py>(
    py: Python<'py>,
    by_label: Vec<(Option<String>, ComputeCounts)>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut groups: HashMap<Option<String>, ComputeCounts> = HashMap::new();
    for (label, counts) in by_label {
        groups
            .entry(label.as_deref().map(|l| label_group(l).to_owned()))
            .or_default()
            .add(&counts);
    }
    let out = PyDict::new(py);
    for (group, counts) in groups {
        let d = PyDict::new(py);
        d.set_item("recomputed", counts.recomputed)?;
        d.set_item("cutoff", counts.cutoff)?;
        d.set_item("verified", counts.verified)?;
        d.set_item("elapsed_ns", counts.elapsed_ns)?;
        out.set_item(group, d)?;
    }
    Ok(out)
}

/// `{"groups": [{"label", "kind", "nodes"}], "edges": [{"from", "to",
/// "count"}]}`, with `None` naming the unlabelled group and `kind` one of
/// "input", "compute" and "mixed". A group whose label has an entry in
/// `keys` also gets `"keys"`, a keyed collection's key count.
pub fn graph_summary_dict<'py>(
    py: Python<'py>,
    summary: GraphSummary,
    keys: &HashMap<String, usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let groups = PyList::empty(py);
    for g in summary.groups {
        let d = PyDict::new(py);
        if let Some(n) = g.label.as_ref().and_then(|l| keys.get(l)) {
            d.set_item("keys", n)?;
        }
        d.set_item("label", g.label)?;
        d.set_item(
            "kind",
            match g.kind {
                GroupKind::Input => "input",
                GroupKind::Compute => "compute",
                GroupKind::Mixed => "mixed",
            },
        )?;
        d.set_item("nodes", g.nodes)?;
        groups.append(d)?;
    }
    let edges = PyList::empty(py);
    for e in summary.edges {
        let d = PyDict::new(py);
        d.set_item("from", e.from)?;
        d.set_item("to", e.to)?;
        d.set_item("count", e.count)?;
        edges.append(d)?;
    }
    let out = PyDict::new(py);
    out.set_item("groups", groups)?;
    out.set_item("edges", edges)?;
    Ok(out)
}
