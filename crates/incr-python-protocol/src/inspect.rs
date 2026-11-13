//! The Python face of `get_traced` entries and compute stats, shared by
//! both bindings, and the label grouping they and `memory_stats` use.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use incr_core::{ComputeCounts, NodeTrace, TraceAction};

/// The group a label reports under: the label up to its first `/`.
pub fn label_group(label: &str) -> &str {
    label.split('/').next().unwrap_or(label)
}

/// A trace entry's `id`, `action`, `dependencies` and `elapsed_ns`.
pub fn trace_entry<'py>(py: Python<'py>, entry: &NodeTrace) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", entry.id.0)?;
    d.set_item(
        "action",
        match entry.action {
            TraceAction::VerifiedClean => "verified_clean",
            TraceAction::Recomputed {
                value_changed: true,
            } => "recomputed_changed",
            TraceAction::Recomputed {
                value_changed: false,
            } => "recomputed_cutoff",
        },
    )?;
    d.set_item(
        "dependencies",
        entry.dependencies.iter().map(|d| d.0).collect::<Vec<_>>(),
    )?;
    d.set_item("elapsed_ns", entry.elapsed_ns)?;
    Ok(d)
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
