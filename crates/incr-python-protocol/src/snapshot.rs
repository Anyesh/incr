//! The Python face of `graph_snapshot`, shared by both bindings: argument
//! parsing for the bounded form and the dict each node is reported as.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use incr_core::{BoundedSnapshot, Direction, GraphQuery, NodeId, NodeInfo, NodeKindInfo};

/// A validated bounded `graph_snapshot` call.
pub struct SnapshotRequest {
    roots: Vec<NodeId>,
    direction: Direction,
    depth: Option<u32>,
    label_prefix: Option<String>,
    limit: Option<usize>,
}

impl SnapshotRequest {
    /// `Ok(None)` when every argument is unset, which asks for the full
    /// snapshot. Otherwise `roots` is required, each root is a node handle
    /// (`slot_of` yields its slot) or an object whose `version_node` is
    /// one, and `direction` is `"dependencies"` (the default),
    /// `"dependents"` or `"both"`; anything else is a `ValueError`.
    pub fn parse<'py>(
        roots: Option<Vec<Bound<'py, PyAny>>>,
        direction: Option<&str>,
        depth: Option<u32>,
        label_prefix: Option<String>,
        limit: Option<usize>,
        slot_of: impl Fn(&Bound<'py, PyAny>) -> Option<u32>,
    ) -> PyResult<Option<Self>> {
        if roots.is_none()
            && direction.is_none()
            && depth.is_none()
            && label_prefix.is_none()
            && limit.is_none()
        {
            return Ok(None);
        }
        let direction = match direction {
            None | Some("dependencies") => Direction::Dependencies,
            Some("dependents") => Direction::Dependents,
            Some("both") => Direction::Both,
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "direction must be 'dependencies', 'dependents' or 'both', not {other:?}"
                )))
            }
        };
        let Some(roots) = roots else {
            return Err(PyValueError::new_err(
                "graph_snapshot needs roots when depth, direction, label_prefix or limit is given",
            ));
        };
        let mut slots = Vec::with_capacity(roots.len());
        for root in &roots {
            let slot = match slot_of(root) {
                Some(slot) => Some(slot),
                None => root
                    .getattr("version_node")
                    .ok()
                    .and_then(|version| slot_of(&version)),
            };
            let slot = slot.ok_or_else(|| {
                PyValueError::new_err("roots must be node handles or collections")
            })?;
            slots.push(NodeId(slot));
        }
        Ok(Some(Self {
            roots: slots,
            direction,
            depth,
            label_prefix,
            limit,
        }))
    }

    pub fn query(&self) -> GraphQuery<'_> {
        GraphQuery {
            roots: &self.roots,
            direction: self.direction,
            depth: self.depth,
            label_prefix: self.label_prefix.as_deref(),
            limit: self.limit,
            max_visited: None,
        }
    }
}

fn info_dict<'py>(py: Python<'py>, info: &NodeInfo) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", info.id.0)?;
    d.set_item(
        "kind",
        match info.kind {
            NodeKindInfo::Input => "input",
            NodeKindInfo::Compute => "compute",
        },
    )?;
    d.set_item("label", &info.label)?;
    let deps: Vec<u32> = info.dependencies.iter().map(|n| n.0).collect();
    let depts: Vec<u32> = info.dependents.iter().map(|n| n.0).collect();
    d.set_item("dependencies", deps)?;
    d.set_item("dependents", depts)?;
    Ok(d)
}

fn info_list<'py>(py: Python<'py>, infos: &[NodeInfo]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for info in infos {
        list.append(info_dict(py, info)?)?;
    }
    Ok(list)
}

/// The full snapshot as a list of node dicts.
pub fn full_snapshot_to_py(py: Python<'_>, infos: &[NodeInfo]) -> PyResult<Py<PyAny>> {
    Ok(info_list(py, infos)?.into_any().unbind())
}

/// A bounded snapshot as `{"nodes": [node dicts], "truncated": bool}`.
pub fn bounded_snapshot_to_py(py: Python<'_>, snap: &BoundedSnapshot) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("nodes", info_list(py, &snap.nodes)?)?;
    d.set_item("truncated", snap.truncated)?;
    Ok(d.into_any().unbind())
}
