//! Reads by snapshot id, shared by both bindings: a `(slot, generation)`
//! pair from `graph_snapshot` resolves to a live node, or raises
//! `StaleNodeError`, and the node's value type is resolved in one place for
//! `get_by_id`, `get_traced_by_id`, trace entries and snapshot values.

use pyo3::prelude::*;

use incr_core::persist_hooks::{inspect_many, Deferred};
use incr_core::{
    Cells, ErasedIncr, Incr, KeptTrace, NodeId, NodeInfo, NodeState, PropagationTrace, Runtime,
    StepValue, TraceCapture,
};

use crate::snapshot::status;
use crate::{PyValue, StaleNodeError};

/// A node whose value type Python can read: a `PyValue`, or the `u64` and
/// `(u64, u64)` of a collection's version and position bookkeeping.
#[derive(Clone, Copy)]
pub enum TypedNode {
    Py(Incr<PyValue>),
    Count(Incr<u64>),
    Pair(Incr<(u64, u64)>),
}

impl TypedNode {
    /// `None` when `handle` is dead or holds another value type.
    pub fn of<C: Cells>(rt: &Runtime<C>, handle: ErasedIncr) -> Option<Self> {
        if let Some(node) = rt.typed_handle::<PyValue>(handle) {
            return Some(Self::Py(node));
        }
        if let Some(node) = rt.typed_handle::<u64>(handle) {
            return Some(Self::Count(node));
        }
        rt.typed_handle::<(u64, u64)>(handle).map(Self::Pair)
    }

    /// The value as stored, computing it if the node is not Clean; `None`
    /// when the read closes a cycle that leaves the node without a value.
    pub fn try_value<C: Cells>(self, rt: &Runtime<C>) -> Option<Py<PyAny>> {
        match self {
            Self::Py(node) => rt.try_get(node).ok().map(|v| v.0),
            Self::Count(node) => rt.try_get(node).ok().map(count_to_py),
            Self::Pair(node) => rt.try_get(node).ok().map(pair_to_py),
        }
    }

    /// [`Self::try_value`] in the form a kept trace stores it: a count or
    /// pair natively, a Python value by reference.
    pub fn kept_value<C: Cells>(self, rt: &Runtime<C>) -> StepValue<Py<PyAny>> {
        let value = match self {
            Self::Py(node) => rt.try_get(node).ok().map(|v| StepValue::Ref(v.0)),
            Self::Count(node) => rt.try_get(node).ok().map(StepValue::Count),
            Self::Pair(node) => rt.try_get(node).ok().map(|(a, b)| StepValue::Pair(a, b)),
        };
        value.unwrap_or(StepValue::None)
    }

    /// [`Runtime::get_traced_retaining`] for any of the readable types.
    pub fn get_traced_retaining<C: Cells>(
        self,
        rt: &Runtime<C>,
    ) -> (Py<PyAny>, PropagationTrace, TraceCapture) {
        match self {
            Self::Py(node) => {
                let (value, trace, capture) = rt.get_traced_retaining(node);
                (value.0, trace, capture)
            }
            Self::Count(node) => {
                let (value, trace, capture) = rt.get_traced_retaining(node);
                (count_to_py(value), trace, capture)
            }
            Self::Pair(node) => {
                let (value, trace, capture) = rt.get_traced_retaining(node);
                (pair_to_py(value), trace, capture)
            }
        }
    }
}

fn count_to_py(n: u64) -> Py<PyAny> {
    Python::attach(|py| n.into_pyobject(py).unwrap().into_any().unbind())
}

fn pair_to_py(pair: (u64, u64)) -> Py<PyAny> {
    Python::attach(|py| pair.into_pyobject(py).unwrap().into_any().unbind())
}

/// A `(slot, generation)` that names no live node of the runtime, or one
/// holding a value type Python cannot read.
pub enum ByIdError {
    Stale { id: u32, generation: u32 },
    Unreadable { id: u32 },
}

impl From<ByIdError> for PyErr {
    fn from(e: ByIdError) -> Self {
        match e {
            ByIdError::Stale { id, generation } => StaleNodeError::new_err(format!(
                "node {id} at generation {generation} is deleted, was reused, or never existed"
            )),
            ByIdError::Unreadable { id } => pyo3::exceptions::PyTypeError::new_err(format!(
                "node {id} holds a value Python cannot read"
            )),
        }
    }
}

/// The readable node at `id` and `generation`. The slot is bounds-checked
/// before the generation, which `Runtime::is_alive_at` assumes in range.
pub fn resolve<C: Cells>(
    rt: &Runtime<C>,
    id: u32,
    generation: u32,
) -> Result<TypedNode, ByIdError> {
    let handle = rt
        .erased_handle_at(NodeId(id), generation)
        .ok_or(ByIdError::Stale { id, generation })?;
    TypedNode::of(rt, handle).ok_or(ByIdError::Unreadable { id })
}

/// The values of `infos`' Clean nodes, in order: `None` for a node that is
/// not Clean, holds an unreadable type, or stopped being Clean while read.
/// Nothing recomputes. `hold_values` runs its argument once; a Shared
/// binding takes the GIL around it so each `PyValue` clone is a refcount
/// bump inside one stretch.
pub fn snapshot_values<C: Cells>(
    rt: &Runtime<C>,
    infos: &[NodeInfo],
    hold_values: impl FnOnce(&mut dyn FnMut()),
) -> Vec<Option<Py<PyAny>>> {
    let mut pys: Vec<(usize, Incr<PyValue>)> = Vec::new();
    let mut counts: Vec<(usize, Incr<u64>)> = Vec::new();
    let mut pairs: Vec<(usize, Incr<(u64, u64)>)> = Vec::new();
    for (index, info) in infos.iter().enumerate() {
        if info.state != NodeState::Clean {
            continue;
        }
        let Some(handle) = rt.erased_handle_at(info.id, info.generation) else {
            continue;
        };
        match TypedNode::of(rt, handle) {
            Some(TypedNode::Py(node)) => pys.push((index, node)),
            Some(TypedNode::Count(node)) => counts.push((index, node)),
            Some(TypedNode::Pair(node)) => pairs.push((index, node)),
            None => {}
        }
    }
    let mut out: Vec<Option<Py<PyAny>>> = infos.iter().map(|_| None).collect();
    let mut bag = Deferred::new();
    let qs: Vec<Incr<PyValue>> = pys.iter().map(|(_, q)| *q).collect();
    let seen = inspect_many(rt, &qs, hold_values, &mut bag);
    for ((index, _), seen) in pys.iter().zip(seen) {
        out[*index] = seen.map(|seen| seen.value.0);
    }
    for ((index, _), seen) in counts.iter().zip(read_plain(rt, &counts, &mut bag)) {
        out[*index] = seen.map(count_to_py);
    }
    for ((index, _), seen) in pairs.iter().zip(read_plain(rt, &pairs, &mut bag)) {
        out[*index] = seen.map(pair_to_py);
    }
    bag.drain(rt);
    out
}

fn read_plain<C: Cells, T: incr_core::Value>(
    rt: &Runtime<C>,
    nodes: &[(usize, Incr<T>)],
    bag: &mut Deferred<C>,
) -> Vec<Option<T>> {
    if nodes.is_empty() {
        return Vec::new();
    }
    let qs: Vec<Incr<T>> = nodes.iter().map(|(_, q)| *q).collect();
    inspect_many(rt, &qs, |f| f(), bag)
        .into_iter()
        .map(|seen| seen.map(|seen| seen.value))
        .collect()
}

/// The status `graph_snapshot(values=True)` reports for the node at `id`
/// and `generation`, and its value when it is Clean and readable. Nothing
/// recomputes.
pub fn peek<C: Cells>(
    rt: &Runtime<C>,
    id: u32,
    generation: u32,
    hold_values: impl FnOnce(&mut dyn FnMut()),
) -> Result<(&'static str, Option<Py<PyAny>>), ByIdError> {
    let info = rt
        .node_info(NodeId(id), generation)
        .ok_or(ByIdError::Stale { id, generation })?;
    let value = snapshot_values(rt, std::slice::from_ref(&info), hold_values)
        .pop()
        .flatten();
    Ok((status(info.state), value))
}

/// The status of the node at `id` and `generation`, as [`peek`] reports
/// it, and the trace a binding kept for it. Nothing recomputes.
pub fn last_trace<C: Cells>(
    rt: &Runtime<C>,
    id: u32,
    generation: u32,
) -> Result<(&'static str, Option<KeptTrace>), ByIdError> {
    let found = rt
        .last_trace(NodeId(id), generation)
        .ok_or(ByIdError::Stale { id, generation })?;
    Ok((status(found.state), found.trace))
}

/// The `(value, trace)` a traced read returns, from `render`, with `kept`
/// kept for [`last_trace`]. Takes the GIL for `render` and keeps the trace
/// after releasing it, so a Shared binding drops a trace it replaces
/// without the GIL, as its engine calls drop values.
pub fn keep_traced<C: Cells>(
    rt: &Runtime<C>,
    capture: TraceCapture,
    kept: KeptTrace,
    render: impl FnOnce(Python<'_>, &KeptTrace) -> PyResult<(Py<PyAny>, Py<PyAny>)>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let out = Python::attach(|py| render(py, &kept))?;
    rt.retain_trace(capture, kept);
    Ok(out)
}
