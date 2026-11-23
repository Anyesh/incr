//! `Runtime.memory_stats()` and `Runtime.memory_walker()`: the engine's
//! figures from an `incr_compute::MemoryWalker`, the Python objects in
//! them counted by type, and the Python callables the engine holds, which it
//! cannot see inside its closures ([`crate::state::RtState`] counts them).

use std::any::TypeId;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use incr_compute::{
    KeptTraceMemory, Labels, MemoryVisitor, MemoryWalker, NodeKind, NodeMemory, StateMemory,
    ValueType, WalkEnd,
};
use incr_python_protocol::inspect::label_group;
use incr_python_protocol::{PyPair, PyValue};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// The Python references one value of `type_id` holds, for the types this
/// binding puts in the engine. A `window`'s `Vec<PyValue>` is not among
/// them, as its length is not known from its type.
fn py_refs_of(type_id: TypeId) -> usize {
    if type_id == TypeId::of::<PyValue>()
        || type_id == TypeId::of::<PyPair>()
        || type_id == TypeId::of::<Py<PyTuple>>()
    {
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

/// What a walk has measured so far, in the shape `Runtime.memory_stats()`
/// returns. `bridges` maps a collection's version node to the node its
/// `version_node` getter hands out, so a label set on that node names the
/// collection's state.
pub(crate) struct Tally {
    bridges: HashMap<u32, u32>,
    by_kind: [Totals; 3],
    by_label: HashMap<String, Totals>,
    nodes: usize,
    bytes: usize,
    end: Option<KeptTraceMemory>,
}

impl Tally {
    pub(crate) fn new(bridges: HashMap<u32, u32>) -> Self {
        Self {
            bridges,
            by_kind: Default::default(),
            by_label: HashMap::new(),
            nodes: 0,
            bytes: 0,
            end: None,
        }
    }

    fn label_group(&mut self, label: &str) -> &mut Totals {
        let group = label_group(label);
        if !self.by_label.contains_key(group) {
            self.by_label.insert(group.to_owned(), Totals::default());
        }
        self.by_label
            .get_mut(group)
            .expect("the group was inserted above")
    }

    /// The dict, or `None` before the walk has ended. `callables` counts
    /// the Python callables the engine holds.
    pub(crate) fn to_dict<'py>(
        &self,
        py: Python<'py>,
        callables: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(kept_traces) = self.end else {
            return Ok(None);
        };
        let kinds = PyDict::new(py);
        for (name, totals) in ["input", "query", "collection"].iter().zip(&self.by_kind) {
            kinds.set_item(name, totals.to_dict(py)?)?;
        }
        let labels = PyDict::new(py);
        for (group, totals) in &self.by_label {
            labels.set_item(group, totals.to_dict(py)?)?;
        }
        let d = PyDict::new(py);
        d.set_item("bytes", self.bytes)?;
        d.set_item("nodes", self.nodes)?;
        d.set_item(
            "py_refs",
            self.by_kind.iter().map(|t| t.py_refs).sum::<usize>() + callables,
        )?;
        d.set_item("by_kind", kinds)?;
        d.set_item("by_label", labels)?;
        let kept = PyDict::new(py);
        kept.set_item("count", kept_traces.count)?;
        kept.set_item("bytes", kept_traces.bytes)?;
        kept.set_item("py_refs", kept_traces.refs)?;
        d.set_item("kept_traces", kept)?;
        Ok(Some(d))
    }
}

impl MemoryVisitor for Tally {
    fn state(&mut self, state: StateMemory, labels: &Labels<'_>) {
        let refs: usize = state
            .held
            .iter()
            .map(|h| h.count * py_refs_of(h.type_id))
            .sum();
        self.by_kind[2].add(0, state.bytes, refs);
        self.bytes += state.bytes;
        let label = state.node.and_then(|n| {
            labels
                .get(n.0)
                .or_else(|| self.bridges.get(&n.0).and_then(|&b| labels.get(b)))
        });
        if let Some(label) = label {
            self.label_group(label).add(0, state.bytes, refs);
        }
    }

    fn node(&mut self, node: NodeMemory<&str>, value_type: Option<&ValueType>) {
        let refs = usize::from(
            node.has_value && value_type.is_some_and(|t| t.type_id == TypeId::of::<PyValue>()),
        );
        let kind = match node.kind {
            NodeKind::Input => 0,
            NodeKind::Query => 1,
            NodeKind::Collection => 2,
        };
        self.by_kind[kind].add(1, node.bytes, refs);
        self.nodes += 1;
        self.bytes += node.bytes;
        if let Some(label) = node.label {
            self.label_group(label).add(1, node.bytes, refs);
        }
    }

    fn finish(&mut self, end: WalkEnd) {
        self.bytes += end.free_bytes;
        self.end = Some(end.kept_traces);
    }
}

/// A `memory_stats` walk taken in steps, from `Runtime.memory_walker()`.
/// The states and node slots are those of the runtime when the walk
/// began, each measured as it is when the walk reaches it, so on a
/// runtime that changes between steps the figures can mix moments; on
/// one that does not, `result()` equals `memory_stats()`.
#[pyclass(name = "MemoryWalker", unsendable)]
pub(crate) struct PyMemoryWalker {
    rt: Py<crate::PyRuntime>,
    walker: MemoryWalker,
    tally: Tally,
    callables: usize,
}

impl PyMemoryWalker {
    pub(crate) fn new(rt: Py<crate::PyRuntime>, walker: MemoryWalker, tally: Tally) -> Self {
        Self {
            rt,
            walker,
            tally,
            callables: 0,
        }
    }
}

#[pymethods]
impl PyMemoryWalker {
    /// Walks for about `budget_seconds` (at least one registered state or
    /// one batch of node slots, so a zero budget still advances) and
    /// returns True once the walk has ended. `math.inf` walks to the end.
    /// Raises ValueError for a negative or NaN budget, and RuntimeError on
    /// a closed runtime.
    fn step(&mut self, py: Python<'_>, budget_seconds: f64) -> PyResult<bool> {
        if budget_seconds.is_nan() || budget_seconds < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "budget_seconds must be a non-negative number of seconds",
            ));
        }
        let deadline = Duration::try_from_secs_f64(budget_seconds)
            .ok()
            .and_then(|budget| Instant::now().checked_add(budget));
        let this = self.rt.bind(py).borrow();
        let rt = this.rt()?;
        let (walker, tally) = (&mut self.walker, &mut self.tally);
        let done = crate::engine_read(rt, || {
            walker
                .belongs_to(rt)
                .then(|| walker.step(rt, tally, deadline))
        })?
        .ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("the runtime this walk began on was replaced")
        })?;
        if done {
            self.callables = this.state.callables();
        }
        Ok(done)
    }

    /// The `memory_stats()` dict this walk measured. Raises RuntimeError
    /// until `step` has returned True.
    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.tally.to_dict(py, self.callables)?.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "the walk has not ended: call step until it returns True",
            )
        })
    }
}
