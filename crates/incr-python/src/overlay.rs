//! The journal behind `Runtime.overlay()`: every mutation the binding
//! exposes records its inverse while an overlay is active, and exit applies
//! the inverses newest first, deletes the nodes the overlay created, and
//! reads the watched outputs and creators the overlay disturbed so the base
//! graph comes back exactly and drains report nothing the overlay caused.

use std::cell::Cell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use incr_compute::{Change, ErasedIncr, Incr, IncrCollection, Runtime};
use incr_core::PairOp;
use incr_python_protocol::PyValue;
use pyo3::prelude::*;

use crate::keyed::{KeyStep, SharedIndex};
use crate::persist::OverlayScope;
use crate::{engine_read, CoreKeyed, PyCollection, PyKeyedCollection, PyNodeId, PyRuntime};

/// Set when the overlay that created an object ends, after which the
/// object's nodes are gone and every use of it raises.
#[derive(Clone, Default)]
pub(crate) struct Born(Option<Rc<Cell<bool>>>);

impl Born {
    pub(crate) fn check(&self) -> PyResult<()> {
        match &self.0 {
            Some(dead) if dead.get() => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "collection was created inside an overlay and deleted at its exit",
            )),
            _ => Ok(()),
        }
    }
}

pub(crate) enum Undo {
    Set(Incr<PyValue>, PyValue),
    Collection(IncrCollection<PyValue>, Vec<Change<PyValue>>),
    /// The pair ops a plain keyed collection's write journalled, undone
    /// by applying their inverses newest first.
    Keyed(CoreKeyed, Vec<PairOp<PyValue, ()>>),
    PersistedKeyed {
        pairs: IncrCollection<PyValue>,
        index: Arc<SharedIndex>,
        applied: Vec<Change<PyValue>>,
        steps: Vec<KeyStep>,
    },
    Label(u32, Option<String>),
}

#[derive(Default)]
pub(crate) struct Overlay {
    undo: Vec<Undo>,
    set_seen: HashSet<Incr<PyValue>>,
    created: Vec<ErasedIncr>,
    dead: Vec<Rc<Cell<bool>>>,
    pub(crate) reported_at_entry: HashSet<ErasedIncr>,
    persist: Option<OverlayScope>,
}

fn no_scope() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("the overlay holds no persisted scope")
}

impl Overlay {
    pub(crate) fn push(&mut self, undo: Undo) {
        self.undo.push(undo);
    }

    /// True the first time `node` is set in this overlay, which is the
    /// only set whose old value the exit restores.
    pub(crate) fn first_set(&mut self, node: Incr<PyValue>) -> bool {
        self.set_seen.insert(node)
    }

    pub(crate) fn forget_set(&mut self, node: Incr<PyValue>) {
        self.set_seen.remove(&node);
    }

    pub(crate) fn with_scope(&mut self, scope: Option<OverlayScope>) {
        self.persist = scope;
    }

    pub(crate) fn take_scope(&mut self) -> Option<OverlayScope> {
        self.persist.take()
    }

    pub(crate) fn touch(&mut self, node: &PyNodeId) -> PyResult<()> {
        self.persist.as_mut().ok_or_else(no_scope)?.touch(node)
    }

    pub(crate) fn writable_collection(
        &mut self,
        collection: &PyCollection,
    ) -> PyResult<IncrCollection<PyValue>> {
        self.persist
            .as_mut()
            .ok_or_else(no_scope)?
            .collection(collection)
    }

    pub(crate) fn writable_pairs(
        &mut self,
        kc: &PyKeyedCollection,
    ) -> PyResult<IncrCollection<PyValue>> {
        self.persist.as_mut().ok_or_else(no_scope)?.pairs(kc)
    }

    pub(crate) fn created(&mut self, node: ErasedIncr) {
        self.created.push(node);
    }

    pub(crate) fn born(&mut self) -> Born {
        let dead = Rc::new(Cell::new(false));
        self.dead.push(Rc::clone(&dead));
        Born(Some(dead))
    }

    /// Applies the inverses newest first, deletes the created nodes newest
    /// first and marks their objects dead. The first failure stops the
    /// restore and is returned with the base half restored.
    pub(crate) fn restore(&mut self, rt: &Runtime) -> PyResult<()> {
        let result = self.apply_inverses(rt);
        for node in self.created.drain(..).rev() {
            rt.delete_with_dependents(node);
        }
        result
    }

    fn apply_inverses(&mut self, rt: &Runtime) -> PyResult<()> {
        while let Some(undo) = self.undo.pop() {
            match undo {
                Undo::Set(node, old) => rt.set(node, old),
                Undo::Collection(collection, applied) => {
                    collection.apply(rt, inverse(applied));
                }
                Undo::Keyed(keyed, ops) => {
                    keyed.apply_pairs_with(
                        rt,
                        ops.into_iter().rev().map(PairOp::inverse),
                        &mut (),
                        None,
                    );
                }
                Undo::PersistedKeyed {
                    pairs,
                    index,
                    applied,
                    steps,
                } => {
                    pairs.apply(rt, inverse(applied));
                    let mut guard = index.try_borrow_mut().ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "KeyedCollection index in use at overlay exit",
                        )
                    })?;
                    Python::attach(|py| guard.undo(py, &steps));
                }
                Undo::Label(slot, Some(old)) => rt.set_label(slot, old),
                Undo::Label(slot, None) => rt.clear_label(slot),
            }
        }
        Ok(())
    }
}

impl Drop for Overlay {
    fn drop(&mut self) {
        for dead in &self.dead {
            dead.set(true);
        }
    }
}

/// The batch that undoes `applied`: each change flipped, last first.
pub(crate) fn inverse(applied: Vec<Change<PyValue>>) -> Vec<Change<PyValue>> {
    applied
        .into_iter()
        .rev()
        .map(|change| match change {
            Change::Insert(v) => Change::Delete(v),
            Change::Delete(v) => Change::Insert(v),
        })
        .collect()
}

/// `with rt.overlay() as ov:` runs candidate changes against the live
/// graph and puts the base graph back exactly on exit, also when the body
/// raises. Mutations journal their inverse; operations that cannot be
/// undone raise inside the block. Inverses re-run user `Hash` and `Eq`, so
/// those must be deterministic and must not raise; if one does, the
/// runtime is poisoned and only `close` still works. A restored collection
/// may hold its elements in another order, so `aggregate` combines must be
/// associative and commutative, `sort_by_key` output is exact up to the
/// order among equal keys, and `group_by` keeps a group a candidate
/// emptied or created.
#[pyclass(name = "Overlay", unsendable)]
pub(crate) struct PyOverlay {
    pub(crate) rt: Py<PyRuntime>,
}

#[pymethods]
impl PyOverlay {
    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        {
            let runtime = slf.rt.bind(slf.py()).borrow();
            if runtime.state.overlay_active() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "an overlay is already active on this runtime",
                ));
            }
            let reported = runtime.engine_read(|rt| {
                rt.hold_child_sweeps(true);
                rt.watch_reported()
            })?;
            let scope = match runtime
                .store_core()
                .map(|core| OverlayScope::open(slf.py(), core))
                .transpose()
            {
                Ok(scope) => scope,
                Err(e) => {
                    let rt = runtime.rt()?;
                    rt.hold_child_sweeps(false);
                    return Err(e);
                }
            };
            let mut overlay = Overlay::default();
            overlay.reported_at_entry = reported.into_iter().collect();
            overlay.with_scope(scope);
            runtime.state.begin_overlay(overlay);
        }
        Ok(slf)
    }

    #[pyo3(signature = (_exc_type=None, _exc=None, _tb=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<Py<PyAny>>,
        _exc: Option<Py<PyAny>>,
        _tb: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        let failure = {
            let runtime = self.rt.bind(py).borrow();
            let Some(mut overlay) = runtime.state.end_overlay() else {
                return Ok(false);
            };
            let scope = overlay.take_scope();
            let restored = runtime.engine_write(|rt| {
                let restored = overlay.restore(rt);
                rt.hold_child_sweeps(false);
                restored
            });
            match restored.and_then(|r| r) {
                Ok(()) => {
                    let settled = runtime.engine_read(|rt| settle(rt, &overlay));
                    if settled.is_err() {
                        let rt = runtime.rt()?;
                        rt.take_held_sweep_creators();
                    }
                    let closed = scope.map(|scope| scope.close(py)).transpose();
                    settled?;
                    closed?;
                    None
                }
                Err(e) => Some(e),
            }
        };
        match failure {
            None => Ok(false),
            Some(e) => {
                self.rt.bind(py).borrow_mut().poison();
                Err(e)
            }
        }
    }

    fn set(&self, py: Python<'_>, node: PyNodeId, value: Py<PyAny>) -> PyResult<()> {
        self.rt.bind(py).borrow().set(node, value)
    }

    fn get(&self, py: Python<'_>, node: PyNodeId) -> PyResult<Py<PyAny>> {
        self.rt.bind(py).borrow().get(py, node)
    }

    fn get_with_status(&self, py: Python<'_>, node: PyNodeId) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        self.rt.bind(py).borrow().get_with_status(py, node)
    }
}

/// Reads what the overlay disturbed so it lands at its base value, then
/// drops the reports the overlay raised. A read that fails leaves that
/// node for its next landing, with its report kept.
fn settle(rt: &Runtime, overlay: &Overlay) {
    for creator in rt.take_held_sweep_creators() {
        read_quietly(rt, creator);
    }
    for reported in rt.watch_reported() {
        if !overlay.reported_at_entry.contains(&reported) && read_quietly(rt, reported) {
            rt.unreport_watched(reported);
        }
    }
}

fn read_quietly(rt: &Runtime, handle: ErasedIncr) -> bool {
    let Some(node) = rt.typed_handle::<PyValue>(handle) else {
        return false;
    };
    engine_read(rt, || rt.try_get(node)).is_ok_and(|read| read.is_ok())
}
