//! `Runtime.overlay()` over the journal in incr-python-persist: exit
//! restores the journal, then reads the watched outputs and creators the
//! overlay disturbed so the base graph comes back exactly and drains report
//! nothing the overlay caused.

use incr_compute::{ErasedIncr, Runtime};
use incr_python_protocol::PyValue;
use pyo3::prelude::*;

use crate::persist::{LocalBinding, OverlayScope};
use crate::{engine_read, PyNodeId, PyRuntime};

pub(crate) use incr_python_persist::overlay::Born;

pub(crate) type Undo = incr_python_persist::overlay::Undo<LocalBinding>;
pub(crate) type Overlay = incr_python_persist::overlay::Overlay<LocalBinding>;

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
