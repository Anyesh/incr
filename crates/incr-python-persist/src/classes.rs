//! What a package's runtime class supplies to the persistence classes
//! [`persist_classes!`](crate::persist_classes) writes, and the method
//! bodies those classes share.

use std::sync::Arc;

use incr_core::{ErasedIncr, Incr, IncrCollection, Runtime};
use incr_python_protocol::{PyKey, PyValue};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::PyClass;

use crate::binding::PyBinding;
use crate::memo::MemoFn;
use crate::overlay::Overlay;
use crate::scope::OverlayScope;
use crate::state::RtState;
use crate::store::{Keyed, StoreCore};

/// The runtime class of a package's `Store`.
pub trait StoreRuntime<B: PyBinding>: PyClass + Sized {
    type Node: PyClass + Clone;

    /// The runtime of a store opened over `core`.
    fn of_store(core: Arc<StoreCore<B>>) -> Self;

    /// Raises when a store call cannot run now.
    fn check_store_call(&self) -> PyResult<()>;

    fn node(inner: Incr<PyValue>) -> Self::Node;
    fn node_inner(node: &Self::Node) -> Incr<PyValue>;
}

/// The runtime class of a package that registers every persistence class:
/// its node, collection and keyed collection classes, its state, and how
/// its engine calls run.
pub trait FullStoreRuntime<B: PyBinding>: StoreRuntime<B> {
    type Extra;
    type Collection: PyClass;
    type Keyed: PyClass;

    fn state(&self) -> &RtState<B, Self::Extra>;
    /// The engine, or the error of a closed or poisoned runtime.
    fn rt(&self) -> PyResult<&Runtime<B::C>>;
    fn store_core(&self) -> Option<&Arc<StoreCore<B>>>;
    /// Keeps the runtime for `close` but refuses every other call.
    fn poison(slf: &Bound<'_, Self>);
    fn close_store_runtime(slf: &Bound<'_, Self>) -> PyResult<()>;

    /// Runs `f`, an engine call that writes `rt`, raising an engine unwind.
    fn write_on<R>(rt: &Runtime<B::C>, f: impl FnOnce() -> R) -> PyResult<R>;
    /// [`Self::write_on`] for a call that reads.
    fn read_on<R>(rt: &Runtime<B::C>, f: impl FnOnce() -> R) -> PyResult<R>;

    /// The closure the engine runs for a memo table whose function is
    /// `func`.
    fn memo_fn(state: &RtState<B, Self::Extra>, func: Py<PyAny>) -> MemoFn<B::C>;

    /// The `Collection` over the persisted collection `handle`.
    fn collection(slf: &Bound<'_, Self>, handle: IncrCollection<PyValue, B::C>)
        -> Self::Collection;
    /// The slot of `collection`'s version node, which store calls name it
    /// by.
    fn version_slot(collection: &Self::Collection) -> u32;
    /// The `KeyedCollection` over `persisted`.
    fn keyed(
        slf: &Bound<'_, Self>,
        persisted: Keyed<B>,
        label: Option<String>,
    ) -> PyResult<Self::Keyed>;
    /// The store handle of `kc` and the runtime it belongs to.
    fn keyed_handle(kc: &Self::Keyed) -> (Option<&Keyed<B>>, &Py<Self>);

    fn get(&self, py: Python<'_>, node: Self::Node) -> PyResult<Py<PyAny>>;
    fn get_with_status(&self, py: Python<'_>, node: Self::Node)
        -> PyResult<(Py<PyAny>, Py<PyAny>)>;
    fn set(&self, node: Self::Node, value: Py<PyAny>) -> PyResult<()>;
}

/// The handle of `kc`, which must be a keyed collection `core` persists.
pub fn keyed_of<B: PyBinding, R: FullStoreRuntime<B>>(
    py: Python<'_>,
    core: &Arc<StoreCore<B>>,
    kc: &R::Keyed,
) -> PyResult<Keyed<B>> {
    let (handle, runtime) = R::keyed_handle(kc);
    let runtime = runtime.bind(py).borrow();
    handle
        .filter(|_| {
            runtime
                .store_core()
                .is_some_and(|owner| Arc::ptr_eq(owner, core))
        })
        .cloned()
        .ok_or_else(|| {
            PyValueError::new_err("collection is not a persisted keyed collection of this store")
        })
}

/// The pairs of `rows`, a mapping or an iterable of `(key, row)` pairs,
/// read whole before any key or row method runs.
pub fn reconcile_pairs(rows: &Bound<'_, PyAny>) -> PyResult<Vec<(PyKey, PyValue)>> {
    let pairs = if rows.hasattr("items")? {
        rows.call_method0("items")?
    } else {
        rows.clone()
    };
    pairs
        .try_iter()?
        .map(|pair| {
            let (key, row): (Py<PyAny>, Py<PyAny>) = pair?.extract()?;
            Ok((PyKey(key), PyValue(row)))
        })
        .collect()
}

/// How many keys an operation inserted, gave a new row, and removed.
pub type KeyCounts = (usize, usize, usize);

pub fn key_counts(counts: incr_core::KeyCounts) -> KeyCounts {
    (counts.inserted, counts.replaced, counts.removed)
}

/// Starts an overlay on `runtime`: holds child sweeps, notes the reported
/// watches, and opens an unlogged scope on its store if it has one.
pub fn enter_overlay<B: PyBinding, R: FullStoreRuntime<B>>(
    py: Python<'_>,
    runtime: &R,
) -> PyResult<()> {
    if runtime.state().overlay_active() {
        return Err(PyRuntimeError::new_err(
            "an overlay is already active on this runtime",
        ));
    }
    let rt = runtime.rt()?;
    let reported = R::read_on(rt, || {
        rt.hold_child_sweeps(true);
        rt.watch_reported()
    })?;
    let scope = match runtime
        .store_core()
        .map(|core| OverlayScope::open(py, core))
        .transpose()
    {
        Ok(scope) => scope,
        Err(e) => {
            rt.hold_child_sweeps(false);
            return Err(e);
        }
    };
    let mut overlay = Overlay::default();
    overlay.reported_at_entry = reported.into_iter().collect();
    overlay.with_scope(scope);
    runtime.state().begin_overlay(overlay);
    Ok(())
}

/// Ends the overlay on `runtime`: restores the journal, settles what it
/// disturbed and closes its scope. A failed restore poisons the runtime.
pub fn exit_overlay<B: PyBinding, R: FullStoreRuntime<B>>(
    py: Python<'_>,
    runtime: &Bound<'_, R>,
) -> PyResult<()> {
    let failure = {
        let runtime = runtime.borrow();
        let Some(mut overlay) = runtime.state().end_overlay() else {
            return Ok(());
        };
        let scope = overlay.take_scope();
        let rt = runtime.rt()?;
        let restored = R::write_on(rt, || {
            let restored = overlay.restore(rt);
            rt.hold_child_sweeps(false);
            restored
        });
        match restored.and_then(|r| r) {
            Ok(()) => {
                let settled = R::read_on(rt, || settle::<B, R>(rt, &overlay));
                if settled.is_err() {
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
        None => Ok(()),
        Some(e) => {
            R::poison(runtime);
            Err(e)
        }
    }
}

/// Reads what the overlay disturbed so it lands at its base value, then
/// drops the reports the overlay raised. A read that fails leaves that
/// node for its next landing, with its report kept.
fn settle<B: PyBinding, R: FullStoreRuntime<B>>(rt: &Runtime<B::C>, overlay: &Overlay<B>) {
    for creator in rt.take_held_sweep_creators() {
        read_quietly::<B, R>(rt, creator);
    }
    for reported in rt.watch_reported() {
        if !overlay.reported_at_entry.contains(&reported) && read_quietly::<B, R>(rt, reported) {
            rt.unreport_watched(reported);
        }
    }
}

fn read_quietly<B: PyBinding, R: FullStoreRuntime<B>>(
    rt: &Runtime<B::C>,
    handle: ErasedIncr,
) -> bool {
    let Some(node) = rt.typed_handle::<PyValue>(handle) else {
        return false;
    };
    R::read_on(rt, || rt.try_get(node)).is_ok_and(|read| read.is_ok())
}
