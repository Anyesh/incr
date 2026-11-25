//! `incr.persist`: durable inputs and collections for a Local runtime,
//! over any Python object that implements the backend protocol.
//!
//! A [`PyStore`] owns an `incr.Runtime` together with the incr-persist
//! store that keeps its persisted inputs and collections in a backend's
//! log. Values cross to bytes through [`codec`], one user codec per
//! persisted key.

mod memo;
mod memory;
mod runtime;

use std::sync::Arc;

use incr_core::Local;
use incr_python_persist::binding::{Carry, CoreStore, Detached, PyBinding};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

pub(crate) use classes::{PyMemoTable, PyOverlay};

pub(crate) type StoreCore = incr_python_persist::store::StoreCore<LocalBinding>;
pub(crate) type PyPersistedKeyed = incr_python_persist::store::Keyed<LocalBinding>;

incr_python_persist::persist_exceptions!(Exceptions, incr.persist);
incr_python_persist::not_ready_signal!(Signal, incr);

/// Local store calls keep the GIL, since the runtime is confined to the
/// opener's thread, and route as `engine_write` and `engine_read` do.
pub(crate) struct LocalBinding;

impl PyBinding for LocalBinding {
    type C = Local;
    type Exceptions = Exceptions;
    type Signal = Signal;
    type Task<T> = LocalTask<T>;

    fn write<R: Send>(
        _py: Python<'_>,
        store: &Arc<CoreStore<Self>>,
        f: impl FnOnce(&Arc<CoreStore<Self>>, Detached) -> R + Send,
    ) -> PyResult<R> {
        crate::engine_write(store.runtime(), || f(store, Detached::assume()))
    }

    fn read<R: Send>(
        _py: Python<'_>,
        store: &Arc<CoreStore<Self>>,
        f: impl FnOnce(&Arc<CoreStore<Self>>, Detached) -> R + Send,
    ) -> PyResult<R> {
        crate::engine_read(store.runtime(), || f(store, Detached::assume()))
    }

    fn drop_detached<T>(_py: Python<'_>, value: T)
    where
        LocalTask<T>: Send,
    {
        drop(value);
    }
}

/// A Local engine handle carried into or out of a store call.
pub(crate) struct LocalTask<T>(T);

impl<T> Carry<T> for LocalTask<T> {
    fn new(value: T) -> Self {
        LocalTask(value)
    }

    fn into_inner(self) -> T {
        self.0
    }
}

// SAFETY: a `LocalTask` holds `Runtime<Local>` handles, which must stay on
// the runtime's thread. Only `LocalBinding::write` and `LocalBinding::read`
// run the tasks that carry one, and both run them in place on the calling
// thread. The type is private to the Local binding, so no Shared path,
// which may run a task on another thread, can receive one.
unsafe impl<T> Send for LocalTask<T> {}

mod classes {
    use super::LocalBinding;
    use crate::PyRuntime;

    incr_python_persist::persist_classes!(
        binding = LocalBinding,
        runtime = PyRuntime,
        module = "incr.persist",
        flags = [unsendable],
        store_methods = {
            /// Starts a checkpoint that runs as a flight and returns its
            /// `PendingCheckpoint`: each `poll` copies the store's tables for
            /// about its budget, and a worker thread folds, encodes and writes
            /// the snapshot while writes go on. Refuses what `checkpoint()`
            /// refuses, and raises `PersistError` for a backend whose class
            /// does not set `concurrent_compaction = True` or has no
            /// `compact_streamed`. Until the flight ends, `checkpoint()` and
            /// `begin_checkpoint()` raise `Busy` and the automatic checkpoint
            /// waits.
            fn begin_checkpoint(
                slf: &Bound<'_, Self>,
            ) -> PyResult<incr_python_persist::flight::PendingCheckpoint> {
                incr_python_persist::flight::begin_store::<LocalBinding, Self>(slf)
            }
        },
        memo_methods = {
            /// Starts a memo checkpoint that runs as a flight and returns its
            /// `PendingCheckpoint`: each `poll` inspects entries for about its
            /// budget, a worker thread encodes and writes the generation, and
            /// the polls after it finish the entries it wrote; `result()`
            /// returns what `checkpoint()` returns. Raises `PersistError` for
            /// a memo backend whose class does not set
            /// `concurrent_compaction = True` or has no
            /// `write_generation_streamed`. Until the flight ends,
            /// `checkpoint()` and `begin_checkpoint()` raise `Busy` and the
            /// automatic checkpoint waits.
            fn begin_checkpoint(
                slf: &Bound<'_, Self>,
            ) -> PyResult<incr_python_persist::flight::PendingCheckpoint> {
                incr_python_persist::flight::begin_memo::<LocalBinding, Self>(slf)
            }
        }
    );
}

#[cfg(feature = "conformance")]
mod conformance {
    use super::Exceptions;

    incr_python_persist::conformance_functions!(
        Exceptions,
        incr.persist.conformance,
        "incr.persist.conformance"
    );
}

/// Creates `incr.persist`, adds it to `parent` and registers it in
/// `sys.modules`, so `import incr.persist` and `from incr.persist import X`
/// work although `incr` is a single extension module.
pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let module = PyModule::new(py, "persist")?;
    module.setattr("__name__", "incr.persist")?;
    module.add("PersistError", py.get_type::<PersistError>())?;
    module.add("Busy", py.get_type::<Busy>())?;
    module.add("ReadOnly", py.get_type::<ReadOnly>())?;
    module.add("InvalidCompaction", py.get_type::<InvalidCompaction>())?;
    module.add("TailMoved", py.get_type::<TailMoved>())?;
    module.add_class::<memory::PyMemoryBackend>()?;
    module.add_class::<memory::PyMemoryMemo>()?;
    module.add_class::<memory::PyMemoryMemoGuard>()?;
    module.add_class::<classes::PyMemo>()?;
    module.add_class::<classes::PyMemoTable>()?;
    module.add_class::<classes::PyStore>()?;
    module.add_class::<classes::PyBatch>()?;
    module.add_class::<incr_python_persist::flight::PendingCheckpoint>()?;
    let join = wrap_pyfunction!(
        incr_python_persist::flight::_join_checkpoint_workers,
        &module
    )?;
    module.add_function(join.clone())?;
    py.import("atexit")?.call_method1("register", (join,))?;
    #[cfg(feature = "gil-checks")]
    {
        let checks = PyModule::new(py, "_gil_checks")?;
        incr_python_protocol::gil::checks::register(&checks)?;
        checks.add_function(wrap_pyfunction!(
            incr_python_persist::flight::_set_memo_frame_limit,
            &checks
        )?)?;
        checks.add_function(wrap_pyfunction!(
            incr_python_persist::flight::_checkpoint_workers_running,
            &checks
        )?)?;
        module.add_submodule(&checks)?;
    }
    #[cfg(feature = "conformance")]
    conformance::register_conformance(&module)?;
    incr_python_persist::codec::install_hook::<Exceptions>();
    parent.add("persist", &module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("incr.persist", &module)?;
    Ok(())
}
