//! How a package runs store calls: [`PyBinding`].

use std::sync::Arc;

use incr_persist::memo::MemoCells;
use incr_persist::Store;
use pyo3::prelude::*;

use crate::backend::PyBackend;
use crate::ready::NotReadySignal;
use crate::PersistExceptions;

/// The incr-persist store a package's Python store wraps.
pub type CoreStore<B> = Store<<B as PyBinding>::C, PyBackend<<B as PyBinding>::Exceptions>>;

/// Marks code running where [`PyBinding::write`] put it. It carries no
/// `Python` token: engine-side code, which on Shared runs with the GIL
/// released, reaches Python only through `Python::attach`.
pub struct Detached(());

impl Detached {
    /// Only [`PyBinding::write`] implementations create one.
    #[doc(hidden)]
    pub fn assume() -> Self {
        Detached(())
    }
}

/// A value a store call carries into and out of [`PyBinding::write`].
pub trait Carry<T> {
    fn new(value: T) -> Self;
    fn into_inner(self) -> T;
}

/// The carrier of a binding whose `write` may run its task on another
/// thread: it is `Send` exactly when the value is.
pub struct Plain<T>(T);

impl<T> Carry<T> for Plain<T> {
    fn new(value: T) -> Self {
        Plain(value)
    }

    fn into_inner(self) -> T {
        self.0
    }
}

/// What a Python package supplies to the shared persistence layer: its
/// runtime's cells, its exception types, and how a store call runs.
pub trait PyBinding: Sized + 'static {
    type C: MemoCells;
    type Exceptions: PersistExceptions;
    /// The package's `_NotReadySignal`, under its own module name.
    type Signal: NotReadySignal;
    /// Wraps an engine handle a task passed to [`Self::write`] captures or
    /// returns, since `write` demands a `Send` task. Code generic over the
    /// binding states the instances it needs as `Send` bounds, which hold
    /// for every binding whose handles are `Send`.
    type Task<T>: Carry<T>;

    /// Runs `f`, a store call that may write the runtime, on `store`, with
    /// a fresh [`Detached`]. The package decides whether the GIL stays
    /// held (Shared releases it, since the store mutex and the engine's
    /// locks run Python under them) and how writes route. An engine unwind
    /// or routing failure is raised as the package's exception.
    fn write<R: Send>(
        py: Python<'_>,
        store: &Arc<CoreStore<Self>>,
        f: impl FnOnce(&Arc<CoreStore<Self>>, Detached) -> R + Send,
    ) -> PyResult<R>;

    /// [`Self::write`] for a store call that only reads the runtime, which
    /// the package may route first.
    fn read<R: Send>(
        py: Python<'_>,
        store: &Arc<CoreStore<Self>>,
        f: impl FnOnce(&Arc<CoreStore<Self>>, Detached) -> R + Send,
    ) -> PyResult<R>;

    /// Drops `value`, whose drop may take the store mutex (a staged batch
    /// that has not landed poisons the store) or join a memo's checkpoint
    /// thread, both of which may wait on a thread that needs the GIL.
    /// Shared releases the GIL for the drop; Local drops in place.
    fn drop_detached<T>(py: Python<'_>, value: T)
    where
        Self::Task<T>: Send;
}

/// `value` wrapped for a task of `B`.
pub fn carry<B: PyBinding, T>(value: T) -> B::Task<T> {
    <B::Task<T> as Carry<T>>::new(value)
}
