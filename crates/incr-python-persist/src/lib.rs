//! The persistence layer of incr's Python bindings, shared by the Local
//! (`incr`) and Shared (`incr_concurrent`) packages.
//!
//! Each package declares its own `persist` exception types with
//! [`persist_exceptions!`]; everything here is generic over them through
//! [`PersistExceptions`]. Values cross to bytes through [`codec`], one user codec
//! per persisted slot.

pub mod backend;
pub mod batch;
pub mod binding;
pub mod classes;
pub mod codec;
#[cfg(feature = "conformance")]
pub mod conformance;
pub mod keyed;
pub mod memo;
pub mod memo_backend;
mod memory;
pub mod overlay;
mod persist_classes;
pub mod ready;
pub mod scope;
pub mod state;
pub mod store;

#[doc(hidden)]
pub use incr_core;
#[doc(hidden)]
pub use incr_persist;
#[doc(hidden)]
pub use incr_python_protocol;
#[doc(hidden)]
pub use pyo3;

use incr_persist::Error;
use incr_python_protocol::PyKey;
use pyo3::prelude::*;
use pyo3::types::PyType;

/// A package's `persist` exception types. Implemented by
/// [`persist_exceptions!`].
pub trait PersistExceptions: Send + Sync + 'static {
    /// A store or backend failure no Python exception describes better.
    fn persist_error(py: Python<'_>) -> Bound<'_, PyType>;
    /// A checkpoint skipped because another handle holds its lock.
    fn busy(py: Python<'_>) -> Bound<'_, PyType>;
    /// A write that reached a read-only handle.
    fn read_only(py: Python<'_>) -> Bound<'_, PyType>;
    /// A compaction outside the snapshot and the tail, constructed as
    /// `InvalidCompaction(upto, snapshot_at, tail)`.
    fn invalid_compaction(py: Python<'_>) -> Bound<'_, PyType>;
    /// A conditional append that found the log's tail elsewhere and wrote
    /// nothing.
    fn tail_moved(py: Python<'_>) -> Bound<'_, PyType>;
}

/// The Python exception for `error`: the exception a Python backend
/// raised, as itself, or a `PersistError` carrying the whole chain.
pub fn persist_error<E: PersistExceptions>(py: Python<'_>, error: Error) -> PyErr {
    if let Error::Backend(source) = &error {
        if let Some(raised) = source.downcast_ref::<backend::PyBackendError>() {
            return raised.err().clone_ref(py);
        }
    }
    match error {
        Error::Busy => return PyErr::from_type(E::busy(py), error.to_string()),
        Error::ReadOnly => return PyErr::from_type(E::read_only(py), error.to_string()),
        Error::TailMoved { .. } => return PyErr::from_type(E::tail_moved(py), error.to_string()),
        Error::KeyedContention { .. } => {
            return pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
        }
        Error::UnstableRow {
            ref key,
            ref row_key,
        } => {
            let row_key = row_key
                .downcast_ref::<PyKey>()
                .and_then(|row_key| row_key.0.bind(py).repr().ok())
                .map_or_else(|| "a key".to_owned(), |repr| repr.to_string());
            return pyo3::exceptions::PyValueError::new_err(format!(
                "keyed collection '{key}', key {row_key}: the held row does not equal its decoded \
                 copy and its codec encodes it differently from the logged bytes, so a replay \
                 would not find it; nothing was written"
            ));
        }
        Error::InvalidCompaction {
            upto,
            snapshot_at,
            tail,
        } => {
            return PyErr::from_type(E::invalid_compaction(py), (upto.0, snapshot_at.0, tail.0));
        }
        _ => {}
    }
    let mut message = error.to_string();
    let mut source = std::error::Error::source(&error);
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    PyErr::from_type(E::persist_error(py), message)
}

/// Declares a package's `persist` exceptions under the Python module path
/// `$module` (such as `incr.persist`) and a unit type `$name` implementing
/// [`PersistExceptions`] for them. Defines `PersistError`, `Busy`, `ReadOnly`,
/// `InvalidCompaction` and `TailMoved` in the calling module.
#[macro_export]
macro_rules! persist_exceptions {
    ($name:ident, $module:expr) => {
        $crate::pyo3::create_exception!(
            $module,
            PersistError,
            $crate::pyo3::exceptions::PyRuntimeError,
            "A store or backend failure that no Python exception describes better."
        );

        $crate::pyo3::create_exception!(
            $module,
            Busy,
            PersistError,
            "Another handle holds the memo checkpoint lock or has written a newer generation: the checkpoint wrote nothing and can be retried."
        );

        $crate::pyo3::create_exception!(
            $module,
            ReadOnly,
            PersistError,
            "A write reached a handle opened read-only."
        );

        $crate::pyo3::create_exception!(
            $module,
            InvalidCompaction,
            PersistError,
            "`compact(upto, ..)` was given an `upto` below the snapshot position or beyond the tail; constructed as `InvalidCompaction(upto, snapshot_at, tail)`."
        );

        $crate::pyo3::create_exception!(
            $module,
            TailMoved,
            PersistError,
            "A backend's `append_at(after, records)` found the log's tail somewhere other than `after` and wrote nothing. A Python backend raises it to say so, using the class of the package whose store it serves: `incr.persist.TailMoved` and `incr_concurrent.persist.TailMoved` are distinct classes, and a store reads the other package's as a plain backend error."
        );

        pub(crate) struct $name;

        impl $crate::PersistExceptions for $name {
            fn persist_error(
                py: $crate::pyo3::Python<'_>,
            ) -> $crate::pyo3::Bound<'_, $crate::pyo3::types::PyType> {
                py.get_type::<PersistError>()
            }

            fn busy(
                py: $crate::pyo3::Python<'_>,
            ) -> $crate::pyo3::Bound<'_, $crate::pyo3::types::PyType> {
                py.get_type::<Busy>()
            }

            fn read_only(
                py: $crate::pyo3::Python<'_>,
            ) -> $crate::pyo3::Bound<'_, $crate::pyo3::types::PyType> {
                py.get_type::<ReadOnly>()
            }

            fn invalid_compaction(
                py: $crate::pyo3::Python<'_>,
            ) -> $crate::pyo3::Bound<'_, $crate::pyo3::types::PyType> {
                py.get_type::<InvalidCompaction>()
            }

            fn tail_moved(
                py: $crate::pyo3::Python<'_>,
            ) -> $crate::pyo3::Bound<'_, $crate::pyo3::types::PyType> {
                py.get_type::<TailMoved>()
            }
        }
    };
}
