//! `KeyedCollection` on a Shared runtime: rows by key, plain or persisted
//! by a store.

use incr_python_persist::classes::{
    key_counts, keyed_rows_refusal, reconcile_pairs, KeyCounts, KeyedSource, PlainKeyed,
};
use incr_python_persist::incr_core::StateStats;
use incr_python_persist::incr_persist::Shared;
use incr_python_persist::store::Keyed;
use incr_python_protocol::{PyKey, PyPair, PyValue};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Weak;

use ::incr_concurrent::{IncrCollection, Runtime};

use crate::persist::SharedBinding;
use crate::{engine_call, PyCollection, PyRuntime, RtRef};

enum KeyedKind {
    Plain(PlainKeyed<Shared>),
    /// A keyed collection a store persists, which only the store writes.
    Persisted(Keyed<SharedBinding>),
}

/// Rows by key: a collection of `(key, row)` pairs holding at most one row
/// per key, with keys matched as a dict matches them. `collection` is the
/// rows as a `Collection`, for the operators.
///
/// Rows need not be hashable: a pair hashes its key alone, and a row is
/// unchanged when it is `==` to the held one, so an upsert of `{"a": True}`
/// over `{"a": 1}` changes nothing. `collection` and operators over it
/// still hash rows, so a collection of unhashable rows raises `TypeError`
/// there. Writes through `collection` raise `RuntimeError`; write with
/// `upsert`, `remove` and `reconcile`.
///
/// Key and row methods run as value code, on
/// whichever thread writes or reads, so they must be thread-safe; one that
/// calls into incr during an operation on the collection raises
/// `RuntimeError`, which the operation raises, changing nothing.
///
/// A key read with `ctx.get_key` has a version input node, which a Shared
/// collection keeps for its whole life, also after the key is removed and
/// no query reads it. Memory therefore grows with the number of distinct
/// keys ever read, not only the keys held; `incr.Runtime` deletes a
/// removed key's node once nothing reads it.
///
/// `label` labels the pair and row collections' version nodes, and the
/// version input each `ctx.get_key` creates as `<label>/<str(key)>`.
#[pyclass(name = "KeyedCollection", module = "incr_concurrent", frozen)]
pub(crate) struct PyKeyedCollection {
    kind: KeyedKind,
    rows: IncrCollection<PyValue>,
    pub(crate) rt: Py<PyRuntime>,
    label: Option<String>,
}

const _: () = assert!(std::mem::align_of::<PyKeyedCollection>() <= 16);

const PERSISTED: &str = "keyed collection is persisted: write it with store.upsert, store.remove, store.reconcile or a store.batch";

impl PyKeyedCollection {
    fn wrap(
        py: Python<'_>,
        runtime: Bound<'_, PyRuntime>,
        kind: KeyedKind,
        pairs: IncrCollection<PyPair>,
        rows: IncrCollection<PyValue>,
        stats: Weak<dyn StateStats + Send + Sync>,
        label: Option<String>,
    ) -> PyResult<Self> {
        if let Some(label) = &label {
            let runtime = runtime.get();
            let rt = runtime.inner.clone();
            let slots = [pairs.version_node().slot(), rows.version_node().slot()];
            let label = label.clone();
            engine_call(py, &rt, || {
                for slot in slots {
                    rt.set_label(slot, label.clone());
                }
            })?;
            runtime
                .keyed_labels
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((stats, label, pairs.version_node().erase()));
        }
        Ok(PyKeyedCollection {
            kind,
            rows,
            rt: runtime.unbind(),
            label,
        })
    }

    /// A keyed collection over one a store persists.
    pub(crate) fn persisted(
        runtime: Bound<'_, PyRuntime>,
        persisted: Keyed<SharedBinding>,
        label: Option<String>,
    ) -> PyResult<Self> {
        let py = runtime.py();
        let pairs = persisted.pairs();
        let rows = persisted.rows().clone();
        let stats = persisted.index_weak();
        Self::wrap(
            py,
            runtime,
            KeyedKind::Persisted(persisted),
            pairs,
            rows,
            stats,
            label,
        )
    }

    /// The handle the store persisting this collection knows it by.
    pub(crate) fn persisted_handle(&self) -> Option<&Keyed<SharedBinding>> {
        match &self.kind {
            KeyedKind::Plain(_) => None,
            KeyedKind::Persisted(persisted) => Some(persisted),
        }
    }

    /// What a per-key read on `rt` reaches, and the label of the version
    /// inputs it creates. Raises `ValueError` when `rt` is not this
    /// collection's runtime.
    pub(crate) fn key_source(
        &self,
        rt: &Runtime,
    ) -> PyResult<(KeyedSource<'_, SharedBinding>, Option<&str>)> {
        if !std::ptr::eq(&*self.rt.get().inner, rt) {
            return Err(PyValueError::new_err(
                "keyed collection belongs to another runtime",
            ));
        }
        let source = match &self.kind {
            KeyedKind::Plain(keyed) => KeyedSource::Plain(keyed),
            KeyedKind::Persisted(persisted) => KeyedSource::Persisted(persisted),
        };
        Ok((source, self.label.as_deref()))
    }

    /// The slot of `key`'s version input, `None` when no read created one.
    pub(crate) fn key_version_slot(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Option<u32>> {
        let rt = self.runtime();
        let (source, _) = self.key_source(rt)?;
        let key = PyKey(key);
        engine_call(py, rt, || {
            source.version_input(&key).map(|handle| handle.slot())
        })
    }

    fn runtime(&self) -> &RtRef {
        &self.rt.get().inner
    }

    fn plain(&self) -> PyResult<&PlainKeyed<Shared>> {
        match &self.kind {
            KeyedKind::Plain(keyed) => Ok(keyed),
            KeyedKind::Persisted(_) => Err(PyRuntimeError::new_err(PERSISTED)),
        }
    }

    /// Runs `f`, a read of the core index, at an engine boundary.
    fn read_index<R: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(IndexOf<'_>) -> R + Send,
    ) -> PyResult<R> {
        let rt = self.runtime();
        let index = match &self.kind {
            KeyedKind::Plain(keyed) => IndexOf::Plain(keyed),
            KeyedKind::Persisted(persisted) => IndexOf::Persisted(persisted),
        };
        engine_call(py, rt, || f(index))
    }
}

/// The index a read without a dependency consults.
enum IndexOf<'a> {
    Plain(&'a PlainKeyed<Shared>),
    Persisted(&'a Keyed<SharedBinding>),
}

#[pymethods]
impl PyKeyedCollection {
    #[new]
    #[pyo3(signature = (runtime, label=None))]
    fn new(runtime: Bound<'_, PyRuntime>, label: Option<String>) -> PyResult<Self> {
        let py = runtime.py();
        let rt = runtime.get().inner.clone();
        let keyed = engine_call(py, &rt, || PlainKeyed::<Shared>::new(&rt))?;
        let pairs = keyed.pairs();
        let rows = keyed.rows().clone();
        let stats = keyed.index_weak();
        Self::wrap(
            py,
            runtime,
            KeyedKind::Plain(keyed),
            pairs,
            rows,
            stats,
            label,
        )
    }

    /// Hold `row` for `key`: nothing happens if the held row is `==` to
    /// it, otherwise the pair is replaced or inserted.
    fn upsert(&self, py: Python<'_>, key: Py<PyAny>, row: Py<PyAny>) -> PyResult<()> {
        let keyed = self.plain()?;
        let rt = self.runtime();
        engine_call(py, rt, || {
            drop(keyed.upsert(rt, PyKey(key), PyValue(row)));
        })
    }

    /// Remove `key`'s pair. Raises `KeyError` if no row is held for it.
    fn remove(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<()> {
        let keyed = self.plain()?;
        let rt = self.runtime();
        let target = PyKey(key.clone_ref(py));
        let removed = engine_call(py, rt, || keyed.remove(rt, &target).counts.removed)?;
        if removed == 0 {
            return Err(PyKeyError::new_err(key));
        }
        Ok(())
    }

    /// Make the held rows equal `rows`, a mapping or an iterable of
    /// `(key, row)` pairs (a later pair for a key wins), in one atomic
    /// change. Returns how many keys were `(inserted, replaced, removed)`.
    fn reconcile(&self, py: Python<'_>, rows: &Bound<'_, PyAny>) -> PyResult<KeyCounts> {
        let keyed = self.plain()?;
        let rt = self.runtime();
        let pairs = reconcile_pairs(rows)?;
        let counts = engine_call(py, rt, || keyed.reconcile(rt, pairs).counts)?;
        Ok(key_counts(counts))
    }

    /// The row held for `key`, or `None`. Registers no dependency: inside a
    /// query, read a key with `ctx.get_key(kc, key)` so the query reruns
    /// when that key's row changes.
    fn get(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Option<Py<PyAny>>> {
        let key = PyKey(key);
        let row = self.read_index(py, |index| match index {
            IndexOf::Plain(keyed) => keyed.row(&key),
            IndexOf::Persisted(persisted) => persisted.row(&key),
        })?;
        Ok(row.map(|row| row.0))
    }

    fn __contains__(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<bool> {
        let key = PyKey(key);
        self.read_index(py, |index| match index {
            IndexOf::Plain(keyed) => keyed.contains(&key),
            IndexOf::Persisted(persisted) => persisted.contains(&key),
        })
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        self.read_index(py, |index| match index {
            IndexOf::Plain(keyed) => keyed.len(),
            IndexOf::Persisted(persisted) => persisted.len(),
        })
    }

    /// A new list of the held keys, in no particular order. Registers no
    /// dependency, like `get`.
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys = self.read_index(py, |index| match index {
            IndexOf::Plain(keyed) => keyed.keys(),
            IndexOf::Persisted(persisted) => persisted.keys(),
        })?;
        PyList::new(py, keys.into_iter().map(|key| key.0))
    }

    /// A new list of the held `(key, row)` tuples, in no particular order.
    /// Registers no dependency, like `get`.
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items = self.read_index(py, |index| match index {
            IndexOf::Plain(keyed) => keyed.items(),
            IndexOf::Persisted(persisted) => persisted.items(),
        })?;
        PyList::new(py, items.into_iter().map(|pair| pair.0))
    }

    #[getter]
    fn collection(&self) -> PyCollection {
        let runtime = self.rt.get();
        let how = match &self.kind {
            KeyedKind::Plain(_) => "kc.upsert, kc.remove or kc.reconcile",
            KeyedKind::Persisted(_) => {
                "store.upsert, store.remove, store.reconcile or a store.batch"
            }
        };
        PyCollection {
            inner: self.rows.clone(),
            rt: runtime.inner.clone(),
            bridges: runtime.bridges.clone(),
            keyed_rows: Some(keyed_rows_refusal(self.label.as_deref(), how)),
        }
    }
}
