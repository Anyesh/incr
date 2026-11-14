//! [`persist_classes!`](crate::persist_classes): the persistence pyclasses,
//! written once for both packages.

/// Defines a package's persistence classes in the calling module, which
/// must hold nothing else, under the Python module path `$module`, with
/// the pyclass flags `$flag` (such as `unsendable` or `frozen`).
///
/// Defines `PyStore`, `PyBatch`, `PyMemo`, `PyMemoTable` and `PyOverlay`
/// for a runtime that implements
/// [`FullStoreRuntime`](crate::classes::FullStoreRuntime). The package
/// registers the classes itself; Shared leaves `PyOverlay` unregistered.
#[macro_export]
macro_rules! persist_classes {
    (binding = $b:ty, runtime = $rt:ty, module = $module:literal, flags = [$($flag:ident),*]) => {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};

        use $crate::binding::PyBinding;
        use $crate::classes::{key_counts, keyed_of, reconcile_pairs, FullStoreRuntime, KeyCounts, StoreRuntime};
        use $crate::incr_core::Runtime;
        use $crate::keyed::Pending;
        use $crate::memo::{MemoCore, TableHandle};
        use $crate::pyo3::exceptions::{PyRuntimeError, PyValueError};
        use $crate::pyo3::prelude::*;
        use $crate::pyo3::types::PyDict;
        use $crate::incr_python_protocol::PyValue;

        /// Persisted inputs and collections over a backend, and the runtime that
        /// reads them.
        ///
        /// `store.input(key, default, codec)` and `store.collection(key, codec)`
        /// declare a key and return its node or `Collection`; `codec` is any object
        /// with `encode(obj) -> bytes` and `decode(bytes) -> obj`. Two contracts
        /// bind it: `decode(encode(x)) == x`, and `encode` is deterministic (equal
        /// values give equal bytes), since fingerprints hash the bytes. A codec
        /// need not know `Pending`, `LookupFailed`, `Unavailable` or `Incomplete`:
        /// those persist as themselves, and only their `row`, `value`, `key` and
        /// `reason` go through the codec. A value written to a slot with no codec
        /// raises, naming the slot.
        ///
        /// Persisted nodes are written through the store (`store.set`,
        /// `store.insert`, ...); `runtime.set` and a persisted collection's own
        /// `insert`, `delete`, `extend` and `replace` raise `RuntimeError`.
        #[pyclass(name = "Store", module = $module, $($flag),*)]
        pub(crate) struct PyStore {
            core: Mutex<Option<Arc<$crate::store::StoreCore<$b>>>>,
            memo: Mutex<Option<Arc<MemoCore<$b>>>>,
            runtime: Py<$rt>,
        }

        const _: () = assert!(std::mem::align_of::<PyStore>() <= 16);

        /// A reference to the store's memo. Whichever reference is the last
        /// drops the memo, joining its checkpoint thread, through the
        /// binding's `drop_detached`, also when it is a call's own after
        /// `close` took the store's.
        struct MemoRef(Option<Arc<MemoCore<$b>>>);

        impl std::ops::Deref for MemoRef {
            type Target = MemoCore<$b>;

            fn deref(&self) -> &MemoCore<$b> {
                self.0.as_deref().expect("a MemoRef holds its memo until it drops")
            }
        }

        impl Drop for MemoRef {
            fn drop(&mut self) {
                if let Some(last) = self.0.take().and_then(Arc::into_inner) {
                    Python::attach(|py| <$b as PyBinding>::drop_detached(py, last));
                }
            }
        }

        impl Drop for PyStore {
            fn drop(&mut self) {
                Python::attach(|py| self.release(py));
            }
        }

        impl PyStore {
            fn new(core: Arc<$crate::store::StoreCore<$b>>, runtime: Py<$rt>) -> Self {
                PyStore {
                    core: Mutex::new(Some(core)),
                    memo: Mutex::new(Some(Arc::new(MemoCore::default()))),
                    runtime,
                }
            }

            fn core(&self) -> PyResult<Arc<$crate::store::StoreCore<$b>>> {
                self.core
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .clone()
                    .ok_or_else(|| PyRuntimeError::new_err("store is closed"))
            }

            /// Drops the memo, stopping its checkpoint thread, and the core,
            /// through the binding's [`drop_detached`](PyBinding::drop_detached).
            fn release(&self, py: Python<'_>) {
                let memo = self.memo.lock().unwrap_or_else(PoisonError::into_inner).take();
                let core = self.core.lock().unwrap_or_else(PoisonError::into_inner).take();
                drop(MemoRef(memo));
                <$b as PyBinding>::drop_detached(py, core);
            }

            fn memo(&self) -> PyResult<MemoRef> {
                self.memo
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .clone()
                    .map(|memo| MemoRef(Some(memo)))
                    .ok_or_else(|| PyRuntimeError::new_err("store is closed"))
            }

            /// The core of a store whose runtime is usable: a runtime poisoned by
            /// a failed overlay restore refuses every store call, since a
            /// checkpoint would fold the half-restored collections.
            fn core_live(&self, py: Python<'_>) -> PyResult<Arc<$crate::store::StoreCore<$b>>> {
                let core = self.core()?;
                self.runtime.bind(py).borrow().rt()?;
                Ok(core)
            }

            fn core_outside_overlay(&self, py: Python<'_>) -> PyResult<Arc<$crate::store::StoreCore<$b>>> {
                self.runtime.bind(py).borrow().check_store_call()?;
                self.core_live(py)
            }

            fn after_write(&self, core: &$crate::store::StoreCore<$b>, py: Python<'_>) {
                if let Ok(memo) = self.memo() {
                    memo.after_write(core, py);
                }
            }

            fn maker(&self, py: Python<'_>) -> impl Fn(Py<PyAny>) -> $crate::memo::MemoFn<<$b as $crate::binding::PyBinding>::C> {
                let state = self.runtime.bind(py).borrow().state().clone();
                move |func| <$rt as FullStoreRuntime<$b>>::memo_fn(&state, func)
            }
        }

        $crate::persist_classes!(@store_methods $b, $rt, {
            /// Declares the collection `key`, which starts with its logged elements,
            /// and returns it as a `Collection`. Write it with `store.insert`,
            /// `store.delete`, `store.replace` and `store.extend`.
            fn collection(
                &self,
                py: Python<'_>,
                key: &str,
                codec: Option<Py<PyAny>>,
            ) -> PyResult<<$rt as FullStoreRuntime<$b>>::Collection> {
                let core = self.core_outside_overlay(py)?;
                let persisted = core.collection(py, key, codec)?;
                Ok(<$rt as FullStoreRuntime<$b>>::collection(
                    self.runtime.bind(py),
                    persisted.handle().clone(),
                ))
            }

            /// Appends one occurrence of `row` to a persisted collection.
            fn insert(
                &self,
                py: Python<'_>,
                collection: PyRef<'_, <$rt as FullStoreRuntime<$b>>::Collection>,
                row: Py<PyAny>,
            ) -> PyResult<()> {
                let core = self.core_outside_overlay(py)?;
                let persisted = core.collection_of(<$rt as FullStoreRuntime<$b>>::version_slot(&collection))?;
                core.insert(py, persisted, row)?;
                self.after_write(&core, py);
                Ok(())
            }

            /// Removes one occurrence of `row` from a persisted collection. Removing
            /// an absent row is logged and changes nothing.
            fn delete(
                &self,
                py: Python<'_>,
                collection: PyRef<'_, <$rt as FullStoreRuntime<$b>>::Collection>,
                row: Py<PyAny>,
            ) -> PyResult<()> {
                let core = self.core_outside_overlay(py)?;
                let persisted = core.collection_of(<$rt as FullStoreRuntime<$b>>::version_slot(&collection))?;
                core.delete(py, persisted, row)?;
                self.after_write(&core, py);
                Ok(())
            }

            /// Removes one occurrence of `old`, if present, and inserts `new`, as
            /// one logged step.
            fn replace(
                &self,
                py: Python<'_>,
                collection: PyRef<'_, <$rt as FullStoreRuntime<$b>>::Collection>,
                old: Py<PyAny>,
                new: Py<PyAny>,
            ) -> PyResult<()> {
                let core = self.core_outside_overlay(py)?;
                let persisted = core.collection_of(<$rt as FullStoreRuntime<$b>>::version_slot(&collection))?;
                core.replace(py, persisted, old, new)?;
                self.after_write(&core, py);
                Ok(())
            }

            /// Appends every row of `rows` in one logged step.
            fn extend(
                &self,
                py: Python<'_>,
                collection: PyRef<'_, <$rt as FullStoreRuntime<$b>>::Collection>,
                rows: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let core = self.core_outside_overlay(py)?;
                let persisted = core.collection_of(<$rt as FullStoreRuntime<$b>>::version_slot(&collection))?;
                core.extend(py, persisted, rows)?;
                self.after_write(&core, py);
                Ok(())
            }

            /// Declares the keyed collection `key`: a `KeyedCollection` over a
            /// persisted collection of `(key, row)` pairs, holding the pairs the
            /// log restores. `key_codec` and `codec` encode the
            /// keys and the rows, under the contracts of `store.input`. Write it
            /// with `store.upsert`, `store.remove`, `store.reconcile` or a
            /// `store.batch`; its own writes and writes through its `collection`
            /// raise. `label` names the collection's group in `graph_summary`,
            /// labelling its pair and row nodes.
            ///
            /// Rows need not be hashable, but `collection` and operators over it
            /// hash them. A row `==` to the held one is unchanged and logs
            /// nothing, so `{"a": True}` over `{"a": 1}` changes nothing although
            /// their encodings differ. A logged pair is matched by the fingerprint
            /// of its bytes, so a row not `==` to its decoded copy (one holding
            /// NaN or an object compared by identity) still replaces, restores
            /// and catches up as one row, unless its codec re-encodes it to
            /// different bytes (a set pickled under another hash seed, say): such
            /// a row can, after a replace or remove and a reopen, leave its old
            /// row held too, and a read may return the old row. Use a codec
            /// that encodes such rows canonically, or rows that equal their
            /// decoded copy.
            #[pyo3(signature = (key, key_codec, codec, label=None))]
            fn keyed_collection(
                &self,
                py: Python<'_>,
                key: &str,
                key_codec: Py<PyAny>,
                codec: Py<PyAny>,
                label: Option<String>,
            ) -> PyResult<<$rt as FullStoreRuntime<$b>>::Keyed> {
                let core = self.core_outside_overlay(py)?;
                let persisted = core.keyed_collection(py, key, key_codec, codec, label.as_deref())?;
                <$rt as FullStoreRuntime<$b>>::keyed(self.runtime.bind(py), persisted, label)
            }

            /// Holds `row` for `key` in a persisted keyed collection, as
            /// `KeyedCollection.upsert` does, logging the change.
            fn upsert(
                &self,
                py: Python<'_>,
                kc: PyRef<'_, <$rt as FullStoreRuntime<$b>>::Keyed>,
                key: Py<PyAny>,
                row: Py<PyAny>,
            ) -> PyResult<()> {
                let core = self.core_outside_overlay(py)?;
                let keyed = keyed_of::<$b, $rt>(py, &core, &kc)?;
                let mut pending = Pending::default();
                pending.upsert(py, &core, &keyed, key, row)?;
                core.write_pending(py, pending)?;
                self.after_write(&core, py);
                Ok(())
            }

            /// Removes `key`'s pair from a persisted keyed collection. Raises
            /// `KeyError` if no row is held for it.
            fn remove(
                &self,
                py: Python<'_>,
                kc: PyRef<'_, <$rt as FullStoreRuntime<$b>>::Keyed>,
                key: Py<PyAny>,
            ) -> PyResult<()> {
                let core = self.core_outside_overlay(py)?;
                let keyed = keyed_of::<$b, $rt>(py, &core, &kc)?;
                let mut pending = Pending::default();
                pending.remove(py, &core, &keyed, key)?;
                core.write_pending(py, pending)?;
                self.after_write(&core, py);
                Ok(())
            }

            /// Makes a persisted keyed collection hold exactly `rows`, as
            /// `KeyedCollection.reconcile` does, in one logged step. Returns how
            /// many keys were `(inserted, replaced, removed)`.
            fn reconcile(
                &self,
                py: Python<'_>,
                kc: PyRef<'_, <$rt as FullStoreRuntime<$b>>::Keyed>,
                rows: &Bound<'_, PyAny>,
            ) -> PyResult<KeyCounts> {
                let core = self.core_outside_overlay(py)?;
                let keyed = keyed_of::<$b, $rt>(py, &core, &kc)?;
                let mut pending = Pending::default();
                pending.reconcile(py, &core, &keyed, reconcile_pairs(rows)?)?;
                let counts = key_counts(pending.counts(&keyed));
                core.write_pending(py, pending)?;
                self.after_write(&core, py);
                Ok(counts)
            }

            /// Starts a batch of `b.set`, `b.insert`, `b.delete`, `b.replace`,
            /// `b.upsert`, `b.remove` and `b.reconcile`, appended to the log as
            /// one run, so a restore sees all of them or none.
            ///
            /// `apply="on_exit"` (the default) collects the writes and appends and
            /// applies them on leaving the block; a block that raises writes
            /// nothing, and reads inside it see the values from before it.
            ///
            /// `apply="on_write"` applies each write at once, so reads inside the
            /// block see it (on a Shared runtime other threads read it too, before
            /// the run lands), and appends the run on leaving the block with the
            /// backend's `append_at`, right after the position the store had
            /// applied; a backend without `append_at` refuses it with
            /// `PersistError`. Entering catches up with the log and runs a due
            /// `auto_checkpoint_records` checkpoint, so none falls between the
            /// batch's records. Until the run lands every other store call that
            /// writes, declares, refreshes or checkpoints, a memo checkpoint and a
            /// second batch raise `PersistError`. If the append raises, the block
            /// raises it and the batch keeps its run: `b.retry()` appends it again
            /// and lands it at most once. A block that raises, or a batch dropped
            /// before its run lands, poisons the store, since the runtime holds
            /// writes the log never receives: every later store call raises
            /// `PersistError` and the way out is to close the store and open the
            /// log again. A backend that fences this writer keeps raising, so its
            /// way out is the same.
            #[pyo3(signature = (apply="on_exit"))]
            fn batch(slf: &Bound<'_, Self>, apply: &str) -> PyResult<PyBatch> {
                let py = slf.py();
                let apply = $crate::batch::Apply::parse(apply)?;
                slf.borrow().core_outside_overlay(py)?;
                Ok(PyBatch {
                    store: slf.clone().unbind(),
                    batch: Mutex::new($crate::batch::Batch::new(apply)),
                    opener: std::thread::current().id(),
                })
            }

            /// Declares the memo table `name`, whose entries `func(ctx, key)`
            /// computes, and returns it. `ctx` is the `RuntimeRef` queries receive,
            /// and `ctx.memo(table, key)` reads another entry as a dependency.
            /// `key_codec` and `codec` encode the keys and the values under the
            /// contracts of `store.input`, and `func` must be a pure function of
            /// its key and what it reads through `ctx`; on a Shared runtime it runs
            /// on whichever thread computes the entry, so it must be thread-safe,
            /// as the codecs must. Bump `version` when it
            /// changes meaning: entries are kept only while the name and version
            /// match. Declare every table before `store.open_memo`. With `label`,
            /// each entry's node is labelled `<label>/<str(key)>` when first
            /// reached through `memo.get_or_insert` or `ctx.memo`, as a keyed
            /// collection labels its per-key inputs, so `graph_summary` and
            /// `compute_stats` group the table's entries under `label`; each such
            /// call then also runs `str(key)`.
            #[pyo3(signature = (name, version, key_codec, codec, func, *, label=None))]
            fn memo_table(
                slf: &Bound<'_, Self>,
                name: &str,
                version: u32,
                key_codec: Py<PyAny>,
                codec: Py<PyAny>,
                func: Py<PyAny>,
                label: Option<String>,
            ) -> PyResult<PyMemoTable> {
                let py = slf.py();
                let store = slf.borrow();
                let core = store.core_outside_overlay(py)?;
                let handle = store.memo()?.declare_table(
                    &core,
                    py,
                    name,
                    version,
                    key_codec,
                    codec,
                    func,
                    label.as_deref(),
                    store.maker(py),
                )?;
                Ok(PyMemoTable {
                    store: slf.clone().unbind(),
                    name: name.to_owned(),
                    handle,
                })
            }

            /// Opens the memo over `backend`, an object that implements the memo
            /// backend protocol (see `incr.persist.MemoryMemo`), and loads its
            /// current generation. Entries written under another `build_id`, or by
            /// a table of another name, version or type, are ignored.
            ///
            /// A checkpoint evicts the oldest entries down to `max_bytes` of
            /// encoded entries (512 MiB by default). With
            /// `checkpoint_after_entries=n` a checkpoint is due once `n` entries
            /// computed since the last one are unsaved: on a Local runtime each
            /// logged store write and batch commit runs a due checkpoint after it
            /// lands, and a read never does (call `memo.checkpoint_if_due()` after
            /// a read-only rebuild); a Shared memo's checkpoint thread runs it. A
            /// triggered checkpoint that fails does not raise from the write that
            /// ran it, which has landed: `memo.stats()` counts it under
            /// `auto_checkpoint_errors` with its text under
            /// `last_auto_checkpoint_error`, and it stays due, retried by the first
            /// write after a new entry computes. `Busy` leaves it due for the
            /// next write.
            #[pyo3(signature = (backend, *, max_bytes=None, checkpoint_after_entries=None))]
            fn open_memo(
                slf: &Bound<'_, Self>,
                backend: Py<PyAny>,
                max_bytes: Option<u64>,
                checkpoint_after_entries: Option<u64>,
            ) -> PyResult<PyMemo> {
                let py = slf.py();
                let options = $crate::memo::OpenOptions::new(max_bytes, checkpoint_after_entries)?;
                let store = slf.borrow();
                let core = store.core_outside_overlay(py)?;
                store.memo()?.open(&core, py, backend, options, store.maker(py))?;
                Ok(PyMemo {
                    store: slf.clone().unbind(),
                })
            }

            /// Applies the records other writers appended since this store last
            /// read the log, across a compaction too, and returns the position it
            /// has applied through.
            fn refresh(&self, py: Python<'_>) -> PyResult<u64> {
                self.core_outside_overlay(py)?.refresh(py)
            }

            /// `position` (the last log position applied), `snapshot_position`
            /// (the snapshot this store last wrote or crossed, 0 if none) and
            /// `records_since_checkpoint`, read without touching the backend.
            fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
                let stats = self.core()?.stats(py)?;
                let dict = PyDict::new(py);
                dict.set_item("position", stats.position.0)?;
                dict.set_item("snapshot_position", stats.snapshot_position.0)?;
                dict.set_item("records_since_checkpoint", stats.records_since_checkpoint)?;
                Ok(dict)
            }

            /// Folds the log into a snapshot, compacting it, and returns the
            /// position the snapshot covers.
            fn checkpoint(&self, py: Python<'_>) -> PyResult<u64> {
                self.core_outside_overlay(py)?.checkpoint(py)
            }

            /// Closes the store: every later store call raises. A Local store
            /// closes its runtime too; a Shared store's runtime stays open with
            /// the values it holds, since other threads may be reading it.
            fn close(&self, py: Python<'_>) -> PyResult<()> {
                <$rt as FullStoreRuntime<$b>>::close_store_runtime(self.runtime.bind(py))?;
                self.release(py);
                Ok(())
            }
        });

        /// The writes of a `with store.batch() as b:` block; see `Store.batch`.
        ///
        /// In an `on_exit` batch keyed writes compose per key against the rows
        /// each key held when first staged, so `b.upsert` twice on one key, or
        /// `b.upsert` then `b.remove`, leave exactly the final pair, and
        /// `b.remove` raises `KeyError` for a key neither held nor staged; the
        /// collection takes the staged keys only once the append has landed. In
        /// an `on_write` batch each write applies at once, and `b.remove` raises
        /// `KeyError` for a key holding no row at that point.
        ///
        /// A batch is used only on the thread that called `store.batch()`: a
        /// call from another thread raises `RuntimeError`.
        #[pyclass(name = "Batch", module = $module, $($flag),*)]
        pub(crate) struct PyBatch {
            store: Py<PyStore>,
            batch: Mutex<$crate::batch::Batch<$b>>,
            opener: std::thread::ThreadId,
        }

        impl Drop for PyBatch {
            fn drop(&mut self) {
                let batch = std::mem::replace(
                    self.batch.get_mut().unwrap_or_else(PoisonError::into_inner),
                    $crate::batch::Batch::new($crate::batch::Apply::OnExit),
                );
                Python::attach(|py| <$b as PyBinding>::drop_detached(py, batch));
            }
        }

        impl PyBatch {
            fn batch(&self) -> PyResult<MutexGuard<'_, $crate::batch::Batch<$b>>> {
                if std::thread::current().id() != self.opener {
                    return Err(PyRuntimeError::new_err(
                        "a batch is used only on the thread that opened it",
                    ));
                }
                match self.batch.try_lock() {
                    Ok(batch) => Ok(batch),
                    Err(TryLockError::Poisoned(poisoned)) => Ok(poisoned.into_inner()),
                    Err(TryLockError::WouldBlock) => Err(PyRuntimeError::new_err(
                        "the batch is in use by another call",
                    )),
                }
            }
        }

        fn batch_core(store: &Py<PyStore>, py: Python<'_>) -> PyResult<Arc<$crate::store::StoreCore<$b>>> {
            store.bind(py).borrow().core_outside_overlay(py)
        }

        #[pymethods]
        impl PyBatch {
            fn __enter__(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
                let py = slf.py();
                let core = batch_core(&slf.borrow().store, py)?;
                slf.borrow().batch()?.enter(py, &core)?;
                Ok(slf)
            }

            #[pyo3(signature = (exc_type, _exc, _traceback))]
            fn __exit__(
                &self,
                py: Python<'_>,
                exc_type: Option<&Bound<'_, PyAny>>,
                _exc: Option<&Bound<'_, PyAny>>,
                _traceback: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<bool> {
                if exc_type.is_some() {
                    let core = self.store.bind(py).borrow().core().ok();
                    self.batch()?.abandon(py, core.as_deref())?;
                } else {
                    let core = batch_core(&self.store, py)?;
                    self.batch()?.exit(py, &core)?;
                    self.store.bind(py).borrow().after_write(&core, py);
                }
                Ok(false)
            }

            /// Appends the run of an `on_write` batch whose append raised, with
            /// the same `append_at`, so it lands at most once, and returns the
            /// position of its last record. Raises what the append raises, keeping
            /// the run for another retry; `PersistError` once the run has landed;
            /// `RuntimeError` for an `on_exit` batch, inside the block, or after a
            /// block that raised.
            fn retry(&self, py: Python<'_>) -> PyResult<u64> {
                let core = batch_core(&self.store, py)?;
                let position = self.batch()?.retry(py, &core)?;
                self.store.bind(py).borrow().after_write(&core, py);
                Ok(position)
            }

            fn set(&self, py: Python<'_>, node: <$rt as StoreRuntime<$b>>::Node, value: Py<PyAny>) -> PyResult<()> {
                let core = batch_core(&self.store, py)?;
                self.batch()?.set(py, &core, <$rt as StoreRuntime<$b>>::node_inner(&node), value)
            }

            fn insert(
                &self,
                py: Python<'_>,
                collection: PyRef<'_, <$rt as FullStoreRuntime<$b>>::Collection>,
                row: Py<PyAny>,
            ) -> PyResult<()> {
                let core = batch_core(&self.store, py)?;
                self.batch()?.insert(py, &core, <$rt as FullStoreRuntime<$b>>::version_slot(&collection), row)
            }

            fn delete(
                &self,
                py: Python<'_>,
                collection: PyRef<'_, <$rt as FullStoreRuntime<$b>>::Collection>,
                row: Py<PyAny>,
            ) -> PyResult<()> {
                let core = batch_core(&self.store, py)?;
                self.batch()?.delete(py, &core, <$rt as FullStoreRuntime<$b>>::version_slot(&collection), row)
            }

            fn replace(
                &self,
                py: Python<'_>,
                collection: PyRef<'_, <$rt as FullStoreRuntime<$b>>::Collection>,
                old: Py<PyAny>,
                new: Py<PyAny>,
            ) -> PyResult<()> {
                let core = batch_core(&self.store, py)?;
                self.batch()?.replace(py, &core, <$rt as FullStoreRuntime<$b>>::version_slot(&collection), old, new)
            }

            fn upsert(
                &self,
                py: Python<'_>,
                kc: PyRef<'_, <$rt as FullStoreRuntime<$b>>::Keyed>,
                key: Py<PyAny>,
                row: Py<PyAny>,
            ) -> PyResult<()> {
                let core = batch_core(&self.store, py)?;
                let keyed = keyed_of::<$b, $rt>(py, &core, &kc)?;
                self.batch()?.upsert(py, &core, &keyed, key, row)
            }

            fn remove(
                &self,
                py: Python<'_>,
                kc: PyRef<'_, <$rt as FullStoreRuntime<$b>>::Keyed>,
                key: Py<PyAny>,
            ) -> PyResult<()> {
                let core = batch_core(&self.store, py)?;
                let keyed = keyed_of::<$b, $rt>(py, &core, &kc)?;
                self.batch()?.remove(py, &core, &keyed, key)
            }

            fn reconcile(
                &self,
                py: Python<'_>,
                kc: PyRef<'_, <$rt as FullStoreRuntime<$b>>::Keyed>,
                rows: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let core = batch_core(&self.store, py)?;
                let keyed = keyed_of::<$b, $rt>(py, &core, &kc)?;
                self.batch()?.reconcile(py, &core, &keyed, reconcile_pairs(rows)?)
            }
        }

        /// A memo table of a store: pass it to `memo.get_or_insert` and
        /// `ctx.memo`.
        #[pyclass(name = "MemoTable", module = $module, $($flag),*)]
        pub(crate) struct PyMemoTable {
            store: Py<PyStore>,
            name: String,
            handle: TableHandle,
        }

        const _: () = assert!(std::mem::align_of::<PyMemoTable>() <= 16);

        impl PyMemoTable {
            /// `key`'s value, restored from the memo or computed, read as a
            /// dependency of whatever query is running on `rt`.
            pub(crate) fn value_of(
                &self,
                py: Python<'_>,
                rt: &Runtime<<$b as $crate::binding::PyBinding>::C>,
                key: PyValue,
            ) -> PyResult<PyValue> {
                let store = self.store.bind(py).borrow();
                let core = store.core()?;
                store.memo()?.value_of(&core, py, rt, &self.name, &self.handle, key)
            }
        }

        #[pymethods]
        impl PyMemoTable {
            #[getter]
            fn name(&self) -> &str {
                &self.name
            }

            fn __repr__(&self) -> String {
                format!("MemoTable({:?})", self.name)
            }
        }

        /// An open memo: `memo.get_or_insert(table, key)` gives an entry's node, and
        /// `memo.checkpoint()` writes the live entries to the backend.
        #[pyclass(name = "Memo", module = $module, $($flag),*)]
        pub(crate) struct PyMemo {
            store: Py<PyStore>,
        }

        const _: () = assert!(std::mem::align_of::<PyMemo>() <= 16);

        #[pymethods]
        impl PyMemo {
            /// The node of `table`'s entry for `key`: the live one, or one restored
            /// from the memo, or a new one. Read it with `runtime.get`. Inside an
            /// overlay it computes fresh and records nothing, so no candidate value
            /// is restored from the memo or reaches it.
            fn get_or_insert(
                &self,
                py: Python<'_>,
                table: PyRef<'_, PyMemoTable>,
                key: Py<PyAny>,
            ) -> PyResult<<$rt as StoreRuntime<$b>>::Node> {
                let core = self.store.bind(py).borrow().core_live(py)?;
                if !table.store.is(&self.store) {
                    return Err(PyValueError::new_err(format!(
                        "memo table '{}' belongs to another store",
                        table.name
                    )));
                }
                let store = self.store.bind(py).borrow();
                let inner = store.memo()?.get_or_insert(&core, py, &table.handle, key)?;
                Ok(<$rt as StoreRuntime<$b>>::node(inner))
            }

            /// Writes every live entry whose dependencies are persisted to the
            /// backend as the next generation, and returns how many it `written`,
            /// `merged` from the previous generation, `evicted` and so on. Raises
            /// `Busy` when another handle is checkpointing. Raises `ValueError`
            /// naming the table when an entry's key or value did not round-trip
            /// through its codec: the generation was still written, without that
            /// entry, which is recomputed on the next run. Raises `PersistError`,
            /// writing nothing, while an `apply="on_write"` batch or an overlay is
            /// open. That refusal is best effort: a batch or scope another thread
            /// opens after the check is seen by the snapshot, and the content
            /// fingerprints stored with each entry keep a restore correct.
            fn checkpoint(&self, py: Python<'_>) -> PyResult<HashMap<&'static str, usize>> {
                let store = self.store.bind(py).borrow();
                let core = store.core_outside_overlay(py)?;
                store.memo()?.checkpoint(&core, py)
            }

            /// Runs `checkpoint()` when `checkpoint_after_entries` entries are
            /// unsaved and returns its counts; `None` when none is due, and
            /// always when `open_memo` set no trigger. For after a read-only
            /// rebuild, since reads never checkpoint.
            fn checkpoint_if_due(&self, py: Python<'_>) -> PyResult<Option<HashMap<&'static str, usize>>> {
                let store = self.store.bind(py).borrow();
                let core = store.core_outside_overlay(py)?;
                store.memo()?.checkpoint_if_due(&core, py)
            }

            /// `entries` and `bytes` of the current generation, `unsaved_entries`
            /// computed since the last checkpoint and not saved,
            /// `unsaveable` entries the last checkpoint could not save,
            /// `evicted_last_checkpoint`, `auto_checkpoint_errors` (triggered
            /// checkpoints that failed) and `last_auto_checkpoint_error` (the
            /// latest such failure a write ran, as text, or `None`), read without
            /// touching the backend.
            fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
                let store = self.store.bind(py).borrow();
                let core = store.core()?;
                let stats = store.memo()?.stats(&core, py)?;
                let dict = PyDict::new(py);
                for (name, count) in stats.counts() {
                    dict.set_item(name, count)?;
                }
                dict.set_item("last_auto_checkpoint_error", stats.last_auto_checkpoint_error)?;
                Ok(dict)
            }
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
        /// emptied or created. The version input a `ctx.get_key` read creates
        /// for a key inside the block stays after exit, so a query that first
        /// read its key there still reruns when that key changes later.
        #[pyclass(name = "Overlay", $($flag),*)]
        pub(crate) struct PyOverlay {
            rt: Py<$rt>,
        }

        impl PyOverlay {
            pub(crate) fn new(rt: Py<$rt>) -> Self {
                PyOverlay { rt }
            }
        }

        #[pymethods]
        impl PyOverlay {
            fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
                $crate::classes::enter_overlay::<$b, $rt>(slf.py(), &slf.rt.bind(slf.py()).borrow())?;
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
                $crate::classes::exit_overlay::<$b, $rt>(py, self.rt.bind(py))?;
                Ok(false)
            }

            fn set(&self, py: Python<'_>, node: <$rt as StoreRuntime<$b>>::Node, value: Py<PyAny>) -> PyResult<()> {
                FullStoreRuntime::<$b>::set(&*self.rt.bind(py).borrow(), node, value)
            }

            fn get(&self, py: Python<'_>, node: <$rt as StoreRuntime<$b>>::Node) -> PyResult<Py<PyAny>> {
                FullStoreRuntime::<$b>::get(&*self.rt.bind(py).borrow(), py, node)
            }

            fn get_with_status(
                &self,
                py: Python<'_>,
                node: <$rt as StoreRuntime<$b>>::Node,
            ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
                FullStoreRuntime::<$b>::get_with_status(&*self.rt.bind(py).borrow(), py, node)
            }
        }
    };

    (@store_methods $b:ty, $rt:ty, { $($extra:tt)* }) => {
        #[pymethods]
        impl PyStore {
            /// Opens a store over `backend`, replaying its log. `build_id`, an int
            /// or a str, names the code that computes memoized values; incr hashes
            /// a str to an id. `durability` is `"sync"` (a write is durable when it
            /// returns) or `"buffered"`. With `auto_checkpoint_records=n`, a
            /// write first checkpoints once `n` records were applied since the last
            /// snapshot; a failed checkpoint fails that write and leaves the log
            /// unchanged.
            #[staticmethod]
            #[pyo3(signature = (backend, *, build_id, durability="sync", auto_checkpoint_records=None))]
            fn open(
                py: Python<'_>,
                backend: Py<PyAny>,
                build_id: &Bound<'_, PyAny>,
                durability: &str,
                auto_checkpoint_records: Option<u64>,
            ) -> PyResult<PyStore> {
                #[allow(
                    clippy::arc_with_non_send_sync,
                    reason = "the store core is reached through `Arc` in both packages; a Local one stays on its runtime's thread"
                )]
                let core = Arc::new($crate::store::StoreCore::<$b>::open(
                    py,
                    backend,
                    build_id,
                    durability,
                    auto_checkpoint_records,
                )?);
                let runtime = Py::new(py, <$rt as StoreRuntime<$b>>::of_store(Arc::clone(&core)))?;
                Ok(PyStore::new(core, runtime))
            }

            /// The runtime that reads the store's nodes. Close the store, not the
            /// runtime.
            #[getter]
            fn runtime(&self, py: Python<'_>) -> Py<$rt> {
                self.runtime.clone_ref(py)
            }

            /// Declares the input `key`, which starts at its logged value or else at
            /// `default`, and returns its node. Write it with `store.set`.
            fn input(
                &self,
                py: Python<'_>,
                key: &str,
                default: Py<PyAny>,
                codec: Option<Py<PyAny>>,
            ) -> PyResult<<$rt as StoreRuntime<$b>>::Node> {
                let core = self.core_outside_overlay(py)?;
                let inner = core.input(py, key, default, codec)?;
                Ok(<$rt as StoreRuntime<$b>>::node(inner))
            }

            /// Sets a persisted input: the value is logged, then applied.
            fn set(
                &self,
                py: Python<'_>,
                node: <$rt as StoreRuntime<$b>>::Node,
                value: Py<PyAny>,
            ) -> PyResult<()> {
                let core = self.core_outside_overlay(py)?;
                core.set(py, <$rt as StoreRuntime<$b>>::node_inner(&node), value)?;
                self.after_write(&core, py);
                Ok(())
            }

            $($extra)*
        }
    };
}
