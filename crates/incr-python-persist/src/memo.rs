//! Persistent memoization for a store: memo tables whose entries survive a
//! restart. A table is declared with its key and value codecs, registered
//! as the slots `MemoKey(name)` and `MemoValue(name)`; the package makes
//! the closure the engine runs for its function.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use incr_core::{Cells, Incr, Runtime};
use incr_persist::memo::{
    BuildId, CheckpointStats, Memo, MemoBuilder, MemoCtx, MemoOptions, MemoStats, MemoTable,
};
use incr_persist::{Error, MemoPart};
use incr_python_protocol::{PyKey, PyValue};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::backend::PyBackend;
use crate::binding::{carry, Carry, PyBinding};
use crate::codec::{capture_keys, Slot};
use crate::memo_backend::PyMemoBackend;
use crate::store::{Keyed, StoreCore};

/// The closure the engine runs for a memo table of a `C` runtime.
pub type MemoFn<C> = Box<dyn Fn(&MemoCtx<'_, C>, &PyValue) -> PyValue + Send + Sync + 'static>;

/// The open memo of a package's store.
pub type OpenMemo<B> = Memo<<B as PyBinding>::C, PyBackend<<B as PyBinding>::Exceptions>>;

/// A declared table's handle, shared with the package's `MemoTable`
/// object. It names the builder that declared it, so it is replaced when
/// a failed open makes the builder again. A leaf lock: no Python runs
/// under it.
#[derive(Clone)]
pub struct TableHandle {
    table: Arc<Mutex<MemoTable<PyValue, PyValue>>>,
    /// The `label` its entries' nodes get as `<label>/<str(key)>`.
    label: Option<Arc<str>>,
}

impl TableHandle {
    fn get(&self) -> MemoTable<PyValue, PyValue> {
        *self.table.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn set(&self, table: MemoTable<PyValue, PyValue>) {
        *self.table.lock().unwrap_or_else(PoisonError::into_inner) = table;
    }

    /// A reference to `key` kept to label its entry, for a labelled table
    /// only. Taken with the GIL, before any store closure.
    fn spare_key(&self, py: Python<'_>, key: &Py<PyAny>) -> Option<Py<PyAny>> {
        self.label.as_ref().map(|_| key.clone_ref(py))
    }

    /// The label of `key`'s entry node, read with the GIL so the label
    /// can be set under the store lock with no Python code; `None` for
    /// an unlabelled table or a key whose `__str__` raises, which is
    /// reported as unraisable.
    fn entry_label(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> Option<String> {
        let label = self.label.as_deref()?;
        match key.str() {
            Ok(text) => Some(format!("{label}/{text}")),
            Err(e) => {
                e.write_unraisable(py, Some(key));
                None
            }
        }
    }
}

/// What the first store call of [`MemoCore::value_of`] returns: the value,
/// or an entry of a labelled table that has no label yet.
enum Entry {
    Value(PyValue),
    Unlabelled(Incr<PyValue>),
}

struct MemoDecl {
    name: String,
    version: u32,
    func: Py<PyAny>,
    handle: TableHandle,
}

/// A store's memo: the builder its tables are declared on until it opens,
/// the declarations, and the open memo. Kept beside the [`StoreCore`]
/// rather than in it, since a builder holds closures that are not `Send`.
/// The locks are leaves.
pub struct MemoCore<B: PyBinding> {
    builder: Mutex<Option<MemoBuilder<B::C>>>,
    decls: Mutex<Vec<MemoDecl>>,
    memo: OnceLock<OpenMemo<B>>,
    auto: Mutex<AutoCheckpoint>,
}

/// The failures of checkpoints a write triggered on a single-threaded
/// runtime, which the write does not raise.
#[derive(Default)]
struct AutoCheckpoint {
    errors: u64,
    last_error: Option<String>,
    /// After a failure, the unsaved count it left: the next triggered
    /// checkpoint waits until a compute raises the count above it, so a
    /// failing backend is not retried on every write.
    retry_above: Option<u64>,
}

impl<B: PyBinding> Default for MemoCore<B> {
    fn default() -> Self {
        MemoCore {
            builder: Mutex::default(),
            decls: Mutex::default(),
            memo: OnceLock::new(),
            auto: Mutex::default(),
        }
    }
}

impl<B: PyBinding> MemoCore<B> {
    fn builder(&self) -> MutexGuard<'_, Option<MemoBuilder<B::C>>> {
        self.builder.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn decls(&self) -> MutexGuard<'_, Vec<MemoDecl>> {
        self.decls.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn auto(&self) -> MutexGuard<'_, AutoCheckpoint> {
        self.auto.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn mismatch_message(tables: &[(String, &'static str)]) -> String {
    let names = tables
        .iter()
        .map(|(table, part)| format!("memo table '{table}' ({part})"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{names} did not round-trip through the codec (decode(encode(x)) != x): the generation \
         was written without those entries, they are recomputed on the next run, and nothing \
         wrong is stored"
    )
}

fn default_options<B: PyBinding>(core: &StoreCore<B>) -> MemoOptions {
    MemoOptions::new(BuildId::Fixed(core.build_id()))
}

/// The options `store.open_memo` takes beyond the backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenOptions {
    /// The size a checkpoint evicts the oldest entries down to; incr's
    /// default when `None`.
    pub max_bytes: Option<u64>,
    /// The unsaved entries that make a checkpoint due; none when `None`.
    pub checkpoint_after_entries: Option<u64>,
}

impl OpenOptions {
    /// Raises `ValueError` for a `checkpoint_after_entries` of 0.
    pub fn new(max_bytes: Option<u64>, checkpoint_after_entries: Option<u64>) -> PyResult<Self> {
        if checkpoint_after_entries == Some(0) {
            return Err(PyValueError::new_err(
                "checkpoint_after_entries must be at least 1",
            ));
        }
        Ok(OpenOptions {
            max_bytes,
            checkpoint_after_entries,
        })
    }

    fn memo_options(self, core: &StoreCore<impl PyBinding>) -> MemoOptions {
        let mut options = default_options(core);
        if let Some(max_bytes) = self.max_bytes {
            options = options.max_bytes(max_bytes);
        }
        if let Some(entries) = self.checkpoint_after_entries {
            options = options.checkpoint_after_entries(entries);
        }
        options
    }
}

/// `stats` as the dict `memo.checkpoint()` returns.
fn checkpoint_counts(stats: &CheckpointStats) -> HashMap<&'static str, usize> {
    HashMap::from([
        ("written", stats.written),
        ("unkept", stats.unkept),
        ("merged", stats.merged),
        ("corrupt", stats.corrupt),
        ("evicted", stats.evicted),
        ("oversized", stats.oversized),
        ("unchanged", usize::from(stats.unchanged)),
    ])
}

/// What `memo.stats()` returns.
pub struct PyMemoStats {
    pub stats: MemoStats,
    /// Triggered checkpoints that failed, by a write on a single-threaded
    /// runtime or by a Shared memo's checkpoint thread.
    pub auto_checkpoint_errors: u64,
    /// The text of the latest failure a write's triggered checkpoint
    /// raised; the checkpoint thread's failures carry none.
    pub last_auto_checkpoint_error: Option<String>,
}

impl PyMemoStats {
    /// The counts `memo.stats()` returns beside `last_auto_checkpoint_error`.
    pub fn counts(&self) -> [(&'static str, u64); 6] {
        let stats = &self.stats;
        [
            ("entries", stats.entries),
            ("bytes", stats.bytes),
            ("unsaved_entries", stats.unsaved_entries),
            ("unsaveable", stats.unsaveable),
            ("evicted_last_checkpoint", stats.evicted_last_checkpoint),
            ("auto_checkpoint_errors", self.auto_checkpoint_errors),
        ]
    }
}

impl<B: PyBinding> MemoCore<B> {
    /// The open memo; `RuntimeError` before `open_memo`.
    pub fn memo(&self) -> PyResult<&OpenMemo<B>> {
        self.memo
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("no memo is open: call store.open_memo first"))
    }

    pub fn is_open(&self) -> bool {
        self.memo.get().is_some()
    }

    /// Declares the memo table `name` with its codecs, whose entries the
    /// closure `make(func)` computes. Raises after `open_memo` has started.
    /// Holds the store's declare lock, as [`Self::open`] does, so a
    /// declaration and an open never interleave.
    #[expect(
        clippy::too_many_arguments,
        reason = "the Python method's arguments, passed through"
    )]
    pub fn declare_table(
        &self,
        core: &StoreCore<B>,
        py: Python<'_>,
        name: &str,
        version: u32,
        key_codec: Py<PyAny>,
        codec: Py<PyAny>,
        func: Py<PyAny>,
        label: Option<&str>,
        make: impl FnOnce(Py<PyAny>) -> MemoFn<B::C>,
    ) -> PyResult<TableHandle> {
        let _declaring = core.declare_lock(py)?;
        if self.is_open() {
            return Err(PyRuntimeError::new_err(
                "memo_table must be called before open_memo",
            ));
        }
        let key_slot = Slot::MemoKey(name.to_owned());
        let value_slot = Slot::MemoValue(name.to_owned());
        for (slot, given) in [(&key_slot, &key_codec), (&value_slot, &codec)] {
            if given.is_none(py) {
                return Err(PyValueError::new_err(format!(
                    "no codec is registered for {}",
                    slot.describe()
                )));
            }
        }
        if core.registry().is_registered(&key_slot) {
            return Err(PyValueError::new_err(format!(
                "memo table '{name}' is already declared"
            )));
        }
        if !func.bind(py).is_callable() {
            return Err(PyTypeError::new_err(
                "a memo table's function must be callable",
            ));
        }
        core.registry().register(key_slot, key_codec)?;
        core.registry().register(value_slot, codec)?;
        let f = make(func.clone_ref(py));
        let handle = {
            let mut builder = self.builder();
            let builder = builder.get_or_insert_with(|| MemoBuilder::new(default_options(core)));
            let table = builder.table::<PyValue, PyValue>(name, version);
            builder.define(table, f);
            TableHandle {
                table: Arc::new(Mutex::new(table)),
                label: label.map(Arc::from),
            }
        };
        self.decls().push(MemoDecl {
            name: name.to_owned(),
            version,
            func,
            handle: handle.clone(),
        });
        Ok(handle)
    }

    /// A builder over `options` made from the declarations, which their
    /// table handles name from now on.
    fn builder_from_decls(
        &self,
        py: Python<'_>,
        options: MemoOptions,
        make: &impl Fn(Py<PyAny>) -> MemoFn<B::C>,
    ) -> MemoBuilder<B::C> {
        let mut builder = MemoBuilder::new(options);
        for decl in self.decls().iter() {
            let table = builder.table::<PyValue, PyValue>(&decl.name, decl.version);
            builder.define(table, make(decl.func.clone_ref(py)));
            decl.handle.set(table);
        }
        builder
    }
}

/// Runs `restore`, a memo call that may restore entries, then labels the
/// per-key nodes of labelled keyed collections it created.
fn restoring<B: PyBinding, R>(core: &StoreCore<B>, py: Python<'_>, restore: impl FnOnce() -> R) -> R
where
    B::Task<Keyed<B>>: Send,
    B::Task<Vec<(Keyed<B>, String, PyKey)>>: Send,
{
    if !core.has_labelled_keys() {
        return restore();
    }
    let (result, keys) = capture_keys(restore);
    core.label_keys(py, keys);
    result
}

impl<B: PyBinding> MemoCore<B>
where
    B::Task<MemoBuilder<B::C>>: Send,
    B::Task<OpenMemo<B>>: Send,
    for<'a> B::Task<&'a OpenMemo<B>>: Send,
    B::Task<Keyed<B>>: Send,
    B::Task<Vec<(Keyed<B>, String, PyKey)>>: Send,
{
    /// `key`'s value in the table `handle` names, restored from the memo or
    /// computed, read as a dependency of whatever query is running on `rt`,
    /// which must be this store's runtime. Runs inside a query, entering
    /// the engine through the binding so a Shared call releases the GIL.
    pub fn value_of(
        &self,
        core: &StoreCore<B>,
        py: Python<'_>,
        rt: &Runtime<B::C>,
        name: &str,
        handle: &TableHandle,
        key: PyValue,
    ) -> PyResult<PyValue> {
        let memo = carry::<B, _>(self.memo()?);
        if !std::ptr::eq(core.runtime(), rt) {
            return Err(PyValueError::new_err(format!(
                "memo table '{name}' belongs to the runtime of its store, not this one"
            )));
        }
        let table = handle.get();
        let registry = core.registry();
        let spare = handle.spare_key(py, &key.0);
        let labelled_table = spare.is_some();
        let first = restoring(core, py, || {
            B::write(py, core.store(), move |store, _| {
                registry.scoped(|| {
                    let node = memo.into_inner().get_or_insert(table, key);
                    let rt = store.runtime();
                    if labelled_table && !rt.has_label(node.slot()) {
                        Entry::Unlabelled(node)
                    } else {
                        Entry::Value(rt.get(node))
                    }
                })
            })
        })?;
        let value = match first {
            Entry::Value(value) => value,
            Entry::Unlabelled(node) => {
                let label = spare.and_then(|key| handle.entry_label(py, key.bind(py)));
                restoring(core, py, || {
                    B::write(py, core.store(), move |store, _| {
                        registry.scoped(|| {
                            let rt = store.runtime();
                            if let Some(label) = label.filter(|_| !rt.has_label(node.slot())) {
                                rt.set_label(node.slot(), label);
                            }
                            rt.get(node)
                        })
                    })
                })?
            }
        };
        drop(core.registry().take_failure());
        Ok(value)
    }

    /// Opens the memo over the Python memo `backend` with `options`,
    /// loading its current generation. The builder is made from the
    /// declarations with `make`, so a later open can succeed after a
    /// failed one. Holds the store's declare lock.
    pub fn open(
        &self,
        core: &StoreCore<B>,
        py: Python<'_>,
        backend: Py<PyAny>,
        options: OpenOptions,
        make: impl Fn(Py<PyAny>) -> MemoFn<B::C>,
    ) -> PyResult<()> {
        let _declaring = core.declare_lock(py)?;
        if self.is_open() {
            return Err(PyRuntimeError::new_err("the memo is already open"));
        }
        let declared = self.builder().take();
        drop(declared);
        let builder = self.builder_from_decls(py, options.memo_options(core), &make);
        let builder = carry::<B, _>(builder);
        let opened = core.write(py, move |store, _| {
            builder
                .into_inner()
                .open(store, PyMemoBackend::<B::Exceptions>::new(backend))
                .map(carry::<B, _>)
        });
        match opened {
            Ok(memo) => self
                .memo
                .set(memo.into_inner())
                .map_err(|_| PyRuntimeError::new_err("the memo is already open")),
            Err(e) => {
                let builder = self.builder_from_decls(py, default_options(core), &make);
                *self.builder() = Some(builder);
                Err(e)
            }
        }
    }

    /// The node of the entry for `key` in the table `handle` names: the
    /// live one, one restored from the memo, or a new one.
    pub fn get_or_insert(
        &self,
        core: &StoreCore<B>,
        py: Python<'_>,
        handle: &TableHandle,
        key: Py<PyAny>,
    ) -> PyResult<Incr<PyValue>> {
        let memo = carry::<B, _>(self.memo()?);
        let table = handle.get();
        let registry = core.registry();
        let spare = handle.spare_key(py, &key);
        let labelled_table = spare.is_some();
        let (node, labelled) = restoring(core, py, || {
            B::read(py, core.store(), move |store, _| {
                registry.scoped(|| {
                    let node = memo.into_inner().get_or_insert(table, PyValue(key));
                    let labelled = !labelled_table || store.runtime().has_label(node.slot());
                    (node, labelled)
                })
            })
        })?;
        if !labelled {
            let label = spare.and_then(|key| handle.entry_label(py, key.bind(py)));
            if let Some(label) = label {
                let slot = node.slot();
                B::read(py, core.store(), move |store, _| {
                    let rt = store.runtime();
                    if !rt.has_label(slot) {
                        rt.set_label(slot, label);
                    }
                })?;
            }
        }
        drop(core.registry().take_failure());
        Ok(node)
    }

    /// Writes every live entry whose dependencies are persisted as the next
    /// generation and returns its counts. Raises `ValueError` naming the
    /// tables whose entries did not round-trip through their codec.
    pub fn checkpoint(
        &self,
        core: &StoreCore<B>,
        py: Python<'_>,
    ) -> PyResult<HashMap<&'static str, usize>> {
        let done = self.run_checkpoint(core, py, |memo| memo.checkpoint().map(Some))?;
        Ok(done.unwrap_or_default())
    }

    /// [`Self::checkpoint`] when `checkpoint_after_entries` entries are
    /// unsaved; `None` otherwise, and always when no trigger is set.
    pub fn checkpoint_if_due(
        &self,
        core: &StoreCore<B>,
        py: Python<'_>,
    ) -> PyResult<Option<HashMap<&'static str, usize>>> {
        self.run_checkpoint(core, py, OpenMemo::<B>::checkpoint_if_due)
    }

    /// Runs a due checkpoint after a logged write or batch commit on a
    /// single-threaded runtime; a Shared memo's checkpoint thread runs it
    /// instead. Does nothing when no memo is open. Never raises, since the
    /// write has landed: a failure is counted in [`Self::stats`] and the
    /// checkpoint stays due, retried by the first write after a compute
    /// adds an unsaved entry. A checkpoint another handle holds (`Busy`)
    /// is left for the next write, which finds it still due.
    pub fn after_write(&self, core: &StoreCore<B>, py: Python<'_>) {
        if !<B::C as Cells>::SINGLE_THREADED || !self.is_open() {
            return;
        }
        let retry_above = self.auto().retry_above;
        if let Some(noted) = retry_above {
            match self.memo_stats(core, py) {
                Ok(stats) if stats.unsaved_entries > noted => {}
                Ok(_) => return,
                Err(e) => return self.note_auto_failure(core, py, &e),
            }
        }
        let done = self.run_checkpoint(core, py, |memo| match memo.checkpoint_if_due() {
            Err(Error::Busy) => Ok(None),
            done => done,
        });
        match done {
            Ok(_) => self.auto().retry_above = None,
            Err(e) => self.note_auto_failure(core, py, &e),
        }
    }

    fn note_auto_failure(&self, core: &StoreCore<B>, py: Python<'_>, error: &PyErr) {
        let text = error.to_string();
        let unsaved = self.memo_stats(core, py).map(|stats| stats.unsaved_entries);
        let mut auto = self.auto();
        auto.errors += 1;
        auto.last_error = Some(text);
        if let Ok(unsaved) = unsaved {
            auto.retry_above = Some(unsaved);
        }
    }

    fn memo_stats(&self, core: &StoreCore<B>, py: Python<'_>) -> PyResult<MemoStats> {
        let memo = carry::<B, _>(self.memo()?);
        B::read(py, core.store(), move |_, _| memo.into_inner().stats())
    }

    /// The memo's counts, read without touching the backend.
    pub fn stats(&self, core: &StoreCore<B>, py: Python<'_>) -> PyResult<PyMemoStats> {
        let stats = self.memo_stats(core, py)?;
        let auto = self.auto();
        Ok(PyMemoStats {
            auto_checkpoint_errors: stats.auto_checkpoint_errors + auto.errors,
            last_auto_checkpoint_error: auto.last_error.clone(),
            stats,
        })
    }

    fn run_checkpoint(
        &self,
        core: &StoreCore<B>,
        py: Python<'_>,
        checkpoint: impl FnOnce(&OpenMemo<B>) -> Result<Option<CheckpointStats>, Error> + Send,
    ) -> PyResult<Option<HashMap<&'static str, usize>>> {
        let memo = self.memo()?;
        drop(core.registry().take_failure());
        let registry = core.registry();
        let task = carry::<B, _>(memo);
        let (done, mismatches) = B::write(py, core.store(), move |_, _| {
            registry.scoped(|| {
                let memo = task.into_inner();
                let done = checkpoint(memo);
                // Takes the memo lock, which a Shared checkpoint thread holds
                // while it needs the GIL: never with the GIL held.
                let mismatches = if done.is_ok() {
                    memo.take_encode_mismatches()
                } else {
                    Vec::new()
                };
                (done, mismatches)
            })
        })?;
        let cause = core.registry().take_failure();
        let stats = done.map_err(|error| crate::persist_error::<B::Exceptions>(py, error))?;
        if stats.is_some() {
            self.auto().retry_above = None;
        }
        if !mismatches.is_empty() {
            let tables = mismatches
                .into_iter()
                .map(|m| {
                    let part = match m.part {
                        MemoPart::Key => "keys",
                        MemoPart::Value => "values",
                    };
                    (m.table, part)
                })
                .collect::<Vec<_>>();
            let err = PyValueError::new_err(mismatch_message(&tables));
            err.set_cause(py, cause);
            return Err(err);
        }
        Ok(stats.as_ref().map(checkpoint_counts))
    }
}
