//! Persistent memoization for a store: memo tables whose entries survive a
//! restart. A table is declared with its key and value codecs, registered
//! as the slots `MemoKey(name)` and `MemoValue(name)`; the package makes
//! the closure the engine runs for its function.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use incr_core::{Incr, Runtime};
use incr_persist::memo::{BuildId, Memo, MemoBuilder, MemoCtx, MemoOptions, MemoTable};
use incr_persist::MemoPart;
use incr_python_protocol::PyValue;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::backend::PyBackend;
use crate::binding::{carry, Carry, PyBinding};
use crate::codec::Slot;
use crate::memo_backend::PyMemoBackend;
use crate::store::StoreCore;

/// The closure the engine runs for a memo table of a `C` runtime.
pub type MemoFn<C> = Box<dyn Fn(&MemoCtx<'_, C>, &PyValue) -> PyValue + Send + Sync + 'static>;

/// The open memo of a package's store.
pub type OpenMemo<B> = Memo<<B as PyBinding>::C, PyBackend<<B as PyBinding>::Exceptions>>;

/// A declared table's handle, shared with the package's `MemoTable`
/// object. It names the builder that declared it, so it is replaced when
/// a failed open makes the builder again. A leaf lock: no Python runs
/// under it.
#[derive(Clone)]
pub struct TableHandle(Arc<Mutex<MemoTable<PyValue, PyValue>>>);

impl TableHandle {
    fn get(&self) -> MemoTable<PyValue, PyValue> {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn set(&self, table: MemoTable<PyValue, PyValue>) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = table;
    }
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
}

impl<B: PyBinding> Default for MemoCore<B> {
    fn default() -> Self {
        MemoCore {
            builder: Mutex::default(),
            decls: Mutex::default(),
            memo: OnceLock::new(),
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

fn new_memo_builder<B: PyBinding>(core: &StoreCore<B>) -> MemoBuilder<B::C> {
    MemoBuilder::new(MemoOptions::new(BuildId::Fixed(core.build_id())))
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
    /// closure `make(func)` computes. Raises before `open_memo` only.
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
        make: impl FnOnce(Py<PyAny>) -> MemoFn<B::C>,
    ) -> PyResult<TableHandle> {
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
            let builder = builder.get_or_insert_with(|| new_memo_builder(core));
            let table = builder.table::<PyValue, PyValue>(name, version);
            builder.define(table, f);
            TableHandle(Arc::new(Mutex::new(table)))
        };
        self.decls().push(MemoDecl {
            name: name.to_owned(),
            version,
            func,
            handle: handle.clone(),
        });
        Ok(handle)
    }

    /// Makes the builder again from the declarations, after a failed open
    /// consumed it.
    fn restore_builder(
        &self,
        core: &StoreCore<B>,
        py: Python<'_>,
        make: impl Fn(Py<PyAny>) -> MemoFn<B::C>,
    ) {
        let mut builder = new_memo_builder(core);
        for decl in self.decls().iter() {
            let table = builder.table::<PyValue, PyValue>(&decl.name, decl.version);
            builder.define(table, make(decl.func.clone_ref(py)));
            decl.handle.set(table);
        }
        *self.builder() = Some(builder);
    }

    /// `key`'s value in the table `handle` names, restored from the memo or
    /// computed, read as a dependency of whatever query is running on `rt`,
    /// which must be this store's runtime. Runs inside a query, so it
    /// takes no engine entry.
    pub fn value_of(
        &self,
        core: &StoreCore<B>,
        rt: &Runtime<B::C>,
        name: &str,
        handle: &TableHandle,
        key: PyValue,
    ) -> PyResult<PyValue> {
        let memo = self.memo()?;
        if !std::ptr::eq(core.runtime(), rt) {
            return Err(PyValueError::new_err(format!(
                "memo table '{name}' belongs to the runtime of its store, not this one"
            )));
        }
        let table = handle.get();
        let value = core.registry().scoped(|| {
            let node = memo.get_or_insert(table, key);
            rt.get(node)
        });
        drop(core.registry().take_failure());
        Ok(value)
    }
}

impl<B: PyBinding> MemoCore<B>
where
    B::Task<MemoBuilder<B::C>>: Send,
    B::Task<OpenMemo<B>>: Send,
    for<'a> B::Task<&'a OpenMemo<B>>: Send,
{
    /// Opens the memo over the Python memo `backend`, loading its current
    /// generation. On failure the builder is made again with `make`, so
    /// a later open can succeed.
    pub fn open(
        &self,
        core: &StoreCore<B>,
        py: Python<'_>,
        backend: Py<PyAny>,
        make: impl Fn(Py<PyAny>) -> MemoFn<B::C>,
    ) -> PyResult<()> {
        if self.is_open() {
            return Err(PyRuntimeError::new_err("the memo is already open"));
        }
        let builder = self
            .builder()
            .take()
            .unwrap_or_else(|| new_memo_builder(core));
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
                self.restore_builder(core, py, make);
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
        let node = B::read(py, core.store(), move |_, _| {
            registry.scoped(|| memo.into_inner().get_or_insert(table, PyValue(key)))
        })?;
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
        let memo = self.memo()?;
        drop(core.registry().take_failure());
        let registry = core.registry();
        let task = carry::<B, _>(memo);
        let done = B::write(py, core.store(), move |_, _| {
            registry.scoped(|| task.into_inner().checkpoint())
        })?;
        let cause = core.registry().take_failure();
        let stats = done.map_err(|error| crate::persist_error::<B::Exceptions>(py, error))?;
        let mismatches = memo.take_encode_mismatches();
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
        Ok(HashMap::from([
            ("written", stats.written),
            ("unkept", stats.unkept),
            ("merged", stats.merged),
            ("corrupt", stats.corrupt),
            ("evicted", stats.evicted),
            ("oversized", stats.oversized),
        ]))
    }
}
