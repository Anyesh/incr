use std::any::{type_name, Any};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use incr_core::{check_value_code, Incr, Runtime, Shared};

use super::handle::{MemoCtx, MemoKey, MemoTable, MemoValue};
use super::memo_core::{CoreRef, IndexGen, MemoCells, MemoCore, MemoStats, Wake};
use super::options::MemoOptions;
use super::table_core::{ErasedTable, TableCore};
use super::{CheckpointStats, GenId, Generation, MemoBackend, TableDesc};
use crate::value_context::EncodeMismatch;
use crate::{Backend, Error, Result, Store};

static NEXT_MEMO: AtomicU64 = AtomicU64::new(1);

type Make<C> = Box<dyn FnOnce(&Runtime<C>) -> Box<dyn ErasedTable<C>>>;

struct Declared<C: MemoCells> {
    desc: TableDesc,
    make: Option<Make<C>>,
}

/// Declares a memo's tables, then opens it over a [`Store`]. Tables are
/// declared before any is defined, so closures can capture any of them.
pub struct MemoBuilder<C: MemoCells> {
    id: u64,
    options: MemoOptions,
    tables: Vec<Declared<C>>,
}

impl<C: MemoCells> MemoBuilder<C> {
    pub fn new(options: MemoOptions) -> Self {
        MemoBuilder {
            id: NEXT_MEMO.fetch_add(1, Ordering::Relaxed),
            options,
            tables: Vec::new(),
        }
    }

    /// Declares table `name`. Entries are kept only while `name`, `version`
    /// and the type names of `K` and `T` stay the same, so bump `version`
    /// when the closure's meaning changes. Panics if `name` is taken.
    pub fn table<K: MemoKey, T: MemoValue>(&mut self, name: &str, version: u32) -> MemoTable<K, T> {
        assert!(
            self.tables.iter().all(|t| t.desc.name != name),
            "incr-persist: memo table {name:?} declared twice"
        );
        let index = self.tables.len() as u32;
        self.tables.push(Declared {
            desc: TableDesc {
                name: name.to_owned(),
                version,
                key_type: type_name::<K>().to_owned(),
                value_type: type_name::<T>().to_owned(),
            },
            make: None,
        });
        MemoTable::new(self.id, index)
    }

    /// Gives `table` its closure. It must be a pure function of `key` and
    /// what it reads through the context or captured handles, and capture
    /// only handles and constants. Panics if `table` is already defined or
    /// was declared by another builder.
    pub fn define<K: MemoKey, T: MemoValue, F>(&mut self, table: MemoTable<K, T>, f: F)
    where
        F: Fn(&MemoCtx<'_, C>, &K) -> T + Send + Sync + 'static,
    {
        assert_eq!(
            table.memo, self.id,
            "incr-persist: a MemoTable defined on a builder other than the one that declared it"
        );
        let declared = &mut self.tables[table.index as usize];
        assert!(
            declared.make.is_none(),
            "incr-persist: memo table {:?} defined twice",
            declared.desc.name
        );
        let index = table.index;
        let name: Arc<str> = declared.desc.name.as_str().into();
        declared.make = Some(Box::new(move |rt| {
            Box::new(TableCore::<K, T, C> {
                index,
                name,
                nodes: rt.create_table(),
                f: Arc::new(f),
            })
        }));
    }

    /// Opens the memo over `store`, loading `backend`'s current generation.
    /// Entries of another build or salt are ignored, and so are a table's
    /// entries under another catalog descriptor.
    pub fn open<B: Backend + 'static>(
        self,
        store: &Arc<Store<C, B>>,
        backend: impl MemoBackend + 'static,
    ) -> Result<Memo<C, B>> {
        let build = self.options.build()?;
        if self.options.auto_checkpoint_secs.is_some() && C::SINGLE_THREADED {
            return Err(Error::AutoCheckpointNeedsShared);
        }
        let mut catalog = Vec::with_capacity(self.tables.len());
        let mut makes = Vec::with_capacity(self.tables.len());
        for declared in self.tables {
            let make = declared
                .make
                .ok_or_else(|| Error::UndefinedTable(declared.desc.name.clone()))?;
            catalog.push(declared.desc);
            makes.push(make);
        }
        let deps = store.open_dep_index()?;
        let generation = backend
            .load()?
            .unwrap_or_else(|| Generation::empty(GenId(0)));
        let index = IndexGen::new(generation, build, &catalog);
        let rt = store.runtime();
        let tables = makes.into_iter().map(|make| make(rt)).collect();
        let core = CoreRef(Arc::new(MemoCore::new(
            self.id,
            build,
            catalog,
            tables,
            Box::new(backend),
            deps,
            rt.create_table(),
            self.options.max_bytes,
            self.options.checkpoint_after_entries,
            index,
        )));
        let every = self.options.auto_checkpoint_secs.map(Duration::from_secs);
        let triggered = self.options.checkpoint_after_entries.is_some();
        let auto = (!C::SINGLE_THREADED && (every.is_some() || triggered))
            .then(|| AutoCheckpoint::start(store, &core, every));
        Ok(Memo {
            store: Arc::clone(store),
            core,
            auto,
        })
    }
}

/// An open memo: tables of query nodes whose values survive restarts. A
/// miss restores a stored entry seeded with its value and deps, so it
/// verifies instead of running when the deps are unchanged.
pub struct Memo<C: MemoCells, B: Backend> {
    store: Arc<Store<C, B>>,
    core: CoreRef<C>,
    auto: Option<AutoCheckpoint>,
}

impl<C: MemoCells, B: Backend> Memo<C, B> {
    pub fn store(&self) -> &Arc<Store<C, B>> {
        &self.store
    }

    pub fn runtime(&self) -> &Runtime<C> {
        self.store.runtime()
    }

    /// `table`'s node for `key`: the live one, or one restored from the
    /// memo, or one created cold. Inside a memo closure use
    /// [`MemoCtx::get_or_insert`].
    pub fn get_or_insert<K: MemoKey, T: MemoValue>(
        &self,
        table: MemoTable<K, T>,
        key: K,
    ) -> Incr<T> {
        self.core.get_or_insert(self.runtime(), table, key)
    }

    /// `table`'s live node for `key`, if any; restores nothing. Not a
    /// dependency, so not allowed inside a compute.
    pub fn get<K: MemoKey, T: MemoValue>(
        &self,
        table: MemoTable<K, T>,
        key: &K,
    ) -> Option<Incr<T>> {
        let t: &TableCore<K, T, C> = self.core.table(table.memo, table.index);
        t.nodes.get(self.runtime(), key)
    }

    /// Sets external `name` to `fp`, creating its input on the first call,
    /// and returns the input for closures to read. Entries over it are
    /// reused across restarts only while it is set to the same `fp`.
    pub fn set_external(&self, name: &str, fp: u128) -> Incr<u128> {
        let rt = self.runtime();
        assert!(
            !rt.frame_active(),
            "incr-persist: Memo::set_external called from inside a query's compute"
        );
        check_value_code();
        self.core.set_external(rt, name, fp)
    }

    /// Writes every live entry whose deps all map into the next
    /// generation, merged with the current one and evicted down to
    /// `max_bytes`. Writes nothing, and reports
    /// [`CheckpointStats::unchanged`], when no entry computed since it was
    /// last saved can be saved. `Err(Busy)` if another handle is
    /// checkpointing.
    pub fn checkpoint(&self) -> Result<CheckpointStats> {
        self.core.checkpoint(self.runtime())
    }

    /// [`Self::checkpoint`] when
    /// [`MemoOptions::checkpoint_after_entries`](super::MemoOptions::checkpoint_after_entries)
    /// entries are unsaved; `Ok(None)` otherwise, and always when no
    /// trigger is set. For after a write or a read-only rebuild, so the
    /// stall lands where the caller chooses.
    pub fn checkpoint_if_due(&self) -> Result<Option<CheckpointStats>> {
        check_value_code();
        if !self.core.checkpoint_due() {
            return Ok(None);
        }
        self.checkpoint().map(Some)
    }

    pub fn stats(&self) -> MemoStats {
        check_value_code();
        self.core.stats()
    }

    /// The entries checkpoints left out since the last call because their
    /// key or value did not round-trip through serde, one per table and
    /// part. The checkpoint itself succeeded; those entries are recomputed
    /// after the next restart.
    pub fn take_encode_mismatches(&self) -> Vec<EncodeMismatch> {
        check_value_code();
        std::mem::take(&mut self.core.lock_memo().mismatches)
    }
}

impl<C: MemoCells, B: Backend> Drop for Memo<C, B> {
    fn drop(&mut self) {
        if let Some(auto) = self.auto.take() {
            auto.stop();
        }
    }
}

struct AutoCheckpoint {
    wake: Arc<Wake>,
    thread: JoinHandle<()>,
}

impl AutoCheckpoint {
    /// Only called for `Shared`, which `open` checks. Checkpoints every
    /// `every` and whenever the trigger makes one due, skipping a turn with
    /// nothing unsaved.
    fn start<C: MemoCells, B: Backend + 'static>(
        store: &Arc<Store<C, B>>,
        core: &CoreRef<C>,
        every: Option<Duration>,
    ) -> Self {
        let store: Box<dyn Any> = Box::new(Arc::clone(store));
        let core: Box<dyn Any> = Box::new(core.clone());
        let (Ok(store), Ok(core)) = (
            store.downcast::<Arc<Store<Shared, B>>>(),
            core.downcast::<CoreRef<Shared>>(),
        ) else {
            unreachable!("incr-persist: automatic checkpoints are Shared-only");
        };
        let wake = Arc::clone(&core.wake);
        let thread = thread::spawn(move || loop {
            core.wake.arm();
            if core.wake.wait(every) {
                return;
            }
            if core.dirty.unsaved() == 0 {
                continue;
            }
            // A panic in a key's or value's code fails this checkpoint, not
            // the thread, so later intervals still run.
            match catch_unwind(AssertUnwindSafe(|| core.checkpoint(store.runtime()))) {
                Ok(
                    Ok(_) | Err(Error::Busy | Error::UnloggedScopeOpen | Error::StagedBatchOpen),
                ) => {}
                Ok(Err(_)) | Err(_) => core.lock_memo().stats.auto_checkpoint_errors += 1,
            }
        });
        AutoCheckpoint { wake, thread }
    }

    fn stop(self) {
        self.wake.stop();
        if let Err(payload) = self.thread.join() {
            if !thread::panicking() {
                resume_unwind(payload);
            }
        }
    }
}
