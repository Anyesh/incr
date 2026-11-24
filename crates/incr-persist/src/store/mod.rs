use std::any::Any;
use std::collections::{BTreeSet, HashMap};
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use incr_core::lock_order::{LockKind, Ordered};
use incr_core::persist_hooks::changed_at;
#[cfg(feature = "yield-hook")]
use incr_core::YieldHook;
use incr_core::{
    in_value_code, with_value_code, Cells, DeleteError, Incr, RoutingHold, Runtime, Value, Yielder,
};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::value_context::{enter_value_scope, ValueScopeGuard};
use crate::{Backend, Error, Kind, Position, ReadOutcome, Record, Result, SnapshotId, ValueScope};

use after_lock::{AfterLock, Drained, Holder, HolderMark};
use binding::{decode, encode, Binding, Folded, InputBinding};
use dep_index::{fp_of, DepFp, DepIndex, Registered};
use logged::{changes_between, fold, is_element_change, Declared, LoggedKey, LoggedKeys};

pub use collection::PersistedCollection;
pub use keyed::{PendingWrite, PersistedKeyed};
pub use scope::UnloggedScope;
pub use staged::StagedBatch;

mod after_lock;
mod binding;
mod collection;
pub(crate) mod dep_index;
#[cfg(test)]
#[path = "dep_index_tests.rs"]
mod dep_index_tests;
#[cfg(test)]
#[path = "drop_tests.rs"]
mod drop_tests;
pub(crate) mod keyed;
#[cfg(test)]
#[path = "keyed_tests.rs"]
mod keyed_tests;
mod logged;
mod rows;
mod scope;
#[cfg(test)]
#[path = "scope_tests.rs"]
mod scope_tests;
mod staged;
#[cfg(test)]
#[path = "staged_tests.rs"]
mod staged_tests;

#[cfg(test)]
thread_local! {
    /// Called by [`Store::undeclare`] on its thread, store mutex held, just
    /// before it deletes the key's node.
    pub(crate) static UNDECLARE_PARK: std::cell::RefCell<Option<Box<dyn Fn()>>> =
        const { std::cell::RefCell::new(None) };
}

/// The bound on values a [`Store`] persists: a runtime [`Value`] that serde
/// can write to and read from the log.
pub trait PersistValue: Value + Serialize + DeserializeOwned {}

impl<T: Value + Serialize + DeserializeOwned> PersistValue for T {}

/// Whether a store write waits for the backend to make it durable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Durability {
    /// Every write calls [`Backend::sync`] before it is applied, so an
    /// acknowledged write survives a crash.
    #[default]
    Sync,
    /// Writes skip `sync`, accepting a window in which acknowledged writes
    /// can be lost to a crash, in exchange for much cheaper writes.
    Buffered,
}

/// How [`Store::open`] runs a store.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Whether writes sync before they are applied. Defaults to
    /// [`Durability::Sync`].
    pub durability: Durability,
    /// When set to `Some(n)`, a write first runs [`Store::checkpoint`] once
    /// the store has applied `n` records since it last checkpointed or
    /// caught up across a compaction. The checkpoint runs before the write
    /// appends anything, so its failure fails the write and leaves the log
    /// unchanged. A [`StagedBatch`] runs a due checkpoint when it opens, so
    /// none falls between its records, and its commit never triggers one.
    /// `None`, the default, compacts only on explicit calls.
    pub auto_checkpoint_records: Option<u64>,
    /// The scope this store's value code finds through
    /// [`current_value_scope`](crate::current_value_scope): it is current
    /// on the thread holding the store's lock, so encodes and decodes the
    /// store runs, on any thread, see it. Value code the caller runs
    /// outside a store call enters it with [`Store::enter_value_scope`].
    pub value_scope: Option<ValueScope>,
    /// The hook of the store's runtime, which restores, checkpoints and
    /// catch-ups also call (`Runtime::set_yield_hook`).
    #[cfg(feature = "yield-hook")]
    pub yield_hook: Option<YieldHook>,
}

/// What [`Store::stats`] reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct StoreStats {
    /// The last log position the store has applied.
    pub position: Position,
    /// The position of the snapshot the store last wrote or crossed,
    /// [`Position::ZERO`] if none.
    pub snapshot_position: Position,
    /// Records applied since that snapshot, the count
    /// [`Options::auto_checkpoint_records`] compares against.
    pub records_since_checkpoint: u64,
}

/// A persisted input: a key in the log bound to an input node of the
/// store's runtime.
pub struct Input<T: 'static> {
    id: u32,
    handle: Incr<T>,
}

/// Where a record passed to a [`Store::subscribe`] callback came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Origin {
    /// Appended by a write through this store.
    Local,
    /// Appended by another writer and read by [`Store::refresh`] or by a
    /// write catching up before its own records.
    Foreign,
    /// Not a log record. The store caught up across a compaction, so it
    /// never saw the folded records one by one; instead it reports, at the
    /// snapshot's position, records that take each changed key from its old
    /// state to the snapshot's. It does the same, at the position it had
    /// applied through, when it finds a compaction at or below that
    /// position which it never crossed, chiefly to report the keys that
    /// compaction forgot. The records are a `Declare` for a new
    /// declaration, a `Set` for a new value, a `Delete` for a value reset to
    /// its default, and a `Remove` or `Insert` per differing element
    /// occurrence. A key the snapshot forgot is a [`Record::Forget`], and
    /// one whose `Declare` was forgotten or changed is a `Forget` followed
    /// by the key's full state, since a first-wins fold ignores a second
    /// `Declare`. A collection this store did not declare whose elements
    /// cannot be compared by bytes, or whose snapshot lists them in an order
    /// its own records do not reach, is reported as a `Delete` followed by
    /// its full contents. A writer that declared the collection lists them
    /// by rank, the fingerprint of each one's encoding, where an insert
    /// rarely lands last, so a key with more than one distinct element is
    /// usually reported that way.
    Snapshot,
}

/// Names a subscription for [`Store::unsubscribe`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubscriptionId(u64);

type Subscriber = Box<dyn FnMut(Position, &Record, Origin) + Send>;

/// Names one key of a store, for [`Store::delete`], [`Batch::delete_key`]
/// and [`Store::undeclare`]: an [`Input`], a `&`[`PersistedCollection`], or
/// the key's name as a string. A name reaches any key, declared in this process or not; a
/// handle is checked against the store, which rejects one it did not issue
/// with [`Error::UnknownHandle`].
pub struct KeyRef<'a>(Target<'a>);

enum Target<'a> {
    Name(&'a str),
    Handle { id: u32, node: Box<dyn Any> },
}

/// Records collected by [`Store::batch`] for one append.
pub struct Batch<'s, C: Cells, B: Backend> {
    store: &'s Store<C, B>,
    records: Vec<Record>,
    error: Option<Error>,
}

/// A runtime kept in sync with an ordered log: a materialized view over a
/// [`Backend`].
///
/// The store changes its runtime only by applying log records in position
/// order. A write appends its records, syncs them when
/// [`Options::durability`] is [`Durability::Sync`], then applies every
/// record after the last one applied up to its own, including records other
/// writers appended in between. All of this runs under one store mutex, so
/// writes from several threads on a `Store<Shared, _>` serialize.
///
/// Inputs are created with [`Store::input`] and read through the runtime
/// handed out by [`Store::runtime`]; queries over them are ordinary runtime
/// queries. Writing an input through the runtime directly bypasses the log
/// and is lost on restart.
///
/// Store methods must not be called from inside a query's compute, because
/// they create and set runtime inputs, nor from a value, key or element
/// type's trait method that the engine or the store is running, because
/// that call may run under the store's own lock. Either call panics, in
/// release builds too, before the store's lock is requested, except that a
/// call on the thread holding this store's lock returns
/// [`Error::Reentrant`].
///
/// A value's `Drop` may call the store. The values a store call displaces,
/// decodes or clones drop after it releases its lock. Where the engine
/// decides when a value drops (a node's values deleted by
/// [`Store::undeclare`], and on `Shared` an input's displaced value, which
/// the engine reclaims later on any thread), the `Drop` may run inside a
/// store call; a call into that same store from it on that thread returns
/// [`Error::Reentrant`] instead of deadlocking.
///
/// A `Store<Local, _>` is confined to its thread like its runtime:
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<incr_persist::Store<incr_persist::Local, incr_persist::MemoryBackend>>();
/// ```
pub struct Store<C: Cells, B: Backend> {
    runtime: Runtime<C>,
    backend: Arc<B>,
    options: Options,
    state: Mutex<State<C>>,
    holder: Holder,
}

/// The store mutex with a routing hold taken before it. Fields drop in
/// declaration order: the holder mark clears before the mutex is
/// released, the mutex before the hold routes what the writes under it
/// queued, and the values the call left in [`State::after`], moved out
/// by `drop`, last.
struct Locked<'a, C: Cells> {
    _holder: HolderMark<'a>,
    guard: Ordered<MutexGuard<'a, State<C>>>,
    _routing: RoutingHold<'a, C>,
    _scope: Option<ValueScopeGuard>,
    drained: Drained<'a>,
}

impl<C: Cells> Drop for Locked<'_, C> {
    fn drop(&mut self) {
        if self._holder.take_poison() {
            self.guard.poison();
        }
        self.drained.take_from(&self.guard.after);
    }
}

impl<C: Cells> std::ops::Deref for Locked<'_, C> {
    type Target = State<C>;

    fn deref(&self) -> &State<C> {
        &self.guard
    }
}

impl<C: Cells> std::ops::DerefMut for Locked<'_, C> {
    fn deref_mut(&mut self) -> &mut State<C> {
        &mut self.guard
    }
}

struct State<C: Cells> {
    applied: Position,
    /// The backend snapshot the folded log and bound nodes account for:
    /// one this store crossed, compacted to, or reconciled with. Any other
    /// snapshot at or below `applied` may have forgotten keys this store
    /// still holds.
    snapshot: Option<SnapshotId>,
    poisoned: bool,
    unlogged: bool,
    /// Set while a [`StagedBatch`] is open, whose writes are applied but
    /// not yet appended.
    staged: bool,
    since_checkpoint: u64,
    /// The position of the snapshot this store last wrote or crossed,
    /// whether or not `snapshot` could name it.
    snapshot_at: Position,
    logged: LoggedKeys,
    bound: HashMap<String, u32>,
    /// Bindings by id. Ids are never reused, so a handle whose key was
    /// undeclared finds nothing rather than a later binding.
    bindings: HashMap<u32, Box<dyn Binding<C>>>,
    next_id: u32,
    subscribers: Vec<(SubscriptionId, Subscriber)>,
    next_subscription: u64,
    /// Kept current by every mapped write once a memo has opened it.
    deps: Option<Arc<DepIndex>>,
    /// Empty whenever the mutex is free.
    after: AfterLock,
}

impl<T: 'static> Input<T> {
    /// The runtime node holding the input's value, for reads and queries.
    /// Write through [`Store::set`], not through this handle: the log never
    /// sees a direct write, and debug builds report one as
    /// [`Error::Bypassed`] at the input's next [`Store::set`] or at the next
    /// [`Store::checkpoint`].
    pub fn handle(self) -> Incr<T> {
        self.handle
    }
}

impl<T: 'static> Clone for Input<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: 'static> Copy for Input<T> {}

impl<T: 'static> PartialEq for Input<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.handle == other.handle
    }
}

impl<T: 'static> Eq for Input<T> {}

impl<T: 'static> std::fmt::Debug for Input<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Input")
            .field("id", &self.id)
            .field("handle", &self.handle)
            .finish()
    }
}

impl KeyRef<'_> {
    fn handle<T: 'static>(id: u32, node: Incr<T>) -> Self {
        KeyRef(Target::Handle {
            id,
            node: Box::new(node),
        })
    }
}

impl<T: 'static> From<Input<T>> for KeyRef<'_> {
    fn from(input: Input<T>) -> Self {
        KeyRef::handle(input.id, input.handle)
    }
}

impl<'a> From<&'a str> for KeyRef<'a> {
    fn from(key: &'a str) -> Self {
        KeyRef(Target::Name(key))
    }
}

impl<'a> From<&'a String> for KeyRef<'a> {
    fn from(key: &'a String) -> Self {
        KeyRef(Target::Name(key))
    }
}

impl<C: Cells, B: Backend> Batch<'_, C, B> {
    /// Adds a `Set` of `input` to the batch. A failure (an unknown handle, a
    /// poisoned store, a value that fails to serialize) is kept and returned
    /// by [`Store::batch`], which then appends nothing.
    pub fn set<T: PersistValue>(&mut self, input: Input<T>, value: T) -> &mut Self {
        let rt = &self.store.runtime;
        self.push(|state| set_record(state, rt, input, &value))
    }

    /// Adds a `Delete` of `key` to the batch, as [`Store::delete`] would
    /// write it. Named `delete_key` because [`Batch::delete`] removes one
    /// element from a collection.
    pub fn delete_key<'k>(&mut self, key: impl Into<KeyRef<'k>>) -> &mut Self {
        let key = key.into();
        self.push(|state| delete_record(state, &key))
    }

    fn push(&mut self, record: impl FnOnce(&State<C>) -> Result<Record>) -> &mut Self {
        if self.error.is_none() {
            match self.store.state().and_then(|state| record(&state)) {
                Ok(record) => self.records.push(record),
                Err(e) => self.error = Some(e),
            }
        }
        self
    }
}

impl<C: Cells> State<C> {
    /// Marks the store poisoned, for its memo too, and closes the staged
    /// flag a staged batch left: nothing will commit.
    fn poison(&mut self) {
        self.poisoned = true;
        if let Some(deps) = &self.deps {
            deps.set_poisoned();
            if self.staged {
                deps.set_staged(false);
            }
        }
        self.staged = false;
    }

    /// Fails while an unlogged scope or a staged batch is open, during
    /// which the runtime holds changes the log has not received.
    fn refuse_in_scope(&self) -> Result<()> {
        if self.unlogged {
            Err(Error::UnloggedScopeOpen)
        } else if self.staged {
            Err(Error::StagedBatchOpen)
        } else {
            Ok(())
        }
    }

    /// The id under which this store bound `key`, if it did. Fails with
    /// [`Error::KindMismatch`] when the key is bound as the other kind.
    fn bound_id(&self, key: &str, kind: Kind) -> Result<Option<u32>> {
        let Some(&id) = self.bound.get(key) else {
            return Ok(None);
        };
        let declared = self.bindings[&id].kind();
        if declared != kind {
            return Err(Error::KindMismatch {
                key: key.to_owned(),
                declared,
            });
        }
        Ok(Some(id))
    }

    /// The name of the key `target` refers to. Fails with
    /// [`Error::UnknownHandle`] for a handle this store did not issue.
    fn key_of(&self, target: &KeyRef<'_>) -> Result<String> {
        match &target.0 {
            Target::Name(key) => Ok((*key).to_owned()),
            Target::Handle { id, node } => {
                Ok(self.handle_binding(*id, node.as_ref())?.key().to_owned())
            }
        }
    }

    /// The id of the binding `target` refers to; `None` for a name this
    /// store has not declared. Fails as [`State::key_of`] does.
    fn binding_id(&self, target: &KeyRef<'_>) -> Result<Option<u32>> {
        match &target.0 {
            Target::Name(key) => Ok(self.bound.get(*key).copied()),
            Target::Handle { id, node } => {
                self.handle_binding(*id, node.as_ref()).map(|_| Some(*id))
            }
        }
    }

    /// The binding issued as `id` for the runtime handle `node`.
    fn handle_binding(&self, id: u32, node: &dyn Any) -> Result<&dyn Binding<C>> {
        self.bindings
            .get(&id)
            .filter(|binding| binding.is_handle(node))
            .map(|binding| binding.as_ref())
            .ok_or(Error::UnknownHandle)
    }

    fn notify(&mut self, position: Position, record: &Record, origin: Origin) {
        for (_, subscriber) in &mut self.subscribers {
            subscriber(position, record, origin);
        }
    }

    fn bind(&mut self, key: &str, binding: Box<dyn Binding<C>>) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.bindings.insert(id, binding);
        self.bound.insert(key.to_owned(), id);
        id
    }
}

impl<C: Cells, B: Backend> Store<C, B> {
    /// Opens a store over `backend`, replaying the log (its snapshot, then
    /// the records after it) into a fresh runtime.
    ///
    /// Replay only folds records; a key's runtime node is created when this
    /// process declares it with [`Store::input`].
    pub fn open(backend: B, options: Options) -> Result<Self> {
        #[cfg_attr(not(feature = "yield-hook"), allow(unused_mut))]
        let mut runtime = Runtime::new();
        #[cfg(feature = "yield-hook")]
        runtime.set_yield_hook(options.yield_hook.clone());
        let store = Store {
            runtime,
            backend: Arc::new(backend),
            options,
            state: Mutex::new(State {
                applied: Position::ZERO,
                snapshot: None,
                poisoned: false,
                unlogged: false,
                staged: false,
                since_checkpoint: 0,
                snapshot_at: Position::ZERO,
                logged: LoggedKeys::new(),
                bound: HashMap::new(),
                bindings: HashMap::new(),
                next_id: 0,
                subscribers: Vec::new(),
                next_subscription: 0,
                deps: None,
                after: AfterLock::default(),
            }),
            holder: Holder::default(),
        };
        {
            let mut state = store.state()?;
            store.catch_up(&mut state, None)?;
        }
        Ok(store)
    }

    /// The store's [`DepIndex`], created on the first call with every key
    /// this store has declared. Encodes nothing: a collection registers a
    /// pending fingerprint that the first memo read needing it computes.
    /// From then on every mapped write keeps it current.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn open_dep_index(&self) -> Result<Arc<DepIndex>> {
        let mut guard = self.state()?;
        let state = &mut *guard;
        state.refuse_in_scope()?;
        if let Some(deps) = &state.deps {
            return Ok(Arc::clone(deps));
        }
        let mut entries = Vec::with_capacity(state.bound.len());
        for (key, id) in &state.bound {
            let binding = &state.bindings[id];
            entries.push((
                key,
                binding.node(),
                binding.kind(),
                binding.dep_fp(self.value_scope()),
                binding.mapped_stamp(),
                binding.row_source(),
            ));
        }
        let deps = Arc::new(DepIndex::default());
        {
            let mut map = deps.write();
            for (key, handle, kind, fp, stamp, rows) in entries {
                map.register(key, handle, kind, fp, stamp);
                if let Some(rows) = rows {
                    map.register_rows(key, rows);
                }
            }
        }
        state.deps = Some(Arc::clone(&deps));
        Ok(deps)
    }

    /// Makes [`Options::value_scope`] current on this thread until the
    /// guard drops; `None` when the store has no scope.
    pub fn enter_value_scope(&self) -> Option<ValueScopeGuard> {
        self.options.value_scope.clone().map(enter_value_scope)
    }

    /// [`Options::value_scope`], for code that enters it on another thread.
    pub(crate) fn value_scope(&self) -> Option<ValueScope> {
        self.options.value_scope.clone()
    }

    /// The runtime the store keeps in sync with the log. Create queries and
    /// read values through it; write inputs through the store.
    pub fn runtime(&self) -> &Runtime<C> {
        &self.runtime
    }

    /// Declares the input `key` and returns its handle.
    ///
    /// If neither this store nor the log, read to its tail first, has a
    /// live `Declare` for `key`, appends one carrying `default` through the
    /// write path, so a read-only store can declare any key another writer
    /// declared without calling [`Store::refresh`]. The first `Declare` in
    /// log order wins: its default, not necessarily `default`, is what the
    /// key resets to, in every process. The input starts at the key's
    /// latest logged value, else at that logged default. Declaring a key
    /// this store already declared returns the same handle. A key that a
    /// compaction at or below this store's position forgot has no live
    /// `Declare`, so it is declared afresh, as after a restart.
    ///
    /// Fails with [`Error::TypeMismatch`] when the logged bytes do not
    /// decode as `T` or the key is already bound to another type, and with
    /// [`Error::KindMismatch`] when the log declares a collection. Neither
    /// poisons the store.
    pub fn input<T: PersistValue>(&self, key: &str, default: T) -> Result<Input<T>> {
        let mut guard = self.state()?;
        let state = &mut *guard;
        if let Some(id) = state.bound_id(key, Kind::Input)? {
            return state.bindings[&id]
                .as_any()
                .downcast_ref::<InputBinding<T>>()
                .map(|binding| Input {
                    id,
                    handle: binding.handle,
                })
                .ok_or_else(|| Error::TypeMismatch {
                    key: key.to_owned(),
                    source: None,
                });
        }
        let declared = self.declare(state, key, Kind::Input, || encode(key, &default))?;
        let logged_default: T = decode(key, &declared.default)?;
        let default_fp = fp_of(&declared.default);
        let (value, fp) = match &state.logged[key].value {
            Some(bytes) => (decode(key, bytes)?, fp_of(bytes)),
            None => (with_value_code(|| logged_default.clone()), default_fp),
        };
        let mut hold = state.deps.as_deref().map(DepIndex::write);
        let handle = self.runtime.create_input(value);
        let stamp = changed_at(&self.runtime, handle);
        if let Some(map) = hold.as_deref_mut() {
            map.register(
                key,
                handle.erase(),
                Kind::Input,
                Registered::Now(DepFp::Known(fp)),
                Some(stamp),
            );
        }
        drop(hold);
        let id = state.bind(
            key,
            Box::new(InputBinding {
                key: key.to_owned(),
                handle,
                default: logged_default,
                default_fp,
                fp,
                stamp,
            }),
        );
        Ok(Input { id, handle })
    }

    /// Writes `value` to `input` through the write path and returns the
    /// record's position. When this returns, the runtime reflects every
    /// record up to that position.
    ///
    /// Fails with [`Error::UnknownHandle`] for a handle another store
    /// issued, and in debug builds with [`Error::Bypassed`] when the input
    /// was written through the runtime directly; neither appends anything.
    /// An error from `append` leaves the store usable; an error
    /// after it (from `sync`, reading records other writers appended first,
    /// or applying a record) poisons the store.
    pub fn set<T: PersistValue>(&self, input: Input<T>, value: T) -> Result<Position> {
        self.write_with(|state| set_record(state, &self.runtime, input, &value))
    }

    /// Resets `key` to its logged default, or empties it if it is a
    /// collection, by appending a `Delete` through the write path; returns
    /// the record's position.
    ///
    /// The default comes from the key's first `Declare` in the log, so
    /// every process applying the `Delete`, and every restart replaying it,
    /// reaches the same state. Queries over the key, here and in other
    /// processes once they apply the record, recompute as after any write.
    /// The runtime node stays; [`Store::undeclare`] removes it from this
    /// process. A key held at its default is forgotten by the next
    /// [`Store::checkpoint`].
    ///
    /// `key` is an [`Input`], a `&`[`PersistedCollection`], or a name,
    /// which may name a key this store never declared. A `Delete` of a key
    /// the log does not know changes nothing. Errors behave as for
    /// [`Store::set`].
    pub fn delete<'k>(&self, key: impl Into<KeyRef<'k>>) -> Result<Position> {
        let key = key.into();
        self.write_with(|state| delete_record(state, &key))
    }

    /// Removes a declared key's node from this process's runtime and
    /// releases the key, so [`Store::restored_keys`] lists it again and a
    /// later [`Store::input`] or [`Store::collection`] binds it afresh.
    /// Nothing is written to the log: the key keeps its logged value or
    /// elements, other processes are unaffected, and records for it go on
    /// folding here. Afterwards every use of the handle fails with
    /// [`Error::UnknownHandle`].
    ///
    /// `key` is an [`Input`], a `&`[`PersistedCollection`], or a name. A
    /// name undeclares the key if this store declared it and, like a
    /// [`Store::delete`] of a key the log does not know, changes nothing
    /// otherwise.
    ///
    /// The nodes the key created (an input's node, a collection's version
    /// node, a keyed collection's pair and row version nodes) go as one
    /// set through the runtime's strict
    /// [`try_delete_nodes`](incr_core::Runtime::try_delete_nodes), so
    /// queries and operators over any of them must be deleted first. While any remain this
    /// fails with [`Error::HasDependents`] and changes nothing; under
    /// `Shared`, a query computing over the key concurrently either counts
    /// as a dependent or finds the node deleted. An input whose node was
    /// already deleted through the runtime just has its key released, but
    /// such a collection fails with [`Error::UnknownHandle`] and stays
    /// bound, because only its node could hand its elements back to the
    /// folded log.
    pub fn undeclare<'k>(&self, key: impl Into<KeyRef<'k>>) -> Result<()> {
        let target = key.into();
        let mut guard = self.state()?;
        let state = &mut *guard;
        state.refuse_in_scope()?;
        let Some(id) = state.binding_id(&target)? else {
            return Ok(());
        };
        let binding = &state.bindings[&id];
        let key = binding.key().to_owned();
        let held = binding.unbind(&self.runtime, &state.after)?;
        #[cfg(test)]
        UNDECLARE_PARK.with(|hook| {
            if let Some(hook) = &*hook.borrow() {
                hook();
            }
        });
        match binding.delete_nodes(&self.runtime) {
            Ok(()) | Err(DeleteError::NotAlive) => {}
            Err(DeleteError::HasDependents) => return Err(Error::HasDependents { key }),
        }
        // Unregistered only now, so readers of the index never see the key
        // unmapped while the delete can still be refused. A deleted handle
        // left mapped meanwhile matches no live node.
        if let Some(deps) = state.deps.as_deref() {
            deps.write().unregister(&key);
        }
        if let Some(binding) = state.bindings.remove(&id) {
            state.after.push(binding);
        }
        state.bound.remove(&key);
        if let Some(elements) = held.filter(|elements| !elements.is_empty()) {
            state.logged.get_or_default(key).elements = elements;
        }
        Ok(())
    }

    /// Runs `f` to collect writes, then appends them as one run with one
    /// sync and applies them, returning the last record's position. An
    /// empty batch writes nothing and returns the position last applied.
    /// The store mutex is not held while `f` runs.
    pub fn batch(&self, f: impl FnOnce(&mut Batch<'_, C, B>)) -> Result<Position> {
        let mut batch = Batch {
            store: self,
            records: Vec::new(),
            error: None,
        };
        f(&mut batch);
        let Batch { records, error, .. } = batch;
        if let Some(e) = error {
            return Err(e);
        }
        let mut state = self.state()?;
        if records.is_empty() {
            return Ok(state.applied);
        }
        self.write(&mut state, records)
    }

    /// Calls `f` with every record this store applies from now on, local
    /// and foreign alike, in position order, together with its position and
    /// [`Origin`]. Returns an id for [`Store::unsubscribe`].
    ///
    /// `f` runs after the record is applied, so the runtime already
    /// reflects it, and it runs under the store mutex. A slow subscriber
    /// therefore blocks every writer and reader of this store; hand records
    /// off to a channel when the work is not trivial. A call from `f` back
    /// into the store fails with [`Error::Reentrant`], and a panic in `f`
    /// poisons the store until it is reopened.
    ///
    /// Records this store applied before subscribing are not replayed.
    /// When a catch-up crosses a compaction, or finds one at or below its
    /// position that it never crossed, the records folded into the snapshot
    /// are reported as [`Origin::Snapshot`] changes instead. Folding
    /// everything `f` receives over the store's state at subscription, by
    /// the log's own rules (the first live `Declare` wins, `Delete` resets a
    /// key to it, [`Record::Forget`] drops the key), yields the state a
    /// restart replaying the log reaches at the store's position.
    pub fn subscribe(
        &self,
        f: impl FnMut(Position, &Record, Origin) + Send + 'static,
    ) -> Result<SubscriptionId> {
        let mut state = self.state()?;
        if state.staged {
            return Err(Error::StagedBatchOpen);
        }
        let id = SubscriptionId(state.next_subscription);
        state.next_subscription += 1;
        state.subscribers.push((id, Box::new(f)));
        Ok(id)
    }

    /// Stops the subscription `id`, so its callback is never called again
    /// and is dropped. An id already unsubscribed is ignored.
    pub fn unsubscribe(&self, id: SubscriptionId) -> Result<()> {
        let mut guard = self.state()?;
        let state = &mut *guard;
        if let Some(at) = state
            .subscribers
            .iter()
            .position(|(subscribed, _)| *subscribed == id)
        {
            state.after.push(state.subscribers.remove(at));
        }
        Ok(())
    }

    /// Keys the log knows that this store has not declared, in order,
    /// filtered to those starting with `prefix`. Applications whose keys
    /// come from data use this to find what to declare after a restart.
    ///
    /// The log is as of the last record this store applied; nothing newer
    /// is read. A compaction another writer ran at or below that position
    /// is taken into account first, so keys it forgot are not listed, as
    /// they would not be after a restart.
    pub fn restored_keys(&self, prefix: &str) -> Result<Vec<String>> {
        let mut state = self.state()?;
        self.reconcile_snapshot(&mut state)?;
        Ok(state
            .logged
            .keys_from(prefix)
            .take_while(|key| key.starts_with(prefix))
            .filter(|key| !state.bound.contains_key(*key))
            .cloned()
            .collect())
    }

    /// Applies every record appended since this store last read the log,
    /// including records from other writers, and returns the position it
    /// has applied through. Behind a compaction it first replaces its state
    /// with the snapshot's: inputs take their folded values, keys missing
    /// from the snapshot reset to their default, and collections change by
    /// only the occurrences that differ, so queries over them survive and
    /// recompute. A compaction at or below its position, of which reading
    /// shows nothing, is taken into account the same way, so the keys it
    /// forgot are forgotten here too.
    ///
    /// An error leaves the store at the last record it applied, except a
    /// failure partway through replacing from a snapshot, which poisons it.
    pub fn refresh(&self) -> Result<Position> {
        let mut state = self.state()?;
        self.catch_up(&mut state, None)?;
        Ok(state.applied)
    }

    /// Folds the log through every record this store has applied into a new
    /// snapshot, compacts the backend to it, and returns the position the
    /// snapshot covers. It catches up first, so records other writers
    /// appended before the call are folded in.
    ///
    /// Each key folds to its first `Declare`, then its latest `Set`, then
    /// one `Insert` per element occurrence; records before a `Delete` are
    /// gone. A key holding only its logged default (an input never
    /// set or deleted back, an empty collection) folds to nothing, `Declare` included, so
    /// it disappears from [`Store::restored_keys`] and the next declaration
    /// appends a fresh first-wins `Declare`, possibly with another default.
    /// A set value is kept even when it equals the default, so a reopen that
    /// declares another default still reads it; only an input declared and
    /// never set, or reset by [`Store::delete`], is forgotten.
    ///
    /// In debug builds it first fails with [`Error::Bypassed`] if any
    /// declared input was written through the runtime directly. In every
    /// build it fails with [`Error::Bypassed`], changing nothing, when a
    /// collection it is folding changes under it, which only a write
    /// around the store can do, or a keyed collection's
    /// `key_partitions` registered meanwhile; a retry succeeds.
    ///
    /// When another writer compacts past this store's position between its
    /// catch-up and its compaction, the backend rejects the stale fold with
    /// [`Error::InvalidCompaction`]; the store catches up across that
    /// compaction and folds again. Fails with [`Error::ReadOnly`] on a
    /// read-only backend. After any other compaction error the store cannot
    /// tell whether the snapshot was replaced, so it reloads its state from
    /// the backend before returning the error, and is poisoned only if that
    /// reload fails too.
    ///
    /// The backend pulls the snapshot through
    /// [`Backend::compact_streamed`], each record folded as it is pulled. A
    /// fold error, such as a value that fails to encode, fails the
    /// checkpoint with that error and changes nothing. A backend that
    /// returns `Ok` without pulling every record has committed a truncated
    /// snapshot: the checkpoint fails with [`Error::Corrupt`] and poisons
    /// the store.
    pub fn checkpoint(&self) -> Result<Position> {
        let mut state = self.state()?;
        self.checkpoint_locked(&mut state)
    }

    /// Where the store stands in its log, without reading the backend.
    pub fn stats(&self) -> Result<StoreStats> {
        let state = self.state()?;
        Ok(StoreStats {
            position: state.applied,
            snapshot_position: state.snapshot_at,
            records_since_checkpoint: state.since_checkpoint,
        })
    }

    fn checkpoint_locked(&self, state: &mut State<C>) -> Result<Position> {
        loop {
            self.catch_up(state, None)?;
            #[cfg(debug_assertions)]
            for (key, id) in &state.bound {
                state.bindings[id].check_bypass(
                    &self.runtime,
                    state.logged.get(key),
                    &state.after,
                )?;
            }
            let upto = state.applied;
            let mut fold = SnapshotFold::new(state, self.runtime.yielder_under_lock());

            let result = self.backend.compact_streamed(upto, &mut fold);
            let pulled = fold.finish(result.is_ok());
            let result = match (result, pulled) {
                (Ok(()), Pulled::Whole(forgotten)) => Ok(forgotten),
                (Err(e), Pulled::Failed) => return Err(e),
                (Ok(()), Pulled::Truncated | Pulled::Failed) => {
                    state.poison();
                    return Err(Error::Corrupt(
                        "the backend's compact_streamed returned Ok after a failed pull or before \
                         the last record, so its snapshot may be truncated",
                    ));
                }
                (Err(e), _) => Err(e),
            };
            match result {
                Ok(forgotten) => {
                    for key in forgotten {
                        state.logged.remove(&key);
                    }
                    state.since_checkpoint = 0;
                    state.snapshot_at = upto;
                    // The new snapshot is this store's own fold, so it needs
                    // no reconciling. It is the one right after the snapshot
                    // the store accounted for unless another writer compacted
                    // in between; a failed or mismatched lookup leaves it
                    // unaccounted, and the next catch-up reconciles with it,
                    // which is correct, only slower.
                    let own = SnapshotId {
                        position: upto,
                        generation: state.snapshot.map_or(0, |id| id.generation) + 1,
                    };
                    if matches!(self.backend.snapshot_id(), Ok(Some(id)) if id == own) {
                        state.snapshot = Some(own);
                    }
                    return Ok(upto);
                }
                Err(Error::InvalidCompaction { snapshot_at, .. }) if snapshot_at > upto => {
                    self.catch_up(state, None)?;
                    if state.applied < snapshot_at {
                        return Err(Error::Corrupt(
                            "backend rejected a compaction below its snapshot but reads continue before it",
                        ));
                    }
                }
                Err(e @ (Error::InvalidCompaction { .. } | Error::ReadOnly)) => return Err(e),
                Err(e) => {
                    if self.resync(state).is_err() {
                        state.poison();
                    }
                    return Err(e);
                }
            }
        }
    }

    /// Rebuilds the folded log and every bound node from the backend's
    /// current snapshot and the records after it, for when the store cannot
    /// tell what a failed backend call changed.
    fn resync(&self, state: &mut State<C>) -> Result<()> {
        let (id, records) = match self.snapshot_with_id()? {
            Some((id, records)) => (Some(id), records),
            None => (None, Vec::new()),
        };
        let at = id.map_or(Position::ZERO, |id| id.position);
        self.replace_with(state, at, records)?;
        state.snapshot = id;
        state.snapshot_at = at;
        self.catch_up(state, None)
    }

    /// Returns the key's live `Declare`, first appending one of `kind` with
    /// the default `default` produces when the log has none. Fails with
    /// [`Error::KindMismatch`] when the live `Declare` is of the other kind.
    fn declare(
        &self,
        state: &mut State<C>,
        key: &str,
        kind: Kind,
        default: impl Fn() -> Result<Vec<u8>>,
    ) -> Result<Declared> {
        // A compaction at or below `applied` may have forgotten the key,
        // which a restart would then declare afresh.
        self.reconcile_snapshot(state)?;
        // A loop because a concurrent compaction may fold away a Declare
        // whose key still held its default, forgetting the key again.
        let mut caught_up = false;
        let declared = loop {
            if let Some(declared) = state.logged.get(key).and_then(|k| k.declared.as_ref()) {
                break declared.clone();
            }
            // Another writer may have declared the key since this store last
            // read the log; appending without looking would add a redundant
            // Declare, and fail outright on a read-only store.
            if !caught_up {
                self.catch_up(state, None)?;
                caught_up = true;
                continue;
            }
            let record = Record::Declare {
                key: key.to_owned(),
                kind,
                default: default()?,
            };
            self.write(state, vec![record])?;
        };
        if declared.kind != kind {
            return Err(Error::KindMismatch {
                key: key.to_owned(),
                declared: declared.kind,
            });
        }
        Ok(declared)
    }

    fn write_with(&self, record: impl FnOnce(&State<C>) -> Result<Record>) -> Result<Position> {
        let mut state = self.state()?;
        let record = record(&state)?;
        self.write(&mut state, vec![record])
    }

    /// The only place the store mutex is locked. The checks run in release
    /// builds and before the lock: this thread may hold the mutex already,
    /// the holder may be waiting on a claim the calling compute or value
    /// code's caller holds, and value code may be running on the thread
    /// that holds the mutex. Re-entry is checked first, so a `Drop` the
    /// engine runs as value code under the mutex gets the error too, and a
    /// call from value code returns [`Error::Reentrant`] instead of
    /// panicking.
    fn drained(&self) -> Drained<'_> {
        let drained = Drained::new(self.options.value_scope.as_ref());
        #[cfg(feature = "yield-hook")]
        let drained = drained.ticking(self.runtime.yield_hook());
        drained
    }

    fn state(&self) -> Result<Locked<'_, C>> {
        if self.holder.is_current() {
            return Err(Error::Reentrant);
        }
        assert!(
            !self.runtime.frame_active(),
            "incr-persist: Store methods must not be called from inside a query's compute",
        );
        if in_value_code() {
            return Err(Error::Reentrant);
        }
        let routing = self.runtime.hold_routing();
        // A panic while the mutex was held may have left the runtime and
        // the folded log out of step, which is the poisoned condition too.
        let guard = Ordered::try_take(LockKind::Store, || {
            self.state.lock().map_err(|_| Error::Poisoned)
        })?;
        if guard.poisoned {
            return Err(Error::Poisoned);
        }
        Ok(Locked {
            _holder: self.holder.mark(),
            guard,
            _routing: routing,
            _scope: self.enter_value_scope(),
            drained: self.drained(),
        })
    }

    /// Poisons the store from a `Drop`. On a thread already inside a call of
    /// this store, that call poisons it before releasing the mutex, since
    /// locking here would deadlock.
    fn poison_from_drop(&self) {
        if self.holder.is_current() {
            self.holder.request_poison();
        } else {
            self.state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .poison();
        }
    }

    /// The write path: append, sync per the durability option, then apply
    /// everything up to the appended records. Any failure after the append
    /// poisons the store, because a retry would append the records again
    /// and a collection insert is not idempotent.
    fn write(&self, state: &mut State<C>, records: Vec<Record>) -> Result<Position> {
        state.refuse_in_scope()?;
        self.checkpoint_if_due(state)?;
        let last = self.backend.append(&records)?;
        let landed = (|| {
            if self.options.durability == Durability::Sync {
                self.backend.sync(last)?;
            }
            self.apply_appended(state, last, records)
        })();
        match landed {
            Ok(()) => Ok(last),
            Err(e) => {
                state.poison();
                Err(e)
            }
        }
    }

    /// Applies `record` as a staged write and keeps it in `run`. A failed
    /// apply poisons the store, since earlier staged writes are already in
    /// the runtime.
    fn apply_staged(
        &self,
        state: &mut State<C>,
        record: Record,
        run: &mut Vec<Record>,
    ) -> Result<()> {
        if let Err(e) = self.apply(state, record.clone()) {
            state.poison();
            return Err(e);
        }
        run.push(record);
        Ok(())
    }

    /// Runs [`Store::checkpoint`] when [`Options::auto_checkpoint_records`]
    /// is due, before anything is appended.
    fn checkpoint_if_due(&self, state: &mut State<C>) -> Result<()> {
        match self.options.auto_checkpoint_records {
            Some(threshold) if state.since_checkpoint >= threshold => {
                self.checkpoint_locked(state).map(drop)
            }
            _ => Ok(()),
        }
    }

    /// Applies everything after `state.applied` through `last`, where the
    /// append of `records` ended.
    ///
    /// Appends are contiguous, so the run is exactly the positions ending at
    /// `last`. When it starts right after `applied`, no other writer's
    /// record lies before it and the records are applied as held, without
    /// reading them back. Otherwise the log is read from `applied`, which
    /// picks up the foreign records and crosses any compaction another
    /// writer ran past `applied`. A compaction at `applied`, or over the
    /// run after it landed, folds only records this store applies anyway,
    /// so it is not crossed here, just as `read_from` at the snapshot
    /// position reports none; the next catch-up reconciles with whatever it
    /// forgot, keeping a backend query off this path.
    fn apply_appended(
        &self,
        state: &mut State<C>,
        last: Position,
        records: Vec<Record>,
    ) -> Result<()> {
        let len = records.len() as u64;
        let first = Position(last.0 + 1 - len);
        if state.applied.0 + len != last.0 {
            return self.catch_up(state, Some(first..=last));
        }
        let yielder = self.runtime.yielder_under_lock();
        for (position, record) in (first.0..).map(Position).zip(records) {
            yielder.tick();
            self.apply_at(state, position, record, Origin::Local)?;
        }
        Ok(())
    }

    /// Applies records after `state.applied` until reaching the end of
    /// `own`, the run this store just appended, or until one read comes
    /// back when `own` is `None`, notifying subscribers after each. A read
    /// behind the snapshot replaces the folded log and every bound node
    /// with the snapshot's content first.
    ///
    /// It reconciles with a snapshot at or below `applied` before reading,
    /// so that records after it, such as a fresh `Declare` of a key it
    /// forgot, apply to the state a restart holds, and again after reading
    /// when it advanced, because a handle may read through a compaction
    /// record by record instead of reporting it.
    fn catch_up(&self, state: &mut State<C>, own: Option<RangeInclusive<Position>>) -> Result<()> {
        state.refuse_in_scope()?;
        self.reconcile_snapshot(state)?;
        let start = state.applied;
        let mut read = Vec::new();
        let yielder = self.runtime.yielder_under_lock();
        loop {
            read.clear();
            match self.backend.read_from(state.applied, &mut read)? {
                ReadOutcome::Compacted { .. } => self.replace_from_snapshot(state)?,
                ReadOutcome::Records => {
                    let progressed = !read.is_empty();
                    for (position, record) in read.drain(..) {
                        yielder.tick();
                        let origin = match &own {
                            Some(run) if run.contains(&position) => Origin::Local,
                            _ => Origin::Foreign,
                        };
                        self.apply_at(state, position, record, origin)?;
                    }
                    match own.as_ref().map(|run| *run.end()) {
                        Some(target) if state.applied < target => {
                            if !progressed {
                                return Err(Error::Corrupt(
                                    "backend read stopped short of a position its append returned",
                                ));
                            }
                        }
                        _ => break,
                    }
                }
            }
        }
        if state.applied != start {
            self.reconcile_snapshot(state)?;
        }
        Ok(())
    }

    /// Applies `record`, the log's record at `position`, and reports it to
    /// subscribers as coming from `origin`.
    fn apply_at(
        &self,
        state: &mut State<C>,
        position: Position,
        record: Record,
        origin: Origin,
    ) -> Result<()> {
        // Cloned only for subscribers, because applying moves the record's
        // bytes into the folded log.
        let notice = (!state.subscribers.is_empty()).then(|| record.clone());
        self.apply(state, record)?;
        state.applied = position;
        state.since_checkpoint += 1;
        if let Some(record) = notice {
            state.notify(position, &record, origin);
        }
        Ok(())
    }

    /// Brings the folded log and every bound node to the state a restart
    /// reaches at `state.applied`, when the backend holds a snapshot at or
    /// below that position which this store has not accounted for.
    ///
    /// Such a snapshot folds only records the store already applied, so
    /// reading from `applied` reports nothing of it, yet it may have
    /// forgotten keys: a key at its logged default folds to nothing,
    /// `Declare` included. The store would go on listing such a key, bind it
    /// with the dropped default instead of declaring it afresh, and reject
    /// the fresh `Declare` every other process adopts. So the state is
    /// rebuilt from the snapshot and the log records after it through
    /// `applied`, and the changes are reported to subscribers at `applied`
    /// as [`Origin::Snapshot`] records. A snapshot beyond `applied` is left
    /// to `catch_up`, which crosses it.
    fn reconcile_snapshot(&self, state: &mut State<C>) -> Result<()> {
        state.refuse_in_scope()?;
        loop {
            let Some(id) = self
                .backend
                .snapshot_id()?
                .filter(|id| Some(*id) != state.snapshot && id.position <= state.applied)
            else {
                return Ok(());
            };
            let loaded = self.backend.load_snapshot()?;
            let mut later = Vec::new();
            let outcome = self.backend.read_from(id.position, &mut later)?;
            // The same id on both sides of the load and the read means both
            // saw this snapshot, not one a racing compaction put in its place.
            if self.backend.snapshot_id()? != Some(id) {
                continue;
            }
            let snapshot = match (loaded, outcome) {
                (Some((at, snapshot)), ReadOutcome::Records) if at == id.position => snapshot,
                _ => {
                    return Err(Error::Corrupt(
                        "backend's snapshot disagrees with the snapshot id it reports",
                    ))
                }
            };
            let applied = state.applied;
            if id.position < applied && later.last().is_none_or(|(last, _)| *last < applied) {
                return Err(Error::Corrupt(
                    "backend read stopped short of a position already applied",
                ));
            }
            let later: Vec<Record> = later
                .into_iter()
                .take_while(|(position, _)| *position <= applied)
                .map(|(_, record)| record)
                .collect();
            let since = later.len() as u64;
            self.replace_with(state, applied, snapshot.into_iter().chain(later))?;
            state.snapshot = Some(id);
            state.snapshot_at = id.position;
            state.since_checkpoint = since;
            return Ok(());
        }
    }

    /// The backend's snapshot and its id, read between two equal ids so that
    /// a compaction racing the load cannot pair one snapshot's records with
    /// another's id.
    fn snapshot_with_id(&self) -> Result<Option<(SnapshotId, Vec<Record>)>> {
        loop {
            let id = self.backend.snapshot_id()?;
            let loaded = self.backend.load_snapshot()?;
            if self.backend.snapshot_id()? != id {
                continue;
            }
            return match (id, loaded) {
                (None, None) => Ok(None),
                (Some(id), Some((at, records))) if at == id.position => Ok(Some((id, records))),
                _ => Err(Error::Corrupt(
                    "backend's snapshot disagrees with the snapshot id it reports",
                )),
            };
        }
    }

    fn apply(&self, state: &mut State<C>, record: Record) -> Result<()> {
        if let Some(&id) = state.bound.get(record.key()) {
            let binding = bound_binding(&mut state.bindings, id);
            let deps = state.deps.as_deref();
            binding.apply(
                &self.runtime,
                &record,
                state.logged.get(record.key()),
                deps,
                &state.after,
            )?;
            // A bound collection holds its elements by decoded value; folding
            // them by bytes as well would let the two disagree.
            if binding.kind() == Kind::Collection && is_element_change(&record) {
                return Ok(());
            }
        }
        fold(&mut state.logged, record);
        Ok(())
    }

    fn replace_from_snapshot(&self, state: &mut State<C>) -> Result<()> {
        let (id, records) = self.snapshot_with_id()?.ok_or(Error::Corrupt(
            "backend reported compaction but has no snapshot",
        ))?;
        if id.position <= state.applied {
            return Err(Error::Corrupt(
                "backend reported compaction below the position already applied",
            ));
        }
        self.replace_with(state, id.position, records)?;
        state.snapshot = Some(id);
        state.snapshot_at = id.position;
        Ok(())
    }

    /// Replaces the folded log and every bound node with the fold of
    /// `records`, the log's content through `at`, and reports the changes
    /// to subscribers as [`Origin::Snapshot`] records at `at`. A binding that
    /// fails after others were replaced leaves the store between two
    /// states, so that failure poisons it.
    fn replace_with(
        &self,
        state: &mut State<C>,
        at: Position,
        records: impl IntoIterator<Item = Record>,
    ) -> Result<()> {
        let mut logged = LoggedKeys::new();
        let yielder = self.runtime.yielder_under_lock();
        for record in records {
            yielder.tick();
            fold(&mut logged, record);
        }
        let notify = !state.subscribers.is_empty();
        let keys: Vec<String> = if notify {
            let all: BTreeSet<&String> = state
                .logged
                .keys()
                .chain(logged.keys())
                .chain(state.bound.keys())
                .collect();
            all.into_iter().cloned().collect()
        } else {
            state.bound.keys().cloned().collect()
        };
        let mut changes = Vec::new();
        for key in &keys {
            yielder.tick();
            let sink = notify.then_some(&mut changes);
            if let Err(e) = self.replace_key(state, key, logged.get_mut(key), sink) {
                state.poison();
                return Err(e);
            }
        }
        state.logged = logged;
        state.applied = at;
        state.since_checkpoint = 0;
        for record in &changes {
            state.notify(at, record, Origin::Snapshot);
        }
        Ok(())
    }

    /// Moves `key`'s bound node, if any, from the key's folded state to
    /// `new`, pushing onto `changes`, when given, the records that take a
    /// subscriber there. The caller stores `new` as the folded state after.
    /// A failing binding has changed nothing.
    fn replace_key(
        &self,
        state: &mut State<C>,
        key: &str,
        new: Option<&mut LoggedKey>,
        mut changes: Option<&mut Vec<Record>>,
    ) -> Result<()> {
        let id = state.bound.get(key).copied();
        let mut rebuilt = false;
        if let Some(changes) = changes.as_deref_mut() {
            let bound_collection =
                id.is_some_and(|id| state.bindings[&id].kind() == Kind::Collection);
            rebuilt = changes_between(
                key,
                state.logged.get(key),
                new.as_deref(),
                !bound_collection,
                changes,
            );
        }
        if let Some(id) = id {
            let changes = if rebuilt { None } else { changes };
            let deps = state.deps.as_deref();
            bound_binding(&mut state.bindings, id).replace(
                &self.runtime,
                new,
                changes,
                deps,
                &state.after,
            )?;
        }
        Ok(())
    }
}

fn bound_binding<C: Cells>(
    bindings: &mut HashMap<u32, Box<dyn Binding<C>>>,
    id: u32,
) -> &mut dyn Binding<C> {
    bindings
        .get_mut(&id)
        .expect("every id in `bound` has a binding")
        .as_mut()
}

/// The binding `input` was issued for, or [`Error::UnknownHandle`] when this
/// store did not issue it or has since undeclared it.
fn input_binding<C: Cells, T: Value>(
    state: &State<C>,
    input: Input<T>,
) -> Result<&InputBinding<T>> {
    state
        .bindings
        .get(&input.id)
        .and_then(|binding| binding.as_any().downcast_ref::<InputBinding<T>>())
        .filter(|binding| binding.handle == input.handle)
        .ok_or(Error::UnknownHandle)
}

/// The `Set` record writing `value` to `input`. In debug builds it first
/// checks that nothing wrote the input around the store.
fn set_record<C: Cells, T: PersistValue>(
    state: &State<C>,
    rt: &Runtime<C>,
    input: Input<T>,
    value: &T,
) -> Result<Record> {
    let binding = input_binding(state, input)?;
    #[cfg(debug_assertions)]
    Binding::<C>::check_bypass(binding, rt, state.logged.get(&binding.key), &state.after)?;
    #[cfg(not(debug_assertions))]
    let _ = rt;
    Ok(Record::Set {
        key: binding.key.clone(),
        value: encode(&binding.key, value)?,
    })
}

fn delete_record<C: Cells>(state: &State<C>, key: &KeyRef<'_>) -> Result<Record> {
    Ok(Record::Delete {
        key: state.key_of(key)?,
    })
}

/// The snapshot of everything a store applied so far, folded key by key in
/// key order as the backend pulls it, gathering the keys it forgets: those
/// the log knows but that fold to nothing. Stops after the first `Err`.
struct SnapshotFold<'a, C: Cells> {
    state: &'a State<C>,
    keys: std::collections::btree_set::IntoIter<&'a str>,
    /// The key being folded, its records, and whether any was pulled.
    current: Option<(&'a str, Folded<'a>, bool)>,
    forgotten: Vec<String>,
    failed: bool,
    yielder: Yielder<'a>,
}

/// How far a backend pulled a [`SnapshotFold`].
enum Pulled {
    /// To the end, with the keys the snapshot forgets.
    Whole(Vec<String>),
    /// Not to the end, though nothing failed.
    Truncated,
    /// Up to an `Err`.
    Failed,
}

impl<'a, C: Cells> SnapshotFold<'a, C> {
    fn new(state: &'a State<C>, yielder: Yielder<'a>) -> Self {
        let keys: BTreeSet<&str> = state
            .logged
            .keys()
            .chain(state.bound.keys())
            .map(String::as_str)
            .collect();
        Self {
            state,
            keys: keys.into_iter(),
            current: None,
            forgotten: Vec::new(),
            failed: false,
            yielder,
        }
    }

    /// Whether the backend pulled every record. After an `Ok` from the
    /// backend, pulls once more to tell a whole snapshot from a truncated
    /// one.
    fn finish(mut self, committed: bool) -> Pulled {
        if self.failed {
            return Pulled::Failed;
        }
        if committed && self.next().is_some() {
            return Pulled::Truncated;
        }
        if self.current.is_some() || self.keys.len() > 0 {
            return Pulled::Truncated;
        }
        Pulled::Whole(self.forgotten)
    }

    fn fold_key(&self, key: &'a str) -> Result<Folded<'a>> {
        let logged = self.state.logged.get(key);
        Ok(match self.state.bound.get(key) {
            Some(&id) => self.state.bindings[&id].fold(logged, &self.yielder)?,
            None => match logged {
                Some(logged) if !logged.at_default() => Box::new(logged.iter_records(key).map(Ok)),
                _ => Box::new(std::iter::empty()),
            },
        })
    }
}

impl<C: Cells> Iterator for SnapshotFold<'_, C> {
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Result<Record>> {
        self.yielder.tick();
        while !self.failed {
            if let Some((key, records, pulled)) = &mut self.current {
                match records.next() {
                    Some(Ok(record)) => {
                        *pulled = true;
                        return Some(Ok(record));
                    }
                    Some(Err(e)) => {
                        self.failed = true;
                        return Some(Err(e));
                    }
                    None => {
                        if !*pulled && self.state.logged.contains_key(key) {
                            self.forgotten.push((*key).to_owned());
                        }
                        self.current = None;
                    }
                }
            }
            let key = self.keys.next()?;
            match self.fold_key(key) {
                Ok(records) => self.current = Some((key, records, false)),
                Err(e) => {
                    self.failed = true;
                    return Some(Err(e));
                }
            }
        }
        None
    }
}
