use std::any::Any;
use std::collections::{BTreeSet, HashMap};
use std::ops::{Bound, RangeInclusive};
use std::sync::{Mutex, MutexGuard};

use incr_core::{Cells, DeleteError, Incr, Runtime, Value};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::{Backend, Error, Kind, Position, ReadOutcome, Record, Result};

use binding::{decode, encode, Binding, InputBinding};
use logged::{changes_between, fold, is_element_change, Declared, LoggedKeys};

pub use collection::PersistedCollection;

mod binding;
mod collection;
mod logged;

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
    /// unchanged. `None`, the default, compacts only on explicit calls.
    pub auto_checkpoint_records: Option<u64>,
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
    /// state to the snapshot's: a `Declare` for a new declaration, a `Set`
    /// for a new value, a `Delete` for a value reset to its default, and a
    /// `Remove` or `Insert` per differing element occurrence. A collection
    /// this store did not declare whose elements cannot be compared by
    /// bytes is reported as a `Delete` followed by its full contents.
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
/// they create and set runtime inputs.
///
/// A `Store<Local, _>` is confined to its thread like its runtime:
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<incr_persist::Store<incr_persist::Local, incr_persist::MemoryBackend>>();
/// ```
pub struct Store<C: Cells, B: Backend> {
    runtime: Runtime<C>,
    backend: B,
    options: Options,
    state: Mutex<State<C>>,
}

struct State<C: Cells> {
    applied: Position,
    poisoned: bool,
    since_checkpoint: u64,
    logged: LoggedKeys,
    bound: HashMap<String, u32>,
    /// Bindings by id. Ids are never reused, so a handle whose key was
    /// undeclared finds nothing rather than a later binding.
    bindings: HashMap<u32, Box<dyn Binding<C>>>,
    next_id: u32,
    subscribers: Vec<(SubscriptionId, Subscriber)>,
    next_subscription: u64,
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
        let store = Store {
            runtime: Runtime::new(),
            backend,
            options,
            state: Mutex::new(State {
                applied: Position::ZERO,
                poisoned: false,
                since_checkpoint: 0,
                logged: LoggedKeys::new(),
                bound: HashMap::new(),
                bindings: HashMap::new(),
                next_id: 0,
                subscribers: Vec::new(),
                next_subscription: 0,
            }),
        };
        {
            let mut state = store.state()?;
            store.catch_up(&mut state, None)?;
        }
        Ok(store)
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
    /// log order
    /// wins: its default, not necessarily `default`, is what the key resets
    /// to, in every process. The input starts at the key's latest logged
    /// value, else at that logged default. Declaring a key this store
    /// already declared returns the same handle.
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
        let declared = self.declare(state, key, Kind::Input, || encode(&default))?;
        let logged_default: T = decode(key, &declared.default)?;
        let value = match &state.logged[key].value {
            Some(bytes) => decode(key, bytes)?,
            None => logged_default.clone(),
        };
        let handle = self.runtime.create_input(value);
        let id = state.bind(
            key,
            Box::new(InputBinding {
                key: key.to_owned(),
                handle,
                default: logged_default,
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
    /// The node (a collection's version node, the only node it creates) goes
    /// through the runtime's strict
    /// [`try_delete_node`](incr_core::Runtime::try_delete_node), so queries
    /// and operators over it must be deleted first. While any remain this
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
        let Some(id) = state.binding_id(&target)? else {
            return Ok(());
        };
        let binding = &state.bindings[&id];
        let key = binding.key().to_owned();
        let held = binding.unbind(&self.runtime)?;
        match self.runtime.try_delete_node(binding.node()) {
            Ok(()) | Err(DeleteError::NotAlive) => {}
            Err(DeleteError::HasDependents) => return Err(Error::HasDependents { key }),
        }
        state.bindings.remove(&id);
        state.bound.remove(&key);
        if let Some(elements) = held.filter(|elements| !elements.is_empty()) {
            state.logged.entry(key).or_default().elements = elements;
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
    /// off to a channel when the work is not trivial. `f` must not call
    /// back into the store, which would deadlock, and a panic in `f`
    /// poisons the store until it is reopened.
    ///
    /// Records this store applied before subscribing are not replayed.
    /// When a catch-up crosses a compaction, the records folded into the
    /// snapshot are reported as [`Origin::Snapshot`] changes instead.
    pub fn subscribe(
        &self,
        f: impl FnMut(Position, &Record, Origin) + Send + 'static,
    ) -> Result<SubscriptionId> {
        let mut state = self.state()?;
        let id = SubscriptionId(state.next_subscription);
        state.next_subscription += 1;
        state.subscribers.push((id, Box::new(f)));
        Ok(id)
    }

    /// Stops the subscription `id`, so its callback is never called again
    /// and is dropped. An id already unsubscribed is ignored.
    pub fn unsubscribe(&self, id: SubscriptionId) -> Result<()> {
        let mut state = self.state()?;
        state
            .subscribers
            .retain(|(subscribed, _)| *subscribed != id);
        Ok(())
    }

    /// Keys the log knows that this store has not declared, in order,
    /// filtered to those starting with `prefix`. Applications whose keys
    /// come from data use this to find what to declare after a restart.
    pub fn restored_keys(&self, prefix: &str) -> Result<Vec<String>> {
        let state = self.state()?;
        Ok(state
            .logged
            .range::<str, _>((Bound::Included(prefix), Bound::Unbounded))
            .map(|(key, _)| key)
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
    /// recompute.
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
    /// gone. A key holding only its logged default (an input at its default
    /// value, an empty collection) folds to nothing, `Declare` included, so
    /// it disappears from [`Store::restored_keys`] and the next declaration
    /// appends a fresh first-wins `Declare`, possibly with another default.
    /// Declared keys compare by decoded value; keys this store has not
    /// declared compare by bytes.
    ///
    /// In debug builds it first fails with [`Error::Bypassed`] if any
    /// declared input was written through the runtime directly.
    ///
    /// When another writer compacts past this store's position between its
    /// catch-up and its compaction, the backend rejects the stale fold with
    /// [`Error::InvalidCompaction`]; the store catches up across that
    /// compaction and folds again. Fails with [`Error::ReadOnly`] on a
    /// read-only backend. After any other compaction error the store cannot
    /// tell whether the snapshot was replaced, so it reloads its state from
    /// the backend before returning the error, and is poisoned only if that
    /// reload fails too.
    pub fn checkpoint(&self) -> Result<Position> {
        let mut state = self.state()?;
        self.checkpoint_locked(&mut state)
    }

    fn checkpoint_locked(&self, state: &mut State<C>) -> Result<Position> {
        loop {
            self.catch_up(state, None)?;
            #[cfg(debug_assertions)]
            for (key, id) in &state.bound {
                state.bindings[id].check_bypass(&self.runtime, state.logged.get(key))?;
            }
            let upto = state.applied;
            let (folded, forgotten) = self.fold_snapshot(state)?;
            match self.backend.compact(upto, folded) {
                Ok(()) => {
                    for key in forgotten {
                        state.logged.remove(&key);
                    }
                    state.since_checkpoint = 0;
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
                        state.poisoned = true;
                    }
                    return Err(e);
                }
            }
        }
    }

    /// The snapshot of everything applied so far, and the keys it forgets:
    /// those the log knows but that fold to nothing.
    fn fold_snapshot(&self, state: &State<C>) -> Result<(Vec<Record>, Vec<String>)> {
        let keys: BTreeSet<&str> = state
            .logged
            .keys()
            .chain(state.bound.keys())
            .map(String::as_str)
            .collect();
        let mut folded = Vec::new();
        let mut forgotten = Vec::new();
        for key in keys {
            let logged = state.logged.get(key);
            let records = match state.bound.get(key) {
                Some(&id) => state.bindings[&id].fold(&self.runtime, logged)?,
                None => match logged {
                    Some(logged) if !logged.at_default() => logged.records(key),
                    _ => Vec::new(),
                },
            };
            if records.is_empty() {
                if logged.is_some() {
                    forgotten.push(key.to_owned());
                }
            } else {
                folded.extend(records);
            }
        }
        Ok((folded, forgotten))
    }

    /// Rebuilds the folded log and every bound node from the backend's
    /// current snapshot and the records after it, for when the store cannot
    /// tell what a failed backend call changed.
    fn resync(&self, state: &mut State<C>) -> Result<()> {
        let (at, records) = self
            .backend
            .load_snapshot()?
            .unwrap_or((Position::ZERO, Vec::new()));
        self.replace_with(state, at, records)?;
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

    fn state(&self) -> Result<MutexGuard<'_, State<C>>> {
        // A panic while the mutex was held may have left the runtime and
        // the folded log out of step, which is the poisoned condition too.
        let state = self.state.lock().map_err(|_| Error::Poisoned)?;
        if state.poisoned {
            return Err(Error::Poisoned);
        }
        Ok(state)
    }

    /// The write path: append, sync per the durability option, then apply
    /// everything up to the appended records. Any failure after the append
    /// poisons the store, because a retry would append the records again
    /// and a collection insert is not idempotent.
    fn write(&self, state: &mut State<C>, records: Vec<Record>) -> Result<Position> {
        if let Some(threshold) = self.options.auto_checkpoint_records {
            if state.since_checkpoint >= threshold {
                self.checkpoint_locked(state)?;
            }
        }
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
                state.poisoned = true;
                Err(e)
            }
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
    /// so it is not crossed, just as `read_from` at the snapshot position
    /// reports none.
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
        for (position, record) in (first.0..).map(Position).zip(records) {
            self.apply_at(state, position, record, Origin::Local)?;
        }
        Ok(())
    }

    /// Applies records after `state.applied` until reaching the end of
    /// `own`, the run this store just appended, or until one read comes
    /// back when `own` is `None`, notifying subscribers after each. A read
    /// behind the snapshot replaces the folded log and every bound node
    /// with the snapshot's content first.
    fn catch_up(&self, state: &mut State<C>, own: Option<RangeInclusive<Position>>) -> Result<()> {
        let mut read = Vec::new();
        loop {
            read.clear();
            match self.backend.read_from(state.applied, &mut read)? {
                ReadOutcome::Compacted { .. } => self.replace_from_snapshot(state)?,
                ReadOutcome::Records => {
                    let progressed = !read.is_empty();
                    for (position, record) in read.drain(..) {
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
                        _ => return Ok(()),
                    }
                }
            }
        }
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

    fn apply(&self, state: &mut State<C>, record: Record) -> Result<()> {
        if let Some(&id) = state.bound.get(record.key()) {
            let binding = bound_binding(&mut state.bindings, id);
            binding.apply(&self.runtime, &record, state.logged.get(record.key()))?;
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
        let (at, records) = self.backend.load_snapshot()?.ok_or(Error::Corrupt(
            "backend reported compaction but has no snapshot",
        ))?;
        if at <= state.applied {
            return Err(Error::Corrupt(
                "backend reported compaction below the position already applied",
            ));
        }
        self.replace_with(state, at, records)
    }

    /// Replaces the folded log and every bound node with the fold of
    /// `records`, a snapshot covering through `at`, and reports the changes
    /// to subscribers as [`Origin::Snapshot`] records. A binding that fails
    /// after others were replaced leaves the store between two states, so
    /// that failure poisons it.
    fn replace_with(&self, state: &mut State<C>, at: Position, records: Vec<Record>) -> Result<()> {
        let mut logged = LoggedKeys::new();
        for record in records {
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
            let id = state.bound.get(key).copied();
            if notify {
                let bound_collection =
                    id.is_some_and(|id| state.bindings[&id].kind() == Kind::Collection);
                changes_between(
                    key,
                    state.logged.get(key),
                    logged.get(key),
                    !bound_collection,
                    &mut changes,
                );
            }
            if let Some(id) = id {
                let binding = bound_binding(&mut state.bindings, id);
                let sink = notify.then_some(&mut changes);
                if let Err(e) = binding.replace(&self.runtime, logged.get_mut(key), sink) {
                    state.poisoned = true;
                    return Err(e);
                }
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
    Binding::<C>::check_bypass(binding, rt, state.logged.get(&binding.key))?;
    #[cfg(not(debug_assertions))]
    let _ = rt;
    Ok(Record::Set {
        key: binding.key.clone(),
        value: encode(value)?,
    })
}

fn delete_record<C: Cells>(state: &State<C>, key: &KeyRef<'_>) -> Result<Record> {
    Ok(Record::Delete {
        key: state.key_of(key)?,
    })
}
