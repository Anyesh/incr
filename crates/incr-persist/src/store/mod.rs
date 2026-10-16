use std::collections::HashMap;
use std::ops::Bound;
use std::sync::{Mutex, MutexGuard};

use incr_core::{Cells, Incr, Runtime, Value};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::{Backend, Error, Kind, Position, ReadOutcome, Record, Result};

use binding::{decode, encode, Binding, InputBinding};
use logged::{fold, is_element_change, Declared, LoggedKeys};

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
}

/// A persisted input: a key in the log bound to an input node of the
/// store's runtime.
pub struct Input<T: 'static> {
    id: u32,
    handle: Incr<T>,
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
    logged: LoggedKeys,
    bound: HashMap<String, u32>,
    bindings: Vec<Box<dyn Binding<C>>>,
}

impl<T: 'static> Input<T> {
    /// The runtime node holding the input's value, for reads and queries.
    /// Write through [`Store::set`], not through this handle.
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

impl<C: Cells, B: Backend> Batch<'_, C, B> {
    /// Adds a `Set` of `input` to the batch. A failure (an unknown handle, a
    /// poisoned store, a value that fails to serialize) is kept and returned
    /// by [`Store::batch`], which then appends nothing.
    pub fn set<T: PersistValue>(&mut self, input: Input<T>, value: T) -> &mut Self {
        self.push(|state| set_record(state, input, &value))
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
        let declared = self.bindings[id as usize].kind();
        if declared != kind {
            return Err(Error::KindMismatch {
                key: key.to_owned(),
                declared,
            });
        }
        Ok(Some(id))
    }

    fn bind(&mut self, key: &str, binding: Box<dyn Binding<C>>) -> u32 {
        let id = self.bindings.len() as u32;
        self.bindings.push(binding);
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
                logged: LoggedKeys::new(),
                bound: HashMap::new(),
                bindings: Vec::new(),
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
    /// If the log has no live `Declare` for `key`, appends one carrying
    /// `default` through the write path. The first `Declare` in log order
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
            return state.bindings[id as usize]
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
    /// issued. An error from `append` leaves the store usable; an error
    /// after it (from `sync`, reading the log back, or applying a record)
    /// poisons the store.
    pub fn set<T: PersistValue>(&self, input: Input<T>, value: T) -> Result<Position> {
        self.write_with(|state| set_record(state, input, &value))
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
        self.write(&mut state, &records)
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
        let declared = loop {
            if let Some(declared) = state.logged.get(key).and_then(|k| k.declared.as_ref()) {
                break declared.clone();
            }
            let record = Record::Declare {
                key: key.to_owned(),
                kind,
                default: default()?,
            };
            self.write(state, &[record])?;
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
        self.write(&mut state, &[record])
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
    fn write(&self, state: &mut State<C>, records: &[Record]) -> Result<Position> {
        let last = self.backend.append(records)?;
        let landed = (|| {
            if self.options.durability == Durability::Sync {
                self.backend.sync(last)?;
            }
            self.catch_up(state, Some(last))
        })();
        match landed {
            Ok(()) => Ok(last),
            Err(e) => {
                state.poisoned = true;
                Err(e)
            }
        }
    }

    /// Applies records after `state.applied` until reaching `until`, or
    /// until one read comes back when `until` is `None`. A read behind the
    /// snapshot replaces the folded log and every bound node with the
    /// snapshot's content first.
    fn catch_up(&self, state: &mut State<C>, until: Option<Position>) -> Result<()> {
        let mut read = Vec::new();
        loop {
            read.clear();
            match self.backend.read_from(state.applied, &mut read)? {
                ReadOutcome::Compacted { .. } => self.replace_from_snapshot(state)?,
                ReadOutcome::Records => {
                    let progressed = !read.is_empty();
                    for (position, record) in read.drain(..) {
                        self.apply(state, record)?;
                        state.applied = position;
                    }
                    match until {
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

    fn apply(&self, state: &mut State<C>, record: Record) -> Result<()> {
        if let Some(&id) = state.bound.get(record.key()) {
            let binding = &state.bindings[id as usize];
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
        let mut logged = LoggedKeys::new();
        for record in records {
            fold(&mut logged, record);
        }
        for (key, &id) in &state.bound {
            state.bindings[id as usize].replace(&self.runtime, logged.get_mut(key))?;
        }
        state.logged = logged;
        state.applied = at;
        Ok(())
    }
}

fn set_record<C: Cells, T: PersistValue>(
    state: &State<C>,
    input: Input<T>,
    value: &T,
) -> Result<Record> {
    let binding = state
        .bindings
        .get(input.id as usize)
        .and_then(|binding| binding.as_any().downcast_ref::<InputBinding<T>>())
        .filter(|binding| binding.handle == input.handle)
        .ok_or(Error::UnknownHandle)?;
    Ok(Record::Set {
        key: binding.key.clone(),
        value: encode(value)?,
    })
}
