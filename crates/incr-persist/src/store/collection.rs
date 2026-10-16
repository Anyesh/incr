use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

use incr_core::{Cells, IncrCollection, Runtime, Value};
use serde::de::DeserializeOwned;

use super::binding::{decode, encode, Binding};
use super::logged::{ElementOp, Elements, LoggedKey};
use super::{Batch, PersistValue, State, Store};
use crate::{Backend, Error, Kind, Position, Record, Result};

/// A persisted collection: a key in the log bound to a multiset collection
/// of the store's runtime.
///
/// Write through [`insert`](Self::insert), [`delete`](Self::delete) and
/// [`replace`](Self::replace), or the [`Batch`] methods of the same names;
/// each appends a record and applies it the way [`IncrCollection`] would.
/// Elements are compared by `T`'s `Eq`, never by their serialized bytes, so
/// a type whose serialization is not canonical (a `HashMap` field, say)
/// folds the same in the log as in the runtime.
pub struct PersistedCollection<T: Value + Hash + Eq, C: Cells> {
    id: u32,
    view: IncrCollection<T, C>,
}

/// The store's mutable side of a [`PersistedCollection`]: the runtime
/// collection is the key's element multiset, by decoded value.
pub(super) struct CollectionBinding<T: Value + Hash + Eq, C: Cells> {
    key: String,
    collection: IncrCollection<T, C>,
}

// SAFETY: the only part of `IncrCollection<T, C>` that is not Send + Sync is
// the marker confining it to C's threading model; under `Shared` it is
// Send + Sync already. A binding is reachable only through the state of the
// `Store` that created it, and that store owns a `Runtime<C>`, which is
// Send or Sync exactly when C's cells are (`Runtime<Local>` is neither, see
// the compile_fail example on `Store`), so under `Local` the binding never
// leaves the store's thread.
unsafe impl<T: Value + Hash + Eq, C: Cells> Send for CollectionBinding<T, C> {}
unsafe impl<T: Value + Hash + Eq, C: Cells> Sync for CollectionBinding<T, C> {}

impl<T: Value + Hash + Eq, C: Cells> PersistedCollection<T, C> {
    /// The runtime collection, for reads, operators (`filter`, `count`,
    /// `group_by`, ...) and queries. It is a derived view, so mutating it
    /// directly panics instead of bypassing the log.
    pub fn handle(&self) -> &IncrCollection<T, C> {
        &self.view
    }
}

impl<T: PersistValue + Hash + Eq, C: Cells> PersistedCollection<T, C> {
    /// Appends one occurrence of `value` through the write path and returns
    /// the record's position. Errors behave as for [`Store::set`].
    pub fn insert<B: Backend>(&self, store: &Store<C, B>, value: T) -> Result<Position> {
        store.write_with(|state| self.insert_record(state, &value))
    }

    /// Removes one occurrence of `value` through the write path and returns
    /// the record's position. Removing an absent element appends a record
    /// that changes nothing when applied, as [`IncrCollection::delete`]
    /// records nothing; presence is decided at the record's position, which
    /// may differ from what this store has applied so far.
    pub fn delete<B: Backend>(&self, store: &Store<C, B>, value: &T) -> Result<Position> {
        store.write_with(|state| self.delete_record(state, value))
    }

    /// Removes one occurrence of `old`, if given and present, and inserts
    /// `new`, as one record applied in one step like
    /// [`IncrCollection::replace`]. `replace(store, None, new)` is an
    /// insert.
    pub fn replace<B: Backend>(
        &self,
        store: &Store<C, B>,
        old: Option<&T>,
        new: T,
    ) -> Result<Position> {
        store.write_with(|state| self.replace_record(state, old, &new))
    }

    fn insert_record(&self, state: &State<C>, value: &T) -> Result<Record> {
        Ok(Record::Insert {
            key: self.key_in(state)?,
            element: encode(value)?,
        })
    }

    fn delete_record(&self, state: &State<C>, value: &T) -> Result<Record> {
        Ok(Record::Remove {
            key: self.key_in(state)?,
            element: encode(value)?,
        })
    }

    fn replace_record(&self, state: &State<C>, old: Option<&T>, new: &T) -> Result<Record> {
        Ok(Record::Replace {
            key: self.key_in(state)?,
            old: old.map(encode).transpose()?,
            new: encode(new)?,
        })
    }

    fn key_in(&self, state: &State<C>) -> Result<String> {
        state
            .bindings
            .get(self.id as usize)
            .and_then(|binding| binding.as_any().downcast_ref::<CollectionBinding<T, C>>())
            .filter(|binding| binding.collection.version_node() == self.view.version_node())
            .map(|binding| binding.key.clone())
            .ok_or(Error::UnknownHandle)
    }
}

impl<T: Value + Hash + Eq, C: Cells> Clone for PersistedCollection<T, C> {
    fn clone(&self) -> Self {
        PersistedCollection {
            id: self.id,
            view: self.view.clone(),
        }
    }
}

impl<T: Value + Hash + Eq, C: Cells> fmt::Debug for PersistedCollection<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PersistedCollection")
            .field("id", &self.id)
            .field("version_node", &self.view.version_node())
            .finish()
    }
}

impl<C: Cells, B: Backend> Batch<'_, C, B> {
    /// Adds an insert of `value` into `collection` to the batch. Failures
    /// are kept as for [`Batch::set`].
    pub fn insert<T: PersistValue + Hash + Eq>(
        &mut self,
        collection: &PersistedCollection<T, C>,
        value: T,
    ) -> &mut Self {
        self.push(|state| collection.insert_record(state, &value))
    }

    /// Adds a removal of one occurrence of `value` from `collection` to the
    /// batch, as [`PersistedCollection::delete`] would write it.
    pub fn delete<T: PersistValue + Hash + Eq>(
        &mut self,
        collection: &PersistedCollection<T, C>,
        value: &T,
    ) -> &mut Self {
        self.push(|state| collection.delete_record(state, value))
    }

    /// Adds a replace in `collection` to the batch, as
    /// [`PersistedCollection::replace`] would write it.
    pub fn replace<T: PersistValue + Hash + Eq>(
        &mut self,
        collection: &PersistedCollection<T, C>,
        old: Option<&T>,
        new: T,
    ) -> &mut Self {
        self.push(|state| collection.replace_record(state, old, &new))
    }
}

impl<C: Cells, B: Backend> Store<C, B> {
    /// Declares the collection `key` and returns its handle.
    ///
    /// If the log has no live `Declare` for `key`, appends one through the
    /// write path; a collection's default is empty. The collection starts
    /// with the key's logged elements, folded by decoded `T`. Declaring a
    /// key this store already declared returns a handle to the same
    /// collection.
    ///
    /// Fails with [`Error::TypeMismatch`] when a logged element does not
    /// decode as `T` or the key is already bound to another element type,
    /// and with [`Error::KindMismatch`] when the key is an input. Neither
    /// poisons the store.
    pub fn collection<T: PersistValue + Hash + Eq>(
        &self,
        key: &str,
    ) -> Result<PersistedCollection<T, C>> {
        let mut guard = self.state()?;
        let state = &mut *guard;
        if let Some(id) = state.bound_id(key, Kind::Collection)? {
            return state.bindings[id as usize]
                .as_any()
                .downcast_ref::<CollectionBinding<T, C>>()
                .map(|binding| PersistedCollection {
                    id,
                    view: binding.collection.as_view(),
                })
                .ok_or_else(|| Error::TypeMismatch {
                    key: key.to_owned(),
                    source: None,
                });
        }
        self.declare(state, key, Kind::Collection, || Ok(Vec::new()))?;
        let logged = state.logged.get_mut(key);
        let elements = match &logged {
            Some(logged) => decode_elements::<T>(key, &logged.elements)?,
            None => HashMap::new(),
        };
        if let Some(logged) = logged {
            logged.elements = Elements::default();
        }
        let binding = CollectionBinding {
            key: key.to_owned(),
            collection: self.runtime.create_collection(),
        };
        binding.converge(&self.runtime, elements);
        let view = binding.collection.as_view();
        let id = state.bind(key, Box::new(binding));
        Ok(PersistedCollection { id, view })
    }
}

impl<T: Value + Hash + Eq, C: Cells> CollectionBinding<T, C> {
    /// Brings the collection to exactly `target` by applying only the
    /// multiplicity differences, removes first, so queries over it see one
    /// change per differing occurrence rather than a rebuild.
    fn converge(&self, rt: &Runtime<C>, target: HashMap<T, usize>) {
        let current = count(self.collection.values(rt));
        for (value, &have) in &current {
            let want = target.get(value).copied().unwrap_or(0);
            for _ in want..have {
                self.collection.delete(rt, value);
            }
        }
        for (value, want) in target {
            let have = current.get(&value).copied().unwrap_or(0);
            for _ in have..want {
                self.collection.insert(rt, value.clone());
            }
        }
    }
}

impl<T, C> Binding<C> for CollectionBinding<T, C>
where
    T: Value + Hash + Eq + serde::Serialize + DeserializeOwned,
    C: Cells,
{
    fn kind(&self) -> Kind {
        Kind::Collection
    }

    fn apply(&self, rt: &Runtime<C>, record: &Record, _before: Option<&LoggedKey>) -> Result<()> {
        match record {
            Record::Insert { element, .. } => {
                self.collection.insert(rt, decode(&self.key, element)?);
            }
            Record::Remove { element, .. } => {
                self.collection.delete(rt, &decode(&self.key, element)?);
            }
            Record::Replace { old, new, .. } => {
                let old: Option<T> = old
                    .as_deref()
                    .map(|old| decode(&self.key, old))
                    .transpose()?;
                let new = decode(&self.key, new)?;
                self.collection.replace(rt, old.as_ref(), new);
            }
            Record::Delete { .. } => self.converge(rt, HashMap::new()),
            Record::Declare { .. } | Record::Set { .. } => {}
        }
        Ok(())
    }

    fn replace(&self, rt: &Runtime<C>, logged: Option<&mut LoggedKey>) -> Result<()> {
        let target = match logged {
            Some(logged) => {
                let target = decode_elements(&self.key, &logged.elements)?;
                logged.elements = Elements::default();
                target
            }
            None => HashMap::new(),
        };
        self.converge(rt, target);
        Ok(())
    }

    fn fold(&self, rt: &Runtime<C>, logged: Option<&LoggedKey>) -> Result<Vec<Record>> {
        let elements = self.collection.values(rt);
        let has_value = logged.is_some_and(|logged| logged.value.is_some());
        if elements.is_empty() && !has_value {
            return Ok(Vec::new());
        }
        let mut out = logged.map_or_else(Vec::new, |logged| logged.records(&self.key));
        for element in &elements {
            out.push(Record::Insert {
                key: self.key.clone(),
                element: encode(element)?,
            });
        }
        Ok(out)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn count<T: Hash + Eq>(values: Vec<T>) -> HashMap<T, usize> {
    let mut counts = HashMap::new();
    for value in values {
        *counts.entry(value).or_insert(0) += 1;
    }
    counts
}

/// Folds byte-level elements into a multiset by decoded value: the live
/// bytes first, then the raw tail in order, removing by `T`'s equality.
fn decode_elements<T: Hash + Eq + DeserializeOwned>(
    key: &str,
    elements: &Elements,
) -> Result<HashMap<T, usize>> {
    let mut counts = HashMap::new();
    for (bytes, n) in elements.live() {
        *counts.entry(decode(key, bytes)?).or_insert(0) += n;
    }
    for op in elements.tail() {
        match op {
            ElementOp::Insert(bytes) => *counts.entry(decode(key, bytes)?).or_insert(0) += 1,
            ElementOp::Remove(bytes) => {
                let value: T = decode(key, bytes)?;
                if let Some(n) = counts.get_mut(&value) {
                    *n -= 1;
                    if *n == 0 {
                        counts.remove(&value);
                    }
                }
            }
        }
    }
    Ok(counts)
}
