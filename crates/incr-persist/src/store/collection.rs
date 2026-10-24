use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;

use incr_core::persist_hooks::{self, Displaced, RepSide, RepSink};
use incr_core::{with_value_code, Cells, ErasedIncr, Incr, IncrCollection, Runtime, Value};
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::binding::{decode, encode, Binding};
use super::dep_index::{fp_of, DepFp, DepIndex, DepMap, PanicGuard};
use super::logged::{ElementOp, Elements, LoggedKey};
use super::{Batch, KeyRef, PersistValue, State, Store};
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
            .get(&self.id)
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

impl<'a, T: Value + Hash + Eq, C: Cells> From<&'a PersistedCollection<T, C>> for KeyRef<'a> {
    fn from(collection: &'a PersistedCollection<T, C>) -> Self {
        KeyRef::handle(collection.id, collection.view.version_node())
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
    /// If neither this store nor the log, read to its tail first, has a
    /// live `Declare` for `key`, appends one through the write path; a
    /// collection's default is empty. As for [`Store::input`], a key that a
    /// compaction at or below this store's position forgot has none. The
    /// collection starts with the key's logged elements, folded by decoded
    /// `T`. Declaring a key this store already declared returns a handle to
    /// the same collection.
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
            return state.bindings[&id]
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
        let plan = Converge {
            current: HashMap::new(),
            target: elements,
            moved: HashSet::new(),
        };
        let mut spent = Spent::default();
        let mut hold = state.deps.as_deref().map(DepIndex::write);
        let binding = CollectionBinding {
            key: key.to_owned(),
            collection: self.runtime.create_collection(),
        };
        let fp = binding.run_converge(&self.runtime, &plan, hold.as_deref_mut(), &mut spent);
        if let (Some(map), Some(fp)) = (hold.as_deref_mut(), fp) {
            map.register(key, binding.node(), Kind::Collection, fp, None);
        }
        drop(hold);
        drop(spent);
        let view = binding.collection.as_view();
        let id = state.bind(key, Box::new(binding));
        Ok(PersistedCollection { id, view })
    }
}

/// The sum, mod 2^128, of the fingerprints of the encodings of `values`,
/// one per stored occurrence: a collection's fingerprint.
pub(super) fn fp_of_elements<T: Serialize>(values: &[T]) -> Result<u128> {
    values.iter().try_fold(0u128, |sum, value| {
        Ok(sum.wrapping_add(fp_of(&encode(value)?)))
    })
}

/// What a single mutation changes in a collection's fingerprint, from the
/// representatives the core reports for it.
#[derive(Default)]
struct FpDelta {
    delta: u128,
    failed: bool,
}

impl<T: Serialize> RepSink<T> for FpDelta {
    fn on_rep(&mut self, side: RepSide, rep: &T) {
        let Ok(bytes) = encode(rep) else {
            self.failed = true;
            return;
        };
        let fp = fp_of(&bytes);
        self.delta = match side {
            RepSide::Removed => self.delta.wrapping_sub(fp),
            RepSide::Added => self.delta.wrapping_add(fp),
        };
    }
}

impl FpDelta {
    fn apply_to(&self, map: &mut DepMap, key: &str) {
        match map.entry(key).map(|entry| entry.fp) {
            _ if self.failed => map.untrust(key),
            Some(DepFp::Known(fp)) => map.set_fp(key, fp.wrapping_add(self.delta)),
            _ => {}
        }
    }
}

enum Op<T> {
    Insert(T),
    Remove(T),
    Replace(Option<T>, T),
}

/// What [`CollectionBinding::converge`] works out before taking the hold.
struct Converge<T> {
    current: HashMap<T, usize>,
    target: HashMap<T, usize>,
    /// Elements in both whose stored values encode differently: all of
    /// their occurrences are removed and the target's inserted, so the
    /// collection ends up storing the target's.
    moved: HashSet<T>,
}

impl<T: Hash + Eq> Converge<T> {
    /// How many occurrences of `value`, an element of one side, the other
    /// side has that converging keeps.
    fn kept(&self, value: &T, other: &HashMap<T, usize>) -> usize {
        with_value_code(|| {
            if self.moved.contains(value) {
                0
            } else {
                other.get(value).copied().unwrap_or(0)
            }
        })
    }
}

/// Values a mapped write is done with, dropped after its hold is released.
struct Spent<T> {
    displaced: Vec<Displaced<T>>,
    values: Vec<T>,
}

impl<T> Default for Spent<T> {
    fn default() -> Self {
        Spent {
            displaced: Vec::new(),
            values: Vec::new(),
        }
    }
}

impl<T: Value + Hash + Eq + Serialize, C: Cells> CollectionBinding<T, C> {
    fn mutate<R: RepSink<T>>(
        &self,
        rt: &Runtime<C>,
        op: Op<T>,
        sink: &mut R,
        spent: &mut Spent<T>,
    ) {
        let keep = &mut spent.displaced;
        match op {
            Op::Insert(value) => {
                persist_hooks::collection_insert(rt, &self.collection, value, sink, keep)
            }
            Op::Remove(value) => {
                persist_hooks::collection_delete(rt, &self.collection, &value, sink, keep);
                spent.values.push(value);
            }
            Op::Replace(old, new) => {
                persist_hooks::collection_replace(
                    rt,
                    &self.collection,
                    old.as_ref(),
                    new,
                    sink,
                    keep,
                );
                spent.values.extend(old);
            }
        }
    }

    /// Applies one mutation as a mapped write.
    fn apply_op(&self, rt: &Runtime<C>, op: Op<T>, deps: Option<&DepIndex>) {
        let mut spent = Spent::default();
        let mut hold = deps.map(DepIndex::write);
        let mut guard = PanicGuard::new(hold.as_deref_mut(), &self.key);
        if guard.deps().is_some() {
            let mut sink = FpDelta::default();
            self.mutate(rt, op, &mut sink, &mut spent);
            if let Some(map) = guard.deps() {
                sink.apply_to(map, &self.key);
            }
        } else {
            self.mutate(rt, op, &mut (), &mut spent);
        }
        guard.disarm();
        drop(hold);
        drop(spent);
    }

    /// Brings the collection to exactly `target`, stored values included,
    /// by applying only the multiplicity differences, removes first, so
    /// queries over it see one change per differing occurrence rather than
    /// a rebuild. Each change is also pushed onto `changes`, when given, as
    /// the record that makes it. The changes are a mapped write, and the
    /// fingerprint is recomputed after them.
    fn converge(
        &self,
        rt: &Runtime<C>,
        target: HashMap<T, usize>,
        changes: Option<&mut Vec<Record>>,
        deps: Option<&DepIndex>,
    ) -> Result<()> {
        let current = count(self.collection.values(rt));
        let mut moved = HashSet::new();
        for stored in current.keys() {
            if let Some((wanted, _)) = with_value_code(|| target.get_key_value(stored)) {
                if encode(stored)? != encode(wanted)? {
                    moved.insert(with_value_code(|| stored.clone()));
                }
            }
        }
        let plan = Converge {
            current,
            target,
            moved,
        };
        if let Some(changes) = changes {
            for (value, &have) in &plan.current {
                let want = plan.kept(value, &plan.target);
                if want < have {
                    let element = encode(value)?;
                    changes.extend((want..have).map(|_| Record::Remove {
                        key: self.key.clone(),
                        element: element.clone(),
                    }));
                }
            }
            for (value, &want) in &plan.target {
                let have = plan.kept(value, &plan.current);
                if have < want {
                    let element = encode(value)?;
                    changes.extend((have..want).map(|_| Record::Insert {
                        key: self.key.clone(),
                        element: element.clone(),
                    }));
                }
            }
        }
        let mut spent = Spent::default();
        let mut hold = deps.map(DepIndex::write);
        if let Some(fp) = self.run_converge(rt, &plan, hold.as_deref_mut(), &mut spent) {
            if let Some(map) = hold.as_deref_mut() {
                map.store_fp(&self.key, fp);
            }
        }
        drop(hold);
        drop(spent);
        Ok(())
    }

    /// The changes of a [`Converge`], made inside the caller's hold of
    /// `map`, if any. With `map`, returns the fingerprint after them.
    fn run_converge(
        &self,
        rt: &Runtime<C>,
        plan: &Converge<T>,
        map: Option<&mut DepMap>,
        spent: &mut Spent<T>,
    ) -> Option<DepFp> {
        let mut guard = PanicGuard::new(map, &self.key);
        for (value, &have) in &plan.current {
            let want = plan.kept(value, &plan.target);
            for _ in want..have {
                persist_hooks::collection_delete(
                    rt,
                    &self.collection,
                    value,
                    &mut (),
                    &mut spent.displaced,
                );
            }
        }
        for (value, &want) in &plan.target {
            let have = plan.kept(value, &plan.current);
            for _ in have..want {
                let value = with_value_code(|| value.clone());
                persist_hooks::collection_insert(
                    rt,
                    &self.collection,
                    value,
                    &mut (),
                    &mut spent.displaced,
                );
            }
        }
        let fp = guard.deps().is_some().then(|| {
            spent.values = self.collection.values(rt);
            match fp_of_elements(&spent.values) {
                Ok(fp) => DepFp::Known(fp),
                Err(_) => DepFp::Untrusted,
            }
        });
        guard.disarm();
        fp
    }
}

impl<T, C> Binding<C> for CollectionBinding<T, C>
where
    T: Value + Hash + Eq + Serialize + DeserializeOwned,
    C: Cells,
{
    fn kind(&self) -> Kind {
        Kind::Collection
    }

    fn key(&self) -> &str {
        &self.key
    }

    fn is_handle(&self, handle: &dyn Any) -> bool {
        handle.downcast_ref::<Incr<u64>>() == Some(&self.collection.version_node())
    }

    fn node(&self) -> ErasedIncr {
        self.collection.version_node().erase()
    }

    fn unbind(&self, rt: &Runtime<C>) -> Result<Option<Elements>> {
        if !rt.is_alive(self.node()) {
            return Err(Error::UnknownHandle);
        }
        let encoded = self
            .collection
            .values(rt)
            .iter()
            .map(encode)
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Elements::from_encoded(encoded)))
    }

    fn mapped_stamp(&self) -> Option<u64> {
        None
    }

    fn dep_fp(&self, rt: &Runtime<C>) -> Result<DepFp> {
        // Deleted through the runtime: nothing can say what it holds.
        if !rt.is_alive(self.node()) {
            return Ok(DepFp::Untrusted);
        }
        Ok(DepFp::Known(fp_of_elements(&self.collection.values(rt))?))
    }

    fn apply(
        &mut self,
        rt: &Runtime<C>,
        record: &Record,
        _before: Option<&LoggedKey>,
        deps: Option<&DepIndex>,
    ) -> Result<()> {
        let op = match record {
            Record::Insert { element, .. } => Op::Insert(decode(&self.key, element)?),
            Record::Remove { element, .. } => Op::Remove(decode(&self.key, element)?),
            Record::Replace { old, new, .. } => {
                let old: Option<T> = old
                    .as_deref()
                    .map(|old| decode(&self.key, old))
                    .transpose()?;
                Op::Replace(old, decode(&self.key, new)?)
            }
            Record::Delete { .. } | Record::Forget { .. } => {
                return self.converge(rt, HashMap::new(), None, deps);
            }
            Record::Declare { .. } | Record::Set { .. } => return Ok(()),
        };
        self.apply_op(rt, op, deps);
        Ok(())
    }

    fn replace(
        &mut self,
        rt: &Runtime<C>,
        logged: Option<&mut LoggedKey>,
        changes: Option<&mut Vec<Record>>,
        deps: Option<&DepIndex>,
    ) -> Result<()> {
        let target = match logged {
            Some(logged) => {
                let target = decode_elements(&self.key, &logged.elements)?;
                logged.elements = Elements::default();
                target
            }
            None => HashMap::new(),
        };
        self.converge(rt, target, changes, deps)
    }

    // The handle a PersistedCollection exposes is a view whose mutators
    // panic, so a collection cannot be written around the store.
    #[cfg(debug_assertions)]
    fn check_bypass(&self, _rt: &Runtime<C>, _logged: Option<&LoggedKey>) -> Result<()> {
        Ok(())
    }

    fn fold(&self, rt: &Runtime<C>, logged: Option<&LoggedKey>) -> Result<Vec<Record>> {
        let mut elements = self
            .collection
            .values(rt)
            .iter()
            .map(encode)
            .collect::<Result<Vec<_>>>()?;
        let has_value = logged.is_some_and(|logged| logged.value.is_some());
        if elements.is_empty() && !has_value {
            return Ok(Vec::new());
        }
        // A store that never declared this key diffs its catch-up by bytes in
        // snapshot order, so listing elements in hash order, which differs
        // per process, would make what it reports differ too. The runtime
        // stores one value per class of equal elements, so no order of them
        // changes the state.
        elements.sort_unstable();
        let mut out = logged.map_or_else(Vec::new, |logged| logged.records(&self.key));
        out.extend(elements.into_iter().map(|element| Record::Insert {
            key: self.key.clone(),
            element,
        }));
        Ok(out)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn count<T: Hash + Eq>(values: Vec<T>) -> HashMap<T, usize> {
    let mut counts = HashMap::new();
    with_value_code(|| {
        for value in values {
            *counts.entry(value).or_insert(0) += 1;
        }
    });
    counts
}

/// Folds byte-level elements into a multiset by decoded value: the live
/// bytes in the order first inserted, then the raw tail in order, removing
/// by `T`'s equality. Each key is the first inserted of its equal elements,
/// the value the runtime stores for them.
fn decode_elements<T: Hash + Eq + DeserializeOwned>(
    key: &str,
    elements: &Elements,
) -> Result<HashMap<T, usize>> {
    let mut counts = HashMap::new();
    with_value_code(|| -> Result<()> {
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
        Ok(())
    })?;
    Ok(counts)
}
