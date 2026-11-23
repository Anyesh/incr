use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use incr_core::persist_hooks::{self, Deferred, Displaced, RepSide, RepSink};
use incr_core::{
    with_value_code, Cells, ElementSlot, ElementsStamp, ErasedIncr, Incr, IncrCollection, Runtime,
    Value, Yielder,
};
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::after_lock::AfterLock;
use super::binding::{decode, encode, Binding, Folded};
use super::dep_index::{fp_of, DepFp, DepIndex, DepMap, FpSource, Fps, PanicGuard, Registered};
use super::logged::{ElementOp, Elements, LoggedKey};
use super::{Batch, KeyRef, PersistValue, State, Store};
use crate::value_context::{enter_value_scope, hold};
use crate::{Backend, Error, Kind, Position, Record, Result, ValueScope};

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

/// A plain collection's [`FpSource`].
struct CollectionFp<T: Value + Hash + Eq, C: Cells> {
    key: String,
    collection: IncrCollection<T, C>,
    scope: Option<ValueScope>,
}

// SAFETY: as for `CollectionBinding`: the source is reachable only through
// the `DepIndex` of the store that created it, and that store owns a
// `Runtime<C>`, which is Send or Sync exactly when C's cells are.
unsafe impl<T: Value + Hash + Eq, C: Cells> Send for CollectionFp<T, C> {}
// SAFETY: see `Send`.
unsafe impl<T: Value + Hash + Eq, C: Cells> Sync for CollectionFp<T, C> {}

impl<T, C> FpSource<C> for CollectionFp<T, C>
where
    T: Value + Hash + Eq + Serialize,
    C: Cells,
{
    fn fp(&self, rt: &Runtime<C>, bag: &mut Deferred<C>) -> DepFp {
        if !rt.is_alive(self.collection.version_node().erase()) {
            return DepFp::Untrusted;
        }
        let values = self.collection.snapshot_values();
        let fp = fp_in_scope(self.scope.as_ref(), &self.key, &values);
        bag.push(values);
        fp
    }
}

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

    pub(super) fn insert_record(&self, state: &State<C>, value: &T) -> Result<Record> {
        let key = self.key_in(state)?;
        Ok(Record::Insert {
            element: encode(&key, value)?,
            key,
        })
    }

    pub(super) fn delete_record(&self, state: &State<C>, value: &T) -> Result<Record> {
        let key = self.key_in(state)?;
        Ok(Record::Remove {
            element: encode(&key, value)?,
            key,
        })
    }

    pub(super) fn replace_record(
        &self,
        state: &State<C>,
        old: Option<&T>,
        new: &T,
    ) -> Result<Record> {
        let key = self.key_in(state)?;
        Ok(Record::Replace {
            old: old.map(|old| encode(&key, old)).transpose()?,
            new: encode(&key, new)?,
            key,
        })
    }

    pub(super) fn unlogged_in(&self, state: &State<C>) -> Result<IncrCollection<T, C>> {
        self.binding_in(state)
            .map(|binding| binding.collection.clone())
    }

    fn binding_in<'s>(&self, state: &'s State<C>) -> Result<&'s CollectionBinding<T, C>> {
        state
            .bindings
            .get(&self.id)
            .and_then(|binding| binding.as_any().downcast_ref::<CollectionBinding<T, C>>())
            .filter(|binding| binding.collection.version_node() == self.view.version_node())
            .ok_or(Error::UnknownHandle)
    }

    fn key_in(&self, state: &State<C>) -> Result<String> {
        self.binding_in(state).map(|binding| binding.key.clone())
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
            Some(logged) => decode_elements::<T>(
                key,
                &logged.elements,
                &state.after,
                &self.runtime.yielder_under_lock(),
            )?,
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
        binding.run_converge(&self.runtime, &plan, hold.as_deref_mut(), &mut spent);
        if let Some(map) = hold.as_deref_mut() {
            let fp = Binding::<C>::dep_fp(&binding, self.value_scope());
            map.register(key, binding.node(), Kind::Collection, fp, None);
        }
        drop(hold);
        spent.keep_in(&state.after);
        plan.keep_in(&state.after);
        let view = binding.collection.as_view();
        let id = state.bind(key, Box::new(binding));
        Ok(PersistedCollection { id, view })
    }
}

/// The sum, mod 2^128, of the fingerprints of the encodings of `values`,
/// one per stored occurrence: a collection's fingerprint.
pub(super) fn fp_of_elements<'a, T: Serialize + 'a>(
    key: &str,
    values: impl IntoIterator<Item = &'a T>,
) -> Result<u128> {
    values.into_iter().try_fold(0u128, |sum, value| {
        Ok(sum.wrapping_add(fp_of(&encode(key, value)?)))
    })
}

/// [`fp_of_elements`] in one hold of `scope`, the store's value scope,
/// made current for the encodes; `Untrusted` when an element does not
/// encode.
pub(super) fn fp_in_scope<'a, T: Serialize + 'a>(
    scope: Option<&ValueScope>,
    key: &str,
    values: impl IntoIterator<Item = &'a T>,
) -> DepFp {
    let _scope = scope.cloned().map(enter_value_scope);
    let mut fp = DepFp::Untrusted;
    hold(scope, || {
        if let Ok(sum) = fp_of_elements(key, values) {
            fp = DepFp::Known(sum);
        }
    });
    fp
}

/// What a single mutation changes in a collection's fingerprint, from the
/// representatives the core reports for it.
pub(super) struct FpDelta<'a> {
    key: &'a str,
    delta: u128,
    failed: bool,
}

impl<T: Serialize> RepSink<T> for FpDelta<'_> {
    fn on_rep(&mut self, side: RepSide, rep: &T) {
        let Ok(bytes) = encode(self.key, rep) else {
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

impl<'a> FpDelta<'a> {
    pub(super) fn new(key: &'a str) -> Self {
        FpDelta {
            key,
            delta: 0,
            failed: false,
        }
    }

    /// Applies the change to `key`'s fingerprint, which the caller found
    /// [tracked](DepMap::tracks).
    pub(super) fn apply_to(&self, map: &mut DepMap, key: &str) {
        if self.failed {
            map.untrust(key);
        } else {
            map.shift_fp(key, self.delta);
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

impl<T: Send + 'static> Converge<T> {
    /// Moves the plan into `bag` unless it holds nothing.
    fn keep_in(self, bag: &AfterLock) {
        if !(self.current.is_empty() && self.target.is_empty() && self.moved.is_empty()) {
            bag.push(self);
        }
    }
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

impl<T: Send + 'static> Spent<T> {
    /// Moves the values into `bag` unless there are none.
    fn keep_in(self, bag: &AfterLock) {
        bag.push_vec(self.displaced);
        bag.push_vec(self.values);
    }
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
    fn apply_op(&self, rt: &Runtime<C>, op: Op<T>, deps: Option<&DepIndex>, bag: &AfterLock) {
        let mut spent = Spent::default();
        let mut hold = deps.map(DepIndex::write);
        let mut guard = PanicGuard::new(hold.as_deref_mut(), &self.key);
        if guard.deps().is_some_and(|map| map.tracks(&self.key)) {
            let mut sink = FpDelta::new(&self.key);
            self.mutate(rt, op, &mut sink, &mut spent);
            if let Some(map) = guard.deps() {
                sink.apply_to(map, &self.key);
            }
        } else {
            self.mutate(rt, op, &mut (), &mut spent);
        }
        guard.disarm();
        drop(hold);
        spent.keep_in(bag);
    }

    /// The elements encoded, in byte order, for the logged state an
    /// unbound key keeps. Any order other than hash order, which differs
    /// per process, would do; the runtime stores one value per class of
    /// equal elements, so no order of them changes the state.
    fn encoded_values(&self, rt: &Runtime<C>, bag: &AfterLock) -> Result<Vec<Vec<u8>>> {
        let values = self.collection.values(rt);
        let encoded = values
            .iter()
            .map(|value| encode(&self.key, value))
            .collect::<Result<Vec<_>>>();
        bag.push_vec(values);
        let mut encoded = encoded?;
        encoded.sort_unstable();
        Ok(encoded)
    }

    /// A snapshot's order of the distinct elements: by the high 64 bits of
    /// each one's encoding's fingerprint, ties by the encodings themselves,
    /// so every process lists the same elements in the same order. Keeps
    /// 16 bytes per element rather than its encoding, which the caller
    /// makes again as it yields.
    fn ranked_slots(&self, yielder: &Yielder<'_>) -> Result<(ElementsStamp, Vec<ElementSlot>)> {
        let mut slots = Vec::new();
        let stamp = self.collection.visit_elements(|slot, value, _| {
            yielder.tick();
            let element = encode(&self.key, value)?;
            slots.push(((fp_of(&element) >> 64) as u64, slot));
            Ok::<(), Error>(())
        })?;
        slots.sort_unstable_by_key(|&(fp, _)| fp);
        let mut at = 0;
        while at < slots.len() {
            yielder.tick();
            let run = slots[at..]
                .iter()
                .take_while(|&&(fp, _)| fp == slots[at].0)
                .count();
            if run > 1 {
                let mut tied = slots[at..at + run]
                    .iter()
                    .map(|&(fp, slot)| Ok((self.encode_slot(stamp, slot)?.0, fp, slot)))
                    .collect::<Result<Vec<_>>>()?;
                tied.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                for (into, (_, fp, slot)) in slots[at..at + run].iter_mut().zip(tied) {
                    *into = (fp, slot);
                }
            }
            at += run;
        }
        Ok((stamp, slots.into_iter().map(|(_, slot)| slot).collect()))
    }

    /// The encoding of the element at `slot` and its multiplicity. Fails
    /// with [`Error::Bypassed`] when the collection changed since `stamp`,
    /// which only a write around the store can do while the store is
    /// locked.
    fn encode_slot(&self, stamp: ElementsStamp, slot: ElementSlot) -> Result<(Vec<u8>, usize)> {
        self.collection
            .with_element(stamp, slot, |value, count| {
                encode(&self.key, value).map(|element| (element, count))
            })
            .unwrap_or_else(|| {
                Err(Error::Bypassed {
                    key: self.key.clone(),
                })
            })
    }

    fn insert(&self, element: Vec<u8>) -> Record {
        Record::Insert {
            key: self.key.clone(),
            element,
        }
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
        bag: &AfterLock,
    ) -> Result<()> {
        let current = count(self.collection.values(rt), bag);
        let mut moved = HashSet::new();
        for stored in current.keys() {
            if let Some((wanted, _)) = with_value_code(|| target.get_key_value(stored)) {
                if encode(&self.key, stored)? != encode(&self.key, wanted)? {
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
                    let element = encode(&self.key, value)?;
                    changes.extend((want..have).map(|_| Record::Remove {
                        key: self.key.clone(),
                        element: element.clone(),
                    }));
                }
            }
            for (value, &want) in &plan.target {
                let have = plan.kept(value, &plan.current);
                if have < want {
                    let element = encode(&self.key, value)?;
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
        spent.keep_in(bag);
        plan.keep_in(bag);
        Ok(())
    }

    /// The changes of a [`Converge`], made inside the caller's hold of
    /// `map`, if any. When `map` [tracks](DepMap::tracks) the key, returns
    /// the fingerprint after them.
    fn run_converge(
        &self,
        rt: &Runtime<C>,
        plan: &Converge<T>,
        map: Option<&mut DepMap>,
        spent: &mut Spent<T>,
    ) -> Option<DepFp> {
        let mut guard = PanicGuard::new(map, &self.key);
        let yielder = rt.yielder_under_lock();
        for (value, &have) in &plan.current {
            let want = plan.kept(value, &plan.target);
            for _ in want..have {
                yielder.tick();
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
                yielder.tick();
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
        let tracked = guard.deps().is_some_and(|map| map.tracks(&self.key));
        let fp = tracked.then(|| {
            spent.values = self.collection.values(rt);
            match fp_of_elements(&self.key, &spent.values) {
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

    fn unbind(&self, rt: &Runtime<C>, bag: &AfterLock) -> Result<Option<Elements>> {
        if !rt.is_alive(self.node()) {
            return Err(Error::UnknownHandle);
        }
        Ok(Some(Elements::from_encoded(self.encoded_values(rt, bag)?)))
    }

    fn mapped_stamp(&self) -> Option<u64> {
        None
    }

    fn dep_fp(&self, scope: Option<ValueScope>) -> Registered {
        let source: Fps<C> = Arc::new(CollectionFp {
            key: self.key.clone(),
            collection: self.collection.as_view(),
            scope,
        });
        Registered::Later(Box::new(source))
    }

    fn apply(
        &mut self,
        rt: &Runtime<C>,
        record: &Record,
        _before: Option<&LoggedKey>,
        deps: Option<&DepIndex>,
        bag: &AfterLock,
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
                return self.converge(rt, HashMap::new(), None, deps, bag);
            }
            Record::Declare { .. } | Record::Set { .. } => return Ok(()),
        };
        self.apply_op(rt, op, deps, bag);
        Ok(())
    }

    fn replace(
        &mut self,
        rt: &Runtime<C>,
        logged: Option<&mut LoggedKey>,
        changes: Option<&mut Vec<Record>>,
        deps: Option<&DepIndex>,
        bag: &AfterLock,
    ) -> Result<()> {
        let target = match logged {
            Some(logged) => {
                let target =
                    decode_elements(&self.key, &logged.elements, bag, &rt.yielder_under_lock())?;
                logged.elements = Elements::default();
                target
            }
            None => HashMap::new(),
        };
        self.converge(rt, target, changes, deps, bag)
    }

    // The handle a PersistedCollection exposes is a view whose mutators
    // panic, so a collection cannot be written around the store.
    #[cfg(debug_assertions)]
    fn check_bypass(
        &self,
        _rt: &Runtime<C>,
        _logged: Option<&LoggedKey>,
        _bag: &AfterLock,
    ) -> Result<()> {
        Ok(())
    }

    fn restamp(
        &mut self,
        _rt: &Runtime<C>,
        _logged: Option<&LoggedKey>,
        _bag: &AfterLock,
    ) -> Result<Option<(u128, u64)>> {
        Ok(None)
    }

    fn fold<'a>(
        &'a self,
        logged: Option<&'a LoggedKey>,
        yielder: &Yielder<'_>,
    ) -> Result<Folded<'a>> {
        let (stamp, slots) = self.ranked_slots(yielder)?;
        let has_value = logged.is_some_and(|logged| logged.value.is_some());
        if slots.is_empty() && !has_value {
            return Ok(Box::new(std::iter::empty()));
        }
        let head = logged.map(|logged| logged.iter_records(&self.key).map(Ok));
        let mut slots = slots.into_iter();
        let mut repeat: Option<(Vec<u8>, usize)> = None;
        let mut failed = false;
        let elements = std::iter::from_fn(move || {
            if failed {
                return None;
            }
            if let Some((element, left)) = &mut repeat {
                *left -= 1;
                let element = if *left == 0 {
                    repeat.take().map(|(element, _)| element)?
                } else {
                    element.clone()
                };
                return Some(Ok(self.insert(element)));
            }
            let slot = slots.next()?;
            match self.encode_slot(stamp, slot) {
                Ok((element, count)) => {
                    if count > 1 {
                        repeat = Some((element.clone(), count - 1));
                    }
                    Some(Ok(self.insert(element)))
                }
                Err(e) => {
                    failed = true;
                    Some(Err(e))
                }
            }
        });
        Ok(Box::new(head.into_iter().flatten().chain(elements)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn count<T: Hash + Eq + Send + 'static>(values: Vec<T>, bag: &AfterLock) -> HashMap<T, usize> {
    let mut counts = HashMap::new();
    with_value_code(|| {
        for value in values {
            add(&mut counts, value, 1, bag);
        }
    });
    counts
}

/// Adds `n` occurrences of `value`, moving it into `bag` when an equal
/// element is already counted.
fn add<T: Hash + Eq + Send + 'static>(
    counts: &mut HashMap<T, usize>,
    value: T,
    n: usize,
    bag: &AfterLock,
) {
    match counts.get_mut(&value) {
        Some(have) => {
            *have += n;
            bag.push(value);
        }
        None => {
            counts.insert(value, n);
        }
    }
}

/// Folds byte-level elements into a multiset by decoded value: the live
/// bytes in the order first inserted, then the raw tail in order, removing
/// by `T`'s equality. Each key is the first inserted of its equal elements,
/// the value the runtime stores for them.
fn decode_elements<T: Hash + Eq + DeserializeOwned + Send + 'static>(
    key: &str,
    elements: &Elements,
    bag: &AfterLock,
    yielder: &Yielder<'_>,
) -> Result<HashMap<T, usize>> {
    let mut counts = HashMap::new();
    with_value_code(|| -> Result<()> {
        for (bytes, n) in elements.live() {
            yielder.tick();
            add(&mut counts, decode(key, bytes)?, n, bag);
        }
        for op in elements.tail() {
            yielder.tick();
            match op {
                ElementOp::Insert(bytes) => add(&mut counts, decode(key, bytes)?, 1, bag),
                ElementOp::Remove(bytes) => {
                    let value: T = decode(key, bytes)?;
                    if let Some(n) = counts.get_mut(&value) {
                        *n -= 1;
                        if *n == 0 {
                            bag.push(counts.remove_entry(&value));
                        }
                    }
                    bag.push(value);
                }
            }
        }
        Ok(())
    })?;
    Ok(counts)
}
