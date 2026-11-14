use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::{Arc, Weak};

use incr_core::persist_hooks::Deferred;
use incr_core::{
    find_element, with_value_code, Cells, DeleteError, ElementRank, ErasedIncr, Incr,
    IncrCollection, KeyCounts, KeyIndex, KeyedChange, KeyedCollection, KeyedSpent, Pair, PairOp,
    Runtime, Shared, Value,
};

use super::binding::{decode, encode, Binding};
use super::collection::{fp_of_elements, FpDelta};
use super::dep_index::{fp_of, DepFp, DepIndex, PanicGuard};
use super::logged::{ElementOp, Elements, LoggedKey};
use super::{set_record, Input, KeyRef, PersistValue, PersistedCollection, State, Store};
use crate::{Backend, Error, Kind, Position, Record, Result};

/// How many times a keyed write appends fixes for keys other writers
/// changed under it before failing with [`Error::KeyedContention`].
const REPAIR_ROUNDS: usize = 3;

type Core<K, V, C, P> = KeyedCollection<K, V, C, P, u128>;
type Ranked<P> = (P, u128);

/// The rank `KeyedCollection::upsert` and `reconcile` give a pair, which
/// an unlogged scope's writes leave behind: no logged bytes back it.
const UNLOGGED_RANK: u128 = incr_core::UNIDENTIFYING_RANK;

/// A key's bytes and its row's fingerprint, `None` for no row.
pub(crate) type RowDep = (Vec<u8>, Option<u128>);

/// What a memo maps of a keyed collection's `get_key` reads. A row's
/// fingerprint is the rank of the pair read for its key: the fingerprint
/// of the bytes that pair was applied from, `None` for a key holding no
/// row.
///
/// Callers hold the memo lock or a `DepIndex` hold, under which a user
/// `Drop` must not run, so every key a method materialises goes to `bag`.
pub(crate) trait RowSource<C: Cells>: Send + Sync {
    /// For a version input of this collection, its key's bytes and its
    /// row's fingerprint, or `Ok(None)` when its row is not a logged one;
    /// `None` for any other node.
    fn live_row(&self, dep: ErasedIncr, bag: &mut Deferred<C>) -> Option<Result<Option<RowDep>>>;

    /// The version input of the key `key_bytes` encode, created if no read
    /// created it; `None` when they do not decode. Takes the version lock,
    /// so not under a `DepIndex` hold.
    fn prepare(
        &self,
        rt: &Runtime<C>,
        key_bytes: &[u8],
        bag: &mut Deferred<C>,
    ) -> Option<ErasedIncr>;

    /// The version input of the key `key_bytes` encode and whether its
    /// row's fingerprint is `fp`; `None` when it has none or they do not
    /// decode.
    fn stored_row(
        &self,
        key_bytes: &[u8],
        fp: Option<u128>,
        bag: &mut Deferred<C>,
    ) -> Option<(ErasedIncr, bool)>;
}

/// A [`RowSource`] as a `DepIndex` holds it, behind `Any`.
pub(crate) type Rows<C> = Arc<dyn RowSource<C>>;

/// The [`Rows`] a `DepIndex` registered, under the store's cells `C`.
pub(crate) fn rows_of<C: Cells>(rows: &(dyn Any + Send + Sync)) -> &Rows<C> {
    rows.downcast_ref::<Rows<C>>()
        .expect("a store registers rows of its own cells")
}

struct KeyedRows<K, V, C, P>
where
    K: 'static,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
{
    key: String,
    keyed: Core<K, V, C, P>,
}

// SAFETY: as for `KeyedBinding`: the rows are reachable only through the
// `DepIndex` of the store that created them, and that store owns a
// `Runtime<C>`, which is Send or Sync exactly when C's cells are.
unsafe impl<K, V, C, P> Send for KeyedRows<K, V, C, P>
where
    K: 'static,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
{
}
// SAFETY: see `Send`.
unsafe impl<K, V, C, P> Sync for KeyedRows<K, V, C, P>
where
    K: 'static,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
{
}

impl<K, V, C, P> KeyedRows<K, V, C, P>
where
    K: PersistValue + Hash + Eq,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V> + PersistValue,
{
    /// The fingerprint of `key`'s row, `Some(None)` for no row, and `None`
    /// for a row an unlogged write left at [`UNLOGGED_RANK`].
    fn row_fp(&self, key: &K) -> Option<Option<u128>> {
        match self.keyed.read_rank(key) {
            Some(UNLOGGED_RANK) => None,
            rank => Some(rank),
        }
    }

    fn decode_key(&self, key_bytes: &[u8]) -> Option<K> {
        decode(&self.key, key_bytes).ok()
    }
}

impl<K, V, C, P> RowSource<C> for KeyedRows<K, V, C, P>
where
    K: PersistValue + Hash + Eq,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V> + PersistValue,
{
    fn live_row(&self, dep: ErasedIncr, bag: &mut Deferred<C>) -> Option<Result<Option<RowDep>>> {
        let key = self.keyed.key_of_version(dep)?;
        let row = match self.row_fp(&key) {
            Some(fp) => encode(&self.key, &key).map(|bytes| Some((bytes, fp))),
            None => Ok(None),
        };
        bag.push(key);
        Some(row)
    }

    fn prepare(
        &self,
        rt: &Runtime<C>,
        key_bytes: &[u8],
        bag: &mut Deferred<C>,
    ) -> Option<ErasedIncr> {
        let key = self.decode_key(key_bytes)?;
        let handle = self.keyed.version_input_or_create(rt, &key);
        bag.push(key);
        Some(handle.erase())
    }

    fn stored_row(
        &self,
        key_bytes: &[u8],
        fp: Option<u128>,
        bag: &mut Deferred<C>,
    ) -> Option<(ErasedIncr, bool)> {
        let key = self.decode_key(key_bytes)?;
        let row = self
            .keyed
            .version_input(&key)
            .map(|handle| (handle.erase(), self.row_fp(&key) == Some(fp)));
        bag.push(key);
        row
    }
}

/// A persisted keyed collection: a collection key in the log whose
/// elements are `P` pairs, bound to a [`KeyedCollection`] that reads at
/// most one row per key. Each pair is ranked by the fingerprint of the
/// bytes it was applied from, so a key that other writers left holding two
/// pairs reads, in every process, as the one with the smallest fingerprint.
///
/// Write through [`Store::upsert`], [`Store::remove`] and
/// [`Store::reconcile`], or stage writes in a [`PendingWrite`]. A changed
/// key appends a `Remove` of each pair it held, then an `Insert` of its new
/// pair. The handle only reads; [`UnloggedScope::keyed`] hands out the
/// writable collection for changes around the log.
///
/// [`UnloggedScope::keyed`]: super::UnloggedScope::keyed
pub struct PersistedKeyed<K, V, C, P = (K, V)>
where
    K: 'static,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
{
    id: u32,
    keyed: Core<K, V, C, P>,
}

/// The store's mutable side of a [`PersistedKeyed`]: the core collection's
/// pairs are the key's element multiset, by decoded value.
pub(super) struct KeyedBinding<K, V, C, P>
where
    K: 'static,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
{
    key: String,
    pub(super) keyed: Core<K, V, C, P>,
    rows: Rows<C>,
}

// SAFETY: the parts of `KeyedCollection<K, V, C, P, R>` that are not
// Send + Sync are its collections' markers confining them to C's threading
// model; under `Shared` it is Send + Sync already. A binding is reachable
// only through the state of the `Store` that created it, and that store
// owns a `Runtime<C>`, which is Send or Sync exactly when C's cells are, so
// under `Local` the binding never leaves the store's thread.
unsafe impl<K, V, C, P> Send for KeyedBinding<K, V, C, P>
where
    K: 'static,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
{
}
// SAFETY: see `Send`.
unsafe impl<K, V, C, P> Sync for KeyedBinding<K, V, C, P>
where
    K: 'static,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
{
}

impl<K, V, C, P> PersistedKeyed<K, V, C, P>
where
    K: Value + Hash + Eq,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
{
    /// A copy of the row read for `key`.
    pub fn row(&self, key: &K) -> Option<V> {
        self.keyed.row(key)
    }

    /// The row read for `key`, as [`KeyedCollection::get_key`] reads it:
    /// inside a query, a dependency on this key alone.
    pub fn get_key(&self, rt: &Runtime<C>, key: &K) -> Option<V> {
        self.keyed.get_key(rt, key)
    }

    /// `key`'s version input, if a read with [`Self::get_key`] or a memo
    /// restore created one.
    pub fn version_input(&self, key: &K) -> Option<Incr<u64>> {
        self.keyed.version_input(key)
    }

    pub fn contains(&self, key: &K) -> bool {
        self.keyed.contains(key)
    }

    /// The number of keys holding a row.
    pub fn len(&self) -> usize {
        self.keyed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keyed.is_empty()
    }

    /// A snapshot of the held keys, in no particular order.
    pub fn keys(&self) -> Vec<K> {
        self.keyed.keys()
    }

    /// A snapshot of the pair read for each key, in no particular order.
    pub fn items(&self) -> Vec<P> {
        self.keyed.items()
    }

    /// The rows collection, for queries.
    pub fn rows(&self) -> &IncrCollection<V, C> {
        self.keyed.rows()
    }

    /// A view of the pair collection, every pair held included, which
    /// panics on a direct write.
    pub fn pairs(&self) -> IncrCollection<P, C> {
        self.keyed.pairs()
    }

    /// The index's epoch, which advances on every change to it.
    pub fn epoch(&self) -> u64 {
        self.keyed.epoch()
    }

    /// A weak handle to the key index, for a caller that reads its
    /// [`StateStats`](incr_core::StateStats) only while the collection
    /// lives.
    pub fn index_weak(&self) -> Weak<KeyIndex<P, u128, C>> {
        self.keyed.index_weak()
    }

    fn version_node(&self) -> Incr<u64> {
        self.keyed.pairs().version_node()
    }

    fn binding_in<'s>(&self, state: &'s State<C>) -> Result<&'s KeyedBinding<K, V, C, P>>
    where
        P: PersistValue,
    {
        state
            .bindings
            .get(&self.id)
            .and_then(|binding| binding.as_any().downcast_ref::<KeyedBinding<K, V, C, P>>())
            .filter(|binding| binding.keyed.pairs().version_node() == self.version_node())
            .ok_or(Error::UnknownHandle)
    }

    pub(super) fn unlogged_in(&self, state: &State<C>) -> Result<Core<K, V, C, P>>
    where
        P: PersistValue,
    {
        self.binding_in(state).map(|binding| binding.keyed.clone())
    }
}

impl<K, V, C, P> Clone for PersistedKeyed<K, V, C, P>
where
    K: 'static,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
{
    fn clone(&self) -> Self {
        PersistedKeyed {
            id: self.id,
            keyed: self.keyed.clone(),
        }
    }
}

impl<K, V, C, P> fmt::Debug for PersistedKeyed<K, V, C, P>
where
    K: Value + Hash + Eq,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PersistedKeyed")
            .field("id", &self.id)
            .field("version_node", &self.version_node())
            .finish()
    }
}

impl<'a, K, V, C, P> From<&'a PersistedKeyed<K, V, C, P>> for KeyRef<'a>
where
    K: Value + Hash + Eq,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
{
    fn from(keyed: &'a PersistedKeyed<K, V, C, P>) -> Self {
        KeyRef::handle(keyed.id, keyed.version_node())
    }
}

type PlainOp<C> = Box<dyn FnOnce(&State<C>, &Runtime<C>) -> Result<Record>>;

pub(super) type WriteRun<'a, C> =
    dyn FnMut(&mut State<C>, Vec<Record>) -> Result<(Position, bool)> + 'a;

/// Writes collected over any number of calls, appended as one run by
/// [`Store::write_pending`]: plain records in the order added, then each
/// keyed collection's staged keys in the order first staged.
///
/// Keyed writes compose per key against the rows each key held when it
/// was first staged, which the write checks again once it has applied
/// (see [`Store::write_pending`]).
pub struct PendingWrite<C: Cells> {
    ops: Erased<PlainOp<C>, C>,
    stages: Erased<Box<dyn Stage<C>>, C>,
}

impl<C: Cells> Default for PendingWrite<C> {
    fn default() -> Self {
        PendingWrite {
            ops: Erased(Vec::new(), PhantomData),
            stages: Erased(Vec::new(), PhantomData),
        }
    }
}

/// A pending write's erased plain ops or keyed stages.
struct Erased<T, C>(Vec<T>, PhantomData<C>);

// SAFETY: the plain ops capture only an `Input`, a `PersistedCollection`
// and values, and `KeyedStage` is the only `Stage`; `_parts_are_send`
// proves each of those `Send` on `Shared` for every value type.
unsafe impl Send for Erased<PlainOp<Shared>, Shared> {}
// SAFETY: as above.
unsafe impl Send for Erased<Box<dyn Stage<Shared>>, Shared> {}

fn _parts_are_send<T, K, V, P>()
where
    T: PersistValue + Hash + Eq,
    K: Value + Hash + Eq,
    V: Value + Hash + Eq,
    P: Pair<K, V> + PersistValue,
{
    fn send<X: Send>() {}
    send::<Input<T>>();
    send::<Option<T>>();
    send::<PersistedCollection<T, Shared>>();
    send::<KeyedStage<K, V, Shared, P>>();
    send::<PendingWrite<Shared>>();
}

impl<T, C> std::ops::Deref for Erased<T, C> {
    type Target = Vec<T>;

    fn deref(&self) -> &Vec<T> {
        &self.0
    }
}

impl<T, C> std::ops::DerefMut for Erased<T, C> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        &mut self.0
    }
}

impl<C: Cells> PendingWrite<C> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether nothing was added. A write staging only unchanged rows is
    /// not empty, though it appends nothing.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty() && self.stages.is_empty()
    }

    /// Adds a `Set` of `input`, as [`Store::set`] would write it. Errors
    /// surface from [`Store::write_pending`].
    pub fn set<T: PersistValue>(&mut self, input: Input<T>, value: T) {
        self.ops
            .push(Box::new(move |state: &State<C>, rt: &Runtime<C>| {
                set_record(state, rt, input, &value)
            }));
    }

    /// Adds an insert into `collection`, as
    /// [`PersistedCollection::insert`] would write it.
    pub fn insert<T: PersistValue + Hash + Eq>(
        &mut self,
        collection: &PersistedCollection<T, C>,
        value: T,
    ) {
        let collection = collection.clone();
        self.ops
            .push(Box::new(move |state: &State<C>, _: &Runtime<C>| {
                collection.insert_record(state, &value)
            }));
    }

    /// Adds a removal from `collection`, as
    /// [`PersistedCollection::delete`] would write it.
    pub fn delete<T: PersistValue + Hash + Eq>(
        &mut self,
        collection: &PersistedCollection<T, C>,
        value: T,
    ) {
        let collection = collection.clone();
        self.ops
            .push(Box::new(move |state: &State<C>, _: &Runtime<C>| {
                collection.delete_record(state, &value)
            }));
    }

    /// Adds a replace in `collection`, as
    /// [`PersistedCollection::replace`] would write it.
    pub fn replace<T: PersistValue + Hash + Eq>(
        &mut self,
        collection: &PersistedCollection<T, C>,
        old: Option<T>,
        new: T,
    ) {
        let collection = collection.clone();
        self.ops
            .push(Box::new(move |state: &State<C>, _: &Runtime<C>| {
                collection.replace_record(state, old.as_ref(), &new)
            }));
    }

    /// Stages `row` for `key`. A row `==` to the one the key held when
    /// first staged leaves the key as it was.
    pub fn upsert<K, V, P>(&mut self, keyed: &PersistedKeyed<K, V, C, P>, key: K, row: V)
    where
        K: Value + Hash + Eq,
        V: Value + Hash + Eq,
        P: Pair<K, V> + PersistValue,
    {
        self.stage(keyed).upsert(key, row);
    }

    /// Stages the removal of `key`; returns whether it held a row as
    /// staged so far.
    pub fn remove<K, V, P>(&mut self, keyed: &PersistedKeyed<K, V, C, P>, key: &K) -> bool
    where
        K: Value + Hash + Eq,
        V: Value + Hash + Eq,
        P: Pair<K, V> + PersistValue,
    {
        self.stage(keyed).remove(key)
    }

    /// Stages `rows` (a later row for a key wins), then the removal of
    /// every held key that nothing staged names. Once the write applies,
    /// any key it holds that the stage does not name is removed too.
    pub fn reconcile<K, V, P>(
        &mut self,
        keyed: &PersistedKeyed<K, V, C, P>,
        rows: impl IntoIterator<Item = (K, V)>,
    ) where
        K: Value + Hash + Eq,
        V: Value + Hash + Eq,
        P: Pair<K, V> + PersistValue,
    {
        self.stage(keyed).reconcile(rows);
    }

    /// How many of `keyed`'s keys the staged writes give a row they
    /// lacked, give another row, and leave without a row.
    pub fn counts<K, V, P>(&self, keyed: &PersistedKeyed<K, V, C, P>) -> KeyCounts
    where
        K: Value + Hash + Eq,
        V: Value + Hash + Eq,
        P: Pair<K, V> + PersistValue,
    {
        self.stages
            .iter()
            .filter_map(|stage| stage.as_any().downcast_ref::<KeyedStage<K, V, C, P>>())
            .find(|stage| stage.is_for(keyed))
            .map_or_else(KeyCounts::default, KeyedStage::counts)
    }

    fn stage<K, V, P>(&mut self, keyed: &PersistedKeyed<K, V, C, P>) -> &mut KeyedStage<K, V, C, P>
    where
        K: Value + Hash + Eq,
        V: Value + Hash + Eq,
        P: Pair<K, V> + PersistValue,
    {
        let at = self.stages.iter().position(|stage| {
            stage
                .as_any()
                .downcast_ref::<KeyedStage<K, V, C, P>>()
                .is_some_and(|stage| stage.is_for(keyed))
        });
        let at = at.unwrap_or_else(|| {
            self.stages.push(Box::new(KeyedStage::new(keyed.clone())));
            self.stages.len() - 1
        });
        self.stages[at]
            .as_any_mut()
            .downcast_mut()
            .expect("the stage at `at` has this type")
    }
}

/// One keyed collection's part of a [`PendingWrite`], erased.
trait Stage<C: Cells> {
    /// Whether the index changed since the stage started, so the held
    /// pairs it staged against may be out of date.
    fn stale(&self) -> bool;

    /// The records that make the staged changes: per key, a `Remove` of
    /// each pair it held, then an `Insert` of its new pair.
    fn records(&self, state: &State<C>) -> Result<Vec<Record>>;

    /// The records that leave each staged key holding exactly its staged
    /// pair, or none, given what the index holds now.
    fn repairs(&self, state: &State<C>) -> Result<Vec<Record>>;

    fn key(&self, state: &State<C>) -> Result<String>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

enum Next<P> {
    Held,
    Row(P),
    Gone,
}

struct StagedKey<K, P> {
    /// The held key object if the index held the key, else the caller's.
    key: K,
    held: Vec<Ranked<P>>,
    next: Next<P>,
}

impl<K, P> StagedKey<K, P> {
    /// The pair read for the key when it was staged: the first of the
    /// smallest rank.
    fn held_read(&self) -> Option<&P> {
        self.held
            .iter()
            .min_by_key(|(_, rank)| *rank)
            .map(|(pair, _)| pair)
    }

    /// [`Self::target`] with the rank it holds once applied: a held
    /// pair's own, a new pair's the fingerprint of its encoding under
    /// `key`.
    fn ranked_target(&self, key: &str) -> Result<Option<(&P, u128)>>
    where
        P: PersistValue,
    {
        Ok(match &self.next {
            Next::Held => self
                .held
                .iter()
                .min_by_key(|(_, rank)| *rank)
                .map(|(pair, rank)| (pair, *rank)),
            Next::Row(pair) => Some((pair, fp_of(&encode(key, pair)?))),
            Next::Gone => None,
        })
    }

    /// The single pair the key is staged to hold, or `None` when it is
    /// staged to hold nothing.
    fn target(&self) -> Option<&P> {
        match &self.next {
            Next::Held => self.held_read(),
            Next::Row(pair) => Some(pair),
            Next::Gone => None,
        }
    }
}

struct KeyedStage<K, V, C, P>
where
    K: 'static,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
{
    keyed: PersistedKeyed<K, V, C, P>,
    epoch: u64,
    keys: Vec<StagedKey<K, P>>,
    by_key: HashMap<K, usize>,
    /// Set by a reconcile: the stage names every key the collection is to
    /// hold.
    complete: bool,
}

impl<K, V, C, P> KeyedStage<K, V, C, P>
where
    K: Value + Hash + Eq,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V> + PersistValue,
{
    fn new(keyed: PersistedKeyed<K, V, C, P>) -> Self {
        KeyedStage {
            epoch: keyed.epoch(),
            keyed,
            keys: Vec::new(),
            by_key: HashMap::new(),
            complete: false,
        }
    }

    fn is_for(&self, keyed: &PersistedKeyed<K, V, C, P>) -> bool {
        self.keyed.id == keyed.id && self.keyed.version_node() == keyed.version_node()
    }

    fn locate(&mut self, key: K) -> usize {
        if let Some(&at) = with_value_code(|| self.by_key.get(&key)) {
            return at;
        }
        let held = self.keyed.keyed.held_pairs(&key);
        self.push(key, held, Next::Held)
    }

    fn push(&mut self, key: K, held: Vec<Ranked<P>>, next: Next<P>) -> usize {
        let at = self.keys.len();
        let key = with_value_code(|| {
            let key = held.first().map_or(key, |(pair, _)| pair.key());
            self.by_key.insert(key.clone(), at);
            key
        });
        self.keys.push(StagedKey { key, held, next });
        at
    }

    fn upsert(&mut self, key: K, row: V) {
        let at = self.locate(key);
        let staged = &mut self.keys[at];
        with_value_code(|| {
            if !staged.target().is_some_and(|pair| pair.row_eq(&row)) {
                staged.next = Next::Row(P::new(staged.key.clone(), row));
            }
        });
    }

    fn remove(&mut self, key: &K) -> bool {
        let at = self.locate(with_value_code(|| key.clone()));
        let staged = &mut self.keys[at];
        let present = staged.target().is_some();
        staged.next = Next::Gone;
        present
    }

    fn reconcile(&mut self, rows: impl IntoIterator<Item = (K, V)>) {
        for (key, row) in rows {
            self.upsert(key, row);
        }
        self.complete = true;
        let held = self.keyed.keyed.ranked_pairs();
        let mut rest: Vec<(K, Vec<Ranked<P>>)> = Vec::new();
        with_value_code(|| {
            let mut at: HashMap<K, usize> = HashMap::new();
            for ranked in held {
                let key = ranked.0.key();
                if self.by_key.contains_key(&key) {
                    continue;
                }
                match at.get(&key) {
                    Some(&i) => rest[i].1.push(ranked),
                    None => {
                        at.insert(key.clone(), rest.len());
                        rest.push((key, vec![ranked]));
                    }
                }
            }
        });
        for (key, held) in rest {
            self.push(key, held, Next::Gone);
        }
    }

    fn counts(&self) -> KeyCounts {
        let mut counts = KeyCounts::default();
        for staged in &self.keys {
            match (staged.held_read(), &staged.next) {
                (_, Next::Held) | (None, Next::Gone) => {}
                (None, Next::Row(_)) => counts.inserted += 1,
                (Some(_), Next::Row(_)) => counts.replaced += 1,
                (Some(_), Next::Gone) => counts.removed += 1,
            }
        }
        counts
    }
}

impl<K, V, C, P> Stage<C> for KeyedStage<K, V, C, P>
where
    K: Value + Hash + Eq,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V> + PersistValue,
{
    fn stale(&self) -> bool {
        self.keyed.epoch() != self.epoch
    }

    fn records(&self, state: &State<C>) -> Result<Vec<Record>> {
        let key = self.key(state)?;
        let mut out = Vec::new();
        for staged in &self.keys {
            let next = match &staged.next {
                Next::Held => continue,
                Next::Row(pair) => Some(pair),
                Next::Gone => None,
            };
            for (pair, _) in &staged.held {
                out.push(remove_record(&key, pair)?);
            }
            if let Some(pair) = next {
                out.push(insert_record(&key, pair)?);
            }
        }
        Ok(out)
    }

    fn repairs(&self, state: &State<C>) -> Result<Vec<Record>> {
        let key = self.key(state)?;
        let mut fixes = Vec::new();
        for staged in &self.keys {
            let checked = match staged.next {
                Next::Held => self.complete && !staged.held.is_empty(),
                Next::Gone => !staged.held.is_empty(),
                Next::Row(_) => true,
            };
            if !checked {
                continue;
            }
            let target = staged.ranked_target(&key)?;
            let held = self.keyed.keyed.held_pairs(&staged.key);
            let keep = with_value_code(|| {
                target.and_then(|target| find_element(held.iter().map(|(p, r)| (p, *r)), target))
            });
            let target = target.map(|(pair, _)| pair);
            if let (Some(target), None) = (target, keep) {
                fixes.push(insert_record(&key, target)?);
            }
            for (at, (pair, _)) in held.iter().enumerate() {
                if Some(at) != keep {
                    fixes.push(remove_record(&key, pair)?);
                }
            }
        }
        if self.complete {
            let held = self.keyed.keyed.ranked_pairs();
            let unnamed: Vec<&P> = with_value_code(|| {
                held.iter()
                    .map(|(pair, _)| pair)
                    .filter(|pair| !self.by_key.contains_key(&pair.key()))
                    .collect()
            });
            for pair in unnamed {
                fixes.push(remove_record(&key, pair)?);
            }
        }
        Ok(fixes)
    }

    fn key(&self, state: &State<C>) -> Result<String> {
        self.keyed
            .binding_in(state)
            .map(|binding| binding.key.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn insert_record<P: PersistValue>(key: &str, pair: &P) -> Result<Record> {
    Ok(Record::Insert {
        key: key.to_owned(),
        element: encode(key, pair)?,
    })
}

fn remove_record<P: PersistValue>(key: &str, pair: &P) -> Result<Record> {
    Ok(Record::Remove {
        key: key.to_owned(),
        element: encode(key, pair)?,
    })
}

fn decode_ranked<P: PersistValue>(key: &str, bytes: &[u8]) -> Result<Ranked<P>> {
    Ok((decode(key, bytes)?, fp_of(bytes)))
}

/// The ops that rebuild `elements` from nothing: an insert per live
/// occurrence in the order first inserted, then the raw tail.
fn element_ops<P: PersistValue>(key: &str, elements: &Elements) -> Result<Vec<PairOp<P, u128>>> {
    let mut ops = Vec::new();
    for (bytes, n) in elements.live() {
        let (pair, rank) = decode_ranked::<P>(key, bytes)?;
        for _ in 1..n {
            ops.push(PairOp::Insert(with_value_code(|| pair.clone()), rank));
        }
        ops.push(PairOp::Insert(pair, rank));
    }
    for op in elements.tail() {
        ops.push(match op {
            ElementOp::Insert(bytes) => {
                let (pair, rank) = decode_ranked(key, bytes)?;
                PairOp::Insert(pair, rank)
            }
            ElementOp::Remove(bytes) => {
                let (pair, rank) = decode_ranked(key, bytes)?;
                PairOp::Remove(pair, rank)
            }
        });
    }
    Ok(ops)
}

/// The ranked multiset `ops` build from nothing, a removal taking the
/// held pair [`find_element`] names, as the index's own replay does.
fn fold_ops<P: Eq>(ops: Vec<PairOp<P, u128>>) -> Vec<Ranked<P>> {
    let mut held: Vec<Ranked<P>> = Vec::new();
    with_value_code(|| {
        for op in ops {
            match op {
                PairOp::Insert(pair, rank) => held.push((pair, rank)),
                PairOp::Remove(pair, rank) => {
                    let found = find_element(held.iter().map(|(p, r)| (p, *r)), (&pair, rank));
                    if let Some(at) = found {
                        held.swap_remove(at);
                    }
                }
            }
        }
    });
    held
}

/// Counts of ranked pairs by element: by rank when the rank identifies
/// the element, else by the pair at that rank, so pairs that are one
/// element by their rank match even when they are not `==`.
struct ElementCounts<'a, P> {
    by_rank: HashMap<u128, usize>,
    by_pair: HashMap<(&'a P, u128), usize>,
}

impl<P> Default for ElementCounts<'_, P> {
    fn default() -> Self {
        ElementCounts {
            by_rank: HashMap::new(),
            by_pair: HashMap::new(),
        }
    }
}

impl<'a, P: Hash + Eq> ElementCounts<'a, P> {
    fn entry(&mut self, (pair, rank): &'a Ranked<P>) -> &mut usize {
        if rank.identifies() {
            self.by_rank.entry(*rank).or_default()
        } else {
            self.by_pair.entry((pair, *rank)).or_default()
        }
    }
}

impl<K, V, C, P> KeyedBinding<K, V, C, P>
where
    K: Value + Hash + Eq,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V> + PersistValue,
{
    /// The fingerprint of the pairs, leaving the copies it read in `read`
    /// for the caller to drop once user `Drop` may run.
    fn fp(&self, rt: &Runtime<C>, read: &mut Vec<P>) -> DepFp {
        *read = self.keyed.pairs().values(rt);
        match fp_of_elements(&self.key, read) {
            Ok(fp) => DepFp::Known(fp),
            Err(_) => DepFp::Untrusted,
        }
    }

    /// Applies `ops` as one mapped write. With `deps`, the fingerprint
    /// moves by what the pair collection reports, or is recomputed from
    /// the pairs when `recompute`, which also trusts it again.
    fn apply_ops(
        &self,
        rt: &Runtime<C>,
        ops: Vec<PairOp<P, u128>>,
        recompute: bool,
        deps: Option<&DepIndex>,
    ) {
        let change: KeyedChange<K>;
        let mut spent = KeyedSpent::default();
        let mut read = Vec::new();
        let mut hold = deps.map(DepIndex::write);
        let mut guard = PanicGuard::new(hold.as_deref_mut(), &self.key);
        change = if guard.deps().is_some() && !recompute {
            let mut sink = FpDelta::new(&self.key);
            let change = self
                .keyed
                .apply_pairs_keeping(rt, ops, &mut sink, None, &mut spent);
            if let Some(map) = guard.deps() {
                sink.apply_to(map, &self.key);
            }
            change
        } else {
            self.keyed
                .apply_pairs_keeping(rt, ops, &mut (), None, &mut spent)
        };
        if recompute && guard.deps().is_some() {
            let fp = self.fp(rt, &mut read);
            if let Some(map) = guard.deps() {
                map.store_fp(&self.key, fp);
            }
        }
        // The keyed write set the changed keys' version inputs inside the
        // hold, so a memo checkpoint never pairs a row with an older entry.
        guard.disarm();
        drop(hold);
        drop((change, spent, read));
    }

    /// Brings the pairs to exactly `target`, ranks included, removing
    /// first, and pushes onto `changes`, when given, the record of each
    /// change.
    fn converge(
        &self,
        rt: &Runtime<C>,
        target: Vec<Ranked<P>>,
        changes: Option<&mut Vec<Record>>,
        deps: Option<&DepIndex>,
    ) -> Result<()> {
        let current = self.keyed.ranked_pairs();
        let (removed, inserted) = with_value_code(|| {
            let mut have = ElementCounts::default();
            for ranked in &current {
                *have.entry(ranked) += 1;
            }
            let mut inserted = Vec::new();
            for ranked in &target {
                match have.entry(ranked) {
                    n if *n > 0 => *n -= 1,
                    _ => inserted.push(ranked.clone()),
                }
            }
            let mut removed = Vec::new();
            for ranked in &current {
                let n = have.entry(ranked);
                if *n > 0 {
                    *n -= 1;
                    removed.push(ranked.clone());
                }
            }
            (removed, inserted)
        });
        if let Some(changes) = changes {
            for (pair, _) in &removed {
                changes.push(remove_record(&self.key, pair)?);
            }
            for (pair, _) in &inserted {
                changes.push(insert_record(&self.key, pair)?);
            }
        }
        let ops = removed
            .into_iter()
            .map(|(pair, rank)| PairOp::Remove(pair, rank))
            .chain(
                inserted
                    .into_iter()
                    .map(|(pair, rank)| PairOp::Insert(pair, rank)),
            )
            .collect();
        self.apply_ops(rt, ops, true, deps);
        Ok(())
    }

    /// Every pair encoded, in byte order, as a plain collection binding
    /// lists its elements.
    fn encoded(&self, rt: &Runtime<C>) -> Result<Vec<Vec<u8>>> {
        let mut encoded = self
            .keyed
            .pairs()
            .values(rt)
            .iter()
            .map(|pair| encode(&self.key, pair))
            .collect::<Result<Vec<_>>>()?;
        encoded.sort_unstable();
        Ok(encoded)
    }
}

impl<K, V, C, P> Binding<C> for KeyedBinding<K, V, C, P>
where
    K: Value + Hash + Eq,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V> + PersistValue,
{
    fn kind(&self) -> Kind {
        Kind::Collection
    }

    fn key(&self) -> &str {
        &self.key
    }

    fn is_handle(&self, handle: &dyn Any) -> bool {
        handle.downcast_ref::<Incr<u64>>() == Some(&self.keyed.pairs().version_node())
    }

    fn node(&self) -> ErasedIncr {
        self.keyed.pairs().version_node().erase()
    }

    /// Deletes the key's version inputs with the collection, so a query
    /// reading any key through `get_key` counts as a dependent.
    fn delete_nodes(&self, rt: &Runtime<C>) -> std::result::Result<(), DeleteError> {
        let mut nodes = vec![self.node(), self.keyed.rows().version_node().erase()];
        nodes.extend(self.keyed.version_inputs());
        rt.try_delete_nodes(&nodes)
    }

    fn unbind(&self, rt: &Runtime<C>) -> Result<Option<Elements>> {
        if !rt.is_alive(self.node()) {
            return Err(Error::UnknownHandle);
        }
        Ok(Some(Elements::from_encoded(self.encoded(rt)?)))
    }

    fn dep_fp(&self, rt: &Runtime<C>) -> Result<DepFp> {
        if !rt.is_alive(self.node()) {
            return Ok(DepFp::Untrusted);
        }
        Ok(DepFp::Known(fp_of_elements(
            &self.key,
            &self.keyed.pairs().values(rt),
        )?))
    }

    fn mapped_stamp(&self) -> Option<u64> {
        None
    }

    fn row_source(&self) -> Option<Box<dyn Any + Send + Sync>> {
        Some(Box::new(Arc::clone(&self.rows)))
    }

    fn apply(
        &mut self,
        rt: &Runtime<C>,
        record: &Record,
        _before: Option<&LoggedKey>,
        deps: Option<&DepIndex>,
    ) -> Result<()> {
        let ops = match record {
            Record::Insert { element, .. } => {
                let (pair, rank) = decode_ranked(&self.key, element)?;
                vec![PairOp::Insert(pair, rank)]
            }
            Record::Remove { element, .. } => {
                let (pair, rank) = decode_ranked(&self.key, element)?;
                vec![PairOp::Remove(pair, rank)]
            }
            Record::Replace { old, new, .. } => {
                let old = old
                    .as_deref()
                    .map(|old| decode_ranked(&self.key, old))
                    .transpose()?;
                let (pair, rank) = decode_ranked(&self.key, new)?;
                old.map(|(pair, rank)| PairOp::Remove(pair, rank))
                    .into_iter()
                    .chain([PairOp::Insert(pair, rank)])
                    .collect()
            }
            Record::Delete { .. } | Record::Forget { .. } => {
                return self.converge(rt, Vec::new(), None, deps);
            }
            Record::Declare { .. } | Record::Set { .. } => return Ok(()),
        };
        self.apply_ops(rt, ops, false, deps);
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
                let target = fold_ops(element_ops(&self.key, &logged.elements)?);
                logged.elements = Elements::default();
                target
            }
            None => Vec::new(),
        };
        self.converge(rt, target, changes, deps)
    }

    // The handle a PersistedKeyed exposes only reads, so the pairs cannot
    // be written around the store but in an unlogged scope.
    #[cfg(debug_assertions)]
    fn check_bypass(&self, _rt: &Runtime<C>, _logged: Option<&LoggedKey>) -> Result<()> {
        Ok(())
    }

    fn restamp(
        &mut self,
        _rt: &Runtime<C>,
        _logged: Option<&LoggedKey>,
    ) -> Result<Option<(u128, u64)>> {
        Ok(None)
    }

    fn fold(&self, rt: &Runtime<C>, logged: Option<&LoggedKey>) -> Result<Vec<Record>> {
        let elements = self.encoded(rt)?;
        let has_value = logged.is_some_and(|logged| logged.value.is_some());
        if elements.is_empty() && !has_value {
            return Ok(Vec::new());
        }
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

impl<C: Cells, B: Backend> Store<C, B> {
    /// Declares the keyed collection `key`, a collection of `P` pairs, and
    /// returns its handle.
    ///
    /// Declares as [`Store::collection`] does, and the collection starts
    /// with the key's logged pairs. Fails with [`Error::TypeMismatch`] when
    /// a logged pair does not decode as `P` or the key is already bound to
    /// another type, a plain collection included, and with
    /// [`Error::KindMismatch`] when the key is an input.
    ///
    /// A memo entry that reads a key through [`PersistedKeyed::get_key`]
    /// stores the key, encoded under `key`, and its row's fingerprint, and
    /// is reused after a restart while that row is unchanged.
    pub fn keyed_collection<K, V, P>(&self, key: &str) -> Result<PersistedKeyed<K, V, C, P>>
    where
        K: PersistValue + Hash + Eq,
        V: Value + Hash + Eq,
        P: Pair<K, V> + PersistValue,
    {
        let mut locked = self.state()?;
        let state = &mut *locked;
        if let Some(id) = state.bound_id(key, Kind::Collection)? {
            return state.bindings[&id]
                .as_any()
                .downcast_ref::<KeyedBinding<K, V, C, P>>()
                .map(|binding| PersistedKeyed {
                    id,
                    keyed: binding.keyed.clone(),
                })
                .ok_or_else(|| Error::TypeMismatch {
                    key: key.to_owned(),
                    source: None,
                });
        }
        self.declare(state, key, Kind::Collection, || Ok(Vec::new()))?;
        let logged = state.logged.get_mut(key);
        let ops = match &logged {
            Some(logged) => element_ops::<P>(key, &logged.elements)?,
            None => Vec::new(),
        };
        if let Some(logged) = logged {
            logged.elements = Elements::default();
        }
        let keyed = KeyedCollection::new(&self.runtime);
        let rows: Rows<C> = Arc::new(KeyedRows {
            key: key.to_owned(),
            keyed: keyed.clone(),
        });
        let binding = KeyedBinding {
            key: key.to_owned(),
            keyed,
            rows,
        };
        let mut spent = KeyedSpent::default();
        let mut read = Vec::new();
        let mut hold = state.deps.as_deref().map(DepIndex::write);
        let mut guard = PanicGuard::new(hold.as_deref_mut(), key);
        let change =
            binding
                .keyed
                .apply_pairs_keeping(&self.runtime, ops, &mut (), None, &mut spent);
        if guard.deps().is_some() {
            let fp = binding.fp(&self.runtime, &mut read);
            let node = Binding::<C>::node(&binding);
            let rows = Binding::<C>::row_source(&binding);
            if let Some(map) = guard.deps() {
                map.register(key, node, Kind::Collection, fp, None);
                if let Some(rows) = rows {
                    map.register_rows(key, rows);
                }
            }
        }
        guard.disarm();
        drop(hold);
        drop((change, spent, read));
        let keyed = binding.keyed.clone();
        let id = state.bind(key, Box::new(binding));
        Ok(PersistedKeyed { id, keyed })
    }

    /// Appends everything `pending` collected as one run and applies it,
    /// returning the last record's position, or the position last applied
    /// when there is nothing to append.
    ///
    /// When a keyed collection's index changed after its stage started, or
    /// the append landed behind records of other writers, the stage's rows
    /// may have been computed from pairs since replaced. The write then
    /// checks each staged key and appends fixes, so that every one ends
    /// holding exactly its staged row as its only pair, and a reconciled
    /// collection holds no other key. A fix that lands behind further
    /// foreign records is checked again, for up to three rounds of fixes,
    /// after which it fails with [`Error::KeyedContention`], leaving the
    /// store usable with every record it appended applied. Other errors
    /// behave as for [`Store::batch`].
    pub fn write_pending(&self, pending: PendingWrite<C>) -> Result<Position> {
        let mut locked = self.state()?;
        self.write_pending_with(&mut locked, pending, &mut |state, records| {
            self.write_counted(state, records)
        })
    }

    /// [`Store::write_pending`] with the store locked, handing each run of
    /// records to `write`, which returns the position it reached and
    /// whether it applied records not in the run.
    pub(super) fn write_pending_with(
        &self,
        state: &mut State<C>,
        pending: PendingWrite<C>,
        write: &mut WriteRun<'_, C>,
    ) -> Result<Position> {
        let PendingWrite { ops, stages } = pending;
        let mut records = Vec::new();
        for op in ops.0 {
            records.push(op(state, &self.runtime)?);
        }
        let mut stale = false;
        for stage in stages.iter() {
            records.extend(stage.records(state)?);
            stale |= stage.stale();
        }
        if records.is_empty() && !stale {
            return Ok(state.applied);
        }
        let mut last = state.applied;
        let mut overtaken = stale;
        if !records.is_empty() {
            let (at, foreign) = write(state, records)?;
            last = at;
            overtaken |= foreign;
        }
        if overtaken {
            if let Some(at) = self.repair(state, &stages, write)? {
                last = at;
            }
        }
        Ok(last)
    }

    /// Writes `row` for `key` in `keyed`, as a [`PendingWrite`] holding
    /// only that upsert would.
    pub fn upsert<K, V, P>(
        &self,
        keyed: &PersistedKeyed<K, V, C, P>,
        key: K,
        row: V,
    ) -> Result<Position>
    where
        K: Value + Hash + Eq,
        V: Value + Hash + Eq,
        P: Pair<K, V> + PersistValue,
    {
        let mut pending = PendingWrite::new();
        pending.upsert(keyed, key, row);
        self.write_pending(pending)
    }

    /// Removes `key` from `keyed`, as [`Store::upsert`] writes.
    pub fn remove<K, V, P>(&self, keyed: &PersistedKeyed<K, V, C, P>, key: &K) -> Result<Position>
    where
        K: Value + Hash + Eq,
        V: Value + Hash + Eq,
        P: Pair<K, V> + PersistValue,
    {
        let mut pending = PendingWrite::new();
        pending.remove(keyed, key);
        self.write_pending(pending)
    }

    /// Makes `keyed` hold exactly `rows`, as [`PendingWrite::reconcile`]
    /// stages it.
    pub fn reconcile<K, V, P>(
        &self,
        keyed: &PersistedKeyed<K, V, C, P>,
        rows: impl IntoIterator<Item = (K, V)>,
    ) -> Result<Position>
    where
        K: Value + Hash + Eq,
        V: Value + Hash + Eq,
        P: Pair<K, V> + PersistValue,
    {
        let mut pending = PendingWrite::new();
        pending.reconcile(keyed, rows);
        self.write_pending(pending)
    }

    /// [`Store::write`], also returning whether any record not in
    /// `records` was applied along the way.
    fn write_counted(
        &self,
        state: &mut State<C>,
        records: Vec<Record>,
    ) -> Result<(Position, bool)> {
        let before = state.applied;
        let len = records.len() as u64;
        let last = self.write(state, records)?;
        Ok((last, last.0 - before.0 != len))
    }

    /// Appends fixes until every stage's keys hold what it staged; returns
    /// the last fix's position, if any was appended.
    fn repair(
        &self,
        state: &mut State<C>,
        stages: &[Box<dyn Stage<C>>],
        write: &mut WriteRun<'_, C>,
    ) -> Result<Option<Position>> {
        let mut last = None;
        for round in 0..=REPAIR_ROUNDS {
            let mut fixes = Vec::new();
            let mut contended = None;
            for stage in stages {
                let stage_fixes = stage.repairs(state)?;
                if !stage_fixes.is_empty() && contended.is_none() {
                    contended = Some(stage.key(state)?);
                }
                fixes.extend(stage_fixes);
            }
            let Some(key) = contended else {
                return Ok(last);
            };
            if round == REPAIR_ROUNDS {
                return Err(Error::KeyedContention { key });
            }
            let (at, foreign) = write(state, fixes)?;
            last = Some(at);
            if !foreign {
                return Ok(last);
            }
        }
        Ok(last)
    }
}
