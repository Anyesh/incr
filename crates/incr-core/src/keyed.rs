//! Keyed collections: rows held by key, over a pair collection of `(key,
//! row)` elements and a rows collection mapped from it. A key index finds a
//! key's pairs without a scan, and every write goes through it, so the
//! index and the pair collection hold the same multiset of pairs.
//!
//! A query reads one key's row with [`KeyedCollection::get_key`], which
//! depends on that key's version input and reads the row from the index
//! untracked, as `IncrCollection::values` reads its log. Every write sets
//! the version input of each key whose row it changed, so a reader reruns
//! only for its own keys.
//!
//! A write runs as `IncrCollection::publish` does. It stages under the
//! index's read guard, as value code: user `Hash`, key and row `Eq`, and
//! `Clone`, changing nothing. It then takes the pair collection's log guard
//! and the index, restaging if another writer advanced the index's epoch
//! meanwhile, runs the log's `Eq` lookups, and changes the log and the
//! index with no user code in between. It releases both, sets the version,
//! and only then drops what it displaced and routes.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher, RandomState};
use std::marker::PhantomData;
use std::mem;
use std::sync::{Arc, Weak};

use crate::cells::Cells;
use crate::collection::{Change, Displaced, IncrCollection, StagedBatch};
use crate::dep_stack::DepStack;
use crate::few::Few;
use crate::fx::FxBuildHasher;
use crate::handle::{ErasedIncr, Incr};
use crate::lock_order::LockKind;
use crate::locks::OrderedLock;
use crate::memory::{Held, StateKind, StateMemory, StateStats};
use crate::persist_hooks::RepSink;
use crate::runtime::{DeleteError, Runtime};
use crate::value::Value;
use crate::value_code;

mod parts;
use parts::{Flip, PartSink, PartState};

/// An element of a keyed collection's pair collection: a key and its row.
///
/// `key_hash` must feed a hasher exactly what `K::hash` of the pair's key
/// would, `key_eq` and `row_eq` must agree with `K::eq` and `V::eq` on the
/// pair's key and row, and two pairs must be `==` exactly when both their
/// keys and their rows are. The engine runs every method but `row` as
/// value code, so none of them may call into the runtime; `row` runs as
/// the rows collection's map closure.
pub trait Pair<K, V>: Value + Hash + Eq {
    fn new(key: K, row: V) -> Self;
    /// A copy of the pair's key.
    fn key(&self) -> K;
    /// A copy of the pair's row.
    fn row(&self) -> V;
    fn key_hash<H: Hasher>(&self, state: &mut H);
    fn key_eq(&self, key: &K) -> bool;
    fn row_eq(&self, row: &V) -> bool;
}

impl<K: Value + Hash + Eq, V: Value + Hash + Eq> Pair<K, V> for (K, V) {
    fn new(key: K, row: V) -> Self {
        (key, row)
    }

    fn key(&self) -> K {
        self.0.clone()
    }

    fn row(&self) -> V {
        self.1.clone()
    }

    fn key_hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }

    fn key_eq(&self, key: &K) -> bool {
        self.0 == *key
    }

    fn row_eq(&self, row: &V) -> bool {
        self.1 == *row
    }
}

/// How many keys a write gave a row they lacked, gave a different row,
/// and left without a row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeyCounts {
    pub inserted: usize,
    pub replaced: usize,
    pub removed: usize,
}

/// What a keyed write changed. `changed` holds a key once for every change
/// of the row read for it: for `upsert`, `remove` and `reconcile` exactly
/// the keys whose row differs after the call, once each; for
/// `apply_pairs_with` one entry per op that changed its key's row at its
/// point in the sequence, so a key can appear more than once. `epoch` is
/// the index's epoch after the write, which advances by one on every write
/// that changes the index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyedChange<K> {
    pub counts: KeyCounts,
    pub changed: Vec<K>,
    pub epoch: u64,
}

/// The rank a keyed collection orders a key's pairs by. A rank that
/// `identifies` names its element: two pairs of one key with equal such
/// ranks are the same element whether or not they are `==`, as two pairs
/// applied from the same bytes are, so two pairs that are not one element
/// must never share an identifying rank.
pub trait ElementRank: Ord + Copy + Default {
    fn identifies(&self) -> bool;
}

impl ElementRank for () {
    fn identifies(&self) -> bool {
        false
    }
}

/// The `u128` rank that never identifies a pair: no logged bytes back the
/// pair, so equal ranks of this value say nothing about the pairs.
pub const UNIDENTIFYING_RANK: u128 = 0;

/// The fingerprint of the bytes a persisted pair was applied from;
/// [`UNIDENTIFYING_RANK`] marks a pair no logged bytes back.
impl ElementRank for u128 {
    fn identifies(&self) -> bool {
        *self != UNIDENTIFYING_RANK
    }
}

/// Whether the ranked pairs `a` and `b` are one element: their ranks are
/// equal and identify it, or the pairs are `==`. Runs `P::eq` only when
/// the ranks do not settle it.
pub fn same_element<P: Eq, R: ElementRank>(a: (&P, R), b: (&P, R)) -> bool {
    (a.1 == b.1 && a.1.identifies()) || a.0 == b.0
}

/// The position in `held` of the element `target` names: the first
/// [`same_element`] of `target`'s rank, else the first of any rank.
pub fn find_element<'a, P: Eq + 'a, R: ElementRank>(
    mut held: impl Iterator<Item = (&'a P, R)> + Clone,
    target: (&P, R),
) -> Option<usize> {
    held.clone()
        .position(|e| e.1 == target.1 && same_element(e, target))
        .or_else(|| held.position(|e| same_element(e, target)))
}

/// One change to a keyed collection's pairs, with the pair's rank. Of the
/// pairs held for one key, the first with the smallest rank is read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PairOp<P, R> {
    Insert(P, R),
    /// Removes one held pair that is the [`same_element`] as this one,
    /// preferring one of this rank; a no-op if no such pair is held.
    Remove(P, R),
}

impl<P, R> PairOp<P, R> {
    /// The op that undoes this one, as journalled by `apply_pairs_with`.
    pub fn inverse(self) -> Self {
        match self {
            PairOp::Insert(pair, rank) => PairOp::Remove(pair, rank),
            PairOp::Remove(pair, rank) => PairOp::Insert(pair, rank),
        }
    }
}

fn few_heap_bytes<T>(few: &Few<T>) -> usize {
    match few {
        Few::Many(v) => v.capacity() * mem::size_of::<T>(),
        Few::None | Few::One(_) => 0,
    }
}

fn few_into_vec<T>(few: Few<T>) -> Vec<T> {
    match few {
        Few::None => Vec::new(),
        Few::One(x) => vec![x],
        Few::Many(v) => v,
    }
}

fn few_of<T>(mut v: Vec<T>) -> Few<T> {
    match v.len() {
        0 => Few::None,
        1 => Few::One(v.swap_remove(0)),
        _ => Few::Many(v),
    }
}

/// The index of the first of the smallest ranks.
fn first_smallest<R: Ord + Copy>(ranks: impl Iterator<Item = R>) -> Option<usize> {
    ranks.enumerate().min_by_key(|(_, r)| *r).map(|(i, _)| i)
}

fn load_limit(capacity: usize) -> usize {
    capacity / 4 * 3
}

struct Bucket<P, R> {
    hash: u64,
    /// Never empty while the bucket is in the table.
    pairs: Few<(P, R)>,
}

impl<P, R: Ord + Copy> Bucket<P, R> {
    fn pairs(&self) -> &[(P, R)] {
        self.pairs.as_slice()
    }

    fn first(&self) -> &P {
        &self.pairs()[0].0
    }

    fn read_at(&self) -> usize {
        first_smallest(self.pairs().iter().map(|p| p.1)).expect("a bucket holds a pair")
    }

    fn read(&self) -> &(P, R) {
        &self.pairs()[self.read_at()]
    }
}

enum Slot<P, R> {
    Empty,
    Tombstone,
    Full(Bucket<P, R>),
}

/// The table behind a [`KeyIndex`]: one bucket per key, found by the key's
/// hash, holding that key's pairs with their ranks. Open addressing with
/// stored hashes, as `HashedMultiset`: only a lookup runs user code, only
/// `reserve` moves buckets, and removal leaves a tombstone unless no probe
/// run passes through the slot.
pub struct KeyTable<P, R> {
    // Invariant: the capacity is zero or a power of two with at least one
    // Empty slot, so every probe loop ends.
    slots: Vec<Slot<P, R>>,
    keys: usize,
    pairs: usize,
    growth_left: usize,
    spilled_bytes: usize,
    /// The key-set partitionings registered with
    /// [`KeyedCollection::key_partitions`], changed with each commit.
    parts: Vec<Box<dyn PartSink<P>>>,
}

impl<P, R: Ord + Copy> KeyTable<P, R> {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            keys: 0,
            pairs: 0,
            growth_left: 0,
            spilled_bytes: 0,
            parts: Vec::new(),
        }
    }

    /// A plan per partitioning for `flips`, run as value code against the
    /// table as it is; empty when none is registered.
    fn plan_parts(&self, flips: &[Flip<'_, P>]) -> Vec<Box<dyn Any + Send>> {
        self.parts.iter().map(|sink| sink.plan(flips)).collect()
    }

    /// Applies the plans made against this table, running no user code,
    /// and returns the (partitioning, partition hash) pairs they touch.
    fn apply_parts(
        &mut self,
        plans: Vec<Box<dyn Any + Send>>,
        gone: &mut Vec<P>,
    ) -> Vec<(usize, u64)> {
        if self.parts.is_empty() {
            return Vec::new();
        }
        let mut touched = Vec::new();
        for (sink, (at, plan)) in self.parts.iter_mut().zip(plans.into_iter().enumerate()) {
            touched.extend(sink.touched(&*plan).into_iter().map(|hash| (at, hash)));
            sink.apply(plan, gone);
        }
        touched
    }

    fn heap_bytes(&self) -> usize {
        self.slots.capacity() * mem::size_of::<Slot<P, R>>() + self.spilled_bytes
    }

    /// The slot of the bucket with `hash` that `matches` accepts. Runs
    /// `matches` (user `Eq`) and changes nothing.
    fn find(&self, hash: u64, mut matches: impl FnMut(&Bucket<P, R>) -> bool) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }
        let mask = self.slots.len() - 1;
        let mut i = hash as usize & mask;
        loop {
            match &self.slots[i] {
                Slot::Empty => return None,
                Slot::Full(b) if b.hash == hash && matches(b) => return Some(i),
                Slot::Full(_) | Slot::Tombstone => {}
            }
            i = (i + 1) & mask;
        }
    }

    fn find_key<K, V>(&self, hash: u64, key: &K) -> Option<usize>
    where
        P: Pair<K, V>,
    {
        self.find(hash, |b| b.first().key_eq(key))
    }

    fn bucket(&self, i: usize) -> &Bucket<P, R> {
        match &self.slots[i] {
            Slot::Full(b) => b,
            Slot::Empty | Slot::Tombstone => panic!("incr-core: key table slot {i} is not live"),
        }
    }

    fn buckets(&self) -> impl Iterator<Item = (usize, &Bucket<P, R>)> {
        self.slots.iter().enumerate().filter_map(|(i, s)| match s {
            Slot::Full(b) => Some((i, b)),
            Slot::Empty | Slot::Tombstone => None,
        })
    }

    /// Makes room for `additional` more keys. Runs no user code. When it
    /// rehashes, returns where each old slot's bucket moved.
    fn reserve(&mut self, additional: usize) -> Option<Vec<usize>> {
        (additional > self.growth_left).then(|| self.rehash(additional))
    }

    #[cold]
    fn rehash(&mut self, additional: usize) -> Vec<usize> {
        let need = self
            .keys
            .checked_add(additional)
            .expect("incr-core: key table capacity overflow");
        let limit = load_limit(self.slots.len());
        let capacity = if need <= limit / 2 {
            self.slots.len()
        } else {
            let need = need.max(limit + 1);
            let mut capacity = 8_usize;
            while load_limit(capacity) < need {
                capacity = capacity
                    .checked_mul(2)
                    .expect("incr-core: key table capacity overflow");
            }
            capacity
        };
        let old = mem::replace(
            &mut self.slots,
            std::iter::repeat_with(|| Slot::Empty)
                .take(capacity)
                .collect(),
        );
        let mut moved = vec![usize::MAX; old.len()];
        for (at, slot) in old.into_iter().enumerate() {
            if let Slot::Full(b) = slot {
                let j = self.free_slot(b.hash);
                self.slots[j] = Slot::Full(b);
                moved[at] = j;
            }
        }
        self.growth_left = load_limit(capacity) - self.keys;
        moved
    }

    fn free_slot(&self, hash: u64) -> usize {
        let mask = self.slots.len() - 1;
        let mut i = hash as usize & mask;
        while matches!(self.slots[i], Slot::Full(_)) {
            i = (i + 1) & mask;
        }
        i
    }

    fn count_in(&mut self, pairs: &Few<(P, R)>) {
        self.pairs += pairs.as_slice().len();
        self.spilled_bytes += few_heap_bytes(pairs);
    }

    fn count_out(&mut self, pairs: &Few<(P, R)>) {
        self.pairs -= pairs.as_slice().len();
        self.spilled_bytes -= few_heap_bytes(pairs);
    }

    /// Adds a bucket for a key no bucket holds. Requires room from
    /// `reserve`.
    fn insert(&mut self, hash: u64, pairs: Few<(P, R)>) {
        let j = self.free_slot(hash);
        if matches!(self.slots[j], Slot::Empty) {
            assert!(
                self.growth_left > 0,
                "incr-core: key table insert without a reserve"
            );
            self.growth_left -= 1;
        }
        self.count_in(&pairs);
        self.slots[j] = Slot::Full(Bucket { hash, pairs });
        self.keys += 1;
    }

    fn replace(&mut self, i: usize, pairs: Few<(P, R)>) -> Few<(P, R)> {
        self.count_in(&pairs);
        let Slot::Full(b) = &mut self.slots[i] else {
            panic!("incr-core: key table slot {i} is not live");
        };
        let old = mem::replace(&mut b.pairs, pairs);
        self.count_out(&old);
        old
    }

    fn remove(&mut self, i: usize) -> Few<(P, R)> {
        let next = (i + 1) & (self.slots.len() - 1);
        let free = if matches!(self.slots[next], Slot::Empty) {
            self.growth_left += 1;
            Slot::Empty
        } else {
            Slot::Tombstone
        };
        let Slot::Full(b) = mem::replace(&mut self.slots[i], free) else {
            panic!("incr-core: key table slot {i} is not live");
        };
        self.keys -= 1;
        self.count_out(&b.pairs);
        b.pairs
    }

    /// Makes the table hold what `stage` staged, running no user code.
    /// `stage` must have been staged against this table as it is now.
    /// Returns the pairs it let go, for the caller to drop once user
    /// `Drop` may run.
    fn commit<K, V>(&mut self, stage: &mut KeyStage<K, V, P, R>) -> Vec<(P, R)> {
        let KeyStage { keys, fresh, .. } = stage;
        let added = keys
            .iter()
            .filter(|k| k.slot.is_none() && !k.now.is_empty())
            .count();
        if let Some(moved) = self.reserve(added) {
            for slot in keys.iter_mut().filter_map(|k| k.slot.as_mut()) {
                *slot = moved[*slot];
            }
        }
        let mut dropped = Vec::new();
        for k in keys.iter().filter(|k| !k.untouched()) {
            let mut held: Vec<Option<(P, R)>> = match k.slot {
                Some(s) => few_into_vec(self.replace(s, Few::None))
                    .into_iter()
                    .map(Some)
                    .collect(),
                None => Vec::new(),
            };
            let next: Vec<(P, R)> = k
                .now
                .iter()
                .map(|src| match *src {
                    Src::Held(i) => held[i].take(),
                    Src::New(j) => fresh[j].take(),
                })
                .map(|pair| pair.expect("incr-core: a staged pair is placed once"))
                .collect();
            dropped.extend(held.into_iter().flatten());
            match (k.slot, next.is_empty()) {
                (Some(s), true) => drop(self.remove(s)),
                (Some(s), false) => drop(self.replace(s, few_of(next))),
                (None, false) => self.insert(k.hash, few_of(next)),
                (None, true) => {}
            }
        }
        dropped
    }

    #[cfg(all(test, not(loom)))]
    fn all_pairs(&self) -> Vec<(P, R)>
    where
        P: Clone,
    {
        self.buckets()
            .flat_map(|(_, b)| b.pairs().iter().cloned())
            .collect()
    }
}

/// A keyed collection's key index: its [`KeyTable`] under a
/// [`LockKind::KeyIndex`] lock, an epoch that starts at 1 and advances on
/// every write that changes the table, and the table's pair count and
/// bytes as of the last such write, readable without the lock.
pub struct KeyIndex<P: 'static, R: 'static, C: Cells> {
    table: OrderedLock<KeyTable<P, R>, C>,
    hasher: RandomState,
    epoch: C::U64,
    settled_pairs: C::U64,
    settled_bytes: C::U64,
}

// SAFETY: the runtime's state registry needs `Send + Sync`. Under `Shared`
// every field is, given `P` and `R` are. Under `Local` the index is reached
// only through a `KeyedCollection<_, _, Local>`, which its collections'
// confinement marker keeps `!Send + !Sync`, and through the registry of a
// `Runtime<Local>`, which is neither, so it never leaves its thread.
unsafe impl<P: Send + Sync, R: Send + Sync, C: Cells> Send for KeyIndex<P, R, C> {}
// SAFETY: see `Send`.
unsafe impl<P: Send + Sync, R: Send + Sync, C: Cells> Sync for KeyIndex<P, R, C> {}

impl<P: 'static, R: Ord + Copy + 'static, C: Cells> KeyIndex<P, R, C> {
    fn new() -> Self {
        Self {
            table: OrderedLock::new(LockKind::KeyIndex, KeyTable::new()),
            hasher: RandomState::new(),
            epoch: C::new_u64(1),
            settled_pairs: C::new_u64(0),
            settled_bytes: C::new_u64(0),
        }
    }

    /// The epoch of the last write that changed the index.
    pub fn epoch(&self) -> u64 {
        C::u64_load_acquire(&self.epoch)
    }

    /// The pairs held and the bytes the table allocates, as of the last
    /// write that changed the index. Takes no lock, so a key or row method
    /// may call it during a write.
    pub fn settled(&self) -> (usize, usize) {
        (
            C::u64_load_acquire(&self.settled_pairs) as usize,
            C::u64_load_acquire(&self.settled_bytes) as usize,
        )
    }

    fn settle(&self, table: &KeyTable<P, R>) {
        C::u64_store_release(&self.settled_pairs, table.pairs as u64);
        C::u64_store_release(&self.settled_bytes, table.heap_bytes() as u64);
    }
}

impl<P: 'static, R: Ord + Copy + 'static, C: Cells> StateStats for KeyIndex<P, R, C> {
    fn state_stats(&self) -> StateMemory {
        let (pairs, bytes) = self.settled();
        StateMemory::new(
            StateKind::Other("keyed_index"),
            pairs,
            vec![Held::of::<P>(pairs)],
            bytes,
        )
    }
}

/// Where a staged key's pair comes from: the held pair at an index of its
/// bucket, or a pair the stage made, at an index of its `fresh` list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Src {
    Held(usize),
    New(usize),
}

struct StagedKey<K> {
    hash: u64,
    /// The held key object if the table holds the key, else the caller's.
    key: K,
    slot: Option<usize>,
    held: usize,
    /// The pairs the key is to hold, in bucket order.
    now: Vec<Src>,
}

impl<K> StagedKey<K> {
    fn untouched(&self) -> bool {
        self.now.len() == self.held && self.now.iter().enumerate().all(|(i, s)| *s == Src::Held(i))
    }
}

/// The ops of one keyed write, staged against a [`KeyTable`] as of an
/// epoch and composed per key: a later op on a key starts from what the
/// earlier ones staged. Staging runs all the write's user key and row code
/// and changes nothing; [`KeyTable::commit`] then applies it with none.
pub(crate) struct KeyStage<K, V, P, R> {
    keys: Vec<StagedKey<K>>,
    by_hash: HashMap<u64, Few<usize>, FxBuildHasher>,
    fresh: Vec<Option<(P, R)>>,
    epoch: u64,
    counts: KeyCounts,
    changed: Vec<K>,
    /// The index hash of each key in `changed`, in the same order.
    changed_hashes: Vec<u64>,
    /// Keys `apply` read from its ops that were staged already.
    spare: Vec<K>,
    journal: Option<Vec<PairOp<P, R>>>,
    _row: PhantomData<fn() -> V>,
}

impl<K, V, P, R> KeyStage<K, V, P, R>
where
    K: Value + Hash + Eq,
    V: Value + Hash + Eq,
    P: Pair<K, V>,
    R: ElementRank,
{
    fn new(journal: bool) -> Self {
        Self {
            keys: Vec::new(),
            by_hash: HashMap::default(),
            fresh: Vec::new(),
            epoch: 0,
            counts: KeyCounts::default(),
            changed: Vec::new(),
            changed_hashes: Vec::new(),
            spare: Vec::new(),
            journal: journal.then(Vec::new),
            _row: PhantomData,
        }
    }

    fn pair<'a>(&'a self, table: &'a KeyTable<P, R>, at: usize, src: Src) -> &'a (P, R) {
        match src {
            Src::Held(i) => {
                let slot = self.keys[at].slot.expect("a held pair has a slot");
                &table.bucket(slot).pairs()[i]
            }
            Src::New(j) => self.fresh[j].as_ref().expect("a staged pair is live"),
        }
    }

    /// The pair read for the key staged at `at`, as staged so far.
    fn read(&self, table: &KeyTable<P, R>, at: usize) -> Option<Src> {
        let now = &self.keys[at].now;
        first_smallest(now.iter().map(|s| self.pair(table, at, *s).1)).map(|i| now[i])
    }

    /// The pair read for the key staged at `at`, as held.
    fn held_read(&self, table: &KeyTable<P, R>, at: usize) -> Option<Src> {
        let slot = self.keys[at].slot?;
        Some(Src::Held(table.bucket(slot).read_at()))
    }

    fn staged(&self, hash: u64, key: &K) -> Option<usize> {
        self.by_hash.get(&hash).and_then(|at| {
            at.as_slice()
                .iter()
                .copied()
                .find(|&i| self.keys[i].key == *key)
        })
    }

    fn locate(&mut self, table: &KeyTable<P, R>, hash: u64, key: &K) -> usize {
        if let Some(at) = self.staged(hash, key) {
            return at;
        }
        let slot = table.find_key(hash, key);
        let (key, held) = match slot {
            Some(s) => {
                let b = table.bucket(s);
                (b.first().key(), b.pairs().len())
            }
            None => (key.clone(), 0),
        };
        self.push(StagedKey {
            hash,
            key,
            slot,
            held,
            now: (0..held).map(Src::Held).collect(),
        })
    }

    /// [`Self::locate`] for an owned key, which the stage keeps either
    /// way, so its `Drop` runs with the stage's.
    fn locate_owned(&mut self, table: &KeyTable<P, R>, hash: u64, key: K) -> usize {
        if let Some(at) = self.staged(hash, &key) {
            self.spare.push(key);
            return at;
        }
        let slot = table.find_key(hash, &key);
        let held = slot.map_or(0, |s| table.bucket(s).pairs().len());
        self.push(StagedKey {
            hash,
            key,
            slot,
            held,
            now: (0..held).map(Src::Held).collect(),
        })
    }

    fn push(&mut self, key: StagedKey<K>) -> usize {
        let at = self.keys.len();
        self.by_hash.entry(key.hash).or_default().push(at);
        self.keys.push(key);
        at
    }

    fn upsert(&mut self, table: &KeyTable<P, R>, hasher: &RandomState, key: &K, row: &V) {
        let at = self.locate(table, hasher.hash_one(key), key);
        if let Some(src) = self.read(table, at) {
            if self.pair(table, at, src).0.row_eq(row) {
                return;
            }
        }
        let pair = P::new(self.keys[at].key.clone(), row.clone());
        self.fresh.push(Some((pair, R::default())));
        self.keys[at].now = vec![Src::New(self.fresh.len() - 1)];
    }

    fn remove(&mut self, table: &KeyTable<P, R>, hasher: &RandomState, key: &K) {
        let at = self.locate(table, hasher.hash_one(key), key);
        self.keys[at].now.clear();
    }

    /// Stages the removal of every held key nothing staged so far names.
    fn remove_the_rest(&mut self, table: &KeyTable<P, R>) {
        let named: HashSet<usize, FxBuildHasher> =
            self.keys.iter().filter_map(|k| k.slot).collect();
        for (slot, b) in table.buckets() {
            if !named.contains(&slot) {
                self.push(StagedKey {
                    hash: b.hash,
                    key: b.first().key(),
                    slot: Some(slot),
                    held: b.pairs().len(),
                    now: Vec::new(),
                });
            }
        }
    }

    /// Ends the staging of `upsert`, `remove` and `reconcile`: a key whose
    /// staged row equals its held one keeps its held pairs, and each key
    /// is counted by its held and staged rows.
    fn settle_keys(&mut self, table: &KeyTable<P, R>) {
        for at in 0..self.keys.len() {
            let before = self.held_read(table, at);
            if let (Some(held), &[Src::New(j)]) = (before, self.keys[at].now.as_slice()) {
                let held_row = self.pair(table, at, held).0.row();
                if self.pair(table, at, Src::New(j)).0.row_eq(&held_row) {
                    let k = &mut self.keys[at];
                    k.now = (0..k.held).map(Src::Held).collect();
                    self.fresh[j] = None;
                }
            }
            let after = self.read(table, at);
            self.tally(at, before, after);
        }
    }

    fn tally(&mut self, at: usize, before: Option<Src>, after: Option<Src>) {
        let counts = &mut self.counts;
        match (before, after) {
            (None, Some(_)) => counts.inserted += 1,
            (Some(_), None) => counts.removed += 1,
            (Some(a), Some(b)) if a != b => counts.replaced += 1,
            _ => return,
        }
        self.changed.push(self.keys[at].key.clone());
        self.changed_hashes.push(self.keys[at].hash);
    }

    /// Stages `op` as the persisted binding replays it, counting it by
    /// the row read for its key before and after it.
    fn apply(&mut self, table: &KeyTable<P, R>, hasher: &RandomState, op: &PairOp<P, R>) {
        let (PairOp::Insert(pair, rank) | PairOp::Remove(pair, rank)) = op;
        let mut state = hasher.build_hasher();
        pair.key_hash(&mut state);
        let at = self.locate_owned(table, state.finish(), pair.key());
        let before = self.read(table, at);
        match op {
            PairOp::Insert(..) => {
                self.fresh.push(Some((pair.clone(), *rank)));
                self.keys[at].now.push(Src::New(self.fresh.len() - 1));
                if let Some(journal) = &mut self.journal {
                    journal.push(PairOp::Insert(pair.clone(), *rank));
                }
            }
            PairOp::Remove(..) => {
                let held = self.keys[at].now.iter().map(|s| {
                    let (p, r) = self.pair(table, at, *s);
                    (p, *r)
                });
                let found = find_element(held, (pair, *rank));
                if let Some(i) = found {
                    let src = self.keys[at].now.remove(i);
                    if self.journal.is_some() {
                        let (held, held_rank) = self.pair(table, at, src);
                        let undo = PairOp::Remove(held.clone(), *held_rank);
                        if let Some(journal) = &mut self.journal {
                            journal.push(undo);
                        }
                    }
                }
            }
        }
        let after = self.read(table, at);
        self.tally(at, before, after);
    }

    /// Journals what `upsert`, `remove` and `reconcile` staged, when
    /// journalling: per key, each held pair it lets go, then each new one.
    fn journal_staged(&mut self, table: &KeyTable<P, R>) {
        let Some(mut journal) = self.journal.take() else {
            return;
        };
        for (at, k) in self.keys.iter().enumerate() {
            if k.untouched() {
                continue;
            }
            for i in (0..k.held).filter(|i| !k.now.contains(&Src::Held(*i))) {
                let (pair, rank) = self.pair(table, at, Src::Held(i));
                journal.push(PairOp::Remove(pair.clone(), *rank));
            }
            for &src in k.now.iter().filter(|s| matches!(s, Src::New(_))) {
                let (pair, rank) = self.pair(table, at, src);
                journal.push(PairOp::Insert(pair.clone(), *rank));
            }
        }
        self.journal = Some(journal);
    }

    fn is_empty(&self) -> bool {
        self.keys.iter().all(StagedKey::untouched)
    }

    /// The keys that go from holding no pair to holding one or back.
    fn flips<'a>(&'a self, table: &'a KeyTable<P, R>) -> Vec<Flip<'a, P>> {
        let mut flips = Vec::new();
        for (at, k) in self.keys.iter().enumerate() {
            let inserted = !k.now.is_empty();
            if (k.held > 0) == inserted {
                continue;
            }
            let src = if inserted {
                self.read(table, at)
                    .expect("incr-core: a held key reads a pair")
            } else {
                Src::Held(0)
            };
            flips.push(Flip {
                pair: &self.pair(table, at, src).0,
                hash: k.hash,
                inserted,
            });
        }
        flips
    }

    /// Whether a key the write staged goes from holding no pair to holding
    /// one or back, judged over the whole write, so a key that loses its
    /// held pair and takes another in one write does not count.
    fn membership_changed(&self) -> bool {
        self.keys.iter().any(|k| (k.held > 0) != !k.now.is_empty())
    }

    /// The changes to the pair collection: each held pair a key lets go,
    /// then each new pair it takes.
    fn log_changes(&self, table: &KeyTable<P, R>) -> Vec<Change<P>> {
        let mut changes = Vec::new();
        for (at, k) in self.keys.iter().enumerate() {
            if k.untouched() {
                continue;
            }
            for i in (0..k.held).filter(|i| !k.now.contains(&Src::Held(*i))) {
                changes.push(Change::Delete(self.pair(table, at, Src::Held(i)).0.clone()));
            }
            for &src in &k.now {
                if let Src::New(_) = src {
                    changes.push(Change::Insert(self.pair(table, at, src).0.clone()));
                }
            }
        }
        changes
    }

    /// Moves the keys and pairs the stage holds into `keep` and drops its
    /// index containers, which hold no user value, so a caller keeping the
    /// stages of many writes keeps only their values.
    fn spend_into(self, keep: &mut KeyedSpent<K, V, P, R>) {
        keep.keys
            .extend(self.keys.into_iter().map(|staged| staged.key));
        keep.keys.extend(self.changed);
        keep.keys.extend(self.spare);
        keep.pairs.extend(self.fresh.into_iter().flatten());
        keep.ops.extend(self.journal.into_iter().flatten());
    }

    fn outcome(&mut self, epoch: u64) -> (KeyedChange<K>, Vec<PairOp<P, R>>) {
        let change = KeyedChange {
            counts: self.counts,
            changed: mem::take(&mut self.changed),
            epoch,
        };
        (change, self.journal.take().unwrap_or_default())
    }
}

/// A keyed collection's version inputs by key, both ways. Keys are found
/// by their hash under the index's hasher, so a writer finds the inputs of
/// the keys it changed from the hashes it staged, running no user code.
struct VersionMap<K> {
    by_hash: HashMap<u64, Vec<(K, Incr<u64>)>, FxBuildHasher>,
    by_handle: HashMap<ErasedIncr, K, FxBuildHasher>,
    /// The membership input, which no key maps to.
    membership: Option<Incr<u64>>,
}

impl<K: Eq> VersionMap<K> {
    /// Runs key `Eq`.
    fn find(&self, hash: u64, key: &K) -> Option<Incr<u64>> {
        self.by_hash
            .get(&hash)?
            .iter()
            .find(|(held, _)| held == key)
            .map(|(_, handle)| *handle)
    }

    /// Every input whose key has one of `hashes`, colliding keys included.
    fn by_hashes<'a>(
        &'a self,
        hashes: &'a [u64],
    ) -> impl Iterator<Item = (u64, &'a K, Incr<u64>)> + 'a {
        hashes.iter().flat_map(move |&hash| {
            self.by_hash
                .get(&hash)
                .into_iter()
                .flatten()
                .map(move |(key, handle)| (hash, key, *handle))
        })
    }

    /// Removes `handle`'s entries and hands back both copies of its key,
    /// so their `Drop` runs where the caller chooses.
    fn forget(&mut self, hash: u64, handle: Incr<u64>) -> Vec<K> {
        let mut keys = Vec::with_capacity(2);
        if let Some(bucket) = self.by_hash.get_mut(&hash) {
            if let Some(at) = bucket.iter().position(|(_, held)| *held == handle) {
                keys.push(bucket.swap_remove(at).0);
            }
            if bucket.is_empty() {
                self.by_hash.remove(&hash);
            }
        }
        keys.extend(self.by_handle.remove(&handle.erase()));
        keys
    }
}

/// The version inputs of the keys queries read through
/// [`KeyedCollection::get_key`]: each created at 0 by its first read in a
/// query and set to a fresh value by every write that changes its key's
/// row.
struct KeyVersions<K: 'static, C: Cells> {
    map: OrderedLock<VersionMap<K>, C>,
    /// The value the next write sets its keys' inputs to. Starts at 1,
    /// above every input's initial 0, so every set changes the input.
    changes: C::U64,
    /// How many inputs `map` holds, so a write to a collection no query
    /// reads by key takes no lock.
    live: C::U64,
    /// 1 once `map` holds the membership input, so a write that inserts or
    /// removes a key in a collection no query lists takes no lock.
    listed: C::U64,
    /// The input each query reading a partition depends on, by partitioning
    /// and partition hash: a partition whose hash collides shares it.
    partition_inputs: OrderedLock<PartitionInputs, C>,
    /// How many inputs `partition_inputs` holds.
    partition_live: C::U64,
}

impl<K: 'static, C: Cells> KeyVersions<K, C> {
    fn new() -> Self {
        Self {
            map: OrderedLock::new(
                LockKind::KeyVersions,
                VersionMap {
                    by_hash: HashMap::default(),
                    by_handle: HashMap::default(),
                    membership: None,
                },
            ),
            changes: C::new_u64(1),
            live: C::new_u64(0),
            listed: C::new_u64(0),
            partition_inputs: OrderedLock::new(LockKind::KeyVersions, HashMap::default()),
            partition_live: C::new_u64(0),
        }
    }

    fn any_listed(&self) -> bool {
        C::u64_load_acquire(&self.listed) != 0
    }

    fn any(&self) -> bool {
        C::u64_load_acquire(&self.live) != 0
    }
}

#[cold]
#[inline(never)]
fn refused() -> ! {
    panic!(
        "incr-core: KeyedCollection used from a key or row method: {}",
        value_code::CONTRACT_MESSAGE
    );
}

/// Panics if this thread is inside value code, naming the keyed collection.
#[inline(always)]
fn refuse_value_code() {
    if value_code::thread_word() & value_code::COUNTER_MASK != 0 {
        refused();
    }
}

/// Whether a read here records a dependency of a compute of `rt`: its
/// frame is on top of this thread's stack. A write checks
/// [`Runtime::frame_active`] instead, which counts every frame.
fn records_reads<C: Cells>(rt: &Runtime<C>) -> bool {
    rt.dep_stack.current_frame_active(rt.runtime_id)
}

/// Rows held by key: at most one row read per key, over a pair collection
/// of `P` and a rows collection mapped from it with [`Pair::row`]. Writes
/// go through the key index; queries read [`Self::rows`] or
/// [`Self::pairs`].
///
/// A key holds more than one pair only through `apply_pairs_with`; it then
/// reads as the first pair with the smallest rank `R`. Every method panics
/// when called from a key or row method run by a keyed write or read, and
/// every write panics inside a compute of any runtime.
///
/// A key's version input stays while the collection lives under `Shared`,
/// where deleting it could race a reader holding its handle, so their
/// number grows with the keys ever read in a query. Under `Local` a write
/// that leaves a key without a row deletes its input if no query depends
/// on it; [`Self::forget_key`] does the same for a key no write changed.
pub struct KeyedCollection<K, V, C, P = (K, V), R = ()>
where
    K: 'static,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
    R: 'static,
{
    pairs: IncrCollection<P, C>,
    rows: IncrCollection<V, C>,
    index: Arc<KeyIndex<P, R, C>>,
    versions: Arc<KeyVersions<K, C>>,
    _key: PhantomData<fn() -> K>,
}

impl<K, V, C, P, R> Clone for KeyedCollection<K, V, C, P, R>
where
    K: 'static,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
    R: 'static,
{
    fn clone(&self) -> Self {
        Self {
            pairs: self.pairs.clone(),
            rows: self.rows.clone(),
            index: Arc::clone(&self.index),
            versions: Arc::clone(&self.versions),
            _key: PhantomData,
        }
    }
}

/// How many times a registration builds its partitions under the table's
/// read lock, which a commit in between invalidates, before it builds them
/// under the write lock, which no commit can interleave with.
const OPTIMISTIC_REGISTRATIONS: usize = 3;

#[cfg(test)]
thread_local! {
    static AFTER_OPTIMISTIC_BUILD: std::cell::RefCell<Option<Box<dyn FnMut()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn set_after_optimistic_build(hook: Option<Box<dyn FnMut()>>) {
    AFTER_OPTIMISTIC_BUILD.with(|slot| *slot.borrow_mut() = hook);
}

#[cfg(test)]
fn after_optimistic_build() {
    let mut hook = AFTER_OPTIMISTIC_BUILD.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook.as_mut() {
        hook();
    }
    AFTER_OPTIMISTIC_BUILD.with(|slot| *slot.borrow_mut() = hook);
}

/// The input per (partitioning, partition hash).
type PartitionInputs = HashMap<(usize, u64), Incr<u64>, FxBuildHasher>;

/// A partitioning of a keyed collection's keys, from
/// [`KeyedCollection::key_partitions`]; use it with that collection only.
pub struct KeyPartitions<Q> {
    sink: usize,
    _q: PhantomData<fn() -> Q>,
}

impl<Q> Clone for KeyPartitions<Q> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Q> Copy for KeyPartitions<Q> {}

/// What keyed writes let go, from [`KeyedCollection::apply_pairs_keeping`]:
/// their ops, staged pairs and keys they did not use, pairs they displaced,
/// the key index's dropped pairs and the keys of version inputs deleted.
/// Dropping it runs their `Drop`.
pub struct KeyedSpent<K, V, P, R = ()> {
    ops: Vec<PairOp<P, R>>,
    keys: Vec<K>,
    batches: Vec<StagedBatch<P>>,
    unused: Vec<P>,
    displaced: Vec<Displaced<P>>,
    pairs: Vec<(P, R)>,
    _row: PhantomData<fn() -> V>,
}

impl<K, V, P, R> Default for KeyedSpent<K, V, P, R> {
    fn default() -> Self {
        Self {
            ops: Vec::new(),
            keys: Vec::new(),
            batches: Vec::new(),
            unused: Vec::new(),
            displaced: Vec::new(),
            pairs: Vec::new(),
            _row: PhantomData,
        }
    }
}

type StageOps<'s, K, V, P, R> =
    dyn Fn(&mut KeyStage<K, V, P, R>, &KeyTable<P, R>, &RandomState) + 's;

impl<K, V, C, P, R> KeyedCollection<K, V, C, P, R>
where
    K: Value + Hash + Eq,
    V: Value + Hash + Eq,
    C: Cells,
    P: Pair<K, V>,
    R: ElementRank + Send + Sync + 'static,
{
    /// An empty keyed collection in `rt`, its index registered for
    /// [`Runtime::memory_stats`] against the rows collection's version node.
    pub fn new(rt: &Runtime<C>) -> Self {
        let pairs = rt.create_collection::<P>();
        let rows = pairs.map(rt, |pair: &P| pair.row());
        let index = Arc::new(KeyIndex::new());
        let weak: Weak<KeyIndex<P, R, C>> = Arc::downgrade(&index);
        rt.register_state(Some(rows.version_node().erase()), weak);
        Self {
            pairs,
            rows,
            index,
            versions: Arc::new(KeyVersions::new()),
            _key: PhantomData,
        }
    }

    /// A view of the pair collection, which panics on a direct write.
    pub fn pairs(&self) -> IncrCollection<P, C> {
        self.pairs.as_view()
    }

    /// The rows collection: every held pair's row, an operator output.
    pub fn rows(&self) -> &IncrCollection<V, C> {
        &self.rows
    }

    pub fn index(&self) -> &KeyIndex<P, R, C> {
        &self.index
    }

    /// A weak handle to the index, for a caller that reads its
    /// [`StateStats`] only while the collection lives.
    pub fn index_weak(&self) -> Weak<KeyIndex<P, R, C>> {
        Arc::downgrade(&self.index)
    }

    /// Holds `row` for `key`. Changes nothing when the row read for `key`
    /// is `==` to `row`; otherwise every pair held for `key` gives way to
    /// one new pair of rank `R::default()`, which keeps the held key
    /// object when there is one.
    pub fn upsert(&self, rt: &Runtime<C>, key: K, row: V) -> KeyedChange<K> {
        self.upsert_journaled(rt, key, row, None)
    }

    /// [`Self::upsert`], journalling into `journal` as
    /// [`Self::apply_pairs_with`] does: each pair the write let go as a
    /// removal, then each it took as an insert.
    pub fn upsert_journaled(
        &self,
        rt: &Runtime<C>,
        key: K,
        row: V,
        journal: Option<&mut Vec<PairOp<P, R>>>,
    ) -> KeyedChange<K> {
        self.write_journaled(rt, journal, &|stage, table, hasher| {
            stage.upsert(table, hasher, &key, &row);
            stage.settle_keys(table);
        })
    }

    /// Drops every pair held for `key`; a no-op for a key with none.
    pub fn remove(&self, rt: &Runtime<C>, key: &K) -> KeyedChange<K> {
        self.remove_journaled(rt, key, None)
    }

    /// [`Self::remove`], journalling as [`Self::upsert_journaled`] does.
    pub fn remove_journaled(
        &self,
        rt: &Runtime<C>,
        key: &K,
        journal: Option<&mut Vec<PairOp<P, R>>>,
    ) -> KeyedChange<K> {
        self.write_journaled(rt, journal, &|stage, table, hasher| {
            stage.remove(table, hasher, key);
            stage.settle_keys(table);
        })
    }

    /// Makes the rows read equal `rows` in one atomic write: a later pair
    /// for a key wins, a key whose row is unchanged keeps its pairs, and a
    /// held key `rows` does not name is removed.
    pub fn reconcile(
        &self,
        rt: &Runtime<C>,
        rows: impl IntoIterator<Item = (K, V)>,
    ) -> KeyedChange<K> {
        self.reconcile_journaled(rt, rows, None)
    }

    /// [`Self::reconcile`], journalling as [`Self::upsert_journaled`] does.
    pub fn reconcile_journaled(
        &self,
        rt: &Runtime<C>,
        rows: impl IntoIterator<Item = (K, V)>,
        journal: Option<&mut Vec<PairOp<P, R>>>,
    ) -> KeyedChange<K> {
        let rows: Vec<(K, V)> = rows.into_iter().collect();
        self.write_journaled(rt, journal, &|stage, table, hasher| {
            for (key, row) in &rows {
                stage.upsert(table, hasher, key, row);
            }
            stage.remove_the_rest(table);
            stage.settle_keys(table);
        })
    }

    fn write_journaled(
        &self,
        rt: &Runtime<C>,
        journal: Option<&mut Vec<PairOp<P, R>>>,
        stage_ops: &StageOps<'_, K, V, P, R>,
    ) -> KeyedChange<K> {
        let (change, steps) =
            self.write(rt, &mut (), journal.is_some(), &|stage, table, hasher| {
                stage_ops(stage, table, hasher);
                stage.journal_staged(table);
            });
        if let Some(journal) = journal {
            journal.extend(steps);
        }
        change
    }

    /// Applies `ops` in order as one atomic write, reporting the pair
    /// collection's representatives to `sink` as its mutations do. Into
    /// `journal`, when given, goes each op that changed the index, a
    /// removal naming the pair and rank it removed, so applying the
    /// journal's inverses newest first restores the index and the pairs.
    pub fn apply_pairs_with<S: RepSink<P>>(
        &self,
        rt: &Runtime<C>,
        ops: impl IntoIterator<Item = PairOp<P, R>>,
        sink: &mut S,
        journal: Option<&mut Vec<PairOp<P, R>>>,
    ) -> KeyedChange<K> {
        let ops: Vec<PairOp<P, R>> = ops.into_iter().collect();
        let (change, steps) = self.write(rt, sink, journal.is_some(), &|stage, table, hasher| {
            for op in &ops {
                stage.apply(table, hasher, op);
            }
        });
        if let Some(journal) = journal {
            journal.extend(steps);
        }
        change
    }

    /// [`Self::apply_pairs_with`] that drops nothing a write lets go: the
    /// staged pairs and keys it did not use, the pairs it displaced and
    /// the index's dropped pairs move into `keep`, for a caller that runs
    /// their `Drop` once it may, such as after releasing a lock user code
    /// must not run under. Version sets, routing and the deletion of
    /// version inputs are as in [`Self::apply_pairs_with`].
    pub fn apply_pairs_keeping<S: RepSink<P>>(
        &self,
        rt: &Runtime<C>,
        ops: impl IntoIterator<Item = PairOp<P, R>>,
        sink: &mut S,
        journal: Option<&mut Vec<PairOp<P, R>>>,
        keep: &mut KeyedSpent<K, V, P, R>,
    ) -> KeyedChange<K> {
        let ops: Vec<PairOp<P, R>> = ops.into_iter().collect();
        let (change, steps) = self.write_keeping(
            rt,
            sink,
            journal.is_some(),
            &|stage, table, hasher| {
                for op in &ops {
                    stage.apply(table, hasher, op);
                }
            },
            Some(keep),
        );
        if let Some(journal) = journal {
            journal.extend(steps);
        }
        keep.ops.extend(ops);
        change
    }

    fn write<S: RepSink<P>>(
        &self,
        rt: &Runtime<C>,
        sink: &mut S,
        journal: bool,
        stage_ops: &StageOps<'_, K, V, P, R>,
    ) -> (KeyedChange<K>, Vec<PairOp<P, R>>) {
        self.write_keeping(rt, sink, journal, stage_ops, None)
    }

    /// Without `keep`, drops what the write lets go after the version set
    /// and before routing, and a failed attempt's stage at once. The changed
    /// keys' version inputs are set after the index and log guards are
    /// released and before the pair collection's version, with no user
    /// code run since the first change.
    fn write_keeping<S: RepSink<P>>(
        &self,
        rt: &Runtime<C>,
        sink: &mut S,
        journal: bool,
        stage_ops: &StageOps<'_, K, V, P, R>,
        mut keep: Option<&mut KeyedSpent<K, V, P, R>>,
    ) -> (KeyedChange<K>, Vec<PairOp<P, R>>) {
        refuse_value_code();
        assert!(
            !rt.frame_active(),
            "incr-core: KeyedCollection written from inside a compute; a run the engine \
             discards would leave rows ahead of their readers' versions",
        );
        let index = &*self.index;
        loop {
            let mut stage = KeyStage::new(journal);
            let (changes, plans) = {
                let table = index.table.read();
                stage.epoch = C::u64_load_relaxed(&index.epoch);
                value_code::run(|| {
                    stage_ops(&mut stage, &table, &index.hasher);
                    let changes = stage.log_changes(&table);
                    let plans = if table.parts.is_empty() {
                        Vec::new()
                    } else {
                        table.plan_parts(&stage.flips(&table))
                    };
                    (changes, plans)
                })
            };
            if stage.is_empty() {
                let outcome = stage.outcome(stage.epoch);
                if let Some(keep) = keep {
                    stage.spend_into(keep);
                }
                return outcome;
            }
            let batch = self.pairs.stage_batch(changes);
            let mut log = self.pairs.log.write();
            let mut table = index.table.write();
            if C::u64_load_relaxed(&index.epoch) != stage.epoch {
                drop(table);
                drop(log);
                if let Some(keep) = keep.as_deref_mut() {
                    stage.spend_into(keep);
                    keep.batches.push(batch);
                }
                continue;
            }
            let committed = log.apply_batch(batch, false, sink);
            let dropped = table.commit(&mut stage);
            let mut gone = Vec::new();
            let touched = table.apply_parts(plans, &mut gone);
            let epoch = stage.epoch + 1;
            C::u64_store_release(&index.epoch, epoch);
            index.settle(&table);
            drop(table);
            drop(log);
            let hashes = mem::take(&mut stage.changed_hashes);
            let membership = self.versions.any_listed() && stage.membership_changed();
            let queued = self.bump(rt, &hashes, membership, &touched);
            let outcome = stage.outcome(epoch);
            match keep.as_deref_mut() {
                Some(keep) => {
                    self.pairs.publish_batch_keeping(
                        rt,
                        committed,
                        &mut keep.unused,
                        &mut keep.displaced,
                    );
                    keep.pairs.extend(dropped);
                    if !gone.is_empty() {
                        keep.unused.extend(gone);
                    }
                    stage.spend_into(keep);
                }
                None => {
                    self.pairs
                        .publish_batch(rt, committed, move || drop((dropped, gone, stage)));
                }
            }
            if queued {
                rt.route_queued();
            }
            self.reclaim(rt, &hashes, keep);
            return outcome;
        }
    }

    /// Sets the version input of every key with one of `hashes`, and the
    /// membership input when `membership`, to a fresh value without
    /// routing: whether a set queued a router, which the caller routes once
    /// the write's other sets are done. Runs no user code.
    fn bump(
        &self,
        rt: &Runtime<C>,
        hashes: &[u64],
        membership: bool,
        touched: &[(usize, u64)],
    ) -> bool {
        let keyed = !hashes.is_empty() && self.versions.any();
        let listed = membership && self.versions.any_listed();
        let parted = !touched.is_empty() && C::u64_load_acquire(&self.versions.partition_live) != 0;
        if !keyed && !listed && !parted {
            return false;
        }
        let value = C::u64_fetch_add(&self.versions.changes, 1);
        let handles: Vec<Incr<u64>> = {
            let map = self.versions.map.read();
            let mut handles: Vec<Incr<u64>> = if keyed {
                map.by_hashes(hashes).map(|(_, _, handle)| handle).collect()
            } else {
                Vec::new()
            };
            if listed {
                handles.extend(map.membership);
            }
            drop(map);
            if parted {
                let inputs = self.versions.partition_inputs.read();
                handles.extend(touched.iter().filter_map(|t| inputs.get(t).copied()));
            }
            handles
        };
        let mut queued = false;
        for handle in handles {
            queued |= rt.set_unrouted(handle, value);
        }
        queued
    }

    /// Under `Local`, deletes the version input of each key with one of
    /// `hashes` that holds no row and has no dependent; its key goes to
    /// `keep` when given.
    fn reclaim(&self, rt: &Runtime<C>, hashes: &[u64], keep: Option<&mut KeyedSpent<K, V, P, R>>) {
        if !C::SINGLE_THREADED || hashes.is_empty() || !self.versions.any() {
            return;
        }
        let candidates: Vec<(u64, K, Incr<u64>)> = {
            let map = self.versions.map.read();
            value_code::run(|| {
                map.by_hashes(hashes)
                    .map(|(hash, key, handle)| (hash, key.clone(), handle))
                    .collect()
            })
        };
        let (_, keys) = self.release_inputs(rt, candidates);
        match keep {
            Some(keep) => keep.keys.extend(keys),
            None => drop(keys),
        }
    }

    /// Deletes the version input of each candidate that holds no row and
    /// has no dependent, and unmaps it. Returns how many inputs it freed
    /// and every key it handled, whose drops the caller places.
    fn release_inputs(
        &self,
        rt: &Runtime<C>,
        candidates: Vec<(u64, K, Incr<u64>)>,
    ) -> (usize, Vec<K>) {
        let mut freed = Vec::new();
        for (hash, key, handle) in &candidates {
            if self.held(key) {
                continue;
            }
            match rt.try_delete_node(*handle) {
                Ok(()) | Err(DeleteError::NotAlive) => freed.push((*hash, *handle)),
                Err(DeleteError::HasDependents) => {}
            }
        }
        let mut keys: Vec<K> = Vec::new();
        let mut unmapped = 0;
        if !freed.is_empty() {
            let mut map = self.versions.map.write();
            for (hash, handle) in freed {
                let forgotten = map.forget(hash, handle);
                if !forgotten.is_empty() {
                    C::u64_fetch_add(&self.versions.live, u64::MAX);
                    unmapped += 1;
                    keys.extend(forgotten);
                }
            }
        }
        keys.extend(candidates.into_iter().map(|(_, key, _)| key));
        (unmapped, keys)
    }

    /// Under `Local`, deletes `key`'s version input when no query depends
    /// on it and the key holds no row, for a key a write never changed
    /// after its readers went away. Returns whether it freed the input;
    /// under `Shared` it frees nothing and returns false, since a reader
    /// may still hold the handle. Panics inside a compute of any runtime,
    /// like every write.
    pub fn forget_key(&self, rt: &Runtime<C>, key: &K) -> bool {
        refuse_value_code();
        assert!(
            !rt.frame_active(),
            "incr-core: KeyedCollection::forget_key called from inside a compute; a run the \
             engine discards would delete an input its own read created",
        );
        if !C::SINGLE_THREADED || !self.versions.any() {
            return false;
        }
        let hash = value_code::run(|| self.index.hasher.hash_one(key));
        let candidates: Vec<(u64, K, Incr<u64>)> = {
            let map = self.versions.map.read();
            value_code::run(|| {
                map.find(hash, key)
                    .map(|handle| (hash, key.clone(), handle))
                    .into_iter()
                    .collect()
            })
        };
        let (freed, keys) = self.release_inputs(rt, candidates);
        value_code::run(|| drop(keys));
        freed > 0
    }

    /// How many version inputs this collection holds.
    pub fn version_input_count(&self) -> usize {
        C::u64_load_acquire(&self.versions.live) as usize
    }

    /// Whether `key` holds a row, read as value code under the index's
    /// read guard. Marks nothing on a running compute.
    fn held(&self, key: &K) -> bool {
        let table = self.index.table.read();
        value_code::run(|| {
            table
                .find_key(self.index.hasher.hash_one(key), key)
                .is_some()
        })
    }

    /// The row read for `key`. Inside a compute of `rt` the read depends
    /// on `key`'s version input, created at 0 on the first such read, so
    /// the query reruns only when a write changes this key's row; the row
    /// itself is read from the index. Outside one it reads as
    /// [`Self::row`] and creates nothing. In a run the engine discards, a
    /// key no read created an input for reads as [`Self::row`] too, as a
    /// `NodeTable` miss there creates nothing.
    pub fn get_key(&self, rt: &Runtime<C>, key: &K) -> Option<V> {
        refuse_value_code();
        if !records_reads(rt) {
            return self.row(key);
        }
        let handle = match self.version_input(key) {
            Some(handle) => handle,
            None if rt.frame_doomed() => return self.row(key),
            None => self.version_input_or_create(rt, key),
        };
        rt.get(handle);
        let table = self.index.table.read();
        value_code::run(|| {
            table
                .find_key(self.index.hasher.hash_one(key), key)
                .map(|s| table.bucket(s).read().0.row())
        })
    }

    /// The keys held. Inside a compute of `rt` the read depends on the
    /// membership input, created at 0 on the first such read, so the query
    /// reruns when a write inserts or removes a key and not when it only
    /// changes a row. Outside one it reads as [`Self::keys`] and creates
    /// nothing; in a run the engine discards, with no input yet, likewise.
    pub fn get_keys(&self, rt: &Runtime<C>) -> Vec<K> {
        refuse_value_code();
        if !records_reads(rt) {
            return self.keys();
        }
        let handle = match self.membership_input() {
            Some(handle) => handle,
            None if rt.frame_doomed() => return self.keys(),
            None => self.membership_input_or_create(rt),
        };
        rt.get(handle);
        let table = self.index.table.read();
        value_code::run(|| table.buckets().map(|(_, b)| b.first().key()).collect())
    }

    /// The membership input, if a read created one.
    pub fn membership_input(&self) -> Option<Incr<u64>> {
        if !self.versions.any_listed() {
            return None;
        }
        self.versions.map.read().membership
    }

    /// Every partition input a read created, in no particular order.
    pub fn partition_inputs(&self) -> Vec<ErasedIncr> {
        self.versions
            .partition_inputs
            .read()
            .values()
            .map(|handle| handle.erase())
            .collect()
    }

    /// Whether a read created the membership input.
    pub fn has_membership_input(&self) -> bool {
        self.versions.any_listed()
    }

    /// The membership input, created at 0 if no read created one yet.
    /// Takes the version lock, like [`Self::version_input_or_create`].
    pub fn membership_input_or_create(&self, rt: &Runtime<C>) -> Incr<u64> {
        if let Some(handle) = self.membership_input() {
            return handle;
        }
        let mut map = self.versions.map.write();
        if let Some(handle) = map.membership {
            return handle;
        }
        let handle = rt.create_input_unchecked(0_u64);
        map.membership = Some(handle);
        C::u64_store_release(&self.versions.listed, 1);
        handle
    }

    /// Registers a partitioning of the keys by `f`, run as value code on
    /// each key a write inserts or removes (in staging, so a function that
    /// panics refuses the write with nothing committed) and on the keys
    /// held now. A row replacement runs nothing.
    ///
    /// `f` must be deterministic and total over every key it was ever given:
    /// the same partition for a key every time, and no panic on a key it
    /// accepted before. A panic on a new key refuses that write; a panic on
    /// a key it accepted before (a replay, a repair, a reconcile removal)
    /// leaves a persisted collection's store poisoned.
    ///
    /// Registration is bounded: it builds under the table's read lock for a
    /// few attempts, each invalidated by a commit in between, then under the
    /// write lock, which blocks commits and readers of the index for the
    /// build.
    pub fn key_partitions<Q>(&self, f: impl Fn(&K) -> Q + Send + Sync + 'static) -> KeyPartitions<Q>
    where
        Q: Value + Hash + Eq,
    {
        refuse_value_code();
        let f: Arc<dyn Fn(&K) -> Q + Send + Sync> = Arc::new(f);
        for _ in 0..OPTIMISTIC_REGISTRATIONS {
            let (epoch, state) = {
                let table = self.index.table.read();
                let epoch = C::u64_load_relaxed(&self.index.epoch);
                let state = value_code::run(|| {
                    PartState::<K, V, Q, P>::build(
                        Arc::clone(&f),
                        table.buckets().map(|(_, b)| (b.hash, &b.read().0)),
                    )
                });
                (epoch, state)
            };
            #[cfg(test)]
            after_optimistic_build();
            let mut table = self.index.table.write();
            if C::u64_load_relaxed(&self.index.epoch) == epoch {
                table.parts.push(Box::new(state));
                C::u64_store_release(&self.index.epoch, epoch + 1);
                return KeyPartitions {
                    sink: table.parts.len() - 1,
                    _q: PhantomData,
                };
            }
        }
        let mut table = self.index.table.write();
        let state = value_code::run(|| {
            PartState::<K, V, Q, P>::build(
                Arc::clone(&f),
                table.buckets().map(|(_, b)| (b.hash, &b.read().0)),
            )
        });
        let epoch = C::u64_load_relaxed(&self.index.epoch);
        table.parts.push(Box::new(state));
        C::u64_store_release(&self.index.epoch, epoch + 1);
        KeyPartitions {
            sink: table.parts.len() - 1,
            _q: PhantomData,
        }
    }

    /// The keys of partition `q` of `parts`. Inside a compute of `rt` the
    /// read depends on that partition's input, created at 0 on the first
    /// such read, so the query reruns when a write inserts or removes a key
    /// of the partition and not for any other change. Outside one it reads
    /// the partition and creates nothing.
    pub fn get_partition<Q>(&self, rt: &Runtime<C>, parts: &KeyPartitions<Q>, q: &Q) -> Vec<K>
    where
        Q: Value + Hash + Eq,
    {
        refuse_value_code();
        if !records_reads(rt) {
            return self.read_partition(parts, q);
        }
        let qhash = value_code::run(|| FxBuildHasher.hash_one(q));
        let slot = (parts.sink, qhash);
        let found = self.versions.partition_inputs.read().get(&slot).copied();
        let handle = match found {
            Some(handle) => handle,
            None if rt.frame_doomed() => return self.read_partition(parts, q),
            None => {
                let mut inputs = self.versions.partition_inputs.write();
                *inputs.entry(slot).or_insert_with(|| {
                    C::u64_fetch_add(&self.versions.partition_live, 1);
                    rt.create_input_unchecked(0_u64)
                })
            }
        };
        rt.get(handle);
        self.read_partition(parts, q)
    }

    fn read_partition<Q>(&self, parts: &KeyPartitions<Q>, q: &Q) -> Vec<K>
    where
        Q: Value + Hash + Eq,
    {
        let table = self.index.table.read();
        value_code::run(|| {
            table.parts[parts.sink]
                .as_any()
                .downcast_ref::<PartState<K, V, Q, P>>()
                .expect("incr-core: a KeyPartitions belongs to its collection")
                .keys_of(q)
        })
    }

    /// Whether a partitioning is registered.
    pub fn has_partitions(&self) -> bool {
        !self.index.table.read().parts.is_empty()
    }

    /// Runs every registered partition function on `pair`'s key and
    /// discards the results, so a caller about to log a write can refuse
    /// it before it is durable when a function panics on one of its keys.
    /// Runs the functions as value code.
    pub fn check_partitions(&self, pair: &P) {
        refuse_value_code();
        let table = self.index.table.read();
        value_code::run(|| table.parts.iter().for_each(|sink| sink.check(pair)));
    }

    /// `key`'s version input, if a read created one.
    pub fn version_input(&self, key: &K) -> Option<Incr<u64>> {
        refuse_value_code();
        let hash = value_code::run(|| self.index.hasher.hash_one(key));
        let map = self.versions.map.read();
        value_code::run(|| map.find(hash, key))
    }

    /// `key`'s version input, created at 0 if no read created one yet.
    /// Takes the version lock, ranked above `DepIndex`, so a caller holding
    /// a `DepIndex` hold must not call it.
    pub fn version_input_or_create(&self, rt: &Runtime<C>, key: &K) -> Incr<u64> {
        if let Some(handle) = self.version_input(key) {
            return handle;
        }
        let hash = value_code::run(|| self.index.hasher.hash_one(key));
        let mut map = self.versions.map.write();
        if let Some(handle) = value_code::run(|| map.find(hash, key)) {
            return handle;
        }
        let (by_hash, by_handle) = value_code::run(|| (key.clone(), key.clone()));
        let handle = rt.create_input_unchecked(0_u64);
        map.by_hash.entry(hash).or_default().push((by_hash, handle));
        map.by_handle.insert(handle.erase(), by_handle);
        C::u64_fetch_add(&self.versions.live, 1);
        handle
    }

    /// A copy of the key whose version input is `handle`.
    pub fn key_of_version(&self, handle: ErasedIncr) -> Option<K> {
        self.keys_of_versions(&[handle], |f| f()).pop().flatten()
    }

    /// [`Self::key_of_version`] for each of `handles`, under one read of
    /// the version map, with every key cloned inside the one call of
    /// `stretch`, which must run its argument exactly once on this thread
    /// (a binding takes its value-code lock around it). Panics if it does
    /// not. `stretch` is not called when no handle is a version input.
    pub fn keys_of_versions(
        &self,
        handles: &[ErasedIncr],
        stretch: impl FnOnce(&mut dyn FnMut()),
    ) -> Vec<Option<K>> {
        refuse_value_code();
        if !self.versions.any() {
            return handles.iter().map(|_| None).collect();
        }
        let map = self.versions.map.read();
        if !handles.iter().any(|h| map.by_handle.contains_key(h)) {
            return handles.iter().map(|_| None).collect();
        }
        let mut keys = Vec::with_capacity(handles.len());
        stretch(&mut || {
            keys.clear();
            value_code::run(|| {
                keys.extend(
                    handles
                        .iter()
                        .map(|handle| map.by_handle.get(handle).cloned()),
                );
            });
        });
        assert_eq!(
            keys.len(),
            handles.len(),
            "incr-core: keys_of_versions' stretch did not run its argument"
        );
        keys
    }

    /// Every version input, in no particular order.
    pub fn version_inputs(&self) -> Vec<ErasedIncr> {
        self.versions.map.read().by_handle.keys().copied().collect()
    }

    /// The rank of the pair read for `key`. Records no dependency and
    /// marks nothing on a running compute.
    pub fn read_rank(&self, key: &K) -> Option<R> {
        self.read_ranks(&[key], |f| f()).pop().flatten()
    }

    /// [`Self::read_rank`] for each of `keys`, under one read of the
    /// table, with every key hashed and compared inside the one call of
    /// `stretch`, which must run its argument exactly once on this thread.
    /// Panics if it does not.
    pub fn read_ranks(
        &self,
        keys: &[&K],
        stretch: impl FnOnce(&mut dyn FnMut()),
    ) -> Vec<Option<R>> {
        refuse_value_code();
        let table = self.index.table.read();
        let mut ranks = Vec::with_capacity(keys.len());
        stretch(&mut || {
            ranks.clear();
            value_code::run(|| {
                ranks.extend(keys.iter().map(|&key| {
                    table
                        .find_key(self.index.hasher.hash_one(key), key)
                        .map(|s| table.bucket(s).read().1)
                }));
            });
        });
        assert_eq!(
            ranks.len(),
            keys.len(),
            "incr-core: read_ranks' stretch did not run its argument"
        );
        ranks
    }

    /// Runs `f` on the table as value code, under its read guard. Records
    /// no dependency.
    fn read<T>(&self, f: impl FnOnce(&KeyTable<P, R>, &RandomState) -> T) -> T {
        refuse_value_code();
        crate::dep_stack::unrecorded_lookup::<C>();
        let table = self.index.table.read();
        value_code::run(|| f(&table, &self.index.hasher))
    }

    /// A copy of the row read for `key`.
    pub fn row(&self, key: &K) -> Option<V> {
        self.read(|table, hasher| {
            table
                .find_key(hasher.hash_one(key), key)
                .map(|s| table.bucket(s).read().0.row())
        })
    }

    pub fn contains(&self, key: &K) -> bool {
        self.read(|table, hasher| table.find_key(hasher.hash_one(key), key).is_some())
    }

    /// The number of keys holding a row.
    pub fn len(&self) -> usize {
        self.read(|table, _| table.keys)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A snapshot of the held keys, in no particular order.
    pub fn keys(&self) -> Vec<K> {
        self.read(|table, _| table.buckets().map(|(_, b)| b.first().key()).collect())
    }

    /// A snapshot of the pair read for each key, in no particular order.
    pub fn items(&self) -> Vec<P> {
        self.read(|table, _| table.buckets().map(|(_, b)| b.read().0.clone()).collect())
    }

    pub fn epoch(&self) -> u64 {
        self.index.epoch()
    }

    /// A copy of every pair held for `key`, with its rank, in bucket order.
    pub fn held_pairs(&self, key: &K) -> Vec<(P, R)> {
        self.read(
            |table, hasher| match table.find_key(hasher.hash_one(key), key) {
                Some(s) => table.bucket(s).pairs().to_vec(),
                None => Vec::new(),
            },
        )
    }

    /// A copy of every pair held, with its rank, in no particular order.
    pub fn ranked_pairs(&self) -> Vec<(P, R)> {
        self.read(|table, _| {
            table
                .buckets()
                .flat_map(|(_, b)| b.pairs().iter().cloned())
                .collect()
        })
    }
}

#[cfg(all(test, not(loom)))]
#[path = "keyed_tests.rs"]
mod tests;
