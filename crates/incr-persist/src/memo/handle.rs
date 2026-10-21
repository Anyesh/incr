use std::fmt;
use std::marker::PhantomData;

use incr_core::{Incr, Runtime, TableKey, Value};
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::memo_core::{CoreRef, MemoCells};

/// A memo table's key: a [`TableKey`] that serializes. Its postcard bytes
/// identify the entry, and the decoded key must `==` the live one.
pub trait MemoKey: TableKey + Serialize + DeserializeOwned {}

impl<K: TableKey + Serialize + DeserializeOwned> MemoKey for K {}

/// A memo table's value: a [`Value`] that serializes. Only values that
/// decode back equal to themselves are checkpointed.
pub trait MemoValue: Value + Serialize + DeserializeOwned {}

impl<T: Value + Serialize + DeserializeOwned> MemoValue for T {}

/// A table declared with [`MemoBuilder::table`](super::MemoBuilder::table):
/// a copyable name for it that memo closures capture, so tables can read
/// each other, mutual recursion included.
pub struct MemoTable<K, T> {
    pub(crate) memo: u64,
    pub(crate) index: u32,
    _types: PhantomData<fn(K) -> T>,
}

impl<K, T> MemoTable<K, T> {
    pub(crate) fn new(memo: u64, index: u32) -> Self {
        MemoTable {
            memo,
            index,
            _types: PhantomData,
        }
    }
}

impl<K, T> Clone for MemoTable<K, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K, T> Copy for MemoTable<K, T> {}

impl<K, T> fmt::Debug for MemoTable<K, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoTable")
            .field("memo", &self.memo)
            .field("index", &self.index)
            .finish()
    }
}

/// What a memo closure computes with: its runtime and the other tables.
pub struct MemoCtx<'a, C: MemoCells> {
    pub(crate) rt: &'a Runtime<C>,
    pub(crate) core: &'a CoreRef<C>,
}

impl<'a, C: MemoCells> MemoCtx<'a, C> {
    pub fn rt(&self) -> &'a Runtime<C> {
        self.rt
    }

    /// `table`'s value for `key`, restored or computed, as a dependency of
    /// the running compute.
    pub fn get<K: MemoKey, T: MemoValue>(&self, table: MemoTable<K, T>, key: K) -> T {
        self.rt.get(self.get_or_insert(table, key))
    }

    /// External `name`'s fp as a dependency, `0` until
    /// [`Memo::set_external`](super::Memo::set_external) first sets it.
    pub fn external(&self, name: &str) -> u128 {
        let handle = self.core.external(self.rt, name);
        self.rt.get(handle)
    }

    /// `table`'s node for `key`, restored on a miss when the memo holds an
    /// entry for it. Reading the node is what records the dependency.
    pub fn get_or_insert<K: MemoKey, T: MemoValue>(
        &self,
        table: MemoTable<K, T>,
        key: K,
    ) -> Incr<T> {
        self.core.get_or_insert(self.rt, table, key)
    }
}
