use std::any::Any;
use std::sync::Arc;

use incr_core::persist_hooks::{
    get_or_insert_query_seeded, inspect, probe_unchecked, seed_query, Deferred, Reuse, SeedBase,
    SeedDep,
};
use incr_core::{with_value_code, ErasedIncr, Incr, NodeTable, QueryOpts, Runtime};
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::handle::{MemoCtx, MemoKey, MemoValue};
use super::mapping::{map_live, LiveDep};
use super::memo_core::{CoreRef, MemoCells};
use crate::store::dep_index::DepMap;

pub(crate) type MemoFn<K, T, C> = dyn Fn(&MemoCtx<'_, C>, &K) -> T + Send + Sync;

/// How a restore's attempt to seed one entry ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Seeded {
    /// Seeded to verify or to run first.
    As(Reuse),
    /// A live node turned up under the miss mutex and was used as it is.
    Live,
    /// The value failed to decode, so the node was created cold.
    Corrupt,
}

/// One live entry a checkpoint's snapshot found, typed values hidden.
pub(crate) struct Snap {
    pub(crate) table: u32,
    pub(crate) handle: ErasedIncr,
    /// The node's `changed_at` when `inspect` read it.
    pub(crate) stamp: u64,
    pub(crate) deps: Vec<LiveDep>,
    pub(crate) in_group: bool,
    pub(crate) item: Box<dyn Encode>,
}

pub(crate) trait Encode {
    /// The key's and value's bytes, or `None` if either does not decode
    /// back equal to itself.
    fn encode(&self) -> Option<(Vec<u8>, Vec<u8>)>;
}

struct Typed<K, T> {
    key: K,
    value: T,
}

impl<K: MemoKey, T: MemoValue> Encode for Typed<K, T> {
    fn encode(&self) -> Option<(Vec<u8>, Vec<u8>)> {
        let key = encode(&self.key)?;
        let value = encode(&self.value)?;
        let same = with_value_code(|| {
            decode::<K>(&key).is_some_and(|k| k == self.key)
                && decode::<T>(&value).is_some_and(|v| v == self.value)
        });
        same.then_some((key, value))
    }
}

pub(crate) fn encode<X: Serialize>(x: &X) -> Option<Vec<u8>> {
    with_value_code(|| postcard::to_allocvec(x).ok())
}

pub(crate) fn decode<X: DeserializeOwned>(bytes: &[u8]) -> Option<X> {
    with_value_code(|| postcard::from_bytes(bytes).ok())
}

/// A table's typed operations for code that walks entries of every table.
pub(crate) trait ErasedTable<C: MemoCells> {
    fn as_any(&self) -> &dyn Any;

    /// The live node for an encoded key; `Err` if the key doesn't decode.
    fn probe_bytes(&self, rt: &Runtime<C>, key: &[u8]) -> Result<Option<ErasedIncr>, ()>;

    /// The node for an encoded key, created cold on a miss; `None` if the
    /// key doesn't decode.
    fn cold_bytes(&self, core: &CoreRef<C>, rt: &Runtime<C>, key: &[u8]) -> Option<ErasedIncr>;

    /// [`TableCore::seed`] for an encoded key; `None` if it doesn't decode.
    #[allow(clippy::too_many_arguments)]
    fn seed_bytes(
        &self,
        core: &CoreRef<C>,
        rt: &Runtime<C>,
        key: &[u8],
        value: &[u8],
        deps: &[SeedDep],
        reuse: Reuse,
        base: &SeedBase,
        bag: &mut Deferred<C>,
    ) -> Option<(ErasedIncr, Seeded)>;

    /// Every live entry that is Clean through `inspect`, with its deps
    /// mapped under `map`'s hold.
    fn snapshot(&self, rt: &Runtime<C>, map: &DepMap, memo: u64, out: &mut Vec<Snap>);
}

pub(crate) struct TableCore<K: MemoKey, T: MemoValue, C: MemoCells> {
    pub(crate) index: u32,
    pub(crate) nodes: NodeTable<K, T, C>,
    pub(crate) f: Arc<MemoFn<K, T, C>>,
}

impl<K: MemoKey, T: MemoValue, C: MemoCells> TableCore<K, T, C> {
    fn closure(&self, core: &CoreRef<C>, key: K) -> impl Fn(&Runtime<C>) -> T + Send + Sync {
        let f = Arc::clone(&self.f);
        let core = core.clone();
        move |rt| f(&MemoCtx { rt, core: &core }, &key)
    }

    pub(crate) fn probe(&self, rt: &Runtime<C>, key: &K) -> Option<Incr<T>> {
        probe_unchecked(&self.nodes, rt, key)
    }

    pub(crate) fn cold(&self, core: &CoreRef<C>, rt: &Runtime<C>, key: K) -> Incr<T> {
        let f = self.closure(core, with_value_code(|| key.clone()));
        self.nodes.get_or_insert_query(rt, key, f)
    }

    /// Creates `key`'s node seeded with the stored `value` and `deps`, or
    /// cold if the value doesn't decode. The decoded value drops into
    /// `bag` when the seed doesn't take it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn seed(
        &self,
        core: &CoreRef<C>,
        rt: &Runtime<C>,
        key: K,
        value: &[u8],
        deps: &[SeedDep],
        reuse: Reuse,
        base: &SeedBase,
        bag: &mut Deferred<C>,
    ) -> (Incr<T>, Seeded) {
        let Some(value) = decode::<T>(value) else {
            return (self.cold(core, rt, key), Seeded::Corrupt);
        };
        // A failed seed's value comes back here rather than to the miss,
        // so it drops in the restore's bag, after M.
        let mut value = Some(value);
        let mut seeded = Seeded::Live;
        let f = self.closure(core, with_value_code(|| key.clone()));
        let handle =
            get_or_insert_query_seeded(&self.nodes, rt, key, f, QueryOpts::default(), |rt, q| {
                let stored = value.take().expect("a miss seeds once");
                match seed_query(rt, base, q, stored, deps, reuse) {
                    Ok(reuse) => {
                        seeded = Seeded::As(reuse);
                        None
                    }
                    Err((_, failed)) => {
                        value = Some(failed);
                        None
                    }
                }
            });
        if let Some(value) = value {
            bag.push(value);
        }
        (handle, seeded)
    }
}

impl<K: MemoKey, T: MemoValue, C: MemoCells> ErasedTable<C> for TableCore<K, T, C> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn probe_bytes(&self, rt: &Runtime<C>, key: &[u8]) -> Result<Option<ErasedIncr>, ()> {
        let key = decode::<K>(key).ok_or(())?;
        Ok(self.probe(rt, &key).map(Incr::erase))
    }

    fn cold_bytes(&self, core: &CoreRef<C>, rt: &Runtime<C>, key: &[u8]) -> Option<ErasedIncr> {
        let key = decode::<K>(key)?;
        Some(self.cold(core, rt, key).erase())
    }

    fn seed_bytes(
        &self,
        core: &CoreRef<C>,
        rt: &Runtime<C>,
        key: &[u8],
        value: &[u8],
        deps: &[SeedDep],
        reuse: Reuse,
        base: &SeedBase,
        bag: &mut Deferred<C>,
    ) -> Option<(ErasedIncr, Seeded)> {
        let key = decode::<K>(key)?;
        let (handle, seeded) = self.seed(core, rt, key, value, deps, reuse, base, bag);
        Some((handle.erase(), seeded))
    }

    fn snapshot(&self, rt: &Runtime<C>, map: &DepMap, memo: u64, out: &mut Vec<Snap>) {
        let mut live = Vec::new();
        self.nodes.for_each(rt, |key, handle| {
            live.push((with_value_code(|| key.clone()), handle));
        });
        for (key, handle) in live {
            let Some(seen) = inspect(rt, handle) else {
                continue;
            };
            // Read after `inspect` returns, so an input's stamp check sees
            // at least the run whose value was cloned.
            let deps = seen
                .deps
                .iter()
                .map(|&d| map_live(map, rt, memo, d))
                .collect();
            out.push(Snap {
                table: self.index,
                handle: handle.erase(),
                stamp: seen.changed_at,
                deps,
                in_group: seen.in_group,
                item: Box::new(Typed {
                    key,
                    value: seen.value,
                }),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use incr_core::{Local, Shared};

    fn send_sync<X: Send + Sync>() {}

    // What `CoreRef`'s `Send` and `Sync` rest on.
    #[test]
    fn table_cores_and_the_externals_table_are_send_and_sync() {
        send_sync::<TableCore<String, Vec<u8>, Shared>>();
        send_sync::<TableCore<String, Vec<u8>, Local>>();
        send_sync::<NodeTable<String, u128, Shared>>();
        send_sync::<NodeTable<String, u128, Local>>();
    }
}
