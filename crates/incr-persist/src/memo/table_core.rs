use std::any::Any;
use std::collections::{HashMap, HashSet};
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
use crate::value_context::{in_memo, EncodeMismatch, MemoPart};

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
    /// `seed_query` refused the node (a dep deleted since the restore
    /// mapped it), so the node was created cold.
    Failed,
}

/// One dirty live entry a checkpoint's snapshot found, typed values
/// hidden.
pub(crate) struct Snap {
    pub(crate) table: u32,
    pub(crate) handle: ErasedIncr,
    /// The entry's dirty mark when the snapshot copied the dirty set.
    pub(crate) mark: u64,
    /// The node's `changed_at` when `inspect` read it.
    pub(crate) stamp: u64,
    pub(crate) deps: Vec<LiveDep>,
    pub(crate) in_group: bool,
    pub(crate) value: Box<dyn Encode>,
}

pub(crate) trait Encode {
    /// The bytes, or the part that does not encode or does not decode back
    /// equal to itself.
    fn encode(&self) -> Result<Vec<u8>, EncodeMismatch>;
}

struct Part<X> {
    table: Arc<str>,
    part: MemoPart,
    x: X,
}

impl<X: Serialize + DeserializeOwned + PartialEq> Encode for Part<X> {
    fn encode(&self) -> Result<Vec<u8>, EncodeMismatch> {
        let mismatch = || EncodeMismatch {
            table: self.table.to_string(),
            part: self.part,
        };
        let bytes = encode(&self.table, self.part, &self.x).ok_or_else(mismatch)?;
        let same = decode::<X>(&self.table, self.part, &bytes)
            .is_some_and(|x| with_value_code(|| x == self.x));
        if same {
            Ok(bytes)
        } else {
            Err(mismatch())
        }
    }
}

/// A live entry's key, found by handle for a checkpoint.
pub(crate) struct KeyOf {
    pub(crate) table: u32,
    pub(crate) key: Box<dyn Encode>,
}

pub(crate) fn encode<X: Serialize>(table: &str, part: MemoPart, x: &X) -> Option<Vec<u8>> {
    in_memo(table, part, || {
        with_value_code(|| postcard::to_allocvec(x).ok())
    })
}

pub(crate) fn decode<X: DeserializeOwned>(table: &str, part: MemoPart, bytes: &[u8]) -> Option<X> {
    in_memo(table, part, || {
        with_value_code(|| postcard::from_bytes(bytes).ok())
    })
}

/// A table's typed operations for code that walks entries of every table.
pub(crate) trait ErasedTable<C: MemoCells> {
    fn as_any(&self) -> &dyn Any;

    /// The live node for an encoded key; `Err` if the key doesn't decode.
    /// The decoded key goes to `bag`.
    fn probe_bytes(
        &self,
        rt: &Runtime<C>,
        key: &[u8],
        bag: &mut Deferred<C>,
    ) -> Result<Option<ErasedIncr>, ()>;

    /// [`TableCore::cold_into`] for an encoded key; `None` if it doesn't
    /// decode.
    fn cold_bytes(
        &self,
        core: &CoreRef<C>,
        rt: &Runtime<C>,
        key: &[u8],
        bag: &mut Deferred<C>,
    ) -> Option<ErasedIncr>;

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

    /// Dirty entry `handle` with its deps mapped under `map`'s hold, if it
    /// is Clean through `inspect`.
    fn inspect_dirty(
        &self,
        rt: &Runtime<C>,
        map: &DepMap,
        memo: u64,
        handle: ErasedIncr,
        mark: u64,
    ) -> Option<Snap>;

    /// The keys of the live entries in `wanted`, cloned into `out`.
    fn keys_of(
        &self,
        rt: &Runtime<C>,
        wanted: &HashSet<ErasedIncr>,
        out: &mut HashMap<ErasedIncr, KeyOf>,
    );
}

pub(crate) struct TableCore<K: MemoKey, T: MemoValue, C: MemoCells> {
    pub(crate) index: u32,
    pub(crate) name: Arc<str>,
    pub(crate) nodes: NodeTable<K, T, C>,
    pub(crate) f: Arc<MemoFn<K, T, C>>,
}

impl<K: MemoKey, T: MemoValue, C: MemoCells> TableCore<K, T, C> {
    /// The node's compute: `f`, then a record in the dirty set. A run that
    /// unwinds records nothing; its node is not Clean, so no checkpoint
    /// takes it.
    fn closure(&self, core: &CoreRef<C>, key: K) -> impl Fn(&Runtime<C>) -> T + Send + Sync {
        let f = Arc::clone(&self.f);
        let core = core.clone();
        let (memo, index) = (core.id, self.index);
        move |rt| {
            let value = f(&MemoCtx { rt, core: &core }, &key);
            let table: &TableCore<K, T, C> = core.table(memo, index);
            if let Some(handle) = table.probe(rt, &key) {
                core.computed(index, handle.erase());
            }
            value
        }
    }

    pub(crate) fn probe(&self, rt: &Runtime<C>, key: &K) -> Option<Incr<T>> {
        probe_unchecked(&self.nodes, rt, key)
    }

    pub(crate) fn cold(&self, core: &CoreRef<C>, rt: &Runtime<C>, key: K) -> Incr<T> {
        let f = self.closure(core, with_value_code(|| key.clone()));
        self.nodes.get_or_insert_query(rt, key, f)
    }

    /// [`Self::cold`] for a caller holding M: what the table doesn't keep
    /// goes to `bag`.
    pub(crate) fn cold_into(
        &self,
        core: &CoreRef<C>,
        rt: &Runtime<C>,
        key: K,
        bag: &mut Deferred<C>,
    ) -> Incr<T> {
        let f = self.closure(core, with_value_code(|| key.clone()));
        get_or_insert_query_seeded(
            &self.nodes,
            rt,
            key,
            f,
            QueryOpts::default(),
            |_, _| None,
            bag,
        )
    }

    /// Creates `key`'s node seeded with the stored `value` and `deps`, or
    /// cold if the value doesn't decode. What the table doesn't keep,
    /// the decoded value of a failed seed included, goes to `bag`.
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
        let Some(value) = decode::<T>(&self.name, MemoPart::Value, value) else {
            return (self.cold_into(core, rt, key, bag), Seeded::Corrupt);
        };
        let mut value = Some(value);
        let mut seeded = Seeded::Live;
        let f = self.closure(core, with_value_code(|| key.clone()));
        let handle = get_or_insert_query_seeded(
            &self.nodes,
            rt,
            key,
            f,
            QueryOpts::default(),
            |rt, q| {
                let stored = value.take().expect("a miss seeds once");
                match seed_query(rt, base, q, stored, deps, reuse) {
                    Ok(reuse) => {
                        seeded = Seeded::As(reuse);
                        None
                    }
                    Err((_, failed)) => {
                        seeded = Seeded::Failed;
                        Some(failed)
                    }
                }
            },
            bag,
        );
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

    fn probe_bytes(
        &self,
        rt: &Runtime<C>,
        key: &[u8],
        bag: &mut Deferred<C>,
    ) -> Result<Option<ErasedIncr>, ()> {
        let key = decode::<K>(&self.name, MemoPart::Key, key).ok_or(())?;
        let live = self.probe(rt, &key).map(Incr::erase);
        bag.push(key);
        Ok(live)
    }

    fn cold_bytes(
        &self,
        core: &CoreRef<C>,
        rt: &Runtime<C>,
        key: &[u8],
        bag: &mut Deferred<C>,
    ) -> Option<ErasedIncr> {
        let key = decode::<K>(&self.name, MemoPart::Key, key)?;
        Some(self.cold_into(core, rt, key, bag).erase())
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
        let key = decode::<K>(&self.name, MemoPart::Key, key)?;
        let (handle, seeded) = self.seed(core, rt, key, value, deps, reuse, base, bag);
        Some((handle.erase(), seeded))
    }

    fn inspect_dirty(
        &self,
        rt: &Runtime<C>,
        map: &DepMap,
        memo: u64,
        handle: ErasedIncr,
        mark: u64,
    ) -> Option<Snap> {
        let typed = rt.typed_handle::<T>(handle)?;
        let seen = inspect(rt, typed)?;
        // Read after `inspect` returns, so an input's stamp check sees at
        // least the run whose value was cloned.
        let deps = seen
            .deps
            .iter()
            .map(|&d| map_live(map, rt, memo, d))
            .collect();
        Some(Snap {
            table: self.index,
            handle,
            mark,
            stamp: seen.changed_at,
            deps,
            in_group: seen.in_group,
            value: Box::new(Part {
                table: Arc::clone(&self.name),
                part: MemoPart::Value,
                x: seen.value,
            }),
        })
    }

    fn keys_of(
        &self,
        rt: &Runtime<C>,
        wanted: &HashSet<ErasedIncr>,
        out: &mut HashMap<ErasedIncr, KeyOf>,
    ) {
        self.nodes.for_each(rt, |key, handle| {
            let handle = handle.erase();
            if wanted.contains(&handle) {
                let key = Part {
                    table: Arc::clone(&self.name),
                    part: MemoPart::Key,
                    x: with_value_code(|| key.clone()),
                };
                out.insert(
                    handle,
                    KeyOf {
                        table: self.index,
                        key: Box::new(key),
                    },
                );
            }
        });
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
