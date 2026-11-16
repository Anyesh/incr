//! Between a live dep and its stored form, under a `DepIndex` hold.

use incr_core::persist_hooks::{changed_at_erased, Deferred};
use incr_core::{Cells, ErasedIncr, Runtime};

use super::{StoredDep, VALUE_BATCH};
use crate::store::dep_index::{DepEntry, DepFp, DepMap};
use crate::store::keyed::rows_of;
use crate::Kind;

/// What a live dep of a checkpointed entry stores as.
pub(crate) enum LiveDep {
    Stored(StoredDep),
    /// Not a store key or external, so a memo entry or nothing.
    Query(ErasedIncr),
    /// A key that no fingerprint vouches for: untrusted after a panic, or
    /// an input whose `changed_at` has left its mapped stamp (a bypass).
    Unmapped,
}

/// Whether a recorded fp still describes `handle`'s value: only while its
/// `changed_at` is the stamp recorded with the fp. A write around the
/// store or `set_external` moves it.
fn vouches<C: Cells>(rt: &Runtime<C>, handle: ErasedIncr, stamp: Option<u64>) -> bool {
    stamp.is_some() && changed_at_erased(rt, handle) == stamp
}

/// What each of `deps`, the live deps of checkpointed entries, stores as.
/// Row deps are mapped a collection at a time, so their keys' value code
/// runs in a few holds of the store's value scope.
pub(crate) fn map_live<C: Cells>(
    map: &DepMap,
    rt: &Runtime<C>,
    memo: u64,
    deps: &[ErasedIncr],
    bag: &mut Deferred<C>,
) -> Vec<LiveDep> {
    let mut mapped: Vec<Option<LiveDep>> = deps
        .iter()
        .map(|&dep| map_unkeyed(map, rt, memo, dep, bag))
        .collect();
    for (collection, rows) in map.all_rows() {
        let (at, pending): (Vec<usize>, Vec<ErasedIncr>) = mapped
            .iter()
            .zip(deps)
            .enumerate()
            .filter(|(_, (live, _))| live.is_none())
            .map(|(i, (_, &dep))| (i, dep))
            .unzip();
        if pending.is_empty() {
            break;
        }
        let trusted = trusted(map, collection);
        let source = rows_of::<C>(rows);
        let found = pending
            .chunks(VALUE_BATCH)
            .flat_map(|chunk| source.live_rows(chunk, bag));
        for (i, row) in at.into_iter().zip(found) {
            let Some(row) = row else {
                continue;
            };
            mapped[i] = Some(match (row, trusted) {
                (Ok(Some((key_bytes, fp))), true) => LiveDep::Stored(StoredDep::Row {
                    collection: collection.to_owned(),
                    key_bytes,
                    fp,
                }),
                _ => LiveDep::Unmapped,
            });
        }
    }
    mapped
        .into_iter()
        .zip(deps)
        .map(|(live, &dep)| live.unwrap_or(LiveDep::Query(dep)))
        .collect()
}

/// What `dep` stores as if it is a store key or an external; `None` for
/// any other node. A collection's pending fingerprint is computed, with
/// what it reads going to `bag`.
fn map_unkeyed<C: Cells>(
    map: &DepMap,
    rt: &Runtime<C>,
    memo: u64,
    dep: ErasedIncr,
    bag: &mut Deferred<C>,
) -> Option<LiveDep> {
    if let Some(key) = map.key_of(dep) {
        let entry = map.entry(key).expect("a handle's key is registered");
        return Some(match (entry.kind, map.fp(entry, rt, bag)) {
            (_, DepFp::Untrusted) => LiveDep::Unmapped,
            (Kind::Input, DepFp::Known(fp)) => {
                if vouches(rt, dep, entry.stamp) {
                    LiveDep::Stored(StoredDep::Input {
                        key: key.to_owned(),
                        fp,
                    })
                } else {
                    LiveDep::Unmapped
                }
            }
            (Kind::Collection, DepFp::Known(fp)) => LiveDep::Stored(StoredDep::Collection {
                key: key.to_owned(),
                fp,
            }),
        });
    }
    let (owner, name) = map.external_of(dep)?;
    let external = map
        .external(owner, name)
        .expect("an external's name is registered");
    Some(if owner == memo && vouches(rt, dep, Some(external.stamp)) {
        LiveDep::Stored(StoredDep::External {
            name: name.to_owned(),
            fp: external.fp,
        })
    } else {
        LiveDep::Unmapped
    })
}

/// Whether the keyed collection `collection` is registered and trusted,
/// so a panic part-way through one of its writes leaves its rows unmapped.
/// Computes no pending fingerprint.
fn trusted(map: &DepMap, collection: &str) -> bool {
    map.entry(collection).is_some_and(DepEntry::trusted)
}

/// A stored store-key or external dep's live node and whether its
/// fingerprint still matches; `None` when nothing live vouches for it.
/// `None` for a query dep, which the restore walk maps. A row dep's
/// decoded key, and what computing a pending collection fingerprint
/// reads, go to `bag`.
pub(crate) fn map_stored<C: Cells>(
    map: &DepMap,
    rt: &Runtime<C>,
    memo: u64,
    dep: &StoredDep,
    bag: &mut Deferred<C>,
) -> Option<(ErasedIncr, bool)> {
    match dep {
        StoredDep::Input { key, fp } => {
            let entry = map.entry(key)?;
            match (entry.kind, map.fp(entry, rt, bag)) {
                (Kind::Input, DepFp::Known(live)) if vouches(rt, entry.handle, entry.stamp) => {
                    Some((entry.handle, live == *fp))
                }
                _ => None,
            }
        }
        StoredDep::Collection { key, fp } => {
            let entry = map.entry(key)?;
            match (entry.kind, map.fp(entry, rt, bag)) {
                (Kind::Collection, DepFp::Known(live)) => Some((entry.handle, live == *fp)),
                _ => None,
            }
        }
        StoredDep::External { name, fp } => {
            let external = map.external(memo, name)?;
            let handle = external.handle.erase();
            vouches(rt, handle, Some(external.stamp)).then_some((handle, external.fp == *fp))
        }
        StoredDep::Row {
            collection,
            key_bytes,
            fp,
        } => {
            if !trusted(map, collection) {
                return None;
            }
            rows_of::<C>(map.rows(collection)?).stored_row(key_bytes, *fp, bag)
        }
        StoredDep::Query { .. } => None,
    }
}
