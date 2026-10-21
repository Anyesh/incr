//! Between a live dep and its stored form, under a `DepIndex` hold.

use incr_core::persist_hooks::changed_at_erased;
use incr_core::{Cells, ErasedIncr, Runtime};

use super::StoredDep;
use crate::store::dep_index::{DepFp, DepMap};
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

pub(crate) fn map_live<C: Cells>(
    map: &DepMap,
    rt: &Runtime<C>,
    memo: u64,
    dep: ErasedIncr,
) -> LiveDep {
    if let Some(key) = map.key_of(dep) {
        let entry = map.entry(key).expect("a handle's key is registered");
        return match (entry.kind, entry.fp) {
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
        };
    }
    if let Some((owner, name)) = map.external_of(dep) {
        let external = map
            .external(owner, name)
            .expect("an external's name is registered");
        return if owner == memo && vouches(rt, dep, Some(external.stamp)) {
            LiveDep::Stored(StoredDep::External {
                name: name.to_owned(),
                fp: external.fp,
            })
        } else {
            LiveDep::Unmapped
        };
    }
    LiveDep::Query(dep)
}

/// A stored store-key or external dep's live node and whether its
/// fingerprint still matches; `None` when nothing live vouches for it.
/// `None` for a query dep, which the restore walk maps.
pub(crate) fn map_stored<C: Cells>(
    map: &DepMap,
    rt: &Runtime<C>,
    memo: u64,
    dep: &StoredDep,
) -> Option<(ErasedIncr, bool)> {
    match dep {
        StoredDep::Input { key, fp } => {
            let entry = map.entry(key)?;
            match (entry.kind, entry.fp) {
                (Kind::Input, DepFp::Known(live)) if vouches(rt, entry.handle, entry.stamp) => {
                    Some((entry.handle, live == *fp))
                }
                _ => None,
            }
        }
        StoredDep::Collection { key, fp } => {
            let entry = map.entry(key)?;
            match (entry.kind, entry.fp) {
                (Kind::Collection, DepFp::Known(live)) => Some((entry.handle, live == *fp)),
                _ => None,
            }
        }
        StoredDep::External { name, fp } => {
            let external = map.external(memo, name)?;
            let handle = external.handle.erase();
            vouches(rt, handle, Some(external.stamp)).then_some((handle, external.fp == *fp))
        }
        StoredDep::Query { .. } => None,
    }
}
