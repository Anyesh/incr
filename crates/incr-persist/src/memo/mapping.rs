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

pub(crate) fn map_live<C: Cells>(map: &DepMap, rt: &Runtime<C>, dep: ErasedIncr) -> LiveDep {
    if let Some(key) = map.key_of(dep) {
        let entry = map.entry(key).expect("a handle's key is registered");
        return match (entry.kind, entry.fp) {
            (_, DepFp::Untrusted) => LiveDep::Unmapped,
            (Kind::Input, DepFp::Known(fp)) => {
                if entry.stamp.is_some() && changed_at_erased(rt, dep) == entry.stamp {
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
    if let Some(name) = map.external_name(dep) {
        let (_, fp) = map
            .external(name)
            .expect("an external's name is registered");
        return LiveDep::Stored(StoredDep::External {
            name: name.to_owned(),
            fp,
        });
    }
    LiveDep::Query(dep)
}

/// A stored store-key or external dep's live node and whether its
/// fingerprint still matches; `None` when nothing live vouches for it.
/// `None` for a query dep, which the restore walk maps.
pub(crate) fn map_stored<C: Cells>(
    map: &DepMap,
    rt: &Runtime<C>,
    dep: &StoredDep,
) -> Option<(ErasedIncr, bool)> {
    match dep {
        StoredDep::Input { key, fp } => {
            let entry = map.entry(key)?;
            match (entry.kind, entry.fp) {
                (Kind::Input, DepFp::Known(live))
                    if entry.stamp.is_some()
                        && changed_at_erased(rt, entry.handle) == entry.stamp =>
                {
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
            let (handle, live) = map.external(name)?;
            Some((handle.erase(), live == *fp))
        }
        StoredDep::Query { .. } => None,
    }
}
