//! Reading a chain's layers newest first: a key's newest run decides, a
//! tombstone hides every older layer's items with its table and key hash,
//! and a live item whose decoded key differs, a hash collision, lets the
//! probe fall through to older layers.

use std::sync::Arc;

use super::{decode_entry, key_hash, Entry, GenId, Generation, IndexItem};
use crate::Result;

/// What a newest-first probe for one key found.
pub(crate) enum Probe {
    Found(Entry),
    Missing,
    /// A layer the probe needed has been removed.
    Gone,
    /// A frame in the deciding layer failed to decode.
    Corrupt,
}

/// The entry for `key` of catalog table `table` across `layers` (oldest
/// first). `read` fetches a frame as [`MemoBackend::read_entry`] does;
/// its errors are returned as they are.
///
/// [`MemoBackend::read_entry`]: super::MemoBackend::read_entry
pub(crate) fn probe(
    layers: &[Arc<Generation>],
    table: u32,
    key: &[u8],
    mut read: impl FnMut(GenId, &IndexItem) -> Result<Option<Vec<u8>>>,
) -> Result<Probe> {
    let hash = key_hash(key);
    for layer in layers.iter().rev() {
        let run = layer.find(table, hash);
        let mut corrupt = false;
        for item in run.iter().filter(|item| !item.is_tombstone()) {
            let Some(frame) = read(layer.id(), item)? else {
                return Ok(Probe::Gone);
            };
            match decode_entry(&frame, layer.frame_tag()) {
                Ok(entry) if entry.table == table && entry.key == key => {
                    return Ok(Probe::Found(entry))
                }
                Ok(_) => {}
                Err(_) => corrupt = true,
            }
        }
        if corrupt {
            return Ok(Probe::Corrupt);
        }
        if run.iter().any(IndexItem::is_tombstone) {
            return Ok(Probe::Missing);
        }
    }
    Ok(Probe::Missing)
}

/// Whether some layer may hold an entry of `table` whose key hashes to
/// `hash`: the newest layer with a run for it has a live item there.
pub(crate) fn may_hold(layers: &[Arc<Generation>], table: u32, hash: u64) -> bool {
    layers
        .iter()
        .rev()
        .map(|layer| layer.find(table, hash))
        .find(|run| !run.is_empty())
        .is_some_and(|run| run.iter().any(|item| !item.is_tombstone()))
}

/// Entries and summed frame lengths a chain shows its readers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Live {
    pub(crate) entries: u64,
    pub(crate) bytes: u64,
}

impl Live {
    fn of(items: &[IndexItem]) -> Live {
        items
            .iter()
            .filter(|item| !item.is_tombstone())
            .fold(Live::default(), |live, item| Live {
                entries: live.entries + 1,
                bytes: live.bytes + u64::from(item.len),
            })
    }

    /// What `layers` (oldest first, a full generation first) show, by key
    /// hash: each delta's runs replace the newest older run for the same
    /// table and hash. Each delta's totals are kept on it once computed,
    /// so only deltas new to this process cost their items times depth.
    pub(crate) fn of_chain(layers: &[Arc<Generation>]) -> Live {
        let Some((base, deltas)) = layers.split_first() else {
            return Live::default();
        };
        let mut live = Live {
            entries: base.entry_count(),
            bytes: base.entry_bytes(),
        };
        for (at, delta) in deltas.iter().enumerate() {
            live = match delta.shown() {
                Some(shown) => shown,
                None => {
                    let shown = live.with_layer(&layers[..=at], delta.items());
                    delta.set_shown(shown);
                    shown
                }
            };
        }
        live
    }

    /// This chain's totals once a layer of `items`, sorted by
    /// `(table, key_hash)`, is put on `below`.
    pub(crate) fn with_layer(self, below: &[Arc<Generation>], items: &[IndexItem]) -> Live {
        let mut live = self;
        for run in items.chunk_by(|a, b| (a.table, a.key_hash) == (b.table, b.key_hash)) {
            let shadowed = below
                .iter()
                .rev()
                .map(|layer| layer.find(run[0].table, run[0].key_hash))
                .find(|older| !older.is_empty())
                .map_or(Live::default(), Live::of);
            let added = Live::of(run);
            live.entries = live.entries - shadowed.entries + added.entries;
            live.bytes = live.bytes - shadowed.bytes + added.bytes;
        }
        live
    }
}

#[cfg(test)]
#[path = "layers_tests.rs"]
mod tests;
