use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

use super::layers::Live;
use super::summary::Summary;
use super::{DeltaHeader, GenHeader, IndexItem};
use crate::{Error, Result};

/// Names one generation of a memo store. Each checkpoint writes the next
/// id after the generation it replaces; the first is `GenId(1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GenId(pub u64);

/// A generation's header and the index of its entry frames, as a backend
/// loaded or wrote it. Entry values stay in the backend until
/// [`MemoBackend::read_entry`](super::MemoBackend::read_entry) fetches one.
#[derive(Clone, Debug)]
pub struct Generation {
    id: GenId,
    header: GenHeader,
    /// `header.frame_tag()`, or the [`DeltaHeader::frame_tag`] of a delta.
    frame_tag: u64,
    /// A delta's `(base, parent)`.
    delta: Option<(GenId, GenId)>,
    /// Sorted by `(table, key_hash, loc)`, so equal keys form one run.
    items: Vec<IndexItem>,
    /// Open addressing over the runs of `items`: a slot is 0 or one plus
    /// the index of a run's first item. The length is a power of two at
    /// least twice the item count, so a probe for a missing key stops soon.
    slots: Vec<u32>,
    /// `64 - log2(slots.len())`, for [`home`].
    shift: u32,
    /// The items that are not tombstones, and their summed lengths.
    entries: u64,
    entry_bytes: u64,
    /// For a delta, what its chain up to and including it shows, once
    /// computed: its older layers are fixed by its base and parent ids.
    shown: OnceLock<Live>,
    /// The eviction metadata of its entries, when a checkpoint of this
    /// process wrote it on a backend without deltas.
    summary: OnceLock<Arc<Summary>>,
}

/// The first slot probed for `(table, key_hash)`: the top bits of the hash,
/// which xxh3 spreads evenly, offset per table. Within one table the home
/// slot then rises with `key_hash`, so `new` fills the slots in one sweep
/// over the sorted items instead of writing them at random.
fn home(table: u32, key_hash: u64, shift: u32, mask: usize) -> usize {
    ((key_hash >> shift) as usize).wrapping_add((table as usize).wrapping_mul(0x9E37_79B9)) & mask
}

fn same_key(item: &IndexItem, table: u32, key_hash: u64) -> bool {
    item.table == table && item.key_hash == key_hash
}

impl Generation {
    /// Panics if `items` holds `u32::MAX` items or more.
    pub fn new(id: GenId, header: GenHeader, items: Vec<IndexItem>) -> Self {
        let frame_tag = header.frame_tag();
        Generation::build(id, header, frame_tag, None, items)
    }

    /// A delta generation, whose frames carry `delta`'s own frame tag and
    /// whose items may be tombstones. Panics like [`new`](Self::new).
    pub fn new_delta(id: GenId, delta: DeltaHeader, items: Vec<IndexItem>) -> Self {
        let frame_tag = delta.frame_tag();
        let link = Some((delta.base, delta.parent));
        Generation::build(id, delta.header, frame_tag, link, items)
    }

    fn build(
        id: GenId,
        header: GenHeader,
        frame_tag: u64,
        delta: Option<(GenId, GenId)>,
        mut items: Vec<IndexItem>,
    ) -> Self {
        items.sort_unstable_by_key(|item| (item.table, item.key_hash, item.loc));
        let len = (items.len() * 2).next_power_of_two().max(2);
        let shift = 64 - len.trailing_zeros();
        let mask = len - 1;
        let mut slots = vec![0_u32; len];
        for (start, item) in items.iter().enumerate() {
            if start > 0 && same_key(&items[start - 1], item.table, item.key_hash) {
                continue;
            }
            let mut at = home(item.table, item.key_hash, shift, mask);
            while slots[at] != 0 {
                at = (at + 1) & mask;
            }
            slots[at] = u32::try_from(start + 1).expect("a generation holds under u32::MAX items");
        }
        let (entries, entry_bytes) = items
            .iter()
            .filter(|item| !item.is_tombstone())
            .fold((0, 0), |(n, bytes), item| {
                (n + 1, bytes + u64::from(item.len))
            });
        Generation {
            id,
            frame_tag,
            delta,
            header,
            items,
            slots,
            shift,
            entries,
            entry_bytes,
            shown: OnceLock::new(),
            summary: OnceLock::new(),
        }
    }

    /// What a backend's `load` returns when the current generation's header
    /// or index is unreadable: a generation that holds nothing, so restores
    /// miss and the next checkpoint replaces it without merging anything.
    /// Its build fingerprint is 0 and its catalog empty.
    pub fn empty(id: GenId) -> Self {
        let header = GenHeader {
            build: 0,
            seq: 0,
            tables: Vec::new(),
        };
        Generation::new(id, header, Vec::new())
    }

    /// What a backend's `load_chain` returns for a delta layer whose header
    /// or index is unreadable: a delta that holds nothing, so probes fall
    /// through to older layers. Its build fingerprint is 0 and its catalog
    /// empty.
    pub fn empty_delta(id: GenId, base: GenId, parent: GenId) -> Self {
        let header = GenHeader {
            build: 0,
            seq: 0,
            tables: Vec::new(),
        };
        Generation::new_delta(
            id,
            DeltaHeader {
                header,
                base,
                parent,
            },
            Vec::new(),
        )
    }

    pub fn id(&self) -> GenId {
        self.id
    }

    pub fn header(&self) -> &GenHeader {
        &self.header
    }

    /// The tag its entry frames are decoded with: the header's
    /// [`GenHeader::frame_tag`], or for a delta its
    /// [`DeltaHeader::frame_tag`].
    pub fn frame_tag(&self) -> u64 {
        self.frame_tag
    }

    /// The full generation a delta builds on, `None` for a full one.
    pub fn base(&self) -> Option<GenId> {
        self.delta.map(|(base, _)| base)
    }

    /// The layer a delta was written on, `None` for a full generation.
    pub fn parent(&self) -> Option<GenId> {
        self.delta.map(|(_, parent)| parent)
    }

    /// Every item, sorted by `(table, key_hash, loc)`.
    pub fn items(&self) -> &[IndexItem] {
        &self.items
    }

    /// The items of table `table` whose key hashes to `key_hash`. More than
    /// one means a hash collision; the decoded keys tell them apart.
    pub fn find(&self, table: u32, key_hash: u64) -> &[IndexItem] {
        let mask = self.slots.len() - 1;
        let mut at = home(table, key_hash, self.shift, mask);
        loop {
            let Some(start) = (self.slots[at] as usize).checked_sub(1) else {
                return &[];
            };
            let run = &self.items[start..];
            if same_key(&run[0], table, key_hash) {
                let len = run
                    .iter()
                    .take_while(|item| same_key(item, table, key_hash))
                    .count();
                return &run[..len];
            }
            at = (at + 1) & mask;
        }
    }

    /// The summed length of every entry frame, which `max_bytes` bounds.
    /// Tombstones add nothing.
    pub fn entry_bytes(&self) -> u64 {
        self.entry_bytes
    }

    /// The items that are not tombstones.
    pub(crate) fn entry_count(&self) -> u64 {
        self.entries
    }

    pub(crate) fn shown(&self) -> Option<Live> {
        self.shown.get().copied()
    }

    pub(crate) fn set_shown(&self, live: Live) {
        let _ = self.shown.set(live);
    }

    pub(crate) fn summary(&self) -> Option<&Summary> {
        self.summary.get().map(|summary| &**summary)
    }

    pub(crate) fn set_summary(&self, summary: Summary) {
        let _ = self.summary.set(Arc::new(summary));
    }
}

/// The layers a store's current state is read through, oldest first: a
/// full generation, then zero or more deltas, each written on the one
/// before it. An entry is found by probing the layers newest first, and a
/// tombstone there is a miss.
#[derive(Clone, Debug)]
pub struct Chain {
    layers: Vec<Arc<Generation>>,
}

impl Chain {
    /// Fails with [`Error::Corrupt`] unless `layers` is a full generation
    /// followed by deltas whose base is that generation and whose parent is
    /// the layer before each.
    pub fn new(layers: Vec<Arc<Generation>>) -> Result<Self> {
        let Some((base, deltas)) = layers.split_first() else {
            return Err(Error::Corrupt("a memo chain has no layers"));
        };
        if base.base().is_some() {
            return Err(Error::Corrupt("a memo chain starts with a delta"));
        }
        let mut parent = base.id();
        for delta in deltas {
            if delta.base() != Some(base.id()) || delta.parent() != Some(parent) {
                return Err(Error::Corrupt("a memo delta is not on the layer before it"));
            }
            parent = delta.id();
        }
        Ok(Chain { layers })
    }

    /// Oldest first.
    pub fn layers(&self) -> &[Arc<Generation>] {
        &self.layers
    }

    pub fn into_layers(self) -> Vec<Arc<Generation>> {
        self.layers
    }

    /// The full generation at the bottom.
    pub fn base(&self) -> &Arc<Generation> {
        &self.layers[0]
    }

    /// The newest layer, which a write must name to replace or extend the
    /// chain.
    pub fn tip(&self) -> &Arc<Generation> {
        self.layers.last().expect("a chain has a layer")
    }

    /// The number of deltas.
    pub fn depth(&self) -> usize {
        self.layers.len() - 1
    }
}

impl From<Generation> for Chain {
    /// A chain of one full generation. Panics if `generation` is a delta.
    fn from(generation: Generation) -> Self {
        assert!(
            generation.base().is_none(),
            "a chain starts with a full generation"
        );
        Chain {
            layers: vec![Arc::new(generation)],
        }
    }
}

#[cfg(test)]
#[path = "generation_tests.rs"]
mod tests;
