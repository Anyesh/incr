use super::{GenHeader, IndexItem};

/// Names one generation of a memo store. Each checkpoint writes the next
/// id after the generation it replaces; the first is `GenId(1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GenId(pub u64);

/// A generation's header and the index of its entry frames, as a backend
/// loaded or wrote it. Entry values stay in the backend until
/// [`MemoBackend::read_entry`](super::MemoBackend::read_entry) fetches one.
#[derive(Clone, Debug)]
pub struct Generation {
    id: GenId,
    header: GenHeader,
    /// Sorted by `(table, key_hash, loc)`, so equal keys form one run.
    items: Vec<IndexItem>,
    /// Open addressing over the runs of `items`: a slot is 0 or one plus
    /// the index of a run's first item. The length is a power of two at
    /// least twice the item count, so a probe for a missing key stops soon.
    slots: Vec<u32>,
    /// `64 - log2(slots.len())`, for [`home`].
    shift: u32,
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
    pub fn new(id: GenId, header: GenHeader, mut items: Vec<IndexItem>) -> Self {
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
        Generation {
            id,
            header,
            items,
            slots,
            shift,
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

    pub fn id(&self) -> GenId {
        self.id
    }

    pub fn header(&self) -> &GenHeader {
        &self.header
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
    pub fn entry_bytes(&self) -> u64 {
        self.items.iter().map(|item| u64::from(item.len)).sum()
    }
}

#[cfg(test)]
#[path = "generation_tests.rs"]
mod tests;
