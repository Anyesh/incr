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
    /// Sorted by `(table, key_hash)` so that `find` is a binary search.
    items: Vec<IndexItem>,
}

impl Generation {
    pub fn new(id: GenId, header: GenHeader, mut items: Vec<IndexItem>) -> Self {
        items.sort_unstable_by_key(|item| (item.table, item.key_hash, item.loc));
        Generation { id, header, items }
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
        let key = (table, key_hash);
        let start = self
            .items
            .partition_point(|item| (item.table, item.key_hash) < key);
        let end =
            start + self.items[start..].partition_point(|item| (item.table, item.key_hash) == key);
        &self.items[start..end]
    }

    /// The summed length of every entry frame, which `max_bytes` bounds.
    pub fn entry_bytes(&self) -> u64 {
        self.items.iter().map(|item| u64::from(item.len)).sum()
    }
}

#[cfg(test)]
#[path = "generation_tests.rs"]
mod tests;
