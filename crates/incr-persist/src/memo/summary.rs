//! What a full write on a backend without deltas keeps of the generation
//! it wrote, so that the next full write can evict and answer query-dep
//! probes without reading that generation first.

use super::codec::decode_view;
use super::{fingerprint, key_hash, EncodedEntry, Generation, StoredDep};

/// Per item of one generation, in [`Generation::items`] order, its entry
/// without the value and the [`fingerprint`] of the value. Every
/// `(table, key_hash)` has one item.
#[derive(Debug)]
pub(crate) struct Summary {
    rows: Vec<Row>,
    fps: Vec<u128>,
}

#[derive(Clone, Debug)]
pub(crate) struct Row {
    pub(crate) key: Box<[u8]>,
    /// The query deps, tables in the catalog the row's table is in.
    pub(crate) deps: Box<[(u32, Box<[u8]>)]>,
    pub(crate) last_used: u64,
}

impl Row {
    pub(crate) fn new(key: &[u8], deps: &[StoredDep], last_used: u64) -> Self {
        Row {
            key: key.into(),
            deps: deps
                .iter()
                .filter_map(|dep| match dep {
                    StoredDep::Query { table, key, .. } => Some((*table, key.as_slice().into())),
                    _ => None,
                })
                .collect(),
            last_used,
        }
    }
}

impl Summary {
    pub(crate) fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// The row and value fp of the entry of `table` and `key` in
    /// `generation`, the generation this summary is of.
    pub(crate) fn find(
        &self,
        generation: &Generation,
        table: u32,
        key: &[u8],
    ) -> Option<(&Row, u128)> {
        let hash = key_hash(key);
        let items = generation.items();
        let at = items.partition_point(|item| (item.table, item.key_hash) < (table, hash));
        let item = items.get(at)?;
        let row = &self.rows[at];
        (item.table == table && item.key_hash == hash && *row.key == *key)
            .then_some((row, self.fps[at]))
    }
}

/// The rows of the frames a full write emits, gathered as they go out.
#[derive(Default)]
pub(crate) struct Summing {
    rows: Vec<((u32, u64, u32), Row, u128)>,
    failed: bool,
}

impl Summing {
    /// Adds `frame`, of the generation whose frame tag is `tag`.
    pub(crate) fn add(&mut self, frame: &EncodedEntry, tag: u64) {
        if self.failed {
            return;
        }
        let row = decode_view(&frame.frame, tag)
            .ok()
            .filter(|entry| entry.table == frame.table && key_hash(entry.key) == frame.key_hash)
            .map(|entry| {
                (
                    Row::new(entry.key, &entry.deps, entry.last_used),
                    fingerprint(entry.value),
                )
            });
        match (row, u32::try_from(frame.frame.len())) {
            (Some((row, fp)), Ok(len)) => {
                self.rows
                    .push(((frame.table, frame.key_hash, len), row, fp))
            }
            _ => {
                self.failed = true;
                self.rows = Vec::new();
            }
        }
    }

    /// The summary of `generation`, written from the frames added, or
    /// `None` when its items do not pair one to one with them by table,
    /// key hash and length, as when two keys of a table share a hash.
    pub(crate) fn finish(mut self, generation: &Generation) -> Option<Summary> {
        let items = generation.items();
        if self.failed || self.rows.len() != items.len() {
            return None;
        }
        self.rows
            .sort_unstable_by_key(|&((table, hash, _), ..)| (table, hash));
        let unique = items
            .windows(2)
            .all(|pair| (pair[0].table, pair[0].key_hash) != (pair[1].table, pair[1].key_hash));
        let paired = unique
            && items
                .iter()
                .zip(&self.rows)
                .all(|(item, &(found, ..))| (item.table, item.key_hash, item.len) == found);
        paired.then(|| {
            let (rows, fps) = self.rows.into_iter().map(|(_, row, fp)| (row, fp)).unzip();
            Summary { rows, fps }
        })
    }
}
