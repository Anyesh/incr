//! `Memo::checkpoint`: snapshot every live entry under one `DepIndex`
//! read hold, encode and write the next generation with no incr lock
//! held, then swap the index to it.

use std::collections::HashMap;

use incr_core::{check_value_code, ErasedIncr, Runtime};

use super::mapping::LiveDep;
use super::memo_core::{CoreRef, IndexGen, MemoCells};
use super::table_core::Snap;
use super::{fingerprint, CheckpointInput, CheckpointStats, Entry, StoredDep};
use crate::Result;

impl<C: MemoCells> CoreRef<C> {
    pub(crate) fn checkpoint(&self, rt: &Runtime<C>) -> Result<CheckpointStats> {
        assert!(
            !rt.frame_active(),
            "incr-persist: Memo::checkpoint called from inside a query's compute"
        );
        check_value_code();
        let _checkpoint = self.lock_checkpoint();
        let snaps = {
            let mut state = self.lock_memo();
            state.seed_info.retain(|&handle, _| rt.is_alive(handle));
            // The read hold excludes every mapped write, and each finishes
            // its walk inside its own hold, so a Clean entry over mapped
            // deps equals its closure of the current mapped values.
            let map = self.deps.read();
            let mut snaps = Vec::new();
            for table in &self.tables {
                table.snapshot(rt, &map, &mut snaps);
            }
            snaps
        };
        let done = self.backend.checkpoint(CheckpointInput {
            build: self.build,
            tables: self.catalog.clone(),
            fresh: fresh(snaps),
            max_bytes: self.max_bytes,
        })?;
        let mut state = self.lock_memo();
        self.swap_index(IndexGen::new(done.generation, self.build, &self.catalog));
        state.stats.checkpoints += 1;
        // Restores hold M, so none is still reading the old generation.
        if let Some(old) = done.replaced {
            self.backend.remove_generation(old)?;
        }
        Ok(done.stats)
    }
}

/// The snapshot's entries that every keep rule but the query-dep fixpoint
/// allows: encoded and decoding back equal, not in a cycle group, and
/// every dep mapped. A query dep maps to another snapshot entry, whose
/// value's fingerprint it records.
fn fresh(snaps: Vec<Snap>) -> Vec<Entry> {
    let encoded: Vec<_> = snaps.iter().map(|snap| snap.item.encode()).collect();
    let by_handle: HashMap<ErasedIncr, (u32, &[u8], u128)> = snaps
        .iter()
        .zip(&encoded)
        .filter_map(|(snap, bytes)| {
            let (key, value) = bytes.as_ref()?;
            Some((
                snap.handle,
                (snap.table, key.as_slice(), fingerprint(value)),
            ))
        })
        .collect();
    let mut fresh = Vec::new();
    for (snap, bytes) in snaps.iter().zip(&encoded) {
        let Some((key, value)) = bytes else { continue };
        if snap.in_group {
            continue;
        }
        let deps: Option<Vec<StoredDep>> = snap
            .deps
            .iter()
            .map(|dep| match dep {
                LiveDep::Stored(stored) => Some(stored.clone()),
                LiveDep::Query(handle) => {
                    let &(table, key, fp) = by_handle.get(handle)?;
                    Some(StoredDep::Query {
                        table,
                        key: key.to_vec(),
                        fp,
                    })
                }
                LiveDep::Unmapped => None,
            })
            .collect();
        let Some(deps) = deps else { continue };
        fresh.push(Entry {
            table: snap.table,
            key: key.clone(),
            value: value.clone(),
            deps,
            last_used: 0,
        });
    }
    fresh
}
