//! `Memo::checkpoint`: snapshot the dirty live entries under one
//! `DepIndex` read hold, encode them and write the next generation, merged
//! with the current one, with no incr lock held, then swap the index to it
//! and clean what reached the backend.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use incr_core::persist_hooks::{changed_at_erased, Deferred};
use incr_core::{check_value_code, ErasedIncr, Runtime};

use super::checkpoint::Fate;
use super::mapping::LiveDep;
use super::memo_core::{CoreRef, IndexGen, MemoCells};
use super::table_core::{KeyOf, Snap};
use super::{fingerprint, CheckpointInput, CheckpointStats, Entry, StoredDep};
use crate::value_context::EncodeMismatch;
use crate::{Error, Result};

/// A clean query dep of a dirty entry: its key, and the fp of the bytes
/// it was last written or seeded with, which is what restore compares.
struct Clean {
    key: KeyOf,
    fp: u128,
}

impl<C: MemoCells> CoreRef<C> {
    pub(crate) fn checkpoint(&self, rt: &Runtime<C>) -> Result<CheckpointStats> {
        assert!(
            !rt.frame_active(),
            "incr-persist: Memo::checkpoint called from inside a query's compute"
        );
        check_value_code();
        // On a Shared store a scope another thread opens after this check is
        // not covered: the snapshot below may then see unlogged changes.
        if self.deps.unlogged() {
            return Err(Error::UnloggedScopeOpen);
        }
        let _checkpoint = self.lock_checkpoint();
        let started = Instant::now();
        let mut attempt = Attempt {
            core: self,
            started: false,
            saved: false,
        };
        let (snaps, mut keys, clean, moved) = {
            let mut state = self.lock_memo();
            state.seed_info.retain(|&handle, _| rt.is_alive(handle));
            let dirty = self.dirty.attempt(|handle| rt.is_alive(handle));
            attempt.started = true;
            // The read hold excludes every mapped write, and each finishes
            // its walk inside its own hold, so a Clean entry over mapped
            // deps equals its closure of the current mapped values.
            let map = self.deps.read();
            let mut snaps = Vec::new();
            for record in &dirty {
                let table = &self.tables[record.table as usize];
                snaps.extend(table.inspect_dirty(rt, &map, self.id, record.handle, record.mark));
            }
            // A node that landed again since its `inspect`, which only a
            // write around the store allows, may have dependents inspected
            // over its new value; dropping it drops them as unmapped.
            let (mut snaps, moved): (Vec<_>, Vec<_>) = snaps
                .into_iter()
                .partition(|snap| changed_at_erased(rt, snap.handle) == Some(snap.stamp));
            {
                let taken: HashSet<ErasedIncr> = snaps.iter().map(|snap| snap.handle).collect();
                for record in &dirty {
                    if !taken.contains(&record.handle) {
                        self.dirty.carry(*record);
                    }
                }
            }
            let dirty: HashSet<ErasedIncr> = dirty.iter().map(|record| record.handle).collect();
            for snap in &mut snaps {
                snap.seeded = state
                    .seed_info
                    .get(&snap.handle)
                    .filter(|&&(_, stamp)| stamp == snap.stamp)
                    .map(|&(fp, _)| fp);
            }
            let is_dirty = |handle: &ErasedIncr| dirty.contains(handle);
            // A clean entry has not run since it was written or seeded,
            // so while it keeps that stamp its value is the stored one.
            let mut clean: HashMap<ErasedIncr, u128> = HashMap::new();
            for snap in &snaps {
                for dep in &snap.deps {
                    let LiveDep::Query(handle) = *dep else {
                        continue;
                    };
                    if is_dirty(&handle) {
                        continue;
                    }
                    if let Some(&(fp, stamp)) = state.seed_info.get(&handle) {
                        if changed_at_erased(rt, handle) == Some(stamp) {
                            clean.insert(handle, fp);
                        }
                    }
                }
            }
            let mut wanted: HashSet<ErasedIncr> = snaps.iter().map(|snap| snap.handle).collect();
            wanted.extend(clean.keys());
            let mut keys = HashMap::new();
            if !wanted.is_empty() {
                for table in &self.tables {
                    table.keys_of(rt, &wanted, &mut keys);
                }
            }
            (snaps, keys, clean, moved)
        };
        drop(moved);
        let clean: HashMap<ErasedIncr, Clean> = clean
            .into_iter()
            .filter_map(|(handle, fp)| {
                Some((
                    handle,
                    Clean {
                        key: keys.remove(&handle)?,
                        fp,
                    },
                ))
            })
            .collect();
        let mut mismatches = Vec::new();
        let index = self.index();
        let stored = |table: u32, key: &[u8], fp: u128| {
            let entry = index.read(&*self.backend, table, key).ok()??;
            (fingerprint(&entry.value) == fp).then_some(entry.value)
        };
        let found = fresh(snaps, &mut keys, &clean, &stored, &mut mismatches);
        drop(index);
        drop((keys, clean));
        let snapshot = started.elapsed();
        if found.entries.is_empty() {
            attempt.saved = true;
            let mut state = self.lock_memo();
            for mismatch in mismatches {
                state.record_mismatch(mismatch);
            }
            state.stats.unsaveable = found.unsaveable;
            state.stats.evicted_last_checkpoint = 0;
            let mut stats = CheckpointStats {
                unchanged: true,
                ..CheckpointStats::default()
            };
            stats.times.snapshot = snapshot;
            return Ok(stats);
        }
        let done = self.backend.checkpoint(CheckpointInput {
            build: self.build,
            tables: self.catalog.clone(),
            fresh: found.entries,
            max_bytes: self.max_bytes,
        });
        let mut state = self.lock_memo();
        for mismatch in mismatches {
            state.record_mismatch(mismatch);
        }
        let mut done = done?;
        attempt.saved = true;
        done.stats.times.snapshot = snapshot;
        self.swap_index(IndexGen::new(done.generation, self.build, &self.catalog));
        state.stats.checkpoints += 1;
        let mut unsaveable = found.unsaveable;
        {
            let dirty = &self.dirty;
            for (fate, origin) in done.fates.iter().zip(&found.origin) {
                match fate {
                    Fate::Written => {
                        dirty.saved(origin.handle, origin.mark);
                        state
                            .seed_info
                            .insert(origin.handle, (origin.fp, origin.stamp));
                    }
                    // Nothing else rewrites a clean dep the base lacks.
                    Fate::Unkept => {
                        unsaveable += 1;
                        for &(table, handle) in &origin.clean {
                            dirty.lost(table, handle);
                        }
                    }
                    Fate::Oversized => unsaveable += 1,
                    Fate::Evicted => {}
                }
            }
        }
        let mut bag = Deferred::<C>::new();
        for (table, key) in &done.evicted_base {
            let Some(t) = self.tables.get(*table as usize) else {
                continue;
            };
            if let Ok(Some(handle)) = t.probe_bytes(rt, key, &mut bag) {
                self.dirty.lost(*table, handle);
            }
        }
        state.stats.unsaveable = unsaveable;
        state.stats.evicted_last_checkpoint = done.stats.evicted as u64;
        // Restores hold M, so none is still reading the old generation.
        // The new one is current either way, so a failed removal only
        // leaves a stale generation behind.
        if let Some(old) = done.replaced {
            if self.backend.remove_generation(old).is_err() {
                state.stats.stale_generations += 1;
            }
        }
        drop(state);
        bag.drain(rt);
        Ok(done.stats)
    }
}

/// Takes back a started dirty-set attempt unless the checkpoint saved,
/// on an error return and on a panic alike.
struct Attempt<'a, C: MemoCells> {
    core: &'a CoreRef<C>,
    started: bool,
    saved: bool,
}

impl<C: MemoCells> Drop for Attempt<'_, C> {
    fn drop(&mut self) {
        if self.started && !self.saved {
            self.core.dirty.failed();
        }
    }
}

/// What [`fresh`] hands the write step and keeps for after it.
struct Found {
    entries: Vec<Entry>,
    /// Per entry, in order.
    origin: Vec<Origin>,
    /// Snapshot entries left out: in a cycle group, a part that does not
    /// round-trip, or a dep that maps to nothing.
    unsaveable: u64,
}

/// The live entry behind a [`Found::entries`] entry.
struct Origin {
    handle: ErasedIncr,
    mark: u64,
    stamp: u64,
    fp: u128,
    /// Its query deps that were clean, by table and handle.
    clean: Vec<(u32, ErasedIncr)>,
}

/// The snapshot's entries that every keep rule but the query-dep fixpoint
/// allows: encoded and decoding back equal, not in a cycle group, and
/// every dep mapped. A query dep maps to another snapshot entry, whose
/// value's fingerprint it records, or to a clean entry, whose stored
/// fingerprint it records. An entry that recomputed to a value equal to
/// its stored one but encodes to other bytes keeps the bytes `stored`
/// finds at its seeded fp, which clean dependents in the base name.
fn fresh(
    snaps: Vec<Snap>,
    keys: &mut HashMap<ErasedIncr, KeyOf>,
    clean: &HashMap<ErasedIncr, Clean>,
    stored: &dyn Fn(u32, &[u8], u128) -> Option<Vec<u8>>,
    mismatches: &mut Vec<EncodeMismatch>,
) -> Found {
    let mut note = |r: Result<Vec<u8>, EncodeMismatch>| match r {
        Ok(bytes) => Some(bytes),
        Err(mismatch) => {
            mismatches.push(mismatch);
            None
        }
    };
    let encoded: Vec<Option<(Vec<u8>, Vec<u8>)>> = snaps
        .iter()
        .map(|snap| {
            let key = note(keys.remove(&snap.handle)?.key.encode())?;
            let value = note(snap.value.encode())?;
            let value = match snap.seeded {
                Some(fp) if fingerprint(&value) != fp => {
                    stored(snap.table, &key, fp).unwrap_or(value)
                }
                _ => value,
            };
            Some((key, value))
        })
        .collect();
    let mut by_handle: HashMap<ErasedIncr, (u32, Vec<u8>, u128)> = snaps
        .iter()
        .zip(&encoded)
        .filter_map(|(snap, bytes)| {
            let (key, value) = bytes.as_ref()?;
            Some((snap.handle, (snap.table, key.clone(), fingerprint(value))))
        })
        .collect();
    for (&handle, dep) in clean {
        if let Some(key) = note(dep.key.key.encode()) {
            by_handle.insert(handle, (dep.key.table, key, dep.fp));
        }
    }
    let mut found = Found {
        entries: Vec::new(),
        origin: Vec::new(),
        unsaveable: 0,
    };
    for (snap, bytes) in snaps.iter().zip(encoded) {
        let Some((key, value)) = bytes.filter(|_| !snap.in_group) else {
            found.unsaveable += 1;
            continue;
        };
        let deps: Option<Vec<StoredDep>> = snap
            .deps
            .iter()
            .map(|dep| match dep {
                LiveDep::Stored(stored) => Some(stored.clone()),
                LiveDep::Query(handle) => {
                    let (table, key, fp) = by_handle.get(handle)?;
                    Some(StoredDep::Query {
                        table: *table,
                        key: key.clone(),
                        fp: *fp,
                    })
                }
                LiveDep::Unmapped => None,
            })
            .collect();
        let Some(deps) = deps else {
            found.unsaveable += 1;
            continue;
        };
        found.origin.push(Origin {
            handle: snap.handle,
            mark: snap.mark,
            stamp: snap.stamp,
            fp: fingerprint(&value),
            clean: snap
                .deps
                .iter()
                .filter_map(|dep| match dep {
                    LiveDep::Query(handle) => Some((clean.get(handle)?.key.table, *handle)),
                    _ => None,
                })
                .collect(),
        });
        found.entries.push(Entry {
            table: snap.table,
            key,
            value,
            deps,
            last_used: 0,
        });
    }
    found
}
