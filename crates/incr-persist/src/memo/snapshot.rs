//! `Memo::checkpoint`: snapshot the dirty live entries under one
//! `DepIndex` read hold, encode them and write the next generation, merged
//! with the current one, with no incr lock held, then swap the index to it
//! and clean what reached the backend.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use incr_core::persist_hooks::{changed_at_erased, Deferred};
use incr_core::{in_value_code, ErasedIncr, Runtime, Yielder};

use super::checkpoint::Fate;
use super::dirty::{Carried, Record};
use super::mapping::LiveDep;
use super::memo_core::{CoreRef, IndexGen, Landing, MemoCells};
use super::table_core::Snap;
use super::{fingerprint, CheckpointInput, CheckpointStats, Entry, StoredDep, VALUE_BATCH};
use crate::value_context::{hold, EncodeMismatch, ValueScope};
use crate::{Error, Result};

#[cfg(test)]
thread_local! {
    /// Keys a checkpoint on this thread read from live entries or stored.
    pub(crate) static KEYS_TOUCHED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn touched_key() {
    KEYS_TOUCHED.with(|n| n.set(n.get() + 1));
}

/// Where a checkpoint calls [`CHECKPOINT_PARK`] on its thread.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CheckpointPark {
    /// One dirty entry is inspected under the `DepIndex` read hold; its
    /// deps, row fingerprints included, are not mapped yet.
    Inspected,
    /// Every snapshot entry and its deps' fingerprints are taken and the
    /// hold is released; nothing is encoded yet.
    Snapshotted,
}

#[cfg(test)]
type CheckpointParkHook = Box<dyn Fn(CheckpointPark)>;

#[cfg(test)]
thread_local! {
    pub(crate) static CHECKPOINT_PARK: std::cell::RefCell<Option<CheckpointParkHook>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn park(point: CheckpointPark) {
    CHECKPOINT_PARK.with(|hook| {
        if let Some(hook) = &*hook.borrow() {
            hook(point);
        }
    });
}

/// A clean query dep of a dirty entry: its table and stored key, and the
/// fp of the bytes it was last written or seeded with, which is what
/// restore compares.
struct Clean {
    table: u32,
    key: Vec<u8>,
    fp: u128,
}

impl<C: MemoCells> CoreRef<C> {
    pub(crate) fn checkpoint(&self, rt: &Runtime<C>) -> Result<CheckpointStats> {
        assert!(
            !rt.frame_active(),
            "incr-persist: Memo::checkpoint called from inside a query's compute"
        );
        if in_value_code() {
            return Err(Error::Reentrant);
        }
        // On a Shared store a scope another thread opens after this check is
        // not covered: the snapshot below may then see unlogged changes.
        if self.deps.poisoned() {
            return Err(Error::Poisoned);
        }
        if self.deps.unlogged() {
            return Err(Error::UnloggedScopeOpen);
        }
        if self.deps.staged() {
            return Err(Error::StagedBatchOpen);
        }
        let _checkpoint = self.lock_checkpoint();
        let started = Instant::now();
        let mut attempt = Attempt {
            core: self,
            started: false,
            saved: false,
        };
        let mut bag = Deferred::<C>::new();
        let mut dead = Vec::new();
        let (snaps, clean, moved) = {
            let mut state = self.lock_memo();
            let yielder = rt.yielder_under_lock();
            // Dead nodes never match a live handle, so pruning them can
            // wait until the map has doubled: linear time per prune,
            // amortized over the inserts since the last one.
            if state.seed_info.len() >= 2 * state.pruned_at.max(1024) {
                state.seed_info.retain(|&handle, _| {
                    yielder.tick();
                    rt.is_alive(handle)
                });
                state.seed_keys.retain(|&handle, _| {
                    yielder.tick();
                    rt.is_alive(handle)
                });
                state.pruned_at = state.seed_info.len();
            }
            let dirty = self.dirty.attempt(
                |handle| {
                    yielder.tick();
                    rt.is_alive(handle)
                },
                &mut dead,
            );
            attempt.started = true;
            // The read hold excludes every mapped write, and each finishes
            // its walk inside its own hold, so a Clean entry over mapped
            // deps equals its closure of the current mapped values.
            let map = self.deps.read();
            let mut snaps = Vec::new();
            let mut by_table: Vec<&Record> = dirty.iter().collect();
            by_table.sort_by_key(|record| record.table);
            for same in by_table.chunk_by(|a, b| a.table == b.table) {
                let table = &self.tables[same[0].table as usize];
                for batch in same.chunks(VALUE_BATCH) {
                    yielder.tick_by(batch.len());
                    table.inspect_dirty(rt, &map, self.id, batch, &mut snaps, &mut bag);
                }
            }
            // A node that landed again since its `inspect`, which only a
            // write around the store allows, may have dependents inspected
            // over its new value; dropping it drops them as unmapped.
            let (mut snaps, moved): (Vec<_>, Vec<_>) = snaps
                .into_iter()
                .inspect(|_| yielder.tick())
                .partition(|snap| changed_at_erased(rt, snap.handle) == Some(snap.stamp));
            {
                let taken: HashSet<ErasedIncr> = snaps
                    .iter()
                    .inspect(|_| yielder.tick())
                    .map(|snap| snap.handle)
                    .collect();
                let mut landing = Landing::default();
                for record in &dirty {
                    yielder.tick();
                    if taken.contains(&record.handle) {
                        continue;
                    }
                    match self.dirty.carry(record) {
                        Carried::No => {}
                        Carried::Counted => landing.push(record.handle, true),
                        Carried::Before => landing.push(record.handle, false),
                    }
                }
                self.wake.await_landing(landing);
            }
            let dirty: HashSet<ErasedIncr> = dirty
                .iter()
                .inspect(|_| yielder.tick())
                .map(|record| record.handle)
                .collect();
            for snap in &mut snaps {
                yielder.tick();
                snap.seeded = state
                    .seed_info
                    .get(&snap.handle)
                    .filter(|&&(_, stamp)| stamp == snap.stamp)
                    .map(|&(fp, _)| fp);
            }
            let is_dirty = |handle: &ErasedIncr| dirty.contains(handle);
            // A clean entry has not run since it was written or seeded,
            // so while it keeps that stamp its value is the stored one.
            let mut clean: HashMap<ErasedIncr, Clean> = HashMap::new();
            for snap in &snaps {
                for dep in &snap.deps {
                    yielder.tick();
                    let LiveDep::Query(handle) = *dep else {
                        continue;
                    };
                    if is_dirty(&handle) {
                        continue;
                    }
                    if clean.contains_key(&handle) {
                        continue;
                    }
                    let Some(&(fp, stamp)) = state.seed_info.get(&handle) else {
                        continue;
                    };
                    let Some((table, key)) = state.seed_keys.get(&handle) else {
                        continue;
                    };
                    if changed_at_erased(rt, handle) == Some(stamp) {
                        #[cfg(test)]
                        touched_key();
                        let (table, key) = (*table, key.clone());
                        clean.insert(handle, Clean { table, key, fp });
                    }
                }
            }
            (snaps, clean, moved)
        };
        drop(moved);
        drop(dead);
        bag.drain(rt);
        #[cfg(test)]
        park(CheckpointPark::Snapshotted);
        let mut mismatches = Vec::new();
        let index = self.index();
        let stored = |table: u32, key: &[u8], fp: u128| {
            let entry = index.read(&*self.backend, table, key).ok()??;
            (fingerprint(&entry.value) == fp).then_some(entry.value)
        };
        let found = fresh(
            snaps,
            &clean,
            self.scope.as_ref(),
            &stored,
            &mut mismatches,
            &rt.yielder_under_lock(),
        );
        drop(index);
        let yielder = rt.yielder_under_lock();
        for dep in clean {
            yielder.tick();
            drop(dep);
        }
        let snapshot = started.elapsed();
        if found.entries.is_empty() {
            attempt.saved = true;
            let mut state = self.lock_memo();
            for mismatch in mismatches {
                state.record_mismatch(mismatch);
            }
            state.stats.unsaveable = found.unsaveable;
            state.stats.evicted_last_checkpoint = 0;
            state.stats.last_bytes_appended = 0;
            let mut stats = CheckpointStats {
                unchanged: true,
                ..CheckpointStats::default()
            };
            stats.times.snapshot = snapshot;
            return Ok(stats);
        }
        let known = self.index().layers.clone();
        let done = rt.with_ambient_under_lock(|| {
            self.backend.checkpoint(
                CheckpointInput {
                    build: self.build,
                    tables: self.catalog.clone(),
                    fresh: found.entries,
                    max_bytes: self.max_bytes,
                },
                &known,
            )
        });
        drop(known);
        let mut state = self.lock_memo();
        for mismatch in mismatches {
            state.record_mismatch(mismatch);
        }
        let mut done = done?;
        attempt.saved = true;
        done.stats.times.snapshot = snapshot;
        self.swap_index(IndexGen::new(done.chain, self.build, &self.catalog));

        state.stats.checkpoints += 1;
        state.stats.delta_checkpoints += u64::from(done.stats.delta);
        state.stats.compactions += u64::from(done.stats.compacted);
        state.stats.last_bytes_appended = done.stats.bytes_appended;
        let mut unsaveable = found.unsaveable;
        let mut saved = Vec::new();
        {
            let dirty = &self.dirty;
            let yielder = rt.yielder_under_lock();
            state.seed_info.reserve(done.fates.len());
            state.seed_keys.reserve(done.fates.len());
            for (fate, origin) in done.fates.iter().zip(&found.origin) {
                yielder.tick();
                match fate {
                    Fate::Written => {
                        saved.extend(dirty.saved(origin.handle, origin.mark));
                        state
                            .seed_info
                            .insert(origin.handle, (origin.fp, origin.stamp));
                        state
                            .seed_keys
                            .insert(origin.handle, (origin.table, origin.key.clone()));
                    }
                    // Nothing else rewrites a clean dep the base lacks.
                    Fate::Unkept => {
                        unsaveable += 1;
                        for (table, handle, key) in &origin.clean {
                            self.lost(*table, *handle, key.clone());
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
                self.dirty.evicted(*table, handle, key.clone());
            }
        }
        state.stats.unsaveable = unsaveable;
        state.stats.evicted_last_checkpoint = done.stats.evicted as u64;
        drop(state);
        drop(saved);
        bag.drain(rt);
        // After the memo lock is released: the new chain is current, and a
        // reader still on the old one gets `None` from `read_entry`. A
        // failed removal only leaves stale generations, which the next
        // checkpoint removes.
        let stale = self.backend.remove_generations(&done.superseded);
        if !stale.is_empty() {
            self.lock_memo().stats.stale_generations += stale.len() as u64;
        }
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

/// The base's value bytes for a live table and key, if they have the
/// given fp.
type StoredAt<'a> = dyn Fn(u32, &[u8], u128) -> Option<Vec<u8>> + 'a;

/// The live entry behind a [`Found::entries`] entry.
struct Origin {
    table: u32,
    key: Vec<u8>,
    handle: ErasedIncr,
    mark: u64,
    stamp: u64,
    fp: u128,
    /// Its query deps that were clean, by table, handle and stored key.
    clean: Vec<(u32, ErasedIncr, Vec<u8>)>,
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
    clean: &HashMap<ErasedIncr, Clean>,
    scope: Option<&ValueScope>,
    stored: &StoredAt<'_>,
    mismatches: &mut Vec<EncodeMismatch>,
    yielder: &Yielder<'_>,
) -> Found {
    let mut note = |r: Result<Vec<u8>, EncodeMismatch>| match r {
        Ok(bytes) => Some(bytes),
        Err(mismatch) => {
            mismatches.push(mismatch);
            None
        }
    };
    let mut encoded: Vec<Option<(Vec<u8>, Vec<u8>)>> = Vec::with_capacity(snaps.len());
    for batch in snaps.chunks(VALUE_BATCH) {
        yielder.tick_by(batch.len());
        hold(scope, || {
            encoded.extend(batch.iter().map(|snap| {
                #[cfg(test)]
                touched_key();
                let key = note(snap.key.encode())?;
                let value = note(snap.value.encode())?;
                Some((key, value))
            }));
        });
    }
    // Outside the holds: `stored` reads the backend, which takes its lock.
    for (snap, bytes) in snaps.iter().zip(&mut encoded) {
        yielder.tick();
        let (Some(fp), Some((key, value))) = (snap.seeded, bytes) else {
            continue;
        };
        if fingerprint(value) != fp {
            if let Some(stored) = stored(snap.table, key, fp) {
                *value = stored;
            }
        }
    }
    let mut by_handle: HashMap<ErasedIncr, (u32, Vec<u8>, u128)> = snaps
        .iter()
        .zip(&encoded)
        .inspect(|_| yielder.tick())
        .filter_map(|(snap, bytes)| {
            let (key, value) = bytes.as_ref()?;
            Some((snap.handle, (snap.table, key.clone(), fingerprint(value))))
        })
        .collect();
    for (&handle, dep) in clean {
        yielder.tick();
        by_handle.insert(handle, (dep.table, dep.key.clone(), dep.fp));
    }
    let mut found = Found {
        entries: Vec::new(),
        origin: Vec::new(),
        unsaveable: 0,
    };
    for (snap, bytes) in snaps.into_iter().zip(encoded) {
        yielder.tick();
        let Some((key, value)) = bytes.filter(|_| !snap.in_group) else {
            found.unsaveable += 1;
            continue;
        };
        let deps: Option<Vec<StoredDep>> = snap
            .deps
            .iter()
            .inspect(|_| yielder.tick())
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
            table: snap.table,
            key: key.clone(),
            handle: snap.handle,
            mark: snap.mark,
            stamp: snap.stamp,
            fp: fingerprint(&value),
            clean: snap
                .deps
                .iter()
                .filter_map(|dep| match dep {
                    LiveDep::Query(handle) => {
                        let dep = clean.get(handle)?;
                        Some((dep.table, *handle, dep.key.clone()))
                    }
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

#[cfg(test)]
#[path = "snapshot_tests.rs"]
mod tests;
