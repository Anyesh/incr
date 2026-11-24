//! `Memo::checkpoint`: snapshot the dirty live entries under one
//! `DepIndex` read hold, encode them and write the next generation, merged
//! with the current one, with no incr lock held, then swap the index to it
//! and clean what reached the backend.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use incr_core::persist_hooks::Deferred;
use incr_core::{in_value_code, ErasedIncr, Runtime, Yielder};

use super::checkpoint::Fate;
use super::dirty::{Carried, Record};
use super::mapping::LiveDep;
use super::memo_core::{CoreRef, ErasedBackend, IndexGen, Landing, MemoCells, MemoState};
use super::table_core::{KeyPart, Snap};
use super::{fingerprint, CheckpointInput, CheckpointStats, Entry, GenId, StoredDep, VALUE_BATCH};
use crate::flight::Parcel;
use crate::value_context::EncodeMismatch;
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

/// The seed of a query dep: the fp of the bytes its node was last written
/// or seeded with, the `changed_at` it holds that value at, and its live
/// table and stored key. Copied under the memo lock, so whoever resolves
/// deps needs no lock.
#[derive(Clone)]
pub(crate) struct Seed {
    pub(crate) fp: u128,
    pub(crate) stamp: u64,
    pub(crate) table: u32,
    pub(crate) key: Vec<u8>,
}

pub(crate) type Seeds = HashMap<ErasedIncr, Seed>;

/// Sets each of `snaps`' own `seeded` and returns the seed of each query
/// dep not in `copied` yet whose seed stamp is the stamp the dep was
/// inspected at, adding every dep it looked at to `copied`. A seed whose
/// stamp a dep has left vouches for no later stamp, since stamps only rise.
pub(crate) fn copy_seeds(
    state: &MemoState,
    snaps: &mut [Snap],
    copied: &mut HashSet<ErasedIncr>,
    yielder: &Yielder<'_>,
) -> Vec<(ErasedIncr, Seed)> {
    let mut seeds = Vec::new();
    for snap in snaps {
        yielder.tick();
        snap.seeded = state
            .seed_info
            .get(&snap.handle)
            .filter(|&&(_, stamp)| stamp == snap.stamp)
            .map(|&(fp, _)| fp);
        for dep in &snap.deps {
            yielder.tick();
            let LiveDep::Query { handle, stamp } = *dep else {
                continue;
            };
            if !copied.insert(handle) {
                continue;
            }
            let Some(&(fp, seeded_at)) = state.seed_info.get(&handle) else {
                continue;
            };
            let Some((table, key)) = state.seed_keys.get(&handle) else {
                continue;
            };
            if stamp == Some(seeded_at) {
                #[cfg(test)]
                touched_key();
                seeds.push((
                    handle,
                    Seed {
                        fp,
                        stamp: seeded_at,
                        table: *table,
                        key: key.clone(),
                    },
                ));
            }
        }
    }
    seeds
}

impl<C: MemoCells> CoreRef<C> {
    pub(crate) fn checkpoint(&self, rt: &Runtime<C>) -> Result<CheckpointStats> {
        self.check_checkpoint(rt)?;
        let mut slot = self.lock_checkpoint();
        if !self.settle_idle(&mut slot) {
            return Err(Error::Busy);
        }
        let started = Instant::now();
        let mut attempt = Attempt {
            core: self,
            started: false,
            saved: false,
        };
        let mut bag = Deferred::<C>::new();
        let mut dead = Vec::new();
        let (snaps, seeds) = {
            let mut state = self.lock_memo();
            let yielder = rt.yielder_under_lock();
            // Dead nodes never match a live handle, so pruning them can
            // wait until the map has doubled: linear time per prune,
            // amortized over the inserts since the last one.
            if state.prune_due() {
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
            let by_table: Vec<&Record> = dirty.iter().collect();
            for same in by_table.chunk_by(|a, b| a.table == b.table) {
                let table = &self.tables[same[0].table as usize];
                for batch in same.chunks(VALUE_BATCH) {
                    yielder.tick_by(batch.len());
                    table.inspect_dirty(rt, &map, self.id, batch, &mut snaps, &mut bag);
                }
            }
            let mut landing = Landing::default();
            self.carry_uninspected(&dirty, &snaps, &yielder, &mut landing);
            self.wake.await_landing(landing);
            let seeds: Seeds = copy_seeds(&state, &mut snaps, &mut HashSet::new(), &yielder)
                .into_iter()
                .collect();
            (snaps, seeds)
        };
        drop(dead);
        bag.drain(rt);
        #[cfg(test)]
        park(CheckpointPark::Snapshotted);
        let mut mismatches = Vec::new();
        let index = self.index();
        let backend = &*self.backend;
        let stored =
            |table: u32, key: &[u8], fp: u128| stored_bytes(&index, backend, table, key, fp);
        let found = fresh(
            Parcel::new(snaps, self.scope.clone()),
            &seeds,
            &stored,
            &mut mismatches,
            &rt.yielder_under_lock(),
        );
        drop(index);
        let yielder = rt.yielder_under_lock();
        for seed in seeds {
            yielder.tick();
            drop(seed);
        }
        let snapshot = started.elapsed();
        if found.entries.is_empty() {
            attempt.saved = true;
            return Ok(self.unchanged(mismatches, found.unsaveable, snapshot));
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
                None,
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
        state.count_written(&done.stats);
        let mut unsaveable = found.unsaveable;
        let mut saved = Vec::new();
        {
            let yielder = rt.yielder_under_lock();
            state.seed_info.reserve(done.fates.len());
            state.seed_keys.reserve(done.fates.len());
            for (fate, origin) in done.fates.iter().zip(&found.origin) {
                yielder.tick();
                unsaveable += u64::from(self.finish_entry(&mut state, *fate, origin, &mut saved));
            }
        }
        let mut bag = Deferred::<C>::new();
        for (table, key) in &done.evicted_base {
            self.probe_evicted(rt, *table, key, &mut bag);
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
        self.remove_superseded(&done.superseded);
        Ok(done.stats)
    }

    /// The refusals every checkpoint entry point shares.
    pub(crate) fn check_checkpoint(&self, rt: &Runtime<C>) -> Result<()> {
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
        Ok(())
    }

    /// Counts again each of `dirty` that `inspect` did not return Clean,
    /// adding to `landing` those whose landing no compute will report.
    pub(crate) fn carry_uninspected(
        &self,
        dirty: &[Record],
        snaps: &[Snap],
        yielder: &Yielder<'_>,
        landing: &mut Landing,
    ) {
        let taken: HashSet<ErasedIncr> = snaps
            .iter()
            .inspect(|_| yielder.tick())
            .map(|snap| snap.handle)
            .collect();
        for record in dirty {
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
    }

    /// The stats of a checkpoint that found nothing to write.
    pub(crate) fn unchanged(
        &self,
        mismatches: Vec<EncodeMismatch>,
        unsaveable: u64,
        snapshot: Duration,
    ) -> CheckpointStats {
        let mut state = self.lock_memo();
        for mismatch in mismatches {
            state.record_mismatch(mismatch);
        }
        state.stats.unsaveable = unsaveable;
        state.stats.evicted_last_checkpoint = 0;
        state.stats.last_bytes_appended = 0;
        let mut stats = CheckpointStats {
            unchanged: true,
            ..CheckpointStats::default()
        };
        stats.times.snapshot = snapshot;
        stats
    }

    /// Applies what the write did with one fresh entry; returns whether
    /// the entry is unsaveable. A written entry's key goes to `saved`, to
    /// drop outside the memo lock.
    pub(crate) fn finish_entry(
        &self,
        state: &mut MemoState,
        fate: Fate,
        origin: &Origin,
        saved: &mut Vec<KeyPart>,
    ) -> bool {
        match fate {
            Fate::Written => {
                saved.extend(self.dirty.saved(origin.handle, origin.mark));
                state
                    .seed_info
                    .insert(origin.handle, (origin.fp, origin.stamp));
                state
                    .seed_keys
                    .insert(origin.handle, (origin.table, origin.key.clone()));
                false
            }
            // Nothing else rewrites a clean dep the base lacks.
            Fate::Unkept => {
                for (table, handle, key) in &origin.clean {
                    self.lost(*table, *handle, key.clone());
                }
                true
            }
            Fate::Oversized => true,
            Fate::Evicted => false,
        }
    }

    /// Keeps dirty the live entry, if any, of a base entry eviction dropped.
    pub(crate) fn probe_evicted(
        &self,
        rt: &Runtime<C>,
        table: u32,
        key: &[u8],
        bag: &mut Deferred<C>,
    ) {
        let Some(t) = self.tables.get(table as usize) else {
            return;
        };
        if let Ok(Some(handle)) = t.probe_bytes(rt, key, bag) {
            self.dirty.evicted(table, handle, key.to_vec());
        }
    }

    fn remove_superseded(&self, superseded: &[GenId]) {
        let stale = self.backend.remove_generations(superseded);
        self.stale_generations
            .fetch_add(stale.len() as u64, Ordering::Relaxed);
    }
}

/// The base's value bytes for live `table` and `key` in `index`, if they
/// have fp `fp`.
pub(crate) fn stored_bytes(
    index: &IndexGen,
    backend: &dyn ErasedBackend,
    table: u32,
    key: &[u8],
    fp: u128,
) -> Option<Vec<u8>> {
    let entry = index.read(backend, table, key).ok()??;
    (fingerprint(&entry.value) == fp).then_some(entry.value)
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
pub(crate) struct Found {
    pub(crate) entries: Vec<Entry>,
    /// Per entry, in order.
    pub(crate) origin: Vec<Origin>,
    /// Snapshot entries left out: in a cycle group, a part that does not
    /// round-trip, or a dep that maps to nothing.
    pub(crate) unsaveable: u64,
}

/// The base's value bytes for a live table and key, if they have the
/// given fp.
pub(crate) type StoredAt<'a> = dyn Fn(u32, &[u8], u128) -> Option<Vec<u8>> + 'a;

/// The live entry behind a [`Found::entries`] entry.
pub(crate) struct Origin {
    table: u32,
    key: Vec<u8>,
    handle: ErasedIncr,
    mark: u64,
    stamp: u64,
    fp: u128,
    /// Its query deps that mapped to their seeds, by table, handle and
    /// stored key.
    clean: Vec<(u32, ErasedIncr, Vec<u8>)>,
}

/// An entry's encoded key and value.
type KeyValue = (Vec<u8>, Vec<u8>);

/// A snap without its key and value, once they are encoded.
struct Taken {
    table: u32,
    handle: ErasedIncr,
    mark: u64,
    stamp: u64,
    seeded: Option<u128>,
    deps: Vec<LiveDep>,
    in_group: bool,
}

/// The snapshot's entries that every keep rule but the query-dep fixpoint
/// allows: encoded and decoding back equal, not in a cycle group, and
/// every dep mapped. A query dep maps to the snapshot entry of its node if
/// that entry was inspected at the stamp the dep was read at, and else to
/// its seed if the seed holds at that stamp; it records that value's
/// fingerprint. An entry that recomputed to a value equal to its stored
/// one but encodes to other bytes keeps the bytes `stored` finds at its
/// seeded fp, which clean dependents in the base name. Each snap's key and
/// value end inside the hold that encodes them.
pub(crate) fn fresh(
    mut snaps: Parcel<Snap>,
    seeds: &Seeds,
    stored: &StoredAt<'_>,
    mismatches: &mut Vec<EncodeMismatch>,
    yielder: &Yielder<'_>,
) -> Found {
    let mut encoded: Vec<(Taken, Option<KeyValue>)> = Vec::with_capacity(snaps.len());
    while !snaps.is_empty() {
        yielder.tick_by(snaps.len().min(VALUE_BATCH));
        let batch = snaps.take_batch(|snap| {
            #[cfg(test)]
            touched_key();
            let bytes = snap
                .key
                .encode()
                .and_then(|key| Ok((key, snap.value.encode()?)));
            let taken = Taken {
                table: snap.table,
                handle: snap.handle,
                mark: snap.mark,
                stamp: snap.stamp,
                seeded: snap.seeded,
                deps: snap.deps,
                in_group: snap.in_group,
            };
            (taken, bytes)
        });
        for (taken, bytes) in batch {
            let bytes = match bytes {
                Ok(bytes) => Some(bytes),
                Err(mismatch) => {
                    mismatches.push(mismatch);
                    None
                }
            };
            encoded.push((taken, bytes));
        }
    }
    // Outside the holds: `stored` reads the backend, which takes its lock.
    for (snap, bytes) in &mut encoded {
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
    let by_handle: HashMap<ErasedIncr, (u32, Vec<u8>, u128, u64)> = encoded
        .iter()
        .inspect(|_| yielder.tick())
        .filter_map(|(snap, bytes)| {
            let (key, value) = bytes.as_ref()?;
            Some((
                snap.handle,
                (snap.table, key.clone(), fingerprint(value), snap.stamp),
            ))
        })
        .collect();
    let resolve = |handle: &ErasedIncr, stamp: Option<u64>| -> Option<(u32, &[u8], u128, bool)> {
        let stamp = stamp?;
        if let Some((table, key, fp, _)) = by_handle.get(handle).filter(|dep| dep.3 == stamp) {
            return Some((*table, key, *fp, false));
        }
        let seed = seeds.get(handle).filter(|seed| seed.stamp == stamp)?;
        Some((seed.table, &seed.key, seed.fp, true))
    };
    let mut found = Found {
        entries: Vec::new(),
        origin: Vec::new(),
        unsaveable: 0,
    };
    for (snap, bytes) in encoded {
        yielder.tick();
        let Some((key, value)) = bytes.filter(|_| !snap.in_group) else {
            found.unsaveable += 1;
            continue;
        };
        let mut clean = Vec::new();
        let deps: Option<Vec<StoredDep>> = snap
            .deps
            .iter()
            .inspect(|_| yielder.tick())
            .map(|dep| match dep {
                LiveDep::Stored(stored) => Some(stored.clone()),
                LiveDep::Query { handle, stamp } => {
                    let (table, key, fp, seeded) = resolve(handle, *stamp)?;
                    if seeded {
                        clean.push((table, *handle, key.to_vec()));
                    }
                    Some(StoredDep::Query {
                        table,
                        key: key.to_vec(),
                        fp,
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
            clean,
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
