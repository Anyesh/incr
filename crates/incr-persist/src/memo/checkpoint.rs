use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use super::{
    decode_entry, fingerprint, key_hash, EncodedEntry, Entry, GenHeader, GenId, Generation,
    IndexItem, MemoBackend, StoredDep, TableDesc,
};
use crate::{Error, Result};

/// The `max_bytes` a memo uses unless configured otherwise: 512 MiB.
pub const DEFAULT_MAX_BYTES: u64 = 512 << 20;

/// What a checkpoint's snapshot hands to [`write_checkpoint`].
#[derive(Clone, Debug)]
pub struct CheckpointInput {
    /// The live build id's fingerprint.
    pub build: u128,
    /// The live catalog. `fresh` entries and their query deps index into it.
    pub tables: Vec<TableDesc>,
    /// The live entries the snapshot found, already filtered by every keep
    /// rule except the query-dep fixpoint: all deps mapped, not in a cycle
    /// group, and `decode(encode(v)) == v`. At most one per `(table, key)`;
    /// their `last_used` is overwritten.
    pub fresh: Vec<Entry>,
    /// The bound on the new generation's summed entry frame lengths.
    pub max_bytes: u64,
}

/// What [`write_checkpoint`] did, by entry count.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckpointStats {
    /// Entries in the new generation.
    pub written: usize,
    /// Fresh entries dropped because a query dep, transitively, was not a
    /// kept fresh entry.
    pub unkept: usize,
    /// Base entries carried into the new generation before eviction.
    pub merged: usize,
    /// Base frames that failed to decode or did not match their index item.
    pub corrupt: usize,
    /// Entries dropped to fit `max_bytes`.
    pub evicted: usize,
    /// Entries dropped because their frame would exceed the frame length
    /// limit, each with every entry that reads it transitively.
    pub oversized: usize,
    pub times: CheckpointTimes,
}

/// Wall time of a checkpoint's steps. [`write_checkpoint`] alone leaves
/// `snapshot` zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CheckpointTimes {
    /// Finding and encoding the live entries to write, under the memo's
    /// locks for the part that reads the runtime.
    pub snapshot: Duration,
    /// Loading the base generation, reading and decoding every base frame
    /// and keeping fresh entries to the query-dep fixpoint.
    pub merge: Duration,
    /// Framing every entry, evicting, and the backend's durable write.
    pub write: Duration,
}

/// What became of one [`CheckpointInput::fresh`] entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fate {
    Written,
    Unkept,
    Oversized,
    Evicted,
}

#[derive(Debug)]
pub struct Checkpointed {
    pub generation: Generation,
    /// The generation this one replaced. The caller removes it with
    /// [`MemoBackend::remove_generation`] once nothing in this process
    /// reads it any more.
    pub replaced: Option<GenId>,
    pub stats: CheckpointStats,
    /// Per `fresh` entry, in input order.
    pub(crate) fates: Vec<Fate>,
    /// The `(table, key)` of every base entry eviction dropped, tables in
    /// the input catalog's numbering extended by carried-over tables.
    pub(crate) evicted_base: Vec<(u32, Vec<u8>)>,
}

/// Runs a checkpoint's write step against `backend`: takes the checkpoint
/// lock, loads the current generation as the base, keeps the fresh entries
/// whose query deps are kept fresh entries or base entries at the same fp
/// (to a fixpoint),
/// stamps them with the new sequence number, merges the base entries that
/// were not rewritten, drops
/// the entries too large for a frame and evicts by oldest `last_used`, each
/// with transitive dependents, down to `max_bytes`, and writes the result as
/// the next generation.
///
/// The base is merged only when its build matches and its seq is below
/// `u64::MAX`; otherwise numbering restarts at seq 1. A base table whose name
/// matches a live table with a different descriptor is dropped; a base
/// table with no live namesake is carried over, appended to the catalog.
/// Fails with [`Error::Busy`](crate::Error::Busy) when another handle holds
/// the lock or replaced the base meanwhile, and with
/// [`Error::Corrupt`](crate::Error::Corrupt) when the base's id is
/// `u64::MAX`; nothing is written then.
pub fn write_checkpoint<B: MemoBackend + ?Sized>(
    backend: &B,
    input: CheckpointInput,
) -> Result<Checkpointed> {
    write_with(backend, input, EncodedEntry::new)
}

fn write_with<B: MemoBackend + ?Sized>(
    backend: &B,
    input: CheckpointInput,
    encode: impl Fn(&Entry, u64) -> Result<EncodedEntry>,
) -> Result<Checkpointed> {
    let _guard = backend.lock_checkpoint()?;
    let started = Instant::now();
    let base = backend.load()?;
    let id = base
        .as_ref()
        .map_or(Some(1), |g| g.id().0.checked_add(1))
        .map(GenId)
        .ok_or(Error::Corrupt(
            "the current memo generation has the largest id",
        ))?;
    let next_seq = base
        .as_ref()
        .map_or(Some(1), |g| g.header().seq.checked_add(1));
    let seq = next_seq.unwrap_or(1);
    let mut stats = CheckpointStats::default();

    let mut tables = input.tables;
    let mergeable = |g: &&Generation| next_seq.is_some() && g.header().build == input.build;
    let based = match base.as_ref().filter(mergeable) {
        Some(base) => read_base(backend, base, &mut tables, &mut stats)?,
        None => Vec::new(),
    };
    let unkept = unkept(&input.fresh, &Held::new(&based));
    let mut fates = vec![Fate::Written; input.fresh.len()];
    let mut entries: Vec<Entry> = Vec::with_capacity(input.fresh.len() + based.len());
    let mut origin: Vec<Option<usize>> = Vec::with_capacity(entries.capacity());
    for (i, (mut entry, drop)) in input.fresh.into_iter().zip(unkept).enumerate() {
        if drop {
            stats.unkept += 1;
            fates[i] = Fate::Unkept;
        } else {
            entry.last_used = seq;
            entries.push(entry);
            origin.push(Some(i));
        }
    }
    let mut present: HashSet<(u32, Vec<u8>)> =
        entries.iter().map(|e| (e.table, e.key.clone())).collect();
    for entry in based {
        if present.insert((entry.table, entry.key.clone())) {
            stats.merged += 1;
            entries.push(entry);
            origin.push(None);
        }
    }
    drop(present);
    let merged_at = Instant::now();
    stats.times.merge = merged_at - started;

    let header = GenHeader {
        build: input.build,
        seq,
        tables,
    };
    let tag = header.frame_tag();
    let encode = |entry: &Entry| encode(entry, tag);
    let frames = encode_fitting(&entries, encode, &mut stats.oversized)?;
    let mut fitting = Vec::with_capacity(entries.len());
    let mut fitting_origin = Vec::with_capacity(entries.len());
    let mut encoded = Vec::with_capacity(entries.len());
    for ((entry, from), frame) in entries.into_iter().zip(origin).zip(frames) {
        match frame {
            Some(frame) => {
                fitting.push(entry);
                fitting_origin.push(from);
                encoded.push(frame);
            }
            None => {
                if let Some(i) = from {
                    fates[i] = Fate::Oversized;
                }
            }
        }
    }
    let sizes: Vec<u64> = encoded.iter().map(|e| e.frame.len() as u64).collect();
    let evicted = evict(&fitting, &sizes, input.max_bytes);
    let mut evicted_base = Vec::new();
    let mut kept = Vec::with_capacity(encoded.len());
    for (((entry, from), frame), gone) in fitting
        .into_iter()
        .zip(fitting_origin)
        .zip(encoded)
        .zip(evicted)
    {
        if !gone {
            kept.push(frame);
            continue;
        }
        stats.evicted += 1;
        match from {
            Some(i) => fates[i] = Fate::Evicted,
            None => evicted_base.push((entry.table, entry.key)),
        }
    }
    stats.written = kept.len();
    let base_id = base.as_ref().map(Generation::id);
    let generation = backend.write_generation(base_id, id, &header, &kept)?;
    stats.times.write = merged_at.elapsed();
    Ok(Checkpointed {
        generation,
        replaced: base_id,
        stats,
        fates,
        evicted_base,
    })
}

type Identity<'a> = (u32, &'a [u8]);

fn query_deps(entry: &Entry) -> impl Iterator<Item = (Identity<'_>, u128)> {
    entry.deps.iter().filter_map(|dep| match dep {
        StoredDep::Query { table, key, fp } => Some(((*table, key.as_slice()), *fp)),
        _ => None,
    })
}

/// The base entries a fresh entry's query dep may name instead of a
/// fresh one: only at the fp of the stored value, which is what a restore
/// compares, so a dep whose rewrite was not kept does not count.
struct Held<'a> {
    at: HashMap<Identity<'a>, &'a Entry>,
    fps: RefCell<HashMap<Identity<'a>, u128>>,
}

impl<'a> Held<'a> {
    fn new(based: &'a [Entry]) -> Self {
        Held {
            at: based
                .iter()
                .map(|e| ((e.table, e.key.as_slice()), e))
                .collect(),
            fps: RefCell::new(HashMap::new()),
        }
    }

    fn none() -> Self {
        Held {
            at: HashMap::new(),
            fps: RefCell::new(HashMap::new()),
        }
    }

    fn holds(&self, dep: Identity<'a>, fp: u128) -> bool {
        let Some(entry) = self.at.get(&dep) else {
            return false;
        };
        let stored = *self
            .fps
            .borrow_mut()
            .entry(dep)
            .or_insert_with(|| fingerprint(&entry.value));
        stored == fp
    }
}

/// The reverse query-dep edges among `entries`, and which entries read a
/// query that is not among them. A dep in `held` is satisfied without an
/// edge.
fn query_edges<'a>(entries: &'a [Entry], held: &Held<'a>) -> (Vec<Vec<usize>>, Vec<bool>) {
    let at: HashMap<Identity, usize> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| ((e.table, e.key.as_slice()), i))
        .collect();
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); entries.len()];
    let mut dangling = vec![false; entries.len()];
    for (i, entry) in entries.iter().enumerate() {
        for (dep, fp) in query_deps(entry) {
            if held.holds(dep, fp) {
                continue;
            }
            match at.get(&dep) {
                Some(&j) => dependents[j].push(i),
                None => dangling[i] = true,
            }
        }
    }
    (dependents, dangling)
}

/// Marks `root` and every entry that reads it transitively, passing each
/// newly marked index to `on_mark` once.
fn mark_with_dependents(
    dependents: &[Vec<usize>],
    marked: &mut [bool],
    root: usize,
    mut on_mark: impl FnMut(usize),
) {
    if marked[root] {
        return;
    }
    marked[root] = true;
    let mut stack = vec![root];
    while let Some(i) = stack.pop() {
        on_mark(i);
        for &j in &dependents[i] {
            if !marked[j] {
                marked[j] = true;
                stack.push(j);
            }
        }
    }
}

/// Index `i` is `true` when `entries[i]` reads, transitively through
/// `entries`, a query that neither `entries` nor `held` has. A dep `held`
/// has is kept whatever becomes of its fresh rewrite. Linear in entries
/// plus edges.
fn unkept<'a>(entries: &'a [Entry], held: &Held<'a>) -> Vec<bool> {
    let (dependents, dangling) = query_edges(entries, held);
    let mut out = vec![false; entries.len()];
    for (i, _) in dangling.iter().enumerate().filter(|(_, &d)| d) {
        mark_with_dependents(&dependents, &mut out, i, |_| {});
    }
    out
}

/// Every base entry whose table survives, decoded and renumbered into
/// `tables`, which gains the base tables with no live namesake.
fn read_base<B: MemoBackend + ?Sized>(
    backend: &B,
    base: &Generation,
    tables: &mut Vec<TableDesc>,
    stats: &mut CheckpointStats,
) -> Result<Vec<Entry>> {
    let remap: Vec<Option<u32>> = base
        .header()
        .tables
        .iter()
        .map(|desc| {
            if let Some(i) = tables.iter().position(|live| live == desc) {
                return Some(i as u32);
            }
            if tables.iter().any(|live| live.name == desc.name) {
                return None;
            }
            tables.push(desc.clone());
            Some(tables.len() as u32 - 1)
        })
        .collect();
    let remap_table = |table: u32| remap.get(table as usize).copied().flatten();

    let mut by_table: Vec<Vec<Entry>> = vec![Vec::new(); tables.len()];
    let mut items: Vec<&IndexItem> = base.items().iter().collect();
    items.sort_unstable_by_key(|item| item.loc);
    for item in items {
        let Some(frame) = backend.read_entry(base.id(), item)? else {
            // Another handle removed the base; nothing more can be merged.
            break;
        };
        let Ok(mut entry) = decode_entry(&frame, base.frame_tag()) else {
            stats.corrupt += 1;
            continue;
        };
        if entry.table != item.table || key_hash(&entry.key) != item.key_hash {
            stats.corrupt += 1;
            continue;
        }
        let Some(table) = remap_table(entry.table) else {
            continue;
        };
        entry.table = table;
        let deps_remapped = entry.deps.iter_mut().all(|dep| match dep {
            StoredDep::Query { table, .. } => remap_table(*table).map(|t| *table = t).is_some(),
            _ => true,
        });
        if deps_remapped {
            by_table[entry.table as usize].push(entry);
        }
    }
    Ok(by_table.into_iter().flatten().collect())
}

/// Each entry's frame, `None` for one whose frame would be too long and
/// for every entry that reads it transitively, all counted in `oversized`.
fn encode_fitting(
    entries: &[Entry],
    encode: impl Fn(&Entry) -> Result<EncodedEntry>,
    oversized: &mut usize,
) -> Result<Vec<Option<EncodedEntry>>> {
    let mut encoded = entries
        .iter()
        .map(|entry| match encode(entry) {
            Err(Error::FrameTooLarge { .. }) => Ok(None),
            other => other.map(Some),
        })
        .collect::<Result<Vec<_>>>()?;
    if encoded.iter().any(Option::is_none) {
        let mut dropped = vec![false; entries.len()];
        let (dependents, _) = query_edges(entries, &Held::none());
        for (i, _) in encoded.iter().enumerate().filter(|(_, e)| e.is_none()) {
            mark_with_dependents(&dependents, &mut dropped, i, |_| *oversized += 1);
        }
        for (frame, gone) in encoded.iter_mut().zip(dropped) {
            if gone {
                *frame = None;
            }
        }
    }
    Ok(encoded)
}

/// Index `i` is `true` when `entries[i]` must go so that the summed `sizes`
/// of the rest fit in `max_bytes`. Entries go oldest `last_used` first, ties
/// in slice order, each with every entry that reads it transitively.
fn evict(entries: &[Entry], sizes: &[u64], max_bytes: u64) -> Vec<bool> {
    let mut total: u64 = sizes.iter().sum();
    let mut out = vec![false; entries.len()];
    if total <= max_bytes {
        return out;
    }
    let (dependents, _) = query_edges(entries, &Held::none());
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by_key(|&i| (entries[i].last_used, i));
    for i in order {
        if total <= max_bytes {
            break;
        }
        mark_with_dependents(&dependents, &mut out, i, |j| total -= sizes[j]);
    }
    out
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
