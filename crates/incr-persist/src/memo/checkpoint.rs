use std::collections::{HashMap, HashSet};

use super::{
    decode_entry, key_hash, EncodedEntry, Entry, GenHeader, GenId, Generation, MemoBackend,
    StoredDep, TableDesc,
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
}

#[derive(Debug)]
pub struct Checkpointed {
    pub generation: Generation,
    /// The generation this one replaced. The caller removes it with
    /// [`MemoBackend::remove_generation`] once nothing in this process
    /// reads it any more.
    pub replaced: Option<GenId>,
    pub stats: CheckpointStats,
}

/// Runs a checkpoint's write step against `backend`: takes the checkpoint
/// lock, loads the current generation as the base, keeps the fresh entries
/// whose query deps are kept (to a fixpoint), stamps them with the new
/// sequence number, merges the base entries that were not rewritten, drops
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
    encode: impl Fn(&Entry) -> Result<EncodedEntry>,
) -> Result<Checkpointed> {
    let _guard = backend.lock_checkpoint()?;
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

    let unkept = unkept(&input.fresh);
    let mut entries: Vec<Entry> = Vec::with_capacity(input.fresh.len());
    for (mut entry, drop) in input.fresh.into_iter().zip(unkept) {
        if drop {
            stats.unkept += 1;
        } else {
            entry.last_used = seq;
            entries.push(entry);
        }
    }

    let mut tables = input.tables;
    let mergeable = |g: &&Generation| next_seq.is_some() && g.header().build == input.build;
    if let Some(base) = base.as_ref().filter(mergeable) {
        merge(backend, base, &mut tables, &mut entries, &mut stats)?;
    }

    let (entries, encoded) = encode_fitting(entries, encode, &mut stats.oversized)?;
    let sizes: Vec<u64> = encoded.iter().map(|e| e.frame.len() as u64).collect();
    let evicted = evict(&entries, &sizes, input.max_bytes);
    let encoded: Vec<_> = encoded
        .into_iter()
        .zip(&evicted)
        .filter_map(|(entry, &gone)| (!gone).then_some(entry))
        .collect();
    stats.evicted = entries.len() - encoded.len();
    stats.written = encoded.len();

    let header = GenHeader {
        build: input.build,
        seq,
        tables,
    };
    let base_id = base.as_ref().map(Generation::id);
    let generation = backend.write_generation(base_id, id, &header, &encoded)?;
    Ok(Checkpointed {
        generation,
        replaced: base_id,
        stats,
    })
}

type Identity<'a> = (u32, &'a [u8]);

fn query_deps(entry: &Entry) -> impl Iterator<Item = Identity<'_>> {
    entry.deps.iter().filter_map(|dep| match dep {
        StoredDep::Query { table, key, .. } => Some((*table, key.as_slice())),
        _ => None,
    })
}

/// The reverse query-dep edges among `entries`, and which entries read a
/// query that is not among them.
fn query_edges(entries: &[Entry]) -> (Vec<Vec<usize>>, Vec<bool>) {
    let at: HashMap<Identity, usize> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| ((e.table, e.key.as_slice()), i))
        .collect();
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); entries.len()];
    let mut dangling = vec![false; entries.len()];
    for (i, entry) in entries.iter().enumerate() {
        for dep in query_deps(entry) {
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

/// Index `i` is `true` when `entries[i]` reads, transitively, a query that
/// is not in `entries`. Linear in entries plus edges.
fn unkept(entries: &[Entry]) -> Vec<bool> {
    let (dependents, dangling) = query_edges(entries);
    let mut out = vec![false; entries.len()];
    for (i, _) in dangling.iter().enumerate().filter(|(_, &d)| d) {
        mark_with_dependents(&dependents, &mut out, i, |_| {});
    }
    out
}

fn merge<B: MemoBackend + ?Sized>(
    backend: &B,
    base: &Generation,
    tables: &mut Vec<TableDesc>,
    entries: &mut Vec<Entry>,
    stats: &mut CheckpointStats,
) -> Result<()> {
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

    let mut present: HashSet<(u32, Vec<u8>)> =
        entries.iter().map(|e| (e.table, e.key.clone())).collect();
    for item in base.items() {
        let Some(frame) = backend.read_entry(base.id(), item)? else {
            // Another handle removed the base; nothing more can be merged.
            break;
        };
        let Ok(mut entry) = decode_entry(&frame) else {
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
        if deps_remapped && present.insert((entry.table, entry.key.clone())) {
            stats.merged += 1;
            entries.push(entry);
        }
    }
    Ok(())
}

/// Encodes `entries`, dropping each whose frame would be too long with
/// every entry that reads it transitively, and counting them in `oversized`.
fn encode_fitting(
    entries: Vec<Entry>,
    encode: impl Fn(&Entry) -> Result<EncodedEntry>,
    oversized: &mut usize,
) -> Result<(Vec<Entry>, Vec<EncodedEntry>)> {
    let encoded = entries
        .iter()
        .map(|entry| match encode(entry) {
            Err(Error::FrameTooLarge { .. }) => Ok(None),
            other => other.map(Some),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut dropped = vec![false; entries.len()];
    if encoded.iter().any(Option::is_none) {
        let (dependents, _) = query_edges(&entries);
        for (i, _) in encoded.iter().enumerate().filter(|(_, e)| e.is_none()) {
            mark_with_dependents(&dependents, &mut dropped, i, |_| *oversized += 1);
        }
    }
    Ok(entries
        .into_iter()
        .zip(encoded)
        .zip(dropped)
        .filter_map(|((entry, frame), gone)| Some((entry, frame.filter(|_| !gone)?)))
        .unzip())
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
    let (dependents, _) = query_edges(entries);
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
