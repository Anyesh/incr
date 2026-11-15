use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::layers::{self, Live, Probe};
use super::{
    decode_entry, fingerprint, key_hash, Chain, DeltaHeader, EncodedEntry, Entry, GenHeader, GenId,
    Generation, IndexItem, MemoBackend, StoredDep, TableDesc, Tombstone,
};
use crate::{Error, Result};

/// The most deltas a chain holds; the checkpoint that would add one more
/// rewrites the chain as a full generation.
pub const MAX_CHAIN_DEPTH: usize = 8;

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
    /// No entry was fresh, so `Memo::checkpoint` wrote nothing: the
    /// current generation stays and every count is zero.
    pub unchanged: bool,
    /// The checkpoint wrote a delta on the current chain instead of a full
    /// generation.
    pub delta: bool,
    /// Fresh entries written, a part of `written`.
    pub fresh_written: usize,
    /// Tombstones the delta wrote; 0 for a full generation.
    pub tombstones: usize,
    /// The summed length of the entry frames written.
    pub bytes_appended: u64,
    /// Deltas in the chain after the checkpoint; 0 after a full write.
    pub chain_depth: usize,
    /// A full write replaced a chain that held deltas.
    pub compacted: bool,
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
    /// The generation written: a delta or a full generation.
    pub generation: Arc<Generation>,
    /// The current chain after the write, `generation` its tip. Its older
    /// layers may have been written by other handles.
    pub chain: Chain,
    /// The layers of the chain a full write replaced, oldest first, empty
    /// after a delta. The caller removes them with
    /// [`MemoBackend::remove_generation`] once nothing in this process
    /// reads them any more.
    pub superseded: Vec<GenId>,
    pub stats: CheckpointStats,
    /// Per `fresh` entry, in input order.
    pub(crate) fates: Vec<Fate>,
    /// The `(table, key)` of every base entry eviction dropped, tables in
    /// the input catalog's numbering extended by carried-over tables.
    pub(crate) evicted_base: Vec<(u32, Vec<u8>)>,
}

/// Runs a checkpoint's write step against `backend`: takes the checkpoint
/// lock, loads the current chain, and writes the fresh entries either as a
/// delta on it or, merged with the entries the chain shows, as the next
/// full generation. Fresh entries are kept when their query deps are kept
/// fresh entries or stored entries at the same fp (to a fixpoint) and
/// stamped with the new sequence number; entries too large for a frame are
/// dropped with their transitive dependents.
///
/// A delta is written when the backend
/// [supports one](MemoBackend::supports_delta), the base's build matches,
/// its seq can advance, every live table is in the base's catalog, the
/// chain holds fewer than [`MAX_CHAIN_DEPTH`] deltas, the delta's frames
/// come to at most a quarter of the base's bytes and the chain's live bytes
/// stay within `max_bytes`. It holds only the fresh entries, numbered in
/// the base's catalog, and a tombstone for each oversized one the chain
/// shows; unkept ones write nothing, so their stored versions stay. A delta
/// with nothing to write publishes no layer.
///
/// Otherwise the full write merges the chain's entries that were not
/// rewritten and evicts by oldest `last_used`, each with transitive
/// dependents, down to `max_bytes`. It merges only when the base's build
/// matches and seq is below `u64::MAX`; otherwise numbering restarts at
/// seq 1. A stored table whose name matches a live table with a different
/// descriptor is dropped; a stored table with no live namesake is carried
/// over, appended to the catalog.
///
/// Fails with [`Error::Busy`](crate::Error::Busy) when another handle holds
/// the lock or changed the chain meanwhile, and with
/// [`Error::Corrupt`](crate::Error::Corrupt) when the tip's id is
/// `u64::MAX`; nothing is written then.
pub fn write_checkpoint<B: MemoBackend + ?Sized>(
    backend: &B,
    input: CheckpointInput,
) -> Result<Checkpointed> {
    write_known(backend, input, &[])
}

/// [`write_checkpoint`] given the layers the caller's index holds, which
/// [`MemoBackend::load_chain`] may reuse.
pub(crate) fn write_known<B: MemoBackend + ?Sized>(
    backend: &B,
    input: CheckpointInput,
    known: &[Arc<Generation>],
) -> Result<Checkpointed> {
    write_chain(backend, input, known, EncodedEntry::new)
}

#[cfg(test)]
fn write_with<B: MemoBackend + ?Sized>(
    backend: &B,
    input: CheckpointInput,
    encode: impl Fn(&Entry, u64) -> Result<EncodedEntry>,
) -> Result<Checkpointed> {
    write_chain(backend, input, &[], encode)
}

fn write_chain<B: MemoBackend + ?Sized>(
    backend: &B,
    input: CheckpointInput,
    known: &[Arc<Generation>],
    encode: impl Fn(&Entry, u64) -> Result<EncodedEntry>,
) -> Result<Checkpointed> {
    let _guard = backend.lock_checkpoint()?;
    let started = Instant::now();
    let chain = backend.load_chain(known)?;
    let tip = chain.as_ref().map(Chain::tip);
    let id = tip
        .map_or(Some(1), |g| g.id().0.checked_add(1))
        .map(GenId)
        .ok_or(Error::Corrupt(
            "the current memo generation has the largest id",
        ))?;
    let newest_seq = chain
        .as_ref()
        .and_then(|chain| chain.layers().iter().map(|layer| layer.header().seq).max());
    let next_seq = newest_seq.map_or(Some(1), |seq| seq.checked_add(1));
    let mergeable = chain
        .as_ref()
        .filter(|chain| next_seq.is_some() && chain.base().header().build == input.build);
    let next = Next {
        id,
        seq: next_seq.unwrap_or(1),
        started,
    };
    let input = match mergeable
        .filter(|chain| backend.supports_delta() && chain.depth() < MAX_CHAIN_DEPTH)
    {
        Some(chain) => match write_delta(backend, chain, input, &next, &encode)? {
            Ok(done) => return Ok(done),
            Err(input) => input,
        },
        None => input,
    };
    let merged = mergeable.is_some();
    write_full(backend, chain.as_ref(), merged, input, &next, encode)
}

/// The id and sequence number the checkpoint writes under.
struct Next {
    id: GenId,
    seq: u64,
    started: Instant,
}

/// Writes `input` as a delta on `chain`, or hands it back when a delta
/// would break one of the rules [`write_checkpoint`] lists.
fn write_delta<B: MemoBackend + ?Sized>(
    backend: &B,
    chain: &Chain,
    input: CheckpointInput,
    next: &Next,
    encode: &impl Fn(&Entry, u64) -> Result<EncodedEntry>,
) -> Result<std::result::Result<Checkpointed, CheckpointInput>> {
    let base = chain.base();
    let base_bytes = base.entry_bytes();
    let to_base: Option<Vec<u32>> = input
        .tables
        .iter()
        .map(|desc| base.header().table_index(desc))
        .collect();
    let Some(to_base) = to_base else {
        return Ok(Err(input));
    };
    let least: u64 = input
        .fresh
        .iter()
        .map(|e| (e.key.len() + e.value.len()) as u64)
        .sum();
    if least.saturating_mul(4) > base_bytes {
        return Ok(Err(input));
    }
    let mut stats = CheckpointStats {
        delta: true,
        chain_depth: chain.depth() + 1,
        ..CheckpointStats::default()
    };
    let held = held_in_chain(backend, chain, &input.fresh, &to_base)?;
    let unkept = unkept(&input.fresh, &held);
    drop(held);
    let mut fates = vec![Fate::Written; input.fresh.len()];
    let mut kept = Vec::new();
    let mut origin = Vec::new();
    for (i, (entry, drop)) in input.fresh.iter().zip(unkept).enumerate() {
        if drop {
            stats.unkept += 1;
            fates[i] = Fate::Unkept;
            continue;
        }
        let mut entry = entry.clone();
        entry.last_used = next.seq;
        renumber(&mut entry, &to_base);
        kept.push(entry);
        origin.push(i);
    }
    let merged_at = Instant::now();
    stats.times.merge = merged_at - next.started;

    let header = DeltaHeader {
        header: GenHeader {
            build: input.build,
            seq: next.seq,
            tables: base.header().tables.clone(),
        },
        base: base.id(),
        parent: chain.tip().id(),
    };
    let tag = header.frame_tag();
    let frames = encode_fitting(&kept, |entry| encode(entry, tag), &mut stats.oversized)?;
    let mut written = Vec::with_capacity(frames.len());
    let mut tombstones = Vec::new();
    for ((entry, i), frame) in kept.iter().zip(origin).zip(frames) {
        match frame {
            Some(frame) => written.push(frame),
            None => {
                fates[i] = Fate::Oversized;
                let hash = key_hash(&entry.key);
                if layers::may_hold(chain.layers(), entry.table, hash) {
                    tombstones.push(Tombstone {
                        table: entry.table,
                        key_hash: hash,
                    });
                }
            }
        }
    }
    drop(kept);
    if written.is_empty() && tombstones.is_empty() {
        stats.delta = false;
        stats.chain_depth = chain.depth();
        stats.times.write = merged_at.elapsed();
        return Ok(Ok(Checkpointed {
            generation: Arc::clone(chain.tip()),
            chain: chain.clone(),
            superseded: Vec::new(),
            stats,
            fates,
            evicted_base: Vec::new(),
        }));
    }
    let bytes: u64 = written.iter().map(|e| e.frame.len() as u64).sum();
    let mut items: Vec<IndexItem> = written
        .iter()
        .map(|e| IndexItem {
            table: e.table,
            key_hash: e.key_hash,
            loc: 0,
            len: e.frame.len() as u32,
        })
        .chain(
            tombstones
                .iter()
                .map(|t| IndexItem::tombstone(t.table, t.key_hash)),
        )
        .collect();
    items.sort_unstable_by_key(|item| (item.table, item.key_hash));
    let live = Live::of_chain(chain.layers()).with_layer(chain.layers(), &items);
    if bytes.saturating_mul(4) > base_bytes || live.bytes > input.max_bytes {
        return Ok(Err(input));
    }
    stats.written = written.len();
    stats.fresh_written = written.len();
    stats.tombstones = tombstones.len();
    stats.bytes_appended = bytes;
    let generation =
        backend.write_delta(chain.tip().id(), next.id, &header, &written, &tombstones)?;
    stats.times.write = merged_at.elapsed();
    let generation = Arc::new(generation);
    let mut layers = chain.layers().to_vec();
    layers.push(Arc::clone(&generation));
    Ok(Ok(Checkpointed {
        generation,
        chain: Chain::new(layers)?,
        superseded: Vec::new(),
        stats,
        fates,
        evicted_base: Vec::new(),
    }))
}

/// Moves `entry`'s table and query-dep tables from the live catalog to
/// the base's through `to_base`.
fn renumber(entry: &mut Entry, to_base: &[u32]) {
    entry.table = to_base[entry.table as usize];
    for dep in &mut entry.deps {
        if let StoredDep::Query { table, .. } = dep {
            *table = to_base[*table as usize];
        }
    }
}

/// The fps of the stored entries the fresh entries' query deps name, read
/// through `chain` newest first; deps are keyed in the live catalog.
fn held_in_chain<'a, B: MemoBackend + ?Sized>(
    backend: &B,
    chain: &Chain,
    fresh: &'a [Entry],
    to_base: &[u32],
) -> Result<Held<'a>> {
    let mut fps = HashMap::new();
    for entry in fresh {
        for (dep, _) in query_deps(entry) {
            if fps.contains_key(&dep) {
                continue;
            }
            let Some(&table) = to_base.get(dep.0 as usize) else {
                continue;
            };
            let probe = layers::probe(chain.layers(), table, dep.1, |id, item| {
                backend.read_entry(id, item)
            })?;
            if let Probe::Found(stored) = probe {
                fps.insert(dep, fingerprint(&stored.value));
            }
        }
    }
    Ok(Held {
        at: HashMap::new(),
        fps: RefCell::new(fps),
    })
}

/// Writes the next full generation from `input` and, when `merged`, every
/// entry `chain` shows that was not rewritten.
fn write_full<B: MemoBackend + ?Sized>(
    backend: &B,
    chain: Option<&Chain>,
    merged: bool,
    input: CheckpointInput,
    next: &Next,
    encode: impl Fn(&Entry, u64) -> Result<EncodedEntry>,
) -> Result<Checkpointed> {
    let seq = next.seq;
    let mut stats = CheckpointStats::default();
    let mut tables = input.tables;
    let based = match chain.filter(|_| merged) {
        Some(chain) => read_chain(backend, chain.layers(), &mut tables, &mut stats)?,
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
    stats.times.merge = merged_at - next.started;

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
    stats.fresh_written = fates.iter().filter(|&&f| f == Fate::Written).count();
    stats.bytes_appended = kept.iter().map(|e| e.frame.len() as u64).sum();
    let superseded: Vec<GenId> = chain
        .map(|chain| chain.layers().iter().map(|layer| layer.id()).collect())
        .unwrap_or_default();
    stats.compacted = superseded.len() > 1;
    let tip = chain.map(|chain| chain.tip().id());
    let generation = Arc::new(backend.write_generation(tip, next.id, &header, &kept)?);
    stats.times.write = merged_at.elapsed();
    Ok(Checkpointed {
        chain: Chain::new(vec![Arc::clone(&generation)])?,
        generation,
        superseded,
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
        if let Some(&stored) = self.fps.borrow().get(&dep) {
            return stored == fp;
        }
        let Some(entry) = self.at.get(&dep) else {
            return false;
        };
        let stored = fingerprint(&entry.value);
        self.fps.borrow_mut().insert(dep, stored);
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

/// Every entry `layers` (oldest first) show whose table survives, decoded
/// and renumbered into `tables`, which gains the stored tables with no live
/// namesake. Newer layers hide older entries of the same key, and a
/// tombstone, a corrupt frame or a dropped entry those of its table and
/// key hash. Entries
/// come grouped by table, older layers first, each layer in frame order.
fn read_chain<B: MemoBackend + ?Sized>(
    backend: &B,
    layers: &[Arc<Generation>],
    tables: &mut Vec<TableDesc>,
    stats: &mut CheckpointStats,
) -> Result<Vec<Entry>> {
    if let [base] = layers {
        return read_base(backend, base, tables, stats);
    }
    let mut hidden: HashSet<(u32, u64)> = HashSet::new();
    let mut seen: HashSet<(u32, Vec<u8>)> = HashSet::new();
    let mut per_layer: Vec<Vec<Entry>> = Vec::with_capacity(layers.len());
    'layers: for layer in layers.iter().rev() {
        let remap = remap_into(layer.header(), tables);
        let remap_table = |table: u32| remap.get(table as usize).copied().flatten();
        let mut found = Vec::new();
        let mut hides = Vec::new();
        let mut items: Vec<&IndexItem> = layer.items().iter().collect();
        items.sort_unstable_by_key(|item| item.loc);
        for item in items {
            let Some(table) = remap_table(item.table) else {
                continue;
            };
            if hidden.contains(&(table, item.key_hash)) {
                continue;
            }
            if item.is_tombstone() {
                hides.push((table, item.key_hash));
                continue;
            }
            let Some(frame) = backend.read_entry(layer.id(), item)? else {
                // Another handle removed the chain; nothing more can be merged.
                per_layer.push(found);
                break 'layers;
            };
            match decode_stored(&frame, layer, item, &remap, stats) {
                Stored::Kept(entry) => {
                    if seen.insert((entry.table, entry.key.clone())) {
                        found.push(entry);
                    }
                }
                Stored::Dropped | Stored::Corrupt => hides.push((table, item.key_hash)),
            }
        }
        hidden.extend(hides);
        per_layer.push(found);
    }
    let mut by_table: Vec<Vec<Entry>> = vec![Vec::new(); tables.len()];
    for entry in per_layer.into_iter().rev().flatten() {
        by_table[entry.table as usize].push(entry);
    }
    Ok(by_table.into_iter().flatten().collect())
}

/// Every entry of the full generation `base` whose table survives, decoded
/// and renumbered into `tables` as [`read_chain`] does.
fn read_base<B: MemoBackend + ?Sized>(
    backend: &B,
    base: &Generation,
    tables: &mut Vec<TableDesc>,
    stats: &mut CheckpointStats,
) -> Result<Vec<Entry>> {
    let remap = remap_into(base.header(), tables);
    let mut by_table: Vec<Vec<Entry>> = vec![Vec::new(); tables.len()];
    let mut items: Vec<&IndexItem> = base.items().iter().collect();
    items.sort_unstable_by_key(|item| item.loc);
    for item in items {
        let Some(frame) = backend.read_entry(base.id(), item)? else {
            // Another handle removed the base; nothing more can be merged.
            break;
        };
        if let Stored::Kept(entry) = decode_stored(&frame, base, item, &remap, stats) {
            by_table[entry.table as usize].push(entry);
        }
    }
    Ok(by_table.into_iter().flatten().collect())
}

enum Stored {
    Kept(Entry),
    /// Its table or a query dep's table did not survive.
    Dropped,
    /// Counted in [`CheckpointStats::corrupt`].
    Corrupt,
}

/// Decodes `item`'s `frame` of `layer` and renumbers it through `remap`.
fn decode_stored(
    frame: &[u8],
    layer: &Generation,
    item: &IndexItem,
    remap: &[Option<u32>],
    stats: &mut CheckpointStats,
) -> Stored {
    let remap_table = |table: u32| remap.get(table as usize).copied().flatten();
    let entry = decode_entry(frame, layer.frame_tag())
        .ok()
        .filter(|entry| entry.table == item.table && key_hash(&entry.key) == item.key_hash);
    let Some(mut entry) = entry else {
        stats.corrupt += 1;
        return Stored::Corrupt;
    };
    let Some(table) = remap_table(entry.table) else {
        return Stored::Dropped;
    };
    entry.table = table;
    let deps_remapped = entry.deps.iter_mut().all(|dep| match dep {
        StoredDep::Query { table, .. } => remap_table(*table).map(|t| *table = t).is_some(),
        _ => true,
    });
    if deps_remapped {
        Stored::Kept(entry)
    } else {
        Stored::Dropped
    }
}

/// Per table of `header`'s catalog, its index in `tables`: a table equal
/// to a live one maps to it, one whose name a live table has under another
/// descriptor maps to nothing, and any other is appended to `tables`.
fn remap_into(header: &GenHeader, tables: &mut Vec<TableDesc>) -> Vec<Option<u32>> {
    header
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
        .collect()
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
