use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::layers::{self, Live, Probe};
use super::summary::{Row, Summary, Summing};
use super::{
    decode_entry, fingerprint, key_hash, retag_frame, Chain, DeltaHeader, EncodedEntry, Entry,
    GenHeader, GenId, Generation, IndexItem, MemoBackend, StoredDep, TableDesc, Tombstone,
};
use crate::{Error, Result};
use incr_core::Yielder;

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
    /// Base frames that failed to decode, did not match their index item,
    /// or that the backend no longer returned.
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
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) mode: Mode,
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
/// A chain frame the backend no longer returns is skipped like a damaged
/// one, so a lost entry costs only itself.
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
    let mut probes = Probes::new(mergeable.and_then(|chain| known_summary(chain, known)));
    let input = match mergeable
        .filter(|chain| backend.supports_delta() && chain.depth() < MAX_CHAIN_DEPTH)
    {
        Some(chain) => match write_delta(backend, chain, input, &next, &encode, &mut probes)? {
            Ok(done) => return Ok(done),
            Err(input) => input,
        },
        None => input,
    };
    let merged = mergeable.is_some();
    write_full(
        backend,
        chain.as_ref(),
        merged,
        input,
        &next,
        encode,
        &mut probes,
    )
}

/// The summary a `known` layer keeps of `chain`, when the chain is one
/// full generation equal to that layer.
fn known_summary<'k>(chain: &Chain, known: &'k [Arc<Generation>]) -> Option<Summarized<'k>> {
    let [base] = chain.layers() else {
        return None;
    };
    let generation = known.iter().find(|layer| layer.id() == base.id())?;
    let summary = generation.summary()?;
    let same = Arc::ptr_eq(generation, base)
        || (generation.base().is_none()
            && generation.header() == base.header()
            && generation.items() == base.items());
    same.then_some(Summarized {
        generation,
        summary,
    })
}

/// A full generation and the summary this process kept of it.
#[derive(Clone, Copy)]
struct Summarized<'k> {
    generation: &'k Generation,
    summary: &'k Summary,
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
    probes: &mut Probes,
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
    let unkept = probes.unkept(
        backend,
        chain.layers(),
        &input.fresh,
        |table| to_base.get(table as usize).copied(),
        None,
    )?;
    let mut fates = vec![Fate::Written; input.fresh.len()];
    let mut kept = Vec::new();
    let mut origin = Vec::new();
    let yielder = Yielder::ambient();
    for (i, (entry, drop)) in input.fresh.iter().zip(unkept).enumerate() {
        yielder.tick();
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
        yielder.tick();
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
            mode: Mode::Delta,
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
    generation.set_shown(live);
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
        mode: Mode::Delta,
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

/// What a checkpoint learned of the stored entries its fresh entries'
/// query deps name: per live table and key, the stored entry's value fp
/// and query-dep tables, or `None` when the chain does not hold it. Deps
/// are looked up in the summary this process kept of the base when it has
/// one, and probed through the chain in batches otherwise.
struct Probes<'k> {
    summary: Option<Summarized<'k>>,
    found: HashMap<u32, HashMap<Vec<u8>, Option<Hit>>>,
}

struct Hit {
    fp: u128,
    /// In the chain's catalog.
    dep_tables: Vec<u32>,
}

impl<'k> Probes<'k> {
    fn new(summary: Option<Summarized<'k>>) -> Self {
        Probes {
            summary,
            found: HashMap::new(),
        }
    }

    fn get(&self, (table, key): Identity) -> Option<&Option<Hit>> {
        self.found.get(&table)?.get(key)
    }

    /// [`unkept`] for `fresh` against the stored entries of `layers`, whose
    /// catalog `to_chain` maps a live table into. With `remap`, a stored
    /// entry counts only when every query-dep table it names survives
    /// `remap`. A dep that names a fresh entry is probed only once that
    /// entry turns out unkept, since a kept one satisfies the dep anyway.
    fn unkept<B: MemoBackend + ?Sized>(
        &mut self,
        backend: &B,
        layers: &[Arc<Generation>],
        fresh: &[Entry],
        to_chain: impl Fn(u32) -> Option<u32>,
        remap: Option<&[Option<u32>]>,
    ) -> Result<Vec<bool>> {
        let at: HashMap<Identity, usize> = fresh
            .iter()
            .enumerate()
            .map(|(i, e)| ((e.table, e.key.as_slice()), i))
            .collect();
        let deps = || fresh.iter().flat_map(query_deps).map(|(dep, _)| dep);
        let stored: Vec<Identity> = deps().filter(|dep| !at.contains_key(dep)).collect();
        self.probe(backend, layers, &stored, &to_chain)?;
        let out = unkept(fresh, &self.held(remap));
        let rescue: Vec<Identity> = deps()
            .filter(|dep| at.get(dep).is_some_and(|&j| out[j]) && self.get(*dep).is_none())
            .collect();
        if rescue.is_empty() {
            return Ok(out);
        }
        self.probe(backend, layers, &rescue, &to_chain)?;
        Ok(unkept(fresh, &self.held(remap)))
    }

    fn probe<B: MemoBackend + ?Sized>(
        &mut self,
        backend: &B,
        layers: &[Arc<Generation>],
        wanted: &[Identity],
        to_chain: &impl Fn(u32) -> Option<u32>,
    ) -> Result<()> {
        let mut pending: Vec<(u32, &[u8])> = Vec::new();
        let mut keys: Vec<(u32, &[u8])> = Vec::new();
        for &(table, key) in wanted {
            let known = self.found.entry(table).or_default();
            if known.contains_key(key) {
                continue;
            }
            let hit = match (to_chain(table), self.summary) {
                (None, _) => None,
                (Some(at), Some(base)) => {
                    base.summary.find(base.generation, at, key).map(Hit::of_row)
                }
                (Some(at), None) => {
                    pending.push((table, key));
                    keys.push((at, key));
                    None
                }
            };
            known.insert(key.to_vec(), hit);
        }
        if keys.is_empty() {
            return Ok(());
        }
        let found = &mut self.found;
        layers::probe_many(
            layers,
            &keys,
            STREAM_BATCH,
            |id, items| backend.read_entries(id, items),
            |i, probe| {
                if let Probe::Found(entry) = probe {
                    let (table, key) = pending[i];
                    let hit = found
                        .get_mut(&table)
                        .and_then(|keys| keys.get_mut(key))
                        .expect("a pending dep was recorded");
                    *hit = Some(Hit::of(&entry));
                }
            },
        )
    }

    fn held<'p>(&'p self, remap: Option<&'p [Option<u32>]>) -> Held<'p> {
        Held {
            probes: Some(self),
            remap,
        }
    }

    fn summary(&self) -> Option<Summarized<'k>> {
        self.summary
    }
}

impl Hit {
    fn of(entry: &Entry) -> Self {
        Hit {
            fp: fingerprint(&entry.value),
            dep_tables: query_deps(entry).map(|((table, _), _)| table).collect(),
        }
    }

    fn of_row((row, fp): (&Row, u128)) -> Self {
        Hit {
            fp,
            dep_tables: row.deps.iter().map(|&(table, _)| table).collect(),
        }
    }
}

/// How a full write carried the chain's entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// No full write: a delta, or nothing written.
    Delta,
    /// One streamed pass over the chain, newest layer first: each frame
    /// retagged, or decoded, renumbered and re-encoded when `remap`.
    Copy { remap: bool },
    /// A pass that keeps only each visible entry's metadata, to evict or
    /// to drop the dependents of an oversized fresh entry, then a streamed
    /// pass over the entries that stay. The first pass reads nothing when
    /// this process kept a summary of the base.
    TwoPass { remap: bool },
}

/// Frames a full write reads from the chain per
/// [`MemoBackend::read_entries`] call.
pub(crate) const STREAM_BATCH: usize = 256;

/// Writes the next full generation from `input` and, when `merged`, every
/// entry `chain` shows that was not rewritten, streaming the chain's
/// frames into the write a batch at a time. Eviction runs when the result
/// would exceed `max_bytes`, down to 0.9 of it on a backend with deltas so
/// that the next checkpoints have room for deltas, and down to `max_bytes`
/// on one without. On a backend without deltas the new generation keeps a
/// summary of its entries for the next full write.
fn write_full<B: MemoBackend + ?Sized>(
    backend: &B,
    chain: Option<&Chain>,
    merged: bool,
    input: CheckpointInput,
    next: &Next,
    encode: impl Fn(&Entry, u64) -> Result<EncodedEntry>,
    probes: &mut Probes,
) -> Result<Checkpointed> {
    let seq = next.seq;
    let mut stats = CheckpointStats::default();
    let layers: &[Arc<Generation>] = match chain.filter(|_| merged) {
        Some(chain) => chain.layers(),
        None => &[],
    };
    let mut tables = input.tables;
    let mut remaps = vec![Vec::new(); layers.len()];
    for (at, layer) in layers.iter().enumerate().rev() {
        remaps[at] = remap_into(layer.header(), &mut tables);
    }
    let remap = remaps
        .iter()
        .any(|remap| (0..).zip(remap).any(|(i, &to)| to != Some(i)));

    let unkept = match layers.first() {
        Some(base) => probes.unkept(
            backend,
            layers,
            &input.fresh,
            |table| base.header().table_index(&tables[table as usize]),
            Some(&remaps[0]),
        )?,
        None => unkept(&input.fresh, &Held::none()),
    };
    let mut fates = vec![Fate::Written; input.fresh.len()];
    let mut fresh = Vec::with_capacity(input.fresh.len());
    let mut origin = Vec::with_capacity(input.fresh.len());
    let yielder = Yielder::ambient();
    for (i, (mut entry, drop)) in input.fresh.into_iter().zip(unkept).enumerate() {
        yielder.tick();
        if drop {
            stats.unkept += 1;
            fates[i] = Fate::Unkept;
        } else {
            entry.last_used = seq;
            fresh.push(entry);
            origin.push(i);
        }
    }
    let header = GenHeader {
        build: input.build,
        seq,
        tables,
    };
    let tag = header.frame_tag();
    let frames = encode_fitting(&fresh, |entry| encode(entry, tag), &mut stats.oversized)?;
    for (frame, &i) in frames.iter().zip(&origin) {
        if frame.is_none() {
            fates[i] = Fate::Oversized;
        }
    }
    let contested = contested(layers, &remaps, &fresh);
    let target = if backend.supports_delta() {
        input.max_bytes - input.max_bytes / 10
    } else {
        input.max_bytes
    };
    let evicting = shown_bytes(layers, &header.tables, &fresh, &frames) > input.max_bytes;
    let two_pass = evicting || frames.iter().any(Option::is_none);

    let source = Source {
        backend,
        layers,
        remaps: &remaps,
        remap,
        tag,
        encode: &encode,
    };
    let mut evicted_base = Vec::new();
    let summing = (!backend.supports_delta()).then(Summing::default);
    let (mut copier, mode) = if two_pass {
        let stored = match probes.summary().filter(|_| !remap) {
            Some(base) => scan_summary(base, &fresh, &contested),
            None => scan(&source, &fresh, &contested, &mut stats)?,
        };
        stats.merged = stored.len();
        let bound = Bound {
            max_bytes: input.max_bytes,
            target,
        };
        let chosen = choose(
            &fresh, frames, stored, bound, &origin, &mut fates, &mut stats,
        );
        let mut picked = vec![Vec::new(); layers.len()];
        for meta in chosen.stored {
            picked[meta.layer].push(meta.item);
        }
        evicted_base = chosen.evicted_base;
        let kept = chosen.fresh;
        let walk = Walk::new(&source, Some(picked));
        let copy = Copier::new(walk, kept, None, summing);
        (copy, Mode::TwoPass { remap })
    } else {
        let kept = frames.into_iter().flatten().collect();
        let walk = Walk::new(&source, None);
        let seen = fresh
            .iter()
            .inspect(|_| yielder.tick())
            .map(|entry| (entry.table, entry.key.clone()))
            .collect();
        let copy = Copier::new(walk, kept, Some((contested, seen)), summing);
        (copy, Mode::Copy { remap })
    };
    drop(fresh);
    let merged_at = Instant::now();
    stats.times.merge = merged_at - next.started;

    let superseded: Vec<GenId> = chain
        .map(|chain| chain.layers().iter().map(|layer| layer.id()).collect())
        .unwrap_or_default();
    let tip = chain.map(|chain| chain.tip().id());
    let generation = backend.write_generation_streamed(tip, next.id, &header, &mut copier)?;
    stats.corrupt += copier.corrupt;
    stats.oversized += copier.oversized;
    if !two_pass {
        stats.merged = copier.carried + copier.oversized;
    }
    stats.written = generation.entry_count() as usize;
    stats.fresh_written = fates.iter().filter(|&&f| f == Fate::Written).count();
    stats.bytes_appended = generation.entry_bytes();
    stats.compacted = superseded.len() > 1;
    if let Some(summary) = copier.summing.take().and_then(|s| s.finish(&generation)) {
        generation.set_summary(summary);
    }
    stats.times.write = merged_at.elapsed();
    let generation = Arc::new(generation);
    Ok(Checkpointed {
        chain: Chain::new(vec![Arc::clone(&generation)])?,
        generation,
        superseded,
        stats,
        fates,
        evicted_base,
        mode,
    })
}

/// When a two-pass write evicts: once its entries sum to more than
/// `max_bytes`, down to `target`.
#[derive(Clone, Copy)]
struct Bound {
    max_bytes: u64,
    target: u64,
}

/// What a two-pass write carries once oversized entries' dependents and
/// evicted entries are gone.
struct Chosen<'s> {
    /// The fitting fresh frames that stay.
    fresh: Vec<EncodedEntry>,
    /// The chain entries that stay, in scan order.
    stored: Vec<Meta<'s>>,
    /// The `(table, key)` of every chain entry evicted.
    evicted_base: Vec<(u32, Vec<u8>)>,
}

/// Drops every entry that reads an oversized fresh entry transitively,
/// then evicts within `bound`, recording each fresh entry's fate.
/// `frames[i]` is fresh entry `i`'s frame, `None` when it was too large,
/// and `origin[i]` its index in the checkpoint's input.
fn choose<'s>(
    fresh: &[Entry],
    frames: Vec<Option<EncodedEntry>>,
    stored: Vec<Meta<'s>>,
    bound: Bound,
    origin: &[usize],
    fates: &mut [Fate],
    stats: &mut CheckpointStats,
) -> Chosen<'s> {
    let nodes: Vec<Node> = fresh
        .iter()
        .zip(&frames)
        .map(|(entry, frame)| {
            let size = frame.as_ref().map_or(0, |f| f.frame.len() as u64);
            Node::of(entry, size)
        })
        .chain(stored.iter().map(Meta::node))
        .collect();
    let dependents = node_edges(&nodes);
    let mut dropped = vec![false; nodes.len()];
    for root in (0..fresh.len()).filter(|&i| frames[i].is_none()) {
        mark_with_dependents(&dependents, &mut dropped, root, |j| match frames.get(j) {
            Some(None) => {}
            Some(Some(_)) => {
                fates[origin[j]] = Fate::Oversized;
                stats.oversized += 1;
            }
            None => stats.oversized += 1,
        });
    }
    let evicted = evict(&nodes, &dependents, &dropped, bound.max_bytes, bound.target);
    drop(nodes);
    let mut gone = dropped.into_iter().zip(evicted);
    let mut chosen = Chosen {
        fresh: Vec::new(),
        stored: Vec::new(),
        evicted_base: Vec::new(),
    };
    for ((i, frame), (dropped, evicted)) in frames.into_iter().enumerate().zip(gone.by_ref()) {
        match frame {
            Some(_) if dropped => {}
            Some(_) if evicted => {
                stats.evicted += 1;
                fates[origin[i]] = Fate::Evicted;
            }
            Some(frame) => chosen.fresh.push(frame),
            None => {}
        }
    }
    for (meta, (dropped, evicted)) in stored.into_iter().zip(gone) {
        if dropped {
            continue;
        }
        if evicted {
            stats.evicted += 1;
            chosen
                .evicted_base
                .push((meta.table, meta.row.key.to_vec()));
        } else {
            chosen.stored.push(meta);
        }
    }
    chosen
}

/// The `(table, key_hash)`, in the new catalog, of every fresh entry and
/// every delta item: the only places two layers, or a layer and a fresh
/// entry, can hold the same key, so only these keys are compared.
fn contested(
    layers: &[Arc<Generation>],
    remaps: &[Vec<Option<u32>>],
    fresh: &[Entry],
) -> HashSet<(u32, u64)> {
    let mut out: HashSet<(u32, u64)> = fresh
        .iter()
        .map(|entry| (entry.table, key_hash(&entry.key)))
        .collect();
    for (layer, remap) in layers.iter().zip(remaps).skip(1) {
        for item in layer.items().iter().filter(|item| !item.is_tombstone()) {
            if let Some(table) = remap.get(item.table as usize).copied().flatten() {
                out.insert((table, item.key_hash));
            }
        }
    }
    out
}

/// The summed frame lengths of the fitting fresh entries and of what
/// `layers` show once the fresh keys replace theirs, by key hash as
/// [`Live`] counts. Fresh tables are in `tables`, the new catalog.
fn shown_bytes(
    layers: &[Arc<Generation>],
    tables: &[TableDesc],
    fresh: &[Entry],
    frames: &[Option<EncodedEntry>],
) -> u64 {
    let fresh_bytes: u64 = frames.iter().flatten().map(|f| f.frame.len() as u64).sum();
    let Some(base) = layers.first() else {
        return fresh_bytes;
    };
    let mut shadow: Vec<IndexItem> = fresh
        .iter()
        .filter_map(|entry| {
            let table = base.header().table_index(&tables[entry.table as usize])?;
            Some(IndexItem::tombstone(table, key_hash(&entry.key)))
        })
        .collect();
    shadow.sort_unstable_by_key(|item| (item.table, item.key_hash));
    let shown = Live::of_chain(layers).with_layer(layers, &shadow);
    fresh_bytes.saturating_add(shown.bytes)
}

/// What a full write reads the chain from and how it carries a frame.
struct Source<'a, B: ?Sized, E> {
    backend: &'a B,
    layers: &'a [Arc<Generation>],
    /// Per layer, its catalog mapped into the new one.
    remaps: &'a [Vec<Option<u32>>],
    /// Some table moves, so frames are decoded and re-encoded instead of
    /// retagged.
    remap: bool,
    tag: u64,
    encode: &'a E,
}

enum Carried {
    Frame(EncodedEntry),
    /// Its table or a query dep's table did not survive.
    Dropped,
    Corrupt,
    TooLarge,
}

impl<B: MemoBackend + ?Sized, E: Fn(&Entry, u64) -> Result<EncodedEntry>> Source<'_, B, E> {
    /// `frame` of `item` in layer `at` as a frame of the new generation,
    /// with its key when `with_key`.
    fn carry(
        &self,
        at: usize,
        item: &IndexItem,
        frame: &[u8],
        with_key: bool,
    ) -> Result<(Carried, Option<Vec<u8>>)> {
        let layer = &self.layers[at];
        if !self.remap {
            let carried = retag_frame(frame, layer.frame_tag(), self.tag)
                .ok()
                .filter(|e| e.table == item.table && e.key_hash == item.key_hash);
            let Some(carried) = carried else {
                return Ok((Carried::Corrupt, None));
            };
            if !with_key {
                return Ok((Carried::Frame(carried), None));
            }
            return Ok(match decode_entry(frame, layer.frame_tag()) {
                Ok(entry) => (Carried::Frame(carried), Some(entry.key)),
                Err(_) => (Carried::Corrupt, None),
            });
        }
        let mut corrupt = 0;
        let entry = match decode_stored(frame, layer, item, &self.remaps[at], &mut corrupt) {
            Stored::Kept(entry) => entry,
            Stored::Dropped => return Ok((Carried::Dropped, None)),
            Stored::Corrupt => return Ok((Carried::Corrupt, None)),
        };
        let key = with_key.then(|| entry.key.clone());
        match (self.encode)(&entry, self.tag) {
            Ok(frame) => Ok((Carried::Frame(frame), key)),
            Err(Error::FrameTooLarge { .. }) => Ok((Carried::TooLarge, None)),
            Err(e) => Err(e),
        }
    }
}

/// Items read from one layer, the layer's index first, each with its frame
/// or `None` when the backend no longer has it.
type Batch = (usize, Vec<(IndexItem, Option<Vec<u8>>)>);

/// The contested `(table, key_hash)` and the keys already carried or fresh
/// among them.
type Dedup = (HashSet<(u32, u64)>, HashSet<(u32, Vec<u8>)>);

/// Reads the chain's items newest layer first, each layer in frame order,
/// [`STREAM_BATCH`] at a time. A layer's tombstones, and the items the
/// reader [`hide`](Self::hide)s, hide the items of older layers with the
/// same table and key hash. With `picked`, only those items are read.
struct Walk<'s, 'a, B: ?Sized, E> {
    source: &'s Source<'a, B, E>,
    picked: Option<Vec<Vec<IndexItem>>>,
    /// Layers not yet started; the next is `layers[unstarted - 1]`.
    unstarted: usize,
    current: Option<(usize, std::vec::IntoIter<IndexItem>)>,
    hidden: HashSet<(u32, u64)>,
    hides: Vec<(u32, u64)>,
}

impl<'s, 'a, B: MemoBackend + ?Sized, E> Walk<'s, 'a, B, E> {
    fn new(source: &'s Source<'a, B, E>, picked: Option<Vec<Vec<IndexItem>>>) -> Self {
        Walk {
            source,
            picked,
            unstarted: source.layers.len(),
            current: None,
            hidden: HashSet::new(),
            hides: Vec::new(),
        }
    }

    /// `item`'s table in the new catalog, if it survives.
    fn table(&self, at: usize, item: &IndexItem) -> Option<u32> {
        self.source.remaps[at]
            .get(item.table as usize)
            .copied()
            .flatten()
    }

    fn hide(&mut self, table: u32, key_hash: u64) {
        self.hides.push((table, key_hash));
    }

    /// The next batch of `(item, frame)` with the layer they are from, or
    /// `None` at the end.
    fn next_batch(&mut self) -> Result<Option<Batch>> {
        loop {
            let Some((at, items)) = self.current.as_mut() else {
                if !self.start_layer() {
                    return Ok(None);
                }
                continue;
            };
            let at = *at;
            let remaps = self.source.remaps;
            let hidden = &self.hidden;
            let batch: Vec<IndexItem> = items
                .by_ref()
                .filter(|item| {
                    remaps[at]
                        .get(item.table as usize)
                        .copied()
                        .flatten()
                        .is_some_and(|table| !hidden.contains(&(table, item.key_hash)))
                })
                .take(STREAM_BATCH)
                .collect();
            if batch.is_empty() {
                self.current = None;
                continue;
            }
            let layer = &self.source.layers[at];
            let frames = self.source.backend.read_entries(layer.id(), &batch)?;
            if frames.len() != batch.len() {
                return Err(Error::Corrupt(
                    "a memo backend returned another number of frames than it was asked for",
                ));
            }
            return Ok(Some((at, batch.into_iter().zip(frames).collect())));
        }
    }

    /// Commits the finished layer's hides and starts the next older one,
    /// or returns `false` when none is left.
    fn start_layer(&mut self) -> bool {
        self.hidden.extend(self.hides.drain(..));
        if self.unstarted == 0 {
            return false;
        }
        self.unstarted -= 1;
        let at = self.unstarted;
        let items = match self.picked.as_mut() {
            Some(picked) => std::mem::take(&mut picked[at]),
            None => {
                let layer = &self.source.layers[at];
                let mut items = Vec::with_capacity(layer.items().len());
                for item in layer.items() {
                    if !item.is_tombstone() {
                        items.push(*item);
                    } else if let Some(table) = self.table(at, item) {
                        self.hides.push((table, item.key_hash));
                    }
                }
                items.sort_by_key(|item| item.loc);
                items
            }
        };
        self.current = Some((at, items.into_iter()));
        true
    }
}

/// The frames of a full write: the fresh ones, then the chain's, pulled a
/// batch at a time. With `dedup`, the chain's entries are chosen as they
/// come, as a newest-first probe sees them; without it the walk reads only
/// entries a scan picked. With `summing`, every frame it hands out is
/// summarized.
struct Copier<'s, 'a, B: ?Sized, E> {
    walk: Walk<'s, 'a, B, E>,
    fresh: std::vec::IntoIter<EncodedEntry>,
    pending: VecDeque<EncodedEntry>,
    dedup: Option<Dedup>,
    summing: Option<Summing>,
    carried: usize,
    corrupt: usize,
    oversized: usize,
    yielder: Yielder<'static>,
}

impl<'s, 'a, B: MemoBackend + ?Sized, E: Fn(&Entry, u64) -> Result<EncodedEntry>>
    Copier<'s, 'a, B, E>
{
    fn new(
        walk: Walk<'s, 'a, B, E>,
        fresh: Vec<EncodedEntry>,
        dedup: Option<Dedup>,
        summing: Option<Summing>,
    ) -> Self {
        Copier {
            walk,
            fresh: fresh.into_iter(),
            pending: VecDeque::new(),
            dedup,
            summing,
            carried: 0,
            corrupt: 0,
            oversized: 0,
            yielder: Yielder::ambient(),
        }
    }

    /// Carries the walk's next batch into `pending`; `false` at the end.
    fn fill(&mut self) -> Result<bool> {
        let Some((at, batch)) = self.walk.next_batch()? else {
            return Ok(false);
        };
        let source = self.walk.source;
        for (item, frame) in batch {
            self.yielder.tick();
            let table = self.walk.table(at, &item).expect("walked items survive");
            let Some(frame) = frame else {
                self.corrupt += 1;
                self.walk.hide(table, item.key_hash);
                continue;
            };
            let contested = self
                .dedup
                .as_ref()
                .is_some_and(|(contested, _)| contested.contains(&(table, item.key_hash)));
            let (carried, key) = source.carry(at, &item, &frame, contested)?;
            match carried {
                Carried::Frame(frame) => {
                    if let (Some((_, seen)), Some(key)) = (self.dedup.as_mut(), key) {
                        if !seen.insert((table, key)) {
                            continue;
                        }
                    }
                    self.carried += 1;
                    self.pending.push_back(frame);
                }
                Carried::Dropped => self.walk.hide(table, item.key_hash),
                Carried::Corrupt => {
                    self.corrupt += 1;
                    self.walk.hide(table, item.key_hash);
                }
                Carried::TooLarge => {
                    self.oversized += 1;
                    self.walk.hide(table, item.key_hash);
                }
            }
        }
        Ok(true)
    }

    fn pull(&mut self) -> Option<Result<EncodedEntry>> {
        if let Some(frame) = self.fresh.next() {
            return Some(Ok(frame));
        }
        loop {
            if let Some(frame) = self.pending.pop_front() {
                return Some(Ok(frame));
            }
            match self.fill() {
                Ok(true) => {}
                Ok(false) => {
                    return None;
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

impl<B: MemoBackend + ?Sized, E: Fn(&Entry, u64) -> Result<EncodedEntry>> Iterator
    for Copier<'_, '_, B, E>
{
    type Item = Result<EncodedEntry>;

    fn next(&mut self) -> Option<Result<EncodedEntry>> {
        self.yielder.tick();
        let next = self.pull();
        if let (Some(summing), Some(Ok(frame))) = (self.summing.as_mut(), &next) {
            summing.add(frame, self.walk.source.tag);
        }
        next
    }
}

/// A chain entry a two-pass write may carry: where its frame is and what
/// eviction needs, without its value.
struct Meta<'s> {
    layer: usize,
    item: IndexItem,
    /// In the new catalog, as are the deps' tables.
    table: u32,
    row: Cow<'s, Row>,
    /// Its frame's length in the new generation.
    size: u64,
}

impl Meta<'_> {
    fn node(&self) -> Node<'_> {
        Node {
            table: self.table,
            key: &self.row.key,
            deps: self.row.deps.iter().map(|(t, k)| (*t, &**k)).collect(),
            last_used: self.row.last_used,
            size: self.size,
        }
    }
}

/// The first pass of a two-pass write: every entry the chain shows that no
/// fresh entry replaces, decoded one batch at a time and kept as [`Meta`].
fn scan<B: MemoBackend + ?Sized, E: Fn(&Entry, u64) -> Result<EncodedEntry>>(
    source: &Source<'_, B, E>,
    fresh: &[Entry],
    contested: &HashSet<(u32, u64)>,
    stats: &mut CheckpointStats,
) -> Result<Vec<Meta<'static>>> {
    let mut seen: HashSet<(u32, Vec<u8>)> = fresh
        .iter()
        .map(|entry| (entry.table, entry.key.clone()))
        .collect();
    let mut walk = Walk::new(source, None);
    let mut out = Vec::new();
    while let Some((at, batch)) = walk.next_batch()? {
        let layer = &source.layers[at];
        for (item, frame) in batch {
            let table = walk.table(at, &item).expect("walked items survive");
            let Some(frame) = frame else {
                stats.corrupt += 1;
                walk.hide(table, item.key_hash);
                continue;
            };
            let stored =
                decode_stored(&frame, layer, &item, &source.remaps[at], &mut stats.corrupt);
            let entry = match stored {
                Stored::Kept(entry) => entry,
                Stored::Dropped | Stored::Corrupt => {
                    walk.hide(table, item.key_hash);
                    continue;
                }
            };
            if contested.contains(&(table, item.key_hash))
                && !seen.insert((entry.table, entry.key.clone()))
            {
                continue;
            }
            let size = if source.remap {
                match (source.encode)(&entry, source.tag) {
                    Ok(frame) => frame.frame.len() as u64,
                    Err(Error::FrameTooLarge { .. }) => {
                        stats.oversized += 1;
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            } else {
                u64::from(item.len)
            };
            out.push(Meta {
                layer: at,
                item,
                table: entry.table,
                row: Cow::Owned(Row::new(&entry.key, &entry.deps, entry.last_used)),
                size,
            });
        }
    }
    Ok(out)
}

/// [`scan`] of a chain that is the one generation `base` summarizes, with
/// no table moving: reads nothing, and takes the entries in the order a
/// scan walks them, by `loc` and then index order.
fn scan_summary<'s>(
    base: Summarized<'s>,
    fresh: &[Entry],
    contested: &HashSet<(u32, u64)>,
) -> Vec<Meta<'s>> {
    let items = base.generation.items();
    let rows = base.summary.rows();
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by_key(|&i| items[i].loc);
    let mut seen: HashSet<Identity> = fresh
        .iter()
        .map(|entry| (entry.table, entry.key.as_slice()))
        .collect();
    let mut out = Vec::with_capacity(items.len());
    for i in order {
        let (item, row) = (items[i], &rows[i]);
        if contested.contains(&(item.table, item.key_hash)) && !seen.insert((item.table, &row.key))
        {
            continue;
        }
        out.push(Meta {
            layer: 0,
            item,
            table: item.table,
            row: Cow::Borrowed(row),
            size: u64::from(item.len),
        });
    }
    out
}

/// An entry as eviction and the oversized-dependents rule see it.
struct Node<'a> {
    table: u32,
    key: &'a [u8],
    deps: Vec<(u32, &'a [u8])>,
    last_used: u64,
    size: u64,
}

impl<'a> Node<'a> {
    fn of(entry: &'a Entry, size: u64) -> Self {
        Node {
            table: entry.table,
            key: &entry.key,
            deps: query_deps(entry).map(|(dep, _)| dep).collect(),
            last_used: entry.last_used,
            size,
        }
    }
}

/// Per node, the nodes that read it through a query dep.
fn node_edges(nodes: &[Node]) -> Vec<Vec<usize>> {
    let at: HashMap<Identity, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| ((n.table, n.key), i))
        .collect();
    let mut dependents = vec![Vec::new(); nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        for dep in &node.deps {
            if let Some(&j) = at.get(dep) {
                dependents[j].push(i);
            }
        }
    }
    dependents
}

type Identity<'a> = (u32, &'a [u8]);

fn query_deps(entry: &Entry) -> impl Iterator<Item = (Identity<'_>, u128)> {
    entry.deps.iter().filter_map(|dep| match dep {
        StoredDep::Query { table, key, fp } => Some(((*table, key.as_slice()), *fp)),
        _ => None,
    })
}

/// The stored entries a fresh entry's query dep may name instead of a
/// fresh one: only at the fp of the stored value, which is what a restore
/// compares, so a dep whose rewrite was not kept does not count. With
/// `remap`, only stored entries whose query-dep tables all survive it.
struct Held<'p> {
    probes: Option<&'p Probes<'p>>,
    remap: Option<&'p [Option<u32>]>,
}

impl Held<'_> {
    fn none() -> Self {
        Held {
            probes: None,
            remap: None,
        }
    }

    fn holds(&self, dep: Identity, fp: u128) -> bool {
        let Some(Some(hit)) = self.probes.and_then(|probes| probes.get(dep)) else {
            return false;
        };
        hit.fp == fp
            && self.remap.is_none_or(|remap| {
                hit.dep_tables
                    .iter()
                    .all(|&table| remap.get(table as usize).copied().flatten().is_some())
            })
    }
}

/// The reverse query-dep edges among `entries`, and which entries read a
/// query that is not among them. A dep in `held` is satisfied without an
/// edge.
fn query_edges(entries: &[Entry], held: &Held) -> (Vec<Vec<usize>>, Vec<bool>) {
    let yielder = Yielder::ambient();
    let at: HashMap<Identity, usize> = entries
        .iter()
        .inspect(|_| yielder.tick())
        .enumerate()
        .map(|(i, e)| ((e.table, e.key.as_slice()), i))
        .collect();
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); entries.len()];
    let mut dangling = vec![false; entries.len()];
    for (i, entry) in entries.iter().enumerate() {
        for (dep, fp) in query_deps(entry) {
            yielder.tick();
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
fn unkept(entries: &[Entry], held: &Held) -> Vec<bool> {
    let (dependents, dangling) = query_edges(entries, held);
    let mut out = vec![false; entries.len()];
    for (i, _) in dangling.iter().enumerate().filter(|(_, &d)| d) {
        mark_with_dependents(&dependents, &mut out, i, |_| {});
    }
    out
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
    corrupt: &mut usize,
) -> Stored {
    let remap_table = |table: u32| remap.get(table as usize).copied().flatten();
    let entry = decode_entry(frame, layer.frame_tag())
        .ok()
        .filter(|entry| entry.table == item.table && key_hash(&entry.key) == item.key_hash);
    let Some(mut entry) = entry else {
        *corrupt += 1;
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
    let yielder = Yielder::ambient();
    let mut encoded = entries
        .iter()
        .inspect(|_| yielder.tick())
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

/// Index `i` is `true` when node `i` must go, once the nodes `skip` does
/// not name sum to more than `max_bytes`, so that the rest fit in
/// `target`. Nodes go oldest `last_used` first, ties in slice order, each
/// with every node that reads it transitively.
fn evict(
    nodes: &[Node],
    dependents: &[Vec<usize>],
    skip: &[bool],
    max_bytes: u64,
    target: u64,
) -> Vec<bool> {
    let mut total: u64 = nodes
        .iter()
        .zip(skip)
        .filter(|(_, &skipped)| !skipped)
        .map(|(node, _)| node.size)
        .sum();
    let mut out = vec![false; nodes.len()];
    if total <= max_bytes {
        return out;
    }
    let mut marked = skip.to_vec();
    let mut order: Vec<usize> = (0..nodes.len()).filter(|&i| !skip[i]).collect();
    order.sort_by_key(|&i| (nodes[i].last_used, i));
    for i in order {
        if total <= target {
            break;
        }
        mark_with_dependents(dependents, &mut marked, i, |j| {
            out[j] = true;
            total -= nodes[j].size;
        });
    }
    out
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
