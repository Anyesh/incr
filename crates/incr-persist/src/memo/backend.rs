use std::sync::Arc;

use super::{
    encode_entry, key_hash, Chain, DeltaHeader, Entry, GenHeader, GenId, Generation, IndexItem,
};
use crate::{Error, Result};

/// An entry frame ready to be written, with the index fields a backend
/// stores beside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedEntry {
    pub table: u32,
    pub key_hash: u64,
    /// The output of [`encode_entry`], stored and returned byte for byte.
    pub frame: Vec<u8>,
}

impl EncodedEntry {
    /// Encodes `entry` for the generation whose header has
    /// [`frame_tag`](super::GenHeader::frame_tag) `tag`.
    pub fn new(entry: &Entry, tag: u64) -> Result<Self> {
        Ok(EncodedEntry {
            table: entry.table,
            key_hash: key_hash(&entry.key),
            frame: encode_entry(entry, tag)?,
        })
    }
}

/// An entry of the generations below a delta that the delta hides: every
/// one of `table` whose key hashes to `key_hash`. A delta stores it as an
/// [`IndexItem::tombstone`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tombstone {
    pub table: u32,
    pub key_hash: u64,
}

/// Storage for the generations of one memo store.
///
/// A store holds a sequence of immutable generations and names one of them
/// current. Several handles, in this process or others, may share a store;
/// every method takes `&self` and implementations serialize internally.
///
/// A backend whose [`supports_delta`](Self::supports_delta) is `true` also
/// keeps chains: a full generation, its base, followed by deltas written
/// with [`write_delta`](Self::write_delta), each on the one before it (see
/// [`Chain`]). The current generation is then the chain's tip, the newest
/// layer: [`write_generation`](Self::write_generation) and `write_delta`
/// take it as their base, and a full write replaces the whole chain.
/// [`load`](Self::load) returns the chain's base, which is what a reader
/// that knows nothing of deltas sees. A backend that does not opt in never
/// holds a delta, so its current generation is its only one.
///
/// [`write_checkpoint`](super::write_checkpoint) calls, in order:
/// [`lock_checkpoint`](Self::lock_checkpoint),
/// [`load_chain`](Self::load_chain), [`read_entry`](Self::read_entry) on
/// the chain's layers, and either [`write_delta`](Self::write_delta) or
/// [`write_generation`](Self::write_generation) from its tip, then drops
/// the guard. Its caller swaps its in-memory index to the new chain and
/// only then calls [`remove_generation`](Self::remove_generation) on the
/// layers a full write replaced. Restores call `read_entry` on whichever
/// layer their index names, which may by then have been removed by another
/// handle.
///
/// The `memo_backend_conformance!` suite, behind the `backend-tests`
/// feature, checks this contract.
pub trait MemoBackend: Send + Sync {
    /// Held for the length of one checkpoint; dropping it releases the
    /// lock.
    type Guard;

    /// Takes the store's checkpoint lock without waiting. Fails with
    /// [`Error::Busy`](crate::Error::Busy) when another handle holds it,
    /// in which case the checkpoint is skipped. A backend whose
    /// [`write_generation`](Self::write_generation) is atomic against
    /// concurrent writers on its own may return a guard that excludes
    /// nothing; its base check then makes the losing writer `Busy`.
    ///
    /// Once taken, it removes every generation not in the current chain:
    /// what an interrupted write left, and old generations whose removal
    /// failed or never ran. Handles whose index names one of them then
    /// miss.
    fn lock_checkpoint(&self) -> Result<Self::Guard>;

    /// Loads the header and index of the current chain's base, without
    /// entry values, or `None` when no generation was ever written. Without
    /// deltas the base is the current generation.
    ///
    /// When the base's header or index is damaged it returns
    /// [`Generation::empty`] with the base's id, so restores miss and the
    /// next checkpoint replaces it. A header of another format version is
    /// treated the same way. Errors are I/O failures only.
    fn load(&self) -> Result<Option<Generation>>;

    /// Returns the frame `item` locates in `generation`, exactly as it was
    /// written, or `None` when the generation has been removed. `item` is
    /// not a tombstone, which has no frame.
    ///
    /// The bytes are not checked here: a damaged or short frame, or one of
    /// another generation, is returned as found, and the caller's
    /// [`decode_entry`](super::decode_entry) with the generation's frame
    /// tag rejects it. Errors are I/O failures other than the generation being
    /// gone.
    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>>;

    /// Durably writes generation `id` with `header` and `entries`, and makes
    /// it current, if the current generation, the chain's tip, is still
    /// `base` (`None`: no generation was ever written). Otherwise it writes
    /// nothing visible and fails with [`Error::Busy`](crate::Error::Busy).
    /// The new generation replaces the whole chain `base` ends.
    ///
    /// - `id` is greater than every id this store has used.
    /// - Atomic: a crash at any point leaves either `base` or the new
    ///   generation, whole, current.
    /// - Durable: once this returns `Ok`, a later open loads the new
    ///   generation.
    /// - The returned [`Generation`] equals what `load` returns next: the
    ///   same id and header, and one item per entry, in any order, whose
    ///   `len` is its frame's length and whose `loc` `read_entry` resolves.
    /// - `base` and every layer below it stay readable until removed.
    fn write_generation(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &[EncodedEntry],
    ) -> Result<Generation>;

    /// [`write_generation`](Self::write_generation) with the entries
    /// pulled one at a time, so a backend can write each frame as it comes
    /// instead of holding them all. Same contract; an `Err` from `entries`
    /// fails the write with that error and writes nothing visible.
    ///
    /// The iterator may call [`read_entry`](Self::read_entry) and
    /// [`read_entries`](Self::read_entries) on this same handle while the
    /// write is in progress, as a compaction reading the old chain does, so
    /// an implementation must not hold a lock across a pull that those
    /// reads also take. The default collects every entry, then calls
    /// `write_generation`.
    fn write_generation_streamed(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &mut dyn Iterator<Item = Result<EncodedEntry>>,
    ) -> Result<Generation> {
        let entries = entries.collect::<Result<Vec<_>>>()?;
        self.write_generation(base, id, header, &entries)
    }

    /// [`read_entry`](Self::read_entry) for each of `items` of
    /// `generation`, in order, none of them a tombstone. A backend whose
    /// reads cost a round trip each can batch them. The default reads them
    /// one by one.
    fn read_entries(&self, generation: GenId, items: &[IndexItem]) -> Result<Vec<Option<Vec<u8>>>> {
        items
            .iter()
            .map(|item| self.read_entry(generation, item))
            .collect()
    }

    /// Whether this backend keeps delta chains: `true` promises that
    /// [`load_chain`](Self::load_chain) returns every layer,
    /// [`write_delta`](Self::write_delta) writes them, and
    /// [`lock_checkpoint`](Self::lock_checkpoint) keeps the current
    /// chain. The default is `false`, and a checkpoint then writes only
    /// full generations, as on a backend from before deltas.
    fn supports_delta(&self) -> bool {
        false
    }

    /// Loads the current chain, oldest layer first: the base, then each
    /// delta on the one before it up to the tip, or `None` when no
    /// generation was ever written. Each layer is what `write_delta` or
    /// `write_generation` returned for it. A delta whose base is no longer
    /// current, or that is not on the chain's tip, is never part of it.
    ///
    /// `known` holds layers this handle loaded earlier. Generations are
    /// immutable, so a backend may return a known layer's `Arc` for the
    /// same id instead of reloading its index. A damaged base is handled
    /// as in `load`, and a damaged delta loads as
    /// [`Generation::empty_delta`] in its place. The default wraps `load`
    /// in a chain of one generation.
    fn load_chain(&self, known: &[Arc<Generation>]) -> Result<Option<Chain>> {
        let _ = known;
        Ok(self.load()?.map(Chain::from))
    }

    /// Durably writes delta `id` with `header`, `entries` and
    /// `tombstones`, and makes it the chain's new tip, if the current tip
    /// is still `tip`. Otherwise it writes nothing visible and fails with
    /// [`Error::Busy`](crate::Error::Busy). `header.parent` is `tip` and
    /// `header.base` is the chain's base; the entries' frames carry
    /// [`DeltaHeader::frame_tag`].
    ///
    /// Atomic and durable as `write_generation`, with `id` greater than
    /// every id used. The returned [`Generation`] is a
    /// [`Generation::new_delta`] equal to the layer `load_chain` returns
    /// next: one item per entry as in `write_generation`, plus one
    /// [`IndexItem::tombstone`] per tombstone. The default fails with
    /// [`Error::Unsupported`].
    fn write_delta(
        &self,
        tip: GenId,
        id: GenId,
        header: &DeltaHeader,
        entries: &[EncodedEntry],
        tombstones: &[Tombstone],
    ) -> Result<Generation> {
        let _ = (tip, id, header, entries, tombstones);
        Err(Error::Unsupported("memo delta"))
    }

    /// Deletes a generation that is no longer in the current chain.
    /// Removing one that is already gone succeeds. Handles still reading
    /// it get `None` from `read_entry`.
    fn remove_generation(&self, generation: GenId) -> Result<()>;

    /// [`remove_generation`](Self::remove_generation) for each of
    /// `generations`, which a backend may do in fewer steps. Every one is
    /// attempted; the ones that could not be removed are returned.
    fn remove_generations(&self, generations: &[GenId]) -> Vec<GenId> {
        generations
            .iter()
            .copied()
            .filter(|&generation| self.remove_generation(generation).is_err())
            .collect()
    }
}
