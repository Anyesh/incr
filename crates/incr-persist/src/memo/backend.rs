use super::{encode_entry, key_hash, Entry, GenHeader, GenId, Generation, IndexItem};
use crate::Result;

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
    pub fn new(entry: &Entry) -> Result<Self> {
        Ok(EncodedEntry {
            table: entry.table,
            key_hash: key_hash(&entry.key),
            frame: encode_entry(entry)?,
        })
    }
}

/// Storage for the generations of one memo store.
///
/// A store holds a sequence of immutable generations and names one of them
/// current. Several handles, in this process or others, may share a store;
/// every method takes `&self` and implementations serialize internally.
/// A checkpoint
/// calls, in order: [`lock_checkpoint`](Self::lock_checkpoint), [`load`](Self::load),
/// [`read_entry`](Self::read_entry) on the loaded generation,
/// [`write_generation`](Self::write_generation) from it, and drops the
/// guard. The caller swaps its in-memory index to the new generation and
/// only then calls [`remove_generation`](Self::remove_generation) on the
/// old one. Restores call `read_entry` on whichever generation their index
/// names, which may by then have been removed by another handle.
///
/// The [`memo_backend_conformance!`](crate::memo_backend_conformance)
/// suite checks this contract.
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
    fn lock_checkpoint(&self) -> Result<Self::Guard>;

    /// Loads the current generation's header and index, without entry
    /// values, or `None` when no generation was ever written.
    ///
    /// When the current generation's header or index is damaged it returns
    /// [`Generation::empty`] with the current id, so restores miss and the
    /// next checkpoint replaces it. A header of another format version is
    /// treated the same way. Errors are I/O failures only.
    fn load(&self) -> Result<Option<Generation>>;

    /// Returns the frame `item` locates in `generation`, exactly as it was
    /// written, or `None` when the generation has been removed.
    ///
    /// The bytes are not checked here: a damaged or short frame is returned
    /// as found, and the caller's [`decode_entry`](super::decode_entry)
    /// rejects it. Errors are I/O failures other than the generation being
    /// gone.
    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>>;

    /// Durably writes generation `id` with `header` and `entries`, and makes
    /// it current, if the current generation is still `base` (`None`: no
    /// generation was ever written). Otherwise it writes nothing visible
    /// and fails with [`Error::Busy`](crate::Error::Busy).
    ///
    /// - `id` is greater than every id this store has used.
    /// - Atomic: a crash at any point leaves `load` returning either `base`
    ///   or the new generation, whole.
    /// - Durable: once this returns `Ok`, a later open loads the new
    ///   generation.
    /// - The returned [`Generation`] equals what `load` returns next: the
    ///   same id and header, and one item per entry, in any order, whose
    ///   `len` is its frame's length and whose `loc` `read_entry` resolves.
    /// - `base` stays readable until it is removed.
    fn write_generation(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &[EncodedEntry],
    ) -> Result<Generation>;

    /// Deletes a generation that is no longer current. Removing one that is
    /// already gone succeeds. Handles still reading it get `None` from
    /// `read_entry`.
    fn remove_generation(&self, generation: GenId) -> Result<()>;
}
