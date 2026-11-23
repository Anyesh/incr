use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::{
    Chain, DeltaHeader, EncodedEntry, GenHeader, GenId, Generation, IndexItem, MemoBackend,
    Tombstone,
};
use crate::{Error, Result};

/// A memo store held in process memory, for tests.
///
/// Clones are handles on the same store, the way separate processes share
/// a directory. Nothing outlives the last handle.
#[derive(Clone, Debug, Default)]
pub struct MemoryMemo {
    state: Arc<Mutex<State>>,
    without_deltas: bool,
}

#[derive(Debug, Default)]
struct State {
    /// The current chain, oldest first; empty before the first write.
    chain: Vec<GenId>,
    generations: BTreeMap<GenId, Stored>,
    locked: bool,
}

impl State {
    fn tip(&self) -> Option<GenId> {
        self.chain.last().copied()
    }

    fn publish(&mut self, id: GenId, stored: Stored) {
        assert!(
            self.generations.keys().all(|&used| used < id),
            "generation {id:?} is not newer than every generation written"
        );
        match stored.delta {
            Some(_) => self.chain.push(id),
            None => self.chain = vec![id],
        }
        self.generations.insert(id, stored);
    }
}

#[derive(Debug)]
struct Stored {
    header: GenHeader,
    delta: Option<DeltaHeader>,
    frames: Vec<Vec<u8>>,
    items: Vec<IndexItem>,
}

impl Stored {
    fn generation(&self, id: GenId) -> Generation {
        let items = self.items.clone();
        match &self.delta {
            Some(delta) => Generation::new_delta(id, delta.clone(), items),
            None => Generation::new(id, self.header.clone(), items),
        }
    }
}

/// Frames pulled for one write, with an item per frame.
#[derive(Default)]
struct Pulled {
    frames: Vec<Vec<u8>>,
    items: Vec<IndexItem>,
}

impl Pulled {
    fn push(&mut self, entry: EncodedEntry) {
        self.items.push(IndexItem {
            table: entry.table,
            key_hash: entry.key_hash,
            loc: self.frames.len() as u64,
            len: entry.frame.len() as u32,
        });
        self.frames.push(entry.frame);
    }

    fn into_stored(self, header: GenHeader, delta: Option<DeltaHeader>) -> Stored {
        Stored {
            header,
            delta,
            frames: self.frames,
            items: self.items,
        }
    }
}

/// The checkpoint lock of a [`MemoryMemo`], released on drop.
#[derive(Debug)]
pub struct MemoryMemoGuard {
    state: Arc<Mutex<State>>,
}

impl Drop for MemoryMemoGuard {
    fn drop(&mut self) {
        lock(&self.state).locked = false;
    }
}

// The state is plain data updated in single assignments, so a panic while
// it is held leaves nothing half-written.
fn lock(state: &Mutex<State>) -> MutexGuard<'_, State> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

impl MemoryMemo {
    pub fn new() -> Self {
        MemoryMemo::default()
    }

    /// A store whose handles report no delta support, so checkpoints write
    /// full generations as on a backend from before deltas.
    pub fn without_deltas() -> Self {
        MemoryMemo {
            without_deltas: true,
            ..MemoryMemo::default()
        }
    }

    /// Overwrites the stored frame of `item` in `generation`, so tests can
    /// damage an entry.
    pub fn replace_frame(&self, generation: GenId, item: &IndexItem, frame: Vec<u8>) {
        let mut state = lock(&self.state);
        let stored = state
            .generations
            .get_mut(&generation)
            .expect("no such generation");
        stored.frames[item.loc as usize] = frame;
    }
}

impl MemoBackend for MemoryMemo {
    type Guard = MemoryMemoGuard;

    fn lock_checkpoint(&self) -> Result<MemoryMemoGuard> {
        let mut state = lock(&self.state);
        if state.locked {
            return Err(Error::Busy);
        }
        state.locked = true;
        let chain = state.chain.clone();
        state.generations.retain(|id, _| chain.contains(id));
        Ok(MemoryMemoGuard {
            state: Arc::clone(&self.state),
        })
    }

    fn load(&self) -> Result<Option<Generation>> {
        let state = lock(&self.state);
        Ok(state
            .chain
            .first()
            .map(|&id| state.generations[&id].generation(id)))
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        assert!(!item.is_tombstone(), "reading a tombstone's frame");
        let state = lock(&self.state);
        Ok(state
            .generations
            .get(&generation)
            .and_then(|stored| stored.frames.get(item.loc as usize).cloned()))
    }

    fn write_generation(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &[EncodedEntry],
    ) -> Result<Generation> {
        self.write_generation_streamed(base, id, header, &mut entries.iter().cloned().map(Ok))
    }

    /// Pulls every entry and builds the generation without the store's
    /// lock, so the iterator may read through this handle, then publishes
    /// under it.
    fn write_generation_streamed(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &mut dyn Iterator<Item = Result<EncodedEntry>>,
    ) -> Result<Generation> {
        if lock(&self.state).tip() != base {
            return Err(Error::Busy);
        }
        let mut pulled = Pulled::default();
        for entry in entries {
            pulled.push(entry?);
        }
        let stored = pulled.into_stored(header.clone(), None);
        let generation = stored.generation(id);
        let mut state = lock(&self.state);
        if state.tip() != base {
            return Err(Error::Busy);
        }
        state.publish(id, stored);
        Ok(generation)
    }

    fn read_entries(&self, generation: GenId, items: &[IndexItem]) -> Result<Vec<Option<Vec<u8>>>> {
        assert!(
            items.iter().all(|item| !item.is_tombstone()),
            "reading a tombstone's frame"
        );
        let state = lock(&self.state);
        let stored = state.generations.get(&generation);
        Ok(items
            .iter()
            .map(|item| stored.and_then(|s| s.frames.get(item.loc as usize).cloned()))
            .collect())
    }

    fn supports_delta(&self) -> bool {
        !self.without_deltas
    }

    fn load_chain(&self, known: &[Arc<Generation>]) -> Result<Option<Chain>> {
        let state = lock(&self.state);
        if state.chain.is_empty() {
            return Ok(None);
        }
        let layers = state
            .chain
            .iter()
            .map(|&id| match known.iter().find(|layer| layer.id() == id) {
                Some(layer) => Arc::clone(layer),
                None => Arc::new(state.generations[&id].generation(id)),
            })
            .collect();
        Chain::new(layers).map(Some)
    }

    fn write_delta(
        &self,
        tip: GenId,
        id: GenId,
        header: &DeltaHeader,
        entries: &[EncodedEntry],
        tombstones: &[Tombstone],
    ) -> Result<Generation> {
        let mut state = lock(&self.state);
        if state.tip() != Some(tip) {
            return Err(Error::Busy);
        }
        assert_eq!(
            header.parent, tip,
            "a delta's parent is the tip it is written on"
        );
        assert_eq!(
            Some(header.base),
            state.chain.first().copied(),
            "a delta's base is its chain's base"
        );
        let mut pulled = Pulled::default();
        for entry in entries {
            pulled.push(entry.clone());
        }
        pulled.items.extend(
            tombstones
                .iter()
                .map(|t| IndexItem::tombstone(t.table, t.key_hash)),
        );
        let stored = pulled.into_stored(header.header.clone(), Some(header.clone()));
        let generation = stored.generation(id);
        state.publish(id, stored);
        Ok(generation)
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        let mut state = lock(&self.state);
        assert!(
            !state.chain.contains(&generation),
            "removing a generation of the current chain"
        );
        state.generations.remove(&generation);
        Ok(())
    }
}
