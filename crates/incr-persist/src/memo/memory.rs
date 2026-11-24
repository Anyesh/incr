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
    items: Arc<Vec<IndexItem>>,
}

impl Stored {
    /// What [`Unbuilt::build`] needs, taken in O(1) under the state lock so
    /// the build runs outside it.
    fn unbuilt(&self, id: GenId) -> Unbuilt {
        Unbuilt {
            id,
            header: self.header.clone(),
            delta: self.delta.clone(),
            items: Arc::clone(&self.items),
        }
    }
}

struct Unbuilt {
    id: GenId,
    header: GenHeader,
    delta: Option<DeltaHeader>,
    items: Arc<Vec<IndexItem>>,
}

impl Unbuilt {
    fn build(self) -> Generation {
        let items = Vec::clone(&self.items);
        match self.delta {
            Some(delta) => Generation::new_delta(self.id, delta, items),
            None => Generation::new(self.id, self.header, items),
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
            items: Arc::new(self.items),
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
        let stale: Vec<Stored> = {
            let mut state = lock(&self.state);
            if state.locked {
                return Err(Error::Busy);
            }
            state.locked = true;
            let stale: Vec<GenId> = state
                .generations
                .keys()
                .filter(|id| !state.chain.contains(id))
                .copied()
                .collect();
            stale
                .iter()
                .filter_map(|id| state.generations.remove(id))
                .collect()
        };
        drop(stale);
        Ok(MemoryMemoGuard {
            state: Arc::clone(&self.state),
        })
    }

    fn load(&self) -> Result<Option<Generation>> {
        let base = {
            let state = lock(&self.state);
            state
                .chain
                .first()
                .map(|&id| state.generations[&id].unbuilt(id))
        };
        Ok(base.map(Unbuilt::build))
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
        let generation = stored.unbuilt(id).build();
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
        let layers: Vec<(Option<Arc<Generation>>, Option<Unbuilt>)> = {
            let state = lock(&self.state);
            state
                .chain
                .iter()
                .map(|&id| match known.iter().find(|layer| layer.id() == id) {
                    Some(layer) => (Some(Arc::clone(layer)), None),
                    None => (None, Some(state.generations[&id].unbuilt(id))),
                })
                .collect()
        };
        if layers.is_empty() {
            return Ok(None);
        }
        let layers = layers
            .into_iter()
            .map(|(known, unbuilt)| {
                known.unwrap_or_else(|| Arc::new(unbuilt.expect("one of the two").build()))
            })
            .collect();
        Chain::new(layers).map(Some)
    }

    /// Builds the layer and its generation without the store's lock, then
    /// publishes under it.
    fn write_delta(
        &self,
        tip: GenId,
        id: GenId,
        header: &DeltaHeader,
        entries: &[EncodedEntry],
        tombstones: &[Tombstone],
    ) -> Result<Generation> {
        if lock(&self.state).tip() != Some(tip) {
            return Err(Error::Busy);
        }
        assert_eq!(
            header.parent, tip,
            "a delta's parent is the tip it is written on"
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
        let generation = stored.unbuilt(id).build();
        let mut state = lock(&self.state);
        if state.tip() != Some(tip) {
            return Err(Error::Busy);
        }
        assert_eq!(
            Some(header.base),
            state.chain.first().copied(),
            "a delta's base is its chain's base"
        );
        state.publish(id, stored);
        Ok(generation)
    }

    fn concurrent_compaction(&self) -> bool {
        true
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        let removed = {
            let mut state = lock(&self.state);
            assert!(
                !state.chain.contains(&generation),
                "removing a generation of the current chain"
            );
            state.generations.remove(&generation)
        };
        drop(removed);
        Ok(())
    }
}
