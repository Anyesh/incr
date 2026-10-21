use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::{EncodedEntry, GenHeader, GenId, Generation, IndexItem, MemoBackend};
use crate::{Error, Result};

/// A memo store held in process memory, for tests.
///
/// Clones are handles on the same store, the way separate processes share
/// a directory. Nothing outlives the last handle.
#[derive(Clone, Debug, Default)]
pub struct MemoryMemo {
    state: Arc<Mutex<State>>,
}

#[derive(Debug, Default)]
struct State {
    current: Option<GenId>,
    generations: BTreeMap<GenId, Stored>,
    locked: bool,
}

#[derive(Debug)]
struct Stored {
    header: GenHeader,
    frames: Vec<Vec<u8>>,
    items: Vec<IndexItem>,
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
        let current = state.current;
        state.generations.retain(|&id, _| Some(id) == current);
        Ok(MemoryMemoGuard {
            state: Arc::clone(&self.state),
        })
    }

    fn load(&self) -> Result<Option<Generation>> {
        let state = lock(&self.state);
        Ok(state.current.map(|id| {
            let stored = &state.generations[&id];
            Generation::new(id, stored.header.clone(), stored.items.clone())
        }))
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
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
        let mut state = lock(&self.state);
        if state.current != base {
            return Err(Error::Busy);
        }
        assert!(
            state.generations.keys().all(|&used| used < id),
            "generation {id:?} is not newer than every generation written"
        );
        let items: Vec<_> = entries
            .iter()
            .enumerate()
            .map(|(loc, entry)| IndexItem {
                table: entry.table,
                key_hash: entry.key_hash,
                loc: loc as u64,
                len: entry.frame.len() as u32,
            })
            .collect();
        let frames = entries.iter().map(|entry| entry.frame.clone()).collect();
        state.generations.insert(
            id,
            Stored {
                header: header.clone(),
                frames,
                items: items.clone(),
            },
        );
        state.current = Some(id);
        Ok(Generation::new(id, header.clone(), items))
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        let mut state = lock(&self.state);
        assert_ne!(
            state.current,
            Some(generation),
            "removing the current generation"
        );
        state.generations.remove(&generation);
        Ok(())
    }
}
